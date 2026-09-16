"""
Command line interface.

    betedge daily                         the one you want: budgeted scan + shortlist
    betedge profiles                      named override bundles, e.g. nfl-week
    betedge budget                        credits left, and today's allowance
    betedge diagnose                      what the whole board looks like
    betedge sports                        list live sport keys and prop coverage
    betedge quota                         check credits (free)
    betedge scan                          run a scan and write a report
    betedge show                          re-print the last scan
    betedge bet <opp_id> --stake 50       log a bet you placed
    betedge settle <bet_id> won           settle it
    betedge close                         capture closing lines for open bets
    betedge report                        performance report
    betedge compare                       single bets vs multi-leg, side by side
    betedge export <file.csv>             dump bets to CSV

    betedge parlay scan                   build and rank multi-leg tickets
    betedge parlay coverage               which sports have usable prop coverage
    betedge parlay correlations --from X  fit correlations from game logs
    betedge parlay verify-payouts         print the payout table being used
    betedge parlay bet <ticket> --stake N log an entry you placed
    betedge parlay settle <id> --hit K    settle it by legs landed
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import budget as B
from . import pricing as P
from . import report as R
from .closing import capture_closing_lines, capture_parlay_closing_lines
from .config import Config
from .db import Database, parse_timestamp
from .markets import SPORTS, expand_sport_keys, markets_for
from .oddsapi import OddsApiClient
from .scan import _events_in_window, best_per_selection, cap_exposure, scan


def build_client(cfg: Config) -> OddsApiClient:
    return OddsApiClient(
        api_key=cfg.api.resolve_key(),
        base_url=cfg.api.base_url,
        timeout=cfg.api.timeout_seconds,
        max_retries=cfg.api.max_retries,
        cache_seconds=cfg.api.cache_seconds,
        cache_dir=Path(cfg.api.cache_dir),
        max_credits_per_scan=cfg.api.max_credits_per_scan,
        min_credits_remaining=cfg.api.min_credits_remaining,
    )


def current_budget(cfg: Config, db: Database, client: OddsApiClient | None = None,
                   now: datetime | None = None) -> B.BudgetStatus:
    """
    Today's spending allowance.

    `remaining` comes from the API when we have spoken to it this run,
    because the provider's own count cannot drift; the local ledger supplies
    what the header cannot, which is how much of today is already gone.
    """
    now = now or datetime.now(timezone.utc)
    cycle_start, cycle_end = B.cycle_bounds(now, cfg.budget.cycle_day)
    day_start, day_end = B.day_bounds(now)
    return B.plan(
        monthly_credits=cfg.budget.monthly_credits,
        cycle_day=cfg.budget.cycle_day,
        reserve=cfg.budget.reserve,
        burst=cfg.budget.daily_burst,
        spent_this_cycle=db.spend_between(cycle_start, cycle_end),
        spent_today=db.spend_between(day_start, day_end),
        api_remaining=client.quota.remaining if client is not None else None,
        now=now,
    )


def _apply_profile(cfg: Config, name: str | None) -> None:
    """
    Apply a --profile and state exactly what it moved.

    Printed, never silent. A profile can reach the staking fractions and
    the guard thresholds, so a scan running under settings the user did
    not state is the same failure as a stale roster: confident output with
    an input nobody checked.
    """
    if not name:
        return
    changes = cfg.apply_profile(name)
    if not changes:
        print(f"Profile '{name}' applied; nothing differed from your config.\n")
        return
    print(f"Profile '{name}' applied:")
    for change in changes:
        print(f"  {change.describe()}")
    print()


def _log_spend(db: Database, client: OddsApiClient, command: str,
               detail: str | None = None) -> int:
    """Record what a command actually cost, straight off the API headers."""
    spent = client.quota.spent_this_session
    db.record_spend(spent, command=command, detail=detail,
                    remaining=client.quota.remaining)
    return spent


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_sports(cfg: Config, args) -> int:
    client = build_client(cfg)
    live = client.sports()
    by_key = {s["key"]: s for s in live}
    print(f"{len(live)} sports live at the API.\n")

    print("Configured in betedge:")
    for key, sport in SPORTS.items():
        status = "live" if key in by_key else "not currently offered"
        n = len(markets_for(key)) if sport.props else 0
        print(f"  {key:38} {sport.title:22} {n:2d} prop markets   [{status}]")
        if sport.notes and args.verbose:
            print(f"      {sport.notes}")

    if args.all:
        print("\nEverything else the API offers right now:")
        for s in sorted(live, key=lambda x: (x["group"], x["key"])):
            if s["key"] not in SPORTS:
                print(f"  {s['key']:44} {s['title']}")
    return 0


def cmd_quota(cfg: Config, args) -> int:
    """
    What the configuration costs, and what the plan affords.

    Every call this makes is free -- /sports and /events are unbilled -- so
    it can be run as often as you like. The point is to answer the question
    that decides everything else: how often can this configuration run?
    """
    client = build_client(cfg)
    q = client.probe_quota()
    if q.remaining is not None:
        print(f"Credits remaining : {q.remaining:,}")
    if q.used is not None:
        print(f"Credits used      : {q.used:,}")

    live = [s["key"] for s in client.sports()]

    # ---- game-level sweep -------------------------------------------
    core_total = 0
    core_rows = []
    if cfg.core_sports:
        resolved, blocked = cfg.allowed(expand_sport_keys(cfg.core_sports, live))
        for sport in blocked:
            print(f"  {sport:34} excluded (see excluded_sports)")
        for sport in resolved:
            n = len(cfg.core_markets_for_sport(sport))
            core_total += n
            core_rows.append((sport, n))

    if core_rows:
        print("\nGame-level sweep (whole sport per call):")
        for sport, n in core_rows:
            print(f"  {sport:34} {n:2d} credit" + ("s" if n != 1 else ""))
        print(f"  {'':34} {'-' * 10}")
        print(f"  {'one full sweep':34} {core_total:2d} credits")

    # ---- prop pass ---------------------------------------------------
    prop_total = 0
    if cfg.sports:
        print("\nPlayer props (per event, per market):")
        prop_sports, prop_blocked = cfg.allowed(cfg.sports)
        for sport in prop_blocked:
            print(f"  {sport:34} excluded (see excluded_sports)")
        for sport in prop_sports:
            markets = cfg.markets_for_sport(sport, cfg.model.include_alternate_lines)
            try:
                events = client.events(sport)
            except Exception as exc:  # noqa: BLE001
                print(f"  {sport:34} unavailable: {exc}")
                continue
            window = cfg.prop_window_hours(sport)
            in_window = _events_in_window(events, datetime.now(timezone.utc), cfg,
                                          max_hours=window)
            cost = len(in_window) * len(markets)
            prop_total += cost
            print(
                f"  {sport:34} {len(in_window):2d} events (of {len(events)} "
                f"posted, {window:.0f}h window) x {len(markets)} markets "
                f"= {cost:,} credits"
            )
        print(f"  {'':34} {'-' * 10}")
        print(f"  {'one full prop pass':34} {prop_total:,} credits")

    # ---- what the plan affords ---------------------------------------
    per_run = core_total + prop_total
    if per_run <= 0:
        print("\nNothing configured to scan.")
        return 0

    monthly = cfg.budget.monthly_credits
    usable = max(0, monthly - cfg.budget.reserve)
    print(f"\nOne `betedge daily` right now: about {per_run:,} credits.")
    print(
        f"Plan: {monthly:,}/month, {cfg.budget.reserve:,} reserved for closing "
        f"lines, so {usable:,} to scan with."
    )
    if core_total:
        print(
            f"  Game lines alone ({core_total} credits) would run "
            f"{usable // core_total:,} times a month "
            f"-- about {usable // core_total // 30:,} an hour, every day."
        )
    runs_per_day = usable / 30.0 / per_run if per_run else 0
    print(f"  This full configuration: about {runs_per_day:.1f} runs a day.")
    if runs_per_day < 1:
        print(
            "  That is under once a day. Narrow `prop_markets`, tighten "
            "`prop_windows`, or scan game lines only with --no-props."
        )
    elif runs_per_day > 12:
        print(
            "  Comfortable. Prices at the soft books move in minutes, so "
            "run it hourly rather than saving the credits."
        )
    return 0


def cmd_scan(cfg: Config, args) -> int:
    _apply_profile(cfg, args.profile)
    if args.sports:
        cfg.sports = args.sports
    if args.min_ev is not None:
        cfg.model.min_ev = args.min_ev
    if args.alternate:
        cfg.model.include_alternate_lines = True
    if args.bankroll is not None:
        cfg.bankroll.amount = args.bankroll

    if args.core_sports:
        cfg.core_sports = args.core_sports
    if args.no_props:
        cfg.sports = []
    if args.markets:
        # A one-off narrowing: apply to every sport being scanned.
        cfg.prop_markets = {sport: list(args.markets) for sport in cfg.sports}

    client = build_client(cfg)
    result = scan(cfg, client, max_events_per_sport=args.max_events)

    if not args.all_lines:
        result.opportunities = best_per_selection(result.opportunities)
    cap_exposure(result.clean, cfg)

    # Stored first so the '#' column is the opportunity id that
    # `betedge bet` takes, rather than a position in the printed list.
    db = Database(cfg.database)
    scan_id = db.record_scan(result)
    _log_spend(db, client, "scan", ",".join(result.sports)[:200])

    print(R.scan_summary(result))
    print()
    print(R.console_table(result.clean, limit=args.limit))
    if result.suspect and not args.hide_suspect:
        print("\nSuspect (sanity checks tripped, no stake recommended):")
        print(R.console_table(result.suspect, limit=10))

    print(f"\nLogged as scan #{scan_id}.")
    if result.clean:
        print("Bet one with:  betedge bet <id> --stake <amount>  (the # column)")

    if not args.no_report:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        path = R.write_report(
            Path(cfg.reports_dir) / f"scan_{stamp}.html",
            R.scan_report_html(result, cfg, title=f"Scan {stamp}"),
        )
        print(f"Report: {path}")
    db.close()
    return 0


def cmd_budget(cfg: Config, args) -> int:
    db = Database(cfg.database)
    client = build_client(cfg)
    try:
        client.probe_quota()          # free
    except Exception as exc:  # noqa: BLE001
        print(f"Could not reach the API ({exc}); using the local ledger only.")
        client = None

    status = current_budget(cfg, db, client)
    print(status.summary())

    history = db.spend_by_day(limit=args.days)
    if history:
        print(f"\nLast {len(history)} day(s) of spending:")
        widest = max(h["credits"] for h in history) or 1
        for h in history:
            bar = "#" * max(1, round(h["credits"] / widest * 34))
            print(f"  {h['day']}  {h['credits']:>5,}  {bar}")
    else:
        print("\nNo spending logged yet.")
    db.close()
    return 0


def cmd_daily(cfg: Config, args) -> int:
    """
    One command for the daily routine.

    Works out what today's credits allow, spends them cheapest-first, prints
    a shortlist you can act on, and captures closing lines for anything
    already bet. The intent is that this is the only command you run.
    """
    _apply_profile(cfg, args.profile)
    if args.bankroll is not None:
        cfg.bankroll.amount = args.bankroll
    if args.min_ev is not None:
        cfg.model.min_ev = args.min_ev

    db = Database(cfg.database)
    client = build_client(cfg)

    try:
        client.probe_quota()          # free, and gives us the real remaining
    except Exception as exc:  # noqa: BLE001
        print(f"Error: cannot reach the API: {exc}", file=sys.stderr)
        db.close()
        return 1

    status = current_budget(cfg, db, client)
    print(status.summary())
    print()

    # Closing-line capture comes out of the reserve and runs first, because
    # props are pulled at kickoff -- a line missed now cannot be recovered
    # later, whereas a scan can always run again in an hour.
    if not args.no_close:
        try:
            stats = capture_closing_lines(cfg, client, db, window_minutes=args.window)
            if stats["captured"] or stats["events_checked"]:
                print(
                    f"Closing lines: captured {stats['captured']} across "
                    f"{stats['events_checked']} event(s)."
                )
        except Exception as exc:  # noqa: BLE001
            print(f"Closing-line capture failed: {exc}")

    allowance = status.spendable if cfg.budget.enabled else cfg.api.max_credits_per_scan
    already = client.quota.spent_this_session
    scan_allowance = max(0, allowance - already)

    if scan_allowance < 3:
        print(
            "\nNo credits left in today's allowance for scanning. "
            "Nothing else to do -- try again tomorrow, or raise "
            "budget.daily_burst if today genuinely deserves more."
        )
        _log_spend(db, client, "daily", "close-only")
        db.close()
        return 0

    # The scan stops itself at the allowance. Budget accounting is per-day,
    # so this is set from the plan rather than from the static per-scan cap.
    client.max_credits_per_scan = already + scan_allowance
    print(f"\nScanning with up to {scan_allowance:,} credits.\n")

    result = scan(cfg, client, max_events_per_sport=args.max_events)
    result.opportunities = best_per_selection(result.opportunities)
    cap_exposure(result.clean, cfg)

    # Stored before it is printed, so every row can show the id that
    # `betedge bet` takes rather than a position in a list.
    scan_id = db.record_scan(result)
    _log_spend(db, client, "daily", ",".join(result.sports)[:200])

    print(R.scan_summary(result))
    print()
    print(R.shortlist(result.clean, limit=args.limit))

    if result.suspect and not args.hide_suspect:
        print("\nSuspect -- shown because they are often interesting, but no "
              "stake is recommended:")
        print(R.console_table(result.suspect, limit=5))

    print(f"\nScan #{scan_id}. Re-print it any time with:  betedge show")

    if not args.no_report:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        path = R.write_report(
            Path(cfg.reports_dir) / f"scan_{stamp}.html",
            R.scan_report_html(result, cfg, title=f"Daily {stamp}"),
        )
        print(f"Report: {path}")

    after = current_budget(cfg, db, client)
    print(f"\n{after.remaining:,} credits left this cycle "
          f"({after.days_left:.1f} days to go).")
    db.close()
    return 0


def cmd_diagnose(cfg: Config, args) -> int:
    """
    Price the board and report the distribution, not just what cleared.

    The question this answers is the one a scan flagging nothing cannot: is
    the bar slightly too high, or is there no edge here at all? Those look
    identical in the rejection counters and call for opposite responses.

    Costs exactly what a scan costs -- it is a scan, with everything kept
    rather than only the winners.
    """
    if args.sports:
        cfg.sports = args.sports
    if args.no_props:
        cfg.sports = []

    db = Database(cfg.database)
    client = build_client(cfg)
    try:
        client.probe_quota()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: cannot reach the API: {exc}", file=sys.stderr)
        db.close()
        return 1

    status = current_budget(cfg, db, client)
    if cfg.budget.enabled:
        client.max_credits_per_scan = (
            client.quota.spent_this_session + status.spendable
        )

    result = scan(cfg, client, max_events_per_sport=args.max_events, collect=True)
    _log_spend(db, client, "diagnose", ",".join(result.sports)[:200])

    print(R.scan_summary(result))
    print()
    print(R.distribution_report(result.assessments, cfg.model.min_ev, top=args.top))
    db.close()
    return 0


def cmd_show(cfg: Config, args) -> int:
    """
    Re-print a scan, showing only what is still actionable.

    With an hourly cron the last scan can be nearly an hour old, and a
    chunk of what it flagged will have started or drifted. Printing all of
    it without saying so invites betting a price that no longer exists, so
    started events are dropped by default and the scan's age is stated up
    front.
    """
    db = Database(cfg.database)
    scan_id = args.scan_id or db.latest_scan_id()
    if scan_id is None:
        print("No scans recorded yet. Run:  betedge daily")
        return 1

    rows = db.opportunities_for_scan(scan_id)
    if not rows:
        print(f"Scan #{scan_id} flagged nothing.")
        return 0

    now = datetime.now(timezone.utc)
    scanned_at = parse_timestamp(rows[0]["scanned_at"])
    age_min = (now - scanned_at).total_seconds() / 60 if scanned_at else None

    age_note = ""
    if age_min is not None:
        age_note = (f"{age_min:.0f} min ago" if age_min < 90
                    else f"{age_min/60:.1f} hours ago")
        if age_min > cfg.model.max_soft_staleness_minutes:
            age_note += " -- prices have probably moved"

    live, started = [], 0
    for r in rows:
        if r["suspect"] and not args.include_suspect:
            continue
        start = parse_timestamp(r["commence_time"])
        mins_out = (start - now).total_seconds() / 60 if start else None
        if mins_out is not None and mins_out < cfg.model.min_minutes_to_start \
                and not args.all:
            started += 1
            continue
        live.append((r, mins_out))

    print(f"Scan #{scan_id}" + (f"  ({age_note})" if age_note else "") + "\n")

    if not live:
        print("Nothing from this scan is still playable.")
        if started:
            print(f"  {started} flagged bet(s) have already started. "
                  "Run `betedge daily` for a fresh look.")
        db.close()
        return 0

    header = (f"{'id':>5}  {'EV':>7}  {'liq':<6} {'bet':<46} {'book':<12} "
              f"{'price':>7} {'stake':>7}  {'starts':>7}  game")
    print(header)
    print("-" * len(header))
    for r, mins_out in live:
        desc = R.describe(r["selection"], r["side"], r["line"], r["market"])
        if mins_out is None:
            when = "-"
        elif mins_out < 60:
            when = f"{mins_out:.0f}m"
        elif mins_out < 60 * 24:
            when = f"{mins_out/60:.1f}h"
        else:
            when = f"{mins_out/1440:.1f}d"
        liq = R._liquidity_label(r["liquidity"] if "liquidity" in r.keys() else None)
        print(
            f"{r['id']:>5}  {r['ev']:>+6.1%}  {liq:<6} {desc:<46.46} "
            f"{r['soft_book']:<12} {R.american(r['soft_price']):>7} "
            f"{(r['recommended_stake'] or 0):>7,.0f}  {when:>7}  "
            f"{r['away_team']} @ {r['home_team']}"
        )

    if started:
        print(f"\n{started} more have already started (--all to include them).")
    print("\nPlace one with:  betedge bet <id> --stake <amount>")
    db.close()
    return 0


def cmd_bet(cfg: Config, args) -> int:
    db = Database(cfg.database)

    # Fields for a bet the scanner never surfaced, or one being logged
    # after the fact. Anything omitted is filled from the opportunity when
    # an id is given. Without these, a manual bet lands with blank columns
    # and `betedge close` can never find it to capture a closing line.
    manual = {
        k: v
        for k, v in {
            "sport": args.sport,
            "event_id": args.event_id,
            "commence_time": args.commence,
            "matchup": args.matchup,
            "market": args.market,
            "selection": args.selection,
            "side": args.side,
            "line": args.line,
        }.items()
        if v is not None
    }

    bet_id = db.place_bet(
        opportunity_id=args.opportunity_id,
        stake=args.stake,
        price=args.price,
        book=args.book,
        notes=args.notes,
        **manual,
    )

    if args.settle:
        db.settle_bet(bet_id, args.settle)
    bet = db.get_bet(bet_id)
    print(
        f"Logged bet #{bet_id}: "
        f"{R.describe(bet['selection'], bet['side'], bet['line'], bet['market'])} "
        f"@ {R.american(bet['price'])} on {bet['book']} for {bet['stake']:,.0f}"
        + (f" (EV {bet['ev_at_bet']:+.1%})" if bet["ev_at_bet"] is not None else "")
    )
    if bet["status"] != "pending":
        print(f"  settled {bet['status']}: {bet['pnl']:+,.2f}")
    if not bet["event_id"] and bet["status"] == "pending":
        # Only worth saying while a closing line could still be captured.
        # A settled bet is past every close there was, so warning about it
        # is noise on exactly the rows where nothing can be done -- and a
        # warning that fires when nothing is wrong teaches you to skip the
        # ones that matter.
        print("  no event id — `betedge close` won't be able to capture a "
              "closing line for this one. Log bets against a flagged "
              "opportunity (`betedge bet <id>`) and it is filled in for you.")
    db.close()
    return 0


def cmd_settle(cfg: Config, args) -> int:
    db = Database(cfg.database)
    pnl = db.settle_bet(args.bet_id, args.status)
    print(f"Bet #{args.bet_id} settled {args.status}: {pnl:+,.2f}")
    db.close()
    return 0


def cmd_delete(cfg: Config, args) -> int:
    """
    Remove bets from the ledger.

    This exists because a wrong row is worse than a missing one. The
    strategy comparison divides realised P&L by modelled P&L to ask whether
    a claimed edge actually shows up; five bets that were never placed make
    that ratio meaningless, and they do it quietly -- the table looks fine,
    it is just answering a different question than the one asked.

    Nothing is deleted until every id has been printed, because the ids are
    the easiest thing in the world to get wrong by one.
    """
    db = Database(cfg.database)
    rows = []
    missing = []
    for bet_id in args.bet_ids:
        row = db.get_bet(bet_id)
        (rows if row is not None else missing).append(row if row is not None else bet_id)

    if missing:
        print(f"No bet with id: {', '.join(str(m) for m in missing)}")
    if not rows:
        db.close()
        return 1

    print("About to delete:")
    for row in rows:
        stake = row["stake"] or 0.0
        pnl = row["pnl"]
        settled = f"{row['status']} {pnl:+,.2f}" if pnl is not None else row["status"]
        print(
            f"  #{row['id']}  {row['selection'] or '?'} "
            f"{row['side'] or ''} {row['line'] if row['line'] is not None else ''}"
            f" ({row['market'] or '?'}) on {row['book']} "
            f"for {stake:,.2f} -- {settled}"
        )

    if not args.yes:
        try:
            answer = input(f"Delete {len(rows)} bet(s)? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            print("Left alone.")
            db.close()
            return 1

    for row in rows:
        db.delete_bet(row["id"])
    print(f"Deleted {len(rows)} bet(s). Re-run `betedge compare` to see the ledger.")
    db.close()
    return 0


def cmd_close(cfg: Config, args) -> int:
    db = Database(cfg.database)
    client = build_client(cfg)
    stats = capture_closing_lines(cfg, client, db, window_minutes=args.window)
    print(
        f"Checked {stats['events_checked']} events, captured {stats['captured']} "
        f"closing lines, {stats['not_found']} not found, {stats['errors']} errors."
    )

    # Multi-leg tickets capture their legs off the same endpoint, and a
    # ticket's joint probability is recomputed once every leg has closed.
    # Ticket CLV is the only fast read on whether the parlay model works,
    # so it runs by default and is skipped only on request.
    if not args.no_parlays:
        try:
            p = capture_parlay_closing_lines(cfg, client, db,
                                             window_minutes=args.window)
            if p["tickets_checked"]:
                print(
                    f"Tickets: {p['legs_captured']} leg(s) closed across "
                    f"{p['events_checked']} event(s); {p['tickets_closed']} "
                    f"ticket(s) now fully priced at the close."
                )
        except Exception as exc:  # noqa: BLE001
            print(f"Parlay closing capture failed: {exc}")

    _log_spend(db, client, "close")
    print(f"({client.quota.spent_this_session} credits)")
    db.close()
    return 0


def cmd_report(cfg: Config, args) -> int:
    db = Database(cfg.database)
    s = db.summary()
    print("Betting performance")
    print("-" * 40)
    for k, v in s.items():
        if v is None:
            print(f"{k:22} -")
        elif isinstance(v, float) and k in {
            "roi", "win_rate", "avg_clv", "clv_positive_rate"
        }:
            print(f"{k:22} {v:+.2%}")
        elif isinstance(v, float):
            print(f"{k:22} {v:,.2f}")
        else:
            print(f"{k:22} {v:,}")

    parlay = db.parlay_summary()
    if parlay["tickets_generated"]:
        print("\nMulti-leg tickets")
        print("-" * 40)
        for k, v in parlay.items():
            if v is None:
                print(f"{k:22} -")
            elif isinstance(v, float) and k in {"roi", "avg_ticket_clv"}:
                print(f"{k:22} {v:+.2%}")
            elif isinstance(v, float):
                print(f"{k:22} {v:,.2f}")
            else:
                print(f"{k:22} {v:,}")

    path = R.write_report(
        Path(cfg.reports_dir) / "performance.html", R.performance_report_html(db, cfg)
    )
    print(f"\nReport: {path}")
    db.close()
    return 0


# ---------------------------------------------------------------------
# Rotten Tomatoes threshold contracts on Kalshi
# ---------------------------------------------------------------------


def _rt_book(cfg, args):
    from .tomatoes import ContractBook

    return ContractBook.load(getattr(args, "contracts", None))


def cmd_rt_contracts(cfg: Config, args) -> int:
    """
    Print the contracts exactly as loaded, so they can be checked against
    the exchange.

    Four things decide a contract and none can be derived: which
    Tomatometer it settles on, whether the threshold itself wins, the
    threshold, and when it is read. An EV built on the wrong one of those
    is not slightly wrong.
    """
    book = _rt_book(cfg, args)
    origin = "your own copy" if book.is_user_copy else "shipped defaults"
    print(f"Contracts: {book.path}  ({origin})")
    print(f"Last verified by you: {book.last_verified_by_user or 'never'}\n")

    if not book.contracts:
        print("No contracts defined yet.\n")
        print("Add them from the Kalshi market pages. Start your own copy "
              "with:\n\n    betedge rt contracts --init\n")
        return 0

    for c in book.contracts:
        extra = book.settings_for(c.ticker)
        mark = "verified" if c.settlement_verified else "NOT VERIFIED"
        print(f"{c.ticker}  [{mark}]")
        print(f"  {c.describe()}   ({c.scope})")
        print(f"  rotten tomatoes: /m/{extra.get('slug') or '?'}"
              f"   settles: {c.settles_at or 'unknown'}")
        ceiling = extra.get("max_new_reviews")
        print(f"  reviews still to come: ceiling "
              f"{ceiling if ceiling is not None else 'from config'}, "
              f"expected {extra.get('expected_new_reviews') or 'from config'}")
        print()

    if book.unverified:
        print("NOTHING IS STAKED ON THESE UNTIL YOU CHECK THEM:\n  "
              + ", ".join(book.unverified))
        print("\nOpen each on Kalshi and confirm the Tomatometer it settles "
              "on, whether the threshold itself wins, and the settlement "
              "time. Then set `verified: true`.")
    return 0


def _init_rt_contracts(cfg: Config) -> int:
    from .tomatoes import CONTRACTS_PATH, USER_CONTRACTS_PATH

    if USER_CONTRACTS_PATH.exists():
        print(f"{USER_CONTRACTS_PATH} already exists -- leaving it alone.")
        return 0
    USER_CONTRACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    USER_CONTRACTS_PATH.write_text(CONTRACTS_PATH.read_text())
    print(f"Wrote {USER_CONTRACTS_PATH}. Edit that -- it is outside the "
          "repository, so a `git pull` will not collide with your work.")
    return 0


def cmd_rt_snapshot(cfg: Config, args) -> int:
    """Read a film's Tomatometer and print it."""
    from . import rtfetch

    try:
        card = rtfetch.fetch(
            args.slug, cache_dir=args.cache,
            max_age_seconds=0.0 if args.fresh else 900.0,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Could not read /m/{args.slug}: {exc}")
        return 1

    print(f"{card.film}  (/m/{card.slug})")
    for scope in sorted(card.scores):
        print(f"  {card.scores[scope].describe()}")
    return 0


def cmd_rt_scan(cfg: Config, args) -> int:
    """
    Price every contract, and say what to buy.

    Prices come from Kalshi's public endpoints -- no key, read only. The
    cost used is what the order book would ACTUALLY fill at the size
    being considered, not the top of the book: on markets this thin the
    best price is often good for twenty contracts and the next level is
    several cents worse.
    """
    from . import kalshi, rtfetch
    from . import tomatoes as T

    book = _rt_book(cfg, args)
    if not book.contracts:
        print(f"No contracts defined in {book.path}.")
        print("See `betedge rt contracts` for how to add them.")
        return 0

    bankroll = args.bankroll or cfg.bankroll.amount
    client = kalshi.MarketData()
    rows = []

    for contract in book.contracts:
        extra = book.settings_for(contract.ticker)
        slug = extra.get("slug")
        if not slug:
            rows.append((contract, None, "no rotten tomatoes slug"))
            continue
        try:
            card = rtfetch.fetch(slug, cache_dir=args.cache)
            snapshot = card.scope(contract.scope)
        except Exception as exc:  # noqa: BLE001
            rows.append((contract, None, f"score unreadable: {exc}"))
            continue

        price = args.price
        if price is None:
            try:
                depth = client.orderbook(contract.ticker)
                price, fillable = depth.cost_to_buy(kalshi.YES, args.size)
                if fillable == 0:
                    rows.append((contract, None, "nothing offered"))
                    continue
            except Exception as exc:  # noqa: BLE001
                rows.append((contract, None, f"no price: {exc}"))
                continue

        rows.append((contract, T.assess(
            snapshot, contract, price, cfg,
            max_new_reviews=extra.get("max_new_reviews"),
            expected_new=extra.get("expected_new_reviews"),
        ), None))

    rows.sort(key=lambda r: (r[1].ev if r[1] else -99), reverse=True)

    print(f"{'ticker':<24} {'bet':<32} {'score':>10} {'price':>7} "
          f"{'fair':>7} {'EV':>8} {'stake':>7}")
    print("-" * 104)
    staked = 0
    for contract, a, problem in rows:
        if a is None:
            print(f"{contract.ticker:<24} {contract.describe():<32}   "
                  f"-- {problem}")
            continue
        stake = T.position_size(a, bankroll, cfg)
        staked += 1 if stake else 0
        print(
            f"{contract.ticker:<24} {contract.describe():<32} "
            f"{a.snapshot.fresh}/{a.snapshot.total:<7} "
            f"{a.price * 100:>6.0f}c {a.probability * 100:>6.1f}c "
            f"{a.ev:>+7.1%} "
            f"{('$' + format(stake, ',.0f')) if stake else '--':>7}"
        )
        for flag in a.flags:
            print(f"{'':<24}   ! {flag}")

    print()
    if staked:
        print(f"{staked} position(s) worth taking.")
    else:
        print("Nothing worth betting. The flags above say why, one by one.")
    if book.unverified:
        print(f"\n{len(book.unverified)} contract(s) unverified, so staked "
              "at nothing whatever their edge. `betedge rt contracts`.")
    return 0


def cmd_compare(cfg: Config, args) -> int:
    """
    Which strategy is actually making money -- and whether the sample can
    yet support an answer.
    """
    db = Database(cfg.database)
    comparison = db.compare_strategies()
    strategies = comparison.strategies

    def cell(value, kind="pct") -> str:
        if value is None:
            return "-"
        if kind == "pct":
            return f"{value:+.2%}"
        if kind == "pct0":
            return f"{value:.0%}"
        if kind == "money":
            return f"{value:+,.2f}"
        if kind == "ratio":
            return f"x{value:,.2f}"
        if kind == "int":
            return f"{value:,}"
        return f"{value:,.2f}"

    rows = [
        ("settled bets", [cell(s.settled, "int") for s in strategies]),
        ("pending", [cell(s.pending, "int") for s in strategies]),
        ("staked", [f"{s.staked:,.0f}" for s in strategies]),
        ("profit and loss", [cell(s.pnl, "money") for s in strategies]),
        ("ROI", [cell(s.roi) for s in strategies]),
        (
            f"ROI {comparison.confidence:.0%} interval",
            [
                "-" if s.roi_interval() is None
                else f"{s.roi_interval()[0]:+.1%} to {s.roi_interval()[1]:+.1%}"
                for s in strategies
            ],
        ),
        ("modelled P&L", [cell(s.modelled_pnl, "money") for s in strategies]),
        ("realised / modelled", [cell(s.realisation, "ratio") for s in strategies]),
        # Which bets that ratio is actually about. A bet logged by hand
        # carries no modelled edge, so it is in the settled count above
        # and not in this one -- and without the row, the ratio silently
        # looks like it covers everything.
        (
            "  ...over how many bets",
            [
                "-" if s.realisation is None
                else cell(s.modelled_settled, "int")
                for s in strategies
            ],
        ),
        ("avg closing-line value", [cell(s.avg_clv) for s in strategies]),
        ("CLV beat rate", [cell(s.clv_beat_rate, "pct0") for s in strategies]),
        ("CLV sample", [cell(len(s.clv_values), "int") for s in strategies]),
        (
            "bets needed for +/-5% ROI",
            [cell(s.bets_needed(), "int") for s in strategies],
        ),
    ]

    width = max(len(label) for label, _ in rows) + 2
    header = " " * width + "".join(f"{s.name:>20}" for s in strategies)
    print("Strategy comparison")
    print(header)
    print("-" * len(header))
    for label, values in rows:
        print(f"{label:<{width}}" + "".join(f"{v:>20}" for v in values))

    print()
    for line in _wrap(comparison.verdict(), 76):
        print(line)

    print()
    for line in _wrap(
        "Read the last two blocks before the first. Closing-line value "
        "converges in dozens of bets where profit needs thousands, and "
        "'realised over modelled' asks the question underneath the "
        "question: not which strategy won more, but whose claimed edge "
        "showed up.",
        76,
    ):
        print(line)
    db.close()
    return 0


def cmd_export(cfg: Config, args) -> int:
    db = Database(cfg.database)
    rows = db.all_bets()
    if not rows:
        print("No bets to export.")
        return 0

    path = Path(args.path)
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        from .tracker import export_tracker

        summary = export_tracker(rows, path, template_path=args.template)
        print(f"Wrote {summary['written']} settled bets to {summary['path']}")
        print(
            f"  staked {summary['total_staked']:,.0f}, "
            f"P&L {summary['pnl']:+,.2f}"
        )
        if summary["skipped"]:
            detail = ", ".join(f"{n} {s}" for s, n in sorted(summary["skipped"].items()))
            print(f"  skipped {detail} (the sheet's P&L formula has no place for them)")
        db.close()
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(r))
    print(f"Wrote {len(rows)} bets to {path}")
    db.close()
    return 0


# --------------------------------------------------------------------------
# Parlay / pick'em commands
# --------------------------------------------------------------------------


def cmd_parlay_scan(cfg: Config, args) -> int:
    from . import parlay as P

    _apply_profile(cfg, args.profile)
    if args.sports:
        cfg.sports = args.sports
    if args.products:
        cfg.parlay.products = args.products
    if args.bankroll is not None:
        cfg.bankroll.amount = args.bankroll
    if args.min_ev is not None:
        cfg.parlay.min_ev = args.min_ev
    if args.min_leg_edge is not None:
        cfg.parlay.min_leg_edge = args.min_leg_edge
    if args.max_legs is not None:
        cfg.parlay.max_legs = args.max_legs
    if args.draws is not None:
        cfg.parlay.draws = args.draws
    if args.rosters:
        cfg.parlay.rosters_path = args.rosters
    if args.interpolate:
        cfg.parlay.allow_line_interpolation = True
    if args.grouping:
        cfg.parlay.grouping = args.grouping

    db = Database(cfg.database)
    client = build_client(cfg)

    # The prop endpoint bills per event per market, which is the one thing
    # here that can eat a month of quota, so the day's allowance caps the
    # run exactly as it does for `daily`.
    if cfg.budget.enabled and not args.ignore_budget:
        try:
            client.probe_quota()          # free
            status = current_budget(cfg, db, client)
            allowance = max(0, status.spendable - client.quota.spent_this_session)
            if allowance < 3:
                print(
                    "No credits left in today's allowance. Run `betedge budget` "
                    "to see the pacing, or pass --ignore-budget to override it."
                )
                db.close()
                return 0
            client.max_credits_per_scan = client.quota.spent_this_session + allowance
            print(f"Scanning with up to {allowance:,} credits.\n")
        except Exception as exc:  # noqa: BLE001
            print(f"Could not check the budget ({exc}); using the per-scan cap.")

    result = P.scan_parlays(
        cfg, client, db=db,
        max_events_per_sport=args.max_events,
        offered_price=args.offered_price,
    )

    ticket_ids = db.record_parlay_tickets(
        result.tickets, draws=cfg.parlay.draws
    )
    _log_spend(db, client, "parlay scan", ",".join(result.sports)[:200])

    print(R.parlay_summary(result))
    if not result.tickets:
        note = R.near_miss_note(result)
        if note:
            print()
            print(note)
    print(R.roster_note(result, cfg))
    print()
    print(R.parlay_console(
        R.sorted_by_ev(result.clean), limit=args.limit,
        title="Ranked by expected value",
    ))
    if result.clean:
        print()
        print(R.parlay_console(
            R.sorted_by_ev_per_variance(result.clean), limit=min(args.limit, 5),
            title="Ranked by expected value per unit of variance",
        ))
    if result.suspect and not args.hide_suspect:
        print("\nSuspect (a guard tripped, no stake recommended):")
        print(R.parlay_console(result.suspect, limit=5))

    # Only warn about the products this run actually used. Naming the
    # whole table would train the reader to skip a line that matters.
    table = P.PayoutTable.load(cfg.parlay.payouts_path)
    used = [k for k in result.products if k in table.unverified]
    if used:
        print(
            "\nNOTE: the payout ladder has not been verified against your "
            f"account for: {', '.join(used)}.\n"
            "      Every EV above is only as right as those multipliers, and "
            "they vary by state.\n"
            "      Check them with `betedge parlay verify-payouts`."
        )

    print(f"\n{len(ticket_ids)} ticket(s) logged.")
    if any(t.recommended_stake > 0 for t in result.clean):
        print("Log one with:  betedge parlay bet <id> --stake <amount>")

    if not args.no_report:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        path = R.write_report(
            Path(cfg.reports_dir) / f"parlay_{stamp}.html",
            R.parlay_report_html(result, cfg, title=f"Parlay scan {stamp}"),
        )
        print(f"Report: {path}")
    db.close()
    return 0


def cmd_parlay_coverage(cfg: Config, args) -> int:
    """
    Which sports actually support this, measured rather than asserted.

    The module ships a default list, but seasons turn over and a book's
    prop coverage changes with them, so the list is re-derivable from what
    the API is serving today.
    """
    from . import parlay as P

    _apply_profile(cfg, args.profile)
    if args.window is not None:
        # A probe is not a scan: it is worth looking further ahead than you
        # would ever bet, because the question is what EXISTS, not what is
        # priced tightly enough to act on yet.
        for sport in (args.sports or list(P.DEFAULT_PROP_SPORTS)):
            cfg.prop_windows[sport] = args.window

    client = build_client(cfg)
    db = Database(cfg.database)
    sports = args.sports or list(P.DEFAULT_PROP_SPORTS)
    report = P.probe_coverage(
        cfg, client, sports=sports, max_events_per_sport=args.max_events,
        books_to_probe=args.books,
    )
    _log_spend(db, client, "parlay coverage", ",".join(sports)[:200])

    header = (
        f"{'sport':<26} {'events':>7} {'probed':>7} {'pinnacle 2-sided':>17} "
        f"{'credits':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in report.rows:
        if row.error:
            print(f"{row.sport:<26} {row.error[:70]}")
            continue
        print(
            f"{row.sport:<26} {row.events_in_window:>7} {row.events_probed:>7} "
            f"{row.two_sided_sharp_markets:>17,} {row.credits_spent:>8,}"
        )

    # Said before anything else, because a probe that asked nobody anything
    # is not a finding about any book -- and the rest of this output would
    # otherwise read as one.
    blank = [r for r in report.rows if not r.error and r.probed_nothing]
    for row in blank:
        if row.events_posted:
            print(
                f"\n{row.sport}: {row.events_posted} event(s) posted, none "
                f"inside the {row.window_hours:g}h window, so NOTHING was "
                f"probed and no book was asked anything.\n"
                f"  A Thursday night game is about 57 hours out on the "
                f"Tuesday before it. To reach it:\n"
                f"    betedge parlay coverage --window 96\n"
                f"    betedge parlay coverage --profile nfl-week"
            )
        else:
            print(
                f"\n{row.sport}: the API has no events posted at all, so "
                f"nothing was probed. Out of season,\n  or the slate is not "
                f"up yet."
            )

    # Per book, because "which pick'em site should I use" is a different
    # question from "does this sport work at all", and the totals above
    # cannot answer it.
    for row in report.rows:
        if row.error or not row.by_book:
            continue
        print(f"\n{row.sport} by book:")
        sub = (
            f"  {'book':<16} {'kind':<9} {'props':>7} {'matched':>8} "
            f"{'match':>6} {'per game':>9} {'players':>8}"
        )
        print(sub)
        print("  " + "-" * (len(sub) - 2))
        for book in sorted(
            row.by_book.values(), key=lambda b: (not b.is_pickem, -b.props)
        ):
            note = "" if book.usable else "  too thin"
            if book.thin:
                note = "  thin"
            print(
                f"  {book.book:<16} {'pickem' if book.is_pickem else 'parlay':<9} "
                f"{book.quotes // 2:>7,} {book.props:>8,} "
                f"{book.match_rate:>5.0%} {book.props_per_event:>9.1f} "
                f"{len(book.players):>8,}{note}"
            )
        if row.silent_books:
            print(
                f"  asked for and got nothing back: {', '.join(row.silent_books)}"
            )

    print(
        "\n'matched' counts distinct props, not quotes -- every prop is posted "
        "on both\nsides, so a quote count reads twice as deep as the board "
        "really is. A leg\ncompared against a line Pinnacle does not price is "
        "not a measurement, so it is\ndropped rather than approximated; a low "
        "match rate usually means a book posting\nalternate lines nobody sharp "
        "prices.\n\n'per game' is what decides whether tickets can be built at "
        "all. Legs are grouped\nby game, so a book with forty usable props "
        "spread over fifteen games can still\nbe unable to fill one ticket."
    )

    silent = sorted({b for row in report.rows for b in row.silent_books})
    if silent:
        print(
            f"\nNo quotes at all from: {', '.join(silent)}. Either the API "
            "does not carry\nthem or they are not pricing these sports right "
            "now -- both mean no legs.\nAsking for them cost nothing extra "
            "(ten books bill as one)."
        )

    usable = report.recommended
    if blank and not usable:
        # The generic "nothing is usable" line below would be misleading
        # here: it invites the reader to conclude something about coverage
        # from a probe that never ran.
        print(
            "\nNo conclusion about any book from this run -- widen the window "
            "and probe again."
        )
    elif usable:
        print(f"\nUsable today: {', '.join(usable)}")
        print("Put these in `sports:` in config.yaml for the prop pass.")
        for row in sorted(report.rows, key=lambda r: r.sport):
            book = row.best_pickem_book
            if book is not None and book.usable:
                print(
                    f"  {row.sport}: best pick'em book is {book.book} "
                    f"({book.props:,} props, {book.props_per_event:.1f} a game)."
                    + ("  Thin -- expect few tickets." if book.thin else "")
                )
            elif book is not None:
                print(
                    f"  {row.sport}: no pick'em book has enough props per game "
                    f"to fill a ticket (best is {book.book} at "
                    f"{book.props_per_event:.1f}; {P.MIN_PROPS_PER_EVENT} needed)."
                )
            # A sportsbook is shown for context and never recommended. It
            # prices its own parlays rather than paying a fixed ladder, so
            # it is not what this optimizer is for -- and the route to
            # using it is only worth naming to someone who has actually
            # configured a parlay product.
            parlay_configured = any(
                P.PayoutTable.load(cfg.parlay.payouts_path)
                .products.get(k, None) is not None
                and P.PayoutTable.load(cfg.parlay.payouts_path)
                .get(k).kind == P.KIND_PARLAY
                for k in cfg.parlay.products
            )
            for other in row.parlay_books:
                if other.usable:
                    tail = (
                        " Use it with `--products draftkings_parlay`."
                        if parlay_configured
                        else " Shown for reference; the pick'em optimizer "
                        "does not use it."
                    )
                    print(
                        f"  {row.sport}: {other.book} is a sportsbook, not a "
                        f"pick'em site -- its parlays are priced by the book "
                        f"rather than paid on a fixed ladder.{tail}"
                    )
    else:
        print(
            "\nNothing has usable coverage right now. Out of season, the "
            "slate has not been\nposted yet, or no probed book prices these "
            "markets -- try again closer to game day."
        )

    print("\nDeliberately excluded from the prop-based optimizer:")
    for pattern, why in sorted(report.excluded.items()):
        print(f"  {pattern:<26} {why}")
    print(f"\n{report.credits_spent} credit(s) spent on this probe.")
    db.close()
    return 0


def cmd_parlay_correlations(cfg: Config, args) -> int:
    from . import correlation as C

    from . import rosters as RO

    db = Database(cfg.database)
    if args.source:
        rows = C.read_game_logs(args.source)
        markets = args.markets or None
        estimates = C.fit_correlations(
            rows, markets=markets, max_pairs_per_bucket=args.max_pairs
        )
        written = db.save_correlation_estimates(estimates)
        print(
            f"Read {len(rows):,} stat lines and fitted {written} pairwise "
            f"correlation(s)."
        )

        # The same logs already carry player -> team, which the fitter
        # needs to bucket its pairs and was otherwise throwing away. Keeping
        # it means the command you have to run anyway is the one that keeps
        # your rosters current.
        learned = RO.from_game_logs(rows)
        by_sport: dict[str, list] = {}
        for entry in learned:
            by_sport.setdefault(entry.sport, []).append(entry)
        for sport, entries in sorted(by_sport.items()):
            for source in {e.source for e in entries}:
                db.replace_rosters(
                    sport, source, [e for e in entries if e.source == source]
                )
        if learned:
            print(
                f"Learned {len(learned):,} player/team mapping(s) from the same "
                f"logs across {len(by_sport)} sport(s) -- no separate roster "
                "file needed for those players."
            )

    store = C.EstimateStore.from_db(db)
    if not len(store):
        print(
            "No fitted correlations stored. Until there are, every ticket's "
            "correlation comes\nfrom the structural priors in "
            "betedge/data/correlation_priors.yaml -- which makes\nevery EV a "
            "hypothesis rather than an estimate. Fit some with:\n\n"
            "  betedge parlay correlations --from game_logs.csv\n\n"
            "CSV columns: " + ", ".join(C.REQUIRED_LOG_COLUMNS) + "[, date]"
        )
        db.close()
        return 0

    threshold = cfg.parlay.min_correlation_sample
    header = (
        f"{'sport':<22} {'market a':<26} {'market b':<26} {'relation':<14} "
        f"{'rho':>6} {'spearman':>9} {'n':>8}  used"
    )
    print(header)
    print("-" * len(header))
    for e in store.all():
        used = "yes" if e.n_observations >= threshold else f"no (<{threshold})"
        print(
            f"{e.sport:<22} {e.market_a:<26.26} {e.market_b:<26.26} "
            f"{e.relation:<14} {e.rho:>+6.2f} {e.spearman:>+9.2f} "
            f"{e.n_observations:>8,}  {used}"
        )
    print(
        f"\nA fitted value replaces the structural prior once it has "
        f"{threshold:,} joint\nobservations behind it (parlay."
        "min_correlation_sample). Below that the prior\nis used and the pair "
        "is reported as prior-based."
    )
    db.close()
    return 0


def cmd_profiles(cfg: Config, args) -> int:
    """List the named override bundles and exactly what each one changes."""
    if not cfg.profiles:
        print("No profiles defined.")
        return 0

    for name in sorted(cfg.profiles):
        spec = cfg.profiles[name]
        print(f"{name}")
        description = " ".join((spec.get("description") or "").split())
        if description:
            for line in _wrap(description, 74):
                print(f"  {line}")
        # Applied to a throwaway copy so listing them never mutates the
        # config the caller is about to scan with.
        preview = Config.load(args.config)
        try:
            changes = preview.apply_profile(name)
        except ValueError as exc:
            print(f"  BROKEN: {exc}")
            print()
            continue
        if changes:
            print("  changes against your current config:")
            for change in changes:
                print(f"    {change.describe()}")
        else:
            print("  nothing would change against your current config.")
        print()

    print("Apply one with, for example:  betedge parlay scan --profile nfl-week")
    print("Define your own under `profiles:` in config.yaml.")
    return 0


def _wrap(text: str, width: int) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def cmd_parlay_rosters(cfg: Config, args) -> int:
    """
    What is known about who plays for whom, and how stale it is.

    Worth looking at before trusting a same-team correlation, because the
    failure this reports is silent otherwise: an unknown team quietly
    downgrades a +0.45 prior to a blended +0.25, and a stale one is worse
    still.
    """
    from . import rosters as RO

    db = Database(cfg.database)
    sports = args.sports or list(cfg.sports) or list(RO.PROVIDERS)

    if args.refresh or cfg.parlay.roster_auto_refresh:
        report = RO.refresh_providers(
            db, sports,
            refresh_days=0 if args.refresh else cfg.parlay.roster_refresh_days,
            force=args.refresh,
        )
        for sport, n in sorted(report.refreshed.items()):
            print(f"Fetched {n:,} players for {sport} from "
                  f"{RO.provider_name(sport)}.")
        for sport, why in sorted(report.skipped.items()):
            print(f"{sport}: {why}")
        for error in report.errors:
            print(f"Could not refresh: {error}")
        if report.refreshed or report.errors:
            print()

    book = RO.load_book(
        db,
        manual_path=cfg.parlay.rosters_path,
        max_age_days=cfg.parlay.roster_max_age_days,
    )
    if book.rejected_stale:
        # Said before the "nothing known" branch below, because "you have
        # rosters and they are all too old" is a completely different
        # problem from "you have none", and the fix is different too.
        print(
            f"{book.rejected_stale:,} entr(y/ies) were dropped for being older "
            f"than {cfg.parlay.roster_max_age_days:g} days. That is deliberate: a "
            "stale roster is\nworse than none, because a wrong team puts a "
            "confident sign on a pair and\nnothing downstream questions it.\n"
        )

    if not len(book):
        print(
            "No rosters usable, so every same-game pair falls back to the "
            "blended priors.\n\nThree ways to fix that, cheapest first:\n"
            "  betedge parlay correlations --from logs.csv   learns them from "
            "the box scores\n"
            "                                                you already fit "
            "correlations from\n"
            "  betedge parlay rosters --refresh              pulls a published "
            "feed where one exists\n"
            f"  parlay.rosters_path in {args.config if hasattr(args, 'config') else 'config.yaml'}"
            "            your own player,team CSV, which wins over both"
        )
        db.close()
        return 0

    header = f"{'sport':<26} {'players':>8}  sources"
    print(header)
    print("-" * max(len(header), 60))
    for sport in book.sports():
        counts = book.source_counts(sport)
        detail = ", ".join(
            f"{src} ({n:,}"
            + (f", newest {book.newest_age(sport, src):.0f}d"
               if book.newest_age(sport, src) is not None else "")
            + ")"
            for src, n in sorted(counts.items())
        )
        print(f"{sport:<26} {sum(counts.values()):>8,}  {detail}")

    if book.ambiguous:
        print(
            f"\n{len(book.ambiguous)} name(s) are claimed by two teams within "
            "one source -- two different\nplayers who share a name. They are "
            "left unresolved rather than guessed at:"
        )
        for sport, key in sorted(book.ambiguous)[:10]:
            print(f"  {sport}: {key}")

    conflicts = book.conflicts()
    if conflicts:
        print(
            f"\n{len(conflicts)} player(s) are placed on different teams by "
            "different sources. Your\nmanual file wins, which is usually right "
            "-- but check it is not simply out of date:"
        )
        for sport, key, entries in conflicts[:10]:
            detail = "  vs  ".join(e.describe(book.today) for e in entries)
            print(f"  {sport}: {key}: {detail}")

    if args.player:
        print()
        for name in args.player:
            for sport in book.sports():
                found = book.lookup(sport, name)
                if found:
                    print(f"{name}: {found.describe(book.today)}  [{sport}]")
                    break
            else:
                print(f"{name}: unknown -- falls back to the blended same-game prior")

    db.close()
    return 0


def cmd_parlay_verify_payouts(cfg: Config, args) -> int:
    """
    Print the payout table exactly as loaded, so it can be checked against
    what the account actually offers.

    These multipliers vary by state and change without notice. They are the
    single input most likely to be silently wrong, and an EV built on the
    wrong ladder is not slightly wrong -- a 5-pick paying 10x instead of 20x
    turns a good bet into a bad one.
    """
    from . import parlay as P

    if args.init:
        return _init_payouts(cfg)

    table = P.PayoutTable.load(cfg.parlay.payouts_path)
    origin = "your own copy" if table.is_user_copy else "shipped defaults"
    print(f"Payout table: {table.path}  ({origin})")
    print(f"Last verified by you: {table.last_verified_by_user or 'never'}\n")

    # Against the shipped defaults, so an edited ladder is visibly edited.
    # Without this the loop has no feedback: you change a number, re-run,
    # and the output looks exactly as it did before.
    shipped = {}
    if table.is_user_copy:
        try:
            shipped = {
                k: v.payouts
                for k, v in P.PayoutTable.load(P.PAYOUTS_PATH).products.items()
            }
        except Exception:  # noqa: BLE001
            shipped = {}
    untouched = []

    for key in sorted(table.products):
        product = table.products[key]
        mark = "verified" if product.verified else "NOT VERIFIED"
        changed = ""
        if shipped:
            if key not in shipped:
                changed = "  [yours only]"
            elif shipped[key] != product.payouts:
                changed = "  [edited]"
            elif product.payouts:
                changed = "  [still the shipped numbers]"
                untouched.append(key)
        print(f"{key}  ({product.title}, {product.book}, {product.kind})  "
              f"[{mark}]{changed}")
        print(
            f"  same player twice: "
            f"{'allowed' if product.allows_same_player else 'not allowed'}"
            f"   |   a pushed leg: {product.void_behaviour}"
            + (f" -> {product.reduces_to}" if product.reduces_to else "")
        )
        for size, target in sorted(product.reduce_map.items()):
            print(f"    except a {size}-pick, which settles on {target}'s "
                  f"{size - 1}-pick table")
        if product.same_game_may_reduce:
            print("  NOTE: the book may cut this ladder for a lineup with "
                  "several players from one game -- which is the lineup this "
                  "tool looks for. Check the entry screen.")
        if product.note:
            for line in _wrap(product.note, width=74):
                print(f"  {line}")
        if not product.payouts:
            print("  priced by the book at entry time, not from this table")
        for legs in product.leg_counts:
            vector = product.payouts[legs]
            parts = "  ".join(
                f"{k}/{legs}: {v:g}x" for k, v in enumerate(vector) if v > 0
            )
            breakeven = product.breakeven_leg_prob(legs)
            print(
                f"  {legs}-pick   {parts or '(nothing pays)'}"
                f"   -- needs {breakeven:.1%} a leg if the legs were independent"
            )
        print()

    if table.unverified:
        print(
            "CHECK THESE BEFORE TRUSTING ANY EV NUMBER:\n  "
            + ", ".join(table.unverified)
        )
        if table.is_user_copy:
            print(
                f"\nOpen your account, read what a 2-pick, a 3-pick and a "
                f"5-pick actually pay\nin your state, edit {table.path}\n"
                "to match, and set `verified: true` on the ones you checked."
            )
        else:
            # Never send anyone to edit the shipped file: it is version
            # controlled, so their verified numbers would collide with the
            # next pull -- a merge conflict on the one input the tool most
            # needs them to get right.
            print(
                "\nThese are the SHIPPED defaults, inside the repository. Do "
                "not edit them there --\nthe next `git pull` would collide "
                "with your numbers. Make your own copy first:\n\n"
                "    betedge parlay verify-payouts --init\n\n"
                f"That writes {P.USER_PAYOUTS_PATH} (gitignored, alongside "
                "your database),\nwhich is then used in preference to the "
                "shipped file."
            )
    else:
        print("Every product is marked verified.")

    if untouched:
        print(
            "\nStill carrying the shipped numbers, unedited: "
            + ", ".join(untouched)
            + "\nThat is fine if your account really pays those, but it is "
            "also what an\nunchecked ladder looks like -- the two are "
            "indistinguishable from here."
        )
    return 0


def _init_payouts(cfg: Config) -> int:
    """Copy the shipped ladders somewhere the user can safely edit them."""
    from . import parlay as P

    target = P.USER_PAYOUTS_PATH
    if target.exists():
        print(
            f"{target} already exists -- leaving it alone.\n"
            "Edit it directly; it is already used in preference to the "
            "shipped file."
        )
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    shipped = P.PAYOUTS_PATH.read_text()
    header = (
        "# YOUR payout ladders. Copied from the betedge defaults and used in\n"
        "# preference to them, so `git pull` can never touch these numbers.\n"
        "#\n"
        "# Open your account, read what each entry size actually pays in your\n"
        "# state, correct the numbers below, and set `verified: true` on the\n"
        "# products you checked. Until you do, every ticket built from them\n"
        "# carries a `payout_table_unverified` flag.\n"
        "#\n"
        "# Check the 3-pick first: it is where the two books differ most.\n"
        "\n"
    )
    target.write_text(header + shipped)
    print(
        f"Wrote {target}.\n\n"
        "It is used in preference to the shipped file from now on, and it is\n"
        "gitignored, so your numbers survive every update.\n\n"
        "Now open your accounts and correct it. Check the 3-pick first --\n"
        "PrizePicks pays 5x and Underdog 6x on the shipped numbers, which is a\n"
        "3.5-point difference in the hit rate each one needs."
    )
    return 0


def cmd_parlay_bet(cfg: Config, args) -> int:
    db = Database(cfg.database)
    bet_id = db.place_parlay_bet(
        args.ticket_id, stake=args.stake, book=args.book, notes=args.notes
    )
    bet = db.get_parlay_bet(bet_id)
    print(
        f"Logged parlay bet #{bet_id}: {bet['n_legs']}-leg {bet['product']} "
        f"on {bet['book']} for {bet['stake']:,.0f} "
        f"(EV {bet['ev_at_bet']:+.1%}, P(all) {bet['joint_prob_at_bet']:.1%})"
    )
    print(f"Settle it with:  betedge parlay settle {bet_id} --hit <legs>")
    db.close()
    return 0


def cmd_parlay_settle(cfg: Config, args) -> int:
    db = Database(cfg.database)
    pnl = db.settle_parlay_bet(
        args.bet_id, args.hit, legs_void=args.void,
        payouts_path=cfg.parlay.payouts_path,
    )
    bet = db.get_parlay_bet(args.bet_id)
    print(
        f"Parlay bet #{args.bet_id} settled {bet['status']} "
        f"({args.hit}/{bet['n_legs']} legs"
        + (f", {args.void} void" if args.void else "")
        + f"): {pnl:+,.2f}"
    )
    db.close()
    return 0


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="betedge", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser(
        "daily",
        help="budgeted scan + shortlist + closing lines. The one to run.",
    )
    s.add_argument("--limit", type=int, default=12, help="rows in the shortlist")
    s.add_argument("--profile", metavar="NAME",
                   help="apply a named bundle of overrides before scanning, "
                        "e.g. nfl-week. Every change it makes is printed. "
                        "See `betedge profiles`.")
    s.add_argument("--bankroll", type=float)
    s.add_argument("--min-ev", type=float, help="e.g. 0.03 for +3%%")
    s.add_argument("--max-events", type=int,
                   help="cap events per sport in the prop pass")
    s.add_argument("--window", type=float, default=20.0,
                   help="minutes before start to capture closing lines")
    s.add_argument("--no-close", action="store_true",
                   help="skip closing-line capture")
    s.add_argument("--hide-suspect", action="store_true")
    s.add_argument("--no-report", action="store_true")
    s.set_defaults(func=cmd_daily)

    s = sub.add_parser(
        "profiles",
        help="named bundles of config overrides, and what each one changes",
    )
    s.set_defaults(func=cmd_profiles)

    s = sub.add_parser(
        "diagnose",
        help="price the whole board and show the distribution, not just hits",
    )
    s.add_argument("--top", type=int, default=15,
                   help="how many near-misses to list")
    s.add_argument("--sports", nargs="+", help="override configured prop sports")
    s.add_argument("--no-props", action="store_true",
                   help="game lines only, the cheap pass")
    s.add_argument("--max-events", type=int)
    s.set_defaults(func=cmd_diagnose)

    s = sub.add_parser("budget", help="credits left and today's allowance")
    s.add_argument("--days", type=int, default=14,
                   help="days of spending history to show")
    s.set_defaults(func=cmd_budget)

    s = sub.add_parser("sports", help="list sport keys and prop coverage")
    s.add_argument("--all", action="store_true", help="include unconfigured sports")
    s.set_defaults(func=cmd_sports)

    s = sub.add_parser("quota", help="check remaining credits and scan cost")
    s.set_defaults(func=cmd_quota)

    s = sub.add_parser("scan", help="find and rank bets")
    s.add_argument("--sports", nargs="+", help="override configured sports")
    s.add_argument("--min-ev", type=float, help="e.g. 0.03 for +3%%")
    s.add_argument("--profile", metavar="NAME",
                   help="apply a named bundle of overrides before scanning, "
                        "e.g. nfl-week. Every change it makes is printed. "
                        "See `betedge profiles`.")
    s.add_argument("--bankroll", type=float)
    s.add_argument("--limit", type=int, default=40)
    s.add_argument("--max-events", type=int, help="cap events per sport (saves credits)")
    s.add_argument("--alternate", action="store_true", help="include alternate lines")
    s.add_argument("--core-sports", nargs="+", metavar="KEY",
                   help="game-level markets for these sports; a trailing * matches "
                        "by prefix, e.g. 'tennis_*'. Cheap: ~3 credits per sport.")
    s.add_argument("--no-props", action="store_true",
                   help="skip the expensive per-event prop scan entirely")
    s.add_argument("--markets", nargs="+", metavar="KEY",
                   help="only these prop markets, e.g. pitcher_strikeouts "
                        "batter_total_bases. Cost is markets x events.")
    s.add_argument("--all-lines", action="store_true",
                   help="do not collapse to one row per player and market")
    s.add_argument("--hide-suspect", action="store_true")
    s.add_argument("--no-report", action="store_true")
    s.set_defaults(func=cmd_scan)

    s = sub.add_parser("show", help="re-print a scan")
    s.add_argument("scan_id", nargs="?", type=int)
    s.add_argument("--ids", action="store_true", help="(ids are always shown)")
    s.add_argument("--include-suspect", action="store_true")
    s.add_argument("--all", action="store_true",
                   help="include bets whose event has already started")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("bet", help="log a bet you placed")
    s.add_argument("opportunity_id", type=int, nargs="?")
    s.add_argument("--stake", type=float, required=True)
    s.add_argument("--price", type=P.parse_price,
                   help="the price you actually got, American (+122, -110) "
                        "or decimal (2.22). Negative or >=100 is read as "
                        "American.")
    s.add_argument("--book")
    s.add_argument("--notes")
    s.add_argument("--settle", choices=["won", "lost", "push", "void",
                                        "half_won", "half_lost"],
                   help="settle it in the same breath, for a bet logged afterwards")
    g = s.add_argument_group(
        "manual bet details",
        "for a bet the scanner didn't surface. Omitted fields are taken "
        "from the opportunity when an id is given.",
    )
    g.add_argument("--selection", help='e.g. "Adam Trautman"')
    g.add_argument("--side", help='"Over", "Under", or the competitor for a moneyline')
    g.add_argument("--line", type=float, help="e.g. 7.5")
    g.add_argument("--market", help="e.g. player_reception_yds")
    g.add_argument("--sport", help="e.g. americanfootball_nfl")
    g.add_argument("--event-id", dest="event_id",
                   help="needed for closing-line capture")
    g.add_argument("--commence", help="ISO start time, e.g. 2026-09-15T00:15:00Z")
    g.add_argument("--matchup", help='e.g. "Denver Broncos @ Kansas City Chiefs"')
    s.set_defaults(func=cmd_bet)

    s = sub.add_parser("settle", help="settle a bet")
    s.add_argument("bet_id", type=int)
    s.add_argument("status", choices=["won", "lost", "push", "void", "half_won", "half_lost"])
    s.set_defaults(func=cmd_settle)

    s = sub.add_parser("delete", help="remove bets from the ledger")
    s.add_argument("bet_ids", type=int, nargs="+",
                   help="bet ids, as `betedge export` or `betedge report` list them")
    s.add_argument("--yes", action="store_true",
                   help="skip the confirmation prompt")
    s.set_defaults(func=cmd_delete)

    s = sub.add_parser("close", help="capture closing lines for open bets")
    s.add_argument("--window", type=float, default=20.0,
                   help="minutes before start to start capturing")
    s.add_argument("--no-parlays", action="store_true",
                   help="skip closing-line capture for multi-leg tickets")
    s.set_defaults(func=cmd_close)

    # ---- parlay / pick'em -------------------------------------------
    p_parlay = sub.add_parser(
        "parlay",
        help="multi-leg pick'em and parlay tickets",
        description="Build multi-leg tickets whose joint probability beats "
                    "what the payout structure assumes.",
    )
    psub = p_parlay.add_subparsers(dest="parlay_command", required=True)

    s = psub.add_parser("scan", help="build and rank tickets")
    s.add_argument("--sports", nargs="+", help="override configured sports")
    s.add_argument("--products", nargs="+", metavar="KEY",
                   help="payout structures to build for, e.g. underdog_standard")
    s.add_argument("--profile", metavar="NAME",
                   help="apply a named bundle of overrides before scanning, "
                        "e.g. nfl-week. Every change it makes is printed. "
                        "See `betedge profiles`.")
    s.add_argument("--limit", type=int, default=8, help="tickets to print")
    s.add_argument("--bankroll", type=float)
    s.add_argument("--min-ev", type=float, help="e.g. 0.03 for +3%%")
    s.add_argument("--min-leg-edge", type=float, metavar="X",
                   help="how far below the ladder's break-even a single leg "
                        "may sit, e.g. -0.10. Lower it to see what "
                        "correlation ALONE would build -- those tickets are "
                        "flagged suspect and staked at zero, so this is a "
                        "diagnostic, not a way to place more bets.")
    s.add_argument("--max-legs", type=int)
    s.add_argument("--draws", type=int,
                   help="Monte Carlo draws for the final estimate")
    s.add_argument("--max-events", type=int,
                   help="cap events per sport (saves credits)")
    s.add_argument("--rosters", metavar="FILE",
                   help="player,team CSV or YAML. Without it, two players in "
                        "one game cannot be told apart as team mates or "
                        "opponents and the weaker blended priors are used.")
    s.add_argument("--interpolate", action="store_true",
                   help="estimate a leg's probability from Pinnacle's "
                        "neighbouring lines when it does not price the exact "
                        "one. Off by default; marks every leg it touches.")
    s.add_argument("--grouping", choices=["same_game", "same_slate", "any"],
                   help="which legs may be combined (default same_game)")
    s.add_argument("--offered-price", type=float, metavar="DECIMAL",
                   help="the parlay price the book actually shows, applied to "
                        "tickets of --max-legs legs. For a same-game parlay "
                        "this is well below the product of the legs, so "
                        "without it the EV is an upper bound the book will "
                        "never pay.")
    s.add_argument("--ignore-budget", action="store_true",
                   help="ignore the daily credit allowance")
    s.add_argument("--hide-suspect", action="store_true")
    s.add_argument("--no-report", action="store_true")
    s.set_defaults(func=cmd_parlay_scan)

    s = psub.add_parser(
        "coverage",
        help="probe the API for usable prop coverage, per sport",
    )
    s.add_argument("--sports", nargs="+")
    s.add_argument("--max-events", type=int, default=2,
                   help="events to probe per sport. This bills like a scan.")
    s.add_argument("--books", nargs="+", metavar="KEY",
                   help="books to ask about. Defaults to your configured "
                        "books plus the known pick'em sites. Asking for more "
                        "costs nothing -- ten books bill as one region.")
    s.add_argument("--window", type=float, metavar="HOURS",
                   help="look this far ahead instead of the configured prop "
                        "window. A probe asks what EXISTS, so it is worth "
                        "looking further out than you would bet.")
    s.add_argument("--profile", metavar="NAME",
                   help="apply a named bundle of overrides first, e.g. "
                        "nfl-week. See `betedge profiles`.")
    s.set_defaults(func=cmd_parlay_coverage)

    s = psub.add_parser(
        "correlations",
        help="fit pairwise correlations from game logs, and show what is stored",
    )
    s.add_argument("--from", dest="source", metavar="CSV",
                   help="game log to fit from. Columns: game_id, sport, "
                        "player, team, market, value[, date]")
    s.add_argument("--markets", nargs="+", help="only fit these market keys")
    s.add_argument("--max-pairs", type=int, default=200_000,
                   help="cap on joint observations kept per market pair")
    s.set_defaults(func=cmd_parlay_correlations)

    s = psub.add_parser(
        "rosters",
        help="who plays for whom, where it came from, and how stale it is",
    )
    s.add_argument("--refresh", action="store_true",
                   help="fetch the roster feed now, ignoring the refresh interval")
    s.add_argument("--sports", nargs="+")
    s.add_argument("--player", nargs="+", metavar="NAME",
                   help="look up specific players")
    s.set_defaults(func=cmd_parlay_rosters)

    s = psub.add_parser(
        "verify-payouts",
        help="print the payout table so it can be checked against your account",
    )
    s.add_argument("--init", action="store_true",
                   help="copy the shipped ladders to data/payouts.yaml, where "
                        "you can edit them without a git pull overwriting your "
                        "numbers")
    s.set_defaults(func=cmd_parlay_verify_payouts)

    s = psub.add_parser("bet", help="log a multi-leg entry you placed")
    s.add_argument("ticket_id", type=int)
    s.add_argument("--stake", type=float, required=True)
    s.add_argument("--book")
    s.add_argument("--notes")
    s.set_defaults(func=cmd_parlay_bet)

    s = psub.add_parser("settle", help="settle an entry by how many legs landed")
    s.add_argument("bet_id", type=int)
    s.add_argument("--hit", type=int, required=True, metavar="N",
                   help="legs that won")
    s.add_argument("--void", type=int, default=0, metavar="N",
                   help="legs that pushed and shrank the entry")
    s.set_defaults(func=cmd_parlay_settle)

    p_rt = sub.add_parser(
        "rt",
        help="Rotten Tomatoes threshold contracts on Kalshi",
        description="Price Kalshi contracts on a film's Tomatometer. The "
                    "score is a running proportion, so a threshold is "
                    "often already decided by arithmetic.",
    )
    rtsub = p_rt.add_subparsers(dest="rt_command", required=True)

    s = rtsub.add_parser("contracts", help="print the contracts as loaded")
    s.add_argument("--contracts", metavar="PATH", help="a contracts file")
    s.add_argument("--init", action="store_true",
                   help="copy the template somewhere you can edit it")
    s.set_defaults(func=lambda cfg, args: (
        _init_rt_contracts(cfg) if args.init else cmd_rt_contracts(cfg, args)
    ))

    s = rtsub.add_parser("snapshot", help="read a film's Tomatometer now")
    s.add_argument("slug", help="the part of the URL after /m/")
    s.add_argument("--cache", metavar="DIR", help="where to cache pages")
    s.add_argument("--fresh", action="store_true", help="ignore the cache")
    s.set_defaults(func=cmd_rt_snapshot)

    s = rtsub.add_parser("scan", help="price every contract and size it")
    s.add_argument("--contracts", metavar="PATH", help="a contracts file")
    s.add_argument("--bankroll", type=float)
    s.add_argument("--size", type=int, default=100, metavar="N",
                   help="contracts to price the fill for. The book is "
                        "walked to this depth, so the price quoted is one "
                        "you could actually get (default 100)")
    s.add_argument("--price", type=float, metavar="P",
                   help="skip Kalshi and use this price, 0..1, for every "
                        "contract. For checking the maths by hand")
    s.add_argument("--cache", metavar="DIR", help="where to cache pages")
    s.set_defaults(func=cmd_rt_scan)

    s = sub.add_parser(
        "compare",
        help="single bets against multi-leg tickets, on the same metrics",
    )
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("report", help="performance and closing-line value")
    s.set_defaults(func=cmd_report)

    s = sub.add_parser(
        "export",
        help="dump bets to CSV, or to your Excel tracker if the path ends .xlsx",
    )
    s.add_argument("path")
    s.add_argument("--template", metavar="XLSX",
                   help="workbook to copy for an .xlsx export; defaults to "
                        "betedge/templates/tracker_template.xlsx")
    s.set_defaults(func=cmd_export)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    cfg = Config.load(args.config)
    try:
        return args.func(cfg, args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:  # noqa: BLE001
        if args.verbose:
            raise
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
