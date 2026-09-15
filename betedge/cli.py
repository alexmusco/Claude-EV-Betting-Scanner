"""
Command line interface.

    betedge daily                         the one you want: budgeted scan + shortlist
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
    betedge export <file.csv>             dump bets to CSV
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
from .closing import capture_closing_lines
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
    if not bet["event_id"]:
        print("  no event id — `betedge close` won't be able to capture a "
              "closing line for this one.")
    db.close()
    return 0


def cmd_settle(cfg: Config, args) -> int:
    db = Database(cfg.database)
    pnl = db.settle_bet(args.bet_id, args.status)
    print(f"Bet #{args.bet_id} settled {args.status}: {pnl:+,.2f}")
    db.close()
    return 0


def cmd_close(cfg: Config, args) -> int:
    db = Database(cfg.database)
    client = build_client(cfg)
    stats = capture_closing_lines(cfg, client, db, window_minutes=args.window)
    _log_spend(db, client, "close")
    print(
        f"Checked {stats['events_checked']} events, captured {stats['captured']} "
        f"closing lines, {stats['not_found']} not found, {stats['errors']} errors. "
        f"({client.quota.spent_this_session} credits)"
    )
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

    path = R.write_report(
        Path(cfg.reports_dir) / "performance.html", R.performance_report_html(db, cfg)
    )
    print(f"\nReport: {path}")
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

    s = sub.add_parser("close", help="capture closing lines for open bets")
    s.add_argument("--window", type=float, default=20.0,
                   help="minutes before start to start capturing")
    s.set_defaults(func=cmd_close)

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
