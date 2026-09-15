"""
Command line interface.

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
from datetime import datetime
from pathlib import Path

from . import report as R
from .closing import capture_closing_lines
from .config import Config
from .db import Database
from .markets import SPORTS, core_markets_for, expand_sport_keys, markets_for
from .oddsapi import OddsApiClient
from .scan import best_per_selection, cap_exposure, scan


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
    client = build_client(cfg)
    q = client.probe_quota()
    print(f"Credits remaining : {q.remaining:,}" if q.remaining is not None else "unknown")
    print(f"Credits used      : {q.used:,}" if q.used is not None else "")
    for sport in cfg.sports:
        markets = cfg.markets_for_sport(sport, cfg.model.include_alternate_lines)
        try:
            events = client.events(sport)
        except Exception as exc:  # noqa: BLE001
            print(f"{sport}: {exc}")
            continue
        cost = len(events) * len(markets)
        print(
            f"{sport:30} {len(events):3d} events x {len(markets):2d} markets "
            f"= up to {cost:,} credits per prop scan"
        )

    if cfg.core_sports:
        live = [s["key"] for s in client.sports()]
        resolved = expand_sport_keys(cfg.core_sports, live)
        total = 0
        print()
        for sport in resolved:
            n = len(core_markets_for(sport))
            total += n
            print(f"{sport:30} {n:2d} core markets = {n} credits for the whole sport")
        print(f"{'':30} core total: {total} credits per sweep")
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

    print(R.scan_summary(result))
    print()
    print(R.console_table(result.clean, limit=args.limit))
    if result.suspect and not args.hide_suspect:
        print("\nSuspect (sanity checks tripped, no stake recommended):")
        print(R.console_table(result.suspect, limit=10))

    db = Database(cfg.database)
    scan_id = db.record_scan(result)
    print(f"\nLogged as scan #{scan_id}.")
    if result.clean:
        print("Bet one with:  betedge bet <id> --stake <amount>")
        print("Opportunity ids are shown by:  betedge show --ids")

    if not args.no_report:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        path = R.write_report(
            Path(cfg.reports_dir) / f"scan_{stamp}.html",
            R.scan_report_html(result, cfg, title=f"Scan {stamp}"),
        )
        print(f"Report: {path}")
    db.close()
    return 0


def cmd_show(cfg: Config, args) -> int:
    db = Database(cfg.database)
    scan_id = args.scan_id or db.latest_scan_id()
    if scan_id is None:
        print("No scans recorded yet.")
        return 1
    rows = db.opportunities_for_scan(scan_id)
    if not rows:
        print(f"Scan #{scan_id} flagged nothing.")
        return 0
    print(f"Scan #{scan_id}\n")
    header = f"{'id':>5}  {'EV':>7}  {'bet':<44} {'book':<12} {'price':>7} {'stake':>7}  game"
    print(header)
    print("-" * len(header))
    for r in rows:
        if r["suspect"] and not args.include_suspect:
            continue
        desc = R.describe(r["selection"], r["side"], r["line"])
        print(
            f"{r['id']:>5}  {r['ev']:>+6.1%}  {desc:<44.44} {r['soft_book']:<12} "
            f"{r['soft_price']:>7.2f} {(r['recommended_stake'] or 0):>7,.0f}  "
            f"{r['away_team']} @ {r['home_team']}"
        )
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
        f"{R.describe(bet['selection'], bet['side'], bet['line'])} "
        f"@ {bet['price']:.2f} on {bet['book']} for {bet['stake']:,.0f}"
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

        summary = export_tracker(rows, path)
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
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("bet", help="log a bet you placed")
    s.add_argument("opportunity_id", type=int, nargs="?")
    s.add_argument("--stake", type=float, required=True)
    s.add_argument("--price", type=float, help="the price you actually got")
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
