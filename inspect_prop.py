#!/usr/bin/env python3
"""
Show every raw quote for one player, from every book, and explain exactly
what the scan engine does with it.

    python3 inspect_prop.py Trautman
    python3 inspect_prop.py Trautman --market player_reception_yds
    python3 inspect_prop.py "Bo Nix" --sport americanfootball_nfl

Costs about 1 credit per market it has to ask for. Use it when the console
output and the sportsbook screen disagree -- it prints the numbers the API
actually returned, so you can see whether the disagreement is a moved line,
a different side, a stale feed, or a real bug.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from betedge import pricing as P
from betedge.config import Config
from betedge.cli import build_client
from betedge.markets import markets_for
from betedge.scan import evaluate_event, pair_sharp_quotes, parse_event_odds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("player", help="any part of the player's name, case-insensitive")
    ap.add_argument("--sport", default=None, help="default: the first sport in config.yaml")
    ap.add_argument("--market", default=None, help="one market key, to keep the cost at 1 credit")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    sport = args.sport or cfg.sports[0]
    client = build_client(cfg)
    needle = args.player.lower()
    now = datetime.now(timezone.utc)

    markets = [args.market] if args.market else markets_for(
        sport, cfg.model.include_alternate_lines
    )
    books = cfg.books.all

    events = client.events(sport)
    if not events:
        print(f"No upcoming {sport} events.")
        return 1

    print(f"Searching {len(events)} {sport} event(s) for '{args.player}'\n")

    found_any = False
    for event in events:
        payload = client.event_odds(sport, event["id"], markets, books)
        if not payload:
            continue
        meta, quotes = parse_event_odds(payload)
        hits = [q for q in quotes if needle in (q.selection or "").lower()]
        if not hits:
            continue

        found_any = True
        print("=" * 78)
        print(f"{meta['away_team']} @ {meta['home_team']}")
        mins = (meta["commence_time"] - now).total_seconds() / 60
        print(f"starts in {mins/60:.1f}h  ({meta['commence_time'].astimezone():%a %d %b %H:%M %Z})")
        print("=" * 78)

        # ---- raw quotes, exactly as the API returned them ----------------
        print("\nRAW QUOTES\n")
        print(f"{'book':<12} {'market':<26} {'side':<6} {'line':>7} {'price':>7} {'age':>7}")
        print("-" * 72)
        for q in sorted(hits, key=lambda x: (x.market, x.line or 0, x.book, x.side)):
            age = ""
            if q.last_update:
                age = f"{(now - q.last_update).total_seconds()/60:.0f}m"
            line = "-" if q.line is None else f"{q.line:g}"
            print(f"{q.book:<12} {q.market:<26} {q.side:<6} {line:>7} {q.price:>7.2f} {age:>7}")

        # ---- what the engine can pair ------------------------------------
        pairs = pair_sharp_quotes(hits, cfg.books.sharp)
        print(f"\nTWO-SIDED {cfg.books.sharp.upper()} MARKETS: {len(pairs)}")
        if not pairs:
            print(f"  None. Without both sides there is nothing to de-vig, so every")
            print(f"  soft quote on this player is skipped.")
        for (market, _sel, line), (over, under) in sorted(pairs.items(), key=lambda kv: str(kv[0])):
            prices = [over.price, under.price]
            orr = P.overround(prices)
            spread = P.devig_spread(prices)
            line_s = "-" if line is None else f"{line:g}"
            print(f"\n  {market} @ {line_s}")
            print(f"    {cfg.books.sharp}: over {over.price:.2f} / under {under.price:.2f}"
                  f"   overround {orr*100:.2f}%")
            for name, probs in P.fair_probs_all_methods(prices).items():
                print(f"      {name:<16} fair over {1/probs[0]:.3f}  fair under {1/probs[1]:.3f}")
            print(f"    de-vig spread {spread*100:.2f}pp"
                  f"  (guard trips above {cfg.model.max_devig_spread*100:.0f}pp)")
            if not (cfg.model.min_overround <= orr <= cfg.model.max_overround):
                print(f"    !! overround outside "
                      f"{cfg.model.min_overround:.1%}-{cfg.model.max_overround:.1%} -- SKIPPED")

            # soft quotes on this exact market and line
            softs = [q for q in hits if q.book in cfg.books.soft
                     and q.market == market and q.line == line]
            if not softs:
                soft_lines = sorted({q.line for q in hits if q.book in cfg.books.soft
                                     and q.market == market and q.line is not None})
                print(f"    No soft quote at this line."
                      + (f" Soft books are on: {soft_lines}" if soft_lines else ""))
            for q in sorted(softs, key=lambda x: (x.book, x.side)):
                idx = 0 if (q.side or "").lower() in ("over", "yes") else 1
                by = {n: p[idx] for n, p in P.fair_probs_all_methods(prices).items()}
                fair = min(by.values()) if cfg.model.devig_method == "worst_case" \
                    else P.devig(prices, method=cfg.model.devig_method)[idx]
                ev = P.expected_value(fair, q.price)
                evs = [P.expected_value(p, q.price) for p in by.values()]
                verdict = (
                    f"SUSPECT (EV above {cfg.model.max_plausible_ev:.0%})"
                    if ev > cfg.model.max_plausible_ev
                    else ("FLAG" if ev >= cfg.model.min_ev else
                          f"below the {cfg.model.min_ev:.0%} bar")
                )
                print(f"    {q.book} {q.side} {q.price:.2f}"
                      f"  -> EV {ev:+.2%}  (range {min(evs):+.2%} to {max(evs):+.2%})"
                      f"  {verdict}")

        # ---- and what the real engine decides ----------------------------
        rej: dict[str, int] = {}
        opps = [o for o in evaluate_event(meta, quotes, cfg, now=now, rejections=rej)
                if needle in o.selection.lower()]
        print(f"\nENGINE VERDICT: {len(opps)} opportunit{'y' if len(opps)==1 else 'ies'}"
              f" for this player")
        for o in opps:
            print(f"  {o.description:<32} {o.soft_book:<12} {o.soft_price:.2f}"
                  f"  EV {o.ev:+.2%}  {'SUSPECT' if o.suspect else 'playable'}"
                  + (f"  flags: {','.join(o.flags)}" if o.flags else ""))
        if rej:
            print("\n  Event-wide rejection counts (all players):")
            for k, v in sorted(rej.items(), key=lambda kv: -kv[1]):
                print(f"    {k:<34} {v}")

    if not found_any:
        print(f"No quotes found for '{args.player}' at any book.")
        print("Check the spelling, or the player may not be posted.")

    print(f"\nCredits used by this check: {client.quota.spent_this_session}"
          + (f", {client.quota.remaining:,} remaining" if client.quota.remaining else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
