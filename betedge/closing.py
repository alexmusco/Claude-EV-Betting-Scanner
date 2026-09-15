"""
Closing-line capture.

The closing line is the market's final word on a game. Beating it
consistently is the only fast evidence that a betting model is real --
profit and loss needs hundreds of bets to separate skill from variance,
whereas closing-line value shows up in dozens.

Run this shortly BEFORE each event starts (props are usually pulled at tip
or kickoff, so waiting until afterwards leaves you with nothing to
capture). A cron entry every 15 minutes is the simple approach:

    */15 * * * * cd /path/to/betedge && python -m betedge close

Each run costs one credit per market per event with an open bet, and only
for events inside the capture window, so it stays cheap.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from . import pricing
from .config import Config
from .db import Database, parse_timestamp
from .oddsapi import OddsApiClient
from .scan import Quote, group_key, group_sharp_markets, outcome_key, parse_event_odds

log = logging.getLogger(__name__)


def capture_closing_lines(
    cfg: Config,
    client: OddsApiClient,
    db: Database,
    window_minutes: float = 20.0,
    now: datetime | None = None,
) -> dict[str, int]:
    """
    For every pending bet whose event starts within `window_minutes` (or has
    already started), pull the sharp book's current price on that exact
    market and line and store it as the close.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now + timedelta(minutes=window_minutes)

    pending = db.open_bets()
    already = {
        r["bet_id"]
        for r in db.conn.execute(
            "SELECT bet_id FROM closing_lines WHERE bet_id IS NOT NULL"
        ).fetchall()
    }

    # Group the work by event so one API call serves every bet on that game.
    by_event: dict[tuple[str, str], list] = {}
    for bet in pending:
        if bet["id"] in already:
            continue
        start = parse_timestamp(bet["commence_time"])
        if start is None or start > cutoff:
            continue
        by_event.setdefault((bet["sport"], bet["event_id"]), []).append(bet)

    stats = {"events_checked": 0, "captured": 0, "not_found": 0, "errors": 0}

    for (sport, event_id), bets in by_event.items():
        markets = sorted({b["market"] for b in bets if b["market"]})
        if not markets:
            continue
        try:
            payload = client.event_odds(sport, event_id, markets, [cfg.books.sharp])
        except Exception as exc:  # noqa: BLE001
            log.warning("closing pull failed for %s: %s", event_id, exc)
            stats["errors"] += 1
            continue

        stats["events_checked"] += 1
        if not payload:
            stats["not_found"] += len(bets)
            continue

        _meta, quotes = parse_event_odds(payload)
        groups = group_sharp_markets(quotes, cfg.books.sharp)

        for bet in bets:
            # Rebuild the quote shape the grouping keys expect, so a
            # moneyline or spread bet is located the same way a prop is.
            probe = Quote(
                book=cfg.books.sharp,
                market=bet["market"],
                selection=bet["selection"],
                side=bet["side"],
                line=float(bet["line"]) if bet["line"] is not None else None,
                price=2.0,
                last_update=None,
            )
            group = groups.get(group_key(probe))
            if group is None:
                stats["not_found"] += 1
                continue

            ordered = sorted(group.items(), key=lambda kv: str(kv[0]))
            try:
                idx = [k for k, _ in ordered].index(outcome_key(probe))
            except ValueError:
                stats["not_found"] += 1
                continue

            prices = [q.price for _, q in ordered]
            taken = prices[idx]
            others = [p for i, p in enumerate(prices) if i != idx]
            other = others[0] if len(others) == 1 else min(others)

            probs = pricing.fair_probs_all_methods(prices)
            method = cfg.model.devig_method
            if method == "worst_case":
                fair_close = min(p[idx] for p in probs.values())
            else:
                fair_close = probs[method][idx]

            db.record_closing_line(
                opportunity_id=bet["opportunity_id"],
                bet_id=bet["id"],
                sharp_price_taken=taken,
                sharp_price_other=other,
                fair_prob_close=fair_close,
                price_taken=bet["price"],
                fair_prob_at_bet=bet["fair_prob_at_bet"],
                captured_at=now,
            )
            stats["captured"] += 1

    return stats
