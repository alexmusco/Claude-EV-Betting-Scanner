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


# --------------------------------------------------------------------------
# Multi-leg tickets
#
# Ticket-level closing-line value is the only fast read on whether the
# multi-leg model works at all. Profit and loss on parlays is hopeless as
# evidence -- a 20x ticket that misses by one leg looks identical to one
# that was never close, and a hundred settled entries still tell you
# nothing. But re-pricing every leg at the close and pushing the result
# back through the same copula gives a directly comparable number, on
# every ticket, bet or not.
# --------------------------------------------------------------------------


def capture_parlay_closing_lines(
    cfg: Config,
    client: OddsApiClient,
    db: Database,
    window_minutes: float = 20.0,
    now: datetime | None = None,
) -> dict[str, int]:
    """
    Re-price each leg of every open ticket at the sharp book's closing
    number, then recompute the ticket's joint probability from the closed
    legs.

    Legs are captured as their own events approach, because a ticket can
    span games hours apart and the first leg's market is pulled long
    before the last one's. The ticket-level row is written once every leg
    has a close, so `joint_prob_close` is always a complete picture rather
    than a mix of closed and stale numbers.
    """
    from . import correlation as corr_mod
    from .parlay import PayoutTable, estimate_push_prob, sharp_ladder

    now = now or datetime.now(timezone.utc)
    cutoff = now + timedelta(minutes=window_minutes)
    stats = {"tickets_checked": 0, "events_checked": 0, "legs_captured": 0,
             "tickets_closed": 0, "not_found": 0, "errors": 0}

    tickets = db.parlay_tickets_for_closing()
    if not tickets:
        return stats

    # One API call per event serves every leg on that game across every
    # ticket, which is what keeps this inside the reserve.
    outstanding: dict[tuple[str, str], list] = {}
    for ticket in tickets:
        stats["tickets_checked"] += 1
        for leg in db.parlay_legs(ticket["id"]):
            if leg["closed_at"] is not None:
                continue
            start = parse_timestamp(leg["commence_time"])
            if start is None or start > cutoff:
                continue
            outstanding.setdefault((leg["sport"], leg["event_id"]), []).append(leg)

    for (sport, event_id), legs in outstanding.items():
        markets = sorted({leg["market"] for leg in legs if leg["market"]})
        if not markets:
            continue
        try:
            payload = client.event_odds(sport, event_id, markets, [cfg.books.sharp])
        except Exception as exc:  # noqa: BLE001
            log.warning("parlay closing pull failed for %s: %s", event_id, exc)
            stats["errors"] += 1
            continue

        stats["events_checked"] += 1
        if not payload:
            stats["not_found"] += len(legs)
            continue

        _meta, quotes = parse_event_odds(payload)
        ladder = sharp_ladder(quotes, cfg.books.sharp, cfg.model.devig_method)

        for leg in legs:
            rungs = ladder.get((leg["market"], leg["selection"]))
            line = leg["line"]
            if not rungs or line is None or float(line) not in rungs:
                stats["not_found"] += 1
                continue
            rung = rungs[float(line)]
            push_close, _source = estimate_push_prob(
                rungs, float(line), cfg.parlay.assumed_push_prob
            )
            db.record_parlay_leg_close(
                leg["id"],
                fair_prob_close=rung.prob_for(leg["side"]),
                push_prob_close=push_close,
                captured_at=now,
            )
            stats["legs_captured"] += 1

    # Recompute the joint probability for any ticket now fully closed.
    table = PayoutTable.load(cfg.parlay.payouts_path)
    priors = corr_mod.PriorSet.load(cfg.parlay.priors_path)
    estimates = corr_mod.EstimateStore.from_db(db)
    already = {
        r["ticket_id"]
        for r in db.conn.execute(
            "SELECT ticket_id FROM parlay_closing_lines"
        ).fetchall()
    }
    bets_by_ticket = {
        r["ticket_id"]: r["id"] for r in db.open_parlay_bets()
    }

    for ticket in tickets:
        if ticket["id"] in already:
            continue
        legs = db.parlay_legs(ticket["id"])
        closed = [leg for leg in legs if leg["fair_prob_close"] is not None]
        if len(closed) != len(legs) or not legs:
            continue
        try:
            joint_close, ev_close = _reprice_ticket(
                ticket, closed, cfg, table, priors, estimates
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("could not reprice ticket %s: %s", ticket["id"], exc)
            stats["errors"] += 1
            continue
        db.record_parlay_closing_line(
            ticket_id=ticket["id"],
            parlay_bet_id=bets_by_ticket.get(ticket["id"]),
            legs_captured=len(closed),
            n_legs=len(legs),
            joint_prob_close=joint_close,
            joint_prob_at_bet=ticket["joint_prob"],
            ev_close=ev_close,
            ev_at_bet=ticket["ev"],
            captured_at=now,
        )
        stats["tickets_closed"] += 1

    return stats


def _reprice_ticket(ticket, closed_legs, cfg, table, priors, estimates):
    """
    The ticket as the closing market sees it.

    The correlation matrix is rebuilt from the same priors and estimates
    rather than stored and reused, so the comparison isolates the thing
    that actually moved -- the market's view of each leg -- instead of
    mixing it with a change in our own correlation inputs.
    """
    from . import copula, correlation as corr_mod
    from .parlay import Product

    probes = [_ClosingLeg(row) for row in closed_legs]
    corr = corr_mod.assemble(
        probes, priors=priors, estimates=estimates,
        min_sample=cfg.parlay.min_correlation_sample,
    )
    sim = copula.simulate(
        [p.hit_prob for p in probes],
        corr.matrix,
        [p.push_prob for p in probes],
        draws=cfg.parlay.draws,
        seed=cfg.parlay.seed,
    )
    product = table.products.get(ticket["product"])
    if product is None:
        # A priced parlay: the entry pays what it was logged at.
        n = len(probes)
        product = Product(
            key=ticket["product"], book=ticket["book"] or "", kind="parlay",
            title=ticket["product"],
            payouts={n: tuple([0.0] * n + [float(ticket["payout_all_hit"] or 0.0)])},
            verified=True, allows_same_player=True,
        )
    multiples = product.multiple_grid(len(probes))
    ev_close = float(sim.payout_per_draw(multiples).mean()) - 1.0
    return sim.joint_prob, ev_close


class _ClosingLeg:
    """
    A stored leg re-read at its closing probability.

    Shaped so `correlation.assemble` can treat it exactly like a live
    `parlay.Leg`: it needs the sport, event, player, market and side to
    work out the relation and the sign, and nothing else.
    """

    __slots__ = ("sport", "event_id", "selection", "market", "side", "team",
                 "fair_prob", "push_prob")

    def __init__(self, row):
        self.sport = row["sport"]
        self.event_id = row["event_id"]
        self.selection = row["selection"]
        self.market = row["market"]
        self.side = row["side"]
        self.team = row["team"]
        self.fair_prob = float(row["fair_prob_close"])
        self.push_prob = float(row["push_prob_close"] or 0.0)

    @property
    def hit_prob(self) -> float:
        return self.fair_prob * (1.0 - self.push_prob)
