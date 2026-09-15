"""
The scan engine: turn raw API payloads into ranked, de-vigged betting
opportunities.

Method
------
1. Normalise every bookmaker payload into flat quotes.
2. For the sharp book, pair the two sides of each (market, player, line)
   and de-vig them into a fair probability.
3. For each soft book quote on that exact same (market, player, line, side),
   compute EV = fair_prob * soft_price - 1.
4. Reject anything that trips a sanity guard, and mark as "suspect"
   anything whose edge is too large to believe.

Step 4 is the part that matters most in practice. A scanner that only does
steps 1-3 will hand you a list dominated by stale numbers and mismatched
players, all of which look like enormous edges. The old R model had no such
guards -- it flagged a +21% EV assist prop as a normal pick.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from . import liquidity, pricing
from .config import Config
from .markets import estimate_credits, expand_sport_keys
from .oddsapi import CreditBudgetExceeded, OddsApiClient

log = logging.getLogger(__name__)

OVER_NAMES = {"over", "yes"}
UNDER_NAMES = {"under", "no"}


# --------------------------------------------------------------------------
# Normalised records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Quote:
    book: str
    market: str
    selection: str      # player name, or team for game markets
    side: str           # "Over" / "Under" / "Yes" / "No" / a team name
    line: float | None
    price: float        # decimal
    last_update: datetime | None

    @property
    def key(self) -> tuple:
        return (self.market, self.selection, self.line)


@dataclass
class Opportunity:
    sport: str
    event_id: str
    commence_time: datetime
    home_team: str
    away_team: str
    market: str
    selection: str
    line: float | None
    side: str

    sharp_book: str
    sharp_price_taken_side: float
    sharp_price_other_side: float
    sharp_overround: float

    fair_prob: float
    fair_price: float
    devig_method: str
    devig_spread: float
    fair_prob_by_method: dict[str, float]

    soft_book: str
    soft_price: float

    ev: float
    ev_range: tuple[float, float]      # min/max EV across de-vig methods
    kelly_fraction: float
    recommended_stake: float

    # How much this fair price is worth trusting, and what that does to the
    # bar and the ranking. See liquidity.py for the reasoning.
    market_tier: str
    liquidity: float
    required_ev: float          # the EV bar this market had to clear
    edge_score: float           # ev * liquidity -- what the report sorts on

    sharp_last_update: datetime | None
    soft_last_update: datetime | None
    scanned_at: datetime

    suspect: bool = False
    flags: list[str] = field(default_factory=list)
    #: Set by Database.record_scan once the row exists, so the number shown
    #: in the report is the number `betedge bet` expects. Without this the
    #: console numbered rows 1..n and the id was somewhere else entirely.
    db_id: int | None = None

    @property
    def minutes_to_start(self) -> float:
        return (self.commence_time - self.scanned_at).total_seconds() / 60.0

    @property
    def matchup(self) -> str:
        return f"{self.away_team} @ {self.home_team}"

    @property
    def description(self) -> str:
        side = (self.side or "").strip().lower()
        if side in OVER_NAMES or side in UNDER_NAMES:
            # A player prop names the player; a game total does not, and
            # its selection field just repeats "Over"/"Under".
            base = (
                self.selection
                if self.selection and self.selection.strip().lower() != side
                else "Total"
            )
            out = f"{base} {self.side}"
            return out if self.line is None else f"{out} {self.line:g}"
        if self.line is None:
            return f"{self.selection} ML"          # moneyline
        return f"{self.selection} {self.line:+g}"  # spread / handicap

    def to_row(self) -> dict:
        d = asdict(self)
        d["commence_time"] = self.commence_time.isoformat()
        d["scanned_at"] = self.scanned_at.isoformat()
        d["sharp_last_update"] = (
            self.sharp_last_update.isoformat() if self.sharp_last_update else None
        )
        d["soft_last_update"] = (
            self.soft_last_update.isoformat() if self.soft_last_update else None
        )
        d["flags"] = ",".join(self.flags)
        d["ev_min"], d["ev_max"] = self.ev_range
        d.pop("ev_range")
        d["fair_prob_by_method"] = ";".join(
            f"{k}={v:.4f}" for k, v in self.fair_prob_by_method.items()
        )
        d["american_price"] = pricing.decimal_to_american(self.soft_price)
        return d


@dataclass(frozen=True)
class QuoteAssessment:
    """
    One soft quote that got far enough to be priced, kept whatever the
    verdict.

    A scan that flags nothing reports only rejection counts, which say what
    tripped but not by how much. The difference between a board whose best
    quote sat at -0.3% and one whose best sat at -6% is the difference
    between a bar set slightly too high and a market there is simply no
    edge in, and the counters cannot tell them apart.
    """

    sport: str
    matchup: str
    market: str
    tier: str
    description: str
    book: str
    soft_price: float
    sharp_price: float
    overround: float
    liquidity: float
    ev: float
    required_ev: float
    minutes_to_start: float

    @property
    def shortfall(self) -> float:
        """How far under its bar this quote fell. Negative means it cleared."""
        return self.required_ev - self.ev


@dataclass
class ScanResult:
    started_at: datetime
    finished_at: datetime
    sports: list[str]
    opportunities: list[Opportunity]
    events_scanned: int = 0
    quotes_seen: int = 0
    sharp_markets_paired: int = 0
    credits_spent: int = 0
    credits_remaining: int | None = None
    rejections: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    #: Every priced quote, when the scan was asked to collect them.
    assessments: list = field(default_factory=list)

    @property
    def clean(self) -> list[Opportunity]:
        return [o for o in self.opportunities if not o.suspect]

    @property
    def suspect(self) -> list[Opportunity]:
        return [o for o in self.opportunities if o.suspect]


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def parse_event_odds(payload: dict) -> tuple[dict, list[Quote]]:
    """
    Flatten one /events/{id}/odds payload.

    Returns (event_meta, quotes). Malformed bookmaker or market blocks are
    skipped rather than raised on -- one bad book should not kill a slate.
    """
    meta = {
        "event_id": payload.get("id"),
        "sport": payload.get("sport_key"),
        "commence_time": _parse_ts(payload.get("commence_time")),
        "home_team": payload.get("home_team"),
        "away_team": payload.get("away_team"),
    }
    quotes: list[Quote] = []

    for book in payload.get("bookmakers") or []:
        book_key = book.get("key")
        book_update = _parse_ts(book.get("last_update"))
        for market in book.get("markets") or []:
            market_key = market.get("key")
            market_update = _parse_ts(market.get("last_update")) or book_update
            for outcome in market.get("outcomes") or []:
                price = outcome.get("price")
                if not isinstance(price, (int, float)) or price <= 1.0:
                    continue
                side = outcome.get("name")
                # Player props carry the player in `description`; game
                # markets put the team in `name` and have no description.
                selection = outcome.get("description") or side
                point = outcome.get("point")
                quotes.append(
                    Quote(
                        book=book_key,
                        market=market_key,
                        selection=selection,
                        side=side,
                        line=float(point) if isinstance(point, (int, float)) else None,
                        price=float(price),
                        last_update=market_update,
                    )
                )
    return meta, quotes


def pair_sharp_quotes(quotes: Iterable[Quote], sharp_book: str) -> dict[tuple, tuple[Quote, Quote]]:
    """
    Build (over, under) pairs from the sharp book, keyed by
    (market, selection, line). Only complete two-sided markets survive --
    you cannot strip vig from a single price.

    Over/under only. `group_sharp_markets` is the general version; this one
    stays because closing-line capture on a prop wants exactly this shape.
    """
    buckets: dict[tuple, dict[str, Quote]] = {}
    for q in quotes:
        if q.book != sharp_book:
            continue
        side = (q.side or "").strip().lower()
        if side in OVER_NAMES:
            buckets.setdefault(q.key, {})["over"] = q
        elif side in UNDER_NAMES:
            buckets.setdefault(q.key, {})["under"] = q
    return {
        key: (sides["over"], sides["under"])
        for key, sides in buckets.items()
        if "over" in sides and "under" in sides
    }


# --------------------------------------------------------------------------
# Generalised market grouping
#
# A market can be de-vigged once you hold every mutually exclusive outcome
# in it. Over/under markets have two outcomes named Over and Under. A
# moneyline's outcomes are named after the competitors -- two of them in
# tennis, MMA and US sports, three in soccer where a draw is possible. A
# spread has two outcomes carrying equal and opposite handicaps.
#
# So each quote gets two keys: which set of outcomes it belongs to, and
# which outcome inside that set it is. Everything downstream works off
# those, which is what lets one code path serve props, moneylines,
# spreads and totals.
# --------------------------------------------------------------------------


def group_key(q: Quote) -> tuple:
    """Which set of mutually exclusive outcomes this quote belongs to."""
    side = (q.side or "").strip().lower()
    if side in OVER_NAMES or side in UNDER_NAMES:
        # Player props carry the player; game totals do not.
        selection = q.selection if q.selection != q.side else None
        return (q.market, selection, q.line)
    # Competitor-named outcome: moneyline or spread. A spread's two sides
    # carry opposite handicaps (-1.5 / +1.5), so the magnitude identifies
    # the pair while the sign identifies the side.
    return (q.market, None, abs(q.line) if q.line is not None else None)


def outcome_key(q: Quote) -> tuple:
    """Which outcome inside its group this quote is."""
    side = (q.side or "").strip().lower()
    if side in OVER_NAMES:
        return ("over",)
    if side in UNDER_NAMES:
        return ("under",)
    # The competitor's name, plus the handicap so a soft book offering a
    # different spread is never matched against the sharp one.
    return (side, q.line)


def group_sharp_markets(
    quotes: Iterable[Quote], sharp_book: str
) -> dict[tuple, dict[tuple, Quote]]:
    """
    Collect the sharp book's quotes into complete outcome sets.

    Incomplete sets are not filtered here -- an incomplete set sums to less
    than 1 before de-vigging, which the overround guard in `evaluate_event`
    rejects. That keeps completeness checking in one place and means a
    three-way soccer market with a missing draw is caught by the same rule
    that catches a one-sided prop.
    """
    groups: dict[tuple, dict[tuple, Quote]] = {}
    for q in quotes:
        if q.book != sharp_book:
            continue
        groups.setdefault(group_key(q), {})[outcome_key(q)] = q
    return {k: v for k, v in groups.items() if len(v) >= 2}


def describe_quote(q: Quote) -> str:
    """
    Full human-readable bet, line included.

    describe_side names only the side, which for a prop reads "Randal
    Grichuk Over" -- true, useless, and impossible to place. The line is
    the part you need.
    """
    side = (q.side or "").strip().lower()
    if side in OVER_NAMES or side in UNDER_NAMES:
        base = (q.selection if q.selection
                and q.selection.strip().lower() != side else "Total")
        out = f"{base} {q.side}"
        return out if q.line is None else f"{out} {q.line:g}"
    if q.line is None:
        return f"{q.selection} ML"
    return f"{q.selection} {q.line:+g}"


def describe_side(q: Quote) -> str:
    """Human-readable side label for the report and the bet log."""
    side = (q.side or "").strip().lower()
    if side in OVER_NAMES or side in UNDER_NAMES:
        return q.side
    if q.line is None:
        return q.side                      # moneyline: just the competitor
    return f"{q.side} {q.line:+g}"         # spread: competitor and handicap


# --------------------------------------------------------------------------
# Edge evaluation
# --------------------------------------------------------------------------


def evaluate_event(
    meta: dict,
    quotes: Sequence[Quote],
    cfg: Config,
    now: datetime | None = None,
    rejections: dict[str, int] | None = None,
    collector: list | None = None,
) -> list[Opportunity]:
    """
    Score every soft quote in one event against the sharp book.

    Pass `collector` to keep a QuoteAssessment for every quote that got as
    far as being priced, rejected or not. That is what `betedge diagnose`
    reads.
    """
    now = now or datetime.now(timezone.utc)
    rejections = rejections if rejections is not None else {}
    m = cfg.model

    def reject(reason: str) -> None:
        rejections[reason] = rejections.get(reason, 0) + 1

    commence = meta.get("commence_time")
    if commence is None:
        reject("no_commence_time")
        return []

    minutes_out = (commence - now).total_seconds() / 60.0
    if minutes_out < m.min_minutes_to_start:
        reject("too_close_to_start")
        return []
    if minutes_out > m.max_hours_to_start * 60:
        reject("too_far_out")
        return []

    groups = group_sharp_markets(quotes, cfg.books.sharp)
    if not groups:
        reject("no_two_sided_sharp_market")
        return []

    soft_quotes = [q for q in quotes if q.book in cfg.books.soft]
    out: list[Opportunity] = []

    for q in soft_quotes:
        group = groups.get(group_key(q))
        if group is None:
            reject("no_matching_sharp_line")
            continue

        # Deterministic outcome order so the de-vig indices are stable.
        ordered = sorted(group.items(), key=lambda kv: str(kv[0]))
        okey = outcome_key(q)
        try:
            idx = [k for k, _ in ordered].index(okey)
        except ValueError:
            # The sharp book prices this market but not this outcome --
            # a different handicap, or a competitor named differently.
            reject("no_matching_sharp_outcome")
            continue

        prices = [quote.price for _, quote in ordered]
        sharp_same = prices[idx]
        others = [p for i, p in enumerate(prices) if i != idx]
        sharp_other = others[0] if len(others) == 1 else min(others)

        over_round = pricing.overround(prices)
        if not (m.min_overround <= over_round <= m.max_overround):
            reject("sharp_overround_out_of_bounds")
            continue

        spread = pricing.devig_spread(prices)
        if spread > m.max_devig_spread:
            reject("devig_methods_disagree")
            continue

        by_method = {
            name: probs[idx]
            for name, probs in pricing.fair_probs_all_methods(prices).items()
        }
        if m.devig_method == "worst_case":
            fair_prob = min(by_method.values())
        else:
            fair_prob = pricing.devig(prices, method=m.devig_method)[idx]

        ev = pricing.expected_value(fair_prob, q.price)
        evs = [pricing.expected_value(p, q.price) for p in by_method.values()]
        ev_range = (min(evs), max(evs))

        # How far to trust this fair price, and therefore how much edge to
        # insist on before flagging it. Ranking on raw EV sorts the board by
        # how likely a number is to be stale rather than by how much money
        # is on offer; liquidity.py explains the correction.
        liq = liquidity.assess(
            market=q.market,
            overround=over_round,
            n_outcomes=len(prices),
            fair_prob=fair_prob,
            minutes_to_start=minutes_out,
        )
        bar = liquidity.required_ev(m.min_ev, liq.score, m.liquidity_ev_penalty)

        if collector is not None:
            collector.append(
                QuoteAssessment(
                    sport=meta.get("sport") or "",
                    matchup=f"{meta.get('away_team')} @ {meta.get('home_team')}",
                    market=q.market,
                    tier=liq.tier,
                    description=describe_quote(q),
                    book=q.book,
                    soft_price=q.price,
                    sharp_price=sharp_same,
                    overround=over_round,
                    liquidity=liq.score,
                    ev=ev,
                    required_ev=bar,
                    minutes_to_start=minutes_out,
                )
            )

        if m.min_liquidity > 0 and liq.score < m.min_liquidity:
            reject("market_too_thin")
            continue

        if ev < bar:
            # Label by what actually did the rejecting, not by how thin the
            # market was. "below_liquidity_bar" means precisely: this would
            # have been flagged under a flat min_ev, and the thin-market
            # penalty is what stopped it. That is the number worth watching
            # when deciding whether the penalty is set sensibly -- tagging
            # every thin-market miss with it, however far below the bar,
            # made it useless for that.
            reject("below_liquidity_bar" if ev >= m.min_ev else "below_min_ev")
            continue

        flags: list[str] = []
        suspect = False

        if ev > m.max_plausible_ev:
            flags.append(f"ev_{ev:.1%}_implausible")
            suspect = True

        # Every outcome in a group comes from the same market block, so
        # they share a last_update timestamp.
        sharp_last_update = ordered[idx][1].last_update
        sharp_age = _age_minutes(sharp_last_update, now)
        soft_age = _age_minutes(q.last_update, now)
        if sharp_age is not None and sharp_age > m.max_sharp_staleness_minutes:
            flags.append(f"sharp_quote_{sharp_age:.0f}m_old")
            suspect = True
        if soft_age is not None and soft_age > m.max_soft_staleness_minutes:
            flags.append(f"soft_quote_{soft_age:.0f}m_old")
            suspect = True
        if ev_range[0] < 0 <= ev_range[1]:
            flags.append("edge_depends_on_devig_method")
            suspect = True
        if minutes_out < 30:
            flags.append("starts_soon")

        kelly = pricing.kelly_fraction(fair_prob, q.price)
        rec_stake = (
            0.0
            if suspect
            else pricing.stake(
                fair_prob,
                q.price,
                bankroll=cfg.bankroll.amount,
                kelly_multiplier=cfg.bankroll.kelly_multiplier,
                max_fraction=cfg.bankroll.max_fraction,
                min_stake=cfg.bankroll.min_stake,
                round_to=cfg.bankroll.round_to,
            )
        )

        out.append(
            Opportunity(
                sport=meta.get("sport") or "",
                event_id=meta.get("event_id") or "",
                commence_time=commence,
                home_team=meta.get("home_team") or "",
                away_team=meta.get("away_team") or "",
                market=q.market,
                selection=q.selection,
                line=q.line,
                side=q.side,
                sharp_book=cfg.books.sharp,
                sharp_price_taken_side=sharp_same,
                sharp_price_other_side=sharp_other,
                sharp_overround=over_round,
                fair_prob=fair_prob,
                fair_price=pricing.fair_decimal(fair_prob),
                devig_method=m.devig_method,
                devig_spread=spread,
                fair_prob_by_method=by_method,
                soft_book=q.book,
                soft_price=q.price,
                ev=ev,
                ev_range=ev_range,
                kelly_fraction=kelly,
                recommended_stake=rec_stake,
                market_tier=liq.tier,
                liquidity=liq.score,
                required_ev=bar,
                edge_score=liquidity.edge_score(ev, liq.score),
                sharp_last_update=sharp_last_update,
                soft_last_update=q.last_update,
                scanned_at=now,
                suspect=suspect,
                flags=flags,
            )
        )

    return out


def _age_minutes(ts: datetime | None, now: datetime) -> float | None:
    if ts is None:
        return None
    return (now - ts).total_seconds() / 60.0


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


@dataclass
class _ScanState:
    """Mutable tally shared by the two passes of a scan."""

    opportunities: list = field(default_factory=list)
    rejections: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    events_scanned: int = 0
    quotes_seen: int = 0
    paired: int = 0
    budget_exhausted: bool = False
    collector: list | None = None


def scan(
    cfg: Config,
    client: OddsApiClient,
    sports: Sequence[str] | None = None,
    now: datetime | None = None,
    max_events_per_sport: int | None = None,
    core_sports: Sequence[str] | None = None,
    collect: bool = False,
) -> ScanResult:
    """
    Run a scan: game-level markets first, then player props.

    The two passes have very different economics. Game-level markets
    (`core_sports`) come off the bulk endpoint and cost markets x 1 for the
    ENTIRE sport -- three credits buys every moneyline, spread and total on
    the board. Player props (`sports`) come off the per-event endpoint and
    cost markets x events, so a single NFL Sunday can run to hundreds.

    Cheap pass first, deliberately. Both passes stop when the credit budget
    is spent, and whichever runs second is the one that gets truncated. The
    cheap pass is also the higher-quality one -- mainlines are deep, sharply
    priced and carry the liquidity scores that survive the sliding EV bar --
    so spending the last of a budget there and truncating props is the right
    way round. Running props first, as this did originally, meant a tight
    budget could burn itself out on the thinnest markets on the board and
    never reach the best ones.
    """
    now = now or datetime.now(timezone.utc)
    started = now
    sports = list(sports if sports is not None else cfg.sports)
    core_sports = list(core_sports if core_sports is not None else cfg.core_sports)
    state = _ScanState()
    state.collector = [] if collect else None
    books = cfg.books.all

    scanned_core = _scan_core_markets(cfg, client, core_sports, books, now, state)
    if not state.budget_exhausted:
        _scan_props(cfg, client, sports, books, now, max_events_per_sport, state)

    # Ranked by liquidity-discounted edge, so the top of the list is where
    # the confidence and the money both are rather than where the number is
    # most likely to be wrong.
    state.opportunities.sort(key=lambda o: o.edge_score, reverse=True)

    return ScanResult(
        started_at=started,
        finished_at=datetime.now(timezone.utc),
        sports=sports + [s for s in scanned_core if s not in sports],
        opportunities=state.opportunities,
        events_scanned=state.events_scanned,
        quotes_seen=state.quotes_seen,
        sharp_markets_paired=state.paired,
        credits_spent=client.quota.spent_this_session,
        credits_remaining=client.quota.remaining,
        rejections=state.rejections,
        errors=state.errors,
        assessments=state.collector or [],
    )


def _scan_core_markets(
    cfg: Config,
    client: OddsApiClient,
    core_sports: Sequence[str],
    books: Sequence[str],
    now: datetime,
    state: _ScanState,
) -> list[str]:
    """
    Game-level markets off the bulk endpoint. Returns the sports actually
    swept, with any wildcards resolved.
    """
    core_sports = list(core_sports)
    if not core_sports:
        return []

    if any(p.endswith("*") for p in core_sports):
        try:
            live = [s["key"] for s in client.sports()]
        except Exception as exc:  # noqa: BLE001
            state.errors.append(f"could not list sports for wildcard expansion: {exc}")
            live = []
        core_sports = expand_sport_keys(core_sports, live)

    # After expansion, not before: a wildcard resolved against the live
    # sport list is exactly how a blocked sport would otherwise creep in.
    core_sports, blocked = cfg.allowed(core_sports)
    for sport in blocked:
        log.info("%s is excluded, skipping", sport)
        state.rejections["sport_excluded"] = (
            state.rejections.get("sport_excluded", 0) + 1
        )

    swept: list[str] = []
    for sport in core_sports:
        markets = cfg.core_markets_for_sport(sport)
        try:
            payloads = client.odds(sport, markets=markets, bookmakers=books)
        except CreditBudgetExceeded as exc:
            state.errors.append(str(exc))
            state.budget_exhausted = True
            break
        except Exception as exc:  # noqa: BLE001
            state.errors.append(f"{sport} (core): {exc}")
            continue

        swept.append(sport)
        log.info(
            "%s: %d events, %d credits for all core markets",
            sport, len(payloads or []), len(markets),
        )
        for payload in payloads or []:
            state.events_scanned += 1
            meta, quotes = parse_event_odds(payload)
            state.quotes_seen += len(quotes)
            state.paired += len(group_sharp_markets(quotes, cfg.books.sharp))
            state.opportunities.extend(
                evaluate_event(meta, quotes, cfg, now=now,
                               rejections=state.rejections,
                               collector=state.collector)
            )
    return swept


def _scan_props(
    cfg: Config,
    client: OddsApiClient,
    sports: Sequence[str],
    books: Sequence[str],
    now: datetime,
    max_events_per_sport: int | None,
    state: _ScanState,
) -> None:
    """Player props off the per-event endpoint. Costs markets x events."""
    sports, blocked = cfg.allowed(list(sports))
    for sport in blocked:
        log.info("%s is excluded, skipping", sport)
        state.rejections["sport_excluded"] = (
            state.rejections.get("sport_excluded", 0) + 1
        )

    for sport in sports:
        if state.budget_exhausted:
            break
        try:
            markets = cfg.markets_for_sport(
                sport, include_alternate=cfg.model.include_alternate_lines
            )
        except ValueError as exc:
            state.errors.append(str(exc))
            continue
        if not markets:
            log.info("%s has no configured prop markets, skipping", sport)
            continue

        try:
            events = client.events(sport)
        except Exception as exc:  # noqa: BLE001 - one bad sport must not kill the scan
            state.errors.append(f"{sport}: could not list events: {exc}")
            continue

        events = _events_in_window(
            events, now, cfg, max_hours=cfg.prop_window_hours(sport)
        )
        if max_events_per_sport:
            events = events[:max_events_per_sport]

        log.info(
            "%s: %d events in window, ~%d credits at most",
            sport, len(events), estimate_credits(len(events), len(markets), len(books)),
        )

        for event in events:
            try:
                payload = client.event_odds(sport, event["id"], markets, books)
            except CreditBudgetExceeded as exc:
                state.errors.append(str(exc))
                log.warning("stopping early: %s", exc)
                state.budget_exhausted = True
                break
            except Exception as exc:  # noqa: BLE001
                state.errors.append(f"{sport}/{event.get('id')}: {exc}")
                continue

            if not payload:
                continue

            state.events_scanned += 1
            meta, quotes = parse_event_odds(payload)
            state.quotes_seen += len(quotes)
            state.paired += len(pair_sharp_quotes(quotes, cfg.books.sharp))
            state.opportunities.extend(
                evaluate_event(meta, quotes, cfg, now=now,
                               rejections=state.rejections,
                               collector=state.collector)
            )


def _events_in_window(
    events: list[dict],
    now: datetime,
    cfg: Config,
    max_hours: float | None = None,
) -> list[dict]:
    """
    Drop events outside the time window before spending credits on them.

    `max_hours` narrows the look-ahead for one sport. This filter runs on
    the free event list, so every event it removes is a per-event odds call
    never made -- which is the whole saving.
    """
    horizon = (max_hours if max_hours is not None else cfg.model.max_hours_to_start) * 60
    keep = []
    for e in events:
        ts = _parse_ts(e.get("commence_time"))
        if ts is None:
            continue
        minutes = (ts - now).total_seconds() / 60.0
        if minutes < cfg.model.min_minutes_to_start:
            continue
        if minutes > horizon:
            continue
        keep.append(e)
    return keep


def best_per_selection(opportunities: Sequence[Opportunity]) -> list[Opportunity]:
    """
    Collapse to one row per (event, market, player, side), keeping the best
    EV. Without this, alternate lines produce a dozen near-duplicate rows
    for the same player and the list becomes unreadable.
    """
    best: dict[tuple, Opportunity] = {}
    for o in opportunities:
        key = (o.event_id, o.market, o.selection, (o.side or "").lower())
        if key not in best or o.edge_score > best[key].edge_score:
            best[key] = o
    return sorted(best.values(), key=lambda o: o.edge_score, reverse=True)


def total_exposure(opportunities: Sequence[Opportunity]) -> float:
    return sum(o.recommended_stake for o in opportunities)


def cap_exposure(opportunities: Sequence[Opportunity], cfg: Config) -> list[Opportunity]:
    """
    Scale stakes down proportionally if the whole board would put more than
    max_total_exposure_fraction of bankroll at risk at once. Correlated
    props on the same slate are not independent bets, and Kelly sizing each
    one in isolation quietly overbets the portfolio.
    """
    cap = cfg.bankroll.amount * cfg.bankroll.max_total_exposure_fraction
    total = total_exposure(opportunities)
    if total <= cap or total == 0:
        return list(opportunities)
    scale = cap / total
    for o in opportunities:
        o.recommended_stake = round(o.recommended_stake * scale / cfg.bankroll.round_to) * cfg.bankroll.round_to
        if o.recommended_stake < cfg.bankroll.min_stake:
            o.recommended_stake = 0.0
        o.flags.append("stake_scaled_for_total_exposure")
    return list(opportunities)
