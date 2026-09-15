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

from . import pricing
from .config import Config
from .markets import core_markets_for, estimate_credits, expand_sport_keys, markets_for
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

    sharp_last_update: datetime | None
    soft_last_update: datetime | None
    scanned_at: datetime

    suspect: bool = False
    flags: list[str] = field(default_factory=list)

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
) -> list[Opportunity]:
    """Score every soft quote in one event against the sharp book."""
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

        if ev < m.min_ev:
            reject("below_min_ev")
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


def scan(
    cfg: Config,
    client: OddsApiClient,
    sports: Sequence[str] | None = None,
    now: datetime | None = None,
    max_events_per_sport: int | None = None,
    core_sports: Sequence[str] | None = None,
) -> ScanResult:
    """
    Run a scan.

    Two passes with very different economics. Player props (`sports`) cost
    one call per event per market off the per-event endpoint. Game-level
    markets (`core_sports`) cost markets x 1 for the entire sport off the
    bulk endpoint -- roughly fifty times cheaper per opportunity, at the
    price of thinner edges, because mainlines are where the sharp money
    concentrates.
    """
    now = now or datetime.now(timezone.utc)
    started = now
    sports = list(sports or cfg.sports)
    rejections: dict[str, int] = {}
    opportunities: list[Opportunity] = []
    errors: list[str] = []
    events_scanned = quotes_seen = paired = 0

    books = cfg.books.all
    budget_exhausted = False

    for sport in sports:
        if budget_exhausted:
            break
        try:
            markets = cfg.markets_for_sport(
                sport, include_alternate=cfg.model.include_alternate_lines
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not markets:
            log.info("%s has no configured prop markets, skipping", sport)
            continue

        try:
            events = client.events(sport)
        except Exception as exc:  # noqa: BLE001 - one bad sport must not kill the scan
            errors.append(f"{sport}: could not list events: {exc}")
            continue

        events = _events_in_window(events, now, cfg)
        if max_events_per_sport:
            events = events[:max_events_per_sport]

        log.info(
            "%s: %d events in window, ~%d credits at most",
            sport,
            len(events),
            estimate_credits(len(events), len(markets), len(books)),
        )

        for event in events:
            try:
                payload = client.event_odds(sport, event["id"], markets, books)
            except CreditBudgetExceeded as exc:
                errors.append(str(exc))
                log.warning("stopping early: %s", exc)
                budget_exhausted = True
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{sport}/{event.get('id')}: {exc}")
                continue

            if not payload:
                continue

            events_scanned += 1
            meta, quotes = parse_event_odds(payload)
            quotes_seen += len(quotes)
            paired += len(pair_sharp_quotes(quotes, cfg.books.sharp))
            opportunities.extend(
                evaluate_event(meta, quotes, cfg, now=now, rejections=rejections)
            )

    # ---- game-level markets -------------------------------------------
    # These come off the bulk endpoint: one call covers every event in the
    # sport, and costs markets x 1 rather than markets x events. Three
    # credits for a whole tennis tour or a whole day of baseball.
    core_sports = list(core_sports if core_sports is not None else cfg.core_sports)
    if core_sports and not budget_exhausted:
        if any(p.endswith("*") for p in core_sports):
            try:
                live = [s["key"] for s in client.sports()]
            except Exception as exc:  # noqa: BLE001
                errors.append(f"could not list sports for wildcard expansion: {exc}")
                live = []
            core_sports = expand_sport_keys(core_sports, live)

        for sport in core_sports:
            markets = core_markets_for(sport)
            try:
                payloads = client.odds(sport, markets=markets, bookmakers=books)
            except CreditBudgetExceeded as exc:
                errors.append(str(exc))
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{sport} (core): {exc}")
                continue

            log.info("%s: %d events, %d credits for all core markets",
                     sport, len(payloads or []), len(markets))

            for payload in payloads or []:
                events_scanned += 1
                meta, quotes = parse_event_odds(payload)
                quotes_seen += len(quotes)
                paired += len(group_sharp_markets(quotes, cfg.books.sharp))
                opportunities.extend(
                    evaluate_event(meta, quotes, cfg, now=now, rejections=rejections)
                )
        sports = sports + [s for s in core_sports if s not in sports]

    opportunities.sort(key=lambda o: o.ev, reverse=True)

    return ScanResult(
        started_at=started,
        finished_at=datetime.now(timezone.utc),
        sports=sports,
        opportunities=opportunities,
        events_scanned=events_scanned,
        quotes_seen=quotes_seen,
        sharp_markets_paired=paired,
        credits_spent=client.quota.spent_this_session,
        credits_remaining=client.quota.remaining,
        rejections=rejections,
        errors=errors,
    )


def _events_in_window(events: list[dict], now: datetime, cfg: Config) -> list[dict]:
    """Drop events outside the configured time window before spending credits."""
    keep = []
    for e in events:
        ts = _parse_ts(e.get("commence_time"))
        if ts is None:
            continue
        minutes = (ts - now).total_seconds() / 60.0
        if minutes < cfg.model.min_minutes_to_start:
            continue
        if minutes > cfg.model.max_hours_to_start * 60:
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
        if key not in best or o.ev > best[key].ev:
            best[key] = o
    return sorted(best.values(), key=lambda o: o.ev, reverse=True)


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
