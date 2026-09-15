"""
Multi-leg tickets: pick'em entries and parlays, priced against Pinnacle.

A different objective from the single-bet scanner
-------------------------------------------------
`scan.py` looks for a soft book whose PRICE is better than Pinnacle's
de-vigged fair price on the same selection. This module looks for
something else entirely: a set of legs whose JOINT probability is higher
than the payout structure assumes.

Where that edge actually lives determines the whole design, so it is worth
stating plainly.

**Fixed-multiplier pick'em (Underdog, PrizePicks) is the primary target.**
Their payouts are set by leg count and nothing else -- a 2-pick pays 3x, a
5-pick pays 20x -- and do not adjust for what the legs are. Two positively
correlated legs therefore hit together more often than a 3x multiple
implies. The gap between the joint probability and the independence the
multiple assumes is the edge, and it is structural rather than a mistake
anyone has to make for it to be there.

**DraftKings same-game parlays are the secondary, harder target.** DK runs
its own correlation model and already discounts correlated SGP legs. An
edge there means their estimate of the correlation is wrong, not that
correlation exists. Expect far fewer hits, and treat the ones you get as
lower confidence -- which is exactly how they are flagged.

**Cross-game parlays are a trap and are not optimised for.** The vig
compounds: four legs at 4.5% hold each is 1.045^4 - 1, about 19% total
hold, and legs in different games have no correlation to claw any of it
back. `copula.compounded_hold` computes this so the tool can say so with
a number, and the search refuses to build these by default.

Where the marginal probabilities come from
------------------------------------------
Pinnacle, always. This is the load-bearing idea of the module.

The pick'em sites post a line and a fixed multiplier. They do not post two
sides, so there is no vig to strip and nothing to de-vig. The edge comes
from having a better estimate of each leg's true probability than the
pick'em site has, and that estimate comes from de-vigging Pinnacle's
two-sided market on the same player, same stat, SAME LINE.

Same line is not a detail. A leg compared against Pinnacle's number at a
different line is not a measurement of anything. So the default is to skip
such a leg; optional interpolation across Pinnacle's alternate lines is
available, is off by default, and marks every leg it touches as estimated.

A leg with no Pinnacle reference is not usable, and the ticket containing
it is not scored. That is a refusal, not a fallback.

Optimise for expected value first
---------------------------------
Payout multiple is a tiebreak and nothing more. A 20x ticket at -8% EV is
a worse bet than a 3x ticket at +4%, and every ranking, print and report
in this module is arranged to make that obvious rather than to bury it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

from . import copula, correlation, liquidity, pricing, rosters as rosters_mod
from .config import Config
from .correlation import CorrelationMatrix, EstimateStore, PriorSet
from .markets import estimate_credits
from .rosters import SOURCE_STRUCTURAL, RosterBook, infer_sides
from .oddsapi import CreditBudgetExceeded, OddsApiClient
from .scan import (
    OVER_NAMES,
    UNDER_NAMES,
    Quote,
    _events_in_window,
    parse_event_odds,
)

log = logging.getLogger(__name__)

PAYOUTS_PATH = Path(__file__).parent / "data" / "payouts.yaml"

KIND_PICKEM = "pickem"
KIND_PARLAY = "parlay"


# --------------------------------------------------------------------------
# Sport coverage
#
# These are a starting point, not a conclusion. `betedge parlay coverage`
# probes the API and reports, per sport, how many two-sided Pinnacle prop
# markets exist and how many have a matching quote at a target book, so
# the list can be re-derived when seasons turn over.
# --------------------------------------------------------------------------

#: Sports where DraftKings and Pinnacle both carry deep player prop
#: markets and daily volume is high enough to produce candidates.
DEFAULT_PROP_SPORTS = (
    "basketball_nba",
    "americanfootball_nfl",
    "baseball_mlb",
    "icehockey_nhl",
)

#: Kept out of the prop-based optimizer on purpose. This is not a
#: judgement about the sports -- Pinnacle is the sharpest book in the world
#: on tennis and soccer, and `core_sports` in the single-bet scanner exists
#: to exploit exactly that. It is a statement about PLAYER PROP coverage,
#: which is the only thing this module can build a marginal from. No
#: two-sided Pinnacle prop means no fair probability means no leg.
PROP_OPTIMIZER_EXCLUDED = {
    "tennis_*": (
        "Pinnacle prices essentially no tennis player props. The sport's "
        "edge is in match markets, which have no second leg to correlate "
        "with inside the same event."
    ),
    "mma_mixed_martial_arts": (
        "Fight cards carry moneylines and little else at Pinnacle. Method "
        "and round props exist at the soft books with no sharp counterpart, "
        "so there is nothing to de-vig against."
    ),
    "soccer_*": (
        "Pinnacle's soccer player-prop coverage is thin and intermittent -- "
        "shots and shots on target at best, rarely two-sided, rarely on the "
        "same line the pick'em sites post."
    ),
}


# --------------------------------------------------------------------------
# Payout structures
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Product:
    """
    One payout structure, loaded from YAML because these are configuration
    rather than facts: they vary by state, change without notice, and an
    EV computed from the wrong table is not slightly wrong.
    """

    key: str
    book: str
    kind: str
    title: str
    payouts: dict[int, tuple[float, ...]]
    verified: bool = False
    allows_same_player: bool = False
    void_behaviour: str = "reduce"
    #: Payouts to fall back on when a push shrinks the entry below this
    #: product's smallest table -- a 3-leg flex entry with one push becomes
    #: an ordinary 2-pick, not a refund.
    fallback_payouts: dict[int, tuple[float, ...]] = field(default_factory=dict)
    reduces_to: str | None = None
    note: str = ""

    @property
    def leg_counts(self) -> list[int]:
        return sorted(self.payouts)

    @property
    def min_legs(self) -> int:
        return min(self.payouts) if self.payouts else 2

    @property
    def max_legs(self) -> int:
        return max(self.payouts) if self.payouts else 2

    @property
    def is_pickem(self) -> bool:
        return self.kind == KIND_PICKEM

    def vector_for(self, legs: int) -> tuple[float, ...] | None:
        """The payout vector for an entry of `legs` legs, if one exists."""
        return self.payouts.get(legs) or self.fallback_payouts.get(legs)

    def all_hit_multiple(self, legs: int) -> float:
        vector = self.vector_for(legs)
        return float(vector[legs]) if vector else 0.0

    def independent_ev(self, legs: int, prob: float) -> float:
        """
        Expected value of an entry of `legs` identical independent legs,
        each hitting with probability `prob`.

        Binomial rather than simulated: with independent, identical legs
        the number of hits IS binomial, so this is exact and needs no
        Monte Carlo. It exists to locate the break-even, which is a
        property of the payout structure and should not inherit sampling
        noise from anything.
        """
        vector = self.vector_for(legs)
        if not vector:
            return -1.0
        total = 0.0
        for k in range(legs + 1):
            weight = math.comb(legs, k) * prob ** k * (1.0 - prob) ** (legs - k)
            total += weight * vector[k]
        return total - 1.0

    def breakeven_leg_prob(self, legs: int) -> float:
        """
        The per-leg probability an entry of this size needs to break even
        if the legs were independent.

        For an all-or-nothing structure this is just multiple^(-1/n): a
        5-pick at 20x needs 54.9% a leg, a 2-pick at 3x needs 57.7%. For a
        FLEX structure it is lower than that, because missing one leg
        still pays something -- and using the all-hit figure there would
        overstate the bar badly and reject legs that were fine. So it is
        solved numerically from the full payout vector, which reduces to
        the closed form on the structures where the closed form is right.

        This is the reference every leg's "edge" is measured against, and
        the number correlation has to make up when a leg falls short.
        """
        vector = self.vector_for(legs)
        if not vector:
            return 1.0
        if self.independent_ev(legs, 1.0 - 1e-12) <= 0:
            return 1.0      # cannot break even at any probability
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self.independent_ev(legs, mid) < 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def easiest_breakeven(self) -> float:
        """
        The most permissive per-leg break-even across this product's sizes.

        Used for PREFILTERING only, so that no leg is dropped which could
        still earn its place in a larger structure. The EV bar does the
        real work afterwards.
        """
        if not self.payouts:
            return 0.0
        return min(self.breakeven_leg_prob(n) for n in self.payouts)

    def multiple_grid(self, legs: int) -> np.ndarray:
        """
        What a 1-unit entry returns for every (voids, hits) outcome.

        Indexed `[v][k]`: v legs voided, k of the survivors hit. This is
        the whole payout structure expressed as data the Monte Carlo can
        index into, which is what lets flex entries, insured entries and
        push-shrunk entries all go through one code path.
        """
        size = legs + 1
        grid = np.zeros((size, size), dtype=float)
        for v in range(size):
            remaining = legs - v
            if v > 0 and self.void_behaviour == "refund":
                grid[v, : remaining + 1] = 1.0
                continue
            vector = self.vector_for(remaining) if remaining > 0 else None
            if vector is None:
                # No table for an entry this small: the stake comes back.
                grid[v, : remaining + 1] = 1.0
                continue
            for k in range(remaining + 1):
                grid[v, k] = float(vector[k])
        return grid


@dataclass
class PayoutTable:
    products: dict[str, Product]
    path: Path | None = None
    last_verified_by_user: Any = None

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PayoutTable":
        p = Path(path or PAYOUTS_PATH)
        raw = yaml.safe_load(p.read_text()) or {}
        declared: dict[str, dict] = raw.get("products") or {}
        products: dict[str, Product] = {}

        for key, spec in declared.items():
            payouts = {}
            for legs, vector in (spec.get("payouts") or {}).items():
                legs = int(legs)
                vector = tuple(float(x) for x in vector)
                if len(vector) != legs + 1:
                    raise ValueError(
                        f"{p}: {key} has a {legs}-leg payout vector of length "
                        f"{len(vector)}; it needs {legs + 1} entries, one for "
                        f"each of k = 0..{legs} legs hitting"
                    )
                if any(x < 0 for x in vector):
                    raise ValueError(f"{p}: {key} has a negative payout in {vector}")
                payouts[legs] = vector
            kind = spec.get("kind", KIND_PICKEM)
            if kind not in (KIND_PICKEM, KIND_PARLAY):
                raise ValueError(
                    f"{p}: {key} has kind {kind!r}; expected "
                    f"{KIND_PICKEM!r} or {KIND_PARLAY!r}"
                )
            void_behaviour = spec.get("void_behaviour", "reduce")
            if void_behaviour not in ("reduce", "refund"):
                raise ValueError(
                    f"{p}: {key} has void_behaviour {void_behaviour!r}; "
                    "expected 'reduce' or 'refund'"
                )
            products[key] = Product(
                key=key,
                book=spec.get("book", ""),
                kind=kind,
                title=spec.get("title", key),
                payouts=payouts,
                verified=bool(spec.get("verified", False)),
                allows_same_player=bool(spec.get("allows_same_player", False)),
                void_behaviour=void_behaviour,
                reduces_to=spec.get("reduces_to"),
                note=(spec.get("note") or "").strip(),
            )

        # Resolve `reduces_to` once everything is parsed, so an insured
        # entry knows what a push turns it into.
        for key, product in list(products.items()):
            if product.reduces_to:
                target = products.get(product.reduces_to)
                if target is None:
                    raise ValueError(
                        f"{p}: {key} reduces_to {product.reduces_to!r}, "
                        "which is not a product in this file"
                    )
                product.fallback_payouts.update(target.payouts)

        meta = raw.get("meta") or {}
        return cls(
            products=products,
            path=p,
            last_verified_by_user=meta.get("last_verified_by_user"),
        )

    def get(self, key: str) -> Product:
        try:
            return self.products[key]
        except KeyError as exc:
            raise ValueError(
                f"unknown payout product {key!r}. Loaded: "
                f"{sorted(self.products)}. Run `betedge parlay verify-payouts` "
                "to see what is in the table."
            ) from exc

    def for_book(self, book: str) -> list[Product]:
        return [p for p in self.products.values() if p.book == book]

    @property
    def unverified(self) -> list[str]:
        return sorted(k for k, p in self.products.items() if not p.verified)


def parlay_product(
    legs: Sequence["Leg"],
    offered_decimal: float | None = None,
    book: str = "draftkings",
) -> Product:
    """
    Build a payout structure for a priced parlay.

    A parlay is not a fixed-multiplier product: the book quotes it. With
    `offered_decimal` we use exactly what the app shows. Without one we
    fall back to the product of the legs' own prices, which for a SAME-GAME
    parlay is an upper bound the book will never actually pay -- DraftKings
    discounts correlated legs, which is the entire difficulty of beating
    them there. Tickets priced this way are flagged for it.
    """
    n = len(legs)
    if offered_decimal is None:
        offered_decimal = 1.0
        for leg in legs:
            if not leg.book_price:
                raise ValueError(
                    f"leg {leg.description} has no price at {leg.book}, so a "
                    "parlay price cannot be computed; pass --offered-price"
                )
            offered_decimal *= leg.book_price
    vector = tuple([0.0] * n + [float(offered_decimal)])
    return Product(
        key=f"{book}_parlay_{n}",
        book=book,
        kind=KIND_PARLAY,
        title=f"{book} parlay @ {offered_decimal:.2f}",
        payouts={n: vector},
        verified=True,
        allows_same_player=True,
        void_behaviour="reduce",
        note="priced from the book, not from the payout table",
    )


# --------------------------------------------------------------------------
# Legs
# --------------------------------------------------------------------------

LINE_EXACT = "exact"
LINE_INTERPOLATED = "interpolated"


def resolve_product(
    product: Product,
    legs: Sequence["Leg"],
    offered_decimal: float | None = None,
) -> Product:
    """
    Turn a product into the concrete payout structure for one leg set.

    Fixed-multiplier pick'em products already are one: the ladder is the
    whole thing. A parlay is not -- `draftkings_parlay` ships with an empty
    table because the price comes from the book, leg by leg -- so its
    payout vector has to be built for the specific legs in hand.
    """
    template = product.kind == KIND_PARLAY and not product.payouts
    if template or (product.kind == KIND_PARLAY and offered_decimal is not None):
        return parlay_product(
            legs, offered_decimal, book=product.book or "draftkings"
        )
    return product


@dataclass(frozen=True)
class SharpLine:
    """Pinnacle's de-vigged view of one (market, player, line)."""

    line: float
    prob_over: float
    prob_under: float
    price_over: float
    price_under: float
    overround: float
    devig_spread: float
    by_method_over: dict[str, float]
    last_update: datetime | None

    def prob_for(self, side: str) -> float:
        return self.prob_over if _is_over(side) else self.prob_under

    def price_for(self, side: str) -> float:
        return self.price_over if _is_over(side) else self.price_under

    def other_price(self, side: str) -> float:
        return self.price_under if _is_over(side) else self.price_over

    def by_method_for(self, side: str) -> dict[str, float]:
        if _is_over(side):
            return dict(self.by_method_over)
        return {k: 1.0 - v for k, v in self.by_method_over.items()}


@dataclass(frozen=True)
class Leg:
    """
    One selection inside a ticket, with its marginal probability and the
    provenance of that probability.

    `fair_prob` is Pinnacle's de-vigged probability for this side at this
    line. On an integer line it is conditional on the stat not landing
    exactly on the line, because a two-sided over/under quote has no room
    for the push -- so the probability the leg actually WINS is
    `hit_prob = fair_prob * (1 - push_prob)`, and that is what the copula
    is given.
    """

    sport: str
    event_id: str
    commence_time: datetime
    home_team: str
    away_team: str

    market: str
    selection: str
    side: str
    line: float | None

    book: str                       #: where the ticket leg would be placed
    book_price: float | None        #: the book's own price, if it posts one
    book_last_update: datetime | None

    fair_prob: float
    push_prob: float
    sharp_price_taken: float
    sharp_price_other: float
    sharp_overround: float
    devig_spread: float
    fair_prob_by_method: dict[str, float]
    sharp_last_update: datetime | None
    sharp_line: float

    market_tier: str
    liquidity: float
    line_source: str = LINE_EXACT
    push_source: str = "half_line"
    #: The player's club, and where that came from. Carried together on
    #: purpose: a team with no provenance cannot be aged, and an entry that
    #: cannot be aged is one the correlation matrix should not trust.
    team: str | None = None
    team_source: str | None = None
    team_as_of: str | None = None
    flags: tuple[str, ...] = ()

    @property
    def hit_prob(self) -> float:
        """Unconditional probability this leg wins outright."""
        return self.fair_prob * (1.0 - self.push_prob)

    @property
    def fair_price(self) -> float:
        return pricing.fair_decimal(self.hit_prob)

    @property
    def single_leg_ev(self) -> float | None:
        """EV of this leg alone at the book's own price, where there is one."""
        if not self.book_price:
            return None
        return pricing.expected_value(self.hit_prob, self.book_price)

    @property
    def matchup(self) -> str:
        return f"{self.away_team} @ {self.home_team}"

    @property
    def description(self) -> str:
        base = (
            self.selection
            if self.selection and self.selection.strip().lower() != (self.side or "").lower()
            else "Total"
        )
        out = f"{base} {self.side}"
        return out if self.line is None else f"{out} {self.line:g}"

    @property
    def identity(self) -> tuple:
        """What makes two legs the same bet, for deduplication."""
        return (self.event_id, self.market, (self.selection or "").lower(),
                (self.side or "").lower(), self.line)

    @property
    def prop_identity(self) -> tuple:
        """The underlying market, ignoring side and line: one leg per prop."""
        return (self.event_id, self.market, (self.selection or "").lower())

    def edge_vs(self, breakeven: float) -> float:
        """
        How far this leg's probability sits above the level an entry needs
        from it. Negative means correlation has to make up the difference.
        """
        return self.hit_prob - breakeven


def _is_over(side: str | None) -> bool:
    return (side or "").strip().lower() in OVER_NAMES


def _is_two_sided(side: str | None) -> bool:
    s = (side or "").strip().lower()
    return s in OVER_NAMES or s in UNDER_NAMES


def _is_half_line(line: float | None) -> bool:
    """A half-point line cannot push, which is why it is the good kind."""
    return line is not None and abs(line - math.floor(line) - 0.5) < 1e-9


def _age_minutes(ts: datetime | None, now: datetime) -> float | None:
    if ts is None:
        return None
    return (now - ts).total_seconds() / 60.0


# --------------------------------------------------------------------------
# Pinnacle's ladder
# --------------------------------------------------------------------------


def sharp_ladder(
    quotes: Iterable[Quote], sharp_book: str, devig_method: str = "worst_case"
) -> dict[tuple[str, str], dict[float, SharpLine]]:
    """
    Every two-sided Pinnacle over/under in one event, indexed by
    (market, player) and then by line.

    Holding the whole ladder rather than one line at a time is what makes
    both interpolation and push estimation possible: the chance a stat
    lands exactly on an integer line is the difference between the over
    probabilities at the two half-lines either side of it, which you can
    only see if you kept them.
    """
    buckets: dict[tuple, dict[float, dict[str, Quote]]] = {}
    for q in quotes:
        if q.book != sharp_book or q.line is None or not _is_two_sided(q.side):
            continue
        key = (q.market, q.selection)
        slot = buckets.setdefault(key, {}).setdefault(float(q.line), {})
        slot["over" if _is_over(q.side) else "under"] = q

    ladder: dict[tuple[str, str], dict[float, SharpLine]] = {}
    for key, by_line in buckets.items():
        for line, sides in by_line.items():
            if "over" not in sides or "under" not in sides:
                continue        # one price is not a market; nothing to de-vig
            over, under = sides["over"], sides["under"]
            prices = [over.price, under.price]
            by_method = {
                name: probs[0]
                for name, probs in pricing.fair_probs_all_methods(prices).items()
            }
            if devig_method == "worst_case":
                prob_over = min(by_method.values())
                prob_under = min(1.0 - v for v in by_method.values())
            else:
                probs = pricing.devig(prices, method=devig_method)
                prob_over, prob_under = probs[0], probs[1]
            ladder.setdefault(key, {})[line] = SharpLine(
                line=line,
                prob_over=prob_over,
                prob_under=prob_under,
                price_over=over.price,
                price_under=under.price,
                overround=pricing.overround(prices),
                devig_spread=pricing.devig_spread(prices),
                by_method_over=by_method,
                last_update=over.last_update or under.last_update,
            )
    return ladder


def interpolate_over_prob(
    rungs: dict[float, SharpLine], line: float, max_distance: float
) -> tuple[float, float, float] | None:
    """
    Estimate P(over `line`) from Pinnacle's neighbouring lines.

    Interpolation runs in probit space -- linear between the two bracketing
    lines' inverse-normal probabilities rather than between the
    probabilities themselves -- because the probability of clearing a
    counting-stat line is close to normal in the line, and linear
    interpolation of the raw probability visibly bends the wrong way near
    the tails.

    Extrapolation is refused outright: without a rung on each side there is
    no shape to interpolate, only a guess. Returns
    (probability, lower rung, upper rung), or None.
    """
    below = [x for x in rungs if x < line]
    above = [x for x in rungs if x > line]
    if not below or not above:
        return None
    lo, hi = max(below), min(above)
    if (line - lo) > max_distance or (hi - line) > max_distance:
        return None
    z_lo = copula.norm_ppf(min(max(rungs[lo].prob_over, 1e-9), 1 - 1e-9))
    z_hi = copula.norm_ppf(min(max(rungs[hi].prob_over, 1e-9), 1 - 1e-9))
    weight = (line - lo) / (hi - lo)
    return copula.norm_cdf(z_lo + weight * (z_hi - z_lo)), lo, hi


def estimate_push_prob(
    rungs: dict[float, SharpLine], line: float, assumed: float
) -> tuple[float, str]:
    """
    The chance the stat lands exactly on `line` and the leg voids.

    A half-point line cannot push -- that is the whole point of the half
    point -- so it returns zero and says so. An integer line can, and
    integer lines at half-point-shy numbers are common on low-count stats
    (receptions, threes, strikeouts) where the push mass is not small.

    Measured where Pinnacle prices both surrounding half-lines:

        P(X = L) = P(X > L - 0.5) - P(X > L + 0.5)

    which is exact, not an approximation. Where it does not, the
    configured assumption is used and the leg is flagged, because a
    guessed push probability shrinks the entry in the model in a way the
    user should be able to see.
    """
    if _is_half_line(line):
        return 0.0, "half_line"
    lower, upper = line - 0.5, line + 0.5
    if lower in rungs and upper in rungs:
        measured = rungs[lower].prob_over - rungs[upper].prob_over
        if measured >= 0.0:
            return min(measured, 0.5), "measured"
    return max(0.0, assumed), "assumed"


# --------------------------------------------------------------------------
# Building legs from one event's payload
#
# The guards here are the same ones `scan.evaluate_event` applies to a
# single bet, and for the same reasons: a stale quote, a malformed
# overround or a fair price that depends on which de-vig method you picked
# is not a foundation to build anything on. A ticket multiplies four of
# them together, so if anything the case is stronger.
# --------------------------------------------------------------------------


def build_legs(
    meta: dict,
    quotes: Sequence[Quote],
    cfg: Config,
    books: Sequence[str],
    now: datetime | None = None,
    rejections: dict[str, int] | None = None,
    rosters: RosterBook | None = None,
) -> list[Leg]:
    """
    Every usable leg in one event, for the books named in `books`.

    A leg is usable when the book posts a line, Pinnacle prices the same
    market and player two-sided at the SAME line, and every sanity guard
    passes. Anything else is counted in `rejections` and dropped -- there
    is no partial credit, because a leg without a Pinnacle reference has
    no probability and a ticket containing it cannot be scored at all.
    """
    now = now or datetime.now(timezone.utc)
    rejections = rejections if rejections is not None else {}
    m = cfg.model
    pc = cfg.parlay

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

    ladder = sharp_ladder(quotes, cfg.books.sharp, m.devig_method)
    if not ladder:
        reject("no_two_sided_sharp_market")
        return []

    out: list[Leg] = []
    seen: set[tuple] = set()

    for q in quotes:
        if q.book not in books or q.book == cfg.books.sharp:
            continue
        if not _is_two_sided(q.side) or q.line is None:
            # Multi-leg tickets are built from over/under selections.
            # Moneylines have no line to match and no push to model.
            reject("not_an_over_under_selection")
            continue

        rungs = ladder.get((q.market, q.selection))
        if not rungs:
            reject("no_pinnacle_reference")
            continue

        line = float(q.line)
        sharp = rungs.get(line)
        line_source = LINE_EXACT
        leg_flags: list[str] = []
        sharp_line_used = line

        if sharp is None:
            if not pc.allow_line_interpolation:
                # Never silently compare different lines. A prop at 6.5 and
                # a prop at 7.5 are different bets, and pretending otherwise
                # manufactures edge out of nothing.
                reject("pinnacle_on_a_different_line")
                continue
            interpolated = interpolate_over_prob(
                rungs, line, pc.max_interpolation_distance
            )
            if interpolated is None:
                reject("pinnacle_line_not_bracketed")
                continue
            prob_over, lo, hi = interpolated
            anchor = rungs[min(rungs, key=lambda x: abs(x - line))]
            sharp = SharpLine(
                line=line,
                prob_over=prob_over,
                prob_under=1.0 - prob_over,
                price_over=pricing.fair_decimal(prob_over),
                price_under=pricing.fair_decimal(1.0 - prob_over),
                overround=anchor.overround,
                devig_spread=anchor.devig_spread,
                by_method_over={"interpolated": prob_over},
                last_update=anchor.last_update,
            )
            line_source = LINE_INTERPOLATED
            sharp_line_used = line
            leg_flags.append(f"line_interpolated_between_{lo:g}_and_{hi:g}")

        if not (m.min_overround <= sharp.overround <= m.max_overround):
            reject("sharp_overround_out_of_bounds")
            continue
        if line_source == LINE_EXACT and sharp.devig_spread > m.max_devig_spread:
            reject("devig_methods_disagree")
            continue

        sharp_age = _age_minutes(sharp.last_update, now)
        if sharp_age is not None and sharp_age > m.max_sharp_staleness_minutes:
            reject("sharp_quote_stale")
            continue
        book_age = _age_minutes(q.last_update, now)
        if book_age is not None and book_age > m.max_soft_staleness_minutes:
            reject("book_quote_stale")
            continue

        fair_prob = sharp.prob_for(q.side)
        push_prob, push_source = estimate_push_prob(rungs, line, pc.assumed_push_prob)
        if push_source == "assumed" and push_prob > 0:
            leg_flags.append(f"push_prob_assumed_{push_prob:.0%}")
        if fair_prob * (1.0 - push_prob) <= 0.0 or fair_prob >= 1.0:
            reject("degenerate_probability")
            continue

        liq = liquidity.assess(
            market=q.market,
            overround=sharp.overround,
            n_outcomes=2,
            fair_prob=fair_prob,
            minutes_to_start=minutes_out,
        )
        if m.min_liquidity > 0 and liq.score < m.min_liquidity:
            reject("market_too_thin")
            continue

        known = (
            rosters.lookup(meta.get("sport") or "", q.selection)
            if rosters is not None else None
        )
        leg = Leg(
            sport=meta.get("sport") or "",
            event_id=meta.get("event_id") or "",
            commence_time=commence,
            home_team=meta.get("home_team") or "",
            away_team=meta.get("away_team") or "",
            market=q.market,
            selection=q.selection,
            side=q.side,
            line=line,
            book=q.book,
            book_price=q.price,
            book_last_update=q.last_update,
            fair_prob=fair_prob,
            push_prob=push_prob,
            sharp_price_taken=sharp.price_for(q.side),
            sharp_price_other=sharp.other_price(q.side),
            sharp_overround=sharp.overround,
            devig_spread=sharp.devig_spread,
            fair_prob_by_method=sharp.by_method_for(q.side),
            sharp_last_update=sharp.last_update,
            sharp_line=sharp_line_used,
            market_tier=liq.tier,
            liquidity=liq.score,
            line_source=line_source,
            push_source=push_source,
            team=known.team if known else None,
            team_source=known.source if known else None,
            team_as_of=known.as_of.isoformat() if known and known.as_of else None,
            flags=tuple(leg_flags),
        )
        if leg.identity in seen:
            continue
        seen.add(leg.identity)
        out.append(leg)

    # Last, and only for players still without a club: some markets carry
    # exactly one player per team, so two of them in one game are
    # necessarily opponents. That needs no roster and no assumption, and it
    # is applied after the roster so it can never overwrite a real answer.
    sides = infer_sides(out)
    if sides:
        out = [
            replace(leg, team=sides[leg.selection], team_source=SOURCE_STRUCTURAL)
            if not leg.team and leg.selection in sides
            else leg
            for leg in out
        ]

    return out


def prefilter_legs(
    legs: Sequence[Leg],
    product: Product,
    cfg: Config,
    rejections: dict[str, int] | None = None,
) -> list[Leg]:
    """
    Narrow the candidate pool before the search sees it.

    Two jobs. The first is quality: a leg whose probability sits far below
    what the product needs is not rescued by correlation, and a leg at the
    extremes of the probability range carries more de-vig model risk than
    edge. The second is cost: the search is exponential in the number of
    candidates, so feeding it everything makes it slow without making it
    better.

    The break-even used is the most permissive one across the product's
    leg counts, so that nothing is dropped here which could still earn a
    place in a larger entry. The EV bar does the real filtering later.
    """
    rejections = rejections if rejections is not None else {}
    pc = cfg.parlay
    breakeven = product.easiest_breakeven()
    kept: list[Leg] = []

    def reject(reason: str) -> None:
        rejections[reason] = rejections.get(reason, 0) + 1

    for leg in legs:
        if leg.book_price is None and product.kind == KIND_PARLAY:
            reject("no_price_for_a_parlay_leg")
            continue
        if not (pc.min_leg_prob <= leg.hit_prob <= pc.max_leg_prob):
            reject("leg_probability_out_of_range")
            continue
        if product.is_pickem:
            if leg.edge_vs(breakeven) < pc.min_leg_edge:
                reject("leg_too_far_below_breakeven")
                continue
        else:
            ev = leg.single_leg_ev
            if ev is not None and ev < pc.min_leg_ev:
                reject("leg_single_bet_ev_too_low")
                continue
        kept.append(leg)

    # Best first, so a trimmed pool keeps the legs most likely to matter.
    kept.sort(key=lambda l: l.edge_vs(breakeven), reverse=True)
    if pc.max_candidates_per_group and len(kept) > pc.max_candidates_per_group:
        rejections["trimmed_to_candidate_cap"] = (
            rejections.get("trimmed_to_candidate_cap", 0)
            + len(kept) - pc.max_candidates_per_group
        )
        kept = kept[: pc.max_candidates_per_group]
    return kept


# --------------------------------------------------------------------------
# Tickets
# --------------------------------------------------------------------------


@dataclass
class Ticket:
    """
    A scored multi-leg entry.

    The two numbers to read together are `ev` and `ev_independent`. The
    second is the same ticket with the correlation matrix replaced by the
    identity -- what the pick'em site's fixed multiplier implicitly
    assumes. If the ticket is positive only in the first, then the entire
    case for betting it is an assumed correlation, and that is a very
    different claim from "this is a good bet". Nothing else in this module
    matters as much as keeping those two side by side.
    """

    legs: list[Leg]
    product: Product
    correlation: CorrelationMatrix

    joint_prob: float
    joint_prob_se: float
    joint_prob_independent: float
    hit_distribution: list[float]

    ev: float
    ev_se: float
    ev_independent: float
    payout_all_hit: float
    variance: float

    kelly_fraction: float
    log_optimal_fraction: float
    recommended_stake: float

    created_at: datetime
    suspect: bool = False
    flags: list[str] = field(default_factory=list)
    db_id: int | None = None

    @property
    def n_legs(self) -> int:
        return len(self.legs)

    @property
    def event_ids(self) -> list[str]:
        seen, out = set(), []
        for leg in self.legs:
            if leg.event_id not in seen:
                seen.add(leg.event_id)
                out.append(leg.event_id)
        return out

    @property
    def same_game(self) -> bool:
        return len(self.event_ids) == 1

    @property
    def sports(self) -> list[str]:
        return sorted({leg.sport for leg in self.legs})

    @property
    def commence_time(self) -> datetime:
        """The first leg to start: after this the ticket cannot be placed."""
        return min(leg.commence_time for leg in self.legs)

    @property
    def ev_per_variance(self) -> float:
        """
        Expected value per unit of variance -- the other question.

        Ranking on EV alone puts the highest-multiple tickets on top
        because a 20x payout carries enormous variance and a little of it
        leaks into the mean. This ranking asks instead which ticket earns
        the most per unit of risk, which for a finite bankroll is closer
        to the question that matters. Both orderings are reported because
        they genuinely are different questions.
        """
        if self.variance <= 0:
            return 0.0
        return self.ev / self.variance

    @property
    def correlation_is_load_bearing(self) -> bool:
        """The ticket is only positive because correlation was assumed."""
        return self.ev > 0 and self.ev_independent <= 0

    @property
    def correlation_lift(self) -> float:
        """How much of the EV the correlation assumption is supplying."""
        return self.ev - self.ev_independent

    @property
    def key(self) -> frozenset:
        """Identity for deduplication: a ticket is its set of legs."""
        return frozenset(leg.identity for leg in self.legs)

    @property
    def description(self) -> str:
        return " + ".join(leg.description for leg in self.legs)

    def to_row(self) -> dict:
        return {
            "created_at": self.created_at.isoformat(),
            "product": self.product.key,
            "book": self.product.book,
            "kind": self.product.kind,
            "n_legs": self.n_legs,
            "sports": ",".join(self.sports),
            "event_ids": ",".join(self.event_ids),
            "commence_time": self.commence_time.isoformat(),
            "same_game": int(self.same_game),
            "joint_prob": self.joint_prob,
            "joint_prob_se": self.joint_prob_se,
            "joint_prob_independent": self.joint_prob_independent,
            "hit_distribution": ";".join(f"{p:.6f}" for p in self.hit_distribution),
            "ev": self.ev,
            "ev_se": self.ev_se,
            "ev_independent": self.ev_independent,
            "payout_all_hit": self.payout_all_hit,
            "variance": self.variance,
            "ev_per_variance": self.ev_per_variance,
            "kelly_fraction": self.kelly_fraction,
            "log_optimal_fraction": self.log_optimal_fraction,
            "recommended_stake": self.recommended_stake,
            "correlation_summary": self.correlation.summary(),
            "correlation_all_prior": int(self.correlation.all_prior),
            "draws": None,
            "suspect": int(self.suspect),
            "flags": ",".join(self.flags),
        }


def effective_decimal(joint_prob: float, ev: float) -> float:
    """
    The single-bet price that would carry this ticket's expected value.

    A flex entry pays on five different outcomes, so it has no "price" in
    the sense `pricing.kelly_fraction` expects. Collapsing it to the
    decimal odds that reproduce the same EV at the same all-hit
    probability is what lets the existing staking code be reused rather
    than reimplemented, and it is exact for the all-or-nothing structures
    that are the main target.
    """
    if joint_prob <= 0:
        return 0.0
    return (1.0 + ev) / joint_prob


def evaluate_ticket(
    legs: Sequence[Leg],
    product: Product,
    cfg: Config,
    priors: PriorSet,
    estimates: EstimateStore | None = None,
    base: np.ndarray | None = None,
    draws: int | None = None,
    now: datetime | None = None,
    full: bool = True,
) -> Ticket:
    """
    Score one candidate ticket.

    `full=False` is the search path: it skips the independence comparison
    and the log-optimal staking solve, which together roughly double the
    cost and neither of which changes the ranking. Finalists are always
    re-scored with `full=True` at the full draw count.
    """
    now = now or datetime.now(timezone.utc)
    pc = cfg.parlay
    legs = list(legs)
    n = len(legs)
    draws = draws or pc.draws

    corr = correlation.assemble(
        legs,
        priors=priors,
        estimates=estimates,
        min_sample=pc.min_correlation_sample,
    )
    hit_probs = [leg.hit_prob for leg in legs]
    push_probs = [leg.push_prob for leg in legs]

    sim = copula.simulate(
        hit_probs, corr.matrix, push_probs,
        draws=draws, seed=pc.seed, base=base,
    )
    multiples = product.multiple_grid(n)
    returns = sim.payout_per_draw(multiples)

    ev = float(returns.mean()) - 1.0
    ev_se = float(returns.std(ddof=1)) / math.sqrt(sim.draws)
    variance = float(returns.var())

    # The independence baseline is computed exactly rather than simulated.
    # It is the number the correlated estimate is judged against, and a
    # comparison between two noisy numbers is half as informative as one
    # between a noisy number and an exact one.
    independent = copula.independent_grid(hit_probs, push_probs)
    ev_independent = float((independent * multiples).sum()) - 1.0
    log_optimal = copula.log_optimal_fraction(returns) if full else 0.0

    joint_prob = sim.joint_prob
    kelly = pricing.kelly_fraction(joint_prob, effective_decimal(joint_prob, ev))

    ticket = Ticket(
        legs=legs,
        product=product,
        correlation=corr,
        joint_prob=joint_prob,
        joint_prob_se=sim.joint_prob_se,
        joint_prob_independent=float(independent[0, n]),
        hit_distribution=sim.hit_distribution,
        ev=ev,
        ev_se=ev_se,
        ev_independent=ev_independent,
        payout_all_hit=product.all_hit_multiple(n),
        variance=variance,
        kelly_fraction=kelly,
        log_optimal_fraction=log_optimal,
        recommended_stake=0.0,
        created_at=now,
    )
    if full:
        apply_guards(ticket, cfg)
        ticket.recommended_stake = ticket_stake(ticket, cfg)
    return ticket


# --------------------------------------------------------------------------
# Guards
#
# scan.py rejects and flags rather than trusting, and the reasoning
# transfers directly: on a board of thousands of candidate tickets the
# ones that look best are, before anything else, the ones most likely to
# be built on something wrong. Every guard below has a test that trips it.
# --------------------------------------------------------------------------


def apply_guards(ticket: Ticket, cfg: Config) -> Ticket:
    """Flag what cannot be trusted, and withhold a stake where it matters."""
    pc = cfg.parlay
    flags: list[str] = []
    suspect = False

    if ticket.ev > pc.max_plausible_ev:
        # A +40% pick'em ticket means a wrong line, a stale quote, or a
        # payout table that does not match what the account offers. It
        # does not mean a 40% edge.
        flags.append(f"ev_{ticket.ev:.0%}_implausible")
        suspect = True

    if ticket.correlation_is_load_bearing:
        # The single most important honesty check in the tool.
        flags.append("only_+ev_because_of_assumed_correlation")
        suspect = True

    if ticket.correlation.all_prior:
        flags.append("correlation_entirely_from_priors")

    if ticket.correlation.projected:
        flags.append("correlation_matrix_projected_to_psd")

    if ticket.ev_se > pc.max_se_ratio * abs(ticket.ev) and ticket.ev != 0:
        flags.append(f"monte_carlo_se_{ticket.ev_se:.1%}_vs_edge_{ticket.ev:+.1%}")
        suspect = True

    negative = ticket.correlation.most_negative
    if negative is not None and negative.rho <= pc.negative_pair_threshold:
        a, b = ticket.legs[negative.i], ticket.legs[negative.j]
        flags.append(
            f"negatively_correlated_pair({a.description} / {b.description} "
            f"{negative.rho:+.2f})"
        )

    if not ticket.product.verified:
        flags.append("payout_table_unverified")

    if all(p.source == correlation.SOURCE_DEFAULT for p in ticket.correlation.pairs):
        # Not one pair matched a structural prior, so the ticket is being
        # scored as very nearly independent -- which is not what the tool
        # is for. Two different causes, and the fix differs, so the flag
        # says which: either the legs' teams are unknown and every pair
        # fell back to the generic number, or the teams ARE known and no
        # prior has been written for this combination of markets.
        if any(leg.team for leg in ticket.legs):
            flags.append(
                "no_prior_for_these_markets(add one to correlation_priors.yaml)"
            )
        else:
            flags.append(
                "no_structural_correlation_matched(teams unknown; see "
                "`betedge parlay rosters`)"
            )

    if any(
        p.relation == correlation.SAME_GAME for p in ticket.correlation.pairs
    ) and not any(leg.team for leg in ticket.legs):
        flags.append("leg_teams_unknown_same_game_priors_are_blends")

    if any(leg.line_source == LINE_INTERPOLATED for leg in ticket.legs):
        flags.append("leg_line_interpolated_not_matched")
        suspect = True

    if any(leg.push_source == "assumed" and leg.push_prob > 0 for leg in ticket.legs):
        flags.append("push_probability_assumed")

    if not ticket.same_game and ticket.product.kind == KIND_PARLAY:
        hold = copula.compounded_hold(
            leg.sharp_overround for leg in ticket.legs
        )
        flags.append(f"cross_game_parlay_compounded_hold_{hold:.1%}")
        suspect = True

    if ticket.product.kind == KIND_PARLAY and ticket.same_game:
        flags.append("sgp_price_may_be_discounted_by_the_book")

    for leg in ticket.legs:
        for f in leg.flags:
            flags.append(f"leg:{f}")

    ticket.flags = flags
    ticket.suspect = suspect
    if suspect:
        ticket.recommended_stake = 0.0
    return ticket


def legs_are_compatible(legs: Sequence[Leg], product: Product) -> tuple[bool, str]:
    """
    Whether a set of legs may be entered together at all.

    Two legs on the same prop are never allowed, in either direction: the
    same player's Over 6.5 and Over 7.5 is one bet entered twice, and his
    Over and Under together is a bet against yourself that no payout
    structure will accept.
    """
    props: set[tuple] = set()
    players: dict[str, int] = {}
    for leg in legs:
        if leg.prop_identity in props:
            return False, "two legs on the same market and player"
        props.add(leg.prop_identity)
        name = (leg.selection or "").strip().lower()
        players[name] = players.get(name, 0) + 1
    if not product.allows_same_player:
        repeated = [name for name, count in players.items() if count > 1]
        if repeated:
            return False, f"{product.key} does not allow the same player twice"
    return True, ""


def ticket_is_scorable(legs: Sequence[Leg]) -> tuple[bool, str]:
    """
    A ticket is scored only when every leg has a real Pinnacle reference.

    Kept as an explicit check rather than left to `build_legs`, because it
    is the one rule that must hold no matter where the legs came from --
    a leg hand-entered at the CLI has to clear it too.
    """
    for leg in legs:
        if leg.fair_prob is None or not (0.0 < leg.fair_prob < 1.0):
            return False, f"{leg.description} has no usable Pinnacle probability"
        if leg.sharp_price_taken is None or leg.sharp_price_other is None:
            return False, f"{leg.description} has no Pinnacle reference price"
    return True, ""


# --------------------------------------------------------------------------
# Staking
# --------------------------------------------------------------------------


def ticket_stake(ticket: Ticket, cfg: Config) -> float:
    """
    Recommended stake for one ticket.

    `pricing.kelly_fraction` is reused, on the collapsed price that
    reproduces this ticket's expected value -- but a parlay breaks Kelly's
    assumptions harder than a single bet does, in three separate ways:

      * the payoff is lumpy, so the log-growth optimum is further from the
        formula's answer than on a two-outcome bet;
      * the probability estimate compounds error across every leg, and
        four legs each 1 point off can be 4 points off together;
      * correlated tickets have fat tails in exactly the direction that
        hurts -- the legs that lose, lose together.

    So the fraction is halved again relative to the single-bet path
    (an eighth of Kelly by default, against a quarter), and capped harder.
    The log-optimal fraction computed from the Monte Carlo draws is
    reported alongside as a cross-check: where the two disagree badly, the
    formula is not describing the bet.
    """
    if ticket.suspect or ticket.ev <= 0:
        return 0.0
    return pricing.stake(
        ticket.joint_prob,
        effective_decimal(ticket.joint_prob, ticket.ev),
        bankroll=cfg.bankroll.amount,
        kelly_multiplier=cfg.parlay.kelly_multiplier,
        max_fraction=cfg.parlay.max_ticket_fraction,
        min_stake=cfg.bankroll.min_stake,
        round_to=cfg.bankroll.round_to,
    )


def cap_exposure(tickets: Sequence[Ticket], cfg: Config) -> list[Ticket]:
    """
    Trim stakes so no one game, and no one board, carries too much.

    The per-game cap is the one that does not exist in the single-bet
    path, and it is the one that matters most here. Five tickets built off
    the same game share most of their legs and all of their weather, their
    injuries and their game script: they are one bet with five names, and
    sizing them as five independent positions is how a bankroll
    disappears in an afternoon.

    Applied greedily in rank order rather than by scaling everything
    proportionally, so the best ticket is funded in full and the marginal
    ones are the ones that get cut -- which is the right way round.
    """
    ordered = sorted(tickets, key=lambda t: t.ev, reverse=True)
    game_cap = cfg.bankroll.amount * cfg.parlay.max_game_exposure_fraction
    total_cap = cfg.bankroll.amount * cfg.bankroll.max_total_exposure_fraction
    per_game: dict[str, float] = {}
    total = 0.0

    for ticket in ordered:
        if ticket.recommended_stake <= 0:
            continue
        headroom = total_cap - total
        for event_id in ticket.event_ids:
            headroom = min(headroom, game_cap - per_game.get(event_id, 0.0))
        if headroom < ticket.recommended_stake:
            allowed = max(0.0, headroom)
            if cfg.bankroll.round_to > 0:
                allowed = (
                    math.floor(allowed / cfg.bankroll.round_to) * cfg.bankroll.round_to
                )
            if allowed < cfg.bankroll.min_stake:
                ticket.recommended_stake = 0.0
                ticket.flags.append("no_room_left_under_the_exposure_caps")
                continue
            ticket.recommended_stake = allowed
            ticket.flags.append("stake_trimmed_for_exposure_caps")
        total += ticket.recommended_stake
        for event_id in ticket.event_ids:
            per_game[event_id] = per_game.get(event_id, 0.0) + ticket.recommended_stake

    return list(tickets)


def exposure_by_game(tickets: Sequence[Ticket]) -> dict[str, float]:
    out: dict[str, float] = {}
    for t in tickets:
        for event_id in t.event_ids:
            out[event_id] = out.get(event_id, 0.0) + t.recommended_stake
    return out


# --------------------------------------------------------------------------
# Search
#
# Full enumeration is hopeless: there are 658,008 five-leg subsets of 40
# candidates, each needing its own correlation matrix and its own
# simulation. Beam search keeps the best `beam_width` partial tickets at
# each size and extends only those, which turns an exponential problem
# into a linear one at the cost of no longer being guaranteed the optimum.
# That trade is easy here -- the ranking is noisy to a percentage point
# anyway, and the tenth-best ticket is as useful as the best.
# --------------------------------------------------------------------------


def group_legs(legs: Sequence[Leg], cfg: Config) -> list[list[Leg]]:
    """
    Split candidates into pools that may be combined with one another.

    Default is one pool per event. Correlation is a property of a shared
    game -- shared weather, shared pace, shared game script, a shared ball
    -- and a quarterback in one game with a centre in another has none of
    it. Combining them adds variance and nothing else, which is the exact
    opposite of what the tool is for.

    `same_slate` widens this to a sport's games starting near each other,
    for the rare structural priors that reach across games. `any` removes
    the restriction entirely and is there for completeness, not for use.
    """
    mode = cfg.parlay.grouping
    if mode == "any":
        return [list(legs)] if legs else []

    if mode == "same_slate":
        pools: dict[tuple, list[Leg]] = {}
        window = cfg.parlay.slate_hours * 3600.0
        for leg in legs:
            bucket = math.floor(leg.commence_time.timestamp() / window) if window else 0
            pools.setdefault((leg.sport, bucket), []).append(leg)
        return [v for v in pools.values() if len(v) >= cfg.parlay.min_legs]

    pools = {}
    for leg in legs:
        pools.setdefault(leg.event_id, []).append(leg)
    return [v for v in pools.values() if len(v) >= cfg.parlay.min_legs]


def beam_search(
    candidates: Sequence[Leg],
    product: Product,
    cfg: Config,
    priors: PriorSet,
    estimates: EstimateStore | None = None,
    base: np.ndarray | None = None,
    now: datetime | None = None,
    offered_decimal: float | None = None,
) -> list[Ticket]:
    """
    Build and rank tickets from one pool of legs.

    Scored on expected value at every step, never on the payout multiple.
    A bigger multiple is available for free by adding another leg, so a
    search that chased it would return a six-leg ticket every time and
    would be wrong every time.
    """
    pc = cfg.parlay
    candidates = list(candidates)
    template = product.kind == KIND_PARLAY and not product.payouts
    max_legs = min(
        pc.max_legs,
        pc.max_legs if template else product.max_legs,
        len(candidates),
    )
    min_legs = max(pc.min_legs, pc.min_legs if template else product.min_legs)
    if max_legs < min_legs:
        return []

    def score(combo: Sequence[Leg]) -> Ticket | None:
        ok, _why = legs_are_compatible(combo, product)
        if not ok:
            return None
        # An offered price describes one specific entry, so it is only
        # applied at the leg count the user asked for. Other sizes of a
        # parlay are priced from the legs, which is an upper bound a
        # same-game parlay will never actually pay -- and is flagged as one.
        price = offered_decimal if len(combo) == max_legs else None
        live = resolve_product(product, combo, price)
        if live.vector_for(len(combo)) is None:
            return None
        return evaluate_ticket(
            combo, live, cfg, priors, estimates,
            base=base, draws=pc.search_draws, now=now, full=False,
        )

    beam: list[tuple[float, list[Leg]]] = [(0.0, [leg]) for leg in candidates]
    found: dict[frozenset, Ticket] = {}

    for size in range(2, max_legs + 1):
        seen: set[frozenset] = set()
        scored: list[tuple[float, list[Leg]]] = []
        for _prev_ev, partial in beam:
            held = {leg.identity for leg in partial}
            for leg in candidates:
                if leg.identity in held:
                    continue
                combo = partial + [leg]
                key = frozenset(l.identity for l in combo)
                if key in seen:
                    continue          # a permutation of something already tried
                seen.add(key)
                ticket = score(combo)
                if ticket is None:
                    continue
                scored.append((ticket.ev, combo))
                if size >= min_legs and (
                    key not in found or ticket.ev > found[key].ev
                ):
                    found[key] = ticket
        if not scored:
            break
        scored.sort(key=lambda item: item[0], reverse=True)
        beam = scored[: pc.beam_width]

    return sorted(found.values(), key=lambda t: t.ev, reverse=True)


def rescore(
    tickets: Sequence[Ticket],
    cfg: Config,
    priors: PriorSet,
    estimates: EstimateStore | None = None,
    now: datetime | None = None,
    limit: int | None = None,
) -> list[Ticket]:
    """
    Re-simulate finalists at the full draw count, with the independence
    comparison and the guards.

    The search runs on few draws and shared random numbers, which ranks
    candidates well but leaves each individual EV noisier than it should
    be before anyone acts on it. So the shortlist is measured again
    properly, on its own draws, and can and does reorder slightly.
    """
    shortlist = list(tickets)[: limit or len(tickets)]
    out = [
        evaluate_ticket(
            t.legs, t.product, cfg, priors, estimates, now=now, full=True
        )
        for t in shortlist
    ]
    return sorted(out, key=lambda t: t.ev, reverse=True)


def rank_by_ev(tickets: Sequence[Ticket], limit: int) -> list[Ticket]:
    return sorted(tickets, key=lambda t: t.ev, reverse=True)[:limit]


def rank_by_ev_per_variance(tickets: Sequence[Ticket], limit: int) -> list[Ticket]:
    """
    The other ranking. Deliberately reported separately rather than blended
    into one score, because "which ticket makes the most money" and "which
    ticket makes the most money per unit of risk" are different questions
    and averaging them answers neither.
    """
    return sorted(tickets, key=lambda t: t.ev_per_variance, reverse=True)[:limit]


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


@dataclass
class ParlayScanResult:
    started_at: datetime
    finished_at: datetime
    sports: list[str]
    products: list[str]
    tickets: list[Ticket]
    legs_built: int = 0
    legs_after_filter: int = 0
    groups_searched: int = 0
    events_scanned: int = 0
    candidates_evaluated: int = 0
    credits_spent: int = 0
    credits_remaining: int | None = None
    rejections: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    rosters: RosterBook | None = None
    roster_refresh: Any = None

    @property
    def team_coverage(self) -> dict[str, int]:
        """
        How many legs the roster book could actually name a club for.

        The number to watch: a book of nine thousand players that covers
        none of tonight's starters buys nothing, and every leg it misses
        falls back to the weaker blended prior.
        """
        return {
            "legs": self.legs_built,
            "with_team": sum(
                1 for t in self.tickets for leg in t.legs if leg.team
            ),
            "ticket_legs": sum(t.n_legs for t in self.tickets),
        }

    @property
    def clean(self) -> list[Ticket]:
        return [t for t in self.tickets if not t.suspect]

    @property
    def suspect(self) -> list[Ticket]:
        return [t for t in self.tickets if t.suspect]

    @property
    def playable(self) -> list[Ticket]:
        return [t for t in self.clean if t.recommended_stake > 0]


@dataclass
class ParlayContext:
    """Everything a parlay run needs that does not come from the odds API."""

    table: PayoutTable
    priors: PriorSet
    estimates: EstimateStore | None
    rosters: RosterBook
    refresh: rosters_mod.RefreshReport | None = None


def load_context(
    cfg: Config,
    db=None,
    sports: Sequence[str] | None = None,
    refresh: bool | None = None,
    now: datetime | None = None,
    opener=None,
) -> ParlayContext:
    """
    Load the payout table, the priors, the fitted correlations and the
    roster book, refreshing any roster feed that has gone cold.

    The refresh never fails a run. A roster feed is an enrichment -- losing
    it costs the sharp same-team priors and falls back to the blended
    ones, which is a smaller loss than not scanning at all.
    """
    now = now or datetime.now(timezone.utc)
    table = PayoutTable.load(cfg.parlay.payouts_path)
    priors = PriorSet.load(cfg.parlay.priors_path)
    estimates = EstimateStore.from_db(db) if db is not None else None

    report = None
    should_refresh = (
        cfg.parlay.roster_auto_refresh if refresh is None else refresh
    )
    if db is not None and should_refresh and sports:
        report = rosters_mod.refresh_providers(
            db, sports,
            refresh_days=cfg.parlay.roster_refresh_days,
            now=now, opener=opener,
        )
        for error in report.errors:
            log.warning("roster refresh: %s", error)

    book = rosters_mod.load_book(
        db,
        manual_path=cfg.parlay.rosters_path,
        max_age_days=cfg.parlay.roster_max_age_days,
        now=now,
    )
    return ParlayContext(table, priors, estimates, book, report)


def scan_parlays(
    cfg: Config,
    client: OddsApiClient,
    db=None,
    sports: Sequence[str] | None = None,
    products: Sequence[str] | None = None,
    now: datetime | None = None,
    max_events_per_sport: int | None = None,
    offered_price: float | None = None,
) -> ParlayScanResult:
    """
    Pull a slate, build legs, search for tickets, rank them.

    Credit discipline is the single-bet scanner's, unchanged: one free
    event list per sport, the time window filtered before anything is
    billed, and the client's own budget ceiling stopping the run rather
    than a count kept here. This must not be able to burn a month's quota,
    and the per-event prop endpoint is precisely the thing that can.
    """
    now = now or datetime.now(timezone.utc)
    started = now
    sports = list(sports if sports is not None else cfg.sports)
    sports, blocked = cfg.allowed(sports)

    context = load_context(cfg, db, sports=sports, now=now)
    table, priors, estimates = context.table, context.priors, context.estimates

    wanted = list(products or cfg.parlay.products)
    chosen = [table.get(key) for key in wanted]

    state = ParlayScanResult(
        started_at=started, finished_at=now, sports=sports,
        products=[p.key for p in chosen], tickets=[],
    )
    for sport in blocked:
        state.rejections["sport_excluded"] = state.rejections.get("sport_excluded", 0) + 1

    target_books = sorted(
        {p.book for p in chosen if p.book} | set(cfg.parlay.pickem_books)
    )
    books = [cfg.books.sharp] + [b for b in target_books if b != cfg.books.sharp]

    all_legs: list[Leg] = []
    budget_exhausted = False

    for sport in sports:
        if budget_exhausted:
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
        except Exception as exc:  # noqa: BLE001
            state.errors.append(f"{sport}: could not list events: {exc}")
            continue

        events = _events_in_window(
            events, now, cfg, max_hours=cfg.prop_window_hours(sport)
        )
        if max_events_per_sport is not None:
            # `is not None` rather than a truth test: this endpoint bills
            # per event per market, and a user who types --max-events 0
            # means zero. Reading it as "no limit" would sweep the whole
            # slate at exactly the moment they asked for none of it.
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
                budget_exhausted = True
                break
            except Exception as exc:  # noqa: BLE001
                state.errors.append(f"{sport}/{event.get('id')}: {exc}")
                continue
            if not payload:
                continue

            state.events_scanned += 1
            meta, quotes = parse_event_odds(payload)
            all_legs.extend(
                build_legs(
                    meta, quotes, cfg, target_books,
                    now=now, rejections=state.rejections,
                    rosters=context.rosters,
                )
            )

    state.legs_built = len(all_legs)
    state.rosters = context.rosters
    state.roster_refresh = context.refresh
    state.tickets = build_tickets(
        all_legs, chosen, cfg, priors, estimates,
        now=now, state=state, offered_price=offered_price,
    )

    state.finished_at = datetime.now(timezone.utc)
    state.credits_spent = client.quota.spent_this_session
    state.credits_remaining = client.quota.remaining
    return state


def build_tickets(
    legs: Sequence[Leg],
    products: Sequence[Product],
    cfg: Config,
    priors: PriorSet,
    estimates: EstimateStore | None = None,
    now: datetime | None = None,
    state: ParlayScanResult | None = None,
    offered_price: float | None = None,
) -> list[Ticket]:
    """
    Search every (product, group) pair and return the ranked shortlist.

    One pool of random numbers is shared by every candidate in the run.
    That is what makes the search's comparisons precise enough to rank on
    at only a few thousand draws -- two tickets scored on the same draws
    differ by their structure, not by their luck.
    """
    now = now or datetime.now(timezone.utc)
    pc = cfg.parlay
    rejections = state.rejections if state is not None else {}
    base = copula.standard_normals(pc.search_draws, max(pc.max_legs, 2), pc.seed)

    found: dict[tuple, Ticket] = {}
    kept_total = 0
    groups = 0

    for product in products:
        for pool in group_legs(
            [leg for leg in legs if _leg_serves(leg, product, cfg)], cfg
        ):
            kept = prefilter_legs(pool, product, cfg, rejections)
            kept_total += len(kept)
            if len(kept) < max(pc.min_legs, product.min_legs):
                continue
            groups += 1
            for ticket in beam_search(
                kept, product, cfg, priors, estimates, base=base, now=now,
                offered_decimal=offered_price,
            ):
                found[(ticket.product.key, ticket.key)] = ticket

    if state is not None:
        state.legs_after_filter = kept_total
        state.groups_searched = groups
        state.candidates_evaluated = len(found)

    shortlist = rescore(
        rank_by_ev(list(found.values()), pc.top_n * 3),
        cfg, priors, estimates, now=now,
    )
    # The EV bar applies to guarded tickets too. A ticket that tripped a
    # guard AND sits below the bar is not an interesting near-miss, it is
    # just a bad ticket, and printing it only dilutes the suspect list.
    playable = [t for t in shortlist if t.ev >= pc.min_ev]
    cap_exposure([t for t in playable if not t.suspect], cfg)
    return playable


def _leg_serves(leg: Leg, product: Product, cfg: Config) -> bool:
    """Whether this leg can be entered on this product's book."""
    if product.book:
        return leg.book == product.book
    return leg.book in cfg.parlay.pickem_books


# --------------------------------------------------------------------------
# Coverage
#
# The sport list at the top of this module is a starting point, not a
# finding. This command re-derives it from what the API is actually
# serving today, which is the only way it stays right across a season
# turnover.
# --------------------------------------------------------------------------


#: Pick'em books worth ASKING about in a coverage probe, whether or not
#: they are configured. Cost is markets x ceil(books / 10), so naming ten
#: books costs exactly what naming one does -- which means the honest way
#: to find out whether a book is available is to ask for it and report
#: what came back, rather than to assert it from memory.
PICKEM_BOOK_CANDIDATES = ("underdog", "prizepicks")

#: Minimum matched legs before a pool is worth searching. Two legs is a
#: ticket, but a pool that small produces nothing worth having.
USABLE_MATCHED_LEGS = 8


@dataclass
class BookCoverage:
    """What one book actually offers against Pinnacle, on one sport."""

    book: str
    quotes: int = 0                 #: two-sided over/under selections seen
    on_a_pinnacle_market: int = 0   #: same player and stat as a Pinnacle market
    matched_on_same_line: int = 0   #: ...and on the same line, so usable
    players: set = field(default_factory=set)
    markets: set = field(default_factory=set)

    @property
    def match_rate(self) -> float:
        """
        Share of this book's quotes that are usable as legs.

        The number that decides whether a book is worth anything here. A
        book can post a thousand props and still be useless if it prices
        them at numbers Pinnacle does not touch, because a leg compared
        against a different line is not a measurement of anything.
        """
        if not self.quotes:
            return 0.0
        return self.matched_on_same_line / self.quotes

    @property
    def usable(self) -> bool:
        return self.matched_on_same_line >= USABLE_MATCHED_LEGS


@dataclass
class CoverageRow:
    sport: str
    events_in_window: int = 0
    events_probed: int = 0
    two_sided_sharp_markets: int = 0
    with_book_quote: int = 0
    matched_on_same_line: int = 0
    players_seen: set = field(default_factory=set)
    markets_seen: set = field(default_factory=set)
    by_book: dict = field(default_factory=dict)
    books_asked: tuple = ()
    credits_spent: int = 0
    error: str = ""

    @property
    def distinct_players(self) -> int:
        return len(self.players_seen)

    @property
    def match_rate(self) -> float:
        if not self.two_sided_sharp_markets:
            return 0.0
        return self.matched_on_same_line / self.two_sided_sharp_markets

    @property
    def silent_books(self) -> list[str]:
        """
        Books that were asked for and returned nothing at all.

        Either the API does not carry them, or they are not pricing this
        sport right now. Both mean the same thing for the optimizer -- no
        legs -- and it is worth saying out loud rather than leaving as an
        absence in a table.
        """
        return [b for b in self.books_asked if not self.by_book.get(b)]

    @property
    def best_book(self) -> "BookCoverage | None":
        return max(
            self.by_book.values(),
            key=lambda b: b.matched_on_same_line,
            default=None,
        )

    @property
    def usable(self) -> bool:
        """Whether any single book offers enough matched legs to search."""
        return any(b.usable for b in self.by_book.values())


@dataclass
class CoverageReport:
    rows: list[CoverageRow]
    excluded: dict[str, str]
    checked_at: datetime
    credits_spent: int = 0

    @property
    def recommended(self) -> list[str]:
        return [r.sport for r in self.rows if r.usable]


def probe_coverage(
    cfg: Config,
    client: OddsApiClient,
    sports: Sequence[str] | None = None,
    max_events_per_sport: int = 2,
    now: datetime | None = None,
    books_to_probe: Sequence[str] | None = None,
) -> CoverageReport:
    """
    Ask the API, per sport, how many two-sided Pinnacle prop markets exist
    and how many of them a target book quotes on the SAME line.

    That last number is the one that matters. A sport can have a thousand
    Pinnacle props and still be useless here if the pick'em site posts
    different numbers, because a leg compared against a different line is
    not a measurement of anything.

    Deliberately capped at a couple of events per sport: this is a probe,
    not a scan, and it bills like the prop endpoint does.
    """
    now = now or datetime.now(timezone.utc)
    sports = list(sports if sports is not None else DEFAULT_PROP_SPORTS)
    sports, _blocked = cfg.allowed(sports)
    target_books = sorted(
        set(books_to_probe)
        if books_to_probe
        else set(cfg.parlay.pickem_books)
        | set(cfg.books.soft)
        | set(PICKEM_BOOK_CANDIDATES)
    )
    books = [cfg.books.sharp] + [b for b in target_books if b != cfg.books.sharp]

    rows: list[CoverageRow] = []
    for sport in sports:
        row = CoverageRow(sport=sport)
        before = client.quota.spent_this_session
        try:
            markets = cfg.markets_for_sport(sport)
            events = client.events(sport)
        except Exception as exc:  # noqa: BLE001
            row.error = str(exc)
            rows.append(row)
            continue

        events = _events_in_window(events, now, cfg, max_hours=cfg.prop_window_hours(sport))
        row.events_in_window = len(events)
        for event in events[:max_events_per_sport]:
            try:
                payload = client.event_odds(sport, event["id"], markets, books)
            except CreditBudgetExceeded as exc:
                row.error = str(exc)
                break
            except Exception as exc:  # noqa: BLE001
                row.error = str(exc)
                continue
            if not payload:
                continue
            row.events_probed += 1
            _meta, quotes = parse_event_odds(payload)
            ladder = sharp_ladder(quotes, cfg.books.sharp, cfg.model.devig_method)
            row.two_sided_sharp_markets += sum(len(v) for v in ladder.values())
            row.players_seen.update(sel for _mk, sel in ladder)
            row.markets_seen.update(mk for mk, _sel in ladder)

            book_quotes = [
                q for q in quotes
                if q.book in target_books and _is_two_sided(q.side) and q.line is not None
            ]
            for q in book_quotes:
                seen = row.by_book.setdefault(q.book, BookCoverage(book=q.book))
                seen.quotes += 1
                seen.players.add(q.selection)
                seen.markets.add(q.market)
                rungs = ladder.get((q.market, q.selection))
                if not rungs:
                    continue
                seen.on_a_pinnacle_market += 1
                row.with_book_quote += 1
                if float(q.line) in rungs:
                    seen.matched_on_same_line += 1
                    row.matched_on_same_line += 1
        row.books_asked = tuple(target_books)
        row.credits_spent = client.quota.spent_this_session - before
        rows.append(row)

    return CoverageReport(
        rows=rows,
        excluded=dict(PROP_OPTIMIZER_EXCLUDED),
        checked_at=now,
        credits_spent=sum(r.credits_spent for r in rows),
    )
