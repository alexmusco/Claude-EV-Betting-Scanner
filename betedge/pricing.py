"""
Pricing math: de-vigging, expected value, and stake sizing.

This is the heart of the model. Everything else is plumbing.

The core idea
-------------
Pinnacle's posted prices contain vig (the book's margin). Raw implied
probability 1/decimal_odds therefore OVERSTATES the true probability -- the
two sides of a market sum to more than 1. To use Pinnacle as a "truth"
estimate you must first strip that margin out to recover a fair probability
p. Then the expected value of a bet at a soft book's decimal price d is:

    EV = p * (d - 1) - (1 - p)  =  p * d - 1

The old R model skipped de-vigging entirely and compared raw 1/price values
between books, then filtered on "difference > 0.5 * average Pinnacle vig".
That is a heuristic proxy for edge, not edge itself: it is biased toward
whichever side carries more of the book's margin, and the threshold drifts
with the vig of whatever slate happened to load that night.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import math

# --------------------------------------------------------------------------
# Odds format conversion
# --------------------------------------------------------------------------


def american_to_decimal(american: float) -> float:
    """+150 -> 2.50, -120 -> 1.8333."""
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def decimal_to_american(decimal: float) -> float:
    """2.50 -> +150, 1.8333 -> -120."""
    if decimal >= 2.0:
        return (decimal - 1.0) * 100.0
    return -100.0 / (decimal - 1.0)


def parse_price(value: float | str) -> float:
    """
    Accept a price in either format and return decimal.

    Sportsbooks in the US quote American, so that is what you read off the
    screen and what you will type. The maths needs decimal. Guessing
    between them is safe because the ranges do not overlap in practice:
    a decimal price is above 1.0 and, at 100.0, already implies +9900,
    while American is at or beyond 100 in either direction.

    Anything negative is unambiguously American.

        parse_price(-110)  -> 1.909
        parse_price(122)   -> 2.22
        parse_price(2.22)  -> 2.22
    """
    if isinstance(value, str):
        value = float(value.strip().lstrip("+"))
    value = float(value)
    if value < 0:
        # American odds never fall between -100 and 0; a value there is a
        # typo, not a price, and converting it yields a plausible-looking
        # number (-0.5 becomes decimal 201) rather than an error.
        if value > -100:
            raise ValueError(
                f"{value} is not a usable price: American odds are at or "
                "beyond -100."
            )
        return american_to_decimal(value)
    if value >= 100:
        return american_to_decimal(value)
    if value <= 1.0:
        raise ValueError(
            f"{value} is not a usable price: decimal odds must exceed 1.0, "
            "and American odds are at or beyond +/-100."
        )
    return value


def format_american(decimal: float) -> str:
    """Decimal to the string a sportsbook shows. 2.22 -> '+122'."""
    return f"{decimal_to_american(decimal):+.0f}"


def implied(decimal: float) -> float:
    """Raw (vigged) implied probability."""
    return 1.0 / decimal


def fair_decimal(prob: float) -> float:
    """Decimal odds that would make `prob` a break-even bet."""
    return 1.0 / prob


# --------------------------------------------------------------------------
# De-vigging
# --------------------------------------------------------------------------


def overround(decimals: Sequence[float]) -> float:
    """Total book margin. 0.045 means a 4.5% overround."""
    return sum(1.0 / d for d in decimals) - 1.0


def devig_multiplicative(decimals: Sequence[float]) -> list[float]:
    """
    Proportional / normalised method. p_i = q_i / sum(q).

    Simplest and most common. Removes margin in proportion to each side's
    raw probability, which means the favourite absorbs more of it in
    absolute terms. Tends to overprice heavy favourites relative to Shin.
    """
    q = [1.0 / d for d in decimals]
    total = sum(q)
    return [x / total for x in q]


def devig_additive(decimals: Sequence[float]) -> list[float]:
    """
    Equal absolute share. p_i = q_i - (sum(q) - 1) / n.

    Splits the margin evenly across outcomes. Can produce negative
    probabilities on very lopsided markets, so we clamp.
    """
    q = [1.0 / d for d in decimals]
    excess = (sum(q) - 1.0) / len(q)
    p = [max(1e-9, x - excess) for x in q]
    total = sum(p)
    return [x / total for x in p]


def devig_power(decimals: Sequence[float], tol: float = 1e-12) -> list[float]:
    """
    Power method. Find k such that sum(q_i ** k) == 1, then p_i = q_i ** k.

    Applies the margin multiplicatively in log-space, which handles
    longshot bias better than the proportional method. Solved by bisection
    on k, which is monotone in sum(q_i ** k) for q_i < 1.
    """
    q = [1.0 / d for d in decimals]
    if any(x >= 1.0 for x in q):
        # A price at or below 1.00 decimal is nonsense; fall back.
        return devig_multiplicative(decimals)

    lo, hi = 1.0, 1.0
    # Expand upper bound until sum drops below 1.
    for _ in range(200):
        if sum(x ** hi for x in q) <= 1.0:
            break
        hi *= 2.0
    else:
        return devig_multiplicative(decimals)

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        s = sum(x ** mid for x in q)
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:
            lo = mid
        else:
            hi = mid
    k = 0.5 * (lo + hi)
    p = [x ** k for x in q]
    total = sum(p)
    return [x / total for x in p]


def devig_shin(decimals: Sequence[float], tol: float = 1e-12) -> list[float]:
    """
    Shin (1992) method. Models the margin as arising from a proportion z of
    insider money, and backs out the probabilities the book would hold if
    there were none.

        p_i = ( sqrt(z^2 + 4(1-z) * q_i^2 / S) - z ) / (2(1-z))

    where S = sum(q) and z is solved so the p_i sum to 1.

    NOTE for two-outcome markets -- which is every market this scanner
    touches, since player props are over/under: Shin reduces exactly to the
    additive method. Verified numerically to ~1e-13 in the tests. So on
    props "shin" and "additive" are the same estimate, and the only real
    choice is between multiplicative, additive/Shin, and power. Shin earns
    its keep on three-way markets (soccer 1X2), which is why it is here.
    """
    q = [1.0 / d for d in decimals]
    S = sum(q)
    if S <= 1.0:
        # No margin (or a genuine arb); nothing to strip.
        return [x / S for x in q]

    def probs_for(z: float) -> list[float]:
        if z <= 0:
            return [x / S for x in q]
        out = []
        for x in q:
            inner = z * z + 4.0 * (1.0 - z) * (x * x) / S
            out.append((math.sqrt(max(inner, 0.0)) - z) / (2.0 * (1.0 - z)))
        return out

    lo, hi = 0.0, 0.99
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        s = sum(probs_for(mid))
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:
            lo = mid
        else:
            hi = mid
    z = 0.5 * (lo + hi)
    p = probs_for(z)
    total = sum(p)
    return [x / total for x in p]


DEVIG_METHODS = {
    "multiplicative": devig_multiplicative,
    "additive": devig_additive,
    "power": devig_power,
    "shin": devig_shin,
}


def devig(decimals: Sequence[float], method: str = "shin") -> list[float]:
    """De-vig a complete market. `decimals` must cover every outcome."""
    if method == "worst_case":
        # Most conservative estimate available: for each outcome take the
        # LOWEST fair probability any method produces. Guards against
        # flagging a bet that only one method's quirks make look good.
        grids = [f(decimals) for f in DEVIG_METHODS.values()]
        return [min(g[i] for g in grids) for i in range(len(decimals))]
    try:
        fn = DEVIG_METHODS[method]
    except KeyError as exc:
        raise ValueError(
            f"unknown devig method {method!r}; "
            f"choose from {sorted(DEVIG_METHODS)} or 'worst_case'"
        ) from exc
    return fn(decimals)


def fair_probs_all_methods(decimals: Sequence[float]) -> dict[str, list[float]]:
    """Fair probabilities under every method, for sensitivity reporting."""
    return {name: fn(decimals) for name, fn in DEVIG_METHODS.items()}


def devig_spread(decimals: Sequence[float]) -> float:
    """
    How much the de-vig methods disagree, in probability points, on the
    outcome where they disagree most. A wide spread means the market is
    lopsided enough that your fair estimate is method-dependent -- treat
    those edges with suspicion.

    Measuring only the first outcome, as this used to, is equivalent for a
    two-way market: the probabilities sum to 1 under every method, so the
    two disagreements are identical by construction. On a three-way market
    it is not equivalent, and worse, it is not even well defined. Which
    outcome lands first depends on how the payload happened to be ordered --
    in `evaluate_event` the outcomes are sorted by name, so it comes down to
    which team is alphabetically first. The same 1.25 / 6.00 / 11.00 market
    scores 0.028 or 0.014 depending on nothing but the teams' initials, and
    a guard set at 0.04 therefore fired or did not fire at random.

    Taking the maximum over every outcome makes the guard independent of
    ordering, which is the property it needed all along.
    """
    grids = [f(decimals) for f in DEVIG_METHODS.values()]
    return max(
        max(g[i] for g in grids) - min(g[i] for g in grids)
        for i in range(len(decimals))
    )


# --------------------------------------------------------------------------
# Expected value and staking
# --------------------------------------------------------------------------


def expected_value(fair_prob: float, offered_decimal: float) -> float:
    """
    EV per unit staked. 0.04 means +4% -- a $100 bet is worth $4 in the
    long run IF fair_prob is right.
    """
    return fair_prob * offered_decimal - 1.0


def kelly_fraction(fair_prob: float, offered_decimal: float) -> float:
    """
    Full-Kelly fraction of bankroll. f* = (p*d - 1) / (d - 1) = EV / (d - 1).
    Negative means no bet.
    """
    b = offered_decimal - 1.0
    if b <= 0:
        return 0.0
    return (fair_prob * offered_decimal - 1.0) / b


def stake(
    fair_prob: float,
    offered_decimal: float,
    bankroll: float,
    kelly_multiplier: float = 0.25,
    max_fraction: float = 0.02,
    min_stake: float = 0.0,
    round_to: float = 1.0,
) -> float:
    """
    Recommended stake in currency units.

    Full Kelly assumes you know p exactly. You do not -- p here is an
    estimate from one book's prices, on a market that book may itself be
    slow on. Fractional Kelly (default one quarter) plus a hard cap as a
    share of bankroll is the standard defence: it costs a little growth
    and removes most of the ruin risk from an overestimated edge.
    """
    f = kelly_fraction(fair_prob, offered_decimal)
    if f <= 0:
        return 0.0
    f = min(f * kelly_multiplier, max_fraction)
    amount = f * bankroll
    if round_to > 0:
        amount = round(amount / round_to) * round_to
    if amount < min_stake:
        return 0.0
    return amount


# --------------------------------------------------------------------------
# Convenience container for a two-way market
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TwoWayFair:
    """De-vigged view of a two-way (over/under, home/away) sharp market."""

    price_a: float
    price_b: float
    prob_a: float
    prob_b: float
    overround: float
    method: str
    method_spread: float

    @property
    def fair_price_a(self) -> float:
        return fair_decimal(self.prob_a)

    @property
    def fair_price_b(self) -> float:
        return fair_decimal(self.prob_b)


def two_way_fair(price_a: float, price_b: float, method: str = "shin") -> TwoWayFair:
    """De-vig a two-way sharp market into fair probabilities for both sides."""
    probs = devig([price_a, price_b], method=method)
    return TwoWayFair(
        price_a=price_a,
        price_b=price_b,
        prob_a=probs[0],
        prob_b=probs[1],
        overround=overround([price_a, price_b]),
        method=method,
        method_spread=devig_spread([price_a, price_b]),
    )


def no_vig_price(price_a: float, price_b: float, method: str = "shin") -> tuple[float, float]:
    """Fair decimal prices for both sides of a two-way market."""
    f = two_way_fair(price_a, price_b, method)
    return f.fair_price_a, f.fair_price_b
