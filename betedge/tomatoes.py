"""
Rotten Tomatoes threshold contracts: when the answer is already known, and
what to think when it is not.

Why this market at all
----------------------
A Tomatometer is not an opinion, it is a running proportion -- the share
of counted reviews that were Fresh. That makes a contract on "will this
film finish above 60%" a question about arithmetic and arrival counts
rather than about whether the film is good, and it is being priced by
people who are answering the second question.

The two regimes
---------------
MECHANICAL. A film at 94% on 180 reviews has 169 Fresh. Even if every
remaining review were Rotten, and even if 40 more arrive, the score
cannot fall below 169/220 = 76.8%. A 60% threshold on that film is
already decided and no model is involved -- the only input is a ceiling
on how many reviews are still to come, and that input enters
CONSERVATIVELY: overestimate it and the bounds widen and the tool takes
fewer bets. The failure mode is timidity, not loss. This is where the
reliable money is, and it is why `bounds` comes before any distribution
in this file.

PROBABILISTIC. When the threshold sits inside those bounds the outcome
genuinely is uncertain, and it is a Beta-Binomial: put a posterior on the
film's true Fresh rate given what has been counted, push it forward over
however many reviews are still to arrive, and read off the probability
that the final proportion clears the line. Exactly computable, no
simulation required.

The drift, and why it is not allowed to carry a bet
---------------------------------------------------
Reviews are not drawn from one urn. Festival and early-embargo reviews
skew positive -- those critics self-select and early access goes to
friendly outlets -- so a film at 94% on 20 reviews tends to settle lower
than 94%. That is a real effect and modelling it properly needs
historical review trajectories, which nobody publishes.

So `drift` defaults to ZERO and is labelled a prior, not a measurement,
exactly as the correlation priors are. An assessment whose sign depends
on the drift term is flagged and staked at nothing. Day one, the only
bets that get money are the ones that stand up with no drift assumed at
all -- which is mostly the mechanical case. The snapshots this tool
records are what eventually turn `drift` into something measured.

Rounding is not a detail
------------------------
Rotten Tomatoes displays a whole-number percentage and contracts settle
on what is displayed. A film sitting at 60.4% settles a "60 or above"
threshold YES. Getting the rounding wrong flips contracts outright, so
the displayed score is computed explicitly rather than left to whatever
the caller's formatter does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Kalshi's published fee coefficients. Fees are charged per ORDER and
# rounded up to the next cent, which matters more than it looks: a single
# contract at 50c pays 2c rather than 1.75c, so small orders are taxed
# harder than the formula alone suggests.
# ---------------------------------------------------------------------------
TAKER_COEFFICIENT = 0.07
MAKER_COEFFICIENT = 0.0175

#: Scopes a contract can settle on. RT publishes both and they differ by
#: several points on the same film, so a contract that does not say which
#: one it means is not scorable.
SCOPE_ALL_CRITICS = "all_critics"
SCOPE_TOP_CRITICS = "top_critics"
VALID_SCOPES = (SCOPE_ALL_CRITICS, SCOPE_TOP_CRITICS)


# ---------------------------------------------------------------------------
# A snapshot of the score
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tomatometer:
    """
    What the Tomatometer said at one moment: how many reviews were counted
    and how many of them were Fresh.

    The count is the whole point. "94%" is not a state -- 94% on 17
    reviews and 94% on 210 reviews are different contracts with different
    answers, and a source that gives the percentage without the count
    cannot drive any of this.
    """

    fresh: int
    total: int
    captured_at: datetime
    scope: str = SCOPE_ALL_CRITICS
    film: str = ""

    def __post_init__(self) -> None:
        if self.total < 0 or self.fresh < 0:
            raise ValueError("review counts cannot be negative")
        if self.fresh > self.total:
            raise ValueError(
                f"{self.fresh} fresh of {self.total} counted: more Fresh "
                "reviews than reviews"
            )
        if self.scope not in VALID_SCOPES:
            raise ValueError(
                f"scope {self.scope!r} is not one of {VALID_SCOPES}"
            )

    @property
    def rotten(self) -> int:
        return self.total - self.fresh

    @property
    def exact_score(self) -> float:
        """The true proportion, 0..100, unrounded."""
        if self.total == 0:
            return 0.0
        return 100.0 * self.fresh / self.total

    @property
    def displayed_score(self) -> int:
        """
        The whole number Rotten Tomatoes shows, which is what contracts
        settle on.

        Half-up, which is the ordinary convention and what the site
        appears to use. It is stated here rather than assumed silently
        because a film at exactly x.5 settles a threshold one way under
        half-up and the other way under banker's rounding, and that is a
        whole contract.
        """
        return int(math.floor(self.exact_score + 0.5))

    def age_hours(self, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        return (now - self.captured_at).total_seconds() / 3600.0

    def describe(self) -> str:
        name = f"{self.film}: " if self.film else ""
        return (
            f"{name}{self.displayed_score}% "
            f"({self.fresh}/{self.total} fresh, {self.scope})"
        )


# ---------------------------------------------------------------------------
# The mechanical half
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreBounds:
    """
    The range the final displayed score must land in, given a ceiling on
    how many more reviews can be counted.

    Both ends are attainable: `low` is every remaining review Rotten,
    `high` is every one Fresh. Nothing about the film's quality enters --
    this is counting.
    """

    low: int
    high: int
    max_new_reviews: int

    @property
    def is_settled(self) -> bool:
        """No room left to move at all."""
        return self.low == self.high

    def contains(self, score: int) -> bool:
        return self.low <= score <= self.high


def bounds(snapshot: Tomatometer, max_new_reviews: int) -> ScoreBounds:
    """
    Hard bounds on the final displayed score.

    `max_new_reviews` is a CEILING, not an estimate, and it is the only
    uncertain input. Getting it too high widens the bounds and makes the
    tool decline bets it could have taken; getting it too low invents
    certainty that is not there. So callers should err high, and the
    guards downstream treat a tight ceiling as the thing to justify.
    """
    if max_new_reviews < 0:
        raise ValueError("max_new_reviews cannot be negative")
    n = snapshot.total + max_new_reviews
    if n == 0:
        # No reviews now and none coming: there is no score to bound.
        return ScoreBounds(low=0, high=0, max_new_reviews=0)
    worst = Tomatometer(
        fresh=snapshot.fresh, total=n,
        captured_at=snapshot.captured_at, scope=snapshot.scope,
    )
    best = Tomatometer(
        fresh=snapshot.fresh + max_new_reviews, total=n,
        captured_at=snapshot.captured_at, scope=snapshot.scope,
    )
    return ScoreBounds(
        low=worst.displayed_score,
        high=best.displayed_score,
        max_new_reviews=max_new_reviews,
    )


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------

ABOVE = "above"
BELOW = "below"


@dataclass(frozen=True)
class Contract:
    """
    One Kalshi threshold market on a film's score.

    `settlement_verified` defaults to False for the same reason the payout
    ladders do: the scope, the threshold's inclusivity and the settlement
    timestamp are read off the contract by a person, and an EV computed
    against the wrong one of those is not slightly wrong. Nothing is
    staked on an unverified contract.
    """

    ticker: str
    film: str
    threshold: int
    direction: str = ABOVE
    #: True when the threshold itself counts as a win -- "60 or above"
    #: rather than "above 60". One point of difference, whole contracts.
    inclusive: bool = True
    scope: str = SCOPE_ALL_CRITICS
    settles_at: datetime | None = None
    settlement_verified: bool = False

    def __post_init__(self) -> None:
        if self.direction not in (ABOVE, BELOW):
            raise ValueError(f"direction must be {ABOVE!r} or {BELOW!r}")
        if not 0 <= self.threshold <= 100:
            raise ValueError("threshold must be a percentage")
        if self.scope not in VALID_SCOPES:
            raise ValueError(f"scope {self.scope!r} is not one of {VALID_SCOPES}")

    def resolves_yes(self, displayed_score: int) -> bool:
        """Whether a final displayed score settles this contract YES."""
        if self.direction == ABOVE:
            return (
                displayed_score >= self.threshold
                if self.inclusive
                else displayed_score > self.threshold
            )
        return (
            displayed_score <= self.threshold
            if self.inclusive
            else displayed_score < self.threshold
        )

    def describe(self) -> str:
        word = "at or above" if self.inclusive else "above"
        if self.direction == BELOW:
            word = "at or below" if self.inclusive else "below"
        return f"{self.film} {word} {self.threshold}%"


def decided(
    snapshot: Tomatometer, contract: Contract, max_new_reviews: int
) -> bool | None:
    """
    Whether the contract is already determined by arithmetic.

    Returns True for a certain YES, False for a certain NO, and None while
    the outcome is genuinely still open. This is the highest-confidence
    signal in the tool and the one that carries no model risk at all: it
    asks only whether every score still reachable settles the same way.
    """
    b = bounds(snapshot, max_new_reviews)
    yes_at_low = contract.resolves_yes(b.low)
    yes_at_high = contract.resolves_yes(b.high)
    if yes_at_low and yes_at_high:
        return True
    if not yes_at_low and not yes_at_high:
        return False
    return None


# ---------------------------------------------------------------------------
# The probabilistic half
# ---------------------------------------------------------------------------


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def beta_binomial_pmf(k: int, n: int, alpha: float, beta: float) -> float:
    """
    P(k successes in n draws) when the success rate is itself Beta(a, b).

    The over-dispersed cousin of the binomial: it is what you get when the
    rate is estimated rather than known, and it is wider than a binomial
    at the same mean. Using a plain binomial here would understate the
    chance of a film drifting across a threshold, which is precisely the
    error that makes a marginal contract look safe.
    """
    if not 0 <= k <= n:
        return 0.0
    if n == 0:
        return 1.0
    log_p = (
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        + _log_beta(k + alpha, n - k + beta)
        - _log_beta(alpha, beta)
    )
    return math.exp(log_p)


def _shift_posterior(
    alpha: float, beta: float, drift: float
) -> tuple[float, float]:
    """
    Move a Beta posterior by `drift` in log-odds, keeping its spread.

    Drift is expressed in log-odds rather than percentage points because
    the same "critics get harsher" effect is worth far less at 97% than at
    60%, and a flat point shift would either be negligible at one end or
    absurd at the other. Positive drift means later reviews are HARSHER,
    so it lowers the rate.

    Matching the mean and concentration is an approximation: the shifted
    distribution is not exactly Beta. It is used only on the path that is
    flagged and unstaked anyway, so it buys an honest headline number
    without pretending to a precision the drift estimate does not have.
    """
    if drift == 0.0:
        return alpha, beta
    concentration = alpha + beta
    mean = alpha / concentration
    mean = min(max(mean, 1e-9), 1 - 1e-9)
    logit = math.log(mean / (1.0 - mean))
    shifted = 1.0 / (1.0 + math.exp(-(logit - drift)))
    shifted = min(max(shifted, 1e-9), 1 - 1e-9)
    return shifted * concentration, (1.0 - shifted) * concentration


def _poisson_pmf(k: int, mean: float) -> float:
    if mean <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-mean + k * math.log(mean) - math.lgamma(k + 1))


def arrival_distribution(
    expected_new: float, max_new: int | None = None
) -> dict[int, float]:
    """
    How many more reviews get counted before settlement, as a distribution.

    Poisson, truncated at `max_new` and renormalised, because the ceiling
    used by `bounds` is a real constraint and the two halves of this
    module must not disagree about what is reachable. Reviews do arrive in
    bursts, which makes Poisson thinner-tailed than reality -- so a caller
    who is unsure should raise `expected_new` rather than trust the tail.
    """
    if expected_new < 0:
        raise ValueError("expected_new cannot be negative")
    cap = max_new if max_new is not None else int(expected_new + 10 * math.sqrt(expected_new + 1) + 10)
    weights = {k: _poisson_pmf(k, expected_new) for k in range(cap + 1)}
    mass = sum(weights.values())
    if mass <= 0:
        return {0: 1.0}
    return {k: w / mass for k, w in weights.items() if w > 0}


def probability_yes(
    snapshot: Tomatometer,
    contract: Contract,
    expected_new: float,
    max_new_reviews: int,
    prior_alpha: float = 2.0,
    prior_beta: float = 2.0,
    drift: float = 0.0,
) -> float:
    """
    The probability the contract settles YES.

    Exact, by summing over how many reviews arrive and over how many of
    those are Fresh. No simulation, so no sampling error to report and no
    seed to pin -- the number is the number.

    The default prior is Beta(2, 2): weak, symmetric, worth about four
    reviews. It matters only when the counted sample is tiny, which is
    exactly when the guards refuse to stake anyway.
    """
    alpha = prior_alpha + snapshot.fresh
    beta = prior_beta + snapshot.rotten
    alpha, beta = _shift_posterior(alpha, beta, drift)

    total = 0.0
    for new, weight in arrival_distribution(expected_new, max_new_reviews).items():
        if weight <= 0:
            continue
        final_total = snapshot.total + new
        if final_total == 0:
            continue
        # Sum only over the Fresh counts that settle YES. Done by testing
        # the displayed score rather than by inverting the threshold,
        # because the inversion has to reproduce RT's rounding and the
        # contract's inclusivity exactly, and one off-by-one there is a
        # whole contract mispriced.
        hit = 0.0
        for k in range(new + 1):
            score = int(math.floor(
                100.0 * (snapshot.fresh + k) / final_total + 0.5
            ))
            if contract.resolves_yes(score):
                hit += beta_binomial_pmf(k, new, alpha, beta)
        total += weight * hit
    return min(max(total, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Money
# ---------------------------------------------------------------------------


def fee(contracts: int, price: float, maker: bool = False) -> float:
    """
    Kalshi's trading fee in dollars, rounded up to the next cent.

    fee = ceil_to_cent( coefficient * contracts * price * (1 - price) )

    The rounding is per order and it bites small ones: one contract at 50c
    pays 2c where the formula says 1.75c, which is 4% of stake rather than
    3.5%. A model that ignores it will systematically overrate exactly the
    small positions these thin markets force you into.
    """
    if contracts <= 0:
        return 0.0
    if not 0.0 <= price <= 1.0:
        raise ValueError("price must be a probability in dollars, 0..1")
    coefficient = MAKER_COEFFICIENT if maker else TAKER_COEFFICIENT
    raw = coefficient * contracts * price * (1.0 - price)
    return math.ceil(raw * 100.0 - 1e-9) / 100.0


def breakeven_edge(price: float, maker: bool = False) -> float:
    """
    How far the true probability must sit above the price before a
    position is worth taking at all, ignoring the cent rounding.

    At 50c a taker needs 1.75 cents and a maker 0.44. That four-fold gap
    is the only structural edge on this exchange that does not require
    being right about anything, and it is why the same contract can be
    worth posting for and not worth crossing the spread for.
    """
    coefficient = MAKER_COEFFICIENT if maker else TAKER_COEFFICIENT
    return coefficient * price * (1.0 - price)


def ev_on_stake(
    probability: float, price: float, contracts: int = 1, maker: bool = False
) -> float:
    """
    Expected return per dollar committed, fees included.

    Committed, not notional: the fee is paid up front, so it belongs in
    the denominator as well as the numerator. Quoting EV against the bare
    contract price would flatter every position by the size of the fee.
    """
    if contracts <= 0:
        raise ValueError("contracts must be positive")
    cost = contracts * price + fee(contracts, price, maker)
    if cost <= 0:
        return 0.0
    expected = probability * contracts
    return (expected - cost) / cost


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------


@dataclass
class Assessment:
    """One contract, priced, with everything that could be wrong about it."""

    contract: Contract
    snapshot: Tomatometer
    price: float
    probability: float
    #: What the probability would be with no drift assumed. The honesty
    #: check: if the two disagree about the sign of the edge, the drift
    #: prior is carrying the bet and the bet does not get money.
    probability_no_drift: float
    bounds: ScoreBounds
    decided: bool | None
    ev: float
    ev_no_drift: float
    maker: bool = False
    flags: list[str] = field(default_factory=list)
    suspect: bool = False

    @property
    def edge(self) -> float:
        """True probability minus price, in cents of probability."""
        return self.probability - self.price

    @property
    def drift_is_load_bearing(self) -> bool:
        return (self.ev > 0) != (self.ev_no_drift > 0)

    def describe(self) -> str:
        return (
            f"{self.contract.describe()} @ {self.price * 100:.0f}c  "
            f"fair {self.probability * 100:.1f}c  EV {self.ev:+.1%}"
        )


def assess(
    snapshot: Tomatometer,
    contract: Contract,
    price: float,
    cfg,
    now: datetime | None = None,
    max_new_reviews: int | None = None,
    expected_new: float | None = None,
    maker: bool | None = None,
) -> Assessment:
    """
    Price one contract and say everything that could be wrong about it.

    `cfg` is a TomatoesConfig. The guards below withhold a stake rather
    than hide a number: an assessment that cannot be trusted is still
    printed, with the reason attached, because "no bets" with no
    explanation is the output that makes a tool get ignored.
    """
    now = now or datetime.now(timezone.utc)
    tc = cfg.tomatoes if hasattr(cfg, "tomatoes") else cfg
    ceiling = (
        max_new_reviews if max_new_reviews is not None
        else tc.default_max_new_reviews
    )
    arrivals = (
        expected_new if expected_new is not None
        else tc.default_expected_new_reviews
    )
    is_maker = tc.assume_maker if maker is None else maker

    b = bounds(snapshot, ceiling)
    verdict = decided(snapshot, contract, ceiling)

    common = dict(
        expected_new=arrivals, max_new_reviews=ceiling,
        prior_alpha=tc.prior_alpha, prior_beta=tc.prior_beta,
    )
    probability = probability_yes(snapshot, contract, drift=tc.drift, **common)
    no_drift = (
        probability if tc.drift == 0.0
        else probability_yes(snapshot, contract, drift=0.0, **common)
    )

    a = Assessment(
        contract=contract, snapshot=snapshot, price=price,
        probability=probability, probability_no_drift=no_drift,
        bounds=b, decided=verdict,
        ev=ev_on_stake(probability, price, contracts=1, maker=is_maker),
        ev_no_drift=ev_on_stake(no_drift, price, contracts=1, maker=is_maker),
        maker=is_maker,
    )
    return apply_guards(a, cfg, now=now)


def apply_guards(a: Assessment, cfg, now: datetime | None = None) -> Assessment:
    """Flag what cannot be trusted, and withhold a stake where it matters."""
    now = now or datetime.now(timezone.utc)
    tc = cfg.tomatoes if hasattr(cfg, "tomatoes") else cfg
    flags: list[str] = []
    suspect = False

    if not a.contract.settlement_verified:
        # The clerical failure, and on these markets the likeliest one.
        # Which Tomatometer, whether the threshold is inclusive, and what
        # time it settles are all read off the contract by a person.
        flags.append("settlement_terms_unverified")
        suspect = True

    if a.contract.scope != a.snapshot.scope:
        # All Critics against Top Critics on the same film differs by
        # several points. Never compare the two.
        flags.append(
            f"scope_mismatch(contract {a.contract.scope} / "
            f"snapshot {a.snapshot.scope})"
        )
        suspect = True

    age = a.snapshot.age_hours(now)
    if age > tc.max_snapshot_age_hours:
        flags.append(f"snapshot_{age:.0f}h_old")
        suspect = True

    if a.decided is not None:
        # The good case, and worth naming: no model is involved, so none
        # of the modelling guards below can apply to it.
        flags.append(
            f"decided_by_arithmetic(score must land in {a.bounds.low}-"
            f"{a.bounds.high}%)"
        )
    else:
        if a.snapshot.total < tc.min_reviews:
            flags.append(f"only_{a.snapshot.total}_reviews_counted")
            suspect = True
        if a.drift_is_load_bearing:
            # The honesty check, and the exact analogue of
            # `only_+ev_because_of_assumed_correlation` on the parlay side.
            flags.append("only_+ev_because_of_assumed_drift")
            suspect = True
        if tc.drift != 0.0:
            flags.append(f"drift_prior_{tc.drift:+.2f}_applied_not_measured")

    if a.ev > tc.max_plausible_ev and a.decided is None:
        # A large edge on an undecided contract means a stale snapshot or
        # the wrong settlement terms far more often than it means an edge.
        flags.append(f"ev_{a.ev:.0%}_implausible")
        suspect = True

    if a.maker:
        # It is only a maker fee if the order rests and gets filled. An EV
        # computed at the maker rate on an order you then cross with is
        # wrong by four times the fee.
        flags.append("priced_as_maker(only true if the order rests)")

    a.flags = flags
    a.suspect = suspect
    return a


def position_size(a: Assessment, bankroll: float, cfg) -> float:
    """
    What to stake, or nothing.

    Fractional Kelly on the fee-adjusted odds, capped. A suspect
    assessment gets zero -- not a reduced size -- because the flags above
    are reasons to think the number is wrong, and a smaller stake on a
    wrong number is still a bet on a wrong number.
    """
    tc = cfg.tomatoes if hasattr(cfg, "tomatoes") else cfg
    if a.suspect or a.ev <= tc.min_ev or bankroll <= 0:
        return 0.0
    cost = a.price + fee(1, a.price, a.maker)
    if cost <= 0 or cost >= 1.0:
        # Nothing to win: the fee has eaten the whole spread to a dollar.
        return 0.0
    # Kelly on a binary paying (1 - cost) per unit committed.
    b = (1.0 - cost) / cost
    edge = a.probability * b - (1.0 - a.probability)
    if edge <= 0:
        return 0.0
    fraction = (edge / b) * tc.kelly_multiplier
    return round(
        min(fraction, tc.max_position_fraction) * bankroll, 2
    )
