"""
Pricing every rung of a game, not just the ones a book quotes back.

The idea
--------
Kalshi lists many strikes on one game -- win by more than 3.5, by more
than 6.5, by more than 9.5, totals at a row of thresholds. Those rungs
are not independent markets. Together they describe one distribution of
one outcome, and they have to be mutually coherent.

Most comparison tools only price a rung where a sportsbook quotes the
same number back, and alternate lines are patchy in every odds feed. So
the outer rungs -- "win by 10+", the ones that appeal to fans and that
almost nobody is checking -- go unexamined. That is where recreational
money distorts prices and where nothing is pushing back.

So: fit a margin distribution to Pinnacle's MAIN line, which is the
sharpest number on the board, and then price every rung from it.

Why the fit is a fit and not an assumption
------------------------------------------
Given only a spread you would have to assume a standard deviation, and
the answer would inherit whatever you assumed. But the spread and the
moneyline together OVER-DETERMINE a two-parameter distribution:

    the spread says where the middle is       ->  mu
    the moneyline says P(margin > 0)          ->  sigma

so sigma is solved for, not guessed:

    sigma = mu / Phi^-1(P(favourite wins))

That also gives a free consistency check. Every sport has a known range
for game-to-game margin variance, and a fitted sigma outside it means the
inputs disagree -- a stale moneyline, a mismatched event, a spread read
off the wrong side. The fit failing loudly is more valuable than the fit
succeeding quietly.

Key numbers
-----------
NFL margins are not smooth. They pile up on 3 and 7, and to a lesser
extent 10 and 14, because of how scoring works. A plain normal underprices
a rung sitting exactly on a key number and overprices its neighbours,
which is precisely the error that makes an outer rung look mispriced when
it is not. The adjustment is data -- a table of empirical margin
frequencies -- and like every other table in this repository it ships
unverified and is labelled a prior until someone checks it.

The part that needs no model at all
-----------------------------------
A ladder must be monotone: P(win by more than 3.5) >= P(win by more than
6.5), always, whatever the distribution. If the exchange's own prices
violate that, there is an arbitrage that does not depend on being right
about the game. That check runs before any fitting, and it is the one
result here that carries no model risk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

PRIORS_PATH = Path(__file__).parent / "data" / "margin_priors.yaml"

#: What a rung asks about.
MARGIN = "margin"
TOTAL = "total"


class LadderError(RuntimeError):
    """The inputs cannot support a distribution."""


# ---------------------------------------------------------------------------
# Normal helpers. Kept local rather than imported from copula.py so that
# this module stands alone -- it is about one game, not about joint
# probabilities across legs.
# ---------------------------------------------------------------------------


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF, by bisection on the CDF.

    Slower than a rational approximation and exact to machine precision
    for this purpose, which matters because sigma is solved through it:
    an error here propagates into every rung on the ladder.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("probability must be strictly between 0 and 1")
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------


@dataclass
class SportPrior:
    """
    What is known about a sport's margins before looking at this game.

    `sigma_low`/`sigma_high` are a plausibility band, not an estimate:
    they exist to catch inputs that disagree, not to constrain the fit.
    """

    key: str
    sigma_typical: float
    sigma_low: float
    sigma_high: float
    total_sigma: float = 0.0
    #: Extra probability mass sitting exactly on these margins, as a
    #: fraction. Priors, not measurements -- see the YAML.
    key_numbers: dict = field(default_factory=dict)
    verified: bool = False
    note: str = ""

    def plausible(self, sigma: float) -> bool:
        return self.sigma_low <= sigma <= self.sigma_high


@dataclass
class PriorSet:
    sports: dict
    path: Path | None = None

    def get(self, sport: str) -> SportPrior | None:
        return self.sports.get(sport)

    @property
    def unverified(self) -> list[str]:
        return sorted(k for k, v in self.sports.items() if not v.verified)

    @classmethod
    def load(cls, path=None) -> "PriorSet":
        import yaml

        p = Path(path) if path else PRIORS_PATH
        raw = yaml.safe_load(p.read_text()) or {}
        sports = {}
        for key, spec in (raw.get("sports") or {}).items():
            sports[key] = SportPrior(
                key=key,
                sigma_typical=float(spec["sigma_typical"]),
                sigma_low=float(spec["sigma_low"]),
                sigma_high=float(spec["sigma_high"]),
                total_sigma=float(spec.get("total_sigma") or 0.0),
                key_numbers={
                    float(k): float(v)
                    for k, v in (spec.get("key_numbers") or {}).items()
                },
                verified=bool(spec.get("verified", False)),
                note=(spec.get("note") or "").strip(),
            )
        return cls(sports=sports, path=p)


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


@dataclass
class MarginModel:
    """
    A distribution of the favourite's margin of victory.

    `sigma_source` says how sigma was arrived at, because a sigma solved
    from a moneyline and a sigma taken from a sport-wide prior deserve
    very different confidence, and a ladder priced from the second should
    not be presented as though it were the first.
    """

    mu: float
    sigma: float
    sport: str = ""
    sigma_source: str = "fitted"
    prior: SportPrior | None = None
    flags: list[str] = field(default_factory=list)
    _pmf: dict | None = field(default=None, repr=False, compare=False)

    @property
    def implied_win_prob(self) -> float:
        """
        P(margin > 0), continuous.

        Deliberately the smooth figure rather than the discrete one: this
        is what sigma was SOLVED against, so reading it off the same
        curve is what makes the fit a closed loop that either holds
        exactly or reveals an algebra error. `prob_margin_over(0)` is the
        discrete answer and will differ slightly, which is the
        discretisation and the key numbers doing their job.
        """
        return 1.0 - norm_cdf((0.0 - self.mu) / self.sigma)

    def pmf(self) -> dict:
        """
        Probability of each whole-number margin, key numbers included.

        Discrete on purpose. A margin is an integer, so the continuous
        picture is a convenience rather than the truth, and the first
        version of this file paid for that: it took the key-number mass
        off the integer rungs and left the half-point rungs alone, which
        made P(margin > 3) come out BELOW P(margin > 3.5). For a sport
        where margins are whole numbers those two are the same event, so
        the ladder was not merely mispriced, it was incoherent -- the
        exact failure this module exists to detect in other people's
        prices.

        Building the mass function first and summing it makes the
        survival function monotone by construction, and puts the lumps
        where they actually are: ON a margin, not on a threshold.
        """
        if self._pmf is not None:
            return self._pmf
        if self.sigma <= 0:
            raise LadderError("sigma must be positive")
        lo = int(math.floor(self.mu - 8 * self.sigma))
        hi = int(math.ceil(self.mu + 8 * self.sigma))
        bumps = (self.prior.key_numbers if self.prior else {}) or {}
        weights = {}
        for m in range(lo, hi + 1):
            # The continuous mass in the unit interval around m.
            w = (norm_cdf((m + 0.5 - self.mu) / self.sigma)
                 - norm_cdf((m - 0.5 - self.mu) / self.sigma))
            # Key numbers attract mass, in either direction: a margin of
            # 3 is common whether the favourite or the underdog wins it.
            bump = bumps.get(float(abs(m)), 0.0)
            weights[m] = w * (1.0 + bump)
        total = sum(weights.values()) or 1.0
        self._pmf = {m: w / total for m, w in weights.items()}
        return self._pmf

    def prob_margin_over(self, threshold: float) -> float:
        """
        P(margin > threshold), from the favourite's perspective.

        A negative threshold asks about the underdog covering, which is
        the same distribution read further left -- no special case
        needed, and that is the point of modelling the margin rather than
        each rung.
        """
        return sum(w for m, w in self.pmf().items() if m > threshold)

    def prob_margin_between(self, low: float, high: float) -> float:
        return max(0.0, self.prob_margin_over(low) - self.prob_margin_over(high))


def fit(
    spread: float,
    win_probability: float | None,
    sport: str = "",
    priors: PriorSet | None = None,
) -> MarginModel:
    """
    Fit a margin distribution to a main line.

    `spread` is the favourite's spread as a positive number of points
    (a -6.5 favourite is 6.5 here). `win_probability` is the favourite's
    DE-VIGGED moneyline probability -- pass the raw one and sigma comes
    out too small, making every outer rung look too likely.
    """
    prior = priors.get(sport) if priors else None
    flags: list[str] = []
    mu = float(spread)

    if win_probability is None:
        if prior is None:
            raise LadderError(
                f"no moneyline and no prior for {sport!r}: there is nothing "
                "to pin the spread of the distribution to, and assuming one "
                "would make every rung an echo of that assumption"
            )
        sigma = prior.sigma_typical
        source = "sport prior"
        flags.append("sigma_from_prior_not_fitted_to_this_game")
    elif not 0.0 < win_probability < 1.0:
        raise LadderError("win probability must be strictly between 0 and 1")
    elif abs(mu) < 1e-9:
        # A pick'em says nothing about the spread of outcomes: every
        # sigma reproduces a 50% moneyline.
        if prior is None:
            raise LadderError(
                "a pick'em game with no sport prior cannot be fitted: every "
                "sigma reproduces a 50% moneyline"
            )
        sigma = prior.sigma_typical
        source = "sport prior"
        flags.append("pick_em_so_sigma_is_not_identified")
    else:
        z = norm_ppf(win_probability)
        if abs(z) < 1e-9:
            raise LadderError("a 50% moneyline cannot identify sigma")
        sigma = mu / z
        source = "fitted"
        if sigma <= 0:
            # The favourite by the spread is the underdog by the
            # moneyline. That is not a distribution, it is mismatched
            # inputs -- most often the spread read off the wrong side.
            raise LadderError(
                f"spread {spread:+g} and win probability {win_probability:.1%} "
                "disagree about who is favoured. Check that both are for the "
                "same side of the same game."
            )

    if prior and not prior.plausible(sigma):
        # The free consistency check. Every sport has a known range, and
        # a fitted sigma outside it means the inputs disagree rather than
        # that this game is unusual.
        flags.append(
            f"sigma_{sigma:.1f}_outside_{sport}_range_"
            f"{prior.sigma_low:g}_to_{prior.sigma_high:g}"
        )
    if prior and not prior.verified:
        flags.append("margin_priors_unverified")

    return MarginModel(mu=mu, sigma=sigma, sport=sport, sigma_source=source,
                       prior=prior, flags=flags)


# ---------------------------------------------------------------------------
# Coherence: the part that needs no model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rung:
    """One strike on the ladder, with what it costs to buy YES."""

    threshold: float
    yes_ask: float | None = None
    yes_bid: float | None = None
    ticker: str = ""
    kind: str = MARGIN

    def describe(self) -> str:
        word = "margin over" if self.kind == MARGIN else "total over"
        return f"{word} {self.threshold:g}"


@dataclass(frozen=True)
class Incoherence:
    """Two rungs whose prices cannot both be right."""

    lower: Rung
    higher: Rung
    kind: str
    detail: str

    def describe(self) -> str:
        return (f"{self.lower.describe()} vs {self.higher.describe()}: "
                f"{self.detail}")


def coherence_violations(rungs) -> list[Incoherence]:
    """
    Rungs whose prices contradict each other, whatever the true
    distribution is.

    A harder threshold can never be more likely than an easier one, so on
    a margin ladder the prices must fall as the threshold rises. Where
    they do not, there is an arbitrage that does not require being right
    about the game: buy the cheaper, harder rung and sell the dearer,
    easier one.

    Compares ASK against BID rather than mid against mid, because a
    crossing of mids is a rounding artefact while a crossing of the
    tradeable sides is money. Fees are not netted here -- the caller
    knows its own maker or taker status -- so a violation is a candidate,
    not a filled trade.
    """
    ordered = sorted(
        [r for r in rungs if r.threshold is not None],
        key=lambda r: r.threshold,
    )
    problems = []
    for i, lower in enumerate(ordered):
        for higher in ordered[i + 1:]:
            if higher.threshold <= lower.threshold:
                continue
            if lower.yes_ask is None or higher.yes_bid is None:
                continue
            # The trade: buy YES on the EASIER rung, buy NO on the
            # HARDER one. The harder outcome implies the easier, so the
            # only losing state -- harder without easier -- cannot occur:
            #
            #   both happen        YES easier pays, NO harder does not  = 1
            #   neither happens    YES easier does not, NO harder pays  = 1
            #   easier only        both pay                             = 2
            #
            # Minimum return is 1 per pair, so it is free money whenever
            # the pair costs less than 1:
            #
            #   ask(easier) + (1 - bid(harder)) < 1   <=>   ask < bid
            if lower.yes_ask < higher.yes_bid:
                edge = higher.yes_bid - lower.yes_ask
                problems.append(Incoherence(
                    lower=lower, higher=higher, kind="monotonicity",
                    detail=(
                        f"the easier rung asks {lower.yes_ask * 100:.0f}c "
                        f"while the harder one bids "
                        f"{higher.yes_bid * 100:.0f}c -- a harder threshold "
                        f"cannot be more likely, so buying the easier and "
                        f"selling the harder locks {edge * 100:.0f}c before "
                        "fees"
                    ),
                ))
    return problems


def price_ladder(model: MarginModel, rungs) -> list[tuple]:
    """
    (rung, fair probability, edge against its ask) for every rung.

    Edge is in probability, not in EV: fees and fill depth belong to the
    caller, which knows whether it is crossing the spread.
    """
    priced = []
    for rung in rungs:
        if rung.kind != MARGIN:
            # Totals need their own distribution -- a different mean and
            # a different sigma -- so pricing them off the margin model
            # would be silently wrong rather than merely unsupported.
            priced.append((rung, None, None))
            continue
        fair = model.prob_margin_over(rung.threshold)
        edge = None if rung.yes_ask is None else fair - rung.yes_ask
        priced.append((rung, fair, edge))
    return priced
