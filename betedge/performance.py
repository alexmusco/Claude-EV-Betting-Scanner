"""
Comparing the two strategies: which one is actually making money?

The two are tracked in separate tables and always have been -- single
bets in `bets`, multi-leg entries in `parlay_bets` -- so the raw material
for the question exists. What did not exist is a fair way to read it,
and reading it unfairly is very easy.

Why a naive ROI comparison misleads
-----------------------------------
Return on turnover is the obvious metric and the worst one to trust
early. A single bet at even money returns +1 or -1 per unit staked, so
the standard deviation of one bet's return is about 1.0 and the error on
your measured ROI after n bets is about 1/sqrt(n). Forty bets gives you
plus or minus sixteen points. Your actual edge is two or three.

Multi-leg entries are far worse, because the whole point of them is a
lumpy payoff. A 20x ticket landing 5% of the time has a per-bet return
standard deviation of about 4.4 -- so after a hundred settled entries the
error on its ROI is still around forty points. You could run a genuinely
+5% strategy and a genuinely -5% one side by side for a season and the
measured ROIs would routinely come out the wrong way round.

So this module does three things rather than printing two ROI numbers:

1. Puts both strategies on identical metrics, so they can be read
   together at all.
2. Bootstraps a confidence interval for each ROI, resampling the actual
   settled bets. That respects the real shape of the payoffs rather than
   assuming a normal one -- which matters most for exactly the lumpy
   parlay case where a formula would be worst.
3. States how many settled bets each strategy would need before ROI could
   resolve a difference of the size being claimed, computed from that
   strategy's own observed spread. Usually the number is large enough to
   settle the argument on its own.

What to read instead, meanwhile
-------------------------------
Closing-line value, and the ratio of realised profit to the profit the
model said to expect. CLV converges in dozens of bets where profit needs
thousands, and the realisation ratio answers the question underneath the
question: not "which won more" but "whose claimed edge is real".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

#: Fixed so a comparison does not reshuffle between two runs on the same
#: data. The interval is an estimate with its own noise; making that noise
#: reproducible is the least it can do.
DEFAULT_SEED = 20260915
DEFAULT_DRAWS = 20_000
DEFAULT_CONFIDENCE = 0.90

#: Below this many settled bets, an ROI is not reported as meaningful at
#: all. Not a statistical threshold -- a floor below which the bootstrap
#: itself has too few distinct outcomes to resample honestly.
MIN_SETTLED_FOR_ROI = 10


@dataclass
class StrategyResult:
    """One strategy's settled record, on the same metrics as the other."""

    name: str
    settled: int = 0
    pending: int = 0
    staked: float = 0.0
    pnl: float = 0.0
    modelled_pnl: float | None = None
    #: How many settled bets carried an EV at the time they were placed,
    #: and what those bets alone actually returned. A bet logged by hand
    #: has no modelled number, so pooling it into the numerator while it
    #: contributes nothing to the denominator inflates the ratio below --
    #: the more hand-logged bets there are, the further off it reads.
    modelled_settled: int = 0
    modelled_realised_pnl: float = 0.0
    clv_values: list[float] = field(default_factory=list)
    #: Per-bet return on stake, the raw material for every interval below.
    returns: list[float] = field(default_factory=list)
    stakes: list[float] = field(default_factory=list)
    pnls: list[float] = field(default_factory=list)

    @property
    def roi(self) -> float | None:
        return (self.pnl / self.staked) if self.staked else None

    @property
    def realisation(self) -> float | None:
        """
        Realised profit over the profit the model said to expect.

        The question underneath the question. Not "which strategy won
        more" -- that is mostly luck at any sample you will have -- but
        "whose claimed edge showed up". Tends to 1.0 if the model is
        right, and its distance from 1.0 is still enormously noisy early,
        which is why it is reported next to a sample size rather than
        alone.
        """
        if not self.modelled_pnl:
            return None
        # Matched numerator: only the bets that HAD a modelled edge, so
        # the ratio compares like with like even when the ledger mixes
        # scanner picks with bets typed in after the fact.
        return self.modelled_realised_pnl / self.modelled_pnl

    @property
    def avg_clv(self) -> float | None:
        return (sum(self.clv_values) / len(self.clv_values)) if self.clv_values else None

    @property
    def clv_beat_rate(self) -> float | None:
        if not self.clv_values:
            return None
        return sum(1 for v in self.clv_values if v > 0) / len(self.clv_values)

    @property
    def return_sd(self) -> float | None:
        """Spread of per-bet returns: what makes ROI hard to measure."""
        if len(self.returns) < 2:
            return None
        return float(np.std(np.asarray(self.returns), ddof=1))

    def roi_interval(
        self,
        confidence: float = DEFAULT_CONFIDENCE,
        draws: int = DEFAULT_DRAWS,
        seed: int = DEFAULT_SEED,
    ) -> tuple[float, float] | None:
        """
        Bootstrap interval for ROI, resampling the settled bets themselves.

        Resampling rather than assuming a shape, because the shape is the
        problem: a handful of large parlay wins among many total losses is
        nothing like normal, and a textbook interval would be narrowest
        exactly where it is least trustworthy.
        """
        if self.settled < MIN_SETTLED_FOR_ROI or not self.staked:
            return None
        stakes = np.asarray(self.stakes, dtype=float)
        pnls = np.asarray(self.pnls, dtype=float)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, stakes.size, size=(draws, stakes.size))
        staked = stakes[idx].sum(axis=1)
        won = pnls[idx].sum(axis=1)
        rois = np.divide(won, staked, out=np.zeros_like(won), where=staked > 0)
        tail = (1.0 - confidence) / 2.0
        return (
            float(np.quantile(rois, tail)),
            float(np.quantile(rois, 1.0 - tail)),
        )

    def bets_needed(self, resolution: float = 0.05, z: float = 1.96) -> int | None:
        """
        Settled bets needed before ROI could resolve a difference this big.

        n = (z * sd / resolution)^2, from this strategy's own observed
        spread of returns. Printed because the number is usually large
        enough to end the argument by itself -- and because "you cannot
        tell yet" is a far more useful answer than a decimal that looks
        like one.
        """
        sd = self.return_sd
        if not sd or resolution <= 0:
            return None
        return int(math.ceil((z * sd / resolution) ** 2))


def summarise(
    name: str,
    settled_rows: Sequence,
    pending_rows: Sequence = (),
    clv_values: Sequence[float] = (),
) -> StrategyResult:
    """
    Build one strategy's record from its settled bets.

    Rows need `stake`, `pnl` and optionally `ev_at_bet`; anything with a
    zero stake is skipped rather than dividing by it. Works on sqlite3
    rows or plain dicts, so the arithmetic is testable without a database.
    """
    result = StrategyResult(name=name)
    modelled = 0.0
    have_modelled = False

    for row in settled_rows:
        stake = float(_get(row, "stake") or 0.0)
        if stake <= 0:
            continue
        pnl = float(_get(row, "pnl") or 0.0)
        result.settled += 1
        result.staked += stake
        result.pnl += pnl
        result.stakes.append(stake)
        result.pnls.append(pnl)
        result.returns.append(pnl / stake)
        ev = _get(row, "ev_at_bet")
        if ev is not None:
            modelled += float(ev) * stake
            result.modelled_settled += 1
            result.modelled_realised_pnl += pnl
            have_modelled = True

    result.pending = len(list(pending_rows))
    result.modelled_pnl = modelled if have_modelled else None
    result.clv_values = [float(v) for v in clv_values if v is not None]
    return result


def _get(row, key):
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return getattr(row, key, None)


@dataclass
class Comparison:
    """Two strategies side by side, and what can honestly be said."""

    strategies: list[StrategyResult]
    confidence: float = DEFAULT_CONFIDENCE

    @property
    def ranked(self) -> list[StrategyResult]:
        return sorted(
            [s for s in self.strategies if s.roi is not None],
            key=lambda s: s.roi,
            reverse=True,
        )

    @property
    def intervals_overlap(self) -> bool | None:
        """
        Whether the ROI intervals overlap.

        Overlapping means the samples are consistent with the two
        strategies performing identically -- which after any realistic
        number of bets is what they will say, and is the honest answer
        rather than a disappointing one.
        """
        bounds = [s.roi_interval(self.confidence) for s in self.strategies]
        if any(b is None for b in bounds) or len(bounds) < 2:
            return None
        lo = max(b[0] for b in bounds)
        hi = min(b[1] for b in bounds)
        return lo <= hi

    def verdict(self) -> str:
        """One paragraph on what the numbers do and do not support."""
        scored = [s for s in self.strategies if s.settled]
        if len(scored) < 2:
            missing = [s.name for s in self.strategies if not s.settled]
            return (
                "Nothing settled yet for: " + ", ".join(missing) + ". "
                "A comparison needs both sides."
            )

        thin = [s for s in scored if s.settled < MIN_SETTLED_FOR_ROI]
        if thin:
            return (
                "Too few settled bets to say anything about profit ("
                + ", ".join(f"{s.name}: {s.settled}" for s in thin)
                + "). Read the closing-line value instead -- it converges in "
                "dozens of bets where profit needs thousands."
            )

        overlap = self.intervals_overlap
        best = self.ranked[0]
        needed = [
            (s.name, s.bets_needed())
            for s in scored
            if s.bets_needed() is not None
        ]
        need_text = ", ".join(
            f"{name} about {n:,}" for name, n in needed
        )

        if overlap:
            return (
                f"{best.name} is ahead on ROI, but the "
                f"{self.confidence:.0%} intervals overlap: this sample is "
                "consistent with the two performing identically, so the gap "
                "is not evidence of anything yet. To resolve a 5-point "
                f"difference on profit alone you would need {need_text} "
                "settled bets. Closing-line value is the faster read."
            )
        return (
            f"{best.name} is ahead and the {self.confidence:.0%} intervals do "
            "not overlap, which is a real signal rather than noise -- though "
            "profit still lags closing-line value as evidence, and one "
            "strategy can be ahead for a stretch while the other has the "
            f"better edge. For reference, resolving a 5-point difference on "
            f"profit alone takes about {need_text} settled bets."
        )
