"""
Gaussian copula: the joint probability that every leg of a ticket lands.

Why not just multiply the marginals
-----------------------------------
A four-leg ticket whose legs each hit 55% of the time does NOT hit
0.55^4 = 9.2% of the time unless the legs are independent. Legs chosen
from the same game are never independent: a quarterback throwing for 320
yards and his top receiver going over 75 are the same event described
twice. Multiplying marginals understates that ticket, and it overstates a
ticket built from legs that compete with each other -- two running backs
splitting one team's carries.

Independence is the assumption that makes fixed-multiplier pick'em
products exploitable, because those products price as if it held. So the
whole model rests on estimating the joint properly, and a "correlation
bonus" bolted onto a product of marginals is not an estimate of anything.

The construction
----------------
Give each leg a latent standard normal Z_i oriented so that LARGE means
the leg wins. Choose the threshold t_i so the leg's win probability comes
out right:

    P(Z_i > t_i) = p_i   =>   t_i = inverse_normal_cdf(1 - p_i)

Couple the legs by drawing (Z_1..Z_n) from a multivariate normal with
correlation matrix R. Then

    P(every leg wins) = P(Z_1 > t_1, ..., Z_n > t_n)

which is the Gaussian orthant probability. It has no closed form beyond
n = 2, so it is estimated by Monte Carlo: draw, count, divide. That is
crude but it is honest, it converges at a known rate, and -- crucially --
the same draws give the FULL distribution of how many legs landed, which
is what flex and insured pick'em entries actually pay on.

The latent-normal dependence is an assumption, not a fact. It is the
standard one, it is the same assumption that underlies every correlation
number anyone quotes for this, and it reduces to the right answer at both
extremes (independence, and perfect dependence). Where it is weakest is
the tails: real sporting outcomes have fatter joint tails than a Gaussian
copula gives them, so an all-hit probability from this machinery is, if
anything, slightly conservative on strongly correlated legs.

Pushes
------
A leg that lands exactly on the line usually voids rather than losing,
and on pick'em sites that shrinks the entry. So each leg gets a second
threshold below the first: above t_hit it wins, between t_void and t_hit
it pushes, below t_void it loses. That ordering is correct in both
directions because the latent is oriented to the leg's own success -- for
an Over the push sits just under the winning region, and for an Under the
latent is the negated stat, so it does too.

Orientation and the sign of R
-----------------------------
Because every latent points at its own leg winning, the correlation
between two legs must be signed for the SIDES taken, not just for the
underlying stats. Two teammates' counting stats may correlate at +0.35,
but taking one Over and the other Under makes the legs' correlation
-0.35. correlation.py applies that sign; this module takes R as given.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: Default RNG seed. Fixed so two runs over the same board produce the
#: same numbers -- a ranking that reshuffles on re-run is unusable for
#: deciding anything, and the Monte Carlo noise is reported separately
#: rather than hidden by pretending it is not there.
DEFAULT_SEED = 20260915

#: Draws for a final, reported estimate. At 200k the standard error on a
#: probability near 0.05 is about 0.05 percentage points, and on the EV of
#: a 20x ticket about 0.9 percentage points -- small enough to rank on,
#: large enough that it is worth printing, which is why it is printed.
DEFAULT_DRAWS = 200_000

#: Draws while searching. The search compares thousands of candidates, and
#: because they share one pool of random numbers (common random numbers)
#: the comparison is far more accurate than either estimate alone.
DEFAULT_SEARCH_DRAWS = 20_000


# --------------------------------------------------------------------------
# Normal distribution helpers
#
# Written out rather than pulled from scipy: the project's only heavy
# dependency is numpy, and these two functions are all that is needed.
# --------------------------------------------------------------------------


def norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
      1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
      6.680131188771972e01, -1.328068155288572e01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
      -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
      3.754408661907416e00)
_P_LOW = 0.02425


def norm_ppf(p: float) -> float:
    """
    Inverse standard normal CDF: the z with P(Z <= z) = p.

    Acklam's rational approximation (about 1e-9 relative) followed by one
    Halley refinement against erfc, which takes it to machine precision.
    Accuracy matters here because a threshold error of 1e-4 in probability
    is comparable to the edges being measured.
    """
    if not 0.0 < p < 1.0:
        if p <= 0.0:
            return -math.inf
        if p >= 1.0:
            return math.inf
        raise ValueError(f"p must be in (0,1), got {p!r}")

    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
            ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    elif p <= 1.0 - _P_LOW:
        q = p - 0.5
        r = q * q
        x = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / \
            (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
            ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)

    # Halley step. e is the residual in probability, u the scaled residual.
    e = norm_cdf(x) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)


def threshold_for(prob: float) -> float:
    """
    The latent cutoff a leg of probability `prob` must exceed.

    Clamped away from 0 and 1: a leg quoted at exactly 0 or 1 is a data
    error, and an infinite threshold would poison the whole matrix rather
    than just that leg.
    """
    p = min(max(prob, 1e-9), 1.0 - 1e-9)
    return norm_ppf(1.0 - p)


def bivariate_normal_upper(t1: float, t2: float, rho: float, nodes: int = 2048) -> float:
    """
    P(Z1 > t1, Z2 > t2) for a standard bivariate normal with correlation
    rho, by Gauss-Legendre quadrature on Sheppard's identity

        d/drho P(Z1 > t1, Z2 > t2) = phi_2(t1, t2; rho)

    integrated from rho = 0, where the answer is the product of the
    marginals. Exact to ~1e-12 for |rho| < 1.

    This is here as the analytic reference for n = 2 -- the one case where
    the orthant probability can be computed without simulating -- so the
    Monte Carlo can be checked against something that does not share a
    line of code with it.
    """
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must be strictly between -1 and 1")
    base = (1.0 - norm_cdf(t1)) * (1.0 - norm_cdf(t2))
    if rho == 0.0:
        return base

    x, w = np.polynomial.legendre.leggauss(nodes)
    # Map the quadrature nodes from [-1, 1] onto [0, rho].
    r = 0.5 * rho * (x + 1.0)
    half = 0.5 * rho
    one_minus = 1.0 - r * r
    density = np.exp(
        -(t1 * t1 - 2.0 * r * t1 * t2 + t2 * t2) / (2.0 * one_minus)
    ) / (2.0 * np.pi * np.sqrt(one_minus))
    return float(base + half * np.dot(w, density))


# --------------------------------------------------------------------------
# Correlation matrix conditioning
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PsdResult:
    """Outcome of forcing a correlation matrix to be usable."""

    matrix: np.ndarray
    projected: bool
    min_eigenvalue: float          #: smallest eigenvalue BEFORE any repair
    max_adjustment: float          #: largest change to any entry

    @property
    def note(self) -> str:
        if not self.projected:
            return ""
        return (
            f"correlation matrix was not positive semi-definite "
            f"(min eigenvalue {self.min_eigenvalue:+.4f}); projected to the "
            f"nearest PSD matrix, moving entries by up to "
            f"{self.max_adjustment:.3f}"
        )


def is_psd(matrix: np.ndarray, tol: float = -1e-10) -> bool:
    """Whether every eigenvalue is non-negative to within `tol`."""
    if matrix.shape[0] == 0:
        return True
    return bool(np.min(np.linalg.eigvalsh(_symmetrise(matrix))) >= tol)


def _symmetrise(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=float)
    return 0.5 * (m + m.T)


def nearest_psd(matrix, min_eigenvalue: float = 1e-8) -> PsdResult:
    """
    Project a would-be correlation matrix onto the nearest PSD one.

    Pairwise correlations assembled from independent sources need not form
    a consistent matrix. Say A and B correlate at +0.8, B and C at +0.8,
    and a third estimate puts A and C at -0.5: no three random variables
    can do all three at once, and the matrix has a negative eigenvalue to
    prove it. Sampling from it is not merely inaccurate, it is impossible
    -- the Cholesky factor does not exist.

    The repair is eigenvalue clipping: decompose, raise every eigenvalue
    to at least `min_eigenvalue`, rebuild, then rescale so the diagonal is
    1 again (clipping perturbs it). This is Higham's first step rather
    than his full alternating projection; for the 2-6 leg matrices here
    the difference is immaterial and one eigendecomposition is honest
    about what it did.

    Callers are expected to surface `.note` -- a projected matrix means
    the correlation inputs disagreed with each other, which is worth
    seeing rather than silently smoothing over.
    """
    m = _symmetrise(matrix)
    n = m.shape[0]
    if n == 0:
        return PsdResult(m, False, 0.0, 0.0)
    np.fill_diagonal(m, 1.0)

    eigenvalues, vectors = np.linalg.eigh(m)
    smallest = float(eigenvalues[0])
    if smallest >= min_eigenvalue:
        return PsdResult(m, False, smallest, 0.0)

    clipped = np.clip(eigenvalues, min_eigenvalue, None)
    repaired = (vectors * clipped) @ vectors.T
    # Clipping moves the diagonal off 1; rescale back to a correlation matrix.
    scale = np.sqrt(np.clip(np.diag(repaired), 1e-12, None))
    repaired = repaired / np.outer(scale, scale)
    repaired = _symmetrise(repaired)
    np.fill_diagonal(repaired, 1.0)
    return PsdResult(
        matrix=repaired,
        projected=True,
        min_eigenvalue=smallest,
        max_adjustment=float(np.max(np.abs(repaired - m))),
    )


def cholesky_factor(matrix: np.ndarray) -> np.ndarray:
    """
    Lower-triangular L with L @ L.T == matrix, adding jitter if needed.

    `nearest_psd` leaves a matrix that is PSD but can still be singular to
    within floating point, which numpy's Cholesky rejects outright. A
    perfectly correlated pair is a legitimate input -- two names for the
    same bet -- so escalating jitter is the pragmatic answer.
    """
    m = _symmetrise(matrix)
    for jitter in (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4):
        try:
            return np.linalg.cholesky(m + jitter * np.eye(m.shape[0]))
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError(
        "correlation matrix is not factorisable even with jitter; "
        "run it through nearest_psd first"
    )


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------


def standard_normals(draws: int, n: int, seed: int | None = DEFAULT_SEED) -> np.ndarray:
    """
    A pool of independent standard normals, shape (draws, n).

    Generated once per run and reused for every candidate ticket. That is
    "common random numbers": it costs nothing, and it means the difference
    between two candidates' estimated EVs is measured far more precisely
    than either estimate alone, which is exactly what a search needs.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((draws, n))


@dataclass
class Simulation:
    """
    The full outcome distribution of one ticket.

    `grid[v][k]` is the probability that exactly v legs voided and exactly
    k of the rest hit. Everything a payout structure needs -- all-hit,
    flex partials, entries shrunk by a push -- is a weighted sum over this
    grid, which is why the simulation returns the grid rather than a
    single number.
    """

    n_legs: int
    draws: int
    grid: np.ndarray                   #: (n+1, n+1) probabilities, sums to 1
    hits: np.ndarray = field(repr=False)     #: per-draw count of winning legs
    voids: np.ndarray = field(repr=False)    #: per-draw count of voided legs
    seed: int | None = None

    @property
    def joint_prob(self) -> float:
        """P(every leg hits, nothing voided) -- the orthant probability."""
        return float(self.grid[0, self.n_legs])

    @property
    def joint_prob_se(self) -> float:
        """Monte Carlo standard error on `joint_prob`."""
        p = self.joint_prob
        return math.sqrt(max(p * (1.0 - p), 0.0) / self.draws)

    def prob_exactly(self, k: int) -> float:
        """P(exactly k legs hit), whatever voided. Sums to 1 over k."""
        return float(self.grid[:, k].sum())

    @property
    def hit_distribution(self) -> list[float]:
        """P(exactly k hit) for k = 0..n. Sums to 1 by construction."""
        return [self.prob_exactly(k) for k in range(self.n_legs + 1)]

    @property
    def prob_any_void(self) -> float:
        return float(1.0 - self.grid[0, :].sum())

    def payout_per_draw(self, multiples: np.ndarray) -> np.ndarray:
        """
        Return multiple for every draw, given a (n+1, n+1) lookup of what
        (voids, hits) pays. Used for EV, variance and log-optimal staking.
        """
        return multiples[self.voids, self.hits]


def simulate(
    hit_probs,
    corr: np.ndarray | None = None,
    push_probs=None,
    draws: int = DEFAULT_DRAWS,
    seed: int | None = DEFAULT_SEED,
    base: np.ndarray | None = None,
) -> Simulation:
    """
    Estimate the joint outcome distribution of a set of legs.

    `hit_probs` are the marginal probabilities each leg wins outright (a
    push is not a win). `push_probs`, if given, is the marginal chance
    each leg lands exactly on its line and voids. `corr` is the legs'
    correlation matrix, already signed for the sides taken and already
    PSD; None means independence.

    `base` is a pool of standard normals from `standard_normals`. Pass one
    when scoring many candidates so they share random numbers.
    """
    p = np.asarray(list(hit_probs), dtype=float)
    n = p.size
    if n == 0:
        raise ValueError("a ticket needs at least one leg")
    if np.any(p <= 0.0) or np.any(p >= 1.0):
        raise ValueError(f"leg probabilities must be strictly inside (0,1): {p}")

    pushes = (
        np.zeros(n)
        if push_probs is None
        else np.asarray(list(push_probs), dtype=float)
    )
    if pushes.size != n:
        raise ValueError("push_probs must have one entry per leg")
    if np.any(pushes < 0.0):
        raise ValueError("push probabilities cannot be negative")
    if np.any(p + pushes >= 1.0):
        raise ValueError("a leg's hit and push probabilities must sum below 1")

    t_hit = np.array([threshold_for(x) for x in p])
    t_void = np.array([threshold_for(x + q) for x, q in zip(p, pushes)])

    if base is None:
        base = standard_normals(draws, n, seed)
    else:
        if base.shape[1] < n:
            raise ValueError(
                f"the shared normal pool has {base.shape[1]} columns but the "
                f"ticket has {n} legs"
            )
        base = base[:, :n]
        draws = base.shape[0]

    if corr is None:
        latent = base
    else:
        corr = np.asarray(corr, dtype=float)
        if corr.shape != (n, n):
            raise ValueError(f"correlation matrix must be {n}x{n}, got {corr.shape}")
        latent = base @ cholesky_factor(corr).T

    won = latent > t_hit
    voided = (latent > t_void) & ~won
    hits = won.sum(axis=1).astype(np.int64)
    voids = voided.sum(axis=1).astype(np.int64)

    size = n + 1
    flat = np.bincount(voids * size + hits, minlength=size * size)
    grid = flat.reshape(size, size).astype(float) / draws

    return Simulation(
        n_legs=n, draws=draws, grid=grid, hits=hits, voids=voids, seed=seed
    )


def independent_grid(hit_probs, push_probs=None) -> np.ndarray:
    """
    The EXACT outcome distribution if the legs were independent.

    This is the number every correlated estimate is reported against, so
    it must not carry Monte Carlo noise of its own. It does not have to:
    with independent legs the joint distribution of (voids, hits) is a
    convolution of one trinomial per leg -- hit, push, miss -- and for the
    handful of legs a ticket ever has, convolving them exactly costs
    nothing.

    Estimating this side by simulation too would mean the most important
    comparison in the tool, correlated against independent, was a
    difference of two noisy numbers. Half of that noise is avoidable, so
    it is avoided.
    """
    probs = list(hit_probs)
    n = len(probs)
    pushes = list(push_probs) if push_probs is not None else [0.0] * n
    grid = np.zeros((n + 1, n + 1))
    grid[0, 0] = 1.0
    for p, q in zip(probs, pushes):
        nxt = np.zeros_like(grid)
        nxt[:, 1:] += grid[:, :-1] * p          # the leg hits
        nxt[1:, :] += grid[:-1, :] * q          # the leg pushes
        nxt += grid * (1.0 - p - q)             # the leg misses
        grid = nxt
    return grid


def independent_joint_prob(hit_probs) -> float:
    """
    The product of the marginals: what a fixed-multiplier pick'em product
    implicitly assumes, and the number every correlated estimate is
    reported against.
    """
    out = 1.0
    for p in hit_probs:
        out *= p
    return out


def log_optimal_fraction(
    returns: np.ndarray, lower: float = 0.0, upper: float = 0.99, tol: float = 1e-6
) -> float:
    """
    The bankroll fraction maximising E[log(1 + f(R-1))] for a sampled
    return multiple R.

    Kelly's usual two-outcome formula assumes the bet either wins a fixed
    amount or loses the stake. A flex pick'em entry has five outcomes with
    wildly different multiples, and `pricing.kelly_fraction` cannot see
    that shape. This is the general answer, computed straight off the
    Monte Carlo draws, and it is reported next to the formula's answer as
    a check: where they disagree badly, the payoff is lumpy enough that
    the formula is not describing the bet.
    """
    profit = np.asarray(returns, dtype=float) - 1.0
    if profit.mean() <= 0.0:
        return 0.0

    def derivative(f: float) -> float:
        return float(np.mean(profit / (1.0 + f * profit)))

    if derivative(upper) > 0:
        return upper
    lo, hi = lower, upper
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if derivative(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def compounded_hold(overrounds) -> float:
    """
    Total margin across independent parlay legs.

    A cross-game parlay multiplies the book's prices, which multiplies its
    margin too: four legs at 4.5% hold compound to 1.045^4 - 1 = 19.3%, and
    no amount of correlation rescues that because legs in different games
    have none. This is the arithmetic behind "don't build cross-game
    parlays", and it is here so the tool can state it with a number rather
    than assert it.
    """
    total = 1.0
    for h in overrounds:
        total *= (1.0 + h)
    return total - 1.0
