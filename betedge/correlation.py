"""
Where the correlation matrix comes from, and how honest it is.

Two sources, in strict order of preference:

1. **Structural priors** (`data/correlation_priors.yaml`). Relationships
   that are true by construction -- a quarterback's passing yards are the
   sum of his receivers' receiving yards, two backs split one pool of
   carries, a goalie's saves are the other team's shots. The SIGN of these
   is close to certain. The MAGNITUDE is judgement.

2. **Empirical estimates** fitted from game logs the user supplies, stored
   in the database with their sample size. Used in place of the prior once
   there are enough joint observations to mean anything.

Every pair records which source it came from and that is carried all the
way to the report, because the difference matters more than any other
number in the tool: an expected value assembled entirely from priors is a
hypothesis about a market, and one assembled from five hundred games of
joint data is an estimate of it. They should never look alike.

What is deliberately absent
---------------------------
There is no attempt to infer correlation from the odds. It is tempting --
the books clearly know something -- but the only observable is DraftKings'
same-game parlay price, and backing a correlation out of it would make
this tool agree with DraftKings by construction. The entire premise of
betting a same-game parlay is that DraftKings' correlation estimate is
wrong, so deriving ours from theirs would guarantee we could never find
the thing we are looking for.

Latent correlation, not outcome correlation
-------------------------------------------
The copula needs the correlation of the LATENT normals, not of the
win/lose indicators. Fitting Pearson correlation on raw stat values would
be close but is distorted by the heavy right tails of counting stats
(rushing yards, strikeouts). So the empirical fit computes Spearman's rank
correlation, which is invariant to any monotone transform, and converts it
to the Gaussian-copula latent correlation with the standard identity

    rho = 2 * sin(pi * rho_spearman / 6)

That identity is exact for a Gaussian copula and needs no assumption about
the marginals at all, which is what makes it the right tool here.
"""

from __future__ import annotations

import csv
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

from . import copula
from .rosters import SOURCE_STRUCTURAL

log = logging.getLogger(__name__)

PRIORS_PATH = Path(__file__).parent / "data" / "correlation_priors.yaml"

# How two legs are related to one another.
SAME_PLAYER = "same_player"
SAME_TEAM = "same_team"
OPPOSING_TEAM = "opposing_team"
SAME_GAME = "same_game"
CROSS_GAME = "cross_game"

RELATIONS = (SAME_PLAYER, SAME_TEAM, OPPOSING_TEAM, SAME_GAME, CROSS_GAME)

# Where a pair's number came from. Reported per pair, never averaged away.
SOURCE_EMPIRICAL = "empirical"
SOURCE_PRIOR = "prior"
SOURCE_DEFAULT = "default"

OVER_SIDES = {"over", "yes"}
UNDER_SIDES = {"under", "no"}


def side_sign(side: str | None) -> int:
    """
    Whether a leg's latent points the same way as the underlying stat.

    Priors are written for both legs taken Over. Taking one Under reverses
    that leg's success direction, so the pair's correlation flips sign.
    Getting this wrong is not a rounding error -- it turns a +0.45 stack
    into a -0.45 hedge and inverts the ticket's whole value.

    A leg named after a competitor (a moneyline or a spread) has no
    over/under orientation; its direction is carried by which team it is,
    which the relation already encodes.
    """
    s = (side or "").strip().lower()
    if s in UNDER_SIDES:
        return -1
    return 1


# --------------------------------------------------------------------------
# Priors
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Prior:
    markets: tuple[str, str]
    relation: str
    rho: float
    why: str = ""

    @property
    def wildcard(self) -> bool:
        return "*" in self.markets

    def matches(self, market_a: str, market_b: str, relation: str) -> bool:
        if relation != self.relation:
            return False
        a, b = self.markets
        pair = (market_a, market_b)
        if a == b:
            return pair[0] == a and pair[1] == a
        for x, y in (pair, pair[::-1]):
            if (a == "*" or a == x) and (b == "*" or b == y):
                return True
        return False


@dataclass
class PriorSet:
    """Structural priors, keyed by sport."""

    by_sport: dict[str, list[Prior]] = field(default_factory=dict)
    defaults: dict[str, float] = field(default_factory=dict)
    path: Path | None = None

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PriorSet":
        p = Path(path or PRIORS_PATH)
        raw = yaml.safe_load(p.read_text()) or {}
        by_sport: dict[str, list[Prior]] = {}
        for sport, entries in (raw.get("priors") or {}).items():
            out = []
            for e in entries or []:
                markets = tuple(e["markets"])
                if len(markets) != 2:
                    raise ValueError(
                        f"{p}: prior for {sport} needs exactly two markets, "
                        f"got {markets!r}"
                    )
                relation = e["relation"]
                if relation not in RELATIONS:
                    raise ValueError(
                        f"{p}: unknown relation {relation!r} for {sport}; "
                        f"choose from {list(RELATIONS)}"
                    )
                rho = float(e["rho"])
                if not -1.0 < rho < 1.0:
                    raise ValueError(
                        f"{p}: rho for {sport} {markets} must be inside "
                        f"(-1, 1), got {rho}"
                    )
                out.append(
                    Prior(markets=markets, relation=relation, rho=rho,
                          why=(e.get("why") or "").strip())
                )
            by_sport[sport] = out
        defaults = {
            k: float(v) for k, v in (raw.get("defaults") or {}).items()
        }
        for relation in RELATIONS:
            defaults.setdefault(relation, 0.0)
        return cls(by_sport=by_sport, defaults=defaults, path=p)

    def lookup(
        self, sport: str, market_a: str, market_b: str, relation: str
    ) -> tuple[float, str, str]:
        """
        The prior correlation for one pair.

        Returns (rho, source, why). The most specific match wins: an exact
        market pair beats a wildcard, and a wildcard beats the per-relation
        default, so ``["*", totals]`` can express "every counting stat
        moves with the total" without overriding the named pairs that say
        it more precisely.
        """
        entries = self.by_sport.get(sport, [])
        exact = [
            e for e in entries
            if not e.wildcard and e.matches(market_a, market_b, relation)
        ]
        if exact:
            return exact[0].rho, SOURCE_PRIOR, exact[0].why
        wild = [
            e for e in entries
            if e.wildcard and e.matches(market_a, market_b, relation)
        ]
        if wild:
            return wild[0].rho, SOURCE_PRIOR, wild[0].why
        return self.defaults.get(relation, 0.0), SOURCE_DEFAULT, ""

    def entries_for(self, sport: str) -> list[Prior]:
        return list(self.by_sport.get(sport, []))


# --------------------------------------------------------------------------
# Relations between two legs
# --------------------------------------------------------------------------


def _norm_name(value: str | None) -> str:
    return (value or "").strip().lower()


def relation_between(a, b) -> str:
    """
    How two legs relate, from whatever is actually known.

    The Odds API does not say which team a player plays for, so the team
    on a leg comes from `rosters.py` -- a roster feed, the game logs being
    fitted, the user's override file, or an inference from the slate
    itself. Where it is not known the pair degrades to `same_game` rather
    than being guessed at, which is the whole reason this returns
    `same_game` at all.

    Two teams are only compared when they are drawn from the same LABEL
    SPACE. A structural inference names its sides `evt#A` and `evt#B` --
    true statements about one game and meaningless outside it -- so
    comparing one against a real club would read as "different team" and
    report a confident relation that nothing supports. Mixing them is
    refused instead.
    """
    if a.event_id != b.event_id:
        return CROSS_GAME
    if _norm_name(a.selection) and _norm_name(a.selection) == _norm_name(b.selection):
        return SAME_PLAYER
    team_a, team_b = _norm_name(getattr(a, "team", None)), _norm_name(getattr(b, "team", None))
    if team_a and team_b:
        inferred_a = getattr(a, "team_source", None) == SOURCE_STRUCTURAL
        inferred_b = getattr(b, "team_source", None) == SOURCE_STRUCTURAL
        if inferred_a != inferred_b:
            return SAME_GAME
        return SAME_TEAM if team_a == team_b else OPPOSING_TEAM
    return SAME_GAME


# --------------------------------------------------------------------------
# Assembling the matrix
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PairCorrelation:
    """One off-diagonal entry, with its provenance attached."""

    i: int
    j: int
    rho: float              #: signed for the sides actually taken
    base_rho: float         #: before the Over/Under sign
    sign: int
    relation: str
    source: str             #: empirical | prior | default
    sample_size: int | None = None
    why: str = ""

    @property
    def measured(self) -> bool:
        return self.source == SOURCE_EMPIRICAL

    def describe(self) -> str:
        if self.source == SOURCE_EMPIRICAL:
            return f"{self.rho:+.2f} measured (n={self.sample_size:,})"
        if self.source == SOURCE_PRIOR:
            return f"{self.rho:+.2f} prior"
        return f"{self.rho:+.2f} default"


@dataclass
class CorrelationMatrix:
    """An assembled, PSD, side-signed correlation matrix plus its receipts."""

    matrix: np.ndarray
    pairs: list[PairCorrelation]
    psd: copula.PsdResult
    min_sample: int

    @property
    def projected(self) -> bool:
        return self.psd.projected

    @property
    def any_measured(self) -> bool:
        return any(p.measured for p in self.pairs)

    @property
    def all_prior(self) -> bool:
        """True when not one pair rests on data. The headline caveat."""
        return bool(self.pairs) and not self.any_measured

    @property
    def strongest(self) -> PairCorrelation | None:
        return max(self.pairs, key=lambda p: abs(p.rho), default=None)

    @property
    def most_negative(self) -> PairCorrelation | None:
        return min(self.pairs, key=lambda p: p.rho, default=None)

    def source_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self.pairs:
            out[p.source] = out.get(p.source, 0) + 1
        return out

    def summary(self) -> str:
        if not self.pairs:
            return "no pairs"
        counts = self.source_counts()
        return ", ".join(
            f"{counts[k]} {k}" for k in (SOURCE_EMPIRICAL, SOURCE_PRIOR, SOURCE_DEFAULT)
            if k in counts
        )


def assemble(
    legs: Sequence[Any],
    priors: PriorSet,
    estimates: "EstimateStore | None" = None,
    min_sample: int = 100,
    max_abs_rho: float = 0.95,
) -> CorrelationMatrix:
    """
    Build the correlation matrix for one candidate ticket.

    For each pair: work out the relation, take the empirical estimate if
    there is one with enough observations behind it, otherwise the prior,
    then apply the Over/Under sign and clamp. The assembled matrix is then
    forced positive semi-definite, because pairwise numbers from different
    sources need not be mutually consistent and an inconsistent matrix
    cannot be sampled from at all.
    """
    n = len(legs)
    matrix = np.eye(n)
    pairs: list[PairCorrelation] = []

    for i in range(n):
        for j in range(i + 1, n):
            a, b = legs[i], legs[j]
            relation = relation_between(a, b)
            base_rho, source, why, sample = _pair_rho(
                a, b, relation, priors, estimates, min_sample
            )
            sign = side_sign(a.side) * side_sign(b.side)
            rho = max(-max_abs_rho, min(max_abs_rho, base_rho * sign))
            matrix[i, j] = matrix[j, i] = rho
            pairs.append(
                PairCorrelation(
                    i=i, j=j, rho=rho, base_rho=base_rho, sign=sign,
                    relation=relation, source=source, sample_size=sample, why=why,
                )
            )

    psd = copula.nearest_psd(matrix)
    return CorrelationMatrix(
        matrix=psd.matrix, pairs=pairs, psd=psd, min_sample=min_sample
    )


def _pair_rho(a, b, relation, priors, estimates, min_sample):
    """Empirical if it is well enough evidenced, otherwise the prior."""
    if estimates is not None and a.sport == b.sport:
        found = estimates.lookup(a.sport, a.market, b.market, relation)
        if found is not None and found.n_observations >= min_sample:
            return found.rho, SOURCE_EMPIRICAL, found.note, found.n_observations
    rho, source, why = priors.lookup(a.sport, a.market, b.market, relation)
    return rho, source, why, None


def identity_matrix(n: int) -> np.ndarray:
    """The independence assumption the pick'em sites price with."""
    return np.eye(n)


# --------------------------------------------------------------------------
# Empirical estimation from game logs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Estimate:
    sport: str
    market_a: str
    market_b: str
    relation: str
    rho: float
    spearman: float
    n_observations: int
    n_games: int
    fitted_at: datetime | None = None
    note: str = ""


class EstimateStore:
    """
    Read side of the fitted-correlation table, with an in-memory index.

    Assembling a matrix asks about every pair of every candidate ticket,
    which in a beam search is tens of thousands of lookups. One query at
    the start is cheaper than one per pair by several orders of magnitude.
    """

    def __init__(self, rows: Iterable[Any] = ()):
        self._by_key: dict[tuple, Estimate] = {}
        for r in rows:
            e = _row_to_estimate(r)
            self._by_key[(e.sport, e.market_a, e.market_b, e.relation)] = e

    @classmethod
    def from_db(cls, db) -> "EstimateStore":
        return cls(db.correlation_estimates())

    def lookup(
        self, sport: str, market_a: str, market_b: str, relation: str
    ) -> Estimate | None:
        a, b = sorted((market_a, market_b))
        return self._by_key.get((sport, a, b, relation))

    def __len__(self) -> int:
        return len(self._by_key)

    def all(self) -> list[Estimate]:
        return sorted(
            self._by_key.values(),
            key=lambda e: (e.sport, e.market_a, e.market_b, e.relation),
        )


def _row_to_estimate(row) -> Estimate:
    get = row.__getitem__ if hasattr(row, "keys") else (lambda k: getattr(row, k))
    fitted = get("fitted_at")
    if isinstance(fitted, str):
        try:
            fitted = datetime.fromisoformat(fitted.replace("Z", "+00:00"))
        except ValueError:
            fitted = None
    return Estimate(
        sport=get("sport"),
        market_a=get("market_a"),
        market_b=get("market_b"),
        relation=get("relation"),
        rho=float(get("rho")),
        spearman=float(get("spearman") or 0.0),
        n_observations=int(get("n_observations")),
        n_games=int(get("n_games") or 0),
        fitted_at=fitted,
        note=(get("note") if "note" in _keys(row) else "") or "",
    )


def _keys(row) -> set:
    try:
        return set(row.keys())
    except AttributeError:
        return set()


@dataclass
class GameLogRow:
    """One player's one stat in one game."""

    game_id: str
    sport: str
    player: str
    team: str
    market: str
    value: float
    date: str = ""


REQUIRED_LOG_COLUMNS = ("game_id", "sport", "player", "team", "market", "value")


def read_game_logs(path: str | Path) -> list[GameLogRow]:
    """
    Load box-score rows from a CSV the user supplies.

    Columns: game_id, sport, player, team, market, value, and optionally
    date. One row per player per stat per game, with `market` using the
    same keys the scanner uses (`player_pass_yds`, `pitcher_strikeouts`)
    so a fitted number lines up with the leg it will be used for.

    Nothing is fetched or scraped. Box scores live behind terms that
    mostly prohibit it, and the ones that do not are a user's choice to
    make, not this tool's.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no game log at {p}")
    rows: list[GameLogRow] = []
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_LOG_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"{p} is missing column(s) {', '.join(missing)}. "
                f"Expected: {', '.join(REQUIRED_LOG_COLUMNS)}[, date]"
            )
        for i, row in enumerate(reader, 2):
            try:
                value = float(row["value"])
            except (TypeError, ValueError):
                log.debug("%s line %d: unparseable value %r, skipped", p, i, row.get("value"))
                continue
            if not row.get("game_id") or not row.get("player") or not row.get("market"):
                continue
            rows.append(
                GameLogRow(
                    game_id=row["game_id"].strip(),
                    sport=(row.get("sport") or "").strip(),
                    player=row["player"].strip(),
                    team=(row.get("team") or "").strip(),
                    market=row["market"].strip(),
                    value=value,
                    date=(row.get("date") or "").strip(),
                )
            )
    return rows


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Spearman's rank correlation, ties averaged.

    Rank-based on purpose. Counting stats have long right tails -- one
    200-yard rushing game moves a Pearson correlation more than fifty
    ordinary ones -- and ranks are immune to that while still capturing
    the monotone dependence the copula needs.
    """
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if a.size < 3:
        return 0.0
    ra, rb = _rankdata(a), _rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks, which is what ties demand and what corrcoef expects."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(1, values.size + 1, dtype=float)
    sorted_values = values[order]
    start = 0
    for i in range(1, values.size + 1):
        if i == values.size or sorted_values[i] != sorted_values[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return ranks


def spearman_to_latent(rho_s: float) -> float:
    """
    Spearman's rho -> Gaussian copula correlation.

        rho = 2 * sin(pi * rho_s / 6)

    Exact for a Gaussian copula and free of any assumption about the
    marginal distributions, which is exactly why the fit goes through
    ranks rather than raw values.
    """
    return 2.0 * math.sin(math.pi * max(-1.0, min(1.0, rho_s)) / 6.0)


def fit_correlations(
    rows: Sequence[GameLogRow],
    markets: Sequence[str] | None = None,
    max_pairs_per_bucket: int = 200_000,
) -> list[Estimate]:
    """
    Fit pairwise latent correlations from game logs.

    Every pair of stat lines inside one game becomes one joint
    observation, bucketed by (sport, market pair, relation). A bucket's
    Spearman correlation is converted to the latent scale and returned
    with the number of observations behind it, which is the number that
    decides whether it is used at all.

    Same-market pairs are added in both orderings so the estimate is
    symmetric rather than depending on which player the box score listed
    first.
    """
    wanted = set(markets) if markets else None
    by_game: dict[tuple[str, str], list[GameLogRow]] = defaultdict(list)
    for r in rows:
        if wanted and r.market not in wanted:
            continue
        by_game[(r.sport, r.game_id)].append(r)

    buckets: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    games: dict[tuple, set] = defaultdict(set)
    dropped = 0

    for (sport, game_id), entries in by_game.items():
        for idx_a in range(len(entries)):
            for idx_b in range(idx_a + 1, len(entries)):
                a, b = entries[idx_a], entries[idx_b]
                if a.player == b.player and a.market == b.market:
                    continue        # the same number twice
                relation = _log_relation(a, b)
                mk_a, mk_b = a.market, b.market
                va, vb = a.value, b.value
                if mk_a > mk_b:
                    mk_a, mk_b, va, vb = mk_b, mk_a, vb, va
                key = (sport, mk_a, mk_b, relation)
                bucket = buckets[key]
                if len(bucket) >= max_pairs_per_bucket:
                    dropped += 1
                    continue
                bucket.append((va, vb))
                if mk_a == mk_b:
                    bucket.append((vb, va))
                games[key].add(game_id)

    if dropped:
        log.warning(
            "%d joint observations dropped at the per-bucket cap of %d; "
            "raise --max-pairs if you need every one",
            dropped, max_pairs_per_bucket,
        )

    fitted_at = datetime.now(timezone.utc)
    out: list[Estimate] = []
    for (sport, mk_a, mk_b, relation), observations in sorted(buckets.items()):
        if len(observations) < 3:
            continue
        xs = [o[0] for o in observations]
        ys = [o[1] for o in observations]
        rho_s = spearman(xs, ys)
        out.append(
            Estimate(
                sport=sport,
                market_a=mk_a,
                market_b=mk_b,
                relation=relation,
                rho=spearman_to_latent(rho_s),
                spearman=rho_s,
                # Same-market buckets hold each pair twice to stay
                # symmetric; the honest observation count is the number of
                # distinct pairs, not the number of rows fed to the fit.
                n_observations=len(observations) // 2 if mk_a == mk_b else len(observations),
                n_games=len(games[(sport, mk_a, mk_b, relation)]),
                fitted_at=fitted_at,
                note="fitted from supplied game logs",
            )
        )
    return out


def _log_relation(a: GameLogRow, b: GameLogRow) -> str:
    if a.player == b.player:
        return SAME_PLAYER
    if a.team and b.team:
        return SAME_TEAM if a.team == b.team else OPPOSING_TEAM
    return SAME_GAME
