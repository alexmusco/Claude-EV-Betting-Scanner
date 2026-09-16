"""
Turning the margin priors from recollection into measurement.

Where the data comes from
-------------------------
nflverse publishes every NFL game since 1999 with its final score AND the
closing spread and moneyline. That is the same project this tool already
uses for rosters, it is public, and it is the right answer to "where do I
get scores" -- better than scores alone, because having the closing line
beside the result is what makes the model checkable rather than merely
fittable.

    https://github.com/nflverse/nfldata  ->  data/games.csv

What measuring it changed
-------------------------
Two things, and the second one inverted a design decision.

The key numbers were wrong. Written from recollection the file claimed 3,
7, 10, 14, 6 and 4. Measured over 6,983 regular-season games, 6 and 4 are
not key numbers at all -- they sit slightly BELOW a smooth fit -- while 1,
17, 21 and 24 are, and 3 is far lumpier than any prior suggested: 15.0%
of games land exactly on it, nearly three times a smooth model's
expectation.

And sigma should not be fitted per game. The module was built around
solving sigma from the spread and the moneyline together, which is
elegant and is BIASED: across every de-vig method the implied sigma comes
out near 11.3-11.6, while the realised standard deviation of (margin
minus closing spread) is 13.19. A too-small sigma understates the outer
rungs, which is exactly where this scan looks, so the error would have
been invisible and would have quietly suppressed the tool's own findings.

The cause is that margins are not normal. Fifteen percent of games land
on exactly three points; a normal has no way to represent that, so the
sigma that best reproduces a moneyline is not the sigma that describes
the spread of outcomes.

So the measured constant is now primary and the moneyline is a CHECK:
where the implied sigma disagrees with the measured one by a lot, that is
reported as a disagreement rather than silently believed.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .ladder import norm_cdf, norm_ppf

NFLVERSE_GAMES_URL = (
    "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
)

#: A bump smaller than this is noise, not a key number.
MIN_BUMP = 0.25
#: Below this many games at a margin, the frequency is not worth trusting.
MIN_GAMES_AT_MARGIN = 30


@dataclass
class Game:
    """One finished game with the line it closed at."""

    season: int
    margin: int          # home score minus away score
    spread: float        # home perspective, positive when home is favoured
    home_moneyline: float | None = None
    away_moneyline: float | None = None

    @property
    def residual(self) -> float:
        """What the line missed by. Its spread IS sigma."""
        return self.margin - self.spread


@dataclass
class Calibration:
    """What a set of finished games says about a sport."""

    sport: str
    games: int = 0
    seasons: tuple = ()
    sigma: float = 0.0
    bias: float = 0.0
    key_numbers: dict = field(default_factory=dict)
    #: The moneyline's own implied sigma, for comparison. Not used to
    #: price anything -- it is the number the measurement replaced.
    implied_sigma: float | None = None
    sigma_by_spread: dict = field(default_factory=dict)

    @property
    def line_is_unbiased(self) -> bool:
        """Whether the closing line is centred, as a sharp line should be."""
        return abs(self.bias) < 0.5

    @property
    def implied_disagrees(self) -> bool:
        if self.implied_sigma is None or not self.sigma:
            return False
        return abs(self.implied_sigma - self.sigma) / self.sigma > 0.08

    def describe(self) -> str:
        lines = [
            f"{self.sport}: {self.games:,} games"
            + (f", {self.seasons[0]}-{self.seasons[1]}" if self.seasons else ""),
            f"  sigma      {self.sigma:.2f}   (residual of margin minus "
            "closing spread)",
            f"  bias       {self.bias:+.3f}   "
            + ("line is centred" if self.line_is_unbiased
               else "LINE IS OFF CENTRE -- check the sign convention"),
        ]
        if self.implied_sigma is not None:
            lines.append(
                f"  moneyline implies {self.implied_sigma:.2f}"
                + ("   <- DISAGREES with the measurement"
                   if self.implied_disagrees else "   (agrees)")
            )
        if self.key_numbers:
            got = ", ".join(f"{int(k)}:{v:+.0%}"
                            for k, v in sorted(self.key_numbers.items()))
            lines.append(f"  key numbers  {got}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reading games
# ---------------------------------------------------------------------------


def read_nflverse_games(path, regular_season_only: bool = True) -> list[Game]:
    """
    Parse nflverse `games.csv`.

    Playoffs are excluded by default: a neutral-site, rested, single-
    elimination game is drawn from a different distribution, and 7,000
    regular-season games is plenty without muddying them.
    """
    games = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if regular_season_only and row.get("game_type") != "REG":
                continue
            if not row.get("result") or not row.get("spread_line"):
                continue
            try:
                margin = int(float(row["result"]))
                spread = float(row["spread_line"])
                season = int(row["season"])
            except (TypeError, ValueError):
                continue
            games.append(Game(
                season=season, margin=margin, spread=spread,
                home_moneyline=_number(row.get("home_moneyline")),
                away_moneyline=_number(row.get("away_moneyline")),
            ))
    return games


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def download_games(destination, url: str = NFLVERSE_GAMES_URL,
                   session=None) -> Path:
    """Fetch nflverse's game file. Public data, no key."""
    import requests

    getter = session.get if session is not None else requests.get
    response = getter(url, timeout=120)
    response.raise_for_status()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(response.text)
    return destination


# ---------------------------------------------------------------------------
# The measurements
# ---------------------------------------------------------------------------


def measure_sigma(games) -> tuple[float, float]:
    """
    (sigma, bias) of the residual -- what the closing line missed by.

    This is the honest number. It asks how far outcomes actually land
    from the line, rather than what sigma would reproduce a moneyline
    under a normal the margins do not follow.
    """
    residuals = [g.residual for g in games]
    if len(residuals) < 2:
        raise ValueError("not enough games to measure a standard deviation")
    return statistics.pstdev(residuals), statistics.fmean(residuals)


def measure_implied_sigma(games, devig_method: str = "multiplicative"):
    """
    The sigma the moneylines imply, for comparison only.

    Kept because the disagreement is the finding: it is what showed the
    per-game fit was biased, and re-measuring it on new data is how that
    conclusion stays checkable rather than becoming folklore.
    """
    from . import pricing

    values = []
    for game in games:
        if game.home_moneyline is None or game.away_moneyline is None:
            continue
        try:
            home = pricing.american_to_decimal(game.home_moneyline)
            away = pricing.american_to_decimal(game.away_moneyline)
        except Exception:  # noqa: BLE001
            continue
        if home <= 1 or away <= 1:
            continue
        fair = pricing.devig([home, away], method=devig_method)
        if game.spread > 0:
            mu, prob = game.spread, fair[0]
        elif game.spread < 0:
            mu, prob = -game.spread, fair[1]
        else:
            continue
        if not 0.5 < prob < 0.999 or mu < 0.5:
            continue
        z = norm_ppf(prob)
        if z <= 0.01:
            continue
        sigma = mu / z
        if 5.0 < sigma < 30.0:
            values.append(sigma)
    return statistics.median(values) if values else None


def measure_sigma_by_spread(games, edges=(0, 3, 7, 10, 14, 99)) -> dict:
    """
    Sigma within each spread band.

    The question this answers is whether sigma scales with the spread,
    which is what a per-game fit implicitly assumes. On NFL data it does
    not -- it is flat to within half a point across the whole range --
    and that is the second reason the fit was the wrong design.
    """
    out = {}
    for low, high in zip(edges, edges[1:]):
        band = [g.residual for g in games if low <= abs(g.spread) < high]
        if len(band) >= 50:
            out[f"{low}-{high}"] = round(statistics.pstdev(band), 2)
    return out


def measure_key_numbers(games, sigma: float) -> dict:
    """
    How much more often each margin happens than a smooth model expects.

    The baseline is not a flat average of neighbours but a normal with
    the measured sigma, centred on EACH GAME'S OWN SPREAD and averaged
    over the real spread distribution. That matters: margins near zero
    are common partly because most games are close, and crediting that
    to a key number would invent lumps that are really just the shape of
    the schedule.

    Returns multiplicative bumps, so +1.8 means "about three times as
    often as a smooth fit predicts". Troughs come back negative and are
    kept -- nine points is genuinely RARE, and a model that only knows
    about the peaks overprices the gaps between them.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    total = len(games)
    if not total:
        return {}

    observed = Counter(abs(g.margin) for g in games)
    expected = Counter()
    for game in games:
        for margin in range(-60, 61):
            expected[abs(margin)] += (
                norm_cdf((margin + 0.5 - game.spread) / sigma)
                - norm_cdf((margin - 0.5 - game.spread) / sigma)
            )

    bumps = {}
    for margin, count in observed.items():
        smooth_count = expected[margin]
        # BOTH counts must be substantial. Requiring only the observed
        # one leaves the far tail wide open: out past thirty points the
        # smooth expectation is a fraction of a game, so a handful of
        # blowouts produces a "+150% key number" that is pure sampling
        # noise -- and it would then be applied to every ladder.
        if count < MIN_GAMES_AT_MARGIN or smooth_count < MIN_GAMES_AT_MARGIN:
            continue
        smooth = smooth_count / total
        if smooth <= 0:
            continue
        bump = (count / total) / smooth - 1.0
        if abs(bump) >= MIN_BUMP:
            bumps[float(margin)] = round(bump, 3)
    return bumps


def calibrate(games, sport: str = "americanfootball_nfl") -> Calibration:
    """Everything measurable about a sport, from finished games."""
    if not games:
        raise ValueError("no games to calibrate from")
    sigma, bias = measure_sigma(games)
    seasons = (min(g.season for g in games), max(g.season for g in games))
    return Calibration(
        sport=sport, games=len(games), seasons=seasons,
        sigma=round(sigma, 3), bias=round(bias, 4),
        key_numbers=measure_key_numbers(games, sigma),
        implied_sigma=(
            round(v, 2) if (v := measure_implied_sigma(games)) else None
        ),
        sigma_by_spread=measure_sigma_by_spread(games),
    )


# ---------------------------------------------------------------------------
# Writing it back
# ---------------------------------------------------------------------------


def compare_with(result: Calibration, prior) -> list[str]:
    """
    How a fresh measurement differs from what is already loaded.

    Empty means the shipped file is up to date, which is the common case
    and deserves to be SAID: a calibration run that always ends "paste
    this in" hands you work that is already done, and after the second
    time you stop reading the output.
    """
    if prior is None:
        return [f"{result.sport} is not in the priors file at all"]
    differences = []
    if not prior.verified:
        differences.append("the loaded priors are still marked unverified")
    if not prior.sigma_measured:
        differences.append("the loaded priors have no measured sigma")
    elif abs(prior.sigma_measured - result.sigma) > 0.05:
        differences.append(
            f"sigma {prior.sigma_measured:g} loaded against "
            f"{result.sigma:g} measured"
        )
    loaded = {float(k): round(float(v), 3)
              for k, v in (prior.key_numbers or {}).items()}
    fresh = {float(k): round(float(v), 3) for k, v in result.key_numbers.items()}
    for margin in sorted(set(loaded) | set(fresh)):
        was, now = loaded.get(margin), fresh.get(margin)
        if was is None:
            differences.append(f"margin {margin:g}: new, {now:+.0%}")
        elif now is None:
            differences.append(f"margin {margin:g}: gone, was {was:+.0%}")
        elif abs(was - now) > 0.02:
            differences.append(
                f"margin {margin:g}: {was:+.0%} loaded, {now:+.0%} measured"
            )
    return differences


def to_yaml_block(result: Calibration, today: str = "") -> str:
    """One sport's measured priors, as YAML ready to paste or write."""
    band = 0.35 * result.sigma
    lines = [
        f"  {result.sport}:",
        f"    # MEASURED from {result.games:,} regular-season games"
        + (f", {result.seasons[0]}-{result.seasons[1]}" if result.seasons else "")
        + ".",
        "    # Source: nflverse/nfldata data/games.csv. Regenerate with",
        "    #   betedge kalshi calibrate --refresh",
        f"    sigma_measured: {result.sigma:g}",
        f"    sigma_typical: {result.sigma:g}",
        f"    sigma_low: {round(result.sigma - band, 1):g}",
        f"    sigma_high: {round(result.sigma + band, 1):g}",
        "    verified: true",
    ]
    if today:
        lines.append(f"    measured_on: \"{today}\"")
    if result.implied_sigma is not None:
        lines.append(
            f"    # The moneylines imply {result.implied_sigma:g}, which is "
            + ("NOT the same" if result.implied_disagrees else "close")
            + " -- see calibrate.py."
        )
    lines.append("    key_numbers:")
    for margin, bump in sorted(result.key_numbers.items()):
        lines.append(f"      {int(margin)}: {bump:g}")
    return "\n".join(lines) + "\n"
