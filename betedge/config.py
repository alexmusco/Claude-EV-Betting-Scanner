"""Configuration loading. YAML file + environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config.yaml")


@dataclass
class ApiConfig:
    base_url: str = "https://api.the-odds-api.com/v4"
    key_env: str = "ODDS_API_KEY"
    key: str | None = None
    timeout_seconds: float = 20.0
    max_retries: int = 3
    # Refuse to start a scan that would leave you below this many credits.
    min_credits_remaining: int = 100
    # Hard ceiling on what one scan may spend.
    max_credits_per_scan: int = 600
    # Seconds to reuse a cached response. Keep at 0 for live scanning;
    # raise it while developing so you stop burning quota on reruns.
    cache_seconds: int = 0
    cache_dir: str = "data/cache"

    def resolve_key(self) -> str:
        key = self.key or os.environ.get(self.key_env)
        if not key:
            raise RuntimeError(
                f"No API key. Set {self.key_env} in your environment or a .env "
                "file, or put it under api.key in config.yaml."
            )
        return key


@dataclass
class BooksConfig:
    sharp: str = "pinnacle"
    soft: list[str] = field(default_factory=lambda: ["draftkings", "underdog"])

    @property
    def all(self) -> list[str]:
        return [self.sharp] + [b for b in self.soft if b != self.sharp]


@dataclass
class ModelConfig:
    # multiplicative | additive | shin | power | worst_case
    # worst_case takes the lowest fair probability any method produces, which
    # is the conservative choice: a bet only clears the bar if it clears it
    # under every method. On lopsided markets the methods disagree a lot.
    devig_method: str = "worst_case"

    # Flag anything at or above this expected value.
    min_ev: float = 0.02
    # Above this, something is probably wrong rather than profitable: a stale
    # line, a mismatched player, a market about to be pulled. Still reported,
    # but marked suspect and given no stake recommendation.
    max_plausible_ev: float = 0.15

    # Sanity bounds on the sharp book's own margin. Outside these the quote
    # is malformed or the market is closing.
    min_overround: float = 0.005
    max_overround: float = 0.15

    # If the de-vig methods disagree by more than this many probability
    # points, the fair estimate is method-dependent -- skip it.
    max_devig_spread: float = 0.04

    # Skip a soft quote whose last_update is older than this.
    max_soft_staleness_minutes: float = 20.0
    # Skip a sharp quote whose last_update is older than this.
    max_sharp_staleness_minutes: float = 20.0

    # Do not flag anything starting sooner than this -- you will not get it
    # down in time, and prices move fastest in the last minutes.
    min_minutes_to_start: float = 5.0
    # Do not look further ahead than this.
    max_hours_to_start: float = 72.0

    include_alternate_lines: bool = False

    # How much extra edge to demand from a thin market. The bar becomes
    #     min_ev * (1 + penalty * (1 - liquidity))
    # so at the default 1.5 a deep market is flagged at +2%, a player prop
    # at roughly +3%, and an alternate line has to show about +4.5%. This is
    # not caution for its own sake: the fair probability is an estimate, and
    # in a thin market its error is comparable to the edge being claimed.
    # Set to 0 for a single flat bar across every market.
    liquidity_ev_penalty: float = 1.5

    # Refuse to flag anything below this liquidity score at all, whatever
    # its EV. 0 disables the floor.
    min_liquidity: float = 0.0


@dataclass
class BankrollConfig:
    # Set this to what you are actually willing to lose, not your net worth.
    amount: float = 1000.0
    # Fraction of full Kelly. Full Kelly assumes your probability estimate is
    # exact; it is not. Quarter Kelly is the usual compromise.
    kelly_multiplier: float = 0.25
    # Hard cap on any single stake as a share of bankroll.
    max_fraction: float = 0.02
    min_stake: float = 5.0
    round_to: float = 5.0
    # Refuse to recommend more than this share of bankroll live at once.
    max_total_exposure_fraction: float = 0.15


@dataclass
class BudgetConfig:
    """
    Monthly credit plan. The per-scan ceiling in ApiConfig stops one runaway
    call; this stops the slower failure of spending three weeks of quota in
    four days. See budget.py.
    """

    # Credits your plan gives you each month.
    monthly_credits: int = 20000
    # Day of the month the plan renews. Leave at 1 if you do not know.
    cycle_day: int = 1
    # Held back so closing-line capture is never starved by scanning. CLV is
    # the only fast evidence that the model works, so it gets paid first.
    reserve: int = 800
    # A day may spend this multiple of the even daily pace, letting an NFL
    # Sunday borrow from a quiet Tuesday. Above 1.0 and below about 4.
    daily_burst: float = 2.0
    # Set false to let scans run unthrottled and use only the per-scan cap.
    enabled: bool = True


@dataclass
class Config:
    api: ApiConfig = field(default_factory=ApiConfig)
    books: BooksConfig = field(default_factory=BooksConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    bankroll: BankrollConfig = field(default_factory=BankrollConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    # Sports scanned for PLAYER PROPS. Expensive: one call per event per
    # market off the per-event endpoint.
    sports: list[str] = field(
        default_factory=lambda: ["basketball_nba", "americanfootball_nfl"]
    )
    # Sports scanned for GAME-LEVEL markets -- moneyline, spreads, totals.
    # Cheap: cost is markets x 1 for the entire sport, not per event, so a
    # full sweep of every match on the board is about three credits. A
    # trailing '*' matches by prefix, which is how you follow tennis, whose
    # sport keys rotate tournament by tournament.
    core_sports: list[str] = field(default_factory=list)
    # Sports never to scan, whatever else asks for them. Patterns may carry
    # a wildcard anywhere. This is enforced after every other selection --
    # config lists, CLI overrides and wildcard expansion alike -- so a
    # blocked sport cannot be reintroduced by a flag.
    #
    # Defaults to excluding college sports: Oregon prohibits collegiate
    # wagering, so DraftKings will not take the bet and a flagged NCAA
    # edge is wasted credits and a wasted look.
    excluded_sports: list[str] = field(
        default_factory=lambda: ["*ncaa*"]
    )
    # Narrow the prop markets for a sport. Cost is markets x events, so
    # cutting MLB from ten markets to two takes a daily scan from ~150
    # credits to ~30. Anything not listed here uses the full set from
    # markets.py.
    prop_markets: dict[str, list[str]] = field(default_factory=dict)
    # Per-sport override of how far ahead the PROP pass will look, in hours.
    # Cost is markets x events, so looking three days out at a sport whose
    # lines are placeholders until game day is the most expensive way to
    # scan nothing. MLB is the clear case: batter props void if the player
    # does not start and pitcher props void on a scratch, so before lineups
    # post -- two to four hours out -- the numbers are not real prices.
    prop_windows: dict[str, float] = field(default_factory=dict)
    # Per-sport override of the game-level market list, same idea as
    # prop_markets. The bulk endpoint bills for markets REQUESTED, not
    # returned, so asking MMA for spreads and totals it does not price costs
    # two credits a sweep for nothing.
    core_markets: dict[str, list[str]] = field(default_factory=dict)
    database: str = "data/betedge.db"
    reports_dir: str = "reports"

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        path = Path(path or DEFAULT_CONFIG_PATH)
        raw: dict[str, Any] = {}
        if path.exists():
            raw = yaml.safe_load(path.read_text()) or {}
        _load_dotenv()
        return cls(
            api=ApiConfig(**raw.get("api", {})),
            books=BooksConfig(**raw.get("books", {})),
            model=ModelConfig(**raw.get("model", {})),
            bankroll=BankrollConfig(**raw.get("bankroll", {})),
            budget=BudgetConfig(**raw.get("budget", {})),
            sports=raw.get("sports", ["basketball_nba", "americanfootball_nfl"]),
            core_sports=raw.get("core_sports", []) or [],
            excluded_sports=(
                raw["excluded_sports"]
                if isinstance(raw.get("excluded_sports"), list)
                else ["*ncaa*"]
            ),
            prop_markets=raw.get("prop_markets", {}) or {},
            prop_windows=raw.get("prop_windows", {}) or {},
            core_markets=raw.get("core_markets", {}) or {},
            database=raw.get("database", "data/betedge.db"),
            reports_dir=raw.get("reports_dir", "reports"),
        )

    def markets_for_sport(self, sport: str, include_alternate: bool = False) -> list[str]:
        """
        Prop markets to request for one sport: the config override if there
        is one, otherwise the registry default.
        """
        from .markets import markets_for

        override = self.prop_markets.get(sport)
        if override:
            return list(override)
        return markets_for(sport, include_alternate=include_alternate)

    def is_excluded(self, sport: str) -> bool:
        """Whether this sport is off limits."""
        from .markets import matches_any

        return matches_any(sport, self.excluded_sports)

    def allowed(self, sports: list[str]) -> tuple[list[str], list[str]]:
        """Split a sport list into (scannable, blocked)."""
        keep, blocked = [], []
        for s in sports:
            (blocked if self.is_excluded(s) else keep).append(s)
        return keep, blocked

    def core_markets_for_sport(self, sport: str) -> list[str]:
        """Game-level markets to request: the config override, else the
        registry default."""
        from .markets import core_markets_for

        override = self.core_markets.get(sport)
        if override:
            return list(override)
        return core_markets_for(sport)

    def prop_window_hours(self, sport: str) -> float:
        """How far ahead the prop pass looks for one sport."""
        return float(self.prop_windows.get(sport, self.model.max_hours_to_start))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["api"].pop("key", None)  # never serialise the key
        return d


def _load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env reader so there is no extra dependency."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        os.environ.setdefault(key, value)
