"""Configuration loading. YAML file + environment overrides."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config.yaml")
PROFILES_PATH = Path(__file__).parent / "data" / "profiles.yaml"

#: Config sections a profile may reach into, field by field.
_PROFILE_SECTIONS = ("model", "parlay", "bankroll", "budget", "books", "api")
#: Per-sport dictionaries a profile merges into rather than replaces. A
#: value of None removes the user's override, falling back to the registry.
_PROFILE_MAPPINGS = ("prop_markets", "prop_windows", "core_markets")
#: Plain lists a profile replaces outright.
_PROFILE_LISTS = ("sports", "core_sports", "excluded_sports")


@dataclass(frozen=True)
class ProfileChange:
    """One override a profile applied, and what the value was before."""

    path: str
    before: Any
    after: Any

    def _absent(self) -> str:
        """
        What "not set" falls back to, which differs by setting -- and
        saying "registry default" for a look-ahead window, where absent
        actually means the global horizon, would be a small confident lie
        in a line whose whole job is to be checkable.
        """
        if self.path.startswith(("prop_markets.", "core_markets.")):
            return "full registry list"
        if self.path.startswith("prop_windows."):
            return "the global look-ahead"
        return "not set"

    def _render(self, value: Any) -> str:
        if value is None:
            return self._absent()
        if isinstance(value, (list, tuple)):
            if len(value) > 3:
                return f"{len(value)} items"
            return "[" + ", ".join(str(v) for v in value) + "]"
        return str(value)

    def describe(self) -> str:
        return f"{self.path:<38} {self._render(self.before)} -> {self._render(self.after)}"


def load_profiles(path: str | Path | None = None) -> dict[str, dict]:
    """
    Shipped profiles, with any the user defined layered on top.

    A profile of the same name in config.yaml replaces the shipped one
    outright rather than merging with it, so a user editing a profile is
    never fighting a default they cannot see.
    """
    profiles: dict[str, dict] = {}
    shipped = Path(PROFILES_PATH)
    if shipped.exists():
        raw = yaml.safe_load(shipped.read_text()) or {}
        profiles.update(raw.get("profiles") or {})
    if path is not None:
        p = Path(path)
        if p.exists():
            raw = yaml.safe_load(p.read_text()) or {}
            profiles.update(raw.get("profiles") or {})
    return profiles



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
class ParlayConfig:
    """
    The multi-leg optimizer. See parlay.py.

    Everything here is deliberately stricter than the single-bet path. A
    parlay compounds the error in every leg's probability estimate, its
    payoff is lumpy, and correlated legs have fatter tails than any of the
    individual bets do -- so the Kelly fraction is halved again, the
    plausibility ceiling is treated as a hard stop, and a ticket that is
    only positive because of an ASSUMED correlation is never staked
    silently.
    """

    # Which payout structures to build tickets for. Names come from
    # betedge/data/payouts.yaml; see `betedge parlay verify-payouts`.
    #: Payout structures to build tickets for. One per book you would
    #: actually enter at -- legs are matched to a product by book, so
    #: listing both keeps the two sets of tickets separate and priced on
    #: their own ladders rather than blended.
    products: list[str] = field(
        default_factory=lambda: ["prizepicks_power", "underdog_standard"]
    )
    # Override the shipped data files. None uses what ships with betedge.
    payouts_path: str | None = None
    priors_path: str | None = None
    # Your own player,team CSV or YAML. Optional, and highest precedence:
    # on the morning of a trade you know before any feed does. Teams
    # otherwise come from the game logs you fit correlations from and from
    # a published roster feed -- see rosters.py.
    rosters_path: str | None = None
    # Pull a roster feed during a scan when its cached snapshot has gone
    # cold. Set false to keep a scan's only network call the Odds API, and
    # refresh by hand with `betedge parlay rosters --refresh`.
    roster_auto_refresh: bool = True
    roster_refresh_days: float = 3.0
    # Past this, a player's team is not used at all. A stale roster is
    # worse than no roster: an unknown team costs the weak blended prior,
    # a WRONG team puts a confident +0.45 on a pair that is really -0.10
    # and nothing downstream questions it.
    roster_max_age_days: float = 30.0

    # Monte Carlo. The seed is fixed so a re-run does not reshuffle the
    # ranking; the standard error is reported rather than hidden.
    draws: int = 200_000
    search_draws: int = 8_000
    seed: int = 20260915

    # Search shape. Beam search, not enumeration: the number of 5-leg
    # subsets of 40 candidates is 658,008, and each one needs a simulation.
    min_legs: int = 2
    max_legs: int = 5
    beam_width: int = 24
    max_candidates_per_group: int = 32
    top_n: int = 12
    # same_game is the default because that is where correlation lives. A
    # quarterback in one game and a centre in another have no structural
    # relationship and only add variance.
    grouping: str = "same_game"
    slate_hours: float = 12.0

    # Bars and guards.
    min_ev: float = 0.02
    # Above this something is wrong -- a mismatched line, a stale quote, or
    # a payout table that does not match what the account actually offers.
    # A genuine pick'em edge is a few points, not forty.
    max_plausible_ev: float = 0.25
    # Flag when the Monte Carlo noise is large next to the edge itself.
    max_se_ratio: float = 0.25
    # How far below the product's break-even a single leg may sit. Negative
    # on purpose: correlation is supposed to make up a small shortfall, and
    # a filter at 0 would reject every ticket the tool exists to find.
    min_leg_edge: float = -0.03
    min_leg_prob: float = 0.25
    max_leg_prob: float = 0.90
    # For books that post a price (DraftKings), the single-bet EV floor.
    min_leg_ev: float = -0.06
    # Report a pair as probable negative correlation at or below this.
    negative_pair_threshold: float = -0.10

    # Staking. An eighth of Kelly rather than the single-bet quarter.
    kelly_multiplier: float = 0.125
    max_ticket_fraction: float = 0.01
    # Five tickets on one game are one bet, not five.
    max_game_exposure_fraction: float = 0.05

    # Joint observations needed before a fitted correlation displaces the
    # structural prior.
    min_correlation_sample: int = 100

    # Comparing a pick'em line against a Pinnacle line at a different
    # number is not a measurement. Off by default; when on, every leg it
    # touches is marked estimated.
    allow_line_interpolation: bool = False
    max_interpolation_distance: float = 1.0

    # Books whose legs are fixed-multiplier pick'em selections.
    pickem_books: list[str] = field(
        default_factory=lambda: ["prizepicks", "underdog"]
    )
    # Push probability to assume on an integer line when Pinnacle does not
    # price both surrounding half-lines. Flagged wherever it is used.
    assumed_push_prob: float = 0.05


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
class TomatoesConfig:
    """
    Rotten Tomatoes threshold contracts on Kalshi. See tomatoes.py.

    Stricter again than the parlay path, for a reason that has nothing to
    do with the maths: these markets are thin and settle on a number a
    third party rounds, so the ways to be wrong are clerical rather than
    statistical and the guards are aimed there.
    """

    # A snapshot older than this is not a measurement of anything -- the
    # score moves whenever a review lands, and the whole edge is knowing
    # the count NOW.
    max_snapshot_age_hours: float = 12.0
    # Below this many counted reviews the posterior is too wide to mean
    # anything, whatever it says.
    min_reviews: int = 15
    # Ceiling on further reviews when nothing better is known. Deliberately
    # generous: it only ever makes the tool more cautious.
    default_max_new_reviews: int = 60
    # Expected further reviews, for the probabilistic branch.
    default_expected_new_reviews: float = 20.0
    # Weak and symmetric, worth about four reviews.
    prior_alpha: float = 2.0
    prior_beta: float = 2.0
    # Log-odds shift applied to LATER reviews. Zero by default and a PRIOR,
    # not a measurement -- see tomatoes.py. Anything only +EV because of
    # it is flagged and staked at nothing.
    drift: float = 0.0
    # The EV bar, and the ceiling above which an edge means a mistake
    # rather than an opportunity.
    min_ev: float = 0.03
    max_plausible_ev: float = 0.40
    # Kelly, stricter than the single-bet quarter. These settle in days and
    # cannot be hedged out of cheaply.
    kelly_multiplier: float = 0.125
    max_position_fraction: float = 0.01
    # Positions are assumed to cross the spread unless told otherwise: the
    # taker fee is four times the maker fee, so assuming the cheap one
    # would flatter every position the tool prints.
    assume_maker: bool = False


@dataclass
class Config:
    api: ApiConfig = field(default_factory=ApiConfig)
    books: BooksConfig = field(default_factory=BooksConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    bankroll: BankrollConfig = field(default_factory=BankrollConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    parlay: ParlayConfig = field(default_factory=ParlayConfig)
    tomatoes: TomatoesConfig = field(default_factory=TomatoesConfig)
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
    #: Named bundles of overrides, applied with --profile. The shipped
    #: ones in betedge/data/profiles.yaml are a default like the payout
    #: table, so they are present on a bare Config() too; a `profiles:`
    #: key in config.yaml adds to them or replaces one by name.
    profiles: dict[str, dict] = field(default_factory=load_profiles)
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
            parlay=ParlayConfig(**raw.get("parlay", {})),
            tomatoes=TomatoesConfig(**raw.get("tomatoes", {})),
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
            profiles=load_profiles(path),
            database=raw.get("database", "data/betedge.db"),
            reports_dir=raw.get("reports_dir", "reports"),
        )

    def apply_profile(self, name: str, profiles: dict | None = None) -> list[ProfileChange]:
        """
        Apply a named bundle of overrides, returning every change it made.

        The return value is the point. A profile can reach the staking
        fractions and the guard thresholds, and a silent change to either
        is exactly what the rest of this tool refuses to do -- so callers
        print what moved and what it was before, rather than letting a
        scan quietly run under settings nobody stated.

        An unrecognised field is an error naming the valid ones. A typo in
        a profile that was ignored would be indistinguishable from a
        setting that did not work.
        """
        profiles = self.profiles if profiles is None else profiles
        try:
            spec = profiles[name]
        except KeyError as exc:
            raise ValueError(
                f"unknown profile {name!r}. Available: "
                f"{sorted(profiles) or 'none'}. Run `betedge profiles` to see "
                "what each one does, or define your own under `profiles:` in "
                "config.yaml."
            ) from exc

        changes: list[ProfileChange] = []

        for key, value in spec.items():
            if key == "description":
                continue

            if key in _PROFILE_LISTS:
                before = list(getattr(self, key))
                if before != list(value):
                    setattr(self, key, list(value))
                    changes.append(ProfileChange(key, before, list(value)))

            elif key in _PROFILE_MAPPINGS:
                mapping = dict(getattr(self, key))
                for sport, override in (value or {}).items():
                    before = mapping.get(sport)
                    if override is None:
                        # Remove the user's narrowing and fall back to the
                        # registry default, which is what `null` means here.
                        if sport in mapping:
                            mapping.pop(sport)
                            changes.append(
                                ProfileChange(f"{key}.{sport}", before, None)
                            )
                    elif before != override:
                        mapping[sport] = override
                        changes.append(
                            ProfileChange(f"{key}.{sport}", before, override)
                        )
                setattr(self, key, mapping)

            elif key in _PROFILE_SECTIONS:
                section = getattr(self, key)
                valid = {f.name for f in fields(section)}
                for field_name, override in (value or {}).items():
                    if field_name not in valid:
                        raise ValueError(
                            f"profile {name!r} sets {key}.{field_name}, which "
                            f"is not a setting. Valid {key} settings: "
                            f"{sorted(valid)}"
                        )
                    before = getattr(section, field_name)
                    if before != override:
                        setattr(section, field_name, override)
                        changes.append(
                            ProfileChange(f"{key}.{field_name}", before, override)
                        )

            else:
                raise ValueError(
                    f"profile {name!r} sets {key!r}, which a profile cannot "
                    f"change. It may set: description, "
                    f"{', '.join(sorted(_PROFILE_LISTS + _PROFILE_MAPPINGS + _PROFILE_SECTIONS))}."
                )

        return changes

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
