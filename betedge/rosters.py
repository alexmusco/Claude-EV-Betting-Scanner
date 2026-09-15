"""
Who plays for whom, and how much that answer can be trusted.

Why this exists
---------------
The Odds API does not say which team a player plays for. It gives a player
name, a stat, a line and a price, and nothing else. But the most valuable
correlation priors in the model are exactly the ones that need the answer:
a quarterback with his OWN receiver is +0.45, two backs in the SAME
backfield are -0.30, a goalie's saves against the OPPOSING shooters is
+0.45. Without a team, every same-game pair collapses to a weak blended
prior and most of the edge the optimizer exists to find goes with it.

So teams come from here, in four layers, cheapest first:

1. **Game logs you already supply.** `parlay correlations --from logs.csv`
   needs a `team` column to bucket its pairs, so the mapping is already in
   your hands -- it was simply being read and thrown away. Fitting
   correlations now maintains your rosters as a side effect, which means
   the thing you have to do anyway is the thing that keeps them current.

2. **A published roster feed**, cached and refreshed on a timer. nflverse
   publishes NFL rosters as an open data release; it is a download, not a
   scrape, and once every few days is not a load on anyone.

3. **Structural inference from the slate itself**, which costs nothing and
   needs no data at all. Some markets have exactly one player per team --
   two starting quarterbacks, two starting pitchers, two goalies -- so
   when a market like that has exactly two players in one game, they are
   necessarily opponents. See `infer_sides`.

4. **Your own CSV**, which overrides everything, because on the morning of
   a trade you know before any feed does.

The rule that matters more than any of them
-------------------------------------------
A stale roster is WORSE than no roster. An unknown team costs you the
weak blended prior; a wrong team puts a confident +0.45 on a pair that is
really -0.10, and nothing downstream will question it. So every entry
carries the date it was true and the source it came from, an entry past
`max_age_days` is not used at all rather than used quietly, and the pair
provenance in the report names the source and the age.

The same reasoning kills a tempting shortcut: asking a language model to
write the roster out from memory. Its answer would be fluent, undated,
and wrong about every transaction since its training cut-off -- which is
the exact failure this module is built to prevent.
"""

from __future__ import annotations

import csv
import io
import logging
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Callable, Iterable, Sequence

log = logging.getLogger(__name__)

SOURCE_MANUAL = "manual"
SOURCE_GAME_LOGS = "game_logs"
SOURCE_STRUCTURAL = "structural"

#: Lower wins. A provider not named here sits between the manual override
#: and the game logs: fresher than a fit you ran last month, but never
#: ahead of a correction you made by hand this morning.
SOURCE_RANK = {SOURCE_MANUAL: 0, SOURCE_GAME_LOGS: 2}
PROVIDER_RANK = 1

#: Name suffixes that appear in one source and not the other.
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalise_name(name: str | None, strip_suffixes: bool = True) -> str:
    """
    A comparison key for a player name.

    Rosters and odds feeds disagree in small, consistent ways: "A.J. Brown"
    against "AJ Brown", "Amon-Ra St. Brown" against "Amon Ra St Brown",
    accents present or absent, a "Jr." on one side only. All of that is
    punctuation and diacritics, so stripping both and folding case matches
    the overwhelming majority without any fuzzy matching -- which is
    deliberate, because a fuzzy match that pairs two different players is
    exactly the confident wrong answer this module exists to avoid.

    `strip_suffixes` is what makes "Odell Beckham Jr." match "Odell
    Beckham". It also collides a father-and-son pair like Michael Carter
    the running back with Michael Carter II the defensive back, so the
    book indexes BOTH spellings and consults the exact one first -- a
    suffix is only discarded when doing so is unambiguous.
    """
    if not name:
        return ""
    folded = unicodedata.normalize("NFKD", str(name))
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in folded)
    parts = cleaned.lower().split()
    while strip_suffixes and len(parts) > 2 and parts[-1] in _SUFFIXES:
        parts.pop()
    # Fuse runs of single letters, so "A.J." and "AJ" land on one key.
    # Splitting on the punctuation leaves the first as ["a", "j"] and the
    # second as ["aj"], and without this they would never match.
    fused: list[str] = []
    for token in parts:
        if len(token) == 1 and fused and len(fused[-1]) <= 2 and fused[-1].isalpha():
            fused[-1] += token
        else:
            fused.append(token)
    return " ".join(fused)


def _as_date(value) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip().replace("Z", "+00:00")
    for parse in (date.fromisoformat, lambda t: datetime.fromisoformat(t).date()):
        try:
            return parse(text)
        except ValueError:
            continue
    return None


@dataclass(frozen=True)
class RosterEntry:
    """One player's team, with the provenance that decides whether to use it."""

    sport: str
    player: str
    team: str
    source: str
    as_of: date | None
    player_key: str = ""
    exact_key: str = ""
    position: str | None = None
    fetched_at: datetime | None = None

    def __post_init__(self):
        if not self.player_key:
            object.__setattr__(self, "player_key", normalise_name(self.player))
        if not self.exact_key:
            object.__setattr__(
                self, "exact_key", normalise_name(self.player, strip_suffixes=False)
            )

    @property
    def rank(self) -> int:
        return SOURCE_RANK.get(self.source, PROVIDER_RANK)

    def age_days(self, today: date | None = None) -> float | None:
        if self.as_of is None:
            return None
        return (( today or datetime.now(timezone.utc).date()) - self.as_of).days

    def stale(self, max_age_days: float, today: date | None = None) -> bool:
        """
        Past its shelf life. An entry with no date at all counts as stale:
        an undated mapping cannot be shown to be current, and the whole
        point of this module is not to pretend otherwise.
        """
        age = self.age_days(today)
        return age is None or age > max_age_days

    def describe(self, today: date | None = None) -> str:
        age = self.age_days(today)
        when = "undated" if age is None else f"{age:.0f}d old"
        return f"{self.team} ({self.source}, {when})"


@dataclass
class RosterBook:
    """
    The resolved view: one team per player, with its receipts.

    Resolution is by source precedence first and recency second, so a hand
    correction beats a feed and a feed beats a fit from last month. Entries
    past `max_age_days` are dropped outright rather than demoted.
    """

    entries: list[RosterEntry] = field(default_factory=list)
    max_age_days: float = 30.0
    today: date | None = None
    #: Normalised names that two teams both claim within one source. Never
    #: resolved, always reported: two different players sharing a key is a
    #: collision, and guessing between them is the confident wrong answer.
    ambiguous: set[tuple[str, str]] = field(default_factory=set)

    def __post_init__(self):
        self.today = self.today or datetime.now(timezone.utc).date()
        self._rejected_stale = 0
        # Two indexes: names exactly as written, and names with a generational
        # suffix removed. The exact one is consulted first, so "Michael
        # Carter" and "Michael Carter II" stay two people while "Odell
        # Beckham Jr." still answers to "Odell Beckham".
        # An entry with no sport is a wildcard, and it is expanded across
        # every league in play BEFORE resolution rather than consulted as a
        # fallback after it. Scoping must not beat precedence: a hand
        # correction written without a sport column is still a correction,
        # and looking it up only when the league-scoped feed had nothing
        # would let the feed quietly win every time.
        self._scopes = sorted({e.sport for e in self.entries if e.sport})
        self._exact = self._index(lambda e: e.exact_key, set(), count_stale=False)
        self._resolved = self._index(lambda e: e.player_key, self.ambiguous)

    def _index(
        self, key_of, ambiguous: set, count_stale: bool = True
    ) -> dict[tuple[str, str], RosterEntry]:
        by_key: dict[tuple[str, str], list[RosterEntry]] = {}
        for e in self.entries:
            k = key_of(e)
            if not k or not e.team:
                continue
            scopes = [e.sport] if e.sport else ["", *self._scopes]
            for scope in scopes:
                by_key.setdefault((scope, k), []).append(e)

        def authority(c: RosterEntry) -> tuple[int, int]:
            # Source precedence first, then how specifically the entry was
            # scoped: a league-scoped line beats the same file's wildcard.
            return (c.rank, 0 if c.sport else 1)

        resolved: dict[tuple[str, str], RosterEntry] = {}
        for key, candidates in by_key.items():
            fresh = [c for c in candidates if not c.stale(self.max_age_days, self.today)]
            if not fresh:
                # Counted on one index only, or every dropped entry would
                # be reported twice in `betedge parlay rosters`.
                if count_stale:
                    self._rejected_stale += 1
                continue
            fresh.sort(
                key=lambda c: (*authority(c), -(c.as_of or date.min).toordinal())
            )
            best = fresh[0]
            tied = [c for c in fresh if authority(c) == authority(best)]
            if len({c.team for c in tied}) > 1:
                # The most authoritative thing available disagrees with
                # itself, which means one name belongs to two players.
                # Guessing between them is the confident wrong answer, so
                # the name stays unknown.
                #
                # Deliberately judged on authority alone and not on dates:
                # whether two rows happen to carry different timestamps says
                # nothing about whether they are the same person.
                ambiguous.add(key)
                continue
            resolved[key] = best
        return resolved

    @classmethod
    def from_mapping(
        cls,
        mapping: dict,
        sport: str = "",
        source: str = SOURCE_MANUAL,
        as_of: date | None = None,
        **kwargs,
    ) -> "RosterBook":
        """A book from a plain {player: team} dict, dated today by default."""
        when = as_of or datetime.now(timezone.utc).date()
        return cls(
            [
                RosterEntry(sport=sport, player=str(p), team=str(t),
                            source=source, as_of=when)
                for p, t in mapping.items()
            ],
            **kwargs,
        )

    # ------------------------------------------------------------- reading

    def lookup(self, sport: str, player: str) -> RosterEntry | None:
        """
        The team for one player, most specific answer first.

        An entry with no sport acts as a wildcard, so a hand-written
        `player,team` file needs no third column to be useful -- almost
        nobody keeps one file spanning four leagues, and demanding the
        column would be friction in front of the override that exists to
        be edited in a hurry. A sport-scoped entry still wins over it.
        """
        exact = normalise_name(player, strip_suffixes=False)
        stripped = normalise_name(player)
        for scope in (sport, ""):
            found = self._exact.get((scope, exact))
            if found is not None:
                return found
            found = self._resolved.get((scope, stripped))
            if found is not None:
                return found
        return None


    def team_for(self, sport: str, player: str) -> str | None:
        found = self.lookup(sport, player)
        return found.team if found else None

    def __len__(self) -> int:
        return len(self._resolved)

    @property
    def rejected_stale(self) -> int:
        return self._rejected_stale

    def sports(self) -> list[str]:
        return sorted({sport for sport, _ in self._resolved})

    def source_counts(self, sport: str | None = None) -> dict[str, int]:
        out: dict[str, int] = {}
        for (s, _key), e in self._resolved.items():
            if sport and s != sport:
                continue
            out[e.source] = out.get(e.source, 0) + 1
        return out

    def oldest_age(self, sport: str, source: str | None = None) -> float | None:
        ages = [
            e.age_days(self.today)
            for (s, _k), e in self._resolved.items()
            if s == sport and (source is None or e.source == source)
        ]
        ages = [a for a in ages if a is not None]
        return max(ages) if ages else None

    def newest_age(self, sport: str, source: str | None = None) -> float | None:
        ages = [
            e.age_days(self.today)
            for (s, _k), e in self._resolved.items()
            if s == sport and (source is None or e.source == source)
        ]
        ages = [a for a in ages if a is not None]
        return min(ages) if ages else None

    def conflicts(self) -> list[tuple[str, str, list[RosterEntry]]]:
        """
        Players two different sources place on different teams.

        Not an error -- the manual override exists precisely to disagree
        with a feed -- but worth showing, because the other reason it
        happens is that a hand-written entry was never cleaned up after
        the player moved back.
        """
        by_key: dict[tuple[str, str], list[RosterEntry]] = {}
        for e in self.entries:
            if e.player_key and e.team:
                by_key.setdefault((e.sport, e.player_key), []).append(e)
        out = []
        for (sport, key), candidates in sorted(by_key.items()):
            if len({c.team for c in candidates}) > 1:
                out.append((sport, key, sorted(candidates, key=lambda c: c.rank)))
        return out

    def coverage(self, sport: str, players: Iterable[str]) -> dict:
        """
        How much of an actual slate this book can answer for.

        The number to watch. A book of nine thousand players that covers
        none of tonight's starters is worth nothing, and the only way to
        know is to ask it about the names the scan actually saw.
        """
        names = sorted({p for p in players if p})
        known = [p for p in names if self.lookup(sport, p)]
        return {
            "players": len(names),
            "known": len(known),
            "rate": (len(known) / len(names)) if names else 0.0,
            "missing": [p for p in names if p not in set(known)],
        }


# --------------------------------------------------------------------------
# Building a book from each source
# --------------------------------------------------------------------------


def from_manual_csv(path, sport: str | None = None) -> list[RosterEntry]:
    """
    The user's own `player,team[,sport,position,as_of]` file, or a YAML
    mapping of names to teams.

    Highest precedence and, by default, dated today: a file you edited by
    hand is a statement about right now. Give an `as_of` column if you want
    the staleness guard to age it properly.
    """
    from pathlib import Path

    import yaml

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no roster file at {p}")
    today = datetime.now(timezone.utc)
    out: list[RosterEntry] = []

    if p.suffix.lower() in (".yaml", ".yml"):
        raw = yaml.safe_load(p.read_text()) or {}
        for name, team in raw.items():
            out.append(RosterEntry(
                sport=sport or "", player=str(name), team=str(team),
                source=SOURCE_MANUAL, as_of=today.date(), fetched_at=today,
            ))
        return out

    with p.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("player") or row.get("name") or "").strip()
            team = (row.get("team") or "").strip()
            if not name or not team:
                continue
            out.append(RosterEntry(
                sport=(row.get("sport") or sport or "").strip(),
                player=name,
                team=team,
                position=(row.get("position") or None),
                source=SOURCE_MANUAL,
                as_of=_as_date(row.get("as_of")) or today.date(),
                fetched_at=today,
            ))
    return out


def from_game_logs(rows: Sequence) -> list[RosterEntry]:
    """
    Harvest player -> team from the box scores already being fitted.

    This is the layer that costs nothing. `correlation.read_game_logs`
    requires a `team` column so its pairs can be bucketed into same-team
    and opposing-team; the mapping was always there and was simply being
    discarded once the bucketing was done.

    A player's team is taken from his LATEST appearance, so a mid-season
    trade corrects itself the next time you refit. Where the logs carry no
    date the run date is used and the source says so, because an undated
    log is a statement about whenever you happened to export it.
    """
    now = datetime.now(timezone.utc)
    latest: dict[tuple[str, str], tuple[date | None, str, str, bool]] = {}
    for r in rows:
        player = getattr(r, "player", None)
        team = getattr(r, "team", None)
        if not player or not team:
            continue
        sport = getattr(r, "sport", "") or ""
        seen = _as_date(getattr(r, "date", None))
        dated = seen is not None
        key = (sport, normalise_name(player))
        previous = latest.get(key)
        if previous is None or (seen or date.min) >= (previous[0] or date.min):
            latest[key] = (seen, player, team, dated)

    out = []
    for (sport, _key), (seen, player, team, dated) in latest.items():
        out.append(RosterEntry(
            sport=sport, player=player, team=team,
            source=SOURCE_GAME_LOGS if dated else f"{SOURCE_GAME_LOGS}(undated)",
            as_of=seen or now.date(), fetched_at=now,
        ))
    return out


# --------------------------------------------------------------------------
# Published feeds
# --------------------------------------------------------------------------

NFLVERSE_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "rosters/roster_{season}.csv"
)

#: A player these statuses no longer plays for that team. Everything else --
#: active, reserve, practice squad -- is on the club and correctly mapped,
#: and a player on injured reserve will not have a prop posted anyway.
NFLVERSE_DEPARTED = {"CUT", "TRD", "RFA", "UFA", "EXE"}

USER_AGENT = "betedge/0.1 (+https://github.com/alexmusco/Claude-EV-Betting-Scanner)"


def nfl_season_for(today: date | None = None) -> int:
    """
    The season a date belongs to. The league year turns over in March, so
    January's games belong to the previous season's file.
    """
    today = today or datetime.now(timezone.utc).date()
    return today.year if today.month >= 3 else today.year - 1


def _http_get(url: str, timeout: float = 30.0) -> str:
    import requests

    response = requests.get(
        url, timeout=timeout, headers={"User-Agent": USER_AGENT}
    )
    response.raise_for_status()
    return response.text


def nflverse_rosters(
    season: int | None = None,
    today: date | None = None,
    opener: Callable[[str], str] | None = None,
    sport: str = "americanfootball_nfl",
    now: datetime | None = None,
) -> list[RosterEntry]:
    """
    NFL rosters from nflverse's published data release.

    A file download from a data project that publishes releases for this
    purpose, not a scrape of anybody's website, and fetched at most once
    every few days. The file carries `full_name`, `team`, `position` and a
    `status` that says whether the player is still on the club.

    Team codes are nflverse's own abbreviations, and deliberately not
    translated into The Odds API's full team names. Nothing downstream
    needs the name: the correlation matrix only ever asks whether two legs
    share a team, so two codes from one source compare perfectly well.

    `opener` exists so the tests can exercise the parsing against a fixture
    without touching the network.
    """
    now = now or datetime.now(timezone.utc)
    today = today or now.date()
    fetch = opener or _http_get
    seasons = [season] if season else [nfl_season_for(today), nfl_season_for(today) - 1]

    last_error: Exception | None = None
    for candidate in seasons:
        url = NFLVERSE_URL.format(season=candidate)
        try:
            text = fetch(url)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            log.debug("nflverse %s unavailable: %s", candidate, exc)
            continue
        entries = _parse_nflverse(text, sport=sport, today=today, now=now)
        if entries:
            log.info("nflverse: %d players for season %s", len(entries), candidate)
            return entries
    if last_error is not None:
        raise RuntimeError(
            f"could not fetch nflverse rosters ({last_error}). The scan will "
            "carry on with whatever is cached."
        ) from last_error
    return []


def _parse_nflverse(
    text: str, sport: str, today: date, now: datetime | None = None
) -> list[RosterEntry]:
    now = now or datetime.now(timezone.utc)
    out: list[RosterEntry] = []
    for row in csv.DictReader(io.StringIO(text)):
        name = (row.get("full_name") or "").strip()
        team = (row.get("team") or "").strip()
        status = (row.get("status") or "").strip().upper()
        if not name or not team or status in NFLVERSE_DEPARTED:
            continue
        out.append(RosterEntry(
            sport=sport,
            player=name,
            team=team,
            position=(row.get("depth_chart_position")
                      or row.get("position") or None),
            source="nflverse",
            # The release is rebuilt continuously, so what it says is true
            # as of the moment it was fetched.
            as_of=today,
            fetched_at=now,
        ))
    return out


#: Which feed serves which sport. Only leagues with a verified, openly
#: published roster release belong here -- an unverified source would hand
#: the model a confident team it has no right to.
PROVIDERS: dict[str, Callable[..., list[RosterEntry]]] = {
    "americanfootball_nfl": nflverse_rosters,
}


def provider_name(sport: str) -> str | None:
    return "nflverse" if sport in PROVIDERS else None


def fetch_provider(
    sport: str,
    today: date | None = None,
    opener: Callable[[str], str] | None = None,
    now: datetime | None = None,
) -> list[RosterEntry]:
    fetcher = PROVIDERS.get(sport)
    if fetcher is None:
        return []
    return fetcher(today=today, opener=opener, sport=sport, now=now)


# --------------------------------------------------------------------------
# Structural inference
# --------------------------------------------------------------------------

#: Markets with exactly one player per team. When one game prices exactly
#: two players in one of these, they are opponents -- no roster, no feed,
#: no assumption. A team has one starting quarterback, one starting
#: pitcher and one goalie in net.
ONE_PER_TEAM_MARKETS = {
    "player_pass_yds",
    "player_pass_tds",
    "player_pass_attempts",
    "player_pass_completions",
    "player_pass_interceptions",
    "pitcher_strikeouts",
    "pitcher_outs",
    "pitcher_hits_allowed",
    "pitcher_earned_runs",
    "player_total_saves",
}


def infer_sides(legs: Sequence) -> dict[str, str]:
    """
    Opponents that can be identified from the slate alone.

    Returns player name -> a synthetic side label, for the players a
    one-per-team market places on opposite sides of one game. The labels
    are event-scoped and meaningless outside it, which is exactly what
    they should be: they say "these two are on opposite sides of this
    game" and claim nothing else.

    Only fires where NEITHER player already has a real team, and the label
    carries its own source so `correlation.relation_between` refuses to
    compare it against a roster team. Mixing the two label spaces would
    let a synthetic side read as "different team" from a real one, which
    is a wrong answer dressed as a confident one.
    """
    by_event: dict[tuple[str, str], set[str]] = {}
    for leg in legs:
        market = getattr(leg, "market", None)
        if market not in ONE_PER_TEAM_MARKETS:
            continue
        if getattr(leg, "team", None):
            continue
        by_event.setdefault(
            (getattr(leg, "event_id", ""), market), set()
        ).add(getattr(leg, "selection", ""))

    sides: dict[str, str] = {}
    for (event_id, _market), players in by_event.items():
        named = sorted(p for p in players if p)
        if len(named) != 2:
            # One player means nobody to oppose; three or more means the
            # market is not one-per-team after all on this slate, and the
            # inference does not hold.
            continue
        for label, player in zip(("A", "B"), named):
            sides.setdefault(player, f"{event_id}#{label}")
    return sides


# --------------------------------------------------------------------------
# Assembling the book
# --------------------------------------------------------------------------


@dataclass
class RefreshReport:
    """What a refresh attempt did, so a scan can say so without stopping."""

    refreshed: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def touched(self) -> bool:
        return bool(self.refreshed or self.errors)


def refresh_providers(
    db,
    sports: Sequence[str],
    refresh_days: float = 3.0,
    force: bool = False,
    now: datetime | None = None,
    opener: Callable[[str], str] | None = None,
) -> RefreshReport:
    """
    Pull any roster feed whose cached snapshot has gone cold.

    Never raises. A scan that cannot reach a roster feed should carry on
    with what it has and say what happened -- the feed is an enrichment,
    and losing it costs the sharp priors, not the run.
    """
    now = now or datetime.now(timezone.utc)
    report = RefreshReport()
    freshness = db.roster_freshness()

    for sport in sports:
        provider = provider_name(sport)
        if provider is None:
            report.skipped[sport] = "no roster feed for this sport"
            continue
        cached = freshness.get((sport, provider))
        if not force and cached and cached["fetched_at"] is not None:
            age_hours = (now - cached["fetched_at"]).total_seconds() / 3600.0
            if age_hours < refresh_days * 24:
                report.skipped[sport] = (
                    f"{provider} snapshot is {age_hours:.0f}h old, "
                    f"under the {refresh_days:g}-day refresh interval"
                )
                continue
        try:
            entries = fetch_provider(
                sport, today=now.date(), opener=opener, now=now
            )
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"{sport}: {exc}")
            continue
        if not entries:
            report.errors.append(f"{sport}: {provider} returned no players")
            continue
        db.replace_rosters(sport, provider, entries)
        report.refreshed[sport] = len(entries)
    return report


def load_book(
    db=None,
    manual_path: str | None = None,
    max_age_days: float = 30.0,
    now: datetime | None = None,
    sport: str | None = None,
) -> RosterBook:
    """
    Every stored layer plus the manual override, resolved into one view.

    The manual file is read fresh each time rather than cached, so editing
    it takes effect on the next run with nothing to refresh.
    """
    now = now or datetime.now(timezone.utc)
    entries: list[RosterEntry] = []
    if db is not None:
        for row in db.roster_rows(sport):
            entries.append(RosterEntry(
                sport=row["sport"],
                player=row["player"],
                team=row["team"],
                source=row["source"],
                as_of=_as_date(row["as_of"]),
                player_key=row["player_key"],
                exact_key=row["exact_key"],
                position=row["position"],
            ))
    if manual_path:
        entries.extend(from_manual_csv(manual_path, sport=sport))
    return RosterBook(entries, max_age_days=max_age_days, today=now.date())
