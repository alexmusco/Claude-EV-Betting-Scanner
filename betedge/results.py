"""
Settling a bet from what actually happened.

Why this exists
---------------
A ledger you fill in by hand measures YOU, not the model. You log what
you bet, you bet what you liked, so the ROI that comes out has your
judgement layered on top of the model's and no way to separate them. A
paper ledger of every flagged bet, settled without your involvement, is
the honest instrument -- and it reaches a thousand settled bets in one
season rather than five.

That only works if settlement is automatic, which means a results feed.
nflverse publishes weekly player statistics for every NFL player, free,
no key, from the same project as the game file the margin model is
calibrated against.

The thing that must not happen
------------------------------
A bet settled against the WRONG PLAYER is worse than a bet left
unsettled. An unsettled bet makes the sample smaller and says so; a
mis-settled one silently corrupts the single number this entire exercise
exists to produce, looks completely plausible, and can never be found
again from the numbers themselves.

So the matcher refuses. Two players normalising to one name, a name that
matches nobody, a player who did not appear -- each returns a reason
rather than a verdict, and the caller reports how many it could not
settle. Every function here would rather answer "I don't know" than
guess.

Not playing is not losing
-------------------------
A player who was inactive did not lose his Over 3.5 receptions -- the
bet voids. Scoring DNPs as losses would drag every measured ROI down by
however often players sit, which is exactly the sort of quiet bias that
makes a tool confidently wrong.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

#: nflverse weekly player statistics, one row per player per game.
WEEKLY_STATS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{season}.csv"
)

WON, LOST, PUSH, VOID = "won", "lost", "push", "void"

#: Our market keys against the nflverse columns that settle them.
#:
#: A tuple sums its columns, which is how a combined market settles --
#: "pass + rush + reception yards" is three columns added, not a column
#: named after the bet.
STAT_COLUMNS: dict[str, tuple[str, ...]] = {
    "player_pass_yds": ("passing_yards",),
    "player_pass_tds": ("passing_tds",),
    "player_pass_completions": ("completions",),
    "player_pass_attempts": ("attempts",),
    "player_pass_interceptions": ("passing_interceptions",),
    "player_rush_yds": ("rushing_yards",),
    "player_rush_attempts": ("carries",),
    "player_receptions": ("receptions",),
    "player_reception_yds": ("receiving_yards",),
    "player_pass_rush_reception_yds": (
        "passing_yards", "rushing_yards", "receiving_yards"),
    "player_tackles_assists": ("def_tackles_solo", "def_tackle_assists"),
}

#: Touchdown markets settle on a COUNT of touchdowns, then a threshold.
#: Special-teams scores count: the market is "anytime touchdown", not
#: "anytime touchdown from scrimmage".
TD_COLUMNS = ("rushing_tds", "receiving_tds", "special_teams_tds")

#: Suffixes that are part of a name on one feed and absent on the other.
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


#: nflverse team abbreviations against the full names an odds feed uses.
#:
#: A lookup table, not a fact worth deriving: the stats feed identifies a
#: club as "NO" and the odds feed as "New Orleans Saints", and nothing in
#: either string tells you they are the same. Without it the team filter
#: silently matches nothing, which is how one player appearing in two
#: weeks came to be reported as two players with the same name.
TEAM_ABBREVIATIONS = {
    "arizona cardinals": "ARI", "atlanta falcons": "ATL",
    "baltimore ravens": "BAL", "buffalo bills": "BUF",
    "carolina panthers": "CAR", "chicago bears": "CHI",
    "cincinnati bengals": "CIN", "cleveland browns": "CLE",
    "dallas cowboys": "DAL", "denver broncos": "DEN",
    "detroit lions": "DET", "green bay packers": "GB",
    "houston texans": "HOU", "indianapolis colts": "IND",
    "jacksonville jaguars": "JAX", "kansas city chiefs": "KC",
    "las vegas raiders": "LV", "los angeles chargers": "LAC",
    "los angeles rams": "LA", "miami dolphins": "MIA",
    "minnesota vikings": "MIN", "new england patriots": "NE",
    "new orleans saints": "NO", "new york giants": "NYG",
    "new york jets": "NYJ", "philadelphia eagles": "PHI",
    "pittsburgh steelers": "PIT", "san francisco 49ers": "SF",
    "seattle seahawks": "SEA", "tampa bay buccaneers": "TB",
    "tennessee titans": "TEN", "washington commanders": "WAS",
}


def abbreviate(team: str) -> str:
    """
    A club's stats-feed abbreviation, or "" when it is not known.

    Returns empty rather than a guess: an unknown club narrows nothing,
    and narrowing by a wrong abbreviation would settle a bet against the
    wrong game, which is the one outcome this module exists to prevent.
    """
    return TEAM_ABBREVIATIONS.get(" ".join(str(team or "").lower().split()), "")


def week_from_game_id(game_id: str):
    """
    The week encoded in an nflverse game id, e.g. "2026_03_PIT_DET" -> 3.

    Taken from the id because this feed carries no date column at all --
    a fact worth stating, since the obvious approach is to map a kickoff
    date onto a week and there is nothing here to map it with.
    """
    parts = str(game_id or "").split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


class ResultsError(RuntimeError):
    """The results feed could not be read."""


def normalise_name(name: str) -> str:
    """
    A player's name in the one spelling both feeds can agree on.

    Accents folded, punctuation dropped, suffixes removed. "Oronde
    Gadsden II" and "Oronde Gadsden" are one player; so are "Amon-Ra
    St. Brown" and "Amon Ra St Brown". Dropping the suffix does risk
    merging a father and son on the same roster, which is why a name
    matching two players is refused rather than resolved.
    """
    folded = unicodedata.normalize("NFKD", name or "")
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", folded.lower())
    words = [w for w in cleaned.split() if w not in SUFFIXES]
    return " ".join(words)


@dataclass(frozen=True)
class PlayerGame:
    """One player's line in one game."""

    player: str
    team: str
    opponent: str
    season: int
    week: int
    game_id: str
    stats: dict

    def total(self, columns) -> float | None:
        """The stat a market settles on, or None if the feed lacks it."""
        found = False
        out = 0.0
        for column in columns:
            raw = self.stats.get(column)
            if raw in (None, ""):
                continue
            try:
                out += float(raw)
            except (TypeError, ValueError):
                return None
            found = True
        return out if found else None


@dataclass(frozen=True)
class Settlement:
    """A verdict, or a refusal with its reason. Never a guess."""

    status: str | None
    actual: float | None = None
    reason: str = ""
    player: str = ""

    @property
    def settled(self) -> bool:
        return self.status is not None


class ResultBook:
    """Every player-game in a season, indexed for matching."""

    def __init__(self, games):
        self.games = list(games)
        self._by_name: dict[str, list[PlayerGame]] = {}
        for game in self.games:
            self._by_name.setdefault(
                normalise_name(game.player), []).append(game)

    def __len__(self) -> int:
        return len(self.games)

    def week_rows(self) -> dict:
        """How many player-games the feed holds for each week."""
        counts: dict[int, int] = {}
        for game in self.games:
            counts[game.week] = counts.get(game.week, 0) + 1
        return dict(sorted(counts.items()))

    def incomplete_weeks(self, fraction: float = 0.5) -> list[tuple]:
        """
        Weeks holding far fewer rows than a full one, with what they hold.

        nflverse publishes on a lag of a day or two, so a week whose
        games finished on Sunday can be a fraction populated on Monday.
        Without this, every bet in that week refuses with "does not
        appear" -- which reads exactly like a broken matcher and is
        nothing of the sort. The difference between "we cannot find this
        player" and "the feed has not caught up" is the difference
        between a bug hunt and waiting a day.
        """
        counts = self.week_rows()
        if len(counts) < 2:
            return []
        full = sorted(counts.values())[len(counts) // 2]   # median week
        if full <= 0:
            return []
        return [(week, n, full) for week, n in counts.items()
                if n < full * fraction]

    def find(self, player: str, week: int | None = None,
             teams=()) -> tuple[list[PlayerGame], str]:
        """
        Every appearance matching this name, narrowed by week and teams.

        Returns (candidates, reason). An empty list always carries a
        reason, because "no result" and "could not look" have to be told
        apart by the caller -- one shrinks the sample honestly and the
        other is a bug.
        """
        key = normalise_name(player)
        if not key:
            return [], "no player name on the bet"
        found = self._by_name.get(key, [])
        if not found:
            return [], f"no player called {player!r} in the results feed"

        if week is not None:
            in_week = [g for g in found if g.week == week]
            if not in_week:
                thin = {w for w, _n, _full in self.incomplete_weeks()}
                if week in thin:
                    # Say which it is. "Not found" and "not published
                    # yet" look identical from here and lead somewhere
                    # completely different.
                    return [], (f"week {week} is only partly published "
                                "-- the feed has not caught up, try again "
                                "tomorrow")
                return [], (f"{player} does not appear in week {week} "
                            "-- inactive, or the week is wrong")
            found = in_week

        if teams:
            # Translate before comparing. Full names never equal
            # abbreviations, so an untranslated filter narrows nothing
            # and quietly leaves every week of the season in play.
            wanted = {a for a in (abbreviate(t) for t in teams if t) if a}
            wanted |= {str(t).upper() for t in teams
                       if t and len(str(t)) <= 3}
            # TWO narrowings, and they answer different questions.
            #
            # Either slot picks the right PLAYER: two men sharing a name
            # are told apart by which club they play for, and a bet only
            # ever names one of the two clubs on the field.
            #
            # Both slots pick the right GAME: a player's own team appears
            # in every one of his seventeen fixtures, so matching one
            # slot narrows nothing at all -- it is the PAIR that names a
            # meeting, and two clubs meet at most twice a season.
            #
            # Applying only the second broke the first; applying only the
            # first left one player in every week he played. In order,
            # each does the job the other cannot.
            if wanted:
                by_club = [g for g in found
                           if g.team.upper() in wanted
                           or g.opponent.upper() in wanted]
                if by_club:
                    found = by_club
                by_fixture = [g for g in found
                              if g.team.upper() in wanted
                              and g.opponent.upper() in wanted]
                if by_fixture:
                    found = by_fixture
        return found, ""


def read_weekly_stats(path) -> ResultBook:
    """Parse an nflverse weekly player-stats CSV."""
    path = Path(path)
    if not path.exists():
        raise ResultsError(f"{path} does not exist")
    games = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("player_display_name")
                    or row.get("player_name") or "").strip()
            if not name:
                continue
            try:
                season = int(float(row.get("season") or 0))
                week = int(float(row.get("week") or 0))
            except (TypeError, ValueError):
                continue
            games.append(PlayerGame(
                player=name,
                team=(row.get("team") or "").strip(),
                opponent=(row.get("opponent_team") or "").strip(),
                season=season, week=week,
                game_id=(row.get("game_id") or "").strip(),
                stats=row,
            ))
    if not games:
        raise ResultsError(
            f"{path} parsed to zero player-games. The feed's shape has "
            "changed, or the file is not what this expects."
        )
    return ResultBook(games)


def download_weekly_stats(season: int, path, session=None) -> Path:
    """Fetch one season's weekly player stats to `path`."""
    url = WEEKLY_STATS_URL.format(season=season)
    if session is not None:
        response = session.get(url, timeout=120)
    else:
        import requests

        response = requests.get(url, timeout=120)
    response.raise_for_status()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return path


def settle_over_under(actual: float, side: str, line: float) -> str:
    """
    Over/Under against a number, with the push handled explicitly.

    An integer line can land exactly on it, and a push is neither a win
    nor a loss. Folding it into either one biases every measured ROI in
    the direction it was folded.
    """
    if actual == line:
        return PUSH
    over = actual > line
    return WON if (side.lower() == "over") == over else LOST


def settle_bet(book: ResultBook, market: str, selection: str, side: str,
               line, week: int | None = None, teams=()) -> Settlement:
    """
    Settle one bet, or explain why it cannot be.

    Refusing is a first-class outcome here. A bet settled against the
    wrong player is worse than one left unsettled: the unsettled one
    makes the sample smaller and says so, while the mis-settled one
    corrupts the measurement invisibly and permanently.
    """
    candidates, reason = book.find(selection, week=week, teams=teams)
    if reason:
        return Settlement(None, reason=reason)
    if len(candidates) > 1:
        # Two different ambiguities, and saying the wrong one sends the
        # reader hunting a name collision that does not exist. One player
        # appearing in several weeks is a NARROWING failure; two players
        # sharing a name is a genuine collision.
        weeks = sorted({g.week for g in candidates})
        teams_seen = sorted({g.team.upper() for g in candidates})
        if len(teams_seen) == 1:
            return Settlement(
                None, player=candidates[0].player,
                reason=(f"{selection} appears in weeks "
                        f"{', '.join(str(w) for w in weeks)} and the game "
                        "was not narrowed to one -- no week, and the teams "
                        "did not match the feed"))
        return Settlement(
            None, reason=(f"{len(candidates)} different players match "
                          f"{selection!r} ({', '.join(teams_seen)}) "
                          "-- ambiguous"))

    game = candidates[0]
    base = market.replace("_alternate", "")

    if base == "player_anytime_td":
        # A Yes/No market, so there is no line to compare against.
        scored = game.total(TD_COLUMNS)
        if scored is None:
            return Settlement(None, player=game.player,
                              reason="no touchdown columns in the feed")
        yes = scored >= 1
        want_yes = (side or "").strip().lower() in ("yes", "over", "")
        return Settlement(WON if yes == want_yes else LOST,
                          actual=scored, player=game.player)

    columns = STAT_COLUMNS.get(base)
    if not columns:
        return Settlement(
            None, player=game.player,
            reason=f"no results column is mapped for market {market!r}")
    if line is None:
        return Settlement(None, player=game.player,
                          reason="the bet has no line to settle against")

    actual = game.total(columns)
    if actual is None:
        return Settlement(
            None, player=game.player,
            reason=f"the feed has no {'/'.join(columns)} for {game.player}")
    return Settlement(settle_over_under(actual, side, float(line)),
                      actual=actual, player=game.player)
