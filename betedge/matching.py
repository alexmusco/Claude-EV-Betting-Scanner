"""
Deciding which Kalshi market is about which game.

Why this is the hard part
-------------------------
Kalshi has no event id in common with the odds feed. Each side names the
same game its own way -- "Denver Broncos @ Kansas City Chiefs" against
"Will the Chiefs beat the Broncos?" -- and the join has to be made from
the words and the clock.

Get it wrong and nothing downstream can save you. A ladder priced against
another game's line produces perfectly valid numbers about the wrong
match: the fit converges, sigma lands in range, every rung is a
probability, and the edge is fiction. There is no guard further along
that can catch it, because nothing further along knows what the words
said. So this module's job is less to match than to REFUSE: a match is
made only when it is unambiguous, and an uncertain one is reported rather
than taken.

How the join is made
--------------------
Not by parsing tickers. Ticker formats are undocumented, vary by series
and change without notice, and a parser built on one is a parser that
breaks silently. The titles are human-readable and far more stable.

The signal is the NICKNAME. Within a league "Chiefs", "Broncos" and
"Dodgers" are unique, and they survive every rendering -- full name,
city-dropped, possessive, headline. City names do not: Los Angeles,
New York and Chicago each field two teams, so a city match alone is
evidence of nothing.

Both teams must be found, they must be different teams, and the start
times must agree to within a few hours. Anything less is not a match.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

#: Cities that field more than one team in the same league, so a city
#: token can never identify a team on its own. Not exhaustive and does
#: not need to be: cities are excluded from matching entirely and this
#: list only documents why.
SHARED_CITIES = (
    "new york", "los angeles", "chicago", "san francisco bay",
    "washington", "florida", "texas", "california",
)

#: Nicknames that collide WITHIN one league, where the sport alone
#: cannot separate them and more of the name is needed.
#:
#: Deliberately short. Almost every nickname collision -- Texas and New
#: York Rangers, the two Giants, the two Cardinals, the two Panthers,
#: both sets of Kings and Jets -- is ACROSS leagues, and the caller
#: scopes its events to one sport. Guarding against those here would
#: demand a city in titles that never carry one ("Bills vs Jets"), which
#: breaks the ordinary case to defend against one that cannot arise.
#: Cross-league confusion is caught instead by requiring BOTH teams to
#: match a single event.
WITHIN_LEAGUE_COLLISIONS = ("sox",)

#: How far apart two start times may be and still be the same game. Wide
#: enough for a timezone slip or a listed-time difference, narrow enough
#: that a rematch a week later cannot collide.
DEFAULT_TIME_TOLERANCE_HOURS = 8.0


class MatchError(RuntimeError):
    """The two sides could not be joined with any confidence."""


def normalise(text: str) -> str:
    """Lowercase, accent-folded, punctuation-stripped."""
    stripped = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(c for c in stripped if not unicodedata.combining(c))
    stripped = stripped.lower().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", stripped).strip()


def nickname(full_name: str) -> str:
    """
    The distinctive part of a team name.

    Usually the last word, but not always -- "White Sox" and "Red Sox"
    need two, and "Sox" alone belongs to both Chicago and Boston. Where
    the trailing word is ambiguous, more of the name is kept.
    """
    words = normalise(full_name).split()
    if not words:
        return ""
    last = words[-1]
    if last in WITHIN_LEAGUE_COLLISIONS and len(words) >= 2:
        return " ".join(words[-2:])
    return last


def team_key(full_name: str) -> str:
    """A team's identity for matching within one league."""
    return nickname(full_name)


def find_team(text: str, full_name: str) -> bool:
    """
    Whether `text` names this team.

    Matches the nickname as a WHOLE WORD. Substring matching would let
    "Jets" find "Jetsons" and, more plausibly, let short abbreviations
    collide with ordinary words -- which is the sort of bug that produces
    a confident match to the wrong game.
    """
    haystack = f" {normalise(text)} "
    nick = nickname(full_name)
    if not nick:
        return False
    return f" {nick} " in haystack


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """One possible join, with everything that argued for and against it."""

    event: dict
    matched_home: bool = False
    matched_away: bool = False
    hours_apart: float | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def both_teams(self) -> bool:
        return self.matched_home and self.matched_away

    @property
    def score(self) -> float:
        """Higher is better. Only used to rank; never to accept."""
        if not self.both_teams:
            return 0.0
        closeness = 1.0
        if self.hours_apart is not None:
            closeness = 1.0 / (1.0 + abs(self.hours_apart))
        return 1.0 + closeness


@dataclass
class MatchResult:
    """What the join decided, and why."""

    event: dict | None
    confident: bool
    reason: str
    candidates: int = 0

    @property
    def event_id(self) -> str | None:
        return (self.event or {}).get("id")


def _start_of(event) -> datetime | None:
    from .db import parse_timestamp

    return parse_timestamp(event.get("commence_time"))


def match_event(
    title: str,
    events,
    close_time: datetime | None = None,
    tolerance_hours: float = DEFAULT_TIME_TOLERANCE_HOURS,
) -> MatchResult:
    """
    Find the odds-feed event a Kalshi title is about.

    Returns a MatchResult whose `confident` flag is the only thing a
    caller should act on. An ambiguous result carries the best guess for
    a human to look at and must not be traded on: a ladder priced against
    the wrong game produces entirely valid-looking numbers, and nothing
    downstream can tell.
    """
    candidates = []
    for event in events:
        home = event.get("home_team") or ""
        away = event.get("away_team") or ""
        if not home or not away:
            continue
        candidate = Candidate(event=event)
        candidate.matched_home = find_team(title, home)
        candidate.matched_away = find_team(title, away)
        if close_time is not None:
            start = _start_of(event)
            if start is not None:
                candidate.hours_apart = (
                    (close_time - start).total_seconds() / 3600.0
                )
        if candidate.both_teams:
            candidates.append(candidate)

    if not candidates:
        return MatchResult(None, False,
                           "no event names both of these teams")

    if close_time is not None:
        timed = [c for c in candidates
                 if c.hours_apart is None
                 or abs(c.hours_apart) <= tolerance_hours]
        if not timed:
            nearest = min(candidates, key=lambda c: abs(c.hours_apart or 0))
            return MatchResult(
                nearest.event, False,
                f"both teams match but the nearest start is "
                f"{abs(nearest.hours_apart):.0f}h away -- probably a "
                "different meeting of the same two teams",
                candidates=len(candidates),
            )
        candidates = timed

    candidates.sort(key=lambda c: c.score, reverse=True)
    if len(candidates) > 1:
        # Two events naming the same pair inside the window is a
        # doubleheader or a duplicated feed entry. Either way, picking
        # one is a coin flip dressed as a decision.
        return MatchResult(
            candidates[0].event, False,
            f"{len(candidates)} events name both teams within "
            f"{tolerance_hours:g}h of each other -- ambiguous",
            candidates=len(candidates),
        )

    best = candidates[0]
    detail = "both teams matched"
    if best.hours_apart is not None:
        detail += f", starts {abs(best.hours_apart):.1f}h from the close"
    return MatchResult(best.event, True, detail, candidates=1)


def match_markets(markets, events, tolerance_hours=DEFAULT_TIME_TOLERANCE_HOURS):
    """
    Join a list of Kalshi markets to odds-feed events.

    Returns (matched, unmatched) where matched is a list of
    (market, MatchResult) for confident joins only. Everything uncertain
    goes to `unmatched` with its reason attached, so that a run reports
    what it could not place instead of quietly covering fewer games than
    the operator thinks.
    """
    matched, unmatched = [], []
    for market in markets:
        title = " ".join(filter(None, [
            getattr(market, "title", ""), getattr(market, "subtitle", ""),
        ]))
        result = match_event(
            title, events, close_time=getattr(market, "close_time", None),
            tolerance_hours=tolerance_hours,
        )
        if result.confident:
            matched.append((market, result))
        else:
            unmatched.append((market, result))
    return matched, unmatched
