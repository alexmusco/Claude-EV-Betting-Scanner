"""
Reading a Tomatometer off a Rotten Tomatoes film page.

Not a scraper in the usual sense
--------------------------------
The visible markup is useless here: the score slots on a film page are
empty web components that JavaScript fills in, so a CSS selector against
the rendered text is both wrong on a plain fetch and fragile forever
after. Every tutorial you will find writes exactly that scraper, against
class names that stopped existing several redesigns ago.

What the page DOES carry, server-rendered, is the site's own structured
data:

    <script id="media-scorecard-json" data-json="mediaScorecard"
            type="application/json"> { ... } </script>

with, under `overlay`, both Tomatometers and -- the part that matters --
the raw counts behind them:

    criticsAll: likedCount 58, notLikedCount 104, reviewCount 162, score "36"
    criticsTop: likedCount 13, notLikedCount 30,  reviewCount 43,  score "30"

Those counts are the whole model. A score without them is not something
this tool can use, so the parser refuses a page it cannot get them from
rather than returning a percentage and letting the caller guess.

Why this holds up better than selectors
---------------------------------------
It is a data contract rather than a presentation detail, so it survives
restyling. And it is self-validating twice over, which is what turns a
silent wrong answer into a loud failure:

  * the counts must add up -- likedCount + notLikedCount == reviewCount
  * the score computed from those counts must equal the score RT itself
    publishes in the same blob

If RT ever changes how it rounds, or what reviewCount counts, the second
check fails immediately and loudly instead of quietly shifting every
contract this tool prices by a point.

Fetching politely
-----------------
robots.txt allows /m/<slug>; it disallows /m/*/pictures and /search,
neither of which is touched here. Responses are cached on disk so that
re-running a scan does not re-fetch, and the caller is expected to leave
the cache alone for the few minutes a Tomatometer takes to move.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .tomatoes import SCOPE_ALL_CRITICS, SCOPE_TOP_CRITICS, Tomatometer

BASE_URL = "https://www.rottentomatoes.com"

#: Paths robots.txt disallows. Checked before any request so that a
#: mistake upstream cannot turn into a request the file forbids.
DISALLOWED = ("/search", "/critics/self-submission/", "/user/account/")

#: Where the scores live, and which of ours each maps to.
_SCOPE_KEYS = {
    SCOPE_ALL_CRITICS: "criticsAll",
    SCOPE_TOP_CRITICS: "criticsTop",
}

_SCORECARD = re.compile(
    r'<script[^>]*data-json="mediaScorecard"[^>]*>(.*?)</script>',
    re.S | re.I,
)

_DEFAULT_CACHE = Path("data") / "rt-cache"


class ParseError(RuntimeError):
    """The page did not contain a Tomatometer this tool can use."""


@dataclass(frozen=True)
class Scorecard:
    """Both Tomatometers from one page, plus what RT itself displayed."""

    film: str
    slug: str
    scores: dict[str, Tomatometer]
    published: dict[str, int]
    fetched_at: datetime

    def scope(self, scope: str) -> Tomatometer:
        try:
            return self.scores[scope]
        except KeyError:
            raise ParseError(
                f"{self.slug}: no {scope} score on this page"
            ) from None


def parse_scorecard(
    html: str, slug: str = "", now: datetime | None = None
) -> Scorecard:
    """
    Pull both Tomatometers, with counts, out of a film page.

    Raises ParseError rather than guessing. A page this cannot read is a
    page whose contract has changed, and the right response to that is a
    loud failure -- a scanner that silently reports no bets because its
    parser broke is worse than one that stops.
    """
    now = now or datetime.now(timezone.utc)
    match = _SCORECARD.search(html)
    if not match:
        raise ParseError(
            "no mediaScorecard JSON on this page. Either the page is a "
            "challenge or error page rather than a film, or Rotten "
            "Tomatoes has changed where it publishes scores -- check "
            "before trusting anything this tool prints."
        )
    try:
        blob = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ParseError(f"mediaScorecard JSON did not parse: {exc}") from exc

    film = _title(html) or slug
    overlay = blob.get("overlay") or {}
    scores: dict[str, Tomatometer] = {}
    published: dict[str, int] = {}

    for scope, key in _SCOPE_KEYS.items():
        section = overlay.get(key) or {}
        if not section:
            continue
        fresh = section.get("likedCount")
        rotten = section.get("notLikedCount")
        if fresh is None or rotten is None:
            continue
        fresh, rotten = int(fresh), int(rotten)
        counted = fresh + rotten
        if counted == 0:
            # A film with no critic reviews yet. Real, and not an error --
            # but there is nothing to price, so it is left out.
            continue

        reported_total = section.get("reviewCount")
        if reported_total is not None and int(reported_total) != counted:
            # The counts are what everything downstream is built on. If
            # they stop meaning what they mean, stop.
            raise ParseError(
                f"{key}: {fresh} fresh + {rotten} rotten = {counted}, but "
                f"reviewCount says {reported_total}. The count no longer "
                "means what this tool assumes it means."
            )

        snapshot = Tomatometer(
            fresh=fresh, total=counted, captured_at=now,
            scope=scope, film=film,
        )

        reported_score = section.get("score")
        if reported_score not in (None, ""):
            shown = int(reported_score)
            published[scope] = shown
            if shown != snapshot.displayed_score:
                # The self-validating check. Contracts settle on the
                # number RT displays, so if ours disagrees with theirs we
                # are pricing a different question than the market is.
                raise ParseError(
                    f"{key}: {fresh}/{counted} rounds to "
                    f"{snapshot.displayed_score}% but Rotten Tomatoes "
                    f"displays {shown}%. Either the rounding rule or the "
                    "meaning of the counts has changed."
                )
        scores[scope] = snapshot

    if not scores:
        raise ParseError(
            "the scorecard carried no critic counts. A percentage alone "
            "cannot drive this model -- see tomatoes.py."
        )
    return Scorecard(
        film=film, slug=slug, scores=scores, published=published,
        fetched_at=now,
    )


_TITLE = re.compile(r"<title>(.*?)</title>", re.S | re.I)


def _title(html: str) -> str:
    m = _TITLE.search(html)
    if not m:
        return ""
    # "Resident Evil | Rotten Tomatoes"
    return m.group(1).split("|")[0].strip()


#: A film slug is a bare identifier. Anything else -- a slash, a dot
#: segment, a query -- can walk out of /m/ and onto a path robots.txt
#: forbids, so the slug is validated rather than the assembled path:
#: "/m/../search" starts with "/m/" and is still a request for /search.
_SLUG = re.compile(r"^[A-Za-z0-9._-]+$")


def path_for(slug: str) -> str:
    """The film path for a slug, validated against robots.txt."""
    slug = slug.strip().strip("/")
    if slug.startswith("m/"):
        slug = slug[2:]
    slug = slug.strip("/")
    if ".." in slug or not _SLUG.match(slug):
        raise ValueError(
            f"{slug!r} is not a film slug. Anything with a path in it can "
            "reach a URL robots.txt disallows, so it is refused here."
        )
    path = f"/m/{slug}"
    for blocked in DISALLOWED:
        if path.startswith(blocked):
            raise ValueError(f"{path} is disallowed by robots.txt")
    return path


def fetch(
    slug: str,
    cache_dir: Path | str | None = None,
    max_age_seconds: float = 900.0,
    session=None,
    now: datetime | None = None,
    timeout: float = 20.0,
) -> Scorecard:
    """
    Fetch one film page and read its Tomatometers.

    Cached on disk, because a scan across a handful of films re-reads the
    same pages and a Tomatometer does not move minute to minute. The
    cache is the rate limiting: fifteen minutes by default, which is far
    slower than any review actually arrives.
    """
    import requests

    now = now or datetime.now(timezone.utc)
    path = path_for(slug)
    directory = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE
    directory.mkdir(parents=True, exist_ok=True)
    cached = directory / f"{path.strip('/').replace('/', '_')}.html"

    html = None
    if cached.exists() and (time.time() - cached.stat().st_mtime) < max_age_seconds:
        html = cached.read_text(encoding="utf-8", errors="replace")

    if html is None:
        getter = session.get if session is not None else requests.get
        response = getter(
            f"{BASE_URL}{path}",
            headers={
                # Identifiable rather than disguised, and honest about
                # what this is. robots.txt permits /m/<slug>.
                "User-Agent": (
                    "betedge/1.0 (personal contract pricing; "
                    "respects robots.txt)"
                ),
                "Accept": "text/html",
            },
            timeout=timeout,
        )
        response.raise_for_status()
        html = response.text
        cached.write_text(html, encoding="utf-8")

    return parse_scorecard(html, slug=slug, now=now)
