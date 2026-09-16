"""
Finding Rotten Tomatoes contracts on Kalshi, and filling in the file.

What this automates, and what it deliberately does not
------------------------------------------------------
It automates the typing: finding the markets, reading their thresholds
out of the tickers and titles, guessing the Rotten Tomatoes slug and
CHECKING that guess against the real page, and writing a contracts file.

It does not automate the verification, and the distinction is the whole
design. Two of the four things that decide a contract -- whether the
threshold itself wins, and which Tomatometer settles it -- live in the
exchange's rules text rather than in any structured field. A parser that
inferred them from a title would be guessing at precisely the moment
guessing is worst: silently, on the number the bet turns on.

So every discovered contract is written `verified: false`, and the rules
text is written INTO the file beside it. Verifying becomes reading a
paragraph in a local file instead of browsing the site, which is the
actual saving. What cannot be read off the rules stays yours to decide.

The slug check
--------------
Rotten Tomatoes slugs are mostly derivable from a title -- "Resident
Evil" becomes resident_evil -- but not reliably: remakes and common
titles carry a year or a disambiguating suffix. So a candidate is not
trusted, it is FETCHED, and accepted only if the page comes back with a
matching title. A wrong slug would price one film's contract against
another film's score, which no downstream guard could catch, because
every number involved would be perfectly valid.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .tomatoes import ABOVE, BELOW, SCOPE_ALL_CRITICS, SCOPE_TOP_CRITICS

#: Words that mark a market as being about a Tomatometer at all. Matched
#: against title, subtitle and rules together.
MARKERS = ("rotten tomatoes", "tomatometer", "rt score", "critic score")

#: Phrases that decide the direction. Order matters: the longer, more
#: specific forms are tested first so that "or above" is not read as
#: "above" with the wrong inclusivity.
_DIRECTION_PATTERNS = [
    (r"\bat or above\b", ABOVE, True),
    (r"\bor above\b", ABOVE, True),
    (r"\bor higher\b", ABOVE, True),
    (r"\bor more\b", ABOVE, True),
    (r"\bgreater than or equal to\b", ABOVE, True),
    (r"\bat least\b", ABOVE, True),
    (r"\bat or below\b", BELOW, True),
    (r"\bor below\b", BELOW, True),
    (r"\bor lower\b", BELOW, True),
    (r"\bor less\b", BELOW, True),
    (r"\bless than or equal to\b", BELOW, True),
    (r"\bat most\b", BELOW, True),
    (r"\bstrictly above\b", ABOVE, False),
    (r"\bstrictly below\b", BELOW, False),
    (r"\bhigher than\b", ABOVE, False),
    (r"\blower than\b", BELOW, False),
    (r"\bgreater than\b", ABOVE, False),
    (r"\bless than\b", BELOW, False),
    (r"\babove\b", ABOVE, None),
    (r"\bbelow\b", BELOW, None),
    (r"\bunder\b", BELOW, None),
    (r"\bover\b", ABOVE, None),
]

_PERCENT = re.compile(r"(\d{1,3})\s*%")
_RANGE = re.compile(r"(\d{1,3})\s*%?\s*(?:-|to|–|—)\s*(\d{1,3})\s*%")
_TICKER_THRESHOLD = re.compile(r"-[TB](\d{1,3})$", re.I)


@dataclass
class Proposal:
    """One discovered market, and everything not yet known about it."""

    ticker: str
    film: str
    title: str
    rules: str
    threshold: int | None = None
    direction: str = ABOVE
    #: None means the rules did not say. It is written into the file as a
    #: question rather than a value, because a wrong guess here is a whole
    #: contract and the guess would be invisible.
    inclusive: bool | None = None
    scope: str | None = None
    slug: str = ""
    slug_confirmed: bool = False
    close_time: datetime | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """Whether there is enough here to write a row worth checking."""
        return self.threshold is not None and self.slug_confirmed

    def describe(self) -> str:
        if self.threshold is None:
            return f"{self.film}: threshold unknown"
        word = "at or above" if self.inclusive else "above"
        if self.direction == BELOW:
            word = "at or below" if self.inclusive else "below"
        if self.inclusive is None:
            word = f"{'above' if self.direction == ABOVE else 'below'}(?)"
        return f"{self.film} {word} {self.threshold}%"


def looks_like_rt_market(*texts: str) -> bool:
    blob = " ".join(t or "" for t in texts).lower()
    return any(marker in blob for marker in MARKERS)


def parse_threshold(
    title: str, subtitle: str = "", ticker: str = "", rules: str = ""
) -> tuple[int | None, str, bool | None, list[str]]:
    """
    Read a threshold, a direction and -- if it is stated -- inclusivity.

    Returns (threshold, direction, inclusive, problems). `inclusive` is
    None when nothing in the text settles it, and that None is carried all
    the way into the file rather than being defaulted, because the default
    would be invisible and wrong half the time.
    """
    problems: list[str] = []
    blob = " ".join(t for t in (subtitle, title, rules) if t)
    lowered = blob.lower()

    # A banded market ("60% to 69%") is a different shape from a
    # threshold and this tool cannot price it. Say so rather than
    # grabbing one of the two numbers.
    band = _RANGE.search(lowered)
    if band and "or above" not in lowered and "or below" not in lowered:
        problems.append(
            f"looks like a band ({band.group(1)}-{band.group(2)}%), not a "
            "threshold -- this tool prices above/below contracts only"
        )
        return None, ABOVE, None, problems

    direction, inclusive = ABOVE, None
    for pattern, this_direction, this_inclusive in _DIRECTION_PATTERNS:
        if re.search(pattern, lowered):
            direction, inclusive = this_direction, this_inclusive
            break
    else:
        problems.append("no direction word found; assuming 'above'")

    threshold = None
    percent = _PERCENT.search(subtitle) or _PERCENT.search(title)
    if percent:
        threshold = int(percent.group(1))
    else:
        from_ticker = _TICKER_THRESHOLD.search(ticker or "")
        if from_ticker:
            threshold = int(from_ticker.group(1))
            problems.append("threshold read from the ticker, not the title")

    if threshold is None:
        problems.append("no threshold found in the title, subtitle or ticker")
    elif not 0 <= threshold <= 100:
        problems.append(f"threshold {threshold} is not a percentage")
        threshold = None

    if inclusive is None and threshold is not None:
        problems.append(
            "the text does not say whether the threshold itself wins -- "
            "read the rules and set `inclusive`"
        )
    return threshold, direction, inclusive, problems


def parse_scope(*texts: str) -> str | None:
    """
    Which Tomatometer, if the text says.

    None when it does not, and None means the contract is not scorable
    until a person decides. All Critics and Top Critics were six points
    apart on Resident Evil the day this was written.
    """
    blob = " ".join(t or "" for t in texts).lower()
    if "top critic" in blob:
        return SCOPE_TOP_CRITICS
    if "all critic" in blob or "all-critic" in blob:
        return SCOPE_ALL_CRITICS
    return None


def slug_candidates(film: str, close_time: datetime | None = None) -> list[str]:
    """
    Plausible Rotten Tomatoes slugs for a film title, best guess first.

    Not trusted -- `confirm_slug` fetches them. This only decides what
    order to try.
    """
    stripped = unicodedata.normalize("NFKD", film or "")
    stripped = "".join(c for c in stripped if not unicodedata.combining(c))
    base = re.sub(r"[^a-z0-9]+", "_", stripped.lower()).strip("_")
    if not base:
        return []
    candidates = [base]
    # Remakes and common titles carry a year.
    years = {str(y) for y in (
        (close_time.year if close_time else None),
        (close_time.year - 1 if close_time else None),
    ) if y}
    for year in sorted(years):
        candidates.append(f"{base}_{year}")
    # "The Thing" is sometimes "the_thing_2011"; also try without a
    # leading article, which RT sometimes drops.
    without_article = re.sub(r"^(the|a|an)_", "", base)
    if without_article != base:
        candidates.append(without_article)
    return candidates


def _comparable(text: str) -> str:
    """
    A title reduced to letters and digits, for comparison only.

    A trailing year is dropped because Rotten Tomatoes disambiguates
    remakes that way -- "The Thing (2011)" is the same film Kalshi calls
    "The Thing".
    """
    stripped = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(c for c in stripped if not unicodedata.combining(c))
    stripped = re.sub(r"[\(\[]?\b(19|20)\d{2}\b[\)\]]?\s*$", "",
                      stripped.strip())
    return re.sub(r"[^a-z0-9]+", "", stripped.lower())


def confirm_slug(
    film: str, candidates, fetcher, max_tries: int = 4
) -> tuple[str, bool, list[str]]:
    """
    Fetch candidate slugs until one comes back as the right film.

    A wrong slug prices one film's contract against another film's score,
    and no guard downstream could catch it -- every number involved would
    be perfectly valid. So the slug is verified here or not used.
    """
    problems: list[str] = []
    wanted = _comparable(film)
    for slug in list(candidates)[:max_tries]:
        try:
            card = fetcher(slug)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"/m/{slug}: {exc}")
            continue
        got = _comparable(card.film)
        # EXACT, not a substring. "Resident Evil" is a substring of
        # "Resident Evil: Afterlife", and accepting that would price one
        # film's contract against a sequel's score -- the one failure no
        # guard downstream could catch, because every number involved
        # would be perfectly valid.
        if got and got == wanted:
            return slug, True, problems
        problems.append(
            f"/m/{slug} is '{card.film}', not '{film}' -- not used"
        )
    return "", False, problems


def discover(
    client,
    fetcher,
    series_ticker: str | None = None,
    status: str | None = "open",
    limit_markets: int = 400,
    now: datetime | None = None,
) -> list[Proposal]:
    """
    Find Rotten Tomatoes markets on Kalshi and propose contract rows.

    `client` is a kalshi.MarketData, `fetcher` takes a slug and returns a
    Scorecard. Both injected so this is testable without a network, which
    matters because the shape of Kalshi's RT markets is not yet confirmed
    against a live response.
    """
    now = now or datetime.now(timezone.utc)
    markets = client.markets(series_ticker=series_ticker, status=status,
                             limit=min(limit_markets, 1000))
    proposals: list[Proposal] = []
    slug_cache: dict[str, tuple[str, bool, list[str]]] = {}

    for market in markets:
        if not looks_like_rt_market(market.title, market.subtitle,
                                    market.rules, market.ticker):
            continue
        film = film_from(market)
        threshold, direction, inclusive, problems = parse_threshold(
            market.title, market.subtitle, market.ticker, market.rules
        )
        scope = parse_scope(market.rules, market.title, market.subtitle)
        if scope is None:
            problems.append(
                "the rules do not say which Tomatometer (All Critics or "
                "Top Critics) -- they were six points apart on Resident "
                "Evil; set `scope` yourself"
            )

        if film not in slug_cache:
            slug_cache[film] = confirm_slug(
                film, slug_candidates(film, market.close_time), fetcher
            )
        slug, confirmed, slug_problems = slug_cache[film]
        problems.extend(slug_problems)
        if not confirmed:
            problems.append(
                f"no Rotten Tomatoes page confirmed for '{film}' -- fill in "
                "`slug` by hand from the film's URL"
            )

        proposals.append(Proposal(
            ticker=market.ticker, film=film, title=market.title,
            rules=market.rules, threshold=threshold, direction=direction,
            inclusive=inclusive, scope=scope, slug=slug,
            slug_confirmed=confirmed, close_time=market.close_time,
            problems=problems,
        ))
    return proposals


_TRAILING_QUESTION = re.compile(
    r"^(will\s+)?(?P<film>.+?)\s+(have|score|get|finish|end)\b", re.I
)


def film_from(market) -> str:
    """
    The film a market is about.

    Kalshi titles vary, so this tries the obvious shapes and falls back to
    the title itself -- a wrong guess here is caught by the slug check
    rather than becoming a silent mispricing.
    """
    for text in (market.title, market.subtitle):
        if not text:
            continue
        cleaned = re.sub(r"\(.*?\)", " ", text)
        for marker in ("Rotten Tomatoes", "Tomatometer", "RT score"):
            cleaned = re.sub(rf"'s?\s*{marker}", " ", cleaned, flags=re.I)
            cleaned = re.sub(rf"\b{marker}\b", " ", cleaned, flags=re.I)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ?:-–—")
        match = _TRAILING_QUESTION.match(cleaned)
        if match:
            return match.group("film").strip(" ?:-–—")
        if cleaned:
            return cleaned
    return market.ticker


def to_yaml(proposals, existing=None) -> str:
    """
    Render proposals as a contracts file.

    Written by hand rather than through a YAML dumper so the rules text
    can sit as a comment above each row: the point of discovery is that
    verifying becomes reading this file, and a dumped string field would
    be one long unreadable line.
    """
    keep = {c.ticker for c in (existing.contracts if existing else [])}
    lines = [
        "# Written by `betedge rt discover`. Every row is UNVERIFIED.",
        "#",
        "# The rules text from the exchange is quoted above each contract.",
        "# Read it, fix anything the parser marked with a QUESTION, and set",
        "# `verified: true`. Until then the row is priced and staked at",
        "# nothing.",
        "",
        "meta:",
        "  version: 1",
        "  last_verified_by_user: null",
        "",
        "contracts:",
    ]
    for p in sorted(proposals, key=lambda x: (x.film, x.threshold or 0)):
        if p.ticker in keep:
            continue
        lines.append("")
        for chunk in _wrapped(f"RULES: {p.rules or '(none given)'}", 72):
            lines.append(f"  # {chunk}")
        for problem in p.problems:
            for chunk in _wrapped(f"QUESTION: {problem}", 72):
                lines.append(f"  # {chunk}")
        lines.append(f"  - ticker: {p.ticker}")
        lines.append(f"    slug: {p.slug}")
        lines.append(f"    film: {_quote(p.film)}")
        lines.append(
            f"    threshold: {p.threshold if p.threshold is not None else '# FILL IN'}"
        )
        lines.append(f"    direction: {p.direction}")
        lines.append(
            "    inclusive: "
            + ("true" if p.inclusive else "false" if p.inclusive is False
               else "# FILL IN -- does the threshold itself win?")
        )
        lines.append(
            f"    scope: {p.scope or '# FILL IN -- all_critics or top_critics'}"
        )
        if p.close_time:
            lines.append(f"    settles_at: {p.close_time.isoformat()}")
        lines.append("    max_new_reviews: 60")
        lines.append("    expected_new_reviews: 20")
        lines.append("    verified: false")
    return "\n".join(lines) + "\n"


def _quote(text: str) -> str:
    escaped = (text or "").replace('"', '\\"')
    return f'"{escaped}"'


def _wrapped(text: str, width: int) -> list[str]:
    words, lines, current = (text or "").split(), [], ""
    for word in words:
        if current and len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or [""]
