"""
Reading a Tomatometer off a real Rotten Tomatoes page.

The fixture is a genuine fetch of /m/resident_evil, gzipped. Tests
against hand-written HTML would pass forever and tell you nothing; this
one fails the day Rotten Tomatoes changes where or how it publishes
scores, which is the only thing worth being told.
"""

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from betedge import rtfetch as R
from betedge import tomatoes as T

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
FIXTURE = Path(__file__).parent / "fixtures" / "rt_resident_evil.html.gz"


@pytest.fixture(scope="module")
def page() -> str:
    with gzip.open(FIXTURE, "rt", encoding="utf-8", errors="replace") as fh:
        return fh.read()


@pytest.fixture
def card(page):
    return R.parse_scorecard(page, slug="resident_evil", now=NOW)


class TestParsingTheRealPage:
    def test_it_finds_the_film(self, card):
        assert card.film == "Resident Evil"

    def test_the_all_critics_counts_are_exact(self, card):
        # Not a percentage: the counts, which are the only thing the
        # model can actually use.
        s = card.scope(T.SCOPE_ALL_CRITICS)
        assert (s.fresh, s.rotten, s.total) == (58, 104, 162)

    def test_the_top_critics_counts_are_exact(self, card):
        s = card.scope(T.SCOPE_TOP_CRITICS)
        assert (s.fresh, s.rotten, s.total) == (13, 30, 43)

    def test_our_rounding_matches_what_the_site_displays(self, card):
        # The self-validating check, on real data: 58/162 = 35.80 -> 36,
        # 13/43 = 30.23 -> 30, and Rotten Tomatoes publishes 36 and 30.
        assert card.scope(T.SCOPE_ALL_CRITICS).displayed_score == 36
        assert card.scope(T.SCOPE_TOP_CRITICS).displayed_score == 30
        assert card.published == {T.SCOPE_ALL_CRITICS: 36,
                                  T.SCOPE_TOP_CRITICS: 30}

    def test_the_two_tomatometers_really_do_differ(self, card):
        # Six points apart on the same film. This is why a contract that
        # does not say which one it settles on is not scorable, and why
        # the scope guard is not pedantry.
        all_c = card.scope(T.SCOPE_ALL_CRITICS).displayed_score
        top_c = card.scope(T.SCOPE_TOP_CRITICS).displayed_score
        assert abs(all_c - top_c) == 6

    def test_the_snapshot_is_stamped_with_when_it_was_read(self, card):
        assert card.scope(T.SCOPE_ALL_CRITICS).captured_at == NOW
        assert card.fetched_at == NOW

    def test_the_audience_score_is_not_mistaken_for_a_tomatometer(self, card):
        # The same blob carries a Popcornmeter at 67% on 48,492 ratings.
        # Picking that up instead would be silent and catastrophic.
        for snapshot in card.scores.values():
            assert snapshot.total < 1000

    def test_the_scores_it_returns_are_usable_straight_away(self, card):
        from betedge.config import TomatoesConfig

        s = card.scope(T.SCOPE_ALL_CRITICS)
        # 58/162 with at most 60 more reviews cannot reach 60%.
        assert T.decided(
            s,
            T.Contract(ticker="X", film="Resident Evil", threshold=60,
                       settlement_verified=True),
            60,
        ) is False


class TestItFailsLoudlyRatherThanQuietly:
    def test_a_page_with_no_scorecard_is_an_error(self):
        with pytest.raises(R.ParseError, match="no mediaScorecard"):
            R.parse_scorecard("<html><title>Nope</title></html>")

    def test_a_challenge_page_does_not_parse_as_a_film(self):
        html = "<html><title>Just a moment...</title><body></body></html>"
        with pytest.raises(R.ParseError):
            R.parse_scorecard(html)

    def test_malformed_json_is_an_error(self):
        html = (
            '<script data-json="mediaScorecard" type="application/json">'
            "{not json}</script>"
        )
        with pytest.raises(R.ParseError, match="did not parse"):
            R.parse_scorecard(html)

    def test_counts_that_do_not_add_up_stop_everything(self):
        # If reviewCount stops meaning "the reviews behind this score",
        # every number downstream is wrong.
        html = _scorecard({
            "criticsAll": {"likedCount": 58, "notLikedCount": 104,
                           "reviewCount": 200, "score": "36"}
        })
        with pytest.raises(R.ParseError, match="no longer means"):
            R.parse_scorecard(html)

    def test_a_rounding_disagreement_stops_everything(self):
        # Contracts settle on the number RT displays. If ours differs,
        # we are pricing a different question than the market is.
        html = _scorecard({
            "criticsAll": {"likedCount": 58, "notLikedCount": 104,
                           "reviewCount": 162, "score": "37"}
        })
        with pytest.raises(R.ParseError, match="displays 37"):
            R.parse_scorecard(html)

    def test_a_percentage_with_no_counts_is_refused(self):
        # The failure mode of every licensed source: a score and no n.
        html = _scorecard({"criticsAll": {"score": "36", "reviewCount": 162}})
        with pytest.raises(R.ParseError, match="cannot drive this model"):
            R.parse_scorecard(html)

    def test_a_film_with_no_reviews_yet_is_not_an_error_but_is_not_priced(self):
        html = _scorecard({
            "criticsAll": {"likedCount": 0, "notLikedCount": 0,
                           "reviewCount": 0, "score": ""},
            "criticsTop": {"likedCount": 13, "notLikedCount": 30,
                           "reviewCount": 43, "score": "30"},
        })
        card = R.parse_scorecard(html, now=NOW)
        assert T.SCOPE_ALL_CRITICS not in card.scores
        assert card.scope(T.SCOPE_TOP_CRITICS).total == 43

    def test_asking_for_a_scope_the_page_lacks_says_so(self):
        html = _scorecard({
            "criticsAll": {"likedCount": 58, "notLikedCount": 104,
                           "reviewCount": 162, "score": "36"}
        })
        card = R.parse_scorecard(html, slug="x", now=NOW)
        with pytest.raises(R.ParseError, match="no top_critics"):
            card.scope(T.SCOPE_TOP_CRITICS)


class TestRobots:
    def test_a_film_path_is_allowed(self):
        assert R.path_for("resident_evil") == "/m/resident_evil"

    def test_a_slug_already_carrying_its_prefix_is_not_doubled(self):
        assert R.path_for("m/resident_evil") == "/m/resident_evil"
        assert R.path_for("/m/resident_evil/") == "/m/resident_evil"

    def test_the_pictures_page_is_refused(self):
        # robots.txt disallows /m/*/pictures explicitly.
        with pytest.raises(ValueError, match="not a film slug"):
            R.path_for("resident_evil/pictures")

    @pytest.mark.parametrize("slug", [
        "../search",
        "resident_evil/../../search",
        "..%2fsearch",
        "resident_evil?x=1",
        "resident_evil/reviews",
        "",
    ])
    def test_anything_that_could_leave_the_film_path_is_refused(self, slug):
        # "/m/../search" starts with "/m/" and is still a request for
        # /search, so validating the assembled path is not enough.
        with pytest.raises(ValueError):
            R.path_for(slug)


class TestFetch:
    def test_a_fresh_cache_file_is_used_without_a_request(self, tmp_path, page):
        # The cache IS the rate limiting. If this ever stops holding, a
        # scan across ten films becomes ten requests every run.
        cache = tmp_path / "rt"
        cache.mkdir()
        (cache / "m_resident_evil.html").write_text(page, encoding="utf-8")

        class Exploding:
            def get(self, *a, **kw):  # pragma: no cover - must not run
                raise AssertionError("a cached page must not be re-fetched")

        card = R.fetch("resident_evil", cache_dir=cache, session=Exploding(),
                       now=NOW)
        assert card.scope(T.SCOPE_ALL_CRITICS).total == 162

    def test_a_stale_cache_file_is_refetched(self, tmp_path, page):
        cache = tmp_path / "rt"
        cache.mkdir()
        stale = cache / "m_resident_evil.html"
        stale.write_text("<html>old</html>", encoding="utf-8")

        class Fake:
            def __init__(self):
                self.calls = 0

            def get(self, url, **kw):
                self.calls += 1
                return _Response(page)

        session = Fake()
        card = R.fetch("resident_evil", cache_dir=cache, session=session,
                       max_age_seconds=0.0, now=NOW)
        assert session.calls == 1
        assert card.scope(T.SCOPE_ALL_CRITICS).total == 162
        # And the refreshed page replaced the stale one.
        assert "mediaScorecard" in stale.read_text(encoding="utf-8")

    def test_it_identifies_itself_rather_than_pretending_to_be_a_browser(
        self, tmp_path, page
    ):
        seen = {}

        class Fake:
            def get(self, url, headers=None, **kw):
                seen["url"] = url
                seen["headers"] = headers or {}
                return _Response(page)

        R.fetch("resident_evil", cache_dir=tmp_path / "rt", session=Fake(),
                now=NOW)
        assert seen["url"] == "https://www.rottentomatoes.com/m/resident_evil"
        assert "betedge" in seen["headers"]["User-Agent"]

    def test_a_disallowed_path_never_reaches_the_network(self, tmp_path):
        class Exploding:
            def get(self, *a, **kw):  # pragma: no cover - must not run
                raise AssertionError("robots.txt should have stopped this")

        with pytest.raises(ValueError, match="robots.txt"):
            R.fetch("resident_evil/pictures", cache_dir=tmp_path,
                    session=Exploding())


# --------------------------------------------------------------------------


class _Response:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        return None


def _scorecard(overlay: dict) -> str:
    return (
        "<html><title>A Film | Rotten Tomatoes</title>"
        '<script id="media-scorecard-json" data-json="mediaScorecard" '
        'type="application/json">'
        + json.dumps({"overlay": overlay})
        + "</script></html>"
    )
