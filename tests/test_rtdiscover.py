"""
Discovery: finding RT markets on Kalshi and writing the contracts file.

The tests that matter are the ones about what discovery REFUSES to
decide. Whether a threshold is inclusive, and which Tomatometer settles
it, are not in any structured field -- so the parser has to carry "I do
not know" all the way into the file rather than defaulting, because a
default would be invisible and wrong about half the time.
"""

from datetime import datetime, timezone

import pytest

from betedge import kalshi as K
from betedge import rtdiscover as D
from betedge import tomatoes as T

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


def market(**kw):
    raw = {
        "ticker": "KXRTSCORE-26-RE-T60",
        "title": "Will Resident Evil have a Tomatometer score above 60%?",
        "subtitle": "Above 60%",
        "rules_primary": "Resolves YES if the Rotten Tomatoes Tomatometer "
                         "(All Critics) is above 60% at settlement.",
        "status": "active",
        "yes_bid": 55, "yes_ask": 58, "no_bid": 42, "no_ask": 45,
        "close_time": "2026-09-26T21:00:00Z",
    }
    raw.update(kw)
    return K.parse_market(raw)


class Card:
    def __init__(self, film):
        self.film = film


def fetcher_for(mapping):
    def fetch(slug):
        if slug not in mapping:
            raise RuntimeError("404")
        return Card(mapping[slug])
    return fetch


class FakeClient:
    def __init__(self, markets):
        self._markets = markets
        self.requests_made = 0

    def markets(self, **kw):
        return self._markets


# --------------------------------------------------------------------------
# Recognising an RT market
# --------------------------------------------------------------------------


class TestRecognition:
    def test_a_tomatometer_market_is_recognised(self):
        assert D.looks_like_rt_market("Tomatometer score for X")

    def test_a_rotten_tomatoes_market_is_recognised(self):
        assert D.looks_like_rt_market("", "", "Rotten Tomatoes score")

    def test_an_unrelated_market_is_not(self):
        assert not D.looks_like_rt_market("Will the Fed cut rates?")

    def test_discovery_ignores_everything_else_on_the_exchange(self):
        client = FakeClient([
            market(),
            K.parse_market({"ticker": "FED-26", "title": "Fed cut?",
                            "status": "active"}),
        ])
        found = D.discover(client, fetcher_for({"resident_evil": "Resident Evil"}))
        assert [p.ticker for p in found] == ["KXRTSCORE-26-RE-T60"]


# --------------------------------------------------------------------------
# What it will and will not decide
# --------------------------------------------------------------------------


class TestThresholdParsing:
    def test_a_plain_threshold_is_read(self):
        threshold, direction, _, _ = D.parse_threshold("above 60%")
        assert threshold == 60
        assert direction == D.ABOVE

    def test_below_is_read(self):
        _, direction, _, _ = D.parse_threshold("below 40%")
        assert direction == D.BELOW

    @pytest.mark.parametrize("text,inclusive", [
        ("60% or above", True),
        ("at or above 60%", True),
        ("at least 60%", True),
        ("greater than or equal to 60%", True),
        ("strictly above 60%", False),
        ("greater than 60%", False),
        ("60% or below", True),
        ("less than 60%", False),
    ])
    def test_inclusivity_is_read_when_the_text_says(self, text, inclusive):
        _, _, got, _ = D.parse_threshold(text)
        assert got is inclusive

    def test_inclusivity_stays_unknown_when_the_text_does_not_say(self):
        # The decisive rule. "above 60%" is genuinely ambiguous in
        # ordinary use, and defaulting would be invisible.
        _, _, inclusive, problems = D.parse_threshold("above 60%")
        assert inclusive is None
        assert any("does not say whether" in p for p in problems)

    def test_a_banded_market_is_refused_rather_than_halved(self):
        # "60% to 69%" is a different shape this tool cannot price.
        # Grabbing one of the two numbers would look like it worked.
        threshold, _, _, problems = D.parse_threshold("Between 60% and 69%",
                                                      "60% to 69%")
        assert threshold is None
        assert any("band" in p for p in problems)

    def test_a_threshold_can_be_recovered_from_the_ticker(self):
        threshold, _, _, problems = D.parse_threshold(
            "Tomatometer", "", "KXRTSCORE-26-RE-T60"
        )
        assert threshold == 60
        assert any("from the ticker" in p for p in problems)

    def test_no_threshold_anywhere_is_reported(self):
        threshold, _, _, problems = D.parse_threshold("Tomatometer score")
        assert threshold is None
        assert any("no threshold found" in p for p in problems)

    def test_a_nonsense_threshold_is_refused(self):
        threshold, _, _, problems = D.parse_threshold("above 250%")
        assert threshold is None
        assert any("not a percentage" in p for p in problems)


class TestScopeParsing:
    def test_top_critics_is_read(self):
        assert D.parse_scope("Top Critics score") == T.SCOPE_TOP_CRITICS

    def test_all_critics_is_read(self):
        assert D.parse_scope("All Critics") == T.SCOPE_ALL_CRITICS

    def test_silence_means_unknown_not_a_default(self):
        # Six points apart on Resident Evil. Guessing is not available.
        assert D.parse_scope("Tomatometer score") is None

    def test_an_unscoped_market_carries_the_question(self):
        m = market(rules_primary="Resolves YES if the Tomatometer is above 60%.")
        found = D.discover(FakeClient([m]),
                           fetcher_for({"resident_evil": "Resident Evil"}))
        assert found[0].scope is None
        assert any("which Tomatometer" in p for p in found[0].problems)


# --------------------------------------------------------------------------
# The slug check
# --------------------------------------------------------------------------


class TestSlugs:
    def test_a_title_becomes_a_candidate(self):
        assert "resident_evil" in D.slug_candidates("Resident Evil")

    def test_accents_and_punctuation_are_folded(self):
        assert "amelie" in D.slug_candidates("Amélie")
        assert "spider_man_no_way_home" in D.slug_candidates(
            "Spider-Man: No Way Home"
        )

    def test_a_year_is_tried_for_remakes(self):
        candidates = D.slug_candidates("The Thing", close_time=NOW)
        assert "the_thing" in candidates
        assert "the_thing_2026" in candidates

    def test_a_confirmed_slug_is_used(self):
        slug, ok, _ = D.confirm_slug(
            "Resident Evil", ["resident_evil"],
            fetcher_for({"resident_evil": "Resident Evil"}),
        )
        assert (slug, ok) == ("resident_evil", True)

    def test_a_slug_for_the_wrong_film_is_refused(self):
        # The failure no downstream guard could catch: every number
        # involved would be perfectly valid, just about another film.
        slug, ok, problems = D.confirm_slug(
            "Resident Evil", ["resident_evil"],
            fetcher_for({"resident_evil": "Resident Evil: Afterlife"}),
        )
        assert ok is False
        assert slug == ""
        assert any("not 'Resident Evil'" in p for p in problems)

    def test_it_falls_through_to_the_next_candidate(self):
        slug, ok, _ = D.confirm_slug(
            "The Thing", ["the_thing", "the_thing_2026"],
            fetcher_for({"the_thing_2026": "The Thing"}),
        )
        assert (slug, ok) == ("the_thing_2026", True)

    @pytest.mark.parametrize("page_title", [
        "Resident Evil: Afterlife",
        "Resident Evil: The Final Chapter",
        "Resident Evil",          # control: this one SHOULD match
    ])
    def test_only_the_exact_film_matches(self, page_title):
        slug, ok, _ = D.confirm_slug(
            "Resident Evil", ["resident_evil"],
            fetcher_for({"resident_evil": page_title}),
        )
        assert ok is (page_title == "Resident Evil")

    def test_a_disambiguating_year_still_matches(self):
        # Rotten Tomatoes titles remakes "The Thing (2011)"; Kalshi does
        # not. That is the same film.
        slug, ok, _ = D.confirm_slug(
            "The Thing", ["the_thing_2011"],
            fetcher_for({"the_thing_2011": "The Thing (2011)"}),
        )
        assert ok is True

    def test_an_unreachable_page_is_not_a_confirmation(self):
        slug, ok, problems = D.confirm_slug(
            "Nope", ["nope"], fetcher_for({})
        )
        assert ok is False
        assert any("404" in p for p in problems)

    def test_a_proposal_without_a_slug_is_not_usable(self):
        found = D.discover(FakeClient([market()]), fetcher_for({}))
        assert found[0].usable is False
        assert any("fill in `slug` by hand" in p for p in found[0].problems)


class TestFilmNames:
    @pytest.mark.parametrize("title,expected", [
        ("Will Resident Evil have a Tomatometer score above 60%?",
         "Resident Evil"),
        ("Will Dune Part Three score above 80%?", "Dune Part Three"),
    ])
    def test_the_film_is_pulled_out_of_the_question(self, title, expected):
        assert D.film_from(market(title=title, subtitle="")) == expected

    def test_one_page_is_fetched_per_film_not_per_market(self):
        # Three thresholds on one film is one film page, not three.
        calls = []

        def counting(slug):
            calls.append(slug)
            return Card("Resident Evil")

        markets = [market(ticker=f"KXRT-RE-T{t}", subtitle=f"Above {t}%")
                   for t in (40, 60, 80)]
        D.discover(FakeClient(markets), counting)
        assert len(calls) == 1


# --------------------------------------------------------------------------
# The file it writes
# --------------------------------------------------------------------------


class TestYamlOutput:
    @pytest.fixture
    def proposals(self):
        return D.discover(FakeClient([market()]),
                          fetcher_for({"resident_evil": "Resident Evil"}))

    def test_it_is_valid_yaml(self, proposals):
        import yaml

        parsed = yaml.safe_load(D.to_yaml(proposals))
        assert parsed["contracts"][0]["ticker"] == "KXRTSCORE-26-RE-T60"

    def test_the_rules_text_is_written_into_the_file(self, proposals):
        # The actual saving: verifying is reading this file, not
        # browsing the exchange.
        body = D.to_yaml(proposals)
        assert "RULES:" in body
        assert "All Critics" in body

    def test_every_row_lands_unverified(self, proposals):
        import yaml

        parsed = yaml.safe_load(D.to_yaml(proposals))
        assert all(c["verified"] is False for c in parsed["contracts"])

    def test_it_loads_back_through_the_contract_book(self, proposals, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(D.to_yaml(proposals))
        book = T.ContractBook.load(path)
        assert book.contracts[0].ticker == "KXRTSCORE-26-RE-T60"
        assert book.unverified == ["KXRTSCORE-26-RE-T60"]

    def test_an_unknown_inclusivity_is_a_question_not_a_value(self):
        # It must not parse as true or false, because a silent default
        # here is a whole contract.
        import yaml

        m = market(subtitle="Above 60%",
                   rules_primary="Resolves on the All Critics Tomatometer.")
        found = D.discover(FakeClient([m]),
                           fetcher_for({"resident_evil": "Resident Evil"}))
        parsed = yaml.safe_load(D.to_yaml(found))
        assert parsed["contracts"][0]["inclusive"] is None

    def test_an_unknown_scope_is_a_question_too(self):
        import yaml

        m = market(rules_primary="Resolves on the Tomatometer above 60%.")
        found = D.discover(FakeClient([m]),
                           fetcher_for({"resident_evil": "Resident Evil"}))
        parsed = yaml.safe_load(D.to_yaml(found))
        assert parsed["contracts"][0]["scope"] is None

    def test_the_questions_are_in_the_file_as_comments(self):
        m = market(rules_primary="Resolves on the Tomatometer above 60%.")
        found = D.discover(FakeClient([m]),
                           fetcher_for({"resident_evil": "Resident Evil"}))
        assert "QUESTION:" in D.to_yaml(found)

    def test_tickers_already_in_the_file_are_not_duplicated(
        self, proposals, tmp_path
    ):
        # Re-running discovery must not add a second row for a contract
        # you already verified -- that would price and stake it twice.
        path = tmp_path / "c.yaml"
        path.write_text(D.to_yaml(proposals))
        existing = T.ContractBook.load(path)
        again = D.to_yaml(proposals, existing=existing)
        assert "KXRTSCORE-26-RE-T60" not in again


# --------------------------------------------------------------------------
# The sweep, and saying honestly what was looked at
# --------------------------------------------------------------------------


class EventClient:
    """A client whose events endpoint works."""

    def __init__(self, events, markets=None):
        self._events = events
        self._markets = markets or []
        self.requests_made = 0
        self.markets_called = False

    def events(self, **kw):
        return self._events

    def markets(self, **kw):
        self.markets_called = True
        return self._markets


class NoEventsClient(EventClient):
    def events(self, **kw):
        raise RuntimeError("no such endpoint")


def event(title, markets):
    return {"title": title, "markets": markets}


class TestSweep:
    def test_the_event_title_is_carried_down_into_its_markets(self):
        # A market's own subtitle is often just "Above 60%". Without the
        # event title there is no film name to look up at all.
        client = EventClient([event(
            "Resident Evil Tomatometer score",
            [{"ticker": "KXRT-RE-T60", "subtitle": "Above 60%",
              "status": "active"}],
        )])
        sweep = D.discover(client,
                           fetcher_for({"resident_evil": "Resident Evil"}))
        assert len(sweep) == 1
        assert sweep[0].film == "Resident Evil"

    def test_it_falls_back_to_the_flat_market_list(self):
        client = NoEventsClient([], [market()])
        sweep = D.discover(client,
                           fetcher_for({"resident_evil": "Resident Evil"}))
        assert client.markets_called
        assert sweep.source == "markets"
        assert len(sweep) == 1

    def test_it_reports_what_it_looked_at(self):
        client = EventClient([event("Fed decision", [
            {"ticker": "FED-1", "subtitle": "Cut", "status": "active"}
        ])])
        sweep = D.discover(client, fetcher_for({}))
        assert len(sweep) == 0
        assert sweep.markets_seen == 1
        assert sweep.events_seen == 1

    def test_a_truncated_sweep_says_so(self):
        # Not finding something in a complete sweep and not finding it in
        # a partial one are different answers. The first version of this
        # gave the confident one after seeing a twentieth of Kalshi.
        events = [event(f"E{i}", []) for i in range(400)]
        sweep = D.discover(EventClient(events), fetcher_for({}), max_pages=2)
        assert sweep.truncated is True

    def test_a_complete_sweep_does_not(self):
        sweep = D.discover(EventClient([event("E", [])]), fetcher_for({}),
                           max_pages=25)
        assert sweep.truncated is False

    def test_a_malformed_nested_market_does_not_lose_the_event(self):
        client = EventClient([event("Resident Evil Tomatometer", [
            {"junk": True},
            {"ticker": "KXRT-RE-T60", "subtitle": "Above 60%",
             "status": "active"},
        ])])
        sweep = D.discover(client,
                           fetcher_for({"resident_evil": "Resident Evil"}))
        assert len(sweep) == 1


class TestGrep:
    def test_it_finds_markets_by_title(self):
        client = EventClient([
            event("Resident Evil Tomatometer score",
                  [{"ticker": "KXRTSCORE-RE-T60", "subtitle": "Above 60%",
                    "status": "active"}]),
            event("Fed decision",
                  [{"ticker": "FED-1", "subtitle": "Cut", "status": "active"}]),
        ])
        hits = D.grep_titles(client, "tomato")
        assert len(hits) == 1
        assert hits[0][0] == "KXRTSCORE"

    def test_it_matches_the_ticker_too(self):
        client = EventClient([event("Something", [
            {"ticker": "KXTOMATO-1", "subtitle": "x", "status": "active"}
        ])])
        assert D.grep_titles(client, "tomato")

    def test_no_match_is_an_empty_list_not_an_error(self):
        client = EventClient([event("Fed", [
            {"ticker": "FED-1", "subtitle": "Cut", "status": "active"}
        ])])
        assert D.grep_titles(client, "tomato") == []
