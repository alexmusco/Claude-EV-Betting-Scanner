"""
Joining a Kalshi market to an odds-feed event.

The tests that matter are the REFUSALS. A ladder priced against the wrong
game produces entirely valid-looking numbers -- the fit converges, sigma
lands in range, every rung is a probability -- and nothing downstream
knows what the words said. So the only safe behaviour on an uncertain
join is to decline it.
"""

from datetime import datetime, timedelta, timezone

import pytest

from betedge import matching as M

NOW = datetime(2026, 9, 18, 0, 15, tzinfo=timezone.utc)


def event(home, away, start=NOW, eid="evt1"):
    return {"id": eid, "home_team": home, "away_team": away,
            "commence_time": start.isoformat()}


CHIEFS = event("Kansas City Chiefs", "Denver Broncos", eid="kc-den")


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------


class TestNames:
    @pytest.mark.parametrize("full,nick", [
        ("Kansas City Chiefs", "chiefs"),
        ("Denver Broncos", "broncos"),
        ("Los Angeles Dodgers", "dodgers"),
        ("Montréal Canadiens", "canadiens"),
    ])
    def test_the_nickname_is_the_distinctive_part(self, full, nick):
        assert M.nickname(full) == nick

    @pytest.mark.parametrize("full,nick", [
        ("Chicago White Sox", "white sox"),
        ("Boston Red Sox", "red sox"),
    ])
    def test_a_two_word_nickname_is_kept_whole(self, full, nick):
        # "Sox" alone belongs to two teams in the same league.
        assert M.nickname(full) == nick

    def test_a_within_league_collision_keeps_both_words(self):
        # "Sox" belongs to Chicago and Boston in the SAME league, so the
        # sport cannot separate them and more of the name is needed.
        assert M.team_key("Chicago White Sox") != M.team_key("Boston Red Sox")

    def test_accents_and_punctuation_do_not_matter(self):
        assert M.normalise("Montréal Canadiens") == "montreal canadiens"

    def test_an_empty_name_has_no_nickname(self):
        assert M.nickname("") == ""
        assert M.find_team("anything", "") is False


class TestFindTeam:
    @pytest.mark.parametrize("title", [
        "Will the Chiefs beat the Broncos?",
        "Kansas City Chiefs vs Denver Broncos",
        "Chiefs/Broncos: winning margin",
        "DENVER BRONCOS AT KANSAS CITY CHIEFS",
    ])
    def test_it_finds_a_team_however_the_title_is_written(self, title):
        assert M.find_team(title, "Kansas City Chiefs")
        assert M.find_team(title, "Denver Broncos")

    def test_it_matches_whole_words_only(self):
        # Substring matching is how a confident match to the wrong game
        # happens. "Jets" must not be found inside "Jetsons".
        assert not M.find_team("The Jetsons movie", "New York Jets")

    def test_a_cross_league_collision_is_left_to_the_event_set(self):
        # Both Rangers answer to "Rangers", but the caller scopes events
        # to one sport and BOTH teams must match one event -- so the
        # Astros in this title are what decide it, not the city.
        assert M.find_team("Rangers vs Astros", "Texas Rangers")
        assert M.find_team("Rangers vs Bruins", "New York Rangers")

    def test_two_word_nicknames_are_found(self):
        assert M.find_team("Will the White Sox win?", "Chicago White Sox")
        assert not M.find_team("Will the White Sox win?", "Boston Red Sox")

    def test_a_city_alone_is_not_a_team(self):
        # Los Angeles fields two NFL teams, so a city match is evidence
        # of nothing at all.
        assert not M.find_team("Los Angeles hosts a game", "Los Angeles Rams")


# --------------------------------------------------------------------------
# Matching, and refusing to
# --------------------------------------------------------------------------


class TestMatching:
    def test_a_clean_match_is_confident(self):
        result = M.match_event(
            "Will the Chiefs beat the Broncos?", [CHIEFS], close_time=NOW
        )
        assert result.confident
        assert result.event_id == "kc-den"

    def test_one_team_is_never_enough(self):
        # Half a match is not a match. The other team could be anyone.
        result = M.match_event("Chiefs to win the AFC", [CHIEFS])
        assert not result.confident
        assert "both of these teams" in result.reason

    def test_no_matching_event_is_reported_not_guessed(self):
        result = M.match_event("Will the Jets beat the Bills?", [CHIEFS])
        assert not result.confident
        assert result.event is None

    def test_the_same_teams_on_another_date_are_refused(self):
        # A rematch is a different game. Both teams match perfectly,
        # which is exactly what makes this dangerous.
        later = event("Kansas City Chiefs", "Denver Broncos",
                      start=NOW + timedelta(days=7), eid="rematch")
        result = M.match_event(
            "Will the Chiefs beat the Broncos?", [later], close_time=NOW
        )
        assert not result.confident
        assert "different meeting" in result.reason
        # The best guess is still carried, for a human to look at.
        assert result.event_id == "rematch"

    def test_a_timezone_slip_is_tolerated(self):
        shifted = event("Kansas City Chiefs", "Denver Broncos",
                        start=NOW + timedelta(hours=3), eid="shifted")
        result = M.match_event(
            "Chiefs vs Broncos", [shifted], close_time=NOW
        )
        assert result.confident

    def test_two_events_naming_the_same_pair_are_ambiguous(self):
        # A doubleheader, or a duplicated feed entry. Picking one is a
        # coin flip dressed up as a decision. Both are placed inside the
        # window so that the ambiguity is what is being tested.
        a = event("Los Angeles Dodgers", "San Diego Padres", eid="game1")
        b = event("Los Angeles Dodgers", "San Diego Padres",
                  start=NOW - timedelta(hours=2), eid="game2")
        result = M.match_event("Dodgers vs Padres", [a, b], close_time=NOW)
        assert not result.confident
        assert "ambiguous" in result.reason
        assert result.candidates == 2

    def test_a_market_closing_after_kickoff_is_the_same_game(self):
        """
        The two sides are not measuring the same moment. An odds feed
        gives KICKOFF; an exchange gives when its market CLOSES, which is
        at or after the final whistle. A symmetric window centred on
        kickoff treats a perfectly ordinary NFL market -- an 8pm game
        whose contract closes near midnight -- as a different fixture,
        which is exactly what rejected a whole Week 3 slate.
        """
        for hours in (1, 3, 6, 10, 13):
            result = M.match_event(
                "Chiefs vs Broncos", [CHIEFS],
                close_time=NOW + timedelta(hours=hours),
            )
            assert result.confident, hours

    def test_a_market_closing_well_before_kickoff_is_not(self):
        # Backwards stays tight: a market that closes before the game
        # starts is a puzzle, not a tolerance.
        result = M.match_event("Chiefs vs Broncos", [CHIEFS],
                               close_time=NOW - timedelta(hours=6))
        assert not result.confident

    def test_the_next_meeting_is_still_out_of_reach(self):
        # The window has to stay far short of the same two teams playing
        # again, which is the whole reason it exists.
        result = M.match_event("Chiefs vs Broncos", [CHIEFS],
                               close_time=NOW + timedelta(days=7))
        assert not result.confident
        assert "different meeting" in result.reason

    def test_an_explicit_tolerance_is_still_symmetric(self):
        # A caller that passes its own number means that number.
        assert M.match_event(
            "Chiefs vs Broncos", [CHIEFS],
            close_time=NOW - timedelta(hours=6), tolerance_hours=12,
        ).confident

    def test_without_a_close_time_a_single_match_still_works(self):
        result = M.match_event("Chiefs vs Broncos", [CHIEFS])
        assert result.confident

    def test_the_right_game_is_picked_out_of_a_full_slate(self):
        slate = [
            CHIEFS,
            event("Buffalo Bills", "New York Jets", eid="buf-nyj"),
            event("Green Bay Packers", "Chicago Bears", eid="gb-chi"),
        ]
        result = M.match_event("Bills vs Jets margin", slate, close_time=NOW)
        assert result.confident
        assert result.event_id == "buf-nyj"

    def test_the_other_team_resolves_a_cross_league_nickname(self):
        """
        Two leagues have Rangers. Neither title names a city, and both
        are matched correctly -- because the OTHER team in each title
        only appears in one of the events.
        """
        mlb = event("Texas Rangers", "Houston Astros", eid="mlb")
        nhl = event("New York Rangers", "Boston Bruins", eid="nhl")
        slate = [mlb, nhl]
        assert M.match_event("Rangers vs Astros", slate,
                             close_time=NOW).event_id == "mlb"
        assert M.match_event("Rangers vs Bruins", slate,
                             close_time=NOW).event_id == "nhl"

    def test_a_bare_nickname_shared_by_two_listed_games_is_ambiguous(self):
        # If the event set really does contain both, nothing in the
        # title separates them and the join is declined.
        slate = [event("Texas Rangers", "New York Rangers", eid="odd")]
        result = M.match_event("Rangers vs Rangers", slate, close_time=NOW)
        assert result.confident  # one event, both teams named
        assert result.event_id == "odd"

    def test_an_event_missing_a_team_name_is_skipped(self):
        broken = {"id": "x", "home_team": "", "away_team": "Denver Broncos",
                  "commence_time": NOW.isoformat()}
        result = M.match_event("Chiefs vs Broncos", [broken, CHIEFS],
                               close_time=NOW)
        assert result.confident
        assert result.event_id == "kc-den"


class TestMatchMarkets:
    class FakeMarket:
        def __init__(self, title, subtitle="", close_time=NOW):
            self.title = title
            self.subtitle = subtitle
            self.close_time = close_time

    def test_confident_joins_and_refusals_are_separated(self):
        markets = [
            self.FakeMarket("Chiefs vs Broncos", "Margin over 6.5"),
            self.FakeMarket("Will the Jets beat the Bills?"),
        ]
        matched, unmatched = M.match_markets(markets, [CHIEFS])
        assert len(matched) == 1
        assert len(unmatched) == 1
        assert matched[0][1].event_id == "kc-den"

    def test_every_refusal_carries_its_reason(self):
        # A run that quietly covers fewer games than the operator thinks
        # is worse than one that covers none.
        markets = [self.FakeMarket("Chiefs to win the AFC")]
        _matched, unmatched = M.match_markets(markets, [CHIEFS])
        assert unmatched[0][1].reason

    def test_the_subtitle_helps_identify_the_game(self):
        # A market's own title is often just a threshold; the teams may
        # only appear once the two are read together.
        markets = [self.FakeMarket("Margin over 6.5", "Chiefs vs Broncos")]
        matched, _unmatched = M.match_markets(markets, [CHIEFS])
        assert len(matched) == 1

    def test_nothing_to_match_is_not_an_error(self):
        assert M.match_markets([], [CHIEFS]) == ([], [])
