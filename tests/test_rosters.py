"""
Rosters: name matching, provenance, staleness, and the refusal to guess.

The tests that matter most are the ones asserting what the module will NOT
do. An unknown team costs a weaker prior; a wrong team puts a confident
sign on a pair that nothing downstream questions, so every path that
could produce one has a test holding it shut.
"""

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from betedge import rosters as R
from betedge.db import Database

TODAY = date(2026, 9, 15)


def entry(player, team, source=R.SOURCE_MANUAL, days_old=0, sport="nfl"):
    return R.RosterEntry(
        sport=sport, player=player, team=team, source=source,
        as_of=TODAY - timedelta(days=days_old),
    )


class TestNameNormalisation:
    @pytest.mark.parametrize("a,b", [
        ("A.J. Brown", "AJ Brown"),
        ("T.J. Hockenson", "TJ Hockenson"),
        ("Amon-Ra St. Brown", "Amon Ra St Brown"),
        ("Odell Beckham Jr.", "Odell Beckham"),
        ("Kenneth Walker III", "Kenneth Walker"),
        ("Patrick  Mahomes ", "patrick mahomes"),
        ("Nikola Jokić", "Nikola Jokic"),
    ])
    def test_the_spellings_that_differ_between_feeds_collapse(self, a, b):
        assert R.normalise_name(a) == R.normalise_name(b)

    def test_different_players_do_not_collapse(self):
        assert R.normalise_name("Josh Allen") != R.normalise_name("Keenan Allen")

    def test_a_suffix_can_be_kept_when_asked(self):
        assert R.normalise_name("Michael Carter II", strip_suffixes=False) != \
            R.normalise_name("Michael Carter", strip_suffixes=False)

    def test_empty_input_is_empty(self):
        assert R.normalise_name(None) == ""
        assert R.normalise_name("") == ""


class TestRosterEntry:
    def test_it_ages(self):
        assert entry("A", "KC", days_old=5).age_days(TODAY) == 5

    def test_staleness_is_measured_against_the_threshold(self):
        assert not entry("A", "KC", days_old=5).stale(30, TODAY)
        assert entry("A", "KC", days_old=45).stale(30, TODAY)

    def test_an_undated_entry_counts_as_stale(self):
        """It cannot be shown to be current, and that is the whole point."""
        undated = R.RosterEntry(sport="nfl", player="A", team="KC",
                                source=R.SOURCE_MANUAL, as_of=None)
        assert undated.stale(3650, TODAY)

    def test_it_describes_its_own_provenance(self):
        text = entry("A", "KC", source="nflverse", days_old=2).describe(TODAY)
        assert "KC" in text and "nflverse" in text and "2d" in text


class TestRosterBook:
    def test_it_answers_for_a_known_player(self):
        book = R.RosterBook([entry("Patrick Mahomes", "KC")], today=TODAY)
        assert book.team_for("nfl", "Patrick Mahomes") == "KC"
        assert book.team_for("nfl", "patrick mahomes") == "KC"

    def test_an_unknown_player_gets_nothing_rather_than_a_guess(self):
        book = R.RosterBook([entry("Patrick Mahomes", "KC")], today=TODAY)
        assert book.lookup("nfl", "Somebody Else") is None

    def test_a_stale_entry_is_dropped_not_demoted(self):
        book = R.RosterBook([entry("A", "KC", days_old=90)],
                            max_age_days=30, today=TODAY)
        assert book.team_for("nfl", "A") is None
        assert book.rejected_stale == 1

    def test_the_manual_override_beats_a_feed(self):
        book = R.RosterBook([
            entry("A", "SF", source="nflverse", days_old=0),
            entry("A", "KC", source=R.SOURCE_MANUAL, days_old=1),
        ], today=TODAY)
        assert book.team_for("nfl", "A") == "KC"

    def test_a_feed_beats_a_fit_from_last_month(self):
        book = R.RosterBook([
            entry("A", "SF", source=R.SOURCE_GAME_LOGS, days_old=20),
            entry("A", "KC", source="nflverse", days_old=1),
        ], today=TODAY)
        assert book.team_for("nfl", "A") == "KC"

    def test_within_one_source_the_newer_entry_wins(self):
        book = R.RosterBook([
            entry("A", "SF", source="nflverse", days_old=10),
            entry("A", "KC", source="nflverse", days_old=1),
        ], today=TODAY)
        # Two teams inside one source means two players sharing a name, so
        # this must NOT silently pick one.
        assert book.team_for("nfl", "A") is None
        assert ("nfl", "a") in book.ambiguous

    def test_two_players_sharing_a_name_are_refused(self):
        book = R.RosterBook([
            entry("Aaron Brewer", "ARI", source="nflverse"),
            entry("Aaron Brewer", "MIA", source="nflverse"),
        ], today=TODAY)
        assert book.lookup("nfl", "Aaron Brewer") is None

    def test_a_suffix_collision_is_resolved_by_the_exact_spelling(self):
        # Michael Carter the running back and Michael Carter II the
        # defensive back only collide because the suffix was stripped, so
        # the exact spelling is consulted first and both survive.
        book = R.RosterBook([
            entry("Michael Carter", "ARI", source="nflverse"),
            entry("Michael Carter II", "PHI", source="nflverse"),
        ], today=TODAY)
        assert book.team_for("nfl", "Michael Carter") == "ARI"
        assert book.team_for("nfl", "Michael Carter II") == "PHI"

    def test_a_suffix_still_matches_when_there_is_no_collision(self):
        book = R.RosterBook([entry("Odell Beckham Jr.", "BAL")], today=TODAY)
        assert book.team_for("nfl", "Odell Beckham") == "BAL"

    def test_an_unscoped_entry_serves_any_sport(self):
        book = R.RosterBook([entry("A", "KC", sport="")], today=TODAY)
        assert book.team_for("americanfootball_nfl", "A") == "KC"

    def test_a_sport_scoped_entry_wins_over_an_unscoped_one(self):
        book = R.RosterBook([
            entry("A", "WILD", sport=""),
            entry("A", "NFL", sport="americanfootball_nfl"),
        ], today=TODAY)
        assert book.team_for("americanfootball_nfl", "A") == "NFL"

    def test_it_reports_disagreement_between_sources(self):
        book = R.RosterBook([
            entry("A", "SF", source="nflverse"),
            entry("A", "KC", source=R.SOURCE_MANUAL),
        ], today=TODAY)
        conflicts = book.conflicts()
        assert len(conflicts) == 1
        assert {e.team for e in conflicts[0][2]} == {"SF", "KC"}

    def test_coverage_is_measured_against_a_real_slate(self):
        book = R.RosterBook([entry("A", "KC"), entry("B", "KC")], today=TODAY)
        stats = book.coverage("nfl", ["A", "B", "C", "D"])
        assert stats["known"] == 2
        assert stats["rate"] == 0.5
        assert stats["missing"] == ["C", "D"]

    def test_from_a_plain_mapping(self):
        book = R.RosterBook.from_mapping({"A": "KC"}, sport="nfl")
        assert book.team_for("nfl", "A") == "KC"

    def test_source_counts_are_broken_out(self):
        book = R.RosterBook([
            entry("A", "KC", source="nflverse"),
            entry("B", "SF", source=R.SOURCE_GAME_LOGS),
        ], today=TODAY)
        assert book.source_counts("nfl") == {"nflverse": 1, "game_logs": 1}


class TestManualFile:
    def test_a_csv_loads_and_wins(self, tmp_path):
        p = tmp_path / "r.csv"
        p.write_text("player,team,position\nPatrick Mahomes,KC,QB\n")
        entries = R.from_manual_csv(p)
        assert entries[0].team == "KC"
        assert entries[0].source == R.SOURCE_MANUAL

    def test_a_hand_edited_file_is_dated_today(self, tmp_path):
        p = tmp_path / "r.csv"
        p.write_text("player,team\nA,KC\n")
        assert R.from_manual_csv(p)[0].as_of == datetime.now(timezone.utc).date()

    def test_an_explicit_as_of_is_respected(self, tmp_path):
        p = tmp_path / "r.csv"
        p.write_text("player,team,as_of\nA,KC,2020-01-01\n")
        assert R.from_manual_csv(p)[0].as_of == date(2020, 1, 1)

    def test_a_yaml_mapping_loads(self, tmp_path):
        p = tmp_path / "r.yaml"
        p.write_text("Patrick Mahomes: KC\nCourtland Sutton: DEN\n")
        teams = {e.player: e.team for e in R.from_manual_csv(p)}
        assert teams["Courtland Sutton"] == "DEN"

    def test_rows_without_a_team_are_skipped(self, tmp_path):
        p = tmp_path / "r.csv"
        p.write_text("player,team\nA,\n,KC\nB,SF\n")
        assert [e.player for e in R.from_manual_csv(p)] == ["B"]

    def test_a_missing_file_says_so(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            R.from_manual_csv(tmp_path / "nope.csv")


class TestFromGameLogs:
    def log(self, player, team, sport="nfl", game="g1", when=None, market="m", value=1):
        from betedge.correlation import GameLogRow

        return GameLogRow(game, sport, player, team, market, value, when or "")

    def test_it_harvests_the_team_column_the_fitter_already_needs(self):
        entries = R.from_game_logs([
            self.log("Mahomes", "KC", when="2026-09-14"),
            self.log("Kelce", "KC", when="2026-09-14"),
        ])
        assert {e.player: e.team for e in entries} == {"Mahomes": "KC", "Kelce": "KC"}
        assert all(e.source == R.SOURCE_GAME_LOGS for e in entries)

    def test_the_latest_appearance_wins_so_a_trade_self_corrects(self):
        entries = R.from_game_logs([
            self.log("A", "NYJ", game="g1", when="2026-09-01"),
            self.log("A", "KC", game="g2", when="2026-09-14"),
        ])
        assert entries[0].team == "KC"
        assert entries[0].as_of == date(2026, 9, 14)

    def test_an_undated_log_says_so_in_its_source(self):
        entries = R.from_game_logs([self.log("A", "KC")])
        assert "undated" in entries[0].source
        assert entries[0].as_of == datetime.now(timezone.utc).date()

    def test_rows_without_a_team_are_ignored(self):
        assert R.from_game_logs([self.log("A", "")]) == []

    def test_sports_are_kept_apart(self):
        entries = R.from_game_logs([
            self.log("A", "KC", sport="nfl"),
            self.log("A", "LAD", sport="mlb"),
        ])
        assert {(e.sport, e.team) for e in entries} == {("nfl", "KC"), ("mlb", "LAD")}


NFLVERSE_CSV = """season,team,position,depth_chart_position,status,full_name
2026,KC,QB,QB,ACT,Patrick Mahomes
2026,KC,TE,TE,ACT,Travis Kelce
2026,DEN,WR,WR,RES,Courtland Sutton
2026,SF,RB,RB,CUT,Somebody Released
2026,,WR,WR,ACT,No Team
2026,PHI,WR,WR,ACT,
"""


class TestNflverseProvider:
    def test_it_parses_the_release_format(self):
        entries = R.nflverse_rosters(
            season=2026, today=TODAY, opener=lambda url: NFLVERSE_CSV
        )
        teams = {e.player: e.team for e in entries}
        assert teams["Patrick Mahomes"] == "KC"
        assert entries[0].source == "nflverse"
        assert entries[0].position == "QB"

    def test_a_released_player_is_dropped(self):
        entries = R.nflverse_rosters(
            season=2026, today=TODAY, opener=lambda url: NFLVERSE_CSV
        )
        assert "Somebody Released" not in {e.player for e in entries}

    def test_a_reserve_player_keeps_his_club(self):
        entries = R.nflverse_rosters(
            season=2026, today=TODAY, opener=lambda url: NFLVERSE_CSV
        )
        assert {e.player: e.team for e in entries}["Courtland Sutton"] == "DEN"

    def test_rows_missing_a_name_or_team_are_skipped(self):
        entries = R.nflverse_rosters(
            season=2026, today=TODAY, opener=lambda url: NFLVERSE_CSV
        )
        assert len(entries) == 3

    def test_it_requests_the_season_it_was_asked_for(self):
        seen = []

        def opener(url):
            seen.append(url)
            return NFLVERSE_CSV

        R.nflverse_rosters(season=2024, today=TODAY, opener=opener)
        assert "roster_2024.csv" in seen[0]

    def test_it_falls_back_to_the_previous_season(self):
        tried = []

        def opener(url):
            tried.append(url)
            if "2026" in url:
                raise RuntimeError("404")
            return NFLVERSE_CSV

        entries = R.nflverse_rosters(today=TODAY, opener=opener)
        assert len(tried) == 2
        assert entries

    def test_a_dead_feed_raises_rather_than_returning_nonsense(self):
        def opener(url):
            raise RuntimeError("no route to host")

        with pytest.raises(RuntimeError, match="carry on with whatever is cached"):
            R.nflverse_rosters(today=TODAY, opener=opener)

    def test_the_league_year_turns_over_in_march(self):
        assert R.nfl_season_for(date(2026, 1, 20)) == 2025
        assert R.nfl_season_for(date(2026, 9, 15)) == 2026

    def test_only_leagues_with_a_verified_feed_have_one(self):
        assert R.provider_name("americanfootball_nfl") == "nflverse"
        assert R.provider_name("baseball_mlb") is None
        assert R.fetch_provider("baseball_mlb") == []


class TestRefresh:
    def test_it_fetches_and_stores(self, tmp_path):
        db = Database(tmp_path / "t.db")
        report = R.refresh_providers(
            db, ["americanfootball_nfl"], opener=lambda url: NFLVERSE_CSV
        )
        assert report.refreshed["americanfootball_nfl"] == 3
        assert len(db.roster_rows("americanfootball_nfl")) == 3
        db.close()

    def test_a_warm_snapshot_is_not_refetched(self, tmp_path):
        db = Database(tmp_path / "t.db")
        R.refresh_providers(db, ["americanfootball_nfl"],
                            opener=lambda url: NFLVERSE_CSV)
        calls = []

        def opener(url):
            calls.append(url)
            return NFLVERSE_CSV

        report = R.refresh_providers(db, ["americanfootball_nfl"],
                                     refresh_days=3, opener=opener)
        assert calls == []
        assert "under the 3-day refresh interval" in report.skipped["americanfootball_nfl"]
        db.close()

    def test_a_cold_snapshot_is_refetched(self, tmp_path):
        db = Database(tmp_path / "t.db")
        now = datetime.now(timezone.utc)
        R.refresh_providers(db, ["americanfootball_nfl"], now=now - timedelta(days=10),
                            opener=lambda url: NFLVERSE_CSV)
        report = R.refresh_providers(db, ["americanfootball_nfl"], refresh_days=3,
                                     now=now, opener=lambda url: NFLVERSE_CSV)
        assert report.refreshed
        db.close()

    def test_force_ignores_the_interval(self, tmp_path):
        db = Database(tmp_path / "t.db")
        R.refresh_providers(db, ["americanfootball_nfl"],
                            opener=lambda url: NFLVERSE_CSV)
        report = R.refresh_providers(db, ["americanfootball_nfl"], force=True,
                                     opener=lambda url: NFLVERSE_CSV)
        assert report.refreshed
        db.close()

    def test_a_failure_is_recorded_and_never_raised(self, tmp_path):
        db = Database(tmp_path / "t.db")

        def dead(url):
            raise RuntimeError("no route to host")

        report = R.refresh_providers(db, ["americanfootball_nfl"], opener=dead)
        assert report.errors
        assert not report.refreshed
        db.close()

    def test_a_sport_with_no_feed_is_skipped_quietly(self, tmp_path):
        db = Database(tmp_path / "t.db")
        report = R.refresh_providers(db, ["baseball_mlb"], opener=lambda url: "")
        assert "no roster feed" in report.skipped["baseball_mlb"]
        assert not report.errors
        db.close()

    def test_a_refresh_replaces_rather_than_accumulates(self, tmp_path):
        # A traded player must not be left on both clubs, which would read
        # as two players sharing a name and lose him entirely.
        db = Database(tmp_path / "t.db")
        R.refresh_providers(db, ["americanfootball_nfl"],
                            opener=lambda url: NFLVERSE_CSV)
        moved = NFLVERSE_CSV.replace("2026,KC,QB,QB,ACT,Patrick Mahomes",
                                     "2026,NYJ,QB,QB,ACT,Patrick Mahomes")
        R.refresh_providers(db, ["americanfootball_nfl"], force=True,
                            opener=lambda url: moved)
        book = R.load_book(db)
        assert book.team_for("americanfootball_nfl", "Patrick Mahomes") == "NYJ"
        db.close()


class TestLoadBook:
    def test_stored_layers_and_the_manual_file_combine(self, tmp_path):
        db = Database(tmp_path / "t.db")
        R.refresh_providers(db, ["americanfootball_nfl"],
                            opener=lambda url: NFLVERSE_CSV)
        manual = tmp_path / "mine.csv"
        manual.write_text("player,team\nPatrick Mahomes,TRADED\n")
        book = R.load_book(db, manual_path=str(manual))
        assert book.team_for("americanfootball_nfl", "Patrick Mahomes") == "TRADED"
        assert book.team_for("americanfootball_nfl", "Travis Kelce") == "KC"
        db.close()

    def test_an_empty_database_gives_an_empty_book(self, tmp_path):
        db = Database(tmp_path / "t.db")
        assert len(R.load_book(db)) == 0
        db.close()


class TestStructuralInference:
    def leg(self, player, market, event="g1", team=None):
        from fixtures import make_leg

        return make_leg(selection=player, market=market, event_id=event, team=team)

    def test_two_quarterbacks_in_one_game_are_opponents(self):
        legs = [self.leg("Mahomes", "player_pass_yds"),
                self.leg("Nix", "player_pass_yds")]
        sides = R.infer_sides(legs)
        assert len(sides) == 2
        assert sides["Mahomes"] != sides["Nix"]

    def test_the_relation_comes_out_as_opposing(self):
        from dataclasses import replace

        from betedge import correlation as C

        legs = [self.leg("Mahomes", "player_pass_yds"),
                self.leg("Nix", "player_pass_yds")]
        sides = R.infer_sides(legs)
        legs = [replace(l, team=sides[l.selection],
                        team_source=R.SOURCE_STRUCTURAL) for l in legs]
        assert C.relation_between(legs[0], legs[1]) == C.OPPOSING_TEAM

    def test_an_inferred_side_is_never_compared_against_a_real_club(self):
        # The labels are true statements about one game and meaningless
        # outside it. Comparing one to "KC" would read as "different team"
        # and report a relation nothing supports.
        from betedge import correlation as C

        inferred = self.leg("Nix", "player_pass_yds")
        inferred = type(inferred)(**{**inferred.__dict__, "team": "g1#B",
                                     "team_source": R.SOURCE_STRUCTURAL})
        real = self.leg("Mahomes", "player_rush_yds", team="KC")
        assert C.relation_between(real, inferred) == C.SAME_GAME

    def test_it_never_overwrites_a_known_club(self):
        legs = [self.leg("Mahomes", "player_pass_yds", team="KC"),
                self.leg("Nix", "player_pass_yds")]
        assert R.infer_sides(legs) == {}

    def test_one_player_alone_infers_nothing(self):
        assert R.infer_sides([self.leg("Mahomes", "player_pass_yds")]) == {}

    def test_three_players_break_the_one_per_team_assumption(self):
        legs = [self.leg(n, "player_pass_yds") for n in ("A", "B", "C")]
        assert R.infer_sides(legs) == {}

    def test_a_market_that_is_not_one_per_team_infers_nothing(self):
        legs = [self.leg("A", "player_reception_yds"),
                self.leg("B", "player_reception_yds")]
        assert R.infer_sides(legs) == {}

    def test_different_games_do_not_pair_up(self):
        legs = [self.leg("A", "player_pass_yds", event="g1"),
                self.leg("B", "player_pass_yds", event="g2")]
        assert R.infer_sides(legs) == {}

    def test_the_markets_it_trusts_are_genuinely_one_per_team(self):
        for market in ("player_pass_yds", "pitcher_strikeouts", "player_total_saves"):
            assert market in R.ONE_PER_TEAM_MARKETS
        for market in ("player_reception_yds", "player_rush_yds", "player_points"):
            assert market not in R.ONE_PER_TEAM_MARKETS
