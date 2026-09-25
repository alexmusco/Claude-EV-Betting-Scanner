"""
Settling a bet from what actually happened.

The invariant under all of this: a bet settled against the WRONG PLAYER
is worse than one left unsettled. The unsettled bet makes the sample
smaller and says so. The mis-settled one corrupts the single number the
paper ledger exists to produce, looks entirely plausible, and can never
be found again from the numbers themselves.
"""

import pytest

from betedge import results as R


def game(player, team="SF", opponent="MIA", week=2, **stats):
    return R.PlayerGame(
        player=player, team=team, opponent=opponent, season=2026,
        week=week, game_id=f"2026_{week:02d}_{team}", stats=stats)


@pytest.fixture
def book():
    return R.ResultBook([
        game("Deebo Samuel", receptions="5", receiving_yards="31",
             rushing_yards="12", rushing_tds="0", receiving_tds="1"),
        game("Chris Olave", team="NO", opponent="ATL", receptions="8",
             receiving_yards="112", receiving_tds="1"),
        game("Rashee Rice", team="KC", opponent="IND", week=1,
             receptions="4", receiving_yards="60"),
        game("Rashee Rice", team="KC", opponent="LV", week=2,
             receptions="4", receiving_yards="55"),
    ])


class TestNameMatching:
    def test_a_suffix_does_not_break_a_match(self):
        assert R.normalise_name("Oronde Gadsden II") == \
            R.normalise_name("Oronde Gadsden")

    def test_punctuation_and_accents_fold(self):
        assert R.normalise_name("Amon-Ra St. Brown") == "amon ra st brown"
        assert R.normalise_name("Nikola Jokić") == "nikola jokic"

    def test_an_unknown_player_is_refused_with_a_reason(self, book):
        found, reason = book.find("Nobody At All")
        assert found == []
        assert "no player called" in reason

    def test_two_players_on_one_name_is_ambiguous_not_a_coin_flip(self):
        """
        Dropping suffixes can merge a father and son. That is resolved by
        REFUSING, never by picking one -- which is the only safe answer
        when the feed genuinely cannot tell them apart.
        """
        book = R.ResultBook([
            game("Marvin Harrison", team="ARI", receptions="3"),
            game("Marvin Harrison Jr.", team="IND", receptions="9"),
        ])
        settled = R.settle_bet(book, "player_receptions",
                               "Marvin Harrison", "Over", 5.5, week=2)
        assert not settled.settled
        assert "ambiguous" in settled.reason

    def test_a_team_narrows_an_otherwise_ambiguous_name(self):
        book = R.ResultBook([
            game("Marvin Harrison", team="ARI", receptions="3"),
            game("Marvin Harrison Jr.", team="IND", receptions="9"),
        ])
        settled = R.settle_bet(book, "player_receptions", "Marvin Harrison",
                               "Over", 5.5, week=2, teams=["IND", "HOU"])
        assert settled.status == R.WON
        assert settled.actual == 9


class TestSettling:
    def test_over_and_under(self, book):
        assert R.settle_bet(book, "player_receptions", "Deebo Samuel",
                            "Over", 4.5, week=2).status == R.WON
        assert R.settle_bet(book, "player_receptions", "Deebo Samuel",
                            "Under", 4.5, week=2).status == R.LOST

    def test_an_exact_landing_is_a_push_not_a_loss(self, book):
        """
        Folding a push into either column biases every measured ROI in
        the direction it was folded.
        """
        assert R.settle_bet(book, "player_receptions", "Deebo Samuel",
                            "Over", 5.0, week=2).status == R.PUSH
        assert R.settle_bet(book, "player_receptions", "Deebo Samuel",
                            "Under", 5.0, week=2).status == R.PUSH

    def test_a_combined_market_sums_its_columns(self, book):
        # 31 receiving + 12 rushing + 0 passing = 43
        settled = R.settle_bet(book, "player_pass_rush_reception_yds",
                               "Deebo Samuel", "Over", 42.5, week=2)
        assert settled.status == R.WON
        assert settled.actual == 43

    def test_anytime_td_needs_no_line(self, book):
        assert R.settle_bet(book, "player_anytime_td", "Deebo Samuel",
                            "Yes", None, week=2).status == R.WON
        assert R.settle_bet(book, "player_anytime_td", "Deebo Samuel",
                            "No", None, week=2).status == R.LOST

    def test_an_alternate_market_key_settles_the_same(self, book):
        assert R.settle_bet(book, "player_receptions_alternate",
                            "Deebo Samuel", "Over", 4.5, week=2).status == R.WON

    def test_an_unmapped_market_refuses_rather_than_guessing(self, book):
        settled = R.settle_bet(book, "player_kicking_points",
                               "Deebo Samuel", "Over", 7.5, week=2)
        assert not settled.settled
        assert "no results column is mapped" in settled.reason

    def test_a_bet_with_no_line_refuses(self, book):
        settled = R.settle_bet(book, "player_receptions", "Deebo Samuel",
                               "Over", None, week=2)
        assert not settled.settled
        assert "no line" in settled.reason

    def test_the_right_week_is_used(self, book):
        assert R.settle_bet(book, "player_reception_yds", "Rashee Rice",
                            "Over", 57.5, week=1).status == R.WON
        assert R.settle_bet(book, "player_reception_yds", "Rashee Rice",
                            "Over", 57.5, week=2).status == R.LOST


class TestTheFeedLag:
    """
    nflverse publishes a day or two after the games. A week that is
    barely populated makes every bet in it refuse with "does not
    appear", which reads exactly like a broken matcher and is nothing of
    the sort.
    """

    def book(self):
        full = [game(f"Player {i}", week=w)
                for w in (1, 2) for i in range(100)]
        thin = [game("Late Arrival", week=3)]
        return R.ResultBook(full + thin)

    def test_a_barely_published_week_is_identified(self):
        assert self.book().incomplete_weeks() == [(3, 1, 100)]

    def test_a_full_week_is_not_flagged(self):
        assert all(w != 1 for w, _n, _f in self.book().incomplete_weeks())

    def test_the_refusal_blames_the_feed_not_the_player(self):
        _found, reason = self.book().find("Player 1", week=3)
        assert "has not caught up" in reason
        assert "does not appear" not in reason

    def test_a_genuinely_missing_player_still_says_so(self):
        _found, reason = self.book().find("Player 1", week=2)
        assert reason == ""      # week 2 is full, and he is in it
        _found, reason = self.book().find("Ghost", week=2)
        assert "no player called" in reason


class TestReadingTheFeed:
    def test_an_empty_file_is_an_error_not_an_empty_book(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("season,week,player_display_name\n")
        with pytest.raises(R.ResultsError, match="zero player-games"):
            R.read_weekly_stats(path)

    def test_a_missing_file_is_an_error(self, tmp_path):
        with pytest.raises(R.ResultsError, match="does not exist"):
            R.read_weekly_stats(tmp_path / "nope.csv")

    def test_rows_without_a_name_are_skipped_not_fatal(self, tmp_path):
        path = tmp_path / "s.csv"
        path.write_text(
            "season,week,player_display_name,team,opponent_team,receptions\n"
            "2026,2,,SF,MIA,3\n"
            "2026,2,Deebo Samuel,SF,MIA,5\n")
        assert len(R.read_weekly_stats(path)) == 1
