"""
The diagnostic view.

A scan that flags nothing reports only rejection counts, which say what
tripped but not by how much. These tests pin the distinction the command
exists to make: a bar set slightly too high looks identical to a market
with no edge in it, and the two call for opposite responses.
"""

from datetime import timedelta

import pytest

from betedge import report as R
from betedge.scan import QuoteAssessment, scan
from conftest import NOW, FakeClient
from test_scan import mlb_prop, nfl_game


def assessment(ev, required_ev=0.02, liquidity=0.95, overround=0.03,
               tier="mainline", price=1.95):
    return QuoteAssessment(
        sport="baseball_mlb", matchup="A @ B", market="h2h", tier=tier,
        description="A ML", book="draftkings", soft_price=price,
        sharp_price=1.95, overround=overround, liquidity=liquidity,
        ev=ev, required_ev=required_ev, minutes_to_start=300,
    )


class TestCollection:
    def test_rejected_quotes_are_kept(self, cfg, now):
        """The whole point: a no-edge board still has to yield data."""
        client = FakeClient(bulk_odds={"baseball_mlb": [
            nfl_game(dk_home_price=1.80)]})   # firmly negative EV
        cfg.core_sports = ["baseball_mlb"]
        result = scan(cfg, client, now=now, collect=True)
        assert result.opportunities == []
        assert len(result.assessments) == 2, "both DK sides were priced"
        assert all(a.ev < 0 for a in result.assessments)

    def test_nothing_is_collected_unless_asked(self, cfg, now):
        client = FakeClient(bulk_odds={"baseball_mlb": [nfl_game()]})
        cfg.core_sports = ["baseball_mlb"]
        assert scan(cfg, client, now=now).assessments == []

    def test_flagged_quotes_are_collected_too(self, cfg, now):
        client = FakeClient(bulk_odds={"baseball_mlb": [
            nfl_game(dk_home_price=2.10)]})
        cfg.core_sports = ["baseball_mlb"]
        result = scan(cfg, client, now=now, collect=True)
        assert result.opportunities
        assert any(a.ev >= a.required_ev for a in result.assessments)

    def test_props_are_collected(self, cfg, now):
        client = FakeClient(
            events_by_sport={"baseball_mlb": [
                {"id": "mlb1",
                 "commence_time": (NOW + timedelta(hours=6)).isoformat()}]},
            event_odds={("baseball_mlb", "mlb1"): mlb_prop(dk_over=2.00)},
        )
        cfg.sports = ["baseball_mlb"]
        cfg.prop_markets = {"baseball_mlb": ["pitcher_strikeouts"]}
        result = scan(cfg, client, now=now, collect=True)
        assert result.assessments
        assert all(a.tier == "primary_prop" for a in result.assessments)

    def test_shortfall_signs_the_gap(self):
        assert assessment(ev=0.01, required_ev=0.02).shortfall == pytest.approx(0.01)
        assert assessment(ev=0.05, required_ev=0.02).shortfall == pytest.approx(-0.03)


class TestDistributionReport:
    def test_an_empty_board_says_so(self):
        text = R.distribution_report([], min_ev=0.02)
        assert "Nothing was priced" in text

    def test_it_separates_a_near_miss_board_from_a_dead_one(self):
        """The distinction the command exists for."""
        near = R.distribution_report([assessment(ev=e) for e in
                                      (0.019, 0.018, 0.015)], min_ev=0.02)
        dead = R.distribution_report([assessment(ev=e) for e in
                                      (-0.06, -0.055, -0.07)], min_ev=0.02)
        assert "+1.90%" in near
        assert "-5.50%" in dead
        assert "at +1.0%:   3 quote(s)" in near
        assert "at +1.0%:   0 quote(s)" in dead

    def test_it_marks_the_configured_bar(self):
        text = R.distribution_report([assessment(ev=0.01)], min_ev=0.02)
        assert "<- your min_ev" in text

    def test_it_counts_what_actually_cleared(self):
        rows = [assessment(ev=0.05), assessment(ev=0.025, required_ev=0.03),
                assessment(ev=-0.01)]
        text = R.distribution_report(rows, min_ev=0.02)
        assert "after the liquidity adjustment: 1 flagged" in text

    def test_it_reports_overround_by_tier(self):
        rows = [assessment(ev=-0.04, overround=0.03, tier="mainline"),
                assessment(ev=-0.05, overround=0.055, tier="primary_prop")]
        text = R.distribution_report(rows, min_ev=0.02)
        assert "mainline" in text and "primary_prop" in text

    def test_near_misses_are_ordered_by_shortfall(self):
        rows = [assessment(ev=-0.05), assessment(ev=0.019), assessment(ev=-0.001)]
        text = R.distribution_report(rows, min_ev=0.02, top=3)
        block = text.split("CLOSEST")[1]
        assert block.index("+1.90%") < block.index("-0.10%") < block.index("-5.00%")

    def test_the_share_above_zero_is_reported(self):
        rows = [assessment(ev=0.01), assessment(ev=-0.01),
                assessment(ev=-0.02), assessment(ev=-0.03)]
        text = R.distribution_report(rows, min_ev=0.02)
        assert "1 of 4 priced above zero (25%)" in text


class TestFlagMarking:
    """
    A negative shortfall means cleared, which is correct and unreadable.
    Without a marker the table looks like a list of bets when only one or
    two rows are.
    """

    def _rows(self):
        return [
            assessment(ev=0.064, required_ev=0.038, price=2.22),   # cleared
            assessment(ev=0.028, required_ev=0.038),               # missed
            assessment(ev=-0.02, required_ev=0.021),               # missed
        ]

    def test_cleared_rows_are_marked(self):
        text = R.distribution_report(self._rows(), min_ev=0.02, top=5)
        # The trailing summary also says FLAG, so count the table rows only.
        table = text.split("CLOSEST")[1].split("Only the")[0]
        assert table.count("FLAG") == 1

    def test_the_count_is_stated_in_words(self):
        text = R.distribution_report(self._rows(), min_ev=0.02, top=5)
        assert "Only the 1 FLAG row(s) cleared" in text
        assert "not as a recommendation" in text

    def test_a_board_with_no_hits_says_so_plainly(self):
        rows = [assessment(ev=-0.03), assessment(ev=0.01, required_ev=0.028)]
        text = R.distribution_report(rows, min_ev=0.02)
        assert "Nothing cleared" in text
        assert "not bets" in text
        assert "FLAG" not in text.split("CLOSEST")[1]


class TestSideLean:
    def test_a_lean_is_surfaced(self):
        rows = [assessment(ev=0.01) for _ in range(5)]
        rows = [
            QuoteAssessment(
                sport="baseball_mlb", matchup="A @ B", market="pitcher_strikeouts",
                tier="primary_prop", description=f"Player {i} Under", book="draftkings",
                soft_price=2.0, sharp_price=1.95, overround=0.07, liquidity=0.75,
                ev=0.01, required_ev=0.028, minutes_to_start=200,
            )
            for i in range(5)
        ]
        text = R.distribution_report(rows, min_ev=0.02)
        assert "SIDE LEAN" in text and "Under 5" in text

    def test_negative_ev_quotes_are_not_counted(self):
        rows = [
            QuoteAssessment(
                sport="s", matchup="A @ B", market="m", tier="primary_prop",
                description=f"P{i} Under", book="draftkings", soft_price=2.0,
                sharp_price=1.95, overround=0.07, liquidity=0.75,
                ev=-0.05, required_ev=0.028, minutes_to_start=200,
            )
            for i in range(6)
        ]
        assert "SIDE LEAN" not in R.distribution_report(rows, min_ev=0.02)

    def test_too_few_positives_to_be_meaningful_is_silent(self):
        rows = [assessment(ev=0.01)]
        assert "SIDE LEAN" not in R.distribution_report(rows, min_ev=0.02)
