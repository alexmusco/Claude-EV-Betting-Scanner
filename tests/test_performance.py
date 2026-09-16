"""
Comparing the two strategies.

The tests that matter are the ones about restraint. A comparison that
declares a winner from forty bets is worse than no comparison at all,
because it looks like an answer — so most of what follows pins down when
the tool refuses to call it.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from betedge import performance as P
from betedge.db import Database


def bets(n, stake=50.0, win_pnl=45.0, lose_pnl=-50.0, win_rate=0.5, ev=0.03):
    """A deterministic run of settled bets at a given hit rate."""
    wins = round(n * win_rate)
    return [
        {"stake": stake, "pnl": win_pnl if i < wins else lose_pnl, "ev_at_bet": ev}
        for i in range(n)
    ]


class TestSummarise:
    def test_it_totals_the_basics(self):
        r = P.summarise("s", bets(10, win_rate=0.5))
        assert r.settled == 10
        assert r.staked == 500
        assert r.pnl == pytest.approx(5 * 45 - 5 * 50)

    def test_roi_is_profit_over_turnover(self):
        r = P.summarise("s", [{"stake": 100, "pnl": 10}])
        assert r.roi == pytest.approx(0.10)

    def test_a_zero_stake_row_is_skipped_not_divided_by(self):
        r = P.summarise("s", [{"stake": 0, "pnl": 5}, {"stake": 10, "pnl": 1}])
        assert r.settled == 1

    def test_no_bets_means_no_roi_rather_than_zero(self):
        assert P.summarise("s", []).roi is None

    def test_modelled_pnl_is_stake_weighted_ev(self):
        r = P.summarise("s", [{"stake": 100, "pnl": 0, "ev_at_bet": 0.05}])
        assert r.modelled_pnl == pytest.approx(5.0)

    def test_modelled_pnl_is_absent_when_no_row_carries_an_ev(self):
        assert P.summarise("s", [{"stake": 10, "pnl": 1}]).modelled_pnl is None

    def test_realisation_is_actual_over_modelled(self):
        r = P.summarise("s", [{"stake": 100, "pnl": 10, "ev_at_bet": 0.05}])
        assert r.realisation == pytest.approx(2.0)

    def test_clv_is_averaged_and_counted(self):
        r = P.summarise("s", bets(4), clv_values=[0.02, -0.01, 0.03])
        assert r.avg_clv == pytest.approx(0.04 / 3)
        assert r.clv_beat_rate == pytest.approx(2 / 3)

    def test_it_reads_sqlite_rows_as_well_as_dicts(self, tmp_path):
        db = Database(tmp_path / "t.db")
        bid = db.place_bet(stake=50, price=1.91, book="dk", ev_at_bet=0.03)
        db.settle_bet(bid, "won")
        rows = db.conn.execute(
            "SELECT stake, pnl, ev_at_bet FROM bets"
        ).fetchall()
        assert P.summarise("s", rows).settled == 1
        db.close()


class TestRoiInterval:
    def test_a_thin_sample_gets_no_interval_at_all(self):
        assert P.summarise("s", bets(5)).roi_interval() is None

    def test_the_interval_brackets_the_point_estimate(self):
        r = P.summarise("s", bets(200, win_rate=0.55))
        lo, hi = r.roi_interval()
        assert lo < r.roi < hi

    def test_more_bets_narrow_it(self):
        few = P.summarise("s", bets(40, win_rate=0.55)).roi_interval()
        many = P.summarise("s", bets(2000, win_rate=0.55)).roi_interval()
        assert (many[1] - many[0]) < (few[1] - few[0])

    def test_a_lumpy_payoff_widens_it(self):
        # Same edge, paid as a rare 10x instead of a frequent near-even
        # bet. This is the whole reason parlay ROI is hard to measure.
        steady = P.summarise("s", bets(200, win_rate=0.55))
        lumpy = P.summarise(
            "s", bets(200, win_pnl=450.0, lose_pnl=-50.0, win_rate=0.12)
        )
        assert lumpy.return_sd > steady.return_sd
        a, b = steady.roi_interval(), lumpy.roi_interval()
        assert (b[1] - b[0]) > (a[1] - a[0])

    def test_it_is_reproducible(self):
        r = P.summarise("s", bets(100, win_rate=0.55))
        assert r.roi_interval() == r.roi_interval()

    def test_a_wider_confidence_gives_a_wider_interval(self):
        r = P.summarise("s", bets(300, win_rate=0.55))
        narrow = r.roi_interval(confidence=0.50)
        wide = r.roi_interval(confidence=0.99)
        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


class TestBetsNeeded:
    def test_it_scales_with_the_spread_of_outcomes(self):
        steady = P.summarise("s", bets(200, win_rate=0.55))
        lumpy = P.summarise(
            "s", bets(200, win_pnl=450.0, lose_pnl=-50.0, win_rate=0.12)
        )
        assert lumpy.bets_needed() > steady.bets_needed()

    def test_an_even_money_bet_needs_roughly_fifteen_hundred(self):
        # sd is about 1.0 per bet, so resolving five points takes
        # (1.96 / 0.05)^2 ~ 1,537 bets. Worth knowing before reading any
        # ROI as a verdict.
        needed = P.summarise("s", bets(500, win_rate=0.5)).bets_needed(0.05)
        assert 1_000 < needed < 2_500

    def test_a_finer_resolution_needs_more(self):
        r = P.summarise("s", bets(200, win_rate=0.55))
        assert r.bets_needed(0.01) > r.bets_needed(0.05)

    def test_one_bet_has_no_spread_to_measure(self):
        assert P.summarise("s", bets(1)).bets_needed() is None


class TestComparison:
    def two(self, a_rate=0.55, b_rate=0.30, n=200):
        return P.Comparison([
            P.summarise("single bets", bets(n, win_rate=a_rate), clv_values=[0.02] * n),
            P.summarise(
                "multi-leg",
                bets(n, win_pnl=100.0, lose_pnl=-50.0, win_rate=b_rate),
                clv_values=[0.01] * n,
            ),
        ])

    def test_it_ranks_by_roi(self):
        assert self.two().ranked[0].name == "single bets"

    def test_it_refuses_to_call_a_thin_sample(self):
        thin = P.Comparison([
            P.summarise("single bets", bets(4)),
            P.summarise("multi-leg", bets(4)),
        ])
        verdict = thin.verdict()
        assert "Too few settled bets" in verdict
        assert "closing-line value" in verdict

    def test_it_says_so_when_one_side_has_nothing(self):
        lopsided = P.Comparison([
            P.summarise("single bets", bets(50)),
            P.summarise("multi-leg", []),
        ])
        assert "Nothing settled yet" in lopsided.verdict()

    def test_overlapping_intervals_are_reported_as_no_evidence(self):
        # Identical strategies: the tool must not pick a winner.
        same = P.Comparison([
            P.summarise("single bets", bets(200, win_rate=0.52)),
            P.summarise("multi-leg", bets(200, win_rate=0.52)),
        ])
        assert same.intervals_overlap
        assert "not evidence of anything yet" in same.verdict()

    def test_a_genuinely_large_gap_is_allowed_to_register(self):
        wide = P.Comparison([
            P.summarise("single bets", bets(4000, win_rate=0.70)),
            P.summarise("multi-leg", bets(4000, win_rate=0.30)),
        ])
        assert wide.intervals_overlap is False
        assert "do not overlap" in wide.verdict()

    def test_even_a_real_signal_is_hedged_against_clv(self):
        wide = P.Comparison([
            P.summarise("single bets", bets(4000, win_rate=0.70)),
            P.summarise("multi-leg", bets(4000, win_rate=0.30)),
        ])
        assert "closing-line value" in wide.verdict()

    def test_the_verdict_quotes_the_sample_size_needed(self):
        assert "settled bets" in self.two().verdict()

    def test_overlap_is_unknown_when_an_interval_cannot_be_built(self):
        thin = P.Comparison([
            P.summarise("single bets", bets(4)),
            P.summarise("multi-leg", bets(400)),
        ])
        assert thin.intervals_overlap is None


def seed_opportunity(db, opp_id=1):
    """
    A flagged opportunity, so bets placed against it count as the model's.

    A bet with no opportunity_id was never surfaced by a scan, and the
    comparison treats it as the user's own pick rather than the model's --
    so a test about the model has to give its bets something to point at.
    """
    db.conn.execute(
        """INSERT INTO opportunities (id, scan_id, scanned_at, sport, event_id,
               commence_time, market, selection, line, side, sharp_book,
               sharp_price_taken_side, sharp_price_other_side, sharp_overround,
               fair_prob, fair_price, devig_method, devig_spread, soft_book,
               soft_price, ev)
           VALUES (?,1,'2026-09-15T00:00:00+00:00','baseball_mlb','e1',
                   '2026-09-15T18:00:00+00:00','pitcher_strikeouts','A Pitcher',
                   5.5,'Over','pinnacle',1.9,1.9,0.03,0.55,1.82,'worst_case',
                   0.004,'draftkings',1.91,0.05)""",
        (opp_id,),
    )
    db.conn.commit()
    return opp_id


class TestDatabaseIntegration:
    def seeded(self, tmp_path):
        db = Database(tmp_path / "t.db")
        opp = seed_opportunity(db)
        for i in range(12):
            bid = db.place_bet(opp, stake=50, price=1.91, book="dk")
            db.settle_bet(bid, "won" if i % 2 == 0 else "lost")
            db.record_closing_line(
                opportunity_id=opp, bet_id=bid, sharp_price_taken=1.9,
                sharp_price_other=1.9, fair_prob_close=0.54, price_taken=1.91,
            )
        db.conn.execute(
            """INSERT INTO parlay_tickets (created_at, product, book, kind,
                   n_legs, joint_prob, ev, payout_all_hit)
               VALUES ('2026-09-15T00:00:00+00:00','underdog_standard',
                       'underdog','pickem',2,0.36,0.08,3.0)"""
        )
        db.conn.commit()
        for i in range(12):
            pb = db.place_parlay_bet(1, stake=20)
            db.settle_parlay_bet(pb, legs_hit=2 if i % 3 == 0 else 1)
        return db

    def test_both_strategies_come_back(self, tmp_path):
        db = self.seeded(tmp_path)
        names = [s.name for s in db.compare_strategies().strategies]
        assert names == ["single bets", "multi-leg"]
        db.close()

    def test_a_bet_logged_by_hand_is_still_a_single_bet(self, tmp_path):
        # How a bet reached the ledger is bookkeeping, not strategy. The
        # comparison is one-leg betting against multi-leg betting, and
        # splitting on whether the scanner printed a row for it answers a
        # different question while making both halves too small to answer
        # anything.
        db = self.seeded(tmp_path)
        mine = db.place_bet(stake=10, price=2.22, book="draftkings",
                            selection="Max Fried", side="Under", line=4.5,
                            market="pitcher_strikeouts", sport="baseball_mlb")
        db.settle_bet(mine, "lost")
        by_name = {s.name: s for s in db.compare_strategies().strategies}
        assert [s.name for s in db.compare_strategies().strategies] == [
            "single bets", "multi-leg"
        ]
        assert by_name["single bets"].settled == 13
        db.close()

    def test_a_hand_logged_bet_does_not_move_the_realisation_ratio(self, tmp_path):
        """
        It has no modelled edge, so it belongs in neither half of
        `realised / modelled`. Counting its P&L in the numerator while it
        contributes nothing to the denominator is how a losing hand-logged
        bet would make the model look worse than it is -- or a winning one
        make it look better.
        """
        db = self.seeded(tmp_path)
        before = {s.name: s for s in db.compare_strategies().strategies}
        ratio_before = before["single bets"].realisation
        covered_before = before["single bets"].modelled_settled

        mine = db.place_bet(stake=50, price=2.22, book="draftkings")
        db.settle_bet(mine, "won")
        after = {s.name: s for s in db.compare_strategies().strategies}

        assert after["single bets"].settled == covered_before + 1
        assert after["single bets"].modelled_settled == covered_before
        assert after["single bets"].realisation == pytest.approx(ratio_before)
        # But it is still in the record it belongs in.
        assert after["single bets"].pnl > before["single bets"].pnl
        db.close()

    def test_every_single_bets_clv_lands_on_one_record(self, tmp_path):
        db = self.seeded(tmp_path)
        mine = db.place_bet(stake=10, price=2.22, book="draftkings")
        db.settle_bet(mine, "lost")
        db.record_closing_line(
            opportunity_id=None, bet_id=mine, sharp_price_taken=2.1,
            sharp_price_other=1.8, fair_prob_close=0.40, price_taken=2.22,
        )
        by_name = {s.name: s for s in db.compare_strategies().strategies}
        assert len(by_name["single bets"].clv_values) == 13
        db.close()

    def test_the_two_tables_stay_separate(self, tmp_path):
        db = self.seeded(tmp_path)
        single, parlay = db.compare_strategies().strategies
        assert single.settled == 12 and single.staked == 600
        assert parlay.settled == 12 and parlay.staked == 240
        db.close()

    def test_a_pending_bet_is_counted_but_not_settled(self, tmp_path):
        db = self.seeded(tmp_path)
        db.place_bet(1, stake=50, price=1.91, book="dk")
        single = db.compare_strategies().strategies[0]
        assert single.pending == 1
        assert single.settled == 12
        db.close()

    def test_a_void_bet_is_excluded_from_both_sides(self, tmp_path):
        # A returned stake is not a result, and counting it would dilute
        # every rate with an outcome that never happened.
        db = self.seeded(tmp_path)
        bid = db.place_bet(1, stake=500, price=1.91, book="dk")
        db.settle_bet(bid, "void")
        assert db.compare_strategies().strategies[0].settled == 12
        db.close()

    def test_single_bet_clv_is_picked_up(self, tmp_path):
        db = self.seeded(tmp_path)
        assert len(db.compare_strategies().strategies[0].clv_values) == 12
        db.close()

    def test_ticket_clv_without_a_bet_is_not_counted_as_earned(self, tmp_path):
        # Closing lines are captured for tickets never entered -- that is
        # the model-quality signal, not a record of what was won.
        db = self.seeded(tmp_path)
        db.record_parlay_closing_line(
            ticket_id=1, parlay_bet_id=None, legs_captured=2, n_legs=2,
            joint_prob_close=0.30, joint_prob_at_bet=0.36,
            ev_close=-0.10, ev_at_bet=0.08,
        )
        assert db.compare_strategies().strategies[1].clv_values == []
        db.close()

    def test_an_empty_database_says_so_rather_than_failing(self, tmp_path):
        db = Database(tmp_path / "t.db")
        assert "Nothing settled yet" in db.compare_strategies().verdict()
        db.close()
