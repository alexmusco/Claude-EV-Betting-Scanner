"""
The multi-leg optimizer: payout structures, leg construction, guards,
search and staking.

The hand calculations in TestPayoutMath are the ones to read first. If the
payout vectors are wrong, every other number in this module is wrong in a
way that looks entirely plausible.
"""

import math
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from conftest import NOW, book, event_payload, outcome
from fixtures import (
    FLAT_STACK,
    KC_STACK,
    KC_TEAMS,
    ladder_event,
    make_leg,
    parlay_config,
    prop_event,
)

from betedge import copula, correlation as C, parlay as P
from betedge.scan import parse_event_odds


@pytest.fixture
def table():
    return P.PayoutTable.load()


@pytest.fixture
def priors():
    return C.PriorSet.load()


@pytest.fixture
def pcfg(tmp_path):
    return parlay_config(tmp_path)


def legs_from(payload, cfg, books=("underdog",), now=None, rejections=None,
              rosters=None):
    meta, quotes = parse_event_odds(payload)
    return P.build_legs(
        meta, quotes, cfg, list(books), now=now or NOW,
        rejections=rejections if rejections is not None else {},
        rosters=rosters,
    )


# --------------------------------------------------------------------------
# Payout structures
# --------------------------------------------------------------------------


class TestPayoutTable:
    def test_the_shipped_table_loads(self, table):
        assert "underdog_standard" in table.products
        assert "prizepicks_power" in table.products

    def test_the_pickem_products_ship_unverified(self, table):
        # They vary by state and change without notice, so shipping them
        # marked verified would be a lie the report then repeats.
        assert "underdog_standard" in table.unverified
        assert "prizepicks_flex" in table.unverified

    def test_an_unknown_product_names_what_is_available(self, table):
        with pytest.raises(ValueError, match="underdog_standard"):
            table.get("fanduel_something")

    def test_a_wrong_length_vector_is_rejected(self, tmp_path):
        bad = tmp_path / "p.yaml"
        bad.write_text(
            "products:\n  x:\n    book: b\n    kind: pickem\n"
            "    payouts:\n      3: [0, 0, 6.0]\n"
        )
        with pytest.raises(ValueError, match="it needs 4 entries"):
            P.PayoutTable.load(bad)

    def test_a_negative_payout_is_rejected(self, tmp_path):
        bad = tmp_path / "p.yaml"
        bad.write_text(
            "products:\n  x:\n    book: b\n    kind: pickem\n"
            "    payouts:\n      2: [0, -1, 3.0]\n"
        )
        with pytest.raises(ValueError, match="negative payout"):
            P.PayoutTable.load(bad)

    def test_an_unknown_kind_is_rejected(self, tmp_path):
        bad = tmp_path / "p.yaml"
        bad.write_text(
            "products:\n  x:\n    book: b\n    kind: teaser\n"
            "    payouts:\n      2: [0, 0, 3.0]\n"
        )
        with pytest.raises(ValueError, match="kind"):
            P.PayoutTable.load(bad)

    def test_a_dangling_reduces_to_is_rejected(self, tmp_path):
        bad = tmp_path / "p.yaml"
        bad.write_text(
            "products:\n  x:\n    book: b\n    kind: pickem\n"
            "    reduces_to: nowhere\n    payouts:\n      2: [0, 0, 3.0]\n"
        )
        with pytest.raises(ValueError, match="reduces_to"):
            P.PayoutTable.load(bad)


class TestPayoutMath:
    """Hand calculations against each shipped structure."""

    def ev_at(self, product, legs, p):
        """EV of `legs` independent legs at probability p, exactly."""
        return product.independent_ev(legs, p)

    def test_underdog_two_pick_by_hand(self, table):
        # 0.6 * 0.6 = 0.36 of the time it returns 3x. EV = 1.08 - 1.
        p = table.get("underdog_standard")
        assert self.ev_at(p, 2, 0.6) == pytest.approx(0.36 * 3.0 - 1.0)
        assert self.ev_at(p, 2, 0.6) == pytest.approx(0.08)

    def test_underdog_three_pick_by_hand(self, table):
        p = table.get("underdog_standard")
        assert self.ev_at(p, 3, 0.6) == pytest.approx(0.216 * 6.0 - 1.0)
        assert self.ev_at(p, 3, 0.6) == pytest.approx(0.296)

    def test_underdog_four_and_five_pick_by_hand(self, table):
        p = table.get("underdog_standard")
        assert self.ev_at(p, 4, 0.6) == pytest.approx(0.6 ** 4 * 10.0 - 1.0)
        assert self.ev_at(p, 5, 0.6) == pytest.approx(0.6 ** 5 * 20.0 - 1.0)

    def test_underdog_three_pick_flex_by_hand(self, table):
        # 3/3 at 0.216 pays 2.25x; 2/3 at 3*0.36*0.4 = 0.432 pays 1.25x.
        p = table.get("underdog_flex")
        expected = 0.216 * 2.25 + 0.432 * 1.25 - 1.0
        assert self.ev_at(p, 3, 0.6) == pytest.approx(expected)

    def test_underdog_five_pick_flex_by_hand(self, table):
        p, q = 0.6, 0.4
        five = p ** 5
        four = 5 * p ** 4 * q
        three = 10 * p ** 3 * q ** 2
        expected = five * 10.0 + four * 2.5 + three * 0.4 - 1.0
        assert self.ev_at(table.get("underdog_flex"), 5, 0.6) == pytest.approx(expected)

    def test_prizepicks_power_by_hand(self, table):
        p = table.get("prizepicks_power")
        assert self.ev_at(p, 3, 0.6) == pytest.approx(0.216 * 5.0 - 1.0)
        assert self.ev_at(p, 6, 0.6) == pytest.approx(0.6 ** 6 * 37.5 - 1.0)

    def test_prizepicks_six_pick_flex_by_hand(self, table):
        p, q, n = 0.6, 0.4, 6
        weight = lambda k: math.comb(n, k) * p ** k * q ** (n - k)
        expected = (
            weight(6) * 25.0 + weight(5) * 2.0 + weight(4) * 0.4
        ) - 1.0
        assert self.ev_at(table.get("prizepicks_flex"), 6, 0.6) == pytest.approx(expected)

    def test_the_simulator_reproduces_the_hand_calculation(self, table):
        product = table.get("underdog_flex")
        grid = copula.independent_grid([0.6] * 3)
        ev = float((grid * product.multiple_grid(3)).sum()) - 1.0
        assert ev == pytest.approx(self.ev_at(product, 3, 0.6), abs=1e-12)

    def test_a_coin_flip_leg_loses_money_on_every_shipped_product(self, table):
        # Every one of these is priced to beat a 50% picker comfortably.
        for key, product in table.products.items():
            for legs in product.leg_counts:
                assert product.independent_ev(legs, 0.5) < 0, f"{key} {legs}"


class TestBreakeven:
    def test_it_matches_the_closed_form_for_all_or_nothing(self, table):
        product = table.get("underdog_standard")
        for legs in product.leg_counts:
            closed = product.all_hit_multiple(legs) ** (-1.0 / legs)
            assert product.breakeven_leg_prob(legs) == pytest.approx(closed, abs=1e-6)

    def test_a_flex_bar_is_lower_than_its_all_hit_bar(self, table):
        # Missing one leg still pays, so the entry needs less from each.
        flex = table.get("underdog_flex")
        assert flex.breakeven_leg_prob(5) < flex.all_hit_multiple(5) ** (-1 / 5)

    def test_the_breakeven_is_actually_break_even(self, table):
        for product in table.products.values():
            for legs in product.leg_counts:
                p = product.breakeven_leg_prob(legs)
                assert product.independent_ev(legs, p) == pytest.approx(0.0, abs=1e-6)

    def test_an_unwinnable_structure_needs_certainty(self, tmp_path):
        bad = tmp_path / "p.yaml"
        bad.write_text(
            "products:\n  x:\n    book: b\n    kind: pickem\n"
            "    payouts:\n      2: [0, 0, 0.5]\n"
        )
        product = P.PayoutTable.load(bad).get("x")
        assert product.breakeven_leg_prob(2) == 1.0


class TestVoidHandling:
    def test_a_push_shrinks_a_standard_entry_to_the_smaller_table(self, table):
        # A 3-pick with one voided leg pays the 2-pick table, so two
        # survivors landing returns 3x, not 6x.
        grid = table.get("underdog_standard").multiple_grid(3)
        assert grid[0, 3] == 6.0
        assert grid[1, 2] == 3.0
        assert grid[1, 1] == 0.0

    def test_a_flex_entry_falls_back_to_the_standard_table(self, table):
        grid = table.get("underdog_flex").multiple_grid(3)
        assert grid[0, 3] == 2.25
        assert grid[0, 2] == 1.25
        assert grid[1, 2] == 3.0        # reduces_to underdog_standard

    def test_shrinking_below_the_smallest_table_refunds_the_stake(self, table):
        grid = table.get("underdog_standard").multiple_grid(3)
        assert grid[2, 1] == 1.0
        assert grid[3, 0] == 1.0

    def test_a_refunding_product_returns_the_stake_on_any_push(self, tmp_path):
        p = tmp_path / "p.yaml"
        p.write_text(
            "products:\n  x:\n    book: b\n    kind: pickem\n"
            "    void_behaviour: refund\n    payouts:\n      2: [0, 0, 3.0]\n"
        )
        grid = P.PayoutTable.load(p).get("x").multiple_grid(2)
        assert grid[1, 1] == 1.0
        assert grid[1, 0] == 1.0

    def test_push_probability_lowers_a_ticket_s_expected_value(self, table, pcfg, priors):
        product = table.get("underdog_standard")
        clean = [make_leg(selection=f"P{i}", market=f"m{i}", fair_prob=0.60)
                 for i in range(3)]
        pushy = [make_leg(selection=f"P{i}", market=f"m{i}", fair_prob=0.60,
                          push_prob=0.10, line=6.0) for i in range(3)]
        a = P.evaluate_ticket(clean, product, pcfg, priors)
        b = P.evaluate_ticket(pushy, product, pcfg, priors)
        assert b.ev < a.ev


class TestParlayPricing:
    def test_a_cross_game_price_is_the_product_of_the_legs(self):
        legs = [make_leg(book_price=2.0), make_leg(book_price=1.5,
                                                   selection="B", market="m")]
        product = P.parlay_product(legs)
        assert product.all_hit_multiple(2) == pytest.approx(3.0)
        assert product.kind == P.KIND_PARLAY

    def test_an_offered_price_overrides_the_product(self):
        legs = [make_leg(), make_leg(selection="B", market="m")]
        assert P.parlay_product(legs, offered_decimal=4.2).all_hit_multiple(2) == 4.2

    def test_an_unpriced_leg_cannot_be_parlayed_without_a_price(self):
        legs = [make_leg(book_price=None), make_leg(selection="B", market="m")]
        with pytest.raises(ValueError, match="offered-price"):
            P.parlay_product(legs)


# --------------------------------------------------------------------------
# Leg construction
# --------------------------------------------------------------------------


class TestBuildLegs:
    def test_both_sides_of_every_matched_prop_become_candidates(self, pcfg):
        legs = legs_from(prop_event(), pcfg)
        assert len(legs) == 2 * len(KC_STACK)
        assert {l.side for l in legs} == {"Over", "Under"}

    def test_the_marginal_is_pinnacle_de_vigged_not_the_book_price(self, pcfg):
        legs = legs_from(prop_event(), pcfg)
        over = [l for l in legs if l.selection == "Patrick Mahomes"
                and l.side == "Over"][0]
        # Pinnacle 1.62/2.42 is a heavy over; the book's 1.91 is not the input.
        assert over.fair_prob > 0.55
        assert over.book_price == 1.91
        assert over.sharp_price_taken == 1.62

    def test_under_worst_case_both_sides_are_priced_conservatively(self, pcfg):
        # worst_case takes the LOWEST probability any de-vig method gives,
        # for whichever side is being taken. So the two sides of one prop
        # sum to slightly under 1 rather than to exactly 1 -- deliberately,
        # since a bet should only clear the bar if it clears it on every
        # method. This mirrors what scan.py does for a single bet.
        assert pcfg.model.devig_method == "worst_case"
        legs = legs_from(prop_event(), pcfg)
        by_side = {l.side: l for l in legs if l.selection == "Travis Kelce"}
        total = by_side["Over"].fair_prob + by_side["Under"].fair_prob
        assert total < 1.0
        assert total > 0.99

    def test_a_named_devig_method_gives_complementary_sides(self, pcfg):
        pcfg.model.devig_method = "shin"
        legs = legs_from(prop_event(), pcfg)
        by_side = {l.side: l for l in legs if l.selection == "Travis Kelce"}
        assert by_side["Over"].fair_prob + by_side["Under"].fair_prob == pytest.approx(1.0)

    def test_a_leg_with_no_pinnacle_market_is_refused(self, pcfg):
        payload = event_payload(
            event_id="x", commence=NOW + timedelta(hours=8),
            bookmakers=[book("underdog", {"player_pass_yds": [
                outcome("Over", 1.91, point=249.5, description="Nobody"),
                outcome("Under", 1.91, point=249.5, description="Nobody"),
            ]})],
        )
        rejections = {}
        assert legs_from(payload, pcfg, rejections=rejections) == []
        assert rejections["no_two_sided_sharp_market"] == 1

    def test_a_one_sided_pinnacle_market_is_not_a_market(self, pcfg):
        payload = event_payload(
            event_id="x", commence=NOW + timedelta(hours=8),
            bookmakers=[
                book("pinnacle", {"player_pass_yds": [
                    outcome("Over", 1.90, point=249.5, description="Mahomes")]}),
                book("underdog", {"player_pass_yds": [
                    outcome("Over", 1.91, point=249.5, description="Mahomes")]}),
            ],
        )
        rejections = {}
        assert legs_from(payload, pcfg, rejections=rejections) == []
        assert "no_two_sided_sharp_market" in rejections

    def test_a_different_line_is_never_silently_compared(self, pcfg):
        rejections = {}
        legs = legs_from(prop_event(target_line_shift=1.0), pcfg,
                         rejections=rejections)
        assert legs == []
        assert rejections["pinnacle_on_a_different_line"] == 2 * len(KC_STACK)

    def test_a_stale_pinnacle_quote_is_refused(self, pcfg):
        rejections = {}
        payload = prop_event(sharp_last_update=NOW - timedelta(minutes=90))
        assert legs_from(payload, pcfg, rejections=rejections) == []
        assert rejections["sharp_quote_stale"] == 2 * len(KC_STACK)

    def test_a_stale_book_quote_is_refused(self, pcfg):
        rejections = {}
        payload = prop_event(book_last_update=NOW - timedelta(minutes=90))
        assert legs_from(payload, pcfg, rejections=rejections) == []
        assert rejections["book_quote_stale"] == 2 * len(KC_STACK)

    def test_an_event_about_to_start_is_refused(self, pcfg):
        rejections = {}
        assert legs_from(prop_event(commence_hours=0.01), pcfg,
                         rejections=rejections) == []
        assert "too_close_to_start" in rejections

    def test_an_event_too_far_out_is_refused(self, pcfg):
        rejections = {}
        assert legs_from(prop_event(commence_hours=24 * 30), pcfg,
                         rejections=rejections) == []
        assert "too_far_out" in rejections

    def test_an_absurd_pinnacle_overround_is_refused(self, pcfg):
        rejections = {}
        payload = prop_event(spec=[("player_pass_yds", "X", 249.5, (1.20, 1.20))])
        assert legs_from(payload, pcfg, rejections=rejections) == []
        assert "sharp_overround_out_of_bounds" in rejections

    def test_the_roster_puts_a_team_on_the_leg(self, pcfg):
        legs = legs_from(prop_event(), pcfg, rosters=KC_TEAMS)
        assert all(l.team == "Kansas City Chiefs" for l in legs)

    def test_without_a_roster_the_team_is_left_unknown(self, pcfg):
        assert all(l.team is None for l in legs_from(prop_event(), pcfg))

    def test_a_book_not_asked_for_is_ignored(self, pcfg):
        assert legs_from(prop_event(target_book="fanduel"), pcfg,
                         books=("underdog",)) == []

    def test_hit_probability_folds_in_the_push(self):
        leg = make_leg(fair_prob=0.60, push_prob=0.10)
        assert leg.hit_prob == pytest.approx(0.54)
        assert leg.fair_price == pytest.approx(1 / 0.54)


class TestPushEstimation:
    def test_a_half_line_can_never_push(self, pcfg):
        legs = legs_from(prop_event(), pcfg)
        assert all(l.push_prob == 0.0 for l in legs)
        assert all(l.push_source == "half_line" for l in legs)

    def test_an_integer_line_measures_the_push_from_the_half_lines(self, pcfg):
        legs = legs_from(ladder_event(), pcfg)
        leg = [l for l in legs if l.side == "Over"][0]
        assert leg.line == 5.0
        assert leg.push_source == "measured"
        assert leg.push_prob > 0.0

    def test_the_measured_push_is_the_gap_between_the_half_lines(self, pcfg):
        meta, quotes = parse_event_odds(ladder_event())
        ladder = P.sharp_ladder(quotes, "pinnacle", pcfg.model.devig_method)
        rungs = ladder[("player_receptions", "Travis Kelce")]
        push, source = P.estimate_push_prob(rungs, 5.0, assumed=0.05)
        assert source == "measured"
        assert push == pytest.approx(rungs[4.5].prob_over - rungs[5.5].prob_over)

    def test_without_the_half_lines_the_push_is_assumed_and_flagged(self, pcfg):
        payload = ladder_event(lines=((5.0, (1.72, 2.20)),))
        legs = legs_from(payload, pcfg)
        leg = legs[0]
        assert leg.push_source == "assumed"
        assert leg.push_prob == pytest.approx(pcfg.parlay.assumed_push_prob)
        assert any("push_prob_assumed" in f for f in leg.flags)


class TestInterpolation:
    def test_it_is_off_by_default(self, pcfg):
        rejections = {}
        payload = ladder_event(
            lines=((4.5, (1.55, 2.55)), (5.5, (1.95, 1.95))), target_line=5.0
        )
        assert legs_from(payload, pcfg, rejections=rejections) == []
        assert "pinnacle_on_a_different_line" in rejections

    def test_when_enabled_it_brackets_and_marks_the_leg(self, tmp_path):
        cfg = parlay_config(tmp_path, allow_line_interpolation=True)
        payload = ladder_event(
            lines=((4.5, (1.55, 2.55)), (5.5, (1.95, 1.95))), target_line=5.0
        )
        legs = legs_from(payload, cfg)
        assert legs
        leg = [l for l in legs if l.side == "Over"][0]
        assert leg.line_source == P.LINE_INTERPOLATED
        assert any("line_interpolated" in f for f in leg.flags)

    def test_the_interpolated_probability_sits_between_the_rungs(self, tmp_path):
        cfg = parlay_config(tmp_path, allow_line_interpolation=True)
        payload = ladder_event(
            lines=((4.5, (1.55, 2.55)), (5.5, (1.95, 1.95))), target_line=5.0
        )
        meta, quotes = parse_event_odds(payload)
        rungs = P.sharp_ladder(quotes, "pinnacle", cfg.model.devig_method)[
            ("player_receptions", "Travis Kelce")
        ]
        probability, lo, hi = P.interpolate_over_prob(rungs, 5.0, 1.0)
        assert (lo, hi) == (4.5, 5.5)
        assert rungs[5.5].prob_over < probability < rungs[4.5].prob_over

    def test_it_refuses_to_extrapolate(self, tmp_path):
        cfg = parlay_config(tmp_path, allow_line_interpolation=True)
        rejections = {}
        payload = ladder_event(
            lines=((4.5, (1.55, 2.55)), (5.5, (1.95, 1.95))), target_line=9.5
        )
        assert legs_from(payload, cfg, rejections=rejections) == []
        assert "pinnacle_line_not_bracketed" in rejections

    def test_it_refuses_a_rung_that_is_too_far_away(self, tmp_path):
        cfg = parlay_config(tmp_path, allow_line_interpolation=True,
                            max_interpolation_distance=0.25)
        rejections = {}
        payload = ladder_event(
            lines=((4.5, (1.55, 2.55)), (5.5, (1.95, 1.95))), target_line=5.0
        )
        assert legs_from(payload, cfg, rejections=rejections) == []
        assert "pinnacle_line_not_bracketed" in rejections


# --------------------------------------------------------------------------
# Prefilter and compatibility
# --------------------------------------------------------------------------


class TestPrefilter:
    def test_a_coin_flip_leg_is_too_far_below_the_bar(self, pcfg, table):
        rejections = {}
        legs = legs_from(prop_event(spec=FLAT_STACK), pcfg)
        kept = P.prefilter_legs(legs, table.get("underdog_standard"), pcfg, rejections)
        assert kept == []
        assert rejections["leg_too_far_below_breakeven"] == len(legs)

    def test_a_leg_above_the_bar_survives(self, pcfg, table):
        legs = legs_from(prop_event(), pcfg)
        kept = P.prefilter_legs(legs, table.get("underdog_standard"), pcfg, {})
        assert kept
        assert all(l.side == "Over" for l in kept)

    def test_an_extreme_longshot_is_dropped(self, pcfg, table):
        rejections = {}
        legs = [make_leg(fair_prob=0.97), make_leg(fair_prob=0.02, selection="B")]
        P.prefilter_legs(legs, table.get("underdog_standard"), pcfg, rejections)
        assert rejections["leg_probability_out_of_range"] == 2

    def test_the_pool_is_capped_and_the_best_kept(self, tmp_path, table):
        cfg = parlay_config(tmp_path, max_candidates_per_group=2)
        legs = [make_leg(selection=f"P{i}", market=f"m{i}", fair_prob=0.60 + i / 100)
                for i in range(6)]
        rejections = {}
        kept = P.prefilter_legs(legs, table.get("underdog_standard"), cfg, rejections)
        assert len(kept) == 2
        assert kept[0].fair_prob > kept[1].fair_prob
        assert rejections["trimmed_to_candidate_cap"] == 4

    def test_a_parlay_leg_without_a_price_is_dropped(self, pcfg, table):
        rejections = {}
        product = P.parlay_product([make_leg(), make_leg(selection="B", market="m")],
                                   offered_decimal=4.0)
        P.prefilter_legs([make_leg(book_price=None)], product, pcfg, rejections)
        assert rejections["no_price_for_a_parlay_leg"] == 1

    def test_a_badly_priced_parlay_leg_is_dropped(self, tmp_path, table):
        cfg = parlay_config(tmp_path, min_leg_ev=-0.01)
        rejections = {}
        product = P.parlay_product([make_leg(), make_leg(selection="B", market="m")],
                                   offered_decimal=4.0)
        P.prefilter_legs([make_leg(fair_prob=0.50, book_price=1.5)],
                         product, cfg, rejections)
        assert rejections["leg_single_bet_ev_too_low"] == 1


class TestCompatibility:
    def test_two_legs_on_one_prop_are_never_allowed(self, table):
        product = table.get("underdog_standard")
        legs = [make_leg(line=249.5), make_leg(line=274.5)]
        ok, why = P.legs_are_compatible(legs, product)
        assert not ok
        assert "same market and player" in why

    def test_both_sides_of_one_prop_are_never_allowed(self, table):
        legs = [make_leg(side="Over"), make_leg(side="Under")]
        ok, _why = P.legs_are_compatible(legs, table.get("underdog_standard"))
        assert not ok

    def test_one_player_twice_is_refused_where_the_product_says_so(self, table):
        legs = [make_leg(market="player_pass_yds"),
                make_leg(market="player_pass_tds", line=1.5)]
        ok, why = P.legs_are_compatible(legs, table.get("underdog_standard"))
        assert not ok
        assert "same player twice" in why

    def test_and_allowed_where_the_product_says_so(self, table):
        legs = [make_leg(market="player_pass_yds"),
                make_leg(market="player_pass_tds", line=1.5)]
        ok, _why = P.legs_are_compatible(legs, table.get("draftkings_parlay"))
        assert ok

    def test_distinct_players_are_fine(self, table):
        legs = [make_leg(selection="A", market="m1"),
                make_leg(selection="B", market="m2")]
        ok, _why = P.legs_are_compatible(legs, table.get("underdog_standard"))
        assert ok


class TestScorability:
    def test_a_leg_without_a_pinnacle_probability_is_not_scorable(self):
        ok, why = P.ticket_is_scorable([make_leg(fair_prob=0.0)])
        assert not ok
        assert "Pinnacle" in why

    def test_a_leg_without_a_reference_price_is_not_scorable(self):
        ok, why = P.ticket_is_scorable([make_leg(sharp_price_taken=None)])
        assert not ok
        assert "reference price" in why

    def test_a_normal_leg_is_scorable(self):
        ok, _why = P.ticket_is_scorable([make_leg()])
        assert ok


# --------------------------------------------------------------------------
# Ticket evaluation
# --------------------------------------------------------------------------


def three_legs(**over):
    base = dict(fair_prob=0.60, team="KC")
    base.update(over)
    return [
        make_leg(selection="Mahomes", market="player_pass_yds", **base),
        make_leg(selection="Kelce", market="player_reception_yds", **base),
        make_leg(selection="Rice", market="player_receptions", line=5.5, **base),
    ]


class TestEvaluateTicket:
    def test_it_produces_a_joint_probability_with_an_error_bar(self, pcfg, priors, table):
        t = P.evaluate_ticket(three_legs(), table.get("underdog_standard"),
                              pcfg, priors)
        assert 0 < t.joint_prob < 1
        assert t.joint_prob_se > 0
        assert t.n_legs == 3

    def test_the_independence_baseline_is_exact_not_simulated(self, pcfg, priors, table):
        t = P.evaluate_ticket(three_legs(), table.get("underdog_standard"),
                              pcfg, priors)
        assert t.joint_prob_independent == pytest.approx(0.6 ** 3, abs=1e-12)
        assert t.ev_independent == pytest.approx(0.6 ** 3 * 6 - 1, abs=1e-12)

    def test_positive_correlation_beats_the_independence_baseline(self, pcfg, priors, table):
        t = P.evaluate_ticket(three_legs(), table.get("underdog_standard"),
                              pcfg, priors)
        assert t.joint_prob > t.joint_prob_independent
        assert t.ev > t.ev_independent
        assert t.correlation_lift > 0

    def test_opposing_sides_of_a_correlated_pair_lower_the_joint(self, pcfg, priors, table):
        aligned = [
            make_leg(selection="Mahomes", market="player_pass_yds", team="KC"),
            make_leg(selection="Kelce", market="player_reception_yds", team="KC"),
        ]
        opposed = [
            aligned[0],
            make_leg(selection="Kelce", market="player_reception_yds",
                     team="KC", side="Under", fair_prob=0.60),
        ]
        product = table.get("underdog_standard")
        a = P.evaluate_ticket(aligned, product, pcfg, priors)
        b = P.evaluate_ticket(opposed, product, pcfg, priors)
        assert b.joint_prob < a.joint_prob

    def test_the_hit_distribution_sums_to_one(self, pcfg, priors, table):
        t = P.evaluate_ticket(three_legs(), table.get("underdog_standard"),
                              pcfg, priors)
        assert sum(t.hit_distribution) == pytest.approx(1.0)
        assert len(t.hit_distribution) == 4

    def test_a_ticket_is_identified_by_its_leg_set_not_their_order(self, pcfg, priors, table):
        legs = three_legs()
        product = table.get("underdog_standard")
        a = P.evaluate_ticket(legs, product, pcfg, priors)
        b = P.evaluate_ticket(list(reversed(legs)), product, pcfg, priors)
        assert a.key == b.key

    def test_the_same_seed_gives_the_same_answer(self, pcfg, priors, table):
        product = table.get("underdog_standard")
        a = P.evaluate_ticket(three_legs(), product, pcfg, priors)
        b = P.evaluate_ticket(three_legs(), product, pcfg, priors)
        assert a.ev == b.ev

    def test_the_effective_price_reproduces_the_expected_value(self):
        d = P.effective_decimal(0.25, 0.08)
        assert 0.25 * d - 1 == pytest.approx(0.08)

    def test_ev_per_variance_ranks_differently_from_ev(self, pcfg, priors, table):
        two = P.evaluate_ticket(three_legs()[:2], table.get("underdog_standard"),
                                pcfg, priors)
        five = P.evaluate_ticket(
            three_legs() + [
                make_leg(selection="Pacheco", market="player_rush_yds", team="KC"),
                make_leg(selection="Brown", market="player_receptions",
                         line=3.5, team="KC"),
            ],
            table.get("underdog_standard"), pcfg, priors,
        )
        # The five-leg ticket carries far more variance per unit of edge.
        assert five.variance > two.variance
        assert five.ev_per_variance < two.ev_per_variance

    def test_a_ticket_spanning_two_games_is_not_same_game(self, pcfg, priors, table):
        legs = [make_leg(selection="A", market="m1", event_id="g1"),
                make_leg(selection="B", market="m2", event_id="g2")]
        t = P.evaluate_ticket(legs, table.get("underdog_standard"), pcfg, priors)
        assert not t.same_game
        assert t.event_ids == ["g1", "g2"]


# --------------------------------------------------------------------------
# Guards -- one test per guard, each tripping it
# --------------------------------------------------------------------------


class TestGuards:
    def evaluate(self, legs, pcfg, priors, table, key="underdog_standard"):
        return P.evaluate_ticket(legs, table.get(key), pcfg, priors)

    def test_an_implausible_edge_is_suspect_and_unstaked(self, tmp_path, priors, table):
        cfg = parlay_config(tmp_path, max_plausible_ev=0.05)
        t = self.evaluate(three_legs(fair_prob=0.75), cfg, priors, table)
        assert t.suspect
        assert t.recommended_stake == 0.0
        assert any("implausible" in f for f in t.flags)

    def test_a_ticket_positive_only_through_correlation_is_called_out(
        self, tmp_path, priors, table
    ):
        # Legs below the independent break-even: the entry is negative
        # unless the assumed correlation rescues it.
        cfg = parlay_config(tmp_path, max_plausible_ev=1.0)
        legs = [
            make_leg(selection="Mahomes", market="player_pass_yds",
                     team="KC", fair_prob=0.545),
            make_leg(selection="Kelce", market="player_reception_yds",
                     team="KC", fair_prob=0.545),
        ]
        t = self.evaluate(legs, cfg, priors, table)
        assert t.ev_independent <= 0 < t.ev
        assert t.correlation_is_load_bearing
        assert "only_+ev_because_of_assumed_correlation" in t.flags
        assert t.suspect and t.recommended_stake == 0.0

    def test_a_prior_only_ticket_says_so(self, pcfg, priors, table):
        t = self.evaluate(three_legs(), pcfg, priors, table)
        assert "correlation_entirely_from_priors" in t.flags

    def test_a_measured_ticket_does_not_say_so(self, pcfg, priors, table):
        store = C.EstimateStore([
            {"sport": "americanfootball_nfl", "market_a": a, "market_b": b,
             "relation": C.SAME_TEAM, "rho": 0.3, "spearman": 0.3,
             "n_observations": 900, "n_games": 900,
             "fitted_at": "2026-09-01T00:00:00+00:00", "note": ""}
            for a, b in [
                ("player_pass_yds", "player_reception_yds"),
                ("player_pass_yds", "player_receptions"),
                ("player_reception_yds", "player_receptions"),
            ]
        ])
        t = P.evaluate_ticket(three_legs(), table.get("underdog_standard"),
                              pcfg, priors, store)
        assert "correlation_entirely_from_priors" not in t.flags
        assert t.correlation.any_measured

    def test_a_projected_matrix_is_reported(self, tmp_path, priors, table):
        impossible = tmp_path / "p.yaml"
        impossible.write_text(
            "priors:\n  americanfootball_nfl:\n"
            "    - {markets: [a, b], relation: same_team, rho: 0.95, why: t}\n"
            "    - {markets: [b, c], relation: same_team, rho: 0.95, why: t}\n"
            "    - {markets: [a, c], relation: same_team, rho: -0.95, why: t}\n"
        )
        cfg = parlay_config(tmp_path)
        legs = [make_leg(selection=n, market=m, team="KC")
                for n, m in (("A", "a"), ("B", "b"), ("C", "c"))]
        t = P.evaluate_ticket(legs, table.get("underdog_standard"), cfg,
                              C.PriorSet.load(impossible))
        assert "correlation_matrix_projected_to_psd" in t.flags

    def test_a_noisy_monte_carlo_estimate_is_flagged(self, tmp_path, priors, table):
        cfg = parlay_config(tmp_path, draws=300, max_se_ratio=0.01,
                            max_plausible_ev=1.0)
        t = self.evaluate(three_legs(), cfg, priors, table)
        assert any("monte_carlo_se" in f for f in t.flags)
        assert t.suspect

    def test_a_negatively_correlated_pair_is_flagged(self, tmp_path, priors, table):
        cfg = parlay_config(tmp_path, max_plausible_ev=1.0)
        legs = [
            make_leg(selection="Pacheco", market="player_rush_yds", team="KC"),
            make_leg(selection="Edwards", market="player_rush_yds", team="KC"),
        ]
        t = self.evaluate(legs, cfg, priors, table)
        assert any("negatively_correlated_pair" in f for f in t.flags)

    def test_an_over_against_the_total_under_is_flagged(self, tmp_path, priors, table):
        cfg = parlay_config(tmp_path, max_plausible_ev=1.0)
        legs = [
            make_leg(selection="Mahomes", market="player_pass_yds", side="Over"),
            make_leg(selection="Total", market="totals", side="Under", line=47.5),
        ]
        t = self.evaluate(legs, cfg, priors, table)
        assert any("negatively_correlated_pair" in f for f in t.flags)

    def test_an_unverified_payout_table_is_flagged(self, pcfg, priors, table):
        t = self.evaluate(three_legs(), pcfg, priors, table)
        assert "payout_table_unverified" in t.flags

    def test_an_interpolated_leg_makes_the_ticket_suspect(self, pcfg, priors, table):
        legs = three_legs()
        legs[0] = make_leg(selection="Mahomes", market="player_pass_yds",
                           team="KC", fair_prob=0.60,
                           line_source=P.LINE_INTERPOLATED)
        t = self.evaluate(legs, pcfg, priors, table)
        assert "leg_line_interpolated_not_matched" in t.flags
        assert t.suspect

    def test_an_assumed_push_probability_is_flagged(self, pcfg, priors, table):
        legs = three_legs()
        legs[0] = make_leg(selection="Mahomes", market="player_pass_yds",
                           team="KC", fair_prob=0.60, line=249.0,
                           push_prob=0.05, push_source="assumed")
        t = self.evaluate(legs, pcfg, priors, table)
        assert "push_probability_assumed" in t.flags

    def test_a_cross_game_parlay_is_suspect_and_states_its_hold(
        self, tmp_path, priors
    ):
        cfg = parlay_config(tmp_path, max_plausible_ev=1.0)
        legs = [make_leg(selection="A", market="m1", event_id="g1",
                         sharp_overround=0.045, book_price=2.0),
                make_leg(selection="B", market="m2", event_id="g2",
                         sharp_overround=0.045, book_price=2.0)]
        t = P.evaluate_ticket(legs, P.parlay_product(legs), cfg, priors)
        assert t.suspect
        assert any("cross_game_parlay_compounded_hold" in f for f in t.flags)

    def test_a_same_game_parlay_warns_that_the_book_discounts_it(
        self, tmp_path, priors
    ):
        cfg = parlay_config(tmp_path, max_plausible_ev=1.0)
        legs = [make_leg(selection="A", market="m1", book_price=2.0),
                make_leg(selection="B", market="m2", book_price=2.0)]
        t = P.evaluate_ticket(legs, P.parlay_product(legs), cfg, priors)
        assert "sgp_price_may_be_discounted_by_the_book" in t.flags

    def test_unknown_teams_are_disclosed(self, pcfg, priors, table):
        legs = [make_leg(selection="A", market="player_pass_yds", team=None),
                make_leg(selection="B", market="player_reception_yds", team=None)]
        t = self.evaluate(legs, pcfg, priors, table)
        assert "leg_teams_unknown_same_game_priors_are_blends" in t.flags

    def test_nothing_structural_matching_is_disclosed(self, pcfg, priors, table):
        legs = [make_leg(selection="A", market="player_steals", team=None),
                make_leg(selection="B", market="player_blocks", team=None)]
        t = self.evaluate(legs, pcfg, priors, table)
        assert any("no_structural_correlation_matched" in f for f in t.flags)

    def test_leg_level_flags_reach_the_ticket(self, pcfg, priors, table):
        legs = three_legs()
        legs[0] = make_leg(selection="Mahomes", market="player_pass_yds",
                           team="KC", fair_prob=0.60, flags=("something_odd",))
        t = self.evaluate(legs, pcfg, priors, table)
        assert "leg:something_odd" in t.flags


# --------------------------------------------------------------------------
# Staking
# --------------------------------------------------------------------------


class TestStaking:
    def test_a_suspect_ticket_gets_nothing(self, tmp_path, priors, table):
        cfg = parlay_config(tmp_path, max_plausible_ev=0.01)
        t = P.evaluate_ticket(three_legs(fair_prob=0.70),
                              table.get("underdog_standard"), cfg, priors)
        assert t.suspect
        assert t.recommended_stake == 0.0

    def test_a_negative_ticket_gets_nothing(self, pcfg, priors, table):
        t = P.evaluate_ticket(three_legs(fair_prob=0.40),
                              table.get("underdog_standard"), pcfg, priors)
        assert t.ev < 0
        assert P.ticket_stake(t, pcfg) == 0.0

    def test_parlay_staking_is_stricter_than_the_single_bet_path(self, pcfg):
        assert pcfg.parlay.kelly_multiplier < pcfg.bankroll.kelly_multiplier
        assert pcfg.parlay.max_ticket_fraction < pcfg.bankroll.max_fraction

    def test_the_ticket_cap_is_respected(self, tmp_path, priors, table):
        cfg = parlay_config(tmp_path, max_plausible_ev=5.0,
                            max_ticket_fraction=0.005)
        cfg.bankroll.amount = 10_000
        cfg.bankroll.round_to = 1
        t = P.evaluate_ticket(three_legs(fair_prob=0.85),
                              table.get("underdog_standard"), cfg, priors)
        assert t.recommended_stake <= 10_000 * 0.005

    def test_the_log_optimal_fraction_is_reported_as_a_cross_check(
        self, tmp_path, priors, table
    ):
        cfg = parlay_config(tmp_path, max_plausible_ev=5.0)
        t = P.evaluate_ticket(three_legs(fair_prob=0.70),
                              table.get("underdog_standard"), cfg, priors)
        assert t.log_optimal_fraction > 0
        assert t.kelly_fraction > 0


class TestExposureCaps:
    def build(self, cfg, priors, table, n_tickets=4, event_ids=None):
        event_ids = event_ids or ["g1"] * n_tickets
        tickets = []
        for i, event_id in enumerate(event_ids):
            legs = [
                make_leg(selection=f"A{i}", market="player_pass_yds",
                         event_id=event_id, team="KC", fair_prob=0.68),
                make_leg(selection=f"B{i}", market="player_reception_yds",
                         event_id=event_id, team="KC", fair_prob=0.68),
            ]
            tickets.append(
                P.evaluate_ticket(legs, table.get("underdog_standard"), cfg, priors)
            )
        return tickets

    def test_one_game_cannot_absorb_the_whole_bankroll(self, tmp_path, priors, table):
        cfg = parlay_config(tmp_path, max_plausible_ev=5.0,
                            max_game_exposure_fraction=0.02)
        cfg.bankroll.amount = 1000
        tickets = self.build(cfg, priors, table, n_tickets=5)
        assert sum(t.recommended_stake for t in tickets) > 20
        P.cap_exposure(tickets, cfg)
        assert P.exposure_by_game(tickets)["g1"] <= 20

    def test_the_best_ticket_is_funded_first(self, tmp_path, priors, table):
        cfg = parlay_config(tmp_path, max_plausible_ev=5.0,
                            max_game_exposure_fraction=0.01)
        cfg.bankroll.amount = 1000
        tickets = self.build(cfg, priors, table, n_tickets=4)
        P.cap_exposure(tickets, cfg)
        best = max(tickets, key=lambda t: t.ev)
        assert best.recommended_stake > 0

    def test_tickets_on_different_games_each_get_their_own_room(
        self, tmp_path, priors, table
    ):
        cfg = parlay_config(tmp_path, max_plausible_ev=5.0,
                            max_game_exposure_fraction=0.02)
        cfg.bankroll.amount = 1000
        tickets = self.build(cfg, priors, table, event_ids=["g1", "g2", "g3"])
        P.cap_exposure(tickets, cfg)
        assert all(v <= 20 for v in P.exposure_by_game(tickets).values())
        assert sum(t.recommended_stake for t in tickets) > 20

    def test_the_total_board_cap_still_applies(self, tmp_path, priors, table):
        cfg = parlay_config(tmp_path, max_plausible_ev=5.0,
                            max_game_exposure_fraction=1.0)
        cfg.bankroll.amount = 1000
        cfg.bankroll.max_total_exposure_fraction = 0.02
        tickets = self.build(cfg, priors, table,
                             event_ids=[f"g{i}" for i in range(6)])
        P.cap_exposure(tickets, cfg)
        assert sum(t.recommended_stake for t in tickets) <= 20

    def test_a_trimmed_ticket_says_so(self, tmp_path, priors, table):
        cfg = parlay_config(tmp_path, max_plausible_ev=5.0,
                            max_game_exposure_fraction=0.012)
        cfg.bankroll.amount = 1000
        tickets = self.build(cfg, priors, table, n_tickets=4)
        P.cap_exposure(tickets, cfg)
        assert any(
            "exposure" in f for t in tickets for f in t.flags
        )


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


class TestGrouping:
    def test_same_game_keeps_events_apart(self, pcfg):
        legs = [make_leg(selection="A", market="m1", event_id="g1"),
                make_leg(selection="B", market="m2", event_id="g1"),
                make_leg(selection="C", market="m3", event_id="g2"),
                make_leg(selection="D", market="m4", event_id="g2")]
        pools = P.group_legs(legs, pcfg)
        assert len(pools) == 2

    def test_a_group_below_the_minimum_leg_count_is_dropped(self, pcfg):
        legs = [make_leg(selection="A", market="m1", event_id="g1"),
                make_leg(selection="B", market="m2", event_id="g2")]
        assert P.group_legs(legs, pcfg) == []

    def test_any_puts_everything_in_one_pool(self, tmp_path):
        cfg = parlay_config(tmp_path, grouping="any")
        legs = [make_leg(selection="A", market="m1", event_id="g1"),
                make_leg(selection="B", market="m2", event_id="g2")]
        assert len(P.group_legs(legs, cfg)) == 1

    def test_same_slate_groups_by_sport_and_start_window(self, tmp_path):
        cfg = parlay_config(tmp_path, grouping="same_slate", slate_hours=12)
        legs = [make_leg(selection="A", market="m1", event_id="g1"),
                make_leg(selection="B", market="m2", event_id="g2"),
                make_leg(selection="C", market="m3", event_id="g3",
                         sport="baseball_mlb"),
                make_leg(selection="D", market="m4", event_id="g4",
                         sport="baseball_mlb")]
        pools = P.group_legs(legs, cfg)
        assert len(pools) == 2
        assert {l.sport for l in pools[0]} == {"americanfootball_nfl"}


class TestBeamSearch:
    def test_it_finds_tickets_and_ranks_them_by_expected_value(
        self, pcfg, priors, table
    ):
        legs = P.prefilter_legs(legs_from(prop_event(), pcfg),
                                table.get("underdog_standard"), pcfg, {})
        tickets = P.beam_search(legs, table.get("underdog_standard"), pcfg, priors)
        assert tickets
        assert tickets == sorted(tickets, key=lambda t: t.ev, reverse=True)

    def test_no_two_returned_tickets_are_permutations_of_each_other(
        self, pcfg, priors, table
    ):
        legs = P.prefilter_legs(legs_from(prop_event(), pcfg),
                                table.get("underdog_standard"), pcfg, {})
        tickets = P.beam_search(legs, table.get("underdog_standard"), pcfg, priors)
        keys = [t.key for t in tickets]
        assert len(keys) == len(set(keys))

    def test_every_ticket_respects_the_product_s_leg_counts(
        self, pcfg, priors, table
    ):
        product = table.get("underdog_standard")
        legs = P.prefilter_legs(legs_from(prop_event(), pcfg), product, pcfg, {})
        for t in P.beam_search(legs, product, pcfg, priors):
            assert product.min_legs <= t.n_legs <= min(product.max_legs,
                                                       pcfg.parlay.max_legs)

    def test_no_ticket_holds_the_same_player_twice(self, pcfg, priors, table):
        product = table.get("underdog_standard")
        legs = P.prefilter_legs(legs_from(prop_event(), pcfg), product, pcfg, {})
        for t in P.beam_search(legs, product, pcfg, priors):
            names = [l.selection for l in t.legs]
            assert len(names) == len(set(names))

    def test_too_few_candidates_yields_nothing(self, pcfg, priors, table):
        assert P.beam_search([make_leg()], table.get("underdog_standard"),
                             pcfg, priors) == []

    def test_the_max_leg_setting_is_honoured(self, tmp_path, priors, table):
        cfg = parlay_config(tmp_path, max_legs=2)
        product = table.get("underdog_standard")
        legs = P.prefilter_legs(legs_from(prop_event(), cfg), product, cfg, {})
        assert all(t.n_legs == 2 for t in P.beam_search(legs, product, cfg, priors))

    def test_a_product_with_no_table_for_a_size_skips_that_size(
        self, pcfg, priors, table
    ):
        # underdog_flex starts at three legs, so no two-leg flex entry
        # should ever be produced.
        product = table.get("underdog_flex")
        legs = P.prefilter_legs(legs_from(prop_event(), pcfg), product, pcfg, {})
        assert all(t.n_legs >= 3 for t in P.beam_search(legs, product, pcfg, priors))


class TestRanking:
    def make(self, pcfg, priors, table):
        product = table.get("underdog_standard")
        legs = P.prefilter_legs(legs_from(prop_event(), pcfg), product, pcfg, {})
        return P.beam_search(legs, product, pcfg, priors)

    def test_the_two_rankings_are_both_available(self, pcfg, priors, table):
        tickets = self.make(pcfg, priors, table)
        by_ev = P.rank_by_ev(tickets, 5)
        by_risk = P.rank_by_ev_per_variance(tickets, 5)
        assert by_ev[0].ev >= by_ev[-1].ev
        assert by_risk[0].ev_per_variance >= by_risk[-1].ev_per_variance

    def test_a_big_multiple_with_a_bad_edge_never_outranks_a_small_good_one(
        self, pcfg, priors, table
    ):
        good = P.evaluate_ticket(
            [make_leg(selection="A", market="m1", fair_prob=0.62),
             make_leg(selection="B", market="m2", fair_prob=0.62)],
            table.get("underdog_standard"), pcfg, priors,
        )
        greedy = P.evaluate_ticket(
            [make_leg(selection=f"P{i}", market=f"m{i}", fair_prob=0.45)
             for i in range(5)],
            table.get("underdog_standard"), pcfg, priors,
        )
        assert greedy.payout_all_hit > good.payout_all_hit
        assert greedy.ev < good.ev
        assert P.rank_by_ev([greedy, good], 2)[0] is good


class TestBuildTickets:
    def test_an_end_to_end_build_from_a_payload(self, pcfg, priors, table):
        legs = legs_from(prop_event(), pcfg)
        state = P.ParlayScanResult(
            started_at=NOW, finished_at=NOW, sports=["americanfootball_nfl"],
            products=["underdog_standard"], tickets=[],
        )
        tickets = P.build_tickets(
            legs, [table.get("underdog_standard")], pcfg, priors,
            now=NOW, state=state,
        )
        assert tickets
        assert state.groups_searched == 1
        assert state.candidates_evaluated > 0
        assert all(t.product.key == "underdog_standard" for t in tickets)

    def test_legs_from_the_wrong_book_are_not_used(self, pcfg, priors, table):
        legs = legs_from(prop_event(target_book="draftkings"), pcfg,
                         books=("draftkings",))
        assert legs
        assert P.build_tickets(legs, [table.get("underdog_standard")],
                               pcfg, priors, now=NOW) == []

    def test_below_the_ev_bar_nothing_clean_comes_back(self, tmp_path, priors, table):
        cfg = parlay_config(tmp_path, min_ev=0.90)
        legs = legs_from(prop_event(), cfg)
        tickets = P.build_tickets(legs, [table.get("underdog_standard")],
                                  cfg, priors, now=NOW)
        assert all(t.suspect for t in tickets)


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------


class TestCoverage:
    def test_the_excluded_sports_each_carry_a_reason(self):
        assert P.PROP_OPTIMIZER_EXCLUDED
        for pattern, why in P.PROP_OPTIMIZER_EXCLUDED.items():
            assert len(why) > 40, pattern

    def test_tennis_mma_and_soccer_are_the_excluded_ones(self):
        keys = set(P.PROP_OPTIMIZER_EXCLUDED)
        assert {"tennis_*", "mma_mixed_martial_arts", "soccer_*"} == keys

    def test_the_default_sports_are_the_four_deep_prop_leagues(self):
        assert set(P.DEFAULT_PROP_SPORTS) == {
            "basketball_nba", "americanfootball_nfl",
            "baseball_mlb", "icehockey_nhl",
        }

    def test_a_probe_counts_matched_lines(self, pcfg):
        from conftest import FakeClient

        client = FakeClient(
            events_by_sport={"americanfootball_nfl": [
                {"id": "kc1", "commence_time": (NOW + timedelta(hours=8)).isoformat()}
            ]},
            event_odds={("americanfootball_nfl", "kc1"): prop_event()},
        )
        report = P.probe_coverage(pcfg, client, sports=["americanfootball_nfl"],
                                  now=NOW)
        row = report.rows[0]
        assert row.two_sided_sharp_markets == len(KC_STACK)
        assert row.matched_on_same_line == 2 * len(KC_STACK)
        assert row.match_rate > 0

    def test_a_book_on_different_lines_counts_as_unmatched(self, pcfg):
        from conftest import FakeClient

        client = FakeClient(
            events_by_sport={"americanfootball_nfl": [
                {"id": "kc1", "commence_time": (NOW + timedelta(hours=8)).isoformat()}
            ]},
            event_odds={
                ("americanfootball_nfl", "kc1"): prop_event(target_line_shift=1.0)
            },
        )
        row = P.probe_coverage(pcfg, client, sports=["americanfootball_nfl"],
                               now=NOW).rows[0]
        assert row.with_book_quote > 0
        assert row.matched_on_same_line == 0
        assert not row.usable

    def test_an_excluded_sport_is_never_probed(self, pcfg):
        from conftest import FakeClient

        client = FakeClient()
        report = P.probe_coverage(pcfg, client, sports=["basketball_ncaab"],
                                  now=NOW)
        assert report.rows == []
        assert client.calls["events"] == 0


# --------------------------------------------------------------------------
# DraftKings parlays -- the secondary, harder target
# --------------------------------------------------------------------------


class TestParlayProducts:
    def test_the_shipped_parlay_ships_with_no_table(self, table):
        product = table.get("draftkings_parlay")
        assert product.kind == P.KIND_PARLAY
        assert product.payouts == {}

    def test_a_template_is_priced_from_the_legs_it_holds(self, table):
        legs = [make_leg(book_price=2.0, selection="A", market="m1"),
                make_leg(book_price=1.8, selection="B", market="m2")]
        live = P.resolve_product(table.get("draftkings_parlay"), legs)
        assert live.all_hit_multiple(2) == pytest.approx(3.6)

    def test_an_offered_price_wins_over_the_product_of_the_legs(self, table):
        legs = [make_leg(book_price=2.0, selection="A", market="m1"),
                make_leg(book_price=1.8, selection="B", market="m2")]
        live = P.resolve_product(table.get("draftkings_parlay"), legs, 3.1)
        assert live.all_hit_multiple(2) == pytest.approx(3.1)

    def test_a_pickem_product_is_never_repriced(self, table):
        product = table.get("underdog_standard")
        assert P.resolve_product(product, [make_leg()], 99.0) is product

    def test_a_parlay_search_works_without_an_offered_price(
        self, tmp_path, priors, table
    ):
        # The whole point of the fix: a DK parlay with no price supplied
        # must still produce tickets, priced from the legs and flagged as
        # an upper bound, rather than silently returning nothing.
        cfg = parlay_config(tmp_path, max_plausible_ev=5.0, min_ev=-1.0)
        legs = legs_from(prop_event(target_book="draftkings"), cfg,
                         books=("draftkings",))
        tickets = P.build_tickets(legs, [table.get("draftkings_parlay")],
                                  cfg, priors, now=NOW)
        assert tickets
        assert all(t.product.kind == P.KIND_PARLAY for t in tickets)
        assert all(t.payout_all_hit > 1 for t in tickets)

    def test_an_offered_price_is_applied_at_the_requested_leg_count(
        self, tmp_path, priors, table
    ):
        cfg = parlay_config(tmp_path, max_legs=2, min_legs=2,
                            max_plausible_ev=5.0, min_ev=-1.0)
        legs = legs_from(prop_event(target_book="draftkings"), cfg,
                         books=("draftkings",))
        tickets = P.build_tickets(legs, [table.get("draftkings_parlay")],
                                  cfg, priors, now=NOW, offered_price=3.1)
        assert tickets
        assert all(t.payout_all_hit == pytest.approx(3.1) for t in tickets)

    def test_a_same_game_parlay_warns_the_price_is_an_upper_bound(
        self, tmp_path, priors, table
    ):
        cfg = parlay_config(tmp_path, max_plausible_ev=5.0, min_ev=-1.0)
        legs = legs_from(prop_event(target_book="draftkings"), cfg,
                         books=("draftkings",))
        tickets = P.build_tickets(legs, [table.get("draftkings_parlay")],
                                  cfg, priors, now=NOW)
        assert all(
            "sgp_price_may_be_discounted_by_the_book" in t.flags for t in tickets
        )

    def test_a_parlay_allows_two_props_on_one_player(self, table):
        legs = [make_leg(market="player_pass_yds"),
                make_leg(market="player_pass_tds", line=1.5)]
        ok, _why = P.legs_are_compatible(
            legs, P.resolve_product(table.get("draftkings_parlay"), legs)
        )
        assert ok
