"""Liquidity scoring: the ordering it produces is the point, so most of
these assert relative rankings rather than exact numbers."""

import pytest

from betedge import liquidity as L


class TestTiers:
    @pytest.mark.parametrize("market,tier", [
        ("h2h", L.TIER_MAINLINE),
        ("spreads", L.TIER_MAINLINE),
        ("totals", L.TIER_MAINLINE),
        ("h2h_3_way", L.TIER_MAINLINE),
        ("totals_h1", L.TIER_DERIVATIVE),
        ("alternate_spreads", L.TIER_DERIVATIVE),
        ("player_pass_yds", L.TIER_PRIMARY_PROP),
        ("pitcher_strikeouts", L.TIER_PRIMARY_PROP),
        ("player_pass_yds_alternate", L.TIER_ALTERNATE),
        ("player_tackles_assists", L.TIER_SECONDARY_PROP),
    ])
    def test_classification(self, market, tier):
        assert L.tier_for(market) == tier

    def test_unknown_market_is_treated_as_a_secondary_prop(self):
        assert L.tier_for("player_something_new") == L.TIER_SECONDARY_PROP
        assert L.tier_for(None) == L.TIER_SECONDARY_PROP

    def test_mainlines_outrank_props_outrank_alternates(self):
        assert (L.TIER_CONFIDENCE[L.TIER_MAINLINE]
                > L.TIER_CONFIDENCE[L.TIER_PRIMARY_PROP]
                > L.TIER_CONFIDENCE[L.TIER_SECONDARY_PROP]
                > L.TIER_CONFIDENCE[L.TIER_ALTERNATE])


class TestOverroundFactor:
    def test_a_tight_market_scores_full_confidence(self):
        assert L.overround_factor(0.020, 2) == pytest.approx(1.0)

    def test_a_wide_market_bottoms_out(self):
        assert L.overround_factor(0.12, 2) == pytest.approx(L.OVERROUND_MIN_FACTOR)

    def test_it_decreases_monotonically(self):
        scores = [L.overround_factor(o, 2) for o in (0.01, 0.03, 0.05, 0.07, 0.09)]
        assert scores == sorted(scores, reverse=True)

    def test_normalised_per_outcome(self):
        """A 4.5% three-way book is 1.5% per outcome and tight; the same
        4.5% on a two-way market is 2.25% a side and merely average."""
        assert L.overround_factor(0.045, 3) > L.overround_factor(0.045, 2)

    def test_zero_outcomes_does_not_divide_by_zero(self):
        assert 0.0 < L.overround_factor(0.05, 0) <= 1.0


class TestTimingAndLongshot:
    def test_closer_events_score_higher(self):
        assert L.timing_factor(60) > L.timing_factor(60 * 30) > L.timing_factor(60 * 100)

    def test_missing_start_time_is_discounted_but_not_zero(self):
        assert 0.5 < L.timing_factor(None) < 1.0

    def test_even_markets_are_not_penalised(self):
        assert L.longshot_factor(0.5) == pytest.approx(1.0)
        assert L.longshot_factor(0.25) == pytest.approx(1.0)

    def test_longshots_are_penalised_symmetrically(self):
        assert L.longshot_factor(0.03) == pytest.approx(L.longshot_factor(0.97))
        assert L.longshot_factor(0.03) < L.longshot_factor(0.30)


class TestAssess:
    def test_a_deep_mainline_scores_near_one(self):
        a = L.assess("h2h", overround=0.022, n_outcomes=2, fair_prob=0.5,
                     minutes_to_start=600)
        assert a.score > 0.90
        assert a.label == "deep"

    def test_an_alternate_longshot_days_out_scores_low(self):
        a = L.assess("player_pass_yds_alternate", overround=0.10, n_outcomes=2,
                     fair_prob=0.08, minutes_to_start=60 * 96)
        assert a.score < 0.30

    def test_score_is_clamped(self):
        a = L.assess("player_x_alternate", overround=0.99, n_outcomes=2,
                     fair_prob=0.001, minutes_to_start=60 * 200)
        assert L.MIN_LIQUIDITY <= a.score <= L.MAX_LIQUIDITY

    def test_mainline_outranks_prop_at_each_tier_normal_margin(self):
        """The comparison that matters: both priced as their kind usually
        is. Measured off a live board -- mainlines ~2.5%, props ~7.0%."""
        mainline = L.assess("h2h", 0.025, 2, 0.5, 300)
        prop = L.assess("pitcher_strikeouts", 0.070, 2, 0.5, 300)
        assert mainline.score > prop.score
        assert mainline.score == pytest.approx(1.0, abs=0.02)
        assert prop.score == pytest.approx(0.75, abs=0.03)

    def test_an_anomalously_wide_mainline_scores_below_a_normal_prop(self):
        """
        Deliberate, and the point of judging margin per tier. 4.5% is
        nearly double what Pinnacle charges on a game line -- it does not
        want that action -- while the same 4.5% on a prop is tighter than
        usual, meaning it is confident. The prop is the better reference
        price, and an absolute overround scale could not see that.
        """
        args = dict(overround=0.045, n_outcomes=2, fair_prob=0.5,
                    minutes_to_start=300)
        assert L.assess("pitcher_strikeouts", **args).score > \
            L.assess("h2h", **args).score


class TestSlidingBar:
    def test_deep_markets_use_the_base_bar(self):
        assert L.required_ev(0.02, 1.0) == pytest.approx(0.02)

    def test_thin_markets_must_show_more(self):
        assert L.required_ev(0.02, 0.4) > L.required_ev(0.02, 0.8) > L.required_ev(0.02, 1.0)

    def test_penalty_of_zero_restores_a_flat_bar(self):
        for liq in (0.2, 0.5, 1.0):
            assert L.required_ev(0.02, liq, penalty=0.0) == pytest.approx(0.02)

    def test_the_documented_defaults_hold(self):
        """The README quotes these, so they are part of the contract."""
        assert L.required_ev(0.02, 1.00) == pytest.approx(0.020, abs=5e-4)
        assert L.required_ev(0.02, 0.65) == pytest.approx(0.0305, abs=1e-3)


class TestEdgeScore:
    def test_liquidity_discounts_the_edge(self):
        assert L.edge_score(0.05, 0.5) == pytest.approx(0.025)

    def test_a_smaller_edge_on_a_deep_market_can_outrank_a_larger_thin_one(self):
        """The ordering the whole module exists to produce."""
        mainline = L.edge_score(0.03, 1.00)
        alt_prop = L.edge_score(0.05, 0.35)
        assert mainline > alt_prop


class TestDiagnosticBar:
    def test_a_zero_bar_stays_zero_at_every_liquidity(self):
        for liq in (0.15, 0.5, 1.0):
            assert L.required_ev(0.0, liq) == 0.0

    def test_a_negative_bar_is_not_scaled(self):
        """`--min-ev -0.03` means "show me everything down to -3%". Scaling
        would make thin markets easier to surface than deep ones, which is
        backwards."""
        for liq in (0.15, 0.5, 1.0):
            assert L.required_ev(-0.03, liq) == pytest.approx(-0.03)


class TestTierRelativeOverround:
    """
    Margin is judged against normal for the market's kind, not on one
    absolute scale. Pinnacle charges ~3.5% a side on every MLB prop; that
    is what the asset class costs, not a signal about any one market.
    Scoring it against a mainline yardstick penalised props twice -- once
    via the tier factor, then again for the margin that defines the tier.
    """

    def test_a_market_at_its_tier_normal_margin_scores_full(self):
        for tier, baseline in L.TIER_OVERROUND_BASELINE.items():
            assert L.overround_factor(baseline * 2, 2, tier) == pytest.approx(1.0)

    def test_tighter_than_normal_also_scores_full(self):
        assert L.overround_factor(0.04, 2, L.TIER_PRIMARY_PROP) == pytest.approx(1.0)

    def test_double_the_normal_margin_bottoms_out(self):
        base = L.TIER_OVERROUND_BASELINE[L.TIER_PRIMARY_PROP]
        assert L.overround_factor(base * 4, 2, L.TIER_PRIMARY_PROP) == \
            pytest.approx(L.OVERROUND_MIN_FACTOR)

    def test_it_decreases_monotonically_within_a_tier(self):
        scores = [L.overround_factor(o, 2, L.TIER_PRIMARY_PROP)
                  for o in (0.06, 0.08, 0.10, 0.12, 0.16)]
        assert scores == sorted(scores, reverse=True)

    def test_the_measured_board_reproduces(self):
        """Live board, Sep 2026: 15 MLB games, 236 DraftKings quotes."""
        assert L.assess("h2h", 0.0250, 2, 0.50, 300).score == pytest.approx(1.0, abs=0.02)
        assert L.assess("pitcher_strikeouts", 0.0703, 2, 0.45, 300).score == \
            pytest.approx(0.75, abs=0.03)

    def test_an_unknown_tier_falls_back_to_a_prop_baseline(self):
        assert L.overround_factor(0.070, 2, "not_a_tier") == pytest.approx(1.0)
