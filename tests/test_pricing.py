"""The maths. These are the tests that matter most -- everything downstream
is plumbing, but an error here silently mis-prices every bet."""

import math

import pytest

from betedge import pricing as P


class TestConversion:
    @pytest.mark.parametrize("american,decimal", [
        (100, 2.0), (150, 2.5), (-120, 1 + 100 / 120), (-110, 1 + 100 / 110),
        (250, 3.5), (-200, 1.5),
    ])
    def test_american_to_decimal(self, american, decimal):
        assert P.american_to_decimal(american) == pytest.approx(decimal)

    @pytest.mark.parametrize("american", [100, 150, -120, -110, 250, -200, 1000])
    def test_round_trip(self, american):
        d = P.american_to_decimal(american)
        assert P.decimal_to_american(d) == pytest.approx(american)


class TestDevig:
    def test_overround_of_a_fair_market_is_zero(self):
        assert P.overround([2.0, 2.0]) == pytest.approx(0.0)

    def test_overround_matches_the_margin(self):
        # 1.91/1.91 is the standard -110/-110, about a 4.7% book.
        assert P.overround([1.91, 1.91]) == pytest.approx(0.04712, abs=1e-5)

    @pytest.mark.parametrize("method", ["multiplicative", "additive", "power", "shin"])
    def test_every_method_sums_to_one(self, method):
        for prices in ([1.91, 1.91], [1.05, 9.5], [1.6, 4.0, 5.5], [2.5, 3.1, 3.0]):
            probs = P.devig(prices, method=method)
            assert sum(probs) == pytest.approx(1.0, abs=1e-9)
            assert all(0.0 < p < 1.0 for p in probs)

    def test_symmetric_market_gives_even_probabilities(self):
        for method in ("multiplicative", "additive", "power", "shin"):
            a, b = P.devig([1.91, 1.91], method=method)
            assert a == pytest.approx(0.5, abs=1e-9)
            assert b == pytest.approx(0.5, abs=1e-9)

    def test_shin_equals_additive_on_two_way_markets(self):
        """The README's claim, and the reason `shin` is not the default:
        on any two-outcome market the two methods are algebraically the
        same, so choosing between them is not a real choice."""
        for prices in ([1.91, 1.91], [1.05, 9.5], [1.4, 3.1], [2.2, 1.72]):
            # 1e-12 is the honest bound: devig_shin bisects to a 1e-12
            # tolerance on the sum, leaving ~1e-13 per outcome.
            assert P.devig(prices, "shin") == pytest.approx(
                P.devig(prices, "additive"), abs=1e-12
            )

    def test_shin_differs_from_additive_on_three_way_markets(self):
        """...and where it stops being the same is where it earns its keep."""
        prices = [1.6, 4.0, 6.5]
        shin = P.devig(prices, "shin")
        additive = P.devig(prices, "additive")
        assert max(abs(a - b) for a, b in zip(shin, additive)) > 1e-4

    def test_methods_disagree_most_on_lopsided_markets(self):
        """The whole reason worst_case is the default."""
        even = P.devig_spread([1.91, 1.91])
        lopsided = P.devig_spread([1.05, 9.5])
        assert even < 0.005
        assert lopsided > 0.03
        assert lopsided > even

    def test_power_is_least_generous_to_longshots(self):
        prices = [1.05, 9.5]
        power = P.devig(prices, "power")[1]
        mult = P.devig(prices, "multiplicative")[1]
        assert power < mult

    def test_worst_case_is_at_or_below_every_method(self):
        for prices in ([1.91, 1.91], [1.05, 9.5], [1.6, 4.0, 5.5]):
            worst = P.devig(prices, "worst_case")
            for name in P.DEVIG_METHODS:
                other = P.devig(prices, name)
                assert all(w <= o + 1e-12 for w, o in zip(worst, other))

    def test_worst_case_does_not_have_to_sum_to_one(self):
        """Taking the minimum per outcome is a bound, not a distribution.
        That is intended -- it is used to test one side at a time."""
        assert sum(P.devig([1.05, 9.5], "worst_case")) <= 1.0 + 1e-12

    def test_unknown_method_is_rejected(self):
        with pytest.raises(ValueError, match="unknown devig method"):
            P.devig([1.91, 1.91], method="vibes")

    def test_devig_spread_does_not_depend_on_outcome_order(self):
        """
        The guard this feeds (max_devig_spread) has to describe the market,
        not the order the payload happened to list it in. Measuring only the
        first outcome scored this same market at 0.028 or 0.014 depending on
        which team sorted first alphabetically.
        """
        base = [1.25, 6.0, 11.0]
        spreads = {
            P.devig_spread(order)
            for order in ([1.25, 6.0, 11.0], [11.0, 6.0, 1.25], [6.0, 11.0, 1.25])
        }
        assert len(spreads) == 1

        first_only = {
            max(f(order)[0] for f in P.DEVIG_METHODS.values())
            - min(f(order)[0] for f in P.DEVIG_METHODS.values())
            for order in ([1.25, 6.0, 11.0], [11.0, 6.0, 1.25], [6.0, 11.0, 1.25])
        }
        assert len(first_only) > 1, "the old measure was order-dependent"

        grids = [f(base) for f in P.DEVIG_METHODS.values()]
        per_outcome = [
            max(g[i] for g in grids) - min(g[i] for g in grids)
            for i in range(len(base))
        ]
        assert P.devig_spread(base) == pytest.approx(max(per_outcome))


class TestExpectedValue:
    def test_fair_bet_has_zero_ev(self):
        assert P.expected_value(0.5, 2.0) == pytest.approx(0.0)

    def test_ev_is_edge_per_unit_staked(self):
        assert P.expected_value(0.52, 2.0) == pytest.approx(0.04)

    def test_negative_edge_is_negative(self):
        assert P.expected_value(0.48, 2.0) < 0


class TestStaking:
    def test_kelly_zero_at_fair_odds(self):
        assert P.kelly_fraction(0.5, 2.0) == pytest.approx(0.0)

    def test_kelly_matches_the_formula(self):
        # p=0.55, d=2.0 -> f* = (0.55*2 - 1)/1 = 0.10
        assert P.kelly_fraction(0.55, 2.0) == pytest.approx(0.10)

    def test_no_stake_without_an_edge(self):
        assert P.stake(0.48, 2.0, bankroll=1000) == 0.0

    def test_quarter_kelly_is_applied(self):
        # full Kelly 10% of 1000 = 100; quarter = 25, under the 2% cap of 20
        s = P.stake(0.55, 2.0, bankroll=1000, kelly_multiplier=0.25,
                    max_fraction=1.0, min_stake=0, round_to=0.01)
        assert s == pytest.approx(25.0)

    def test_max_fraction_caps_the_stake(self):
        s = P.stake(0.80, 2.0, bankroll=1000, kelly_multiplier=0.25,
                    max_fraction=0.02, min_stake=0, round_to=0.01)
        assert s == pytest.approx(20.0)

    def test_rounding_and_minimum(self):
        assert P.stake(0.51, 2.0, bankroll=1000, round_to=5, min_stake=5) % 5 == 0
        # A tiny edge rounds below the minimum and becomes no bet at all.
        assert P.stake(0.5005, 2.0, bankroll=100, round_to=5, min_stake=5) == 0.0


class TestTwoWayFair:
    def test_fair_prices_are_reciprocals_of_fair_probs(self):
        f = P.two_way_fair(1.91, 1.91, method="multiplicative")
        assert f.fair_price_a == pytest.approx(2.0)
        assert f.fair_price_b == pytest.approx(2.0)

    def test_no_vig_price_beats_the_posted_price(self):
        a, b = P.no_vig_price(1.91, 1.91)
        assert a > 1.91 and b > 1.91
