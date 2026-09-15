"""
The Gaussian copula, checked against results it must reproduce.

A Monte Carlo estimator is easy to write and easy to get subtly wrong, so
every claim here is anchored to something computed a different way: an
exact closed form, a quadrature, or an analytic convolution. Agreement
within the reported standard error is the bar.
"""

import math

import numpy as np
import pytest

from betedge import copula as C


class TestNormalHelpers:
    def test_ppf_inverts_cdf(self):
        for p in (0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999):
            assert C.norm_cdf(C.norm_ppf(p)) == pytest.approx(p, abs=1e-12)

    def test_known_quantiles(self):
        assert C.norm_ppf(0.975) == pytest.approx(1.959963984540054, abs=1e-9)
        assert C.norm_ppf(0.5) == pytest.approx(0.0, abs=1e-12)
        assert C.norm_ppf(0.05) == pytest.approx(-1.6448536269514722, abs=1e-9)

    def test_threshold_is_exceeded_with_the_right_probability(self):
        for p in (0.05, 0.4, 0.62, 0.95):
            t = C.threshold_for(p)
            assert 1.0 - C.norm_cdf(t) == pytest.approx(p, abs=1e-12)

    def test_degenerate_probabilities_are_clamped_not_infinite(self):
        assert math.isfinite(C.threshold_for(0.0))
        assert math.isfinite(C.threshold_for(1.0))

    def test_ppf_rejects_nonsense(self):
        with pytest.raises(ValueError):
            C.norm_ppf(float("nan"))


class TestBivariateReference:
    """
    The n = 2 orthant has a closed form at the median, and a convergent
    quadrature everywhere else. Both are independent of the simulator.
    """

    def test_matches_the_arcsine_identity_at_zero_thresholds(self):
        # P(Z1 > 0, Z2 > 0) = 1/4 + arcsin(rho) / (2 pi), exactly.
        for rho in (-0.8, -0.3, 0.0, 0.25, 0.5, 0.9):
            expected = 0.25 + math.asin(rho) / (2 * math.pi)
            assert C.bivariate_normal_upper(0.0, 0.0, rho) == pytest.approx(
                expected, abs=1e-12
            )

    def test_reduces_to_the_product_when_independent(self):
        t1, t2 = C.threshold_for(0.6), C.threshold_for(0.45)
        assert C.bivariate_normal_upper(t1, t2, 0.0) == pytest.approx(
            0.6 * 0.45, abs=1e-12
        )

    def test_rejects_a_degenerate_correlation(self):
        with pytest.raises(ValueError):
            C.bivariate_normal_upper(0.0, 0.0, 1.0)


class TestSimulationAgainstKnownResults:
    def test_independent_legs_give_the_product_of_the_marginals(self):
        probs = [0.62, 0.55, 0.48, 0.71]
        sim = C.simulate(probs, corr=None, draws=400_000, seed=7)
        expected = C.independent_joint_prob(probs)
        assert abs(sim.joint_prob - expected) < 4 * sim.joint_prob_se

    def test_an_identity_matrix_behaves_exactly_like_independence(self):
        probs = [0.6, 0.55, 0.5]
        a = C.simulate(probs, corr=None, draws=50_000, seed=3)
        b = C.simulate(probs, corr=np.eye(3), draws=50_000, seed=3)
        assert a.joint_prob == pytest.approx(b.joint_prob, abs=1e-12)

    def test_perfectly_correlated_identical_legs_give_the_single_leg_probability(self):
        # Three descriptions of one event cannot be harder to hit than one.
        probs = [0.58, 0.58, 0.58]
        corr = C.nearest_psd(np.ones((3, 3))).matrix
        sim = C.simulate(probs, corr, draws=200_000, seed=11)
        assert sim.joint_prob == pytest.approx(0.58, abs=0.005)

    def test_two_legs_match_the_bivariate_normal(self):
        for rho in (-0.5, 0.0, 0.35, 0.8):
            probs = [0.6, 0.45]
            corr = np.array([[1.0, rho], [rho, 1.0]])
            sim = C.simulate(probs, corr, draws=400_000, seed=23)
            exact = C.bivariate_normal_upper(
                C.threshold_for(0.6), C.threshold_for(0.45), rho
            )
            assert abs(sim.joint_prob - exact) < 4 * sim.joint_prob_se

    def test_positive_correlation_raises_the_joint_probability(self):
        probs = [0.6, 0.6, 0.6]
        base = C.standard_normals(200_000, 3, seed=5)
        independent = C.simulate(probs, np.eye(3), base=base)
        correlated = C.simulate(probs, np.full((3, 3), 0.4) + 0.6 * np.eye(3), base=base)
        assert correlated.joint_prob > independent.joint_prob

    def test_negative_correlation_lowers_it(self):
        probs = [0.6, 0.6]
        base = C.standard_normals(200_000, 2, seed=5)
        independent = C.simulate(probs, np.eye(2), base=base)
        opposed = C.simulate(probs, np.array([[1.0, -0.5], [-0.5, 1.0]]), base=base)
        assert opposed.joint_prob < independent.joint_prob


class TestOutcomeDistribution:
    def test_partial_hit_probabilities_sum_to_one(self):
        sim = C.simulate([0.6, 0.5, 0.45, 0.7, 0.55], draws=50_000, seed=1)
        assert sum(sim.hit_distribution) == pytest.approx(1.0, abs=1e-12)
        assert len(sim.hit_distribution) == 6

    def test_the_whole_grid_sums_to_one_with_pushes(self):
        sim = C.simulate(
            [0.5, 0.5, 0.5], push_probs=[0.1, 0.05, 0.2], draws=50_000, seed=2
        )
        assert sim.grid.sum() == pytest.approx(1.0, abs=1e-12)

    def test_partial_hits_match_the_binomial_when_independent(self):
        p, n = 0.6, 4
        sim = C.simulate([p] * n, draws=400_000, seed=4)
        for k in range(n + 1):
            expected = math.comb(n, k) * p ** k * (1 - p) ** (n - k)
            assert sim.prob_exactly(k) == pytest.approx(expected, abs=0.004)

    def test_a_push_takes_its_mass_from_the_misses_not_the_hits(self):
        # `hit_probs` are unconditional: the push region sits BELOW the
        # winning threshold, so adding push mass converts losses into
        # voids and leaves the all-hit probability exactly where it was.
        # `parlay.Leg.hit_prob` is what folds the push into the marginal.
        without = C.simulate([0.5, 0.5], draws=100_000, seed=6)
        with_push = C.simulate(
            [0.5, 0.5], push_probs=[0.2, 0.2], draws=100_000, seed=6
        )
        assert with_push.joint_prob == pytest.approx(without.joint_prob)
        assert with_push.prob_any_void > 0.3
        assert without.prob_any_void == 0.0
        # No-hit outcomes lose exactly the mass the voids gained.
        assert with_push.grid[0, 0] < without.grid[0, 0]

    def test_standard_error_shrinks_with_the_square_root_of_the_draws(self):
        small = C.simulate([0.5, 0.5], draws=10_000, seed=8)
        large = C.simulate([0.5, 0.5], draws=1_000_000, seed=8)
        assert large.joint_prob_se == pytest.approx(small.joint_prob_se / 10, rel=0.1)


class TestExactIndependentGrid:
    def test_matches_the_simulation_within_error(self):
        probs = [0.6, 0.55, 0.48]
        exact = C.independent_grid(probs)
        sim = C.simulate(probs, draws=400_000, seed=9)
        assert abs(exact[0, 3] - sim.joint_prob) < 4 * sim.joint_prob_se

    def test_sums_to_one_with_and_without_pushes(self):
        assert C.independent_grid([0.5, 0.4, 0.3]).sum() == pytest.approx(1.0)
        assert C.independent_grid(
            [0.5, 0.4], [0.1, 0.25]
        ).sum() == pytest.approx(1.0)

    def test_all_hit_is_the_product_of_the_marginals(self):
        probs = [0.61, 0.52, 0.49, 0.7]
        grid = C.independent_grid(probs)
        assert grid[0, 4] == pytest.approx(C.independent_joint_prob(probs), abs=1e-15)

    def test_a_pushed_leg_moves_mass_into_the_void_row(self):
        grid = C.independent_grid([0.5, 0.5], [0.0, 0.3])
        assert grid[1, :].sum() == pytest.approx(0.3, abs=1e-12)


class TestPsdProjection:
    def test_a_valid_matrix_is_left_alone(self):
        m = np.array([[1.0, 0.3], [0.3, 1.0]])
        result = C.nearest_psd(m)
        assert not result.projected
        assert result.note == ""
        assert np.allclose(result.matrix, m)

    def test_a_deliberately_impossible_matrix_is_repaired(self):
        # A and B move together, B and C move together, but A and C are
        # said to move apart. No three variables can do all three.
        bad = np.array([
            [1.0, 0.9, -0.9],
            [0.9, 1.0, 0.9],
            [-0.9, 0.9, 1.0],
        ])
        assert not C.is_psd(bad)
        result = C.nearest_psd(bad)
        assert result.projected
        assert result.min_eigenvalue < 0
        assert C.is_psd(result.matrix)
        assert "not positive semi-definite" in result.note

    def test_projection_keeps_a_unit_diagonal(self):
        bad = np.array([[1.0, 0.95, -0.95], [0.95, 1.0, 0.95], [-0.95, 0.95, 1.0]])
        repaired = C.nearest_psd(bad).matrix
        assert np.allclose(np.diag(repaired), 1.0)
        assert np.allclose(repaired, repaired.T)

    def test_projection_reports_how_far_it_moved_things(self):
        bad = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
        assert C.nearest_psd(bad).max_adjustment > 0.05

    def test_a_projected_matrix_can_then_be_simulated_from(self):
        bad = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
        sim = C.simulate([0.5, 0.5, 0.5], C.nearest_psd(bad).matrix, draws=5_000)
        assert 0.0 <= sim.joint_prob <= 1.0

    def test_a_singular_matrix_still_factorises(self):
        ones = np.ones((3, 3))
        factor = C.cholesky_factor(C.nearest_psd(ones).matrix)
        assert factor.shape == (3, 3)
        assert np.all(np.isfinite(factor))


class TestSimulationGuards:
    def test_probabilities_outside_the_open_unit_interval_are_refused(self):
        with pytest.raises(ValueError):
            C.simulate([0.5, 1.0])
        with pytest.raises(ValueError):
            C.simulate([0.0, 0.5])

    def test_hit_plus_push_must_stay_below_one(self):
        with pytest.raises(ValueError):
            C.simulate([0.8, 0.5], push_probs=[0.3, 0.1])

    def test_a_mismatched_matrix_is_refused(self):
        with pytest.raises(ValueError):
            C.simulate([0.5, 0.5], np.eye(3))

    def test_a_shared_pool_too_narrow_for_the_ticket_is_refused(self):
        base = C.standard_normals(100, 2, seed=1)
        with pytest.raises(ValueError):
            C.simulate([0.5, 0.5, 0.5], base=base)

    def test_the_same_seed_gives_the_same_answer_twice(self):
        a = C.simulate([0.6, 0.55], draws=20_000, seed=99)
        b = C.simulate([0.6, 0.55], draws=20_000, seed=99)
        assert a.joint_prob == b.joint_prob


class TestStakingHelpers:
    def test_log_optimal_matches_kelly_on_a_two_outcome_bet(self):
        # A bet at even money winning 60% of the time: Kelly says 20%.
        returns = np.where(np.arange(1_000_000) < 600_000, 2.0, 0.0)
        assert C.log_optimal_fraction(returns) == pytest.approx(0.20, abs=0.005)

    def test_a_negative_edge_gets_nothing(self):
        returns = np.where(np.arange(100_000) < 40_000, 2.0, 0.0)
        assert C.log_optimal_fraction(returns) == 0.0

    def test_a_lumpy_payoff_gets_a_smaller_fraction_than_its_edge_suggests(self):
        # Same +8% edge, paid as a rare 10x instead of a frequent 2x.
        frequent = np.where(np.arange(1_000_000) < 540_000, 2.0, 0.0)
        lumpy = np.where(np.arange(1_000_000) < 108_000, 10.0, 0.0)
        assert frequent.mean() == pytest.approx(lumpy.mean(), rel=0.01)
        assert C.log_optimal_fraction(lumpy) < C.log_optimal_fraction(frequent)


class TestCompoundedHold:
    def test_four_legs_at_four_and_a_half_percent(self):
        # The arithmetic behind "don't build cross-game parlays".
        assert C.compounded_hold([0.045] * 4) == pytest.approx(1.045 ** 4 - 1)
        assert C.compounded_hold([0.045] * 4) > 0.19

    def test_no_legs_means_no_hold(self):
        assert C.compounded_hold([]) == 0.0

    def test_it_is_worse_than_adding_the_margins_up(self):
        holds = [0.05, 0.06, 0.04]
        assert C.compounded_hold(holds) > sum(holds)
