"""
The Rotten Tomatoes threshold engine.

The load-bearing tests here are the ones that check the two halves agree:
`decided` says a contract is settled by arithmetic, and `probability_yes`
independently arrives at 1.0 or 0.0 for the same contract. They are
computed by completely different routes -- bounds versus a Beta-Binomial
sum -- so agreement between them is real evidence, and disagreement means
one of the rounding rules is wrong.
"""

import math
from datetime import datetime, timedelta, timezone

import pytest

from betedge import tomatoes as T

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


def snap(fresh, total, scope=T.SCOPE_ALL_CRITICS, film="A Film", at=NOW):
    return T.Tomatometer(fresh=fresh, total=total, captured_at=at,
                         scope=scope, film=film)


def contract(threshold=60, direction=T.ABOVE, inclusive=True, **kw):
    return T.Contract(
        ticker=kw.pop("ticker", "RT-FILM-60"),
        film=kw.pop("film", "A Film"),
        threshold=threshold, direction=direction, inclusive=inclusive, **kw,
    )


# --------------------------------------------------------------------------
# The snapshot
# --------------------------------------------------------------------------


class TestTomatometer:
    def test_the_score_is_the_proportion(self):
        assert snap(169, 180).exact_score == pytest.approx(93.888, abs=1e-3)

    def test_the_displayed_score_is_what_settles(self):
        # 93.888 -> 94, and it is 94 that a contract is written against.
        assert snap(169, 180).displayed_score == 94

    def test_rounding_is_half_up_and_stated(self):
        # Exactly x.5 has to go somewhere, and which way is a whole
        # contract on a threshold sitting at the boundary.
        assert snap(121, 200).displayed_score == 61   # 60.5 -> 61
        assert snap(1, 8).displayed_score == 13       # 12.5 -> 13

    def test_more_fresh_than_reviews_is_rejected(self):
        with pytest.raises(ValueError, match="more Fresh"):
            snap(10, 5)

    def test_a_negative_count_is_rejected(self):
        with pytest.raises(ValueError, match="negative"):
            snap(-1, 5)

    def test_an_unknown_scope_is_rejected(self):
        # All Critics and Top Critics differ by several points on the same
        # film, so a snapshot that does not say which is not usable.
        with pytest.raises(ValueError, match="scope"):
            snap(10, 20, scope="whatever")

    def test_no_reviews_is_not_a_division_by_zero(self):
        assert snap(0, 0).exact_score == 0.0
        assert snap(0, 0).displayed_score == 0

    def test_age_is_reported_for_staleness_checks(self):
        s = snap(90, 100, at=NOW - timedelta(hours=5))
        assert s.age_hours(NOW) == pytest.approx(5.0)

    def test_it_describes_itself_with_the_count(self):
        # The count is the point -- a description that hid it would be
        # the same mistake as a source that only gives the percentage.
        text = snap(169, 180).describe()
        assert "94%" in text and "169/180" in text


# --------------------------------------------------------------------------
# Bounds: the arithmetic half
# --------------------------------------------------------------------------


class TestBounds:
    def test_the_worked_example(self):
        # 169 fresh of 180, 40 more reviews possible:
        #   worst 169/220 = 76.8 -> 77
        #   best  209/220 = 95.0 -> 95
        b = T.bounds(snap(169, 180), max_new_reviews=40)
        assert (b.low, b.high) == (77, 95)

    def test_no_further_reviews_pins_the_score_exactly(self):
        b = T.bounds(snap(169, 180), max_new_reviews=0)
        assert b.low == b.high == 94
        assert b.is_settled

    def test_bounds_only_widen_as_more_reviews_are_allowed(self):
        # The ceiling is conservative by construction: raising it can
        # never make the tool more certain.
        s = snap(169, 180)
        previous = T.bounds(s, 0)
        for n in range(1, 60):
            current = T.bounds(s, n)
            assert current.low <= previous.low
            assert current.high >= previous.high
            previous = current

    def test_both_ends_are_attainable(self):
        s = snap(80, 100)
        b = T.bounds(s, 20)
        assert b.low == snap(80, 120).displayed_score
        assert b.high == snap(100, 120).displayed_score

    def test_a_negative_ceiling_is_rejected(self):
        with pytest.raises(ValueError, match="negative"):
            T.bounds(snap(80, 100), -1)

    def test_no_reviews_at_all_is_not_a_crash(self):
        b = T.bounds(snap(0, 0), 0)
        assert (b.low, b.high) == (0, 0)


# --------------------------------------------------------------------------
# Contracts
# --------------------------------------------------------------------------


class TestContract:
    def test_inclusive_thresholds_count_the_boundary(self):
        c = contract(threshold=60, inclusive=True)
        assert c.resolves_yes(60)
        assert c.resolves_yes(61)
        assert not c.resolves_yes(59)

    def test_exclusive_thresholds_do_not(self):
        # One point of difference, and it is a whole contract.
        c = contract(threshold=60, inclusive=False)
        assert not c.resolves_yes(60)
        assert c.resolves_yes(61)

    def test_a_below_contract_runs_the_other_way(self):
        c = contract(threshold=40, direction=T.BELOW, inclusive=True)
        assert c.resolves_yes(40)
        assert c.resolves_yes(10)
        assert not c.resolves_yes(41)

    def test_settlement_is_unverified_until_a_person_checks(self):
        # Same rule as the payout ladders: the scope, the inclusivity and
        # the settlement time are read off the contract by a human.
        assert contract().settlement_verified is False

    def test_a_nonsense_direction_is_rejected(self):
        with pytest.raises(ValueError, match="direction"):
            contract(direction="sideways")

    def test_a_threshold_outside_a_percentage_is_rejected(self):
        with pytest.raises(ValueError, match="percentage"):
            contract(threshold=140)


# --------------------------------------------------------------------------
# Decided: where the money is
# --------------------------------------------------------------------------


class TestDecided:
    def test_a_high_scorer_is_already_past_a_low_threshold(self):
        # 169/180 with 40 to come cannot fall below 77.
        assert T.decided(snap(169, 180), contract(60), 40) is True

    def test_a_low_scorer_cannot_reach_a_high_threshold(self):
        assert T.decided(snap(20, 180), contract(80), 40) is False

    def test_a_threshold_inside_the_bounds_is_open(self):
        assert T.decided(snap(169, 180), contract(85), 40) is None

    def test_a_wide_enough_ceiling_reopens_a_decided_contract(self):
        # The honest direction: allowing more reviews can only remove
        # certainty, never create it.
        assert T.decided(snap(169, 180), contract(80), 10) is True
        assert T.decided(snap(169, 180), contract(80), 400) is None

    def test_it_agrees_with_the_bounds_it_is_built_from(self):
        s = snap(140, 160)
        b = T.bounds(s, 30)
        for threshold in range(0, 101):
            c = contract(threshold)
            verdict = T.decided(s, c, 30)
            if threshold <= b.low:
                assert verdict is True, threshold
            elif threshold > b.high:
                assert verdict is False, threshold
            else:
                assert verdict is None, threshold


# --------------------------------------------------------------------------
# The Beta-Binomial
# --------------------------------------------------------------------------


class TestBetaBinomial:
    def test_the_pmf_sums_to_one(self):
        total = sum(T.beta_binomial_pmf(k, 25, 6.0, 3.0) for k in range(26))
        assert total == pytest.approx(1.0, abs=1e-12)

    def test_it_reduces_to_the_binomial_when_the_rate_is_known(self):
        # A hugely concentrated Beta is a point mass at its mean, and then
        # the Beta-Binomial IS the binomial. Computed here independently
        # from the closed form.
        n, p, c = 10, 0.7, 4_000_000.0
        alpha, beta = p * c, (1 - p) * c
        for k in range(n + 1):
            expected = math.comb(n, k) * p ** k * (1 - p) ** (n - k)
            assert T.beta_binomial_pmf(k, n, alpha, beta) == pytest.approx(
                expected, abs=1e-6
            )

    def test_a_flat_prior_makes_every_count_equally_likely(self):
        # Beta(1,1) over n draws is the discrete uniform on 0..n -- a
        # known result, and a good check that the Beta functions are the
        # right way round.
        for k in range(9):
            assert T.beta_binomial_pmf(k, 8, 1.0, 1.0) == pytest.approx(1 / 9)

    def test_it_is_wider_than_a_binomial_at_the_same_mean(self):
        # The whole reason for using it: an estimated rate spreads the
        # outcome, and a binomial would understate the chance of crossing
        # a threshold.
        n, p = 20, 0.6
        alpha, beta = 6.0, 4.0        # mean 0.6, weakly held
        centre = 12
        bb = T.beta_binomial_pmf(centre, n, alpha, beta)
        binomial = math.comb(n, centre) * p ** centre * (1 - p) ** (n - centre)
        assert bb < binomial

    def test_counts_outside_the_range_have_no_mass(self):
        assert T.beta_binomial_pmf(-1, 5, 2.0, 2.0) == 0.0
        assert T.beta_binomial_pmf(6, 5, 2.0, 2.0) == 0.0

    def test_no_draws_is_a_certainty(self):
        assert T.beta_binomial_pmf(0, 0, 2.0, 2.0) == 1.0


class TestArrivalDistribution:
    def test_it_is_a_distribution(self):
        d = T.arrival_distribution(12.0, max_new=60)
        assert sum(d.values()) == pytest.approx(1.0)
        assert all(v >= 0 for v in d.values())

    def test_it_never_exceeds_the_ceiling_the_bounds_used(self):
        # The two halves of the module must agree about what is
        # reachable, or a contract can be "decided" and still be given a
        # probability strictly between 0 and 1.
        d = T.arrival_distribution(30.0, max_new=10)
        assert max(d) <= 10

    def test_expecting_none_means_none(self):
        assert T.arrival_distribution(0.0, max_new=5) == {0: 1.0}

    def test_a_negative_expectation_is_rejected(self):
        with pytest.raises(ValueError, match="negative"):
            T.arrival_distribution(-1.0)


# --------------------------------------------------------------------------
# probability_yes -- and its agreement with `decided`
# --------------------------------------------------------------------------


class TestProbabilityYes:
    def test_a_decided_yes_prices_at_one(self):
        s, c = snap(169, 180), contract(60)
        assert T.decided(s, c, 40) is True
        assert T.probability_yes(s, c, expected_new=20, max_new_reviews=40) == \
            pytest.approx(1.0, abs=1e-9)

    def test_a_decided_no_prices_at_zero(self):
        s, c = snap(20, 180), contract(80)
        assert T.decided(s, c, 40) is False
        assert T.probability_yes(s, c, expected_new=20, max_new_reviews=40) == \
            pytest.approx(0.0, abs=1e-9)

    def test_the_two_halves_never_disagree(self):
        """
        The load-bearing test. `decided` works from bounds, this works
        from a Beta-Binomial sum, and they must agree everywhere -- if
        one of the rounding rules were wrong they would not.
        """
        s = snap(120, 150)
        for threshold in range(30, 100, 3):
            for inclusive in (True, False):
                c = contract(threshold, inclusive=inclusive)
                verdict = T.decided(s, c, 25)
                p = T.probability_yes(
                    s, c, expected_new=12, max_new_reviews=25
                )
                if verdict is True:
                    assert p == pytest.approx(1.0, abs=1e-9), (threshold, inclusive)
                elif verdict is False:
                    assert p == pytest.approx(0.0, abs=1e-9), (threshold, inclusive)
                else:
                    assert 0.0 < p < 1.0, (threshold, inclusive)

    def test_with_no_reviews_coming_it_is_the_current_score(self):
        s = snap(94, 100)
        assert T.probability_yes(s, contract(94), 0, 0) == pytest.approx(1.0)
        assert T.probability_yes(s, contract(95), 0, 0) == pytest.approx(0.0)

    def test_a_higher_threshold_is_never_more_likely(self):
        s = snap(120, 150)
        previous = 1.0
        for threshold in range(50, 96):
            p = T.probability_yes(s, contract(threshold), 20, 40)
            assert p <= previous + 1e-12, threshold
            previous = p

    def test_a_below_contract_is_the_complement_of_its_above_twin(self):
        s = snap(120, 150)
        above = T.probability_yes(s, contract(70, inclusive=True), 20, 40)
        below = T.probability_yes(
            s, contract(69, direction=T.BELOW, inclusive=True), 20, 40
        )
        assert above + below == pytest.approx(1.0, abs=1e-9)

    def test_the_prior_barely_matters_on_a_large_sample(self):
        s = snap(120, 150)
        weak = T.probability_yes(s, contract(78), 20, 40,
                                 prior_alpha=2.0, prior_beta=2.0)
        different = T.probability_yes(s, contract(78), 20, 40,
                                      prior_alpha=1.0, prior_beta=3.0)
        assert abs(weak - different) < 0.02

    def test_the_prior_does_matter_on_a_tiny_one(self):
        # Which is exactly when the guards must refuse to stake.
        s = snap(4, 5)
        optimistic = T.probability_yes(s, contract(70), 40, 60,
                                       prior_alpha=8.0, prior_beta=1.0)
        pessimistic = T.probability_yes(s, contract(70), 40, 60,
                                        prior_alpha=1.0, prior_beta=8.0)
        assert optimistic - pessimistic > 0.1


class TestDrift:
    def test_zero_drift_changes_nothing(self):
        assert T._shift_posterior(30.0, 10.0, 0.0) == (30.0, 10.0)

    def test_it_preserves_the_concentration(self):
        a, b = T._shift_posterior(30.0, 10.0, 0.5)
        assert a + b == pytest.approx(40.0)

    def test_positive_drift_means_harsher_later_reviews(self):
        a, b = T._shift_posterior(30.0, 10.0, 0.5)
        assert a / (a + b) < 0.75

    def test_it_lowers_the_chance_of_clearing_a_threshold(self):
        s = snap(19, 20)      # 95% on twenty festival reviews
        c = contract(85)
        naive = T.probability_yes(s, c, expected_new=120, max_new_reviews=200)
        drifted = T.probability_yes(s, c, expected_new=120,
                                    max_new_reviews=200, drift=0.8)
        assert drifted < naive

    def test_it_cannot_rescue_a_decided_contract(self):
        # Arithmetic outranks the prior: no drift assumption may move a
        # contract that the counting has already settled.
        s, c = snap(169, 180), contract(60)
        assert T.probability_yes(s, c, 20, 40, drift=2.0) == pytest.approx(1.0)
        assert T.probability_yes(s, c, 20, 40, drift=-2.0) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Fees and EV
# --------------------------------------------------------------------------


class TestFees:
    def test_the_published_formula_at_fifty_cents(self):
        # 0.07 * 100 * 0.5 * 0.5 = $1.75
        assert T.fee(100, 0.50) == pytest.approx(1.75)

    def test_the_maker_coefficient_is_a_quarter_of_the_taker(self):
        assert T.fee(100, 0.50, maker=True) == pytest.approx(0.44)  # 0.4375 -> 0.44

    def test_rounding_up_bites_a_single_contract(self):
        # 1.75c rounds to 2c: 4% of a 50c stake, not 3.5%. A model that
        # used the bare formula would overrate exactly the small
        # positions these thin markets force on you.
        assert T.fee(1, 0.50) == pytest.approx(0.02)

    def test_the_fee_peaks_at_the_coin_flip_per_contract(self):
        at_half = T.fee(1000, 0.50)
        assert at_half > T.fee(1000, 0.20)
        assert at_half > T.fee(1000, 0.80)

    def test_but_as_a_share_of_stake_it_falls_with_price(self):
        # 0.07 * (1 - P) of stake: cheap on favourites, dear on
        # longshots. The opposite of where most people look.
        for price in (0.2, 0.5, 0.8, 0.9):
            share = T.fee(10_000, price) / (10_000 * price)
            assert share == pytest.approx(0.07 * (1 - price), abs=1e-4)

    def test_no_contracts_costs_nothing(self):
        assert T.fee(0, 0.5) == 0.0

    def test_a_price_outside_a_probability_is_rejected(self):
        with pytest.raises(ValueError, match="probability"):
            T.fee(10, 1.4)


class TestBreakevenEdge:
    def test_the_taker_needs_one_and_three_quarter_cents_at_the_coin_flip(self):
        assert T.breakeven_edge(0.50) == pytest.approx(0.0175)

    def test_the_maker_needs_a_quarter_of_that(self):
        assert T.breakeven_edge(0.50, maker=True) == pytest.approx(0.004375)

    def test_it_is_cheaper_at_the_tails(self):
        assert T.breakeven_edge(0.90) < T.breakeven_edge(0.50)
        assert T.breakeven_edge(0.10) < T.breakeven_edge(0.50)


class TestExpectedValue:
    def test_a_fair_price_is_negative_by_the_fee(self):
        """
        Paying the true probability still loses. Note the size: the fee is
        3.5% OF STAKE at the coin flip, but the loss is 3.38% of capital
        COMMITTED, because the fee is part of what you committed. The
        smaller number is the honest one and the difference is exactly
        why the fee belongs in the denominator.
        """
        n, price = 10_000, 0.50
        ev = T.ev_on_stake(price, price, contracts=n)
        paid = T.fee(n, price)
        assert ev == pytest.approx(-paid / (n * price + paid), abs=1e-12)
        assert ev == pytest.approx(-0.0338, abs=1e-4)
        # The naive figure, for contrast -- what quoting EV against the
        # bare stake would have told you.
        assert paid / (n * price) == pytest.approx(0.035, abs=1e-6)

    def test_the_breakeven_edge_is_where_the_sign_flips(self):
        price = 0.50
        edge = T.breakeven_edge(price)
        contracts = 100_000          # large, so cent rounding is nothing
        assert T.ev_on_stake(price + edge * 0.5, price, contracts) < 0
        assert T.ev_on_stake(price + edge * 1.5, price, contracts) > 0

    def test_a_maker_clears_an_edge_a_taker_cannot(self):
        # The one structural advantage on this exchange that needs no
        # opinion about the film at all.
        price, contracts = 0.50, 100_000
        probability = price + 0.008
        assert T.ev_on_stake(probability, price, contracts, maker=False) < 0
        assert T.ev_on_stake(probability, price, contracts, maker=True) > 0

    def test_the_fee_is_in_the_denominator_too(self):
        # It is paid up front, so it is committed capital. Quoting EV
        # against the bare price would flatter every position.
        p, price, n = 0.60, 0.50, 1000
        cost = n * price + T.fee(n, price)
        assert T.ev_on_stake(p, price, n) == pytest.approx(
            (p * n - cost) / cost
        )

    def test_a_certain_winner_bought_cheap_is_a_large_edge(self):
        assert T.ev_on_stake(1.0, 0.92, 10_000) > 0.05

    def test_zero_contracts_is_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            T.ev_on_stake(0.6, 0.5, contracts=0)


# --------------------------------------------------------------------------
# The assessment
# --------------------------------------------------------------------------


class TestAssessment:
    def build(self, ev, ev_no_drift):
        return T.Assessment(
            contract=contract(), snapshot=snap(120, 150), price=0.5,
            probability=0.55, probability_no_drift=0.52,
            bounds=T.bounds(snap(120, 150), 30), decided=None,
            ev=ev, ev_no_drift=ev_no_drift,
        )

    def test_drift_carrying_the_bet_is_visible(self):
        # The honesty check, and the direct analogue of
        # `only_+ev_because_of_assumed_correlation` on the parlay side.
        assert self.build(ev=0.04, ev_no_drift=-0.01).drift_is_load_bearing

    def test_a_bet_that_stands_without_drift_is_not_flagged(self):
        assert not self.build(ev=0.06, ev_no_drift=0.02).drift_is_load_bearing

    def test_a_loser_either_way_is_not_flagged_as_drift_driven(self):
        assert not self.build(ev=-0.03, ev_no_drift=-0.05).drift_is_load_bearing

    def test_the_edge_is_reported_in_probability(self):
        assert self.build(0.1, 0.1).edge == pytest.approx(0.05)


# --------------------------------------------------------------------------
# assess(): the guards
# --------------------------------------------------------------------------


@pytest.fixture
def tcfg():
    from betedge.config import TomatoesConfig

    # Verified-by-default is wrong for real use but right for these tests:
    # otherwise every case trips the settlement guard and nothing else is
    # ever exercised. The unverified path has its own test below.
    return TomatoesConfig()


def verified(threshold=60, **kw):
    kw.setdefault("settlement_verified", True)
    return contract(threshold, **kw)


class TestAssessGuards:
    def test_an_unverified_contract_is_never_staked(self, tcfg):
        # The likeliest failure on these markets is clerical, not
        # statistical: the wrong Tomatometer or the wrong inclusivity.
        a = T.assess(snap(169, 180), contract(60), price=0.80, cfg=tcfg, now=NOW)
        assert "settlement_terms_unverified" in a.flags
        assert a.suspect
        assert T.position_size(a, 1000.0, tcfg) == 0.0

    def test_a_decided_contract_says_so_and_names_the_range(self, tcfg):
        a = T.assess(snap(169, 180), verified(60), price=0.80, cfg=tcfg,
                     now=NOW, max_new_reviews=40)
        assert a.decided is True
        assert any("decided_by_arithmetic" in f for f in a.flags)
        assert "77-95%" in " ".join(a.flags)
        assert not a.suspect

    def test_a_decided_contract_bought_below_par_is_a_real_position(self, tcfg):
        a = T.assess(snap(169, 180), verified(60), price=0.80, cfg=tcfg,
                     now=NOW, max_new_reviews=40)
        assert a.probability == pytest.approx(1.0)
        assert a.ev > 0.2
        assert T.position_size(a, 1000.0, tcfg) > 0

    def test_a_stale_snapshot_is_not_a_measurement(self, tcfg):
        old = snap(169, 180, at=NOW - timedelta(hours=30))
        a = T.assess(old, verified(60), price=0.80, cfg=tcfg, now=NOW)
        assert any("snapshot_30h_old" in f for f in a.flags)
        assert a.suspect
        assert T.position_size(a, 1000.0, tcfg) == 0.0

    def test_comparing_two_different_tomatometers_is_refused(self, tcfg):
        # All Critics against Top Critics on the same film differs by
        # several points. Same rule as never comparing two prop lines.
        a = T.assess(
            snap(169, 180, scope=T.SCOPE_ALL_CRITICS),
            verified(60, scope=T.SCOPE_TOP_CRITICS),
            price=0.80, cfg=tcfg, now=NOW,
        )
        assert any("scope_mismatch" in f for f in a.flags)
        assert a.suspect

    def test_too_few_reviews_is_refused_on_an_open_contract(self, tcfg):
        a = T.assess(snap(8, 9), verified(70), price=0.50, cfg=tcfg, now=NOW)
        assert a.decided is None
        assert any("only_9_reviews_counted" in f for f in a.flags)
        assert a.suspect

    def test_but_a_tiny_sample_can_still_decide_a_contract(self, tcfg):
        # Nine reviews all Fresh cannot produce a score below 13% even if
        # sixty rotten ones arrive. Arithmetic does not care about sample
        # size, and the guard must not override it.
        a = T.assess(snap(9, 9), verified(10), price=0.90, cfg=tcfg, now=NOW)
        assert a.decided is True
        assert not a.suspect

    def test_an_implausible_edge_means_a_mistake_not_an_opportunity(self, tcfg):
        a = T.assess(snap(100, 150), verified(55), price=0.20, cfg=tcfg, now=NOW)
        assert a.decided is None
        assert any("implausible" in f for f in a.flags)
        assert a.suspect

    def test_a_decided_contract_is_exempt_from_the_plausibility_ceiling(self, tcfg):
        # A 60% edge on a contract settled by counting is not implausible,
        # it is the product working.
        a = T.assess(snap(169, 180), verified(60), price=0.55, cfg=tcfg,
                     now=NOW, max_new_reviews=40)
        assert a.ev > tcfg.max_plausible_ev
        assert not any("implausible" in f for f in a.flags)
        assert T.position_size(a, 1000.0, tcfg) > 0

    def test_maker_pricing_is_disclosed(self, tcfg):
        a = T.assess(snap(169, 180), verified(60), price=0.80, cfg=tcfg,
                     now=NOW, maker=True)
        assert any("priced_as_maker" in f for f in a.flags)

    def test_takers_are_assumed_by_default(self, tcfg):
        # Assuming the maker fee would flatter every position by four
        # times the fee.
        assert tcfg.assume_maker is False


class TestAssessDrift:
    def test_drift_that_carries_the_bet_is_refused(self):
        """
        Built directly rather than searched for, so the assertion cannot
        quietly evaporate: an assessment that is +EV with the drift and
        -EV without it must be flagged and must get no money.
        """
        from betedge.config import TomatoesConfig

        cfg = TomatoesConfig(drift=-1.5)
        s = snap(120, 150)
        a = T.Assessment(
            contract=verified(78), snapshot=s, price=0.50,
            probability=0.58, probability_no_drift=0.48,
            bounds=T.bounds(s, 60), decided=None,
            ev=0.12, ev_no_drift=-0.06,
        )
        assert a.drift_is_load_bearing
        T.apply_guards(a, cfg, now=NOW)
        assert "only_+ev_because_of_assumed_drift" in a.flags
        assert a.suspect
        assert T.position_size(a, 1000.0, cfg) == 0.0

    def test_a_bet_that_survives_without_drift_still_gets_money(self):
        from betedge.config import TomatoesConfig

        cfg = TomatoesConfig(drift=-1.5)
        s = snap(120, 150)
        a = T.Assessment(
            contract=verified(78), snapshot=s, price=0.50,
            probability=0.62, probability_no_drift=0.58,
            bounds=T.bounds(s, 60), decided=None,
            ev=0.20, ev_no_drift=0.12,
        )
        T.apply_guards(a, cfg, now=NOW)
        assert "only_+ev_because_of_assumed_drift" not in a.flags
        assert not a.suspect
        assert T.position_size(a, 1000.0, cfg) > 0

    def test_an_applied_drift_is_always_disclosed(self):
        from betedge.config import TomatoesConfig

        cfg = TomatoesConfig(drift=0.6)
        a = T.assess(snap(120, 150), verified(78), price=0.50, cfg=cfg, now=NOW)
        assert any("drift_prior" in f and "not_measured" in f for f in a.flags)

    def test_with_no_drift_the_two_numbers_are_the_same(self, tcfg):
        a = T.assess(snap(120, 150), verified(78), price=0.5, cfg=tcfg, now=NOW)
        assert a.probability == a.probability_no_drift
        assert not a.drift_is_load_bearing


class TestPositionSize:
    def test_nothing_is_staked_below_the_ev_bar(self, tcfg):
        a = T.assess(snap(120, 150), verified(60), price=0.99, cfg=tcfg, now=NOW)
        assert a.ev < tcfg.min_ev
        assert T.position_size(a, 1000.0, tcfg) == 0.0

    def test_a_suspect_assessment_gets_nothing_not_less(self, tcfg):
        # A smaller stake on a number you think is wrong is still a bet on
        # a number you think is wrong.
        a = T.assess(snap(169, 180), contract(60), price=0.60, cfg=tcfg, now=NOW)
        assert a.suspect and a.ev > 0
        assert T.position_size(a, 1000.0, tcfg) == 0.0

    def test_the_cap_binds_before_kelly_does(self, tcfg):
        a = T.assess(snap(169, 180), verified(60), price=0.50, cfg=tcfg,
                     now=NOW, max_new_reviews=40)
        size = T.position_size(a, 10_000.0, tcfg)
        assert size == pytest.approx(10_000.0 * tcfg.max_position_fraction)

    def test_it_scales_with_the_bankroll(self, tcfg):
        a = T.assess(snap(169, 180), verified(60), price=0.80, cfg=tcfg,
                     now=NOW, max_new_reviews=40)
        assert T.position_size(a, 2000.0, tcfg) == pytest.approx(
            2 * T.position_size(a, 1000.0, tcfg)
        )

    def test_an_empty_bankroll_stakes_nothing(self, tcfg):
        a = T.assess(snap(169, 180), verified(60), price=0.80, cfg=tcfg, now=NOW)
        assert T.position_size(a, 0.0, tcfg) == 0.0

    def test_a_price_the_fee_has_eaten_stakes_nothing(self, tcfg):
        # At 99c the fee alone can leave nothing to win.
        a = T.assess(snap(169, 180), verified(60), price=0.995, cfg=tcfg, now=NOW)
        assert T.position_size(a, 1000.0, tcfg) == 0.0

    def test_staking_is_stricter_than_the_single_bet_path(self):
        from betedge.config import Config

        cfg = Config()
        assert cfg.tomatoes.kelly_multiplier < cfg.bankroll.kelly_multiplier
        assert cfg.tomatoes.max_position_fraction < cfg.bankroll.max_fraction
