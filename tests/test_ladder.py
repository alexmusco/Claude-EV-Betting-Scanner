"""
Pricing every rung of a game from Pinnacle's main line.

Two things carry the weight here. The fit must REPRODUCE the moneyline
it was fitted to -- that is a closed loop and it either holds exactly or
the algebra is wrong. And the coherence check must find contradictions
without any model at all, because that is the one result on the ladder
that carries no model risk.
"""

import math

import pytest

from betedge import ladder as L


@pytest.fixture(scope="module")
def priors():
    return L.PriorSet.load()


@pytest.fixture
def nfl(priors):
    # A 6.5-point favourite that the moneyline makes a 70% winner.
    return L.fit(6.5, 0.70, "americanfootball_nfl", priors)


# --------------------------------------------------------------------------
# Normal helpers
# --------------------------------------------------------------------------


class TestNormal:
    def test_the_cdf_is_a_half_at_zero(self):
        assert L.norm_cdf(0.0) == pytest.approx(0.5)

    @pytest.mark.parametrize("p,z", [
        (0.5, 0.0), (0.975, 1.959964), (0.025, -1.959964), (0.90, 1.281552),
    ])
    def test_the_inverse_matches_known_values(self, p, z):
        assert L.norm_ppf(p) == pytest.approx(z, abs=1e-5)

    def test_the_two_round_trip(self):
        for p in (0.01, 0.3, 0.5, 0.77, 0.99):
            assert L.norm_cdf(L.norm_ppf(p)) == pytest.approx(p, abs=1e-9)

    def test_a_certainty_is_rejected(self):
        with pytest.raises(ValueError):
            L.norm_ppf(1.0)


# --------------------------------------------------------------------------
# The fit
# --------------------------------------------------------------------------


class TestFit:
    def test_it_reproduces_the_moneyline_it_was_fitted_to(self, nfl):
        # The closed loop. Sigma is solved so that P(margin > 0) comes
        # out at the moneyline; if it does not, the algebra is wrong and
        # every rung on the ladder inherits the error.
        assert nfl.implied_win_prob == pytest.approx(0.70, abs=1e-9)

    @pytest.mark.parametrize("spread,prob", [
        (3.0, 0.58), (6.5, 0.70), (10.0, 0.78), (1.5, 0.53), (14.0, 0.85),
    ])
    def test_the_loop_closes_across_the_board(self, priors, spread, prob):
        model = L.fit(spread, prob, "americanfootball_nfl", priors)
        assert model.implied_win_prob == pytest.approx(prob, abs=1e-9)

    def test_sigma_is_solved_not_assumed(self, nfl, priors):
        # sigma = mu / z, and for a 70% winner z is about 0.5244.
        assert nfl.sigma == pytest.approx(6.5 / L.norm_ppf(0.70), abs=1e-9)
        assert nfl.sigma != priors.get("americanfootball_nfl").sigma_typical
        assert nfl.sigma_source == "fitted"

    def test_a_bigger_favourite_at_the_same_price_means_a_wider_game(
        self, priors
    ):
        narrow = L.fit(3.0, 0.70, "americanfootball_nfl", priors)
        wide = L.fit(10.0, 0.70, "americanfootball_nfl", priors)
        assert wide.sigma > narrow.sigma

    def test_without_a_moneyline_it_falls_back_and_says_so(self, priors):
        model = L.fit(6.5, None, "americanfootball_nfl", priors)
        assert model.sigma_source == "sport prior"
        assert "sigma_from_prior_not_fitted_to_this_game" in model.flags

    def test_with_neither_a_moneyline_nor_a_prior_it_refuses(self):
        # Assuming a sigma would make every rung an echo of the
        # assumption rather than a price.
        with pytest.raises(L.LadderError, match="nothing to pin"):
            L.fit(6.5, None, "quidditch", None)

    def test_a_pick_em_cannot_identify_sigma(self, priors):
        # Every sigma reproduces a 50% moneyline, so there is nothing to
        # solve and the fallback must be flagged rather than silent.
        model = L.fit(0.0, 0.50, "americanfootball_nfl", priors)
        assert "pick_em_so_sigma_is_not_identified" in model.flags
        assert model.sigma_source == "sport prior"

    def test_inputs_that_disagree_about_the_favourite_are_refused(self, priors):
        # A positive spread with a sub-50% moneyline is the spread read
        # off the wrong side. Fitting it would produce a negative sigma
        # and price the whole ladder backwards.
        with pytest.raises(L.LadderError, match="disagree about who is favoured"):
            L.fit(6.5, 0.35, "americanfootball_nfl", priors)

    def test_an_impossible_probability_is_refused(self, priors):
        with pytest.raises(L.LadderError, match="strictly between"):
            L.fit(6.5, 1.0, "americanfootball_nfl", priors)

    def test_an_implausible_sigma_is_flagged_not_rejected(self, priors):
        # A 14-point favourite at 52% implies an absurdly wide game. The
        # inputs disagree; the fit still returns, with the disagreement
        # attached.
        model = L.fit(14.0, 0.52, "americanfootball_nfl", priors)
        assert any("outside_americanfootball_nfl_range" in f
                   for f in model.flags)

    def test_an_ordinary_game_is_not_flagged(self, nfl):
        assert not any("outside" in f for f in nfl.flags)

    def test_unverified_priors_ride_along_on_every_model(self, nfl):
        assert "margin_priors_unverified" in nfl.flags


# --------------------------------------------------------------------------
# Pricing the rungs
# --------------------------------------------------------------------------


class TestRungPricing:
    def test_the_ladder_falls_as_the_threshold_rises(self, nfl):
        previous = 1.0
        for threshold in [x * 0.5 for x in range(-20, 61)]:
            p = nfl.prob_margin_over(threshold)
            assert p <= previous + 1e-12, threshold
            previous = p

    def test_every_rung_is_a_probability(self, nfl):
        for threshold in range(-40, 60):
            assert 0.0 <= nfl.prob_margin_over(threshold) <= 1.0

    def test_the_spread_itself_is_near_a_coin_flip(self, nfl):
        # By construction: the spread is the middle of the distribution.
        assert nfl.prob_margin_over(6.5) == pytest.approx(0.5, abs=0.02)

    def test_a_negative_threshold_asks_about_the_underdog(self, nfl):
        # The same distribution read further left, which is the point of
        # modelling the margin rather than each rung separately.
        assert nfl.prob_margin_over(-7.0) > nfl.prob_margin_over(0.0)

    def test_a_band_is_the_difference_of_two_rungs(self, nfl):
        band = nfl.prob_margin_between(3.5, 10.5)
        assert band == pytest.approx(
            nfl.prob_margin_over(3.5) - nfl.prob_margin_over(10.5)
        )

    def test_a_band_is_never_negative(self, nfl):
        assert nfl.prob_margin_between(10.5, 3.5) == 0.0


class TestKeyNumbers:
    def exactly(self, model, margin):
        """P(margin == m), which is the pair of rungs straddling it."""
        return (model.prob_margin_over(margin - 0.5)
                - model.prob_margin_over(margin + 0.5))

    def test_the_lump_sits_on_the_margin_not_on_a_threshold(self, priors):
        # Three is common; two and four are not. That is a fact about
        # outcomes, and putting it on a THRESHOLD instead is what made
        # the first version of this incoherent.
        model = L.fit(3.0, 0.58, "americanfootball_nfl", priors)
        assert self.exactly(model, 3) > self.exactly(model, 2)
        assert self.exactly(model, 3) > self.exactly(model, 4)

    def test_seven_is_lumpy_too_and_less_so_than_three(self, priors):
        model = L.fit(3.0, 0.58, "americanfootball_nfl", priors)
        assert self.exactly(model, 7) > self.exactly(model, 8)
        assert self.exactly(model, 7) > self.exactly(model, 6)

    def test_a_key_number_works_for_the_underdog_too(self, priors):
        # A three-point game is a three-point game whoever won it.
        model = L.fit(3.0, 0.58, "americanfootball_nfl", priors)
        assert self.exactly(model, -3) > self.exactly(model, -2)

    def test_a_sport_without_key_numbers_is_smooth(self, priors):
        model = L.fit(6.0, 0.68, "basketball_nba", priors)
        peak = model.mu
        near = [self.exactly(model, int(round(peak)) + d) for d in range(0, 5)]
        assert near == sorted(near, reverse=True)

    def test_the_ladder_stays_monotone_with_key_numbers_on(self, priors):
        # The bug that caused this rewrite: taking mass off integer
        # rungs while leaving half-point rungs alone made P(over 3) come
        # out BELOW P(over 3.5), which for a whole-number margin is the
        # same event priced two different ways.
        model = L.fit(3.0, 0.58, "americanfootball_nfl", priors)
        previous = 1.0
        for threshold in [x * 0.5 for x in range(-30, 41)]:
            p = model.prob_margin_over(threshold)
            assert p <= previous + 1e-12, threshold
            previous = p

    def test_over_three_and_over_three_and_a_half_are_one_event(self, priors):
        # Margins are whole numbers, so there is nothing between them.
        model = L.fit(3.0, 0.58, "americanfootball_nfl", priors)
        assert model.prob_margin_over(3.0) == pytest.approx(
            model.prob_margin_over(3.5)
        )

    def test_the_mass_function_is_a_distribution(self, priors):
        for sport in ("americanfootball_nfl", "icehockey_nhl"):
            model = L.fit(2.0, 0.56, sport, priors)
            assert sum(model.pmf().values()) == pytest.approx(1.0, abs=1e-9)
            assert all(w >= 0 for w in model.pmf().values())

    def test_every_rung_stays_a_probability(self, priors):
        model = L.fit(2.0, 0.55, "icehockey_nhl", priors)
        for threshold in range(-6, 7):
            assert 0.0 <= model.prob_margin_over(threshold) <= 1.0


# --------------------------------------------------------------------------
# Coherence: no model involved
# --------------------------------------------------------------------------


def rung(threshold, ask=None, bid=None, ticker=""):
    return L.Rung(threshold=threshold, yes_ask=ask, yes_bid=bid,
                  ticker=ticker)


class TestCoherence:
    def test_a_normal_ladder_has_no_violations(self):
        rungs = [rung(3.5, ask=0.60, bid=0.57),
                 rung(6.5, ask=0.45, bid=0.42),
                 rung(9.5, ask=0.30, bid=0.27)]
        assert L.coherence_violations(rungs) == []

    def test_an_easier_rung_cheaper_than_a_harder_one_is_caught(self):
        # Buy "over 3.5" at 40c and sell "over 9.5" at 55c. Winning by
        # ten implies winning by four, so the losing state cannot occur
        # and the pair returns at least 1. No view about the game needed.
        rungs = [rung(3.5, ask=0.40, bid=0.37),
                 rung(9.5, ask=0.58, bid=0.55)]
        problems = L.coherence_violations(rungs)
        assert len(problems) == 1
        assert problems[0].kind == "monotonicity"
        assert "cannot be more likely" in problems[0].detail
        assert "15c" in problems[0].detail

    def test_the_ordinary_shape_of_a_ladder_is_not_a_violation(self):
        # The easier rung trading ABOVE the harder one is exactly what a
        # working ladder looks like. An earlier version of this check had
        # the inequality backwards and would have flagged every healthy
        # ladder on the exchange while missing every real one.
        rungs = [rung(3.5, ask=0.58, bid=0.55),
                 rung(9.5, ask=0.50, bid=0.47)]
        assert L.coherence_violations(rungs) == []

    def test_it_compares_the_tradeable_sides_not_the_mids(self):
        # Mids crossing by a hair is a rounding artefact; the tradeable
        # sides crossing is money. Here the mids are within a cent of
        # each other and there is nothing to take.
        rungs = [rung(3.5, ask=0.52, bid=0.48),
                 rung(6.5, ask=0.53, bid=0.49)]
        assert L.coherence_violations(rungs) == []

    def test_every_offending_pair_is_reported_not_just_neighbours(self):
        # The violation may be between rungs several steps apart, so
        # checking only adjacent ones would miss it.
        rungs = [rung(3.5, ask=0.30, bid=0.28),
                 rung(6.5, ask=0.40, bid=0.38),
                 rung(9.5, ask=0.50, bid=0.48)]
        problems = L.coherence_violations(rungs)
        assert len(problems) == 3

    def test_a_rung_missing_a_price_is_skipped_not_guessed(self):
        rungs = [rung(3.5, ask=None, bid=0.37), rung(9.5, ask=0.58, bid=None)]
        assert L.coherence_violations(rungs) == []

    def test_order_of_the_input_does_not_matter(self):
        rungs = [rung(9.5, ask=0.58, bid=0.55), rung(3.5, ask=0.40, bid=0.37)]
        assert len(L.coherence_violations(rungs)) == 1

    def test_one_rung_cannot_contradict_itself(self):
        assert L.coherence_violations([rung(3.5, ask=0.6, bid=0.5)]) == []


class TestPriceLadder:
    def test_it_prices_every_rung_and_reports_the_edge(self, nfl):
        rungs = [rung(3.5, ask=0.60), rung(9.5, ask=0.30)]
        priced = L.price_ladder(nfl, rungs)
        assert len(priced) == 2
        for r, fair, edge in priced:
            assert fair is not None
            assert edge == pytest.approx(fair - r.yes_ask)

    def test_a_rung_with_no_ask_gets_a_fair_price_but_no_edge(self, nfl):
        (_r, fair, edge), = L.price_ladder(nfl, [rung(3.5)])
        assert fair is not None and edge is None

    def test_a_total_is_not_priced_off_the_margin_model(self, nfl):
        # Totals have their own mean and their own sigma. Pricing one
        # here would be silently wrong rather than merely unsupported.
        total = L.Rung(threshold=44.5, yes_ask=0.5, kind=L.TOTAL)
        (_r, fair, edge), = L.price_ladder(nfl, [total])
        assert fair is None and edge is None


class TestShippedPriors:
    def test_every_sport_ships_unverified(self, priors):
        # They are recollection, not a fit, and the flag rides along on
        # every model built from them.
        assert set(priors.unverified) == set(priors.sports)

    def test_the_bands_are_ordered_and_contain_the_typical(self, priors):
        for key, prior in priors.sports.items():
            assert prior.sigma_low < prior.sigma_typical < prior.sigma_high, key

    def test_the_bands_are_wide_enough_not_to_cry_wolf(self, priors):
        # A band that flags ordinary games teaches you to ignore it.
        for key, prior in priors.sports.items():
            span = prior.sigma_high - prior.sigma_low
            assert span > 0.5 * prior.sigma_typical, key

    def test_key_number_mass_is_a_sane_fraction(self, priors):
        for key, prior in priors.sports.items():
            total = sum(prior.key_numbers.values())
            assert total < 0.5, f"{key} claims {total:.0%} on key numbers"

    def test_the_four_sports_this_tool_bets_are_present(self, priors):
        for key in ("americanfootball_nfl", "basketball_nba",
                    "baseball_mlb", "icehockey_nhl"):
            assert priors.get(key) is not None
