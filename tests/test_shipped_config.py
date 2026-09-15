"""
Guards on the config.yaml that actually ships.

Everything else in the suite tests the code with synthetic configs. These
check the file the scanner really loads, because a plausible-looking typo
there is silent: nothing raises, the scan just quietly does the wrong
thing. The billing cycle is the sharpest example -- the wrong cycle_day
paces spending against the wrong number of days AND resets on a day the
provider does not refill, so the quota runs out before the month does.
"""

from pathlib import Path

import pytest

from betedge.config import Config

CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


@pytest.fixture(scope="module")
def cfg():
    return Config.load(CONFIG)


def test_the_shipped_config_loads(cfg):
    assert cfg is not None


class TestBooks:
    def test_draftkings_is_the_only_soft_book(self, cfg):
        """Oregon permits no other, and every extra book is parsing work
        for a bet that cannot be placed."""
        assert cfg.books.soft == ["draftkings"]

    def test_the_sharp_book_is_not_also_a_soft_book(self, cfg):
        assert cfg.books.sharp not in cfg.books.soft

    def test_the_sharp_book_is_included_in_requests(self, cfg):
        assert cfg.books.sharp in cfg.books.all


class TestBudget:
    def test_the_cycle_day_matches_the_subscription(self, cfg):
        """Subscription started 13 Sep; the cycle runs 13th to 13th."""
        assert cfg.budget.cycle_day == 13

    def test_the_plan_size_is_right(self, cfg):
        assert cfg.budget.monthly_credits == 20000

    def test_a_reserve_is_held_for_closing_lines(self, cfg):
        assert 0 < cfg.budget.reserve < cfg.budget.monthly_credits / 4

    def test_the_burst_is_sane(self, cfg):
        """At or below 1 the pacing cannot absorb a heavy slate; far above
        4 it stops being a budget."""
        assert 1.0 <= cfg.budget.daily_burst <= 4.0

    def test_budgeting_is_switched_on(self, cfg):
        assert cfg.budget.enabled


class TestExclusions:
    @pytest.mark.parametrize("key", [
        "americanfootball_ncaaf", "basketball_ncaab",
        "basketball_wncaab", "baseball_ncaa",
    ])
    def test_college_sports_are_blocked(self, cfg, key):
        assert cfg.is_excluded(key)

    @pytest.mark.parametrize("key", [
        "americanfootball_nfl", "baseball_mlb", "basketball_nba",
        "icehockey_nhl", "mma_mixed_martial_arts",
        "tennis_wta_guadalajara_open",
    ])
    def test_bettable_sports_are_not_blocked(self, cfg, key):
        assert not cfg.is_excluded(key)

    def test_no_configured_sport_is_also_excluded(self, cfg):
        """A sport listed for scanning AND blocked is a contradiction that
        would silently scan nothing."""
        for sport in cfg.sports + cfg.core_sports:
            if "*" not in sport:
                assert not cfg.is_excluded(sport), f"{sport} is both listed and blocked"


class TestModel:
    def test_the_devig_method_is_valid(self, cfg):
        from betedge.pricing import DEVIG_METHODS

        assert cfg.model.devig_method in set(DEVIG_METHODS) | {"worst_case"}

    def test_the_ev_bar_is_plausible(self, cfg):
        assert 0.0 < cfg.model.min_ev < cfg.model.max_plausible_ev

    def test_the_bar_is_not_so_low_it_is_inside_the_estimate_error(self, cfg):
        """
        Below about 1%, the edge being claimed is smaller than the error on
        a fair probability de-vigged from a 7%-margin prop market. That is
        not a bet, it is a rounding difference.
        """
        assert cfg.model.min_ev >= 0.01

    def test_the_overround_band_is_ordered(self, cfg):
        assert 0 < cfg.model.min_overround < cfg.model.max_overround

    def test_the_liquidity_penalty_is_sane(self, cfg):
        assert 0.0 <= cfg.model.liquidity_ev_penalty <= 5.0


class TestBankroll:
    def test_stake_caps_are_ordered(self, cfg):
        assert cfg.bankroll.max_fraction <= cfg.bankroll.max_total_exposure_fraction

    def test_kelly_is_fractional(self, cfg):
        """Full Kelly assumes p is known exactly. It is not."""
        assert 0 < cfg.bankroll.kelly_multiplier <= 0.5

    def test_no_single_bet_can_be_a_large_share_of_bankroll(self, cfg):
        assert cfg.bankroll.max_fraction <= 0.05


class TestPropCost:
    def test_the_prop_market_list_is_wide_enough_to_produce_candidates(self, cfg):
        """
        Two markets on a 15-game slate yields a handful of quotes a day,
        which never accumulates the sample closing-line value needs.
        Widening beats re-scanning: DraftKings is not repricing hourly.
        """
        assert len(cfg.prop_markets.get("baseball_mlb", [])) >= 4

    def test_mlb_prop_markets_are_all_primary_tier(self, cfg):
        """Secondary-tier markets carry a materially higher bar, so paying
        per-event for them buys candidates that mostly cannot clear."""
        from betedge.liquidity import TIER_PRIMARY_PROP, tier_for

        for market in cfg.prop_markets.get("baseball_mlb", []):
            assert tier_for(market) == TIER_PRIMARY_PROP, market

    def test_a_full_prop_run_fits_the_daily_allowance(self, cfg):
        """15 events x markets, plus an hourly core sweep, against the pace
        a 20,000-credit month allows."""
        markets = len(cfg.prop_markets.get("baseball_mlb", []))
        per_prop_run = 15 * markets
        daily_pace = (cfg.budget.monthly_credits - cfg.budget.reserve) / 30
        core_all_day = 24 * 10
        assert per_prop_run * 5 + core_all_day <= daily_pace

    def test_every_prop_sport_has_a_narrowed_market_list(self, cfg):
        """Cost is markets x events, so an unnarrowed sport is the single
        easiest way to blow the month."""
        for sport in cfg.sports:
            assert cfg.prop_markets.get(sport), f"{sport} has no prop_markets override"

    def test_mlb_props_wait_for_lineups(self, cfg):
        """Batter props void if the player does not start, so anything
        further out than a few hours is a placeholder."""
        assert cfg.prop_window_hours("baseball_mlb") <= 12

    def test_alternate_lines_are_off(self, cfg):
        assert not cfg.model.include_alternate_lines


class TestParlaySection:
    """
    The multi-leg settings that ship.

    Same reasoning as the rest of this file: a wrong number here does not
    raise, it just quietly stakes too much or lets through a ticket that
    should have been refused. The staking ones matter most, because a
    parlay breaks Kelly's assumptions harder than a single bet does.
    """

    def test_it_loads(self, cfg):
        assert cfg.parlay.products

    def test_every_configured_product_exists_in_the_payout_table(self, cfg):
        from betedge.parlay import PayoutTable

        table = PayoutTable.load(cfg.parlay.payouts_path)
        for key in cfg.parlay.products:
            assert key in table.products, key

    def test_parlay_staking_is_stricter_than_single_bet_staking(self, cfg):
        assert cfg.parlay.kelly_multiplier < cfg.bankroll.kelly_multiplier
        assert cfg.parlay.max_ticket_fraction < cfg.bankroll.max_fraction

    def test_one_game_cannot_take_the_whole_board_s_exposure(self, cfg):
        assert 0 < cfg.parlay.max_game_exposure_fraction
        assert (cfg.parlay.max_game_exposure_fraction
                <= cfg.bankroll.max_total_exposure_fraction)

    def test_a_single_ticket_cannot_exceed_one_game_s_cap(self, cfg):
        assert cfg.parlay.max_ticket_fraction <= cfg.parlay.max_game_exposure_fraction

    def test_line_interpolation_is_off(self, cfg):
        """Comparing against a different line is not a measurement."""
        assert cfg.parlay.allow_line_interpolation is False

    def test_the_plausibility_ceiling_is_above_the_bar_it_guards(self, cfg):
        assert cfg.parlay.min_ev < cfg.parlay.max_plausible_ev

    def test_enough_draws_for_the_error_to_be_small_next_to_the_edge(self, cfg):
        # At 200k draws the standard error on a 20x ticket's EV is under a
        # point; at 2k it would be larger than the edges being measured.
        assert cfg.parlay.draws >= 50_000
        assert cfg.parlay.search_draws >= 1_000

    def test_the_seed_is_fixed_so_a_rerun_does_not_reshuffle(self, cfg):
        assert isinstance(cfg.parlay.seed, int)

    def test_the_leg_count_range_is_sane(self, cfg):
        assert 2 <= cfg.parlay.min_legs <= cfg.parlay.max_legs <= 8

    def test_the_correlation_sample_bar_is_meaningful(self, cfg):
        """Ten joint observations is not an estimate of anything."""
        assert cfg.parlay.min_correlation_sample >= 50

    def test_legs_are_grouped_where_correlation_can_exist(self, cfg):
        assert cfg.parlay.grouping in ("same_game", "same_slate")

    def test_the_pickem_books_are_ones_the_odds_api_carries(self, cfg):
        # Nothing here is scraped; if a book is not on the API it has no
        # business in this list.
        assert set(cfg.parlay.pickem_books) <= {"underdog", "prizepicks"}


class TestShippedDataFiles:
    def test_the_payout_table_ships_unverified(self):
        """
        The multipliers vary by state and change without notice, so
        shipping them marked verified would be a lie the report repeats on
        every ticket.
        """
        from betedge.parlay import PayoutTable

        table = PayoutTable.load()
        assert table.unverified
        assert table.last_verified_by_user is None

    def test_no_shipped_structure_is_beatable_by_a_coin_flip(self):
        """
        A 50% picker must lose on every ladder. If one of these comes out
        positive, the vector is mistyped -- which is exactly how a payout
        table silently manufactures edge.
        """
        from betedge.parlay import PayoutTable

        for key, product in PayoutTable.load().products.items():
            for legs in product.leg_counts:
                assert product.independent_ev(legs, 0.5) < 0, f"{key} {legs}-pick"

    def test_every_shipped_structure_needs_a_plausible_hit_rate(self):
        """
        Break-evens should land in the low-to-mid fifties. Far outside that
        and the numbers do not describe a real product.
        """
        from betedge.parlay import PayoutTable

        for key, product in PayoutTable.load().products.items():
            for legs in product.leg_counts:
                bar = product.breakeven_leg_prob(legs)
                assert 0.50 < bar < 0.65, f"{key} {legs}-pick needs {bar:.1%}"

    def test_the_correlation_priors_load_and_are_documented(self):
        from betedge.correlation import PriorSet

        priors = PriorSet.load()
        assert priors.by_sport
        for sport, entries in priors.by_sport.items():
            assert entries, sport
            for e in entries:
                assert e.why, f"{sport} {e.markets} has no stated reasoning"

    def test_the_default_correlations_are_small(self):
        """
        An unknown relationship must not be able to manufacture an edge on
        its own.
        """
        from betedge.correlation import PriorSet

        for relation, rho in PriorSet.load().defaults.items():
            assert abs(rho) <= 0.25, relation


class TestRosterSection:
    def test_the_staleness_guard_is_on_and_sane(self, cfg):
        """
        A stale roster is worse than none, so this has to be finite and
        short enough to matter. Disabling it by setting a huge number would
        turn every old mapping into a confident wrong sign.
        """
        assert 1 <= cfg.parlay.roster_max_age_days <= 120

    def test_the_feed_is_refreshed_more_often_than_entries_expire(self, cfg):
        assert cfg.parlay.roster_refresh_days < cfg.parlay.roster_max_age_days

    def test_a_roster_path_if_set_is_not_the_shipped_data_directory(self, cfg):
        # The override is the user's own file; pointing it inside the
        # package would have it wiped by the next install.
        if cfg.parlay.rosters_path:
            assert "betedge/data" not in cfg.parlay.rosters_path.replace("\\", "/")

    def test_every_league_with_a_feed_is_one_we_verified(self):
        from betedge import rosters as R

        assert set(R.PROVIDERS) == {"americanfootball_nfl"}
        assert R.provider_name("baseball_mlb") is None

    def test_the_one_per_team_markets_really_are_one_per_team(self):
        """
        The structural inference is only sound for markets where a team
        fields exactly one player. A receiving market here would pair two
        team mates as opponents and invert the sign.
        """
        from betedge import rosters as R

        for market in R.ONE_PER_TEAM_MARKETS:
            assert (
                market.startswith("pitcher_")
                or market.startswith("player_pass_")
                or market == "player_total_saves"
            ), market
