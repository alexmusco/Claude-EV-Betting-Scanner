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
