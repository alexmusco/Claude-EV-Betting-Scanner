"""Credit pacing. The failure this prevents is spending three weeks of
quota in four days, so most of these check that the allowance reacts."""

from datetime import datetime, timezone

import pytest

from betedge import budget as B


def at(day, hour=12, month=9, year=2026):
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


def plan(**kw):
    base = dict(monthly_credits=20000, cycle_day=1, reserve=800, burst=2.0,
                spent_this_cycle=0, spent_today=0, api_remaining=None)
    base.update(kw)
    return B.plan(**base)


class TestCycleBounds:
    def test_cycle_starting_on_the_first(self):
        start, end = B.cycle_bounds(at(15), cycle_day=1)
        assert (start.month, start.day) == (9, 1)
        assert (end.month, end.day) == (10, 1)

    def test_before_the_cycle_day_belongs_to_the_previous_cycle(self):
        start, end = B.cycle_bounds(at(5), cycle_day=12)
        assert (start.month, start.day) == (8, 12)
        assert (end.month, end.day) == (9, 12)

    def test_on_the_cycle_day_a_new_cycle_has_started(self):
        start, _ = B.cycle_bounds(at(12), cycle_day=12)
        assert (start.month, start.day) == (9, 12)

    def test_year_boundary(self):
        start, end = B.cycle_bounds(datetime(2026, 1, 5, tzinfo=timezone.utc), cycle_day=20)
        assert (start.year, start.month) == (2025, 12)
        assert (end.year, end.month) == (2026, 1)

    def test_a_cycle_day_past_the_end_of_a_short_month_is_clamped(self):
        start, end = B.cycle_bounds(datetime(2026, 2, 15, tzinfo=timezone.utc), cycle_day=31)
        assert start.day <= 28 and end.day <= 31


class TestPacing:
    def test_even_pace_divides_the_remainder_by_days_left(self):
        s = plan(api_remaining=15800, now=at(16))
        # 15800 - 800 reserve = 15000 over ~15 days
        assert s.even_pace == pytest.approx(1000, rel=0.05)

    def test_burst_lets_a_good_day_spend_more_than_the_even_pace(self):
        s = plan(api_remaining=15800, burst=2.0, now=at(16))
        assert s.allowance_today == pytest.approx(s.even_pace * 2, rel=0.05)

    def test_spending_today_reduces_what_is_left_today(self):
        a = plan(api_remaining=15800, spent_today=0, now=at(16))
        b = plan(api_remaining=15800, spent_today=500, now=at(16))
        assert b.left_today == a.left_today - 500

    def test_overspending_early_throttles_later_days(self):
        """The self-correcting property that makes this work."""
        healthy = plan(api_remaining=16000, now=at(10))
        burned = plan(api_remaining=3000, now=at(10))
        assert burned.allowance_today < healthy.allowance_today / 4

    def test_unspent_credits_roll_forward_without_special_handling(self):
        """A quiet day leaves `remaining` higher, so tomorrow's pace rises."""
        spent_daily = plan(api_remaining=10000, now=at(21))
        saved_up = plan(api_remaining=14000, now=at(21))
        assert saved_up.allowance_today > spent_daily.allowance_today

    def test_the_reserve_is_never_offered_for_scanning(self):
        s = plan(api_remaining=900, reserve=800, now=at(2))
        assert s.spendable <= 100

    def test_nothing_spendable_once_below_the_reserve(self):
        s = plan(api_remaining=500, reserve=800, now=at(2))
        assert s.spendable == 0

    def test_allowance_never_exceeds_what_is_actually_left(self):
        """Late in a cycle the burst multiplier must not invent credits."""
        s = plan(api_remaining=1000, reserve=0, burst=4.0, now=at(30, hour=20))
        assert s.allowance_today <= 1000

    def test_spendable_is_never_negative(self):
        s = plan(api_remaining=1000, spent_today=99999, now=at(15))
        assert s.spendable == 0

    def test_api_count_is_preferred_over_the_local_ledger(self):
        s = plan(api_remaining=7777, spent_this_cycle=100, now=at(15))
        assert s.remaining == 7777 and s.remaining_is_measured

    def test_falls_back_to_the_ledger_when_the_api_is_silent(self):
        s = plan(api_remaining=None, spent_this_cycle=6000, now=at(15))
        assert s.remaining == 14000 and not s.remaining_is_measured

    def test_on_pace_flags_heavy_spending(self):
        assert plan(api_remaining=16000, spent_this_cycle=4000, now=at(16)).on_pace
        assert not plan(api_remaining=3000, spent_this_cycle=17000, now=at(10)).on_pace

    def test_summary_mentions_the_key_numbers(self):
        text = plan(api_remaining=15800, now=at(16)).summary()
        assert "Remaining" in text and "15,800" in text and "Available" in text
