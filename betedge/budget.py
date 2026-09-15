"""
Monthly credit budgeting.

The Odds API sells credits by the month. A plan of 20,000 sounds like a
lot until one careless prop scan of a full NFL Sunday across every market
takes 2,400 of them, or a cron job stuck in a loop takes the rest. The
existing per-scan ceiling stops a single runaway call; it does nothing
about the slower failure, which is spending three weeks of quota in the
first four days and going dark for the rest of the month.

So this module answers one question: **how many credits may be spent right
now?**

    remaining -> a daily pace -> today's allowance -> what is left today

Ground truth for `remaining` is the API itself. Every response carries
`x-requests-remaining`, which is the provider's own count and cannot drift
the way a local tally does. The local ledger exists for the things the
header cannot tell you: what was spent *today* specifically, and what the
spending has looked like over the cycle.

Pacing
------
Even pacing is `remaining / days_left`. Spending exactly that every day is
the wrong target though, because opportunity is not spread evenly: an NFL
Sunday is worth more credits than a Tuesday in September, and a day with
no lineups posted is worth almost none. So the allowance is the even pace
multiplied by a burst factor (default 2.0), which lets a good day borrow
from a quiet one while still making it arithmetically impossible to burn
the month in a week.

Unspent credits roll forward automatically and need no special handling:
`remaining` is simply larger tomorrow than the even pace assumed, so the
next day's allowance rises on its own.

A reserve is held back so that closing-line capture -- which is what tells
you whether the model works at all -- never gets starved by scanning.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


def _clamp_day(year: int, month: int, day: int) -> int:
    """Handle a cycle day of 31 in a 30-day month."""
    return min(day, calendar.monthrange(year, month)[1])


def _prev_month(year: int, month: int) -> tuple[int, int]:
    return (year - 1, 12) if month == 1 else (year, month - 1)


def _next_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


def cycle_bounds(now: datetime, cycle_day: int = 1) -> tuple[datetime, datetime]:
    """
    Start and end of the billing cycle containing `now`.

    `cycle_day` is the day of the month your plan renews on. Leave it at 1
    if you do not know; the pacing still works, it just resets on a
    different day than the provider does.
    """
    cycle_day = max(1, min(28, cycle_day)) if cycle_day > 28 else max(1, cycle_day)
    anchor = _clamp_day(now.year, now.month, cycle_day)
    if now.day >= anchor:
        sy, sm = now.year, now.month
    else:
        sy, sm = _prev_month(now.year, now.month)
    start = datetime(sy, sm, _clamp_day(sy, sm, cycle_day), tzinfo=timezone.utc)
    ey, em = _next_month(sy, sm)
    end = datetime(ey, em, _clamp_day(ey, em, cycle_day), tzinfo=timezone.utc)
    return start, end


@dataclass
class BudgetStatus:
    cycle_start: datetime
    cycle_end: datetime
    monthly_credits: int
    #: Provider's own count when available, else inferred from the ledger.
    remaining: int
    remaining_is_measured: bool
    spent_this_cycle: int
    spent_today: int
    days_left: float
    reserve: int
    even_pace: float
    allowance_today: int
    left_today: int

    @property
    def spendable(self) -> int:
        """What a scan starting now is allowed to cost."""
        return max(0, self.left_today)

    @property
    def on_pace(self) -> bool:
        """
        Whether spending is inside the even-pace envelope for the cycle so
        far. False means recent days have been heavy -- not an error, but
        worth seeing before you start another scan.
        """
        elapsed = (self.cycle_end - self.cycle_start).days - self.days_left
        if elapsed <= 0:
            return True
        budget_by_now = self.monthly_credits * (
            elapsed / max(1.0, (self.cycle_end - self.cycle_start).days)
        )
        return self.spent_this_cycle <= budget_by_now * 1.15

    def summary(self) -> str:
        measured = "API" if self.remaining_is_measured else "estimated"
        lines = [
            f"Cycle       {self.cycle_start:%d %b} to {self.cycle_end:%d %b}"
            f"   ({self.days_left:.1f} days left)",
            f"Remaining   {self.remaining:,} of {self.monthly_credits:,} ({measured})",
            f"Spent       {self.spent_this_cycle:,} this cycle, "
            f"{self.spent_today:,} today",
            f"Pace        {self.even_pace:,.0f}/day even; "
            f"today's allowance {self.allowance_today:,}",
            f"Available   {self.spendable:,} credits for this run",
        ]
        if not self.on_pace:
            lines.append(
                "            running ahead of even pace -- allowance is "
                "throttled accordingly"
            )
        if self.remaining <= self.reserve:
            lines.append(
                f"            at or below the {self.reserve:,} reserve; only "
                "closing-line capture should run"
            )
        return "\n".join(lines)


def plan(
    *,
    monthly_credits: int,
    cycle_day: int,
    reserve: int,
    burst: float,
    spent_this_cycle: int,
    spent_today: int,
    api_remaining: int | None,
    now: datetime | None = None,
) -> BudgetStatus:
    """
    Work out today's spending allowance.

    Pure arithmetic on numbers the caller has already gathered, so it is
    trivially testable and has no opinion about where they came from.
    """
    now = now or datetime.now(timezone.utc)
    start, end = cycle_bounds(now, cycle_day)

    if api_remaining is not None:
        remaining, measured = int(api_remaining), True
    else:
        remaining, measured = max(0, monthly_credits - spent_this_cycle), False

    seconds_left = (end - now).total_seconds()
    days_left = max(0.25, seconds_left / 86400.0)

    usable = max(0, remaining - reserve)
    even_pace = usable / days_left
    allowance_today = int(even_pace * max(1.0, burst))
    # Never let a single day's allowance exceed what is actually there.
    allowance_today = min(allowance_today, usable)
    left_today = max(0, allowance_today - spent_today)

    return BudgetStatus(
        cycle_start=start,
        cycle_end=end,
        monthly_credits=monthly_credits,
        remaining=remaining,
        remaining_is_measured=measured,
        spent_this_cycle=spent_this_cycle,
        spent_today=spent_today,
        days_left=days_left,
        reserve=reserve,
        even_pace=even_pace,
        allowance_today=allowance_today,
        left_today=left_today,
    )


def day_bounds(now: datetime | None = None) -> tuple[datetime, datetime]:
    """UTC midnight-to-midnight around `now`, for the 'spent today' tally."""
    now = now or datetime.now(timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)
