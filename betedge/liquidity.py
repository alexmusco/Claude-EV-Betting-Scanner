"""
Liquidity scoring: how much to trust a fair price, and how much edge to
demand before betting it.

The problem this solves
-----------------------
Ranking opportunities by raw expected value sorts the board in almost
exactly the wrong order. The biggest apparent edges cluster in the
thinnest markets -- alternate lines, obscure props, markets posted days
early -- because that is where Pinnacle's own price is least certain and
where a stale quote survives longest. A scanner that ranks on EV alone
hands you a list sorted by "how likely is this number to be wrong",
which correlates with, but is not, "how much money is here".

The fix is to treat the de-vigged Pinnacle probability as an ESTIMATE
WITH ERROR rather than as truth. The error is small on a market Pinnacle
will take $50,000 on and large on one it will take $250 on. A 3% edge
against a high-limit number is worth more than a 6% edge against a
low-limit one, because the 6% is mostly estimation error.

What we can actually observe
----------------------------
The Odds API does not publish limits, which is what we would really
want. Three observable proxies stand in:

1. **Pinnacle's own margin.** This is the strongest signal available.
   Pinnacle sets margin inversely to limit as a matter of policy: around
   2% on a game side it will take five figures on, 5-7% on a player prop
   it will take a few hundred. A tight margin is Pinnacle telling you it
   is confident and the market is deep.

2. **Market type.** Game sides and totals are where the sharp money goes.
   Primary props (passing yards, strikeouts, points) are heavily bet but
   an order of magnitude thinner. Alternate lines are thinner again and
   move last.

3. **Time to start.** Limits rise as an event approaches and the market
   converges. A prop posted on Tuesday for a Sunday game is a placeholder.

A fourth adjustment is about the maths rather than the market: on
lopsided prices the de-vig methods disagree most, so a fair probability
near 0 or 1 carries more model risk regardless of how liquid the market
is. Anytime-touchdown and similar yes/no longshots live here.

How the score is used
---------------------
Two ways, both of which push toward the liquid end of the board:

* **A sliding EV bar.** `required_ev` rises as liquidity falls, so a thin
  market has to show substantially more edge to be flagged at all. This
  is the Bayesian-correct response to a noisier estimate, not timidity:
  if your fair probability could be off by two points, a two-point edge
  is not an edge.

* **Ranked ordering.** `edge_score = ev * liquidity` is what the report
  sorts on, so the top of the list is where the money and the confidence
  both are.

Neither throws information away -- the raw EV is still recorded and still
shown.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------
# Market tiers
# --------------------------------------------------------------------------

TIER_MAINLINE = "mainline"
TIER_DERIVATIVE = "derivative"
TIER_PRIMARY_PROP = "primary_prop"
TIER_SECONDARY_PROP = "secondary_prop"
TIER_ALTERNATE = "alternate"

#: Base confidence by tier, before any market-specific adjustment.
#: These are deliberately coarse -- the overround factor does the fine
#: grading, and pretending to more precision here would be false.
TIER_CONFIDENCE = {
    TIER_MAINLINE: 1.00,
    TIER_DERIVATIVE: 0.80,
    TIER_PRIMARY_PROP: 0.75,
    TIER_SECONDARY_PROP: 0.55,
    TIER_ALTERNATE: 0.40,
}

#: Game-level markets: moneyline, spread, total. Where the sharp money is.
MAINLINE_MARKETS = {
    "h2h",
    "h2h_3_way",
    "spreads",
    "totals",
}

#: Game-level but a step removed -- halves, team totals, derived markets.
DERIVATIVE_MARKETS = {
    "h2h_h1", "h2h_h2", "h2h_q1", "h2h_p1",
    "spreads_h1", "spreads_h2", "spreads_q1", "spreads_p1",
    "totals_h1", "totals_h2", "totals_q1", "totals_p1",
    "team_totals", "alternate_spreads", "alternate_totals",
    "btts", "draw_no_bet", "double_chance",
}

#: Player props that carry real volume. These are the ones the public bets
#: in size, which means Pinnacle prices them carefully and the soft books
#: still move slowly on news. That combination is the whole strategy.
PRIMARY_PROP_MARKETS = {
    # NFL
    "player_pass_yds",
    "player_pass_tds",
    "player_rush_yds",
    "player_reception_yds",
    "player_receptions",
    "player_anytime_td",
    "player_pass_attempts",
    "player_pass_completions",
    "player_rush_attempts",
    # NBA
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
    "player_points_rebounds_assists",
    # MLB
    "pitcher_strikeouts",
    "batter_total_bases",
    "batter_hits",
    "batter_home_runs",
    "batter_runs_scored",
    "batter_rbis",
    # NHL
    "player_shots_on_goal",
    "player_goal_scorer_anytime",
    "player_total_saves",
}


def tier_for(market_key: str | None) -> str:
    """Classify a market key into a liquidity tier."""
    if not market_key:
        return TIER_SECONDARY_PROP
    key = market_key.strip().lower()
    if key.endswith("_alternate") or key.startswith("alternate_"):
        # alternate_spreads / alternate_totals are game-level and deeper
        # than an alternate player prop, so they stay in DERIVATIVE.
        if key in DERIVATIVE_MARKETS:
            return TIER_DERIVATIVE
        return TIER_ALTERNATE
    if key in MAINLINE_MARKETS:
        return TIER_MAINLINE
    if key in DERIVATIVE_MARKETS:
        return TIER_DERIVATIVE
    if key in PRIMARY_PROP_MARKETS:
        return TIER_PRIMARY_PROP
    return TIER_SECONDARY_PROP


# --------------------------------------------------------------------------
# Component factors
# --------------------------------------------------------------------------

#: Per-outcome overround at or below which a market is treated as fully
#: liquid. Pinnacle runs about 1.0-1.2% per side on a major game line.
OVERROUND_FLOOR = 0.0125
#: Per-outcome overround at or above which confidence bottoms out. Around
#: 3.5-4% per side is prop territory with limits in the hundreds.
OVERROUND_CEILING = 0.040
OVERROUND_MIN_FACTOR = 0.45


def overround_factor(overround: float, n_outcomes: int = 2) -> float:
    """
    Confidence from Pinnacle's margin, normalised per outcome.

    Comparing raw overround across market shapes is misleading: a 4.5%
    total on a three-way soccer market is 1.5% per outcome and perfectly
    tight, while 4.5% on a two-way prop is 2.25% per side and middling.
    Dividing by the number of outcomes puts them on one scale.
    """
    if n_outcomes < 1:
        n_outcomes = 1
    per_outcome = max(0.0, overround) / n_outcomes
    if per_outcome <= OVERROUND_FLOOR:
        return 1.0
    if per_outcome >= OVERROUND_CEILING:
        return OVERROUND_MIN_FACTOR
    span = OVERROUND_CEILING - OVERROUND_FLOOR
    travelled = (per_outcome - OVERROUND_FLOOR) / span
    return 1.0 - travelled * (1.0 - OVERROUND_MIN_FACTOR)


def timing_factor(minutes_to_start: float | None) -> float:
    """
    Confidence from how close the event is.

    Limits climb and prices converge as an event approaches. A prop posted
    four days out is a placeholder that Pinnacle will take pocket change
    on. This is a mild factor on purpose -- some of the best soft-book lag
    is precisely in the midweek window, so we discount early prices
    without discarding them.
    """
    if minutes_to_start is None:
        return 0.85
    hours = minutes_to_start / 60.0
    if hours <= 12:
        return 1.0
    if hours <= 24:
        return 0.95
    if hours <= 48:
        return 0.88
    return 0.80


def longshot_factor(fair_prob: float) -> float:
    """
    Confidence from where the fair probability sits.

    Near the extremes the de-vig methods disagree most and a small error in
    the margin assumption becomes a large relative error in the
    probability. A fair price of 1.05 versus 9.50 puts the longshot
    anywhere from 5.9% to 9.9% depending on method -- that spread is larger
    than any edge you would bet on.
    """
    p = min(max(fair_prob, 0.0), 1.0)
    tail = min(p, 1.0 - p)
    if tail >= 0.20:
        return 1.0
    if tail <= 0.05:
        return 0.55
    # Linear between a 5% and a 20% tail.
    return 0.55 + (tail - 0.05) / 0.15 * 0.45


# --------------------------------------------------------------------------
# Composite
# --------------------------------------------------------------------------

MIN_LIQUIDITY = 0.15
MAX_LIQUIDITY = 1.0


@dataclass(frozen=True)
class Liquidity:
    score: float
    tier: str
    overround_factor: float
    timing_factor: float
    longshot_factor: float

    @property
    def label(self) -> str:
        """Coarse bucket for the console, where a decimal is just noise."""
        if self.score >= 0.80:
            return "deep"
        if self.score >= 0.60:
            return "good"
        if self.score >= 0.40:
            return "thin"
        return "v.thin"


def assess(
    market: str | None,
    overround: float,
    n_outcomes: int,
    fair_prob: float,
    minutes_to_start: float | None,
) -> Liquidity:
    """
    Combine the observable proxies into one 0-1 confidence score.

    Multiplicative rather than additive because these are independent
    reasons to doubt the fair price, and any one of them being bad should
    drag the result down rather than be averaged away by the others.
    """
    tier = tier_for(market)
    o = overround_factor(overround, n_outcomes)
    t = timing_factor(minutes_to_start)
    l = longshot_factor(fair_prob)
    raw = TIER_CONFIDENCE.get(tier, 0.55) * o * t * l
    return Liquidity(
        score=min(MAX_LIQUIDITY, max(MIN_LIQUIDITY, raw)),
        tier=tier,
        overround_factor=o,
        timing_factor=t,
        longshot_factor=l,
    )


def required_ev(base_min_ev: float, liquidity: float, penalty: float = 1.5) -> float:
    """
    The EV bar this market has to clear, raised as liquidity falls.

    With the defaults (2% base, 1.5 penalty) a deep market is flagged at
    +2.0%, a primary prop at roughly +3.0%, and a thin alternate line has
    to show about +4.5% before it is worth your attention. Set
    `model.liquidity_ev_penalty` to 0 to switch this off and go back to one
    flat bar for everything.
    """
    liquidity = min(max(liquidity, 0.0), 1.0)
    return base_min_ev * (1.0 + penalty * (1.0 - liquidity))


def edge_score(ev: float, liquidity: float) -> float:
    """
    Ranking key: expected value discounted by how much we trust it.

    A +3% edge on an NFL side (liquidity ~1.0) scores 0.030 and outranks a
    +5% edge on an alternate prop (liquidity ~0.35) at 0.018 -- which is
    the ordering you want, because the first is a bet you can get real
    money down on and the second is probably a stale number.
    """
    return ev * min(max(liquidity, 0.0), 1.0)
