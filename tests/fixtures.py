"""
API-shaped payloads for the multi-leg optimizer.

Built the same way conftest's builders are: dictionaries in exactly the
shape The Odds API returns, assembled locally. Nothing here touches the
network or spends a credit, and every timestamp is anchored to
`conftest.NOW` so a test never depends on what time it happens to run.
"""

from __future__ import annotations

from datetime import timedelta

from conftest import NOW, book, event_payload, outcome

from betedge.config import Config
from betedge.parlay import Leg
from betedge.rosters import RosterBook

# A player prop: (market, player, line, (pinnacle over, pinnacle under)).
# The pick'em book posts the same line at a nominal price -- which is what
# the real products look like, since their payout comes from the leg count
# rather than from a price.
KC_STACK = [
    ("player_pass_yds", "Patrick Mahomes", 249.5, (1.62, 2.42)),
    ("player_reception_yds", "Travis Kelce", 64.5, (1.66, 2.33)),
    ("player_reception_yds", "Rashee Rice", 52.5, (1.70, 2.25)),
    ("player_rush_yds", "Isiah Pacheco", 48.5, (1.68, 2.28)),
]

#: Two legs priced at a coin flip, which every product loses money on.
FLAT_STACK = [
    ("player_pass_yds", "Patrick Mahomes", 249.5, (1.95, 1.95)),
    ("player_reception_yds", "Travis Kelce", 64.5, (1.95, 1.95)),
]

KC_TEAMS = {
    "patrick mahomes": "Kansas City Chiefs",
    "travis kelce": "Kansas City Chiefs",
    "rashee rice": "Kansas City Chiefs",
    "isiah pacheco": "Kansas City Chiefs",
}


def prop_event(
    spec=None,
    event_id="kc1",
    sport="americanfootball_nfl",
    target_book="underdog",
    commence_hours=8,
    book_price=1.91,
    sharp_last_update=None,
    book_last_update=None,
    home="Kansas City Chiefs",
    away="Denver Broncos",
    include_target=True,
    target_line_shift=0.0,
):
    """
    One event with Pinnacle two-sided on every prop and a target book
    quoting the same lines.

    `target_line_shift` moves the target book off Pinnacle's number, which
    is how the "never compare different lines" rule gets tested.
    """
    spec = KC_STACK if spec is None else spec
    pinnacle: dict[str, list] = {}
    target: dict[str, list] = {}
    for market, player, line, (over, under) in spec:
        pinnacle.setdefault(market, []).extend([
            outcome("Over", over, point=line, description=player),
            outcome("Under", under, point=line, description=player),
        ])
        if include_target:
            shifted = line + target_line_shift
            target.setdefault(market, []).extend([
                outcome("Over", book_price, point=shifted, description=player),
                outcome("Under", book_price, point=shifted, description=player),
            ])

    books = [book("pinnacle", pinnacle, last_update=sharp_last_update)]
    if include_target:
        books.append(book(target_book, target, last_update=book_last_update))
    return event_payload(
        event_id=event_id,
        sport=sport,
        commence=NOW + timedelta(hours=commence_hours),
        home=home,
        away=away,
        bookmakers=books,
    )


def ladder_event(
    market="player_receptions",
    player="Travis Kelce",
    lines=((4.5, (1.55, 2.55)), (5.0, (1.72, 2.20)), (5.5, (1.95, 1.95))),
    target_line=5.0,
    event_id="kc1",
    target_book="underdog",
):
    """
    Pinnacle pricing several rungs of one prop.

    Needed for the two things a single line cannot support: interpolating
    a probability onto a line Pinnacle does not price, and measuring the
    chance a stat lands exactly on an integer line from the two half-lines
    either side of it.
    """
    pinnacle: dict[str, list] = {}
    for line, (over, under) in lines:
        pinnacle.setdefault(market, []).extend([
            outcome("Over", over, point=line, description=player),
            outcome("Under", under, point=line, description=player),
        ])
    target = {market: [
        outcome("Over", 1.91, point=target_line, description=player),
        outcome("Under", 1.91, point=target_line, description=player),
    ]}
    return event_payload(
        event_id=event_id,
        commence=NOW + timedelta(hours=8),
        bookmakers=[book("pinnacle", pinnacle), book(target_book, target)],
    )


def parlay_config(tmp_path, **overrides):
    """A config wired for fast, deterministic parlay tests."""
    cfg = Config()
    cfg.books.soft = ["draftkings"]
    cfg.sports = ["americanfootball_nfl"]
    cfg.core_sports = []
    cfg.database = str(tmp_path / "parlay.db")
    cfg.reports_dir = str(tmp_path / "reports")
    cfg.parlay.draws = 20_000
    cfg.parlay.search_draws = 4_000
    cfg.parlay.beam_width = 8
    # Tests never reach the network; the roster layer is exercised
    # explicitly with a fake opener where it is the thing under test.
    cfg.parlay.roster_auto_refresh = False
    for key, value in overrides.items():
        setattr(cfg.parlay, key, value)
    return cfg


def roster_book(mapping=None, sport="americanfootball_nfl", **kwargs) -> RosterBook:
    """A RosterBook over a plain {player: team} map, dated today."""
    return RosterBook.from_mapping(mapping or KC_TEAMS, sport=sport, **kwargs)


def make_leg(**overrides) -> Leg:
    """A Leg with sane defaults, for testing one rule at a time."""
    fields = dict(
        sport="americanfootball_nfl",
        event_id="kc1",
        commence_time=NOW + timedelta(hours=8),
        home_team="Kansas City Chiefs",
        away_team="Denver Broncos",
        market="player_pass_yds",
        selection="Patrick Mahomes",
        side="Over",
        line=249.5,
        book="underdog",
        book_price=1.91,
        book_last_update=NOW,
        fair_prob=0.60,
        push_prob=0.0,
        sharp_price_taken=1.62,
        sharp_price_other=2.42,
        sharp_overround=0.03,
        devig_spread=0.005,
        fair_prob_by_method={"shin": 0.60},
        sharp_last_update=NOW,
        sharp_line=249.5,
        market_tier="primary_prop",
        liquidity=0.75,
    )
    fields.update(overrides)
    return Leg(**fields)
