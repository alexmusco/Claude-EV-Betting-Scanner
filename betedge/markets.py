"""
Sport and market registry.

Market keys are The Odds API's own keys. Player props live on the
per-event endpoint (`/events/{id}/odds`), never the bulk `/odds` one.

Credit cost, straight from the API docs:

    /sports                  free
    /events                  free
    /odds                    markets x regions
    /events/{id}/odds        markets RETURNED x regions

and "every 10 bookmakers counts as 1 region equivalent". So passing
`bookmakers=pinnacle,draftkings,underdog` and NO `regions` parameter puts
the multiplier at 1. The old R scripts passed `regions=us,eu` alongside
`bookmakers=`, which doubled the cost of every single call for no benefit.

Because event-level cost counts markets *returned*, asking for a market no
book prices is free. That makes it cheap to request a wide market list and
let the API tell you what exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Sport:
    key: str
    title: str
    # Two-way player prop markets, safe for over/under de-vigging.
    props: tuple[str, ...] = ()
    # Alternate-line versions. Softer books update these slowest, so edges
    # look biggest here -- and are most often stale rather than real.
    props_alternate: tuple[str, ...] = ()
    # Game-level markets. h2h_3_way flags sports where h2h includes a draw.
    core: tuple[str, ...] = ("h2h", "spreads", "totals")
    three_way_h2h: bool = False
    notes: str = ""


NBA_PROPS = (
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
    "player_blocks",
    "player_steals",
    "player_turnovers",
    "player_points_rebounds_assists",
    "player_points_rebounds",
    "player_points_assists",
    "player_rebounds_assists",
    "player_blocks_steals",
)

NBA_PROPS_ALT = tuple(f"{m}_alternate" for m in NBA_PROPS)

NFL_PROPS = (
    "player_pass_yds",
    "player_pass_tds",
    "player_pass_completions",
    "player_pass_attempts",
    "player_pass_interceptions",
    "player_rush_yds",
    "player_rush_attempts",
    "player_receptions",
    "player_reception_yds",
    "player_pass_rush_reception_yds",
    "player_kicking_points",
    "player_tackles_assists",
    # Yes/No market. Two outcomes, so it de-vigs like any over/under, but the
    # "Yes" side is a longshot and that is exactly where the de-vig methods
    # disagree most -- worth watching the reported spread on these.
    "player_anytime_td",
)

NFL_PROPS_ALT = tuple(
    f"{m}_alternate"
    for m in (
        "player_pass_yds",
        "player_pass_tds",
        "player_rush_yds",
        "player_receptions",
        "player_reception_yds",
    )
)

NHL_PROPS = (
    "player_points",
    "player_assists",
    "player_shots_on_goal",
    "player_blocked_shots",
    "player_total_saves",
)

MLB_PROPS = (
    "batter_hits",
    "batter_total_bases",
    "batter_rbis",
    "batter_runs_scored",
    "batter_strikeouts",
    "batter_walks",
    "pitcher_strikeouts",
    "pitcher_hits_allowed",
    "pitcher_earned_runs",
    "pitcher_outs",
)

SOCCER_PROPS = (
    "player_shots_on_target",
    "player_shots",
    "player_assists",
)


SPORTS: dict[str, Sport] = {
    s.key: s
    for s in [
        Sport(
            key="basketball_nba",
            title="NBA",
            props=NBA_PROPS,
            props_alternate=NBA_PROPS_ALT,
            notes="Deepest prop coverage at both Pinnacle and the soft books. "
            "Also the most picked-over, so edges are smaller but frequent.",
        ),
        Sport(
            key="americanfootball_nfl",
            title="NFL",
            props=NFL_PROPS,
            props_alternate=NFL_PROPS_ALT,
            notes="Weekly slate. Props post days early and soft books are slow "
            "to react to injury and weather news midweek -- that lag is the edge.",
        ),
        Sport(
            key="icehockey_nhl",
            title="NHL",
            props=NHL_PROPS,
            notes="Shots on goal and blocked shots are consistently soft. "
            "Lower public volume than NBA/NFL means slower line correction.",
        ),
        Sport(
            key="baseball_mlb",
            title="MLB",
            props=MLB_PROPS,
            notes="Huge daily volume of prop markets. Pitcher strikeouts and "
            "batter total bases are the usual hunting ground. In season Apr-Oct.",
        ),
        Sport(
            key="soccer_epl",
            title="EPL",
            props=SOCCER_PROPS,
            three_way_h2h=True,
            notes="Pinnacle is very sharp on soccer, but its PLAYER PROP coverage "
            "is thin compared to US sports. Expect few prop matches; the real "
            "soccer edge is in core markets (1X2, totals, Asian handicap).",
        ),
        Sport(
            key="soccer_uefa_champs_league",
            title="UCL",
            props=SOCCER_PROPS,
            three_way_h2h=True,
        ),
        Sport(
            key="soccer_spain_la_liga",
            title="La Liga",
            props=SOCCER_PROPS,
            three_way_h2h=True,
        ),
        Sport(
            key="soccer_germany_bundesliga",
            title="Bundesliga",
            props=SOCCER_PROPS,
            three_way_h2h=True,
        ),
        Sport(
            key="soccer_italy_serie_a",
            title="Serie A",
            props=SOCCER_PROPS,
            three_way_h2h=True,
        ),
        Sport(
            key="basketball_ncaab",
            title="NCAAB",
            props=NBA_PROPS,
            notes="Soft books price mid-major games lazily. Prop coverage is "
            "patchy but core markets are a genuine soft spot.",
        ),
        Sport(
            key="americanfootball_ncaaf",
            title="NCAAF",
            props=NFL_PROPS,
            notes="Same story as NCAAB -- the edge is in games nobody watches.",
        ),
        Sport(
            key="tennis",
            title="Tennis (all tournaments)",
            props=(),
            core=("h2h", "spreads", "totals"),
            notes="Pinnacle is the reference book for tennis and US books lag it "
            "badly. Effectively no player props -- this is a core-market sport, "
            "which is the cheap kind: one call covers every match on the board. "
            "Tennis sport keys are per-tournament and rotate weekly, so use the "
            "wildcard 'tennis_*' in core_sports rather than naming tournaments.",
        ),
        Sport(
            key="mma_mixed_martial_arts",
            title="MMA",
            props=(),
            notes="Pinnacle sharp, soft books slow to move on fight-week news. "
            "Moneyline only in practice.",
        ),
    ]
}


# Sports whose h2h has three outcomes because a draw is possible. The
# de-vig handles n outcomes, but it matters for interpreting the result.
THREE_WAY_H2H_PREFIXES = ("soccer_",)


def is_three_way(sport_key: str) -> bool:
    sport = SPORTS.get(sport_key)
    if sport is not None:
        return sport.three_way_h2h
    return sport_key.startswith(THREE_WAY_H2H_PREFIXES)


def core_markets_for(sport_key: str) -> list[str]:
    """
    Game-level markets for a sport.

    These come off the bulk /odds endpoint, which costs markets x 1 region
    for the WHOLE sport rather than per event. Three credits buys every
    moneyline, spread and total on the board -- roughly fifty times cheaper
    per opportunity than player props.
    """
    sport = SPORTS.get(sport_key)
    if sport is not None:
        return list(sport.core)
    # Unknown key (a tennis tournament, say): the three standard markets.
    return ["h2h", "spreads", "totals"]


def expand_sport_keys(patterns: Sequence[str], available: Sequence[str]) -> list[str]:
    """
    Resolve config entries against the sport keys the API currently serves.

    A trailing '*' matches by prefix, which is how you follow a sport whose
    keys rotate. Tennis publishes one key per tournament -- today's
    'tennis_atp_china_open' is gone next month -- so 'tennis_*' is the only
    maintainable way to say "all tennis". Exact keys pass through whether or
    not they are currently live, so a typo still surfaces as an error rather
    than silently matching nothing.
    """
    resolved: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            matches = sorted(k for k in available if k.startswith(prefix))
            for k in matches:
                if k not in seen:
                    seen.add(k)
                    resolved.append(k)
        elif pattern not in seen:
            seen.add(pattern)
            resolved.append(pattern)
    return resolved


def get_sport(key: str) -> Sport:
    try:
        return SPORTS[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown sport key {key!r}. Known keys: {sorted(SPORTS)}. "
            "Run `betedge sports` to list every key the API currently offers."
        ) from exc


def markets_for(key: str, include_alternate: bool = False) -> list[str]:
    sport = get_sport(key)
    out = list(sport.props)
    if include_alternate:
        out += list(sport.props_alternate)
    return out


def estimate_credits(n_events: int, n_markets: int, n_books: int = 3) -> int:
    """
    Upper bound on credits for a prop scan.

    cost = events x markets_returned x region_equivalents, and
    region_equivalents = ceil(n_books / 10), which is 1 for any realistic
    book list. This is an upper bound because markets neither book prices
    are not returned and so are not billed.
    """
    region_equivalents = max(1, -(-n_books // 10))
    return n_events * n_markets * region_equivalents
