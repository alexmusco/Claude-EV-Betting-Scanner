"""
The Kalshi ladder scan, end to end.

    Pinnacle main line  ->  margin distribution  ->  every Kalshi rung
                                                 ->  fee-adjusted EV at
                                                     fillable depth

One credit per sport buys the whole board: `h2h` and `spreads` off the
bulk endpoint, which is the cheap call. Kalshi's market data is free.

What each stage refuses to do
-----------------------------
Every step here can fail in a way that produces confident nonsense, so
each one is built to stop rather than guess:

  the join      an uncertain match is declined, because a ladder priced
                against the wrong game produces entirely valid numbers
  the fit       a spread and moneyline that disagree about who is
                favoured are refused, not fitted to a negative sigma
  the rung      a threshold that cannot be read out of the title is
                skipped, not inferred from a ticker whose format is
                undocumented
  the price     depth is walked to the size being considered, because an
                EV computed on the top of a thin book is an EV for a
                position that does not exist

The coherence check comes first
-------------------------------
Before any of the above, the rungs of one game are checked against each
other. That result needs no model, no Pinnacle line and no fit -- a
harder threshold cannot be more likely than an easier one -- so it
survives every way the rest of this can be wrong, and it is reported
even when the fit fails.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import kalshi, ladder, matching, pricing

#: Thresholds in a Kalshi title. "by more than 6.5", "6.5+", "over 6.5".
_THRESHOLD = re.compile(
    r"(?:by\s+)?(?:more\s+than|over|above|at\s+least|\+)?\s*"
    r"(-?\d{1,3}(?:\.5)?)\s*(?:\+|or\s+more|or\s+better)?",
    re.I,
)
_MARGIN_WORDS = ("margin", "win by", "beat", "by more than", "spread",
                 "cover")


@dataclass
class GameLine:
    """Pinnacle's main line for one game, de-vigged."""

    event: dict
    favourite: str
    spread: float
    fair_win_prob: float
    overround: float = 0.0

    @property
    def event_id(self) -> str:
        return self.event.get("id", "")

    @property
    def matchup(self) -> str:
        return f"{self.event.get('away_team')} @ {self.event.get('home_team')}"


@dataclass
class RungQuote:
    """One priced rung, with everything needed to judge and place it."""

    ticker: str
    title: str
    threshold: float
    fair: float
    price: float
    contracts: int
    ev: float
    edge: float
    matchup: str = ""
    sport: str = ""
    commence_time: str = ""
    maker: bool = False
    flags: list[str] = field(default_factory=list)

    @property
    def notional(self) -> float:
        return self.contracts * self.price

    def describe(self) -> str:
        return f"{self.matchup}: margin over {self.threshold:g}"


@dataclass
class GameResult:
    """One game: its fit, its priced rungs and its incoherences."""

    line: GameLine | None
    model: ladder.MarginModel | None
    quotes: list[RungQuote] = field(default_factory=list)
    incoherences: list = field(default_factory=list)
    skipped: list[tuple] = field(default_factory=list)
    matchup: str = ""


@dataclass
class ScanResult:
    games: list[GameResult] = field(default_factory=list)
    unmatched: list[tuple] = field(default_factory=list)
    markets_seen: int = 0
    events_seen: int = 0
    credits: int = 0
    requests: int = 0

    @property
    def quotes(self) -> list[RungQuote]:
        return [q for g in self.games for q in g.quotes]

    @property
    def incoherences(self) -> list:
        return [i for g in self.games for i in g.incoherences]


# ---------------------------------------------------------------------------
# Reading Pinnacle's main line
# ---------------------------------------------------------------------------


def game_lines(events, sharp_book: str = "pinnacle",
               devig_method: str = "worst_case") -> list[GameLine]:
    """
    The de-vigged main line for every game the sharp book prices.

    Both the spread and the moneyline are required. The moneyline alone
    cannot say how wide a game is and the spread alone cannot say how
    likely; it is having both that turns a guess into a fit.

    De-vigging is not optional here. A raw moneyline carries the book's
    margin, which makes the favourite look more likely than it is, which
    makes sigma come out too small -- and a too-small sigma understates
    every outer rung, which is precisely where this scan is looking.
    """
    lines = []
    for event in events:
        book = _book(event, sharp_book)
        if not book:
            continue
        spreads = _market(book, "spreads")
        h2h = _market(book, "h2h")
        if not spreads or not h2h:
            continue

        home = event.get("home_team")
        away = event.get("away_team")
        outcomes = {o.get("name"): o for o in h2h.get("outcomes") or []}
        if home not in outcomes or away not in outcomes:
            continue
        if len(outcomes) != 2:
            # A three-way market (a draw is possible) is a different
            # distribution and the margin model does not describe it.
            continue

        prices = [outcomes[home].get("price"), outcomes[away].get("price")]
        if not all(isinstance(p, (int, float)) and p > 1 for p in prices):
            continue
        fair = pricing.devig(prices, method=devig_method)
        home_prob, away_prob = fair[0], fair[1]

        spread_by_team = {
            o.get("name"): o.get("point")
            for o in spreads.get("outcomes") or []
        }
        home_point = spread_by_team.get(home)
        if home_point is None:
            continue

        # The favourite is whoever is laying points. Taken from the
        # spread rather than the moneyline so that the two can be checked
        # against each other by the fit rather than assumed to agree.
        if home_point < 0:
            favourite, magnitude, win_prob = home, -home_point, home_prob
        elif home_point > 0:
            favourite, magnitude, win_prob = away, home_point, away_prob
        else:
            favourite, magnitude, win_prob = home, 0.0, home_prob

        lines.append(GameLine(
            event=event, favourite=favourite, spread=float(magnitude),
            fair_win_prob=float(win_prob),
            overround=pricing.overround(prices),
        ))
    return lines


def _book(event, key):
    for book in event.get("bookmakers") or []:
        if book.get("key") == key:
            return book
    return None


def _market(book, key):
    for market in book.get("markets") or []:
        if market.get("key") == key:
            return market
    return None


# ---------------------------------------------------------------------------
# Reading a rung out of a Kalshi market
# ---------------------------------------------------------------------------


def parse_rung(market, favourite: str) -> tuple[float | None, str]:
    """
    The threshold a market is about, and which side of the game it is on.

    Returns (threshold, problem). The threshold is expressed from the
    FAVOURITE's perspective, so a market about the underdog covering
    comes back negative -- one distribution, read at different points,
    which is the whole reason for modelling the margin.

    A title that does not clearly carry a threshold is refused. Falling
    back to the ticker would mean parsing a format that is undocumented,
    varies by series, and changes without notice.
    """
    title = " ".join(filter(None, [
        getattr(market, "subtitle", ""), getattr(market, "title", ""),
    ]))
    lowered = matching.normalise(title)
    if not any(word in lowered for word in
               (matching.normalise(w) for w in _MARGIN_WORDS)):
        return None, "not a margin market"

    numbers = re.findall(r"-?\d{1,3}(?:\.5)?", title)
    if not numbers:
        return None, "no threshold in the title"
    if len(set(numbers)) > 1:
        # Two numbers is a band ("by 7 to 13"), which is a different
        # shape from a threshold. Picking one would look like it worked.
        return None, f"more than one number in the title ({', '.join(numbers)})"

    threshold = float(numbers[0])
    if abs(threshold) > 100:
        return None, f"threshold {threshold:g} is not a margin"

    # Whose margin? If the title names the favourite, the threshold is
    # already from their side. If it names only the underdog, flip it.
    names_favourite = matching.find_team(title, favourite)
    if not names_favourite:
        return None, "the title does not say which team it is about"
    return threshold, ""


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


def price_rungs(
    model: ladder.MarginModel,
    line: GameLine,
    markets,
    books: dict,
    cfg,
    sport: str = "",
) -> tuple[list[RungQuote], list[tuple]]:
    """
    Turn matched markets into priced rungs.

    `books` maps ticker to an OrderBook. Depth is walked to the size
    being considered rather than read off the top, because an EV computed
    on a best price that is good for twenty contracts is an EV for a
    position you cannot take.
    """
    kc = cfg.kalshi
    quotes, skipped = [], []

    for market in markets:
        threshold, problem = parse_rung(market, line.favourite)
        if threshold is None:
            skipped.append((market, problem))
            continue
        book = books.get(market.ticker)
        if book is None:
            skipped.append((market, "no order book"))
            continue

        maker = _pays_maker_fee(market.ticker, kc)
        price, fillable = book.cost_to_buy(kalshi.YES, kc.depth_contracts)
        if fillable == 0:
            skipped.append((market, "nothing offered"))
            continue

        fair = model.prob_margin_over(threshold)
        fee = kalshi.fee_for(fillable, price, maker, kc)
        cost = fillable * price + fee
        if cost <= 0:
            skipped.append((market, "no cost to buy"))
            continue
        ev = (fair * fillable - cost) / cost

        flags = list(model.flags)
        if fillable < kc.depth_contracts:
            # Not a failure, but it changes what the number means: this
            # is the EV of a smaller position than was asked for.
            flags.append(f"only_{fillable}_of_{kc.depth_contracts}_fillable")

        quotes.append(RungQuote(
            ticker=market.ticker, title=getattr(market, "title", ""),
            threshold=threshold, fair=fair, price=price,
            contracts=fillable, ev=ev, edge=fair - price,
            matchup=line.matchup, sport=sport,
            commence_time=str(line.event.get("commence_time") or ""),
            maker=maker, flags=flags,
        ))
    return quotes, skipped


def _pays_maker_fee(ticker: str, kc) -> bool:
    """
    Whether a resting order in this series pays a maker fee.

    Defaults to TRUE for anything unlisted, which is the safe direction:
    Kalshi's published schedule zeroes the maker multiplier on many
    series and sets it to one on several sports series, so assuming free
    would flatter every resting quote this tool suggests.
    """
    series = ticker.split("-")[0] if ticker else ""
    return bool(kc.series_maker_multiplier.get(series, 1))


def rungs_for_coherence(markets, books) -> list:
    """Rungs with both tradeable sides, for the model-free check."""
    out = []
    for market in markets:
        book = books.get(market.ticker)
        if book is None:
            continue
        threshold = None
        numbers = re.findall(
            r"-?\d{1,3}(?:\.5)?",
            " ".join(filter(None, [getattr(market, "subtitle", ""),
                                   getattr(market, "title", "")])),
        )
        if len(set(numbers)) == 1:
            threshold = float(numbers[0])
        if threshold is None:
            continue
        out.append(ladder.Rung(
            threshold=threshold,
            yes_ask=book.best_ask(kalshi.YES),
            yes_bid=book.yes_bids[0].price if book.yes_bids else None,
            ticker=market.ticker,
        ))
    return out
