"""
Kalshi public market data. No key, no signing, read only.

Scope
-----
Market discovery, prices and order-book depth. Nothing here can place an
order: that needs a signed, key-bearing client, and keeping the two apart
means a bug in a scanner cannot become a trade.

Prices are CENTS
----------------
Kalshi quotes whole cents, 1..99, and a contract settles at 100 or 0. So
a price is a probability in percent and the two are the same number. This
module converts once, at the boundary, and everything above it works in
probabilities -- mixing the two is how a 3% edge becomes a 300% one in a
log line.

The inversion that is easy to get backwards
-------------------------------------------
The order book has a `yes` side and a `no` side, and BOTH are lists of
RESTING BIDS -- people waiting to buy. There is no "ask" array. To buy
YES you cross with someone bidding on NO, because one contract of each
side is what gets created:

    a resting NO bid at 40c  ==  a YES offer at 60c

So the YES asks are `100 - p` over the NO bids, best ask first, carrying
the same quantities. Reading the `yes` array as your buy prices would
have you computing edge against the wrong side of the spread on every
single market, and it would look plausible the whole time.

  *** This shape is from Kalshi's published API and has NOT yet been
      confirmed against a live response from this account. `Market.check`
      and the order-book validation below are what turn a wrong
      assumption into a loud failure rather than a quiet mispricing.
      Confirm it once against a real market before trading on it. ***

Depth, not the top of the book
------------------------------
The scanner asks `cost_to_buy`, never `best_ask`. On books this thin the
best price is often good for twenty contracts and the next level is seven
cents worse, so an EV computed on the top of the book is an EV for a
position you cannot actually take.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

YES = "yes"
NO = "no"


class KalshiError(RuntimeError):
    """The exchange returned something this client cannot use."""


CERTIFICATE_HELP = """could not verify the TLS certificate.

Two quite different causes, and they need opposite responses. Find out
which one this is before doing anything:

    openssl s_client -connect <host>:443 -servername <host> </dev/null \\
        2>&1 | grep -E "^(subject|issuer)="

INTERCEPTION -- the issuer is some product or network appliance rather
than a public CA (Amazon, Let's Encrypt, DigiCert), and the certificate
often has a validity of a day or less. Something is sitting between you
and the venue and reading the traffic. Frequently a DNS filter: check
whether several hostnames on the domain resolve to one shared address,
which a real CDN-backed site would not. Updating certifi will not help
and trusting that issuer would mean deliberately letting a third party
read everything you send the exchange -- including, later, orders.
Change network, or deal with the filter at its source.

A STALE OR INCOMPLETE CHAIN -- the issuer IS a public CA. Then:

    pip install --upgrade certifi

inside the virtualenv you actually run from, and if that fails and you
installed Python from python.org, its Install Certificates.command --
though note that script targets the framework Python and ignores your
virtualenv, so it often changes nothing.

Either way: do NOT disable verification to get past this. An unverified
connection to a trading venue is worth less than no connection to one,
and it is exactly the situation where knowing who you are talking to is
the point."""


def _explain_ssl(exc: Exception, url: str) -> Exception:
    """Turn an opaque TLS failure into something actionable."""
    return KalshiError(f"{url}: {CERTIFICATE_HELP}\n\n(underlying error: {exc})")


def _is_certificate_error(exc: Exception) -> bool:
    """
    Whether a failure is a TLS trust problem.

    Matched on the message rather than the type so that this works
    whether the caller is using requests, urllib or a stub -- the point
    is to say something useful, and being wrong costs only a slightly
    off error message.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    return "certificate" in text or "sslcertverification" in text


def _cents_to_probability(value) -> float | None:
    if value in (None, ""):
        return None
    cents = float(value)
    if not 0.0 <= cents <= 100.0:
        raise KalshiError(
            f"{cents} is not a price in cents. Kalshi quotes 1..99 and a "
            "settled contract is 0 or 100; anything else means the field "
            "is not what this client assumes it is."
        )
    return cents / 100.0


@dataclass(frozen=True)
class Level:
    """One price level: a probability and how many contracts are there."""

    price: float
    quantity: int


@dataclass
class OrderBook:
    """
    Resting orders on both sides, expressed as what it costs YOU to buy.

    `yes_asks` and `no_asks` are already inverted from the raw bid arrays
    (see the module docstring) and sorted cheapest first, which is the
    order a taker actually fills in.
    """

    ticker: str
    yes_asks: list[Level] = field(default_factory=list)
    no_asks: list[Level] = field(default_factory=list)
    yes_bids: list[Level] = field(default_factory=list)
    no_bids: list[Level] = field(default_factory=list)

    def asks(self, side: str) -> list[Level]:
        return self.yes_asks if side == YES else self.no_asks

    def best_ask(self, side: str) -> float | None:
        levels = self.asks(side)
        return levels[0].price if levels else None

    def depth(self, side: str) -> int:
        return sum(level.quantity for level in self.asks(side))

    def cost_to_buy(self, side: str, contracts: int) -> tuple[float, int]:
        """
        Walk the book: what `contracts` would actually average, and how
        many of them are available.

        Returns (average price, contracts fillable). The second number is
        the one that decides whether a position exists at all -- a 20%
        edge on eight contracts is four dollars, and not a strategy.
        """
        if contracts <= 0:
            raise ValueError("contracts must be positive")
        spent = 0.0
        filled = 0
        for level in self.asks(side):
            take = min(level.quantity, contracts - filled)
            if take <= 0:
                break
            spent += take * level.price
            filled += take
            if filled >= contracts:
                break
        if filled == 0:
            return (0.0, 0)
        return (spent / filled, filled)


@dataclass(frozen=True)
class Market:
    """One Kalshi market, with prices as probabilities."""

    ticker: str
    title: str
    status: str
    yes_bid: float | None = None
    yes_ask: float | None = None
    no_bid: float | None = None
    no_ask: float | None = None
    last_price: float | None = None
    volume: int = 0
    open_interest: int = 0
    close_time: datetime | None = None
    event_ticker: str = ""
    #: The exchange's own settlement rules. Carried through so that
    #: verifying a contract is reading a paragraph in a local file rather
    #: than going back to the site -- which is the whole point of
    #: discovering markets automatically.
    rules: str = ""
    subtitle: str = ""

    @property
    def is_open(self) -> bool:
        return self.status in ("active", "open")

    @property
    def spread(self) -> float | None:
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return self.yes_ask - self.yes_bid

    def check(self) -> None:
        """
        Sanity-check the exchange's own numbers against each other.

        The YES and NO sides of one market are two names for the same
        thing, so their prices must complement to 1. If that stops
        holding, this client is reading fields that do not mean what it
        thinks -- and every edge it computes afterwards is fiction.
        """
        if self.yes_ask is not None and self.no_bid is not None:
            if abs((self.yes_ask + self.no_bid) - 1.0) > 0.02:
                raise KalshiError(
                    f"{self.ticker}: yes_ask {self.yes_ask:.2f} and no_bid "
                    f"{self.no_bid:.2f} do not complement to 1. The price "
                    "fields are not what this client assumes."
                )
        if self.yes_bid is not None and self.yes_ask is not None:
            if self.yes_bid > self.yes_ask:
                raise KalshiError(
                    f"{self.ticker}: bid {self.yes_bid:.2f} above ask "
                    f"{self.yes_ask:.2f}, which is not a book."
                )


def parse_market(raw: dict) -> Market:
    """Build a Market from the exchange's JSON, converting cents once."""
    if not isinstance(raw, dict) or not raw.get("ticker"):
        raise KalshiError(f"not a market object: {raw!r}")
    market = Market(
        ticker=str(raw["ticker"]),
        title=str(raw.get("title") or raw.get("subtitle") or ""),
        subtitle=str(raw.get("subtitle") or raw.get("yes_sub_title") or ""),
        rules=" ".join(
            str(raw.get(key) or "").strip()
            for key in ("rules_primary", "rules_secondary")
        ).strip(),
        status=str(raw.get("status") or "unknown"),
        yes_bid=_cents_to_probability(raw.get("yes_bid")),
        yes_ask=_cents_to_probability(raw.get("yes_ask")),
        no_bid=_cents_to_probability(raw.get("no_bid")),
        no_ask=_cents_to_probability(raw.get("no_ask")),
        last_price=_cents_to_probability(raw.get("last_price")),
        volume=int(raw.get("volume") or 0),
        open_interest=int(raw.get("open_interest") or 0),
        close_time=_timestamp(raw.get("close_time")),
        event_ticker=str(raw.get("event_ticker") or ""),
    )
    market.check()
    return market


def parse_orderbook(raw: dict, ticker: str = "") -> OrderBook:
    """
    Build an OrderBook from the exchange's JSON.

    Both arrays arrive as resting BIDS. The yes-side asks are therefore
    derived from the no-side bids and vice versa -- see the module
    docstring, because this is the one thing here worth getting right.
    """
    book = raw.get("orderbook") if "orderbook" in raw else raw
    if not isinstance(book, dict):
        raise KalshiError(f"not an order book: {raw!r}")

    yes_bids = _levels(book.get(YES))
    no_bids = _levels(book.get(NO))
    return OrderBook(
        ticker=ticker,
        yes_bids=yes_bids,
        no_bids=no_bids,
        # Cheapest first: the best YES ask comes from the HIGHEST NO bid.
        yes_asks=sorted(
            (Level(round(1.0 - b.price, 4), b.quantity) for b in no_bids),
            key=lambda level: level.price,
        ),
        no_asks=sorted(
            (Level(round(1.0 - b.price, 4), b.quantity) for b in yes_bids),
            key=lambda level: level.price,
        ),
    )


def _levels(raw) -> list[Level]:
    if not raw:
        return []
    levels = []
    for entry in raw:
        try:
            price, quantity = entry[0], entry[1]
        except (TypeError, IndexError, KeyError) as exc:
            raise KalshiError(
                f"order-book level {entry!r} is not [price, quantity]"
            ) from exc
        probability = _cents_to_probability(price)
        if probability is None or int(quantity) <= 0:
            continue
        levels.append(Level(probability, int(quantity)))
    # Bids: best (highest) first.
    return sorted(levels, key=lambda level: level.price, reverse=True)


def _timestamp(value) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def fee(contracts: int, price: float, maker: bool = False,
        cfg=None) -> float:
    """
    Kalshi's trading fee in dollars, rounded up to the next cent.

        fee = ceil_to_cent( multiplier * contracts * price * (1 - price) )

    The rounding is per order and it bites small ones: a single contract
    at 50c pays 2c where the formula says 1.75c -- 4% of stake rather
    than 3.5%. On books this thin, small orders are the normal case, so
    a model using the bare formula would overrate almost everything.
    """
    if contracts <= 0:
        return 0.0
    if not 0.0 <= price <= 1.0:
        raise KalshiError("price must be a probability in dollars, 0..1")
    taker = getattr(cfg, "taker_coefficient", 0.07) if cfg else 0.07
    maker_c = getattr(cfg, "maker_coefficient", 0.0175) if cfg else 0.0175
    coefficient = maker_c if maker else taker
    raw = coefficient * contracts * price * (1.0 - price)
    return math.ceil(raw * 100.0 - 1e-9) / 100.0


def fee_for(contracts: int, price: float, maker: bool, cfg) -> float:
    """`fee`, with the coefficients a config carries."""
    return fee(contracts, price, maker, cfg)


def breakeven_edge(price: float, maker: bool = False, cfg=None) -> float:
    """
    How far the true probability must sit above the price before a
    position is worth taking, ignoring the cent rounding.

    At a coin flip a taker needs 1.75 cents and a maker 0.44. That
    four-fold gap is the only structural edge on this exchange that does
    not require being right about anything, and it is why the same
    contract can be worth posting for and not worth crossing for.
    """
    taker = getattr(cfg, "taker_coefficient", 0.07) if cfg else 0.07
    maker_c = getattr(cfg, "maker_coefficient", 0.0175) if cfg else 0.0175
    coefficient = maker_c if maker else taker
    return coefficient * price * (1.0 - price)


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------


class MarketData:
    """
    Read-only access to Kalshi's public endpoints.

    Deliberately incapable of trading. `session` is injectable so the
    whole thing is testable without a network, which is the only way to
    test a client whose response shape has not yet been confirmed
    against the live exchange.
    """

    def __init__(self, session=None, base_url: str = BASE_URL,
                 timeout: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = session
        self.requests_made = 0

    def _get(self, path: str, params: dict | None = None) -> dict:
        if self._session is not None:
            getter = self._session.get
        else:
            import requests

            getter = requests.get
        self.requests_made += 1
        url = f"{self.base_url}{path}"
        try:
            response = getter(
                url,
                params=params or {},
                headers={"Accept": "application/json",
                         "User-Agent": "betedge/1.0 (read-only market data)"},
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            if _is_certificate_error(exc):
                raise _explain_ssl(exc, url) from exc
            raise
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise KalshiError(f"{path}: expected a JSON object")
        return payload

    def market(self, ticker: str) -> Market:
        payload = self._get(f"/markets/{ticker}")
        raw = payload.get("market", payload)
        return parse_market(raw)

    def orderbook(self, ticker: str, depth: int = 10) -> OrderBook:
        payload = self._get(f"/markets/{ticker}/orderbook",
                            {"depth": depth})
        return parse_orderbook(payload, ticker=ticker)

    def events(self, series_ticker: str | None = None, status: str | None = "open",
               limit: int = 200, max_pages: int = 10) -> list[dict]:
        """
        Raw event objects, for discovery.

        Returned as dicts rather than parsed: discovery is looking for
        markets whose SHAPE is not yet known, and a parser would have to
        guess at exactly the moment guessing is worst.
        """
        params: dict = {"limit": min(limit, 200), "with_nested_markets": "true"}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status
        found: list[dict] = []
        cursor = None
        for _ in range(max_pages):
            if cursor:
                params["cursor"] = cursor
            payload = self._get("/events", params)
            found.extend(payload.get("events") or [])
            cursor = payload.get("cursor")
            if not cursor:
                break
        return found

    def markets(
        self,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        status: str | None = "open",
        limit: int = 100,
        max_pages: int = 10,
    ) -> list[Market]:
        """
        List markets, following the cursor.

        `max_pages` is a stop, not a target: without one a wrong filter
        walks the whole exchange.
        """
        params: dict = {"limit": min(limit, 1000)}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if status:
            params["status"] = status

        found: list[Market] = []
        cursor = None
        for _ in range(max_pages):
            if cursor:
                params["cursor"] = cursor
            payload = self._get("/markets", params)
            for raw in payload.get("markets") or []:
                try:
                    found.append(parse_market(raw))
                except KalshiError:
                    # One malformed market must not lose the rest of the
                    # page; the ones that parsed are still usable.
                    continue
            cursor = payload.get("cursor")
            if not cursor:
                break
        return found
