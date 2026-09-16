"""
The read-only Kalshi client.

The load-bearing tests are the order-book inversion ones. Both sides of
Kalshi's book are resting BIDS, so the price you pay to buy YES comes
from the NO side. Reading it the other way round computes edge against
the wrong side of the spread on every market and looks plausible
throughout, which is exactly the failure a test has to catch.
"""

import pytest

from betedge import kalshi as K


def market_json(**kw):
    base = {
        "ticker": "KXRT-FILM-T60",
        "event_ticker": "KXRT-FILM",
        "title": "Resident Evil above 60%",
        "status": "active",
        "yes_bid": 55,
        "yes_ask": 58,
        "no_bid": 42,
        "no_ask": 45,
        "last_price": 57,
        "volume": 1200,
        "open_interest": 800,
        "close_time": "2026-09-26T21:00:00Z",
    }
    base.update(kw)
    return base


def book_json(yes=None, no=None):
    return {"orderbook": {"yes": yes or [], "no": no or []}}


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.calls = []

    def get(self, url, params=None, **kw):
        self.calls.append((url, dict(params or {})))
        return _Response(self.payloads.pop(0))


# --------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------


class TestPrices:
    def test_cents_become_probabilities_once(self):
        m = K.parse_market(market_json())
        assert m.yes_ask == pytest.approx(0.58)
        assert m.yes_bid == pytest.approx(0.55)

    def test_the_spread_is_in_probability(self):
        assert K.parse_market(market_json()).spread == pytest.approx(0.03)

    def test_a_missing_price_stays_missing_rather_than_becoming_zero(self):
        # A market with no bid is not a market priced at zero.
        m = K.parse_market(market_json(yes_bid=None, no_ask=None))
        assert m.yes_bid is None
        assert m.yes_ask == pytest.approx(0.58)

    def test_a_price_outside_cents_is_refused(self):
        with pytest.raises(K.KalshiError, match="not a price in cents"):
            K.parse_market(market_json(yes_ask=580))

    def test_an_object_with_no_ticker_is_not_a_market(self):
        with pytest.raises(K.KalshiError, match="not a market"):
            K.parse_market({"title": "nope"})

    def test_the_close_time_is_parsed_with_a_zone(self):
        m = K.parse_market(market_json())
        assert m.close_time.tzinfo is not None
        assert m.close_time.year == 2026

    def test_an_unparseable_close_time_is_not_fatal(self):
        assert K.parse_market(market_json(close_time="soon")).close_time is None


class TestSanityChecks:
    def test_the_two_sides_must_complement(self):
        # yes and no are two names for one thing. If they stop summing to
        # a dollar, this client is reading the wrong fields and every
        # edge it computes afterwards is fiction.
        with pytest.raises(K.KalshiError, match="complement"):
            K.parse_market(market_json(yes_ask=58, no_bid=90))

    def test_a_crossed_book_is_refused(self):
        # no_bid moved with yes_ask so this isolates the crossed book
        # rather than tripping the complement check first.
        with pytest.raises(K.KalshiError, match="not a book"):
            K.parse_market(market_json(yes_bid=70, yes_ask=60, no_bid=40))

    def test_a_penny_of_rounding_is_tolerated(self):
        # Exchanges round. One cent is not evidence of a broken schema.
        K.parse_market(market_json(yes_ask=58, no_bid=41)).check()


# --------------------------------------------------------------------------
# The inversion
# --------------------------------------------------------------------------


class TestOrderBookInversion:
    def test_buying_yes_crosses_the_no_bids(self):
        # A resting NO bid at 40c IS a YES offer at 60c.
        book = K.parse_orderbook(book_json(no=[[40, 25]]))
        assert book.yes_asks == [K.Level(0.60, 25)]

    def test_buying_no_crosses_the_yes_bids(self):
        book = K.parse_orderbook(book_json(yes=[[55, 10]]))
        assert book.no_asks == [K.Level(0.45, 10)]

    def test_the_best_yes_ask_comes_from_the_highest_no_bid(self):
        # Highest NO bid (45c) is the cheapest YES offer (55c).
        book = K.parse_orderbook(book_json(no=[[30, 100], [45, 20], [38, 50]]))
        assert book.best_ask(K.YES) == pytest.approx(0.55)

    def test_asks_come_back_cheapest_first(self):
        book = K.parse_orderbook(book_json(no=[[30, 100], [45, 20], [38, 50]]))
        prices = [level.price for level in book.yes_asks]
        assert prices == sorted(prices)

    def test_bids_come_back_best_first(self):
        book = K.parse_orderbook(book_json(yes=[[50, 5], [56, 9], [53, 7]]))
        assert [level.price for level in book.yes_bids] == [0.56, 0.53, 0.50]

    def test_the_two_sides_of_one_book_are_consistent(self):
        # Cross-check by construction: what it costs to buy YES plus what
        # it costs to buy NO must exceed a dollar, or the book is
        # arbitrageable and something is parsed wrong.
        book = K.parse_orderbook(book_json(yes=[[55, 10]], no=[[42, 25]]))
        assert book.best_ask(K.YES) + book.best_ask(K.NO) > 1.0

    def test_a_bare_book_without_the_wrapper_key_also_parses(self):
        book = K.parse_orderbook({"yes": [[55, 10]], "no": [[42, 25]]})
        assert book.best_ask(K.YES) == pytest.approx(0.58)

    def test_an_empty_side_is_empty_not_an_error(self):
        book = K.parse_orderbook(book_json(no=[]))
        assert book.yes_asks == []
        assert book.best_ask(K.YES) is None

    def test_zero_quantity_levels_are_dropped(self):
        book = K.parse_orderbook(book_json(no=[[40, 0], [38, 5]]))
        assert [level.quantity for level in book.yes_asks] == [5]

    def test_a_malformed_level_is_refused(self):
        with pytest.raises(K.KalshiError, match="price, quantity"):
            K.parse_orderbook(book_json(no=[[40]]))

    def test_something_that_is_not_a_book_is_refused(self):
        with pytest.raises(K.KalshiError, match="not an order book"):
            K.parse_orderbook({"orderbook": [1, 2, 3]})


# --------------------------------------------------------------------------
# Depth
# --------------------------------------------------------------------------


class TestDepth:
    @pytest.fixture
    def book(self):
        # NO bids at 45, 38, 30 -> YES asks at 55 (20), 62 (50), 70 (100)
        return K.parse_orderbook(
            book_json(no=[[45, 20], [38, 50], [30, 100]])
        )

    def test_a_fill_inside_the_top_level_pays_the_top_price(self, book):
        price, filled = book.cost_to_buy(K.YES, 10)
        assert filled == 10
        assert price == pytest.approx(0.55)

    def test_a_larger_order_walks_the_book(self, book):
        # 20 at 55c and 30 at 62c -> (20*55 + 30*62) / 50 = 59.2c
        price, filled = book.cost_to_buy(K.YES, 50)
        assert filled == 50
        assert price == pytest.approx(0.592)

    def test_walking_the_book_is_the_whole_point(self, book):
        # The top of the book flatters a real position by nearly five
        # cents here, which is more than any edge this tool looks for.
        top = book.best_ask(K.YES)
        average, _ = book.cost_to_buy(K.YES, 50)
        assert average - top == pytest.approx(0.042, abs=1e-3)

    def test_an_order_larger_than_the_book_reports_what_it_could_fill(
        self, book
    ):
        price, filled = book.cost_to_buy(K.YES, 5000)
        assert filled == 170            # 20 + 50 + 100
        assert 0.55 < price < 0.70

    def test_an_empty_side_fills_nothing(self):
        book = K.parse_orderbook(book_json(no=[]))
        assert book.cost_to_buy(K.YES, 10) == (0.0, 0)

    def test_total_depth_is_reported(self, book):
        assert book.depth(K.YES) == 170

    def test_asking_for_nothing_is_rejected(self, book):
        with pytest.raises(ValueError, match="positive"):
            book.cost_to_buy(K.YES, 0)


# --------------------------------------------------------------------------
# The client
# --------------------------------------------------------------------------


class TestClient:
    def test_it_fetches_one_market(self):
        session = FakeSession({"market": market_json()})
        client = K.MarketData(session=session)
        m = client.market("KXRT-FILM-T60")
        assert m.ticker == "KXRT-FILM-T60"
        assert session.calls[0][0].endswith("/markets/KXRT-FILM-T60")

    def test_a_market_returned_bare_also_parses(self):
        client = K.MarketData(session=FakeSession(market_json()))
        assert client.market("X").yes_ask == pytest.approx(0.58)

    def test_it_fetches_an_order_book(self):
        session = FakeSession(book_json(no=[[45, 20]]))
        book = K.MarketData(session=session).orderbook("T", depth=5)
        assert book.ticker == "T"
        assert book.best_ask(K.YES) == pytest.approx(0.55)
        assert session.calls[0][1]["depth"] == 5

    def test_listing_follows_the_cursor(self):
        session = FakeSession(
            {"markets": [market_json(ticker="A")], "cursor": "next"},
            {"markets": [market_json(ticker="B")], "cursor": ""},
        )
        found = K.MarketData(session=session).markets(series_ticker="KXRT")
        assert [m.ticker for m in found] == ["A", "B"]
        assert session.calls[1][1]["cursor"] == "next"

    def test_paging_stops_rather_than_walking_the_exchange(self):
        pages = [{"markets": [market_json(ticker=f"T{i}")], "cursor": "more"}
                 for i in range(50)]
        session = FakeSession(*pages)
        found = K.MarketData(session=session).markets(max_pages=3)
        assert len(found) == 3
        assert session.calls and len(session.calls) == 3

    def test_one_bad_market_does_not_lose_the_page(self):
        session = FakeSession({
            "markets": [market_json(ticker="A"), {"junk": True},
                        market_json(ticker="C")],
            "cursor": "",
        })
        found = K.MarketData(session=session).markets()
        assert [m.ticker for m in found] == ["A", "C"]

    def test_the_filters_reach_the_request(self):
        session = FakeSession({"markets": [], "cursor": ""})
        K.MarketData(session=session).markets(
            series_ticker="KXRTSCORE", status="open"
        )
        params = session.calls[0][1]
        assert params["series_ticker"] == "KXRTSCORE"
        assert params["status"] == "open"

    def test_a_non_object_response_is_refused(self):
        class Bad(FakeSession):
            def get(self, url, params=None, **kw):
                return _Response([1, 2, 3])

        with pytest.raises(K.KalshiError, match="JSON object"):
            K.MarketData(session=Bad()).market("X")

    def test_requests_are_counted(self):
        session = FakeSession({"market": market_json()},
                              book_json(no=[[45, 20]]))
        client = K.MarketData(session=session)
        client.market("X")
        client.orderbook("X")
        assert client.requests_made == 2

    def test_it_cannot_trade(self):
        # Structural, not a policy: a scanner bug cannot become an order
        # if the client has no method that places one.
        for name in ("order", "place_order", "create_order", "buy", "sell"):
            assert not hasattr(K.MarketData, name)
