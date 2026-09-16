"""
The Kalshi ladder scan, end to end, with no network.

Every stage here can fail in a way that produces confident nonsense, so
the tests are mostly about what each one REFUSES. The one that matters
most is de-vigging: a raw moneyline makes the favourite look likelier
than it is, which makes sigma come out too small, which understates every
outer rung -- exactly where this scan is looking.
"""

from datetime import datetime, timedelta, timezone

import pytest

from betedge import kalshi as K
from betedge import kalshiscan as KS
from betedge import ladder as L
from betedge.config import Config

NOW = datetime(2026, 9, 18, 0, 15, tzinfo=timezone.utc)


def odds_event(home="Kansas City Chiefs", away="Denver Broncos",
               home_price=1.43, away_price=3.05, home_point=-6.5,
               eid="kc-den", markets=("h2h", "spreads")):
    book_markets = []
    if "h2h" in markets:
        book_markets.append({"key": "h2h", "outcomes": [
            {"name": home, "price": home_price},
            {"name": away, "price": away_price},
        ]})
    if "spreads" in markets:
        book_markets.append({"key": "spreads", "outcomes": [
            {"name": home, "price": 1.91, "point": home_point},
            {"name": away, "price": 1.91, "point": -home_point},
        ]})
    return {
        "id": eid, "home_team": home, "away_team": away,
        "commence_time": NOW.isoformat(),
        "bookmakers": [{"key": "pinnacle", "markets": book_markets}],
    }


class FakeMarket:
    def __init__(self, ticker, title, subtitle="", close_time=NOW):
        self.ticker = ticker
        self.title = title
        self.subtitle = subtitle
        self.close_time = close_time


def book(no_bids, yes_bids=()):
    return K.parse_orderbook({"yes": list(yes_bids), "no": list(no_bids)})


# --------------------------------------------------------------------------
# Reading the sharp line
# --------------------------------------------------------------------------


class TestGameLines:
    def test_it_reads_a_complete_line(self):
        (line,) = KS.game_lines([odds_event()])
        assert line.favourite == "Kansas City Chiefs"
        assert line.spread == 6.5
        assert 0.5 < line.fair_win_prob < 1.0

    def test_the_moneyline_is_de_vigged(self):
        """
        The decisive one. A raw moneyline carries the book's margin, so
        the favourite looks likelier than it is -- and since sigma is
        mu / Phi^-1(p), too high a p means too small a sigma, which
        understates every outer rung. That is precisely where this scan
        is looking, so the error would be invisible AND in the direction
        that suppresses its own findings.
        """
        from betedge import pricing

        event = odds_event(home_price=1.43, away_price=3.05)
        (line,) = KS.game_lines([event])
        raw = pricing.implied(1.43)
        assert line.fair_win_prob < raw
        assert line.overround > 0

    def test_a_de_vigged_fit_gives_a_wider_game_than_a_raw_one(self):
        priors = L.PriorSet.load()
        (line,) = KS.game_lines([odds_event()])
        from betedge import pricing

        fitted = L.fit(line.spread, line.fair_win_prob,
                       "americanfootball_nfl", priors)
        naive = L.fit(line.spread, pricing.implied(1.43),
                      "americanfootball_nfl", priors)
        assert fitted.sigma > naive.sigma

    def test_the_favourite_comes_from_the_spread(self):
        # So that the spread and the moneyline can be checked against
        # each other by the fit, rather than assumed to agree.
        (line,) = KS.game_lines([odds_event(home_point=+3.5)])
        assert line.favourite == "Denver Broncos"
        assert line.spread == 3.5

    def test_a_game_without_a_spread_is_skipped(self):
        assert KS.game_lines([odds_event(markets=("h2h",))]) == []

    def test_a_game_without_a_moneyline_is_skipped(self):
        # The spread alone cannot say how wide a game is.
        assert KS.game_lines([odds_event(markets=("spreads",))]) == []

    def test_a_three_way_market_is_skipped(self):
        # A draw is a different distribution and the margin model does
        # not describe it.
        event = odds_event()
        event["bookmakers"][0]["markets"][0]["outcomes"].append(
            {"name": "Draw", "price": 3.4}
        )
        assert KS.game_lines([event]) == []

    def test_another_book_is_not_mistaken_for_the_sharp_one(self):
        event = odds_event()
        event["bookmakers"][0]["key"] = "draftkings"
        assert KS.game_lines([event]) == []

    def test_a_nonsense_price_is_skipped(self):
        assert KS.game_lines([odds_event(home_price=0.5)]) == []


# --------------------------------------------------------------------------
# Reading a rung
# --------------------------------------------------------------------------


class TestParseRung:
    def test_it_reads_a_threshold_from_the_title(self):
        market = FakeMarket("T", "Chiefs to beat the Broncos by more than 6.5")
        threshold, problem = KS.parse_rung(market, "Kansas City Chiefs")
        assert threshold == 6.5
        assert problem == ""

    def test_a_band_is_refused_rather_than_halved(self):
        # "by 7 to 13" is a different shape. Taking one of the numbers
        # would look exactly like it had worked.
        market = FakeMarket("T", "Chiefs win by 7 to 13")
        threshold, problem = KS.parse_rung(market, "Kansas City Chiefs")
        assert threshold is None
        assert "more than one number" in problem

    def test_a_market_that_is_not_about_a_margin_is_refused(self):
        market = FakeMarket("T", "Chiefs total points over 24.5")
        threshold, problem = KS.parse_rung(market, "Kansas City Chiefs")
        assert threshold is None

    def test_a_title_naming_neither_team_is_refused(self):
        market = FakeMarket("T", "Win by more than 6.5")
        threshold, problem = KS.parse_rung(market, "Kansas City Chiefs")
        assert threshold is None
        assert "which team" in problem

    def test_a_title_about_the_underdog_is_refused_not_guessed(self):
        # It might mean the underdog covering, which is the same
        # distribution read at a negative threshold -- but "might" is
        # not good enough when the sign flips the whole answer.
        market = FakeMarket("T", "Broncos to win by more than 3.5")
        threshold, _problem = KS.parse_rung(market, "Kansas City Chiefs")
        assert threshold is None

    def test_an_absurd_threshold_is_refused(self):
        market = FakeMarket("T", "Chiefs win by more than 500")
        threshold, problem = KS.parse_rung(market, "Kansas City Chiefs")
        assert threshold is None
        assert "not a margin" in problem

    def test_it_does_not_fall_back_to_the_ticker(self):
        # Ticker formats are undocumented and change without notice, so
        # a parser built on one breaks silently.
        market = FakeMarket("KXNFL-T65", "Chiefs game")
        threshold, _problem = KS.parse_rung(market, "Kansas City Chiefs")
        assert threshold is None


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------


class TestPriceRungs:
    @pytest.fixture
    def setup(self):
        cfg = Config()
        cfg.kalshi.depth_contracts = 100
        priors = L.PriorSet.load()
        (line,) = KS.game_lines([odds_event()])
        model = L.fit(line.spread, line.fair_win_prob,
                      "americanfootball_nfl", priors)
        return cfg, line, model

    def test_it_prices_a_rung_and_reports_fillable_size(self, setup):
        cfg, line, model = setup
        markets = [FakeMarket("KXNFL-1",
                              "Chiefs to beat the Broncos by more than 3.5")]
        books = {"KXNFL-1": book(no_bids=[[60, 40], [55, 200]])}
        quotes, skipped = KS.price_rungs(model, line, markets, books, cfg)
        assert len(quotes) == 1
        assert skipped == []
        assert quotes[0].contracts == 100
        assert 0 < quotes[0].fair < 1

    def test_the_book_is_walked_not_read_off_the_top(self, setup):
        # 40 at 40c then 60 at 45c is not 100 at 40c, and the difference
        # is larger than any edge this scan looks for.
        cfg, line, model = setup
        markets = [FakeMarket("T", "Chiefs to beat the Broncos by more than 3.5")]
        books = {"T": book(no_bids=[[60, 40], [55, 200]])}
        (quote,), _ = KS.price_rungs(model, line, markets, books, cfg)
        assert quote.price > 0.40
        assert quote.price == pytest.approx((40 * 0.40 + 60 * 0.45) / 100)

    def test_a_partial_fill_is_flagged_not_hidden(self, setup):
        # It changes what the number means: this is the EV of a smaller
        # position than was asked for.
        cfg, line, model = setup
        markets = [FakeMarket("T", "Chiefs to beat the Broncos by more than 3.5")]
        books = {"T": book(no_bids=[[60, 12]])}
        (quote,), _ = KS.price_rungs(model, line, markets, books, cfg)
        assert quote.contracts == 12
        assert any("only_12_of_100_fillable" in f for f in quote.flags)

    def test_an_empty_book_is_skipped_with_a_reason(self, setup):
        cfg, line, model = setup
        markets = [FakeMarket("T", "Chiefs to beat the Broncos by more than 3.5")]
        (quotes, skipped) = KS.price_rungs(model, line, markets,
                                           {"T": book(no_bids=[])}, cfg)
        assert quotes == []
        assert skipped[0][1] == "nothing offered"

    def test_a_market_with_no_book_is_skipped(self, setup):
        cfg, line, model = setup
        markets = [FakeMarket("T", "Chiefs to beat the Broncos by more than 3.5")]
        quotes, skipped = KS.price_rungs(model, line, markets, {}, cfg)
        assert quotes == []
        assert skipped[0][1] == "no order book"

    def test_the_fee_is_inside_the_ev(self, setup):
        # Paying exactly the fair price must still lose, by the fee.
        cfg, line, model = setup
        markets = [FakeMarket("T", "Chiefs to beat the Broncos by more than 3.5")]
        fair_cents = int(round(
            model.prob_margin_over(3.5) * 100
        ))
        books = {"T": book(no_bids=[[100 - fair_cents, 5000]])}
        (quote,), _ = KS.price_rungs(model, line, markets, books, cfg)
        assert quote.ev < 0
        assert abs(quote.edge) < 0.01

    def test_the_model_flags_ride_along_on_every_quote(self, setup):
        cfg, line, model = setup
        markets = [FakeMarket("T", "Chiefs to beat the Broncos by more than 3.5")]
        books = {"T": book(no_bids=[[60, 200]])}
        (quote,), _ = KS.price_rungs(model, line, markets, books, cfg)
        assert "margin_priors_unverified" in quote.flags


class TestMakerFees:
    def test_an_unlisted_series_is_assumed_to_charge_one(self):
        # The safe direction: assuming free would flatter every resting
        # quote the tool suggests.
        cfg = Config()
        assert KS._pays_maker_fee("KXNFLGAME-26SEP18", cfg.kalshi) is True

    def test_a_series_listed_as_zero_does_not(self):
        cfg = Config()
        cfg.kalshi.series_maker_multiplier = {"KXNFLGAME": 0}
        assert KS._pays_maker_fee("KXNFLGAME-26SEP18", cfg.kalshi) is False


# --------------------------------------------------------------------------
# Coherence, which needs none of the above
# --------------------------------------------------------------------------


class TestCoherenceFromMarkets:
    def test_an_incoherent_pair_is_found_without_any_model(self):
        markets = [
            FakeMarket("A", "Chiefs win by more than 3.5"),
            FakeMarket("B", "Chiefs win by more than 9.5"),
        ]
        books = {
            # easier rung asks 40c, harder rung bids 55c
            "A": book(no_bids=[[60, 100]], yes_bids=[[37, 100]]),
            "B": book(no_bids=[[42, 100]], yes_bids=[[55, 100]]),
        }
        rungs = KS.rungs_for_coherence(markets, books)
        problems = L.coherence_violations(rungs)
        assert len(problems) == 1

    def test_a_healthy_ladder_reports_nothing(self):
        markets = [
            FakeMarket("A", "Chiefs win by more than 3.5"),
            FakeMarket("B", "Chiefs win by more than 9.5"),
        ]
        books = {
            "A": book(no_bids=[[42, 100]], yes_bids=[[55, 100]]),
            "B": book(no_bids=[[70, 100]], yes_bids=[[27, 100]]),
        }
        rungs = KS.rungs_for_coherence(markets, books)
        assert L.coherence_violations(rungs) == []

    def test_a_banded_market_is_left_out(self):
        markets = [FakeMarket("A", "Chiefs win by 7 to 13")]
        books = {"A": book(no_bids=[[60, 100]], yes_bids=[[37, 100]])}
        assert KS.rungs_for_coherence(markets, books) == []
