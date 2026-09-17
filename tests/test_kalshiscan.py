"""
The Kalshi ladder scan, end to end, with no network.

Every stage here can fail in a way that produces confident nonsense, so
the tests are mostly about what each one REFUSES. The one that matters
most is de-vigging: a raw moneyline makes the favourite look likelier
than it is, which makes sigma come out too small, which understates every
outer rung -- exactly where this scan is looking.
"""

from collections import Counter
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
        """
        Exercised on a sport with no MEASURED sigma, because the NFL no
        longer fits sigma at all -- measuring it showed the fit was
        biased low. The reasoning still holds wherever the fit is used:
        the vig inflates the favourite, which shrinks sigma, which
        understates the outer rungs.
        """
        from betedge import pricing

        priors = L.PriorSet.load()
        (line,) = KS.game_lines([odds_event()])
        fitted = L.fit(line.spread, line.fair_win_prob,
                       "basketball_nba", priors)
        naive = L.fit(line.spread, pricing.implied(1.43),
                      "basketball_nba", priors)
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
        threshold, _fav, problem = KS.parse_rung(market, "Kansas City Chiefs")
        assert threshold == 6.5
        assert problem == ""

    def test_a_band_is_refused_rather_than_halved(self):
        # "by 7 to 13" is a different shape. Taking one of the numbers
        # would look exactly like it had worked.
        market = FakeMarket("T", "Chiefs win by 7 to 13")
        threshold, _fav, problem = KS.parse_rung(market, "Kansas City Chiefs")
        assert threshold is None
        assert "more than one number" in problem

    def test_a_market_that_is_not_about_a_margin_is_refused(self):
        market = FakeMarket("T", "Chiefs total points over 24.5")
        threshold, _fav, problem = KS.parse_rung(market, "Kansas City Chiefs")
        assert threshold is None

    def test_a_title_naming_neither_team_is_refused(self):
        market = FakeMarket("T", "Win by more than 6.5")
        threshold, _fav, problem = KS.parse_rung(market, "Kansas City Chiefs")
        assert threshold is None
        assert "which team" in problem

    def test_a_title_about_the_underdog_is_refused_not_guessed(self):
        # It might mean the underdog covering, which is the same
        # distribution read at a negative threshold -- but "might" is
        # not good enough when the sign flips the whole answer.
        market = FakeMarket("T", "Broncos to win by more than 3.5")
        threshold, _fav, _problem = KS.parse_rung(market, "Kansas City Chiefs")
        assert threshold is None

    def test_an_absurd_threshold_is_refused(self):
        market = FakeMarket("T", "Chiefs win by more than 500")
        threshold, _fav, problem = KS.parse_rung(market, "Kansas City Chiefs")
        assert threshold is None
        assert "not a margin" in problem

    def test_it_does_not_fall_back_to_the_ticker(self):
        # Ticker formats are undocumented and change without notice, so
        # a parser built on one breaks silently.
        market = FakeMarket("KXNFL-T65", "Chiefs game")
        threshold, _fav, _problem = KS.parse_rung(market, "Kansas City Chiefs")
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

    def test_an_empty_book_says_it_may_be_a_parser_fault(self, setup):
        # A book empty on BOTH sides across a liquid slate is far more
        # likely to be this client reading the wrong fields than 32
        # NFL markets nobody wants, so the message has to point there.
        cfg, line, model = setup
        markets = [FakeMarket("T", "Chiefs to beat the Broncos by more than 3.5")]
        (quotes, skipped) = KS.price_rungs(model, line, markets,
                                           {"T": book(no_bids=[])}, cfg)
        assert quotes == []
        assert "EMPTY ON BOTH SIDES" in skipped[0][1]
        assert "kalshi raw" in skipped[0][1]

    def test_a_one_sided_book_is_not_called_empty(self, setup):
        # YES asks come from NO bids, so a book full of YES bids offers a
        # YES buyer nothing while being a perfectly real, thin market.
        # Reporting that as "empty" would send the reader hunting a bug
        # that is not there.
        cfg, line, model = setup
        markets = [FakeMarket("T", "Chiefs to beat the Broncos by more than 3.5")]
        (quotes, skipped) = KS.price_rungs(
            model, line, markets,
            {"T": book(no_bids=[], yes_bids=[[40, 250]])}, cfg,
        )
        assert quotes == []
        assert "no one is offering yes" in skipped[0][1]
        assert "BOTH SIDES" not in skipped[0][1]

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
        # Whatever the model has to say about itself must reach the
        # quote, or a caveat dies between the fit and the decision.
        cfg, line, _model = setup
        model = L.fit(line.spread, 0.95, "americanfootball_nfl",
                      L.PriorSet.load())
        assert model.flags
        markets = [FakeMarket("T", "Chiefs to beat the Broncos by more than 3.5")]
        books = {"T": book(no_bids=[[60, 200]])}
        (quote,), _ = KS.price_rungs(model, line, markets, books, cfg)
        for flag in model.flags:
            assert flag in quote.flags


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


class TestCollectGameMarkets:
    """
    The targeting bug, and why it was worth a module change.

    The first version paged Kalshi's flat market list and took the first
    thousand. Kalshi has tens of thousands of markets and only a handful
    are today's games, so it came back with a thousand movie contracts
    and reported "0 games priced" as though the exchange had nothing to
    offer.
    """

    class Client:
        def __init__(self, events):
            self._events = events
            self.requests_made = 0

        def events(self, **kw):
            return self._events

        def markets(self, **kw):  # pragma: no cover - must not be used
            raise AssertionError("the flat market list is the wrong pool")

    def event(self, title, tickers):
        return {"title": title, "markets": [
            {"ticker": t, "subtitle": "Above 6.5", "status": "active"}
            for t in tickers
        ]}

    @pytest.fixture
    def lines(self):
        return KS.game_lines([odds_event()])

    def test_it_keeps_only_markets_naming_a_game_we_have_a_line_for(
        self, lines
    ):
        client = self.Client([
            self.event("Chiefs vs Broncos: winning margin",
                       ["KXNFLSPREAD-1", "KXNFLSPREAD-2"]),
            self.event("Best Picture winner", ["KXOSCARPIC-1"]),
            self.event("Will the Fed cut rates?", ["KXFED-1"]),
        ])
        markets, series, _near = KS.collect_game_markets(client, lines)
        assert [m.ticker for m in markets] == ["KXNFLSPREAD-1",
                                               "KXNFLSPREAD-2"]
        assert series == {"KXNFLSPREAD": 2}

    def test_it_reports_which_series_the_games_live_in(self, lines):
        # So a narrowed re-run is a command rather than a guess.
        client = self.Client([
            self.event("Chiefs vs Broncos margin", ["KXNFLSPREAD-1"]),
            self.event("Chiefs vs Broncos total", ["KXNFLTOTAL-1"]),
        ])
        _markets, series, _near = KS.collect_game_markets(client, lines)
        assert dict(series) == {"KXNFLSPREAD": 1, "KXNFLTOTAL": 1}

    def test_the_event_title_is_carried_into_each_market(self, lines):
        # A market's own subtitle is often just a threshold. Without the
        # event title there is no team name to match on at all, which is
        # the other half of why nothing joined.
        client = self.Client([
            self.event("Chiefs vs Broncos margin", ["KXNFLSPREAD-1"]),
        ])
        (market,), _series, _near = KS.collect_game_markets(client, lines)
        assert "Chiefs" in market.title and "Broncos" in market.title

    def test_an_unrelated_board_yields_nothing_rather_than_noise(self, lines):
        client = self.Client([self.event("Best Picture", ["KXOSCARPIC-1"])])
        markets, series, _near = KS.collect_game_markets(client, lines)
        assert markets == [] and not series

    def test_a_malformed_market_does_not_lose_its_event(self, lines):
        client = self.Client([{
            "title": "Chiefs vs Broncos margin",
            "markets": [{"junk": True},
                        {"ticker": "KXNFLSPREAD-2", "subtitle": "Above 6.5",
                         "status": "active"}],
        }])
        markets, _series, _near = KS.collect_game_markets(client, lines)
        assert [m.ticker for m in markets] == ["KXNFLSPREAD-2"]

    def test_one_team_name_is_not_enough(self):
        """
        The collision that produced twenty-eight phantom candidates on a
        live run: Winnipeg's JETS on an NFL board, and US Treasury BILLS
        against Buffalo's. Requiring one team let both through, and they
        then buried the real finding -- that nothing named two.
        """
        lines = KS.game_lines([odds_event(
            home="Buffalo Bills", away="New York Jets",
            home_point=-3.5, eid="buf-nyj",
        )])
        client = self.Client([
            self.event("Winnipeg Jets season points", ["KXNHLSEASONPTS-1"]),
            self.event("US Treasury bills above 4%", ["KXUSDTTBILL-1"]),
        ])
        markets, series, near = KS.collect_game_markets(client, lines)
        assert markets == []
        assert not series
        # But the collision is REPORTED, because "the board is empty" and
        # "something matched for the wrong reason" are different answers.
        assert near["jets"] == 1
        assert near["bills"] == 1

    def test_a_real_game_still_passes_with_both_teams(self):
        lines = KS.game_lines([odds_event(
            home="Buffalo Bills", away="New York Jets",
            home_point=-3.5, eid="buf-nyj",
        )])
        client = self.Client([
            self.event("Jets at Bills: winning margin", ["KXNFLSPREAD-1"]),
        ])
        markets, series, near = KS.collect_game_markets(client, lines)
        assert len(markets) == 1
        assert series == {"KXNFLSPREAD": 1}
        assert not near

    def test_an_unreachable_events_endpoint_is_not_fatal(self, lines):
        class Broken(self.Client):
            def events(self, **kw):
                raise RuntimeError("nope")

        assert KS.collect_game_markets(Broken([]), lines) == (
            [], Counter(), Counter())


class TestTheMoneylineRung:
    """
    A market asking only who wins is a rung at zero, and it may be all an
    exchange lists for a game. Refusing it would have skipped every
    market on Kalshi's KXNFLGAME series.
    """

    @pytest.fixture
    def setup(self):
        cfg = Config()
        priors = L.PriorSet.load()
        (line,) = KS.game_lines([odds_event()])
        model = L.fit(line.spread, line.fair_win_prob,
                      "americanfootball_nfl", priors)
        return cfg, line, model

    @pytest.mark.parametrize("title", [
        "Will the Chiefs beat the Broncos?",
        "Chiefs vs Broncos: Chiefs win",
        "Will the Chiefs defeat the Broncos?",
    ])
    def test_a_winner_market_is_a_rung_at_zero(self, title):
        threshold, _fav, problem = KS.parse_rung(
            FakeMarket("T", title), "Kansas City Chiefs", "Denver Broncos"
        )
        assert threshold == 0.0
        assert problem == ""

    @pytest.mark.parametrize("title", [
        "Chiefs to win by more than 6.5",
        "Chiefs vs Broncos total points over 44.5",
        "Chiefs first quarter winner",
    ])
    def test_something_else_is_not_mistaken_for_one(self, title):
        threshold, _fav, _problem = KS.parse_rung(
            FakeMarket("T", title), "Kansas City Chiefs", "Denver Broncos"
        )
        assert threshold != 0.0

    @pytest.mark.parametrize("title", [
        "Chiefs to win the AFC",
        "Will the Chiefs win the Super Bowl?",
        "Chiefs to win their division",
    ])
    def test_a_futures_market_is_not_this_game_s_moneyline(self, title):
        """
        Priced as a moneyline, "Chiefs to win the AFC" would be compared
        against Thursday's de-vigged win probability. That is not
        slightly wrong, it is a different question entirely -- and the
        numbers would all look perfectly reasonable.
        """
        threshold, _fav, problem = KS.parse_rung(
            FakeMarket("T", title), "Kansas City Chiefs", "Denver Broncos"
        )
        assert threshold is None
        assert problem

    def test_a_game_names_both_teams_and_a_future_names_one(self):
        # The principled check behind the word list: two teams means a
        # game, one means something season-long.
        game, _fav, _p = KS.parse_rung(
            FakeMarket("T", "Will the Chiefs beat the Broncos?"),
            "Kansas City Chiefs", "Denver Broncos",
        )
        future, _fav2, problem = KS.parse_rung(
            FakeMarket("T", "Will the Chiefs win it all?"),
            "Kansas City Chiefs", "Denver Broncos",
        )
        assert game == 0.0
        assert future is None
        assert "names one team only" in problem

    def test_it_still_has_to_say_which_team(self):
        threshold, _fav, problem = KS.parse_rung(
            FakeMarket("T", "Who will win?"), "Kansas City Chiefs",
            "Denver Broncos"
        )
        assert threshold is None
        assert "which team" in problem

    def test_it_is_priced_from_the_sharp_moneyline_not_the_model(self, setup):
        """
        The one rung the sharp book quotes DIRECTLY. Its de-vigged
        moneyline is a better estimate of who wins than this model
        produces -- the model exists for the rungs Pinnacle does not
        price, and using it here would mean disagreeing with the
        sharpest number on the board for no reason.
        """
        cfg, line, model = setup
        markets = [FakeMarket("KXNFLGAME-1", "Will the Chiefs beat the Broncos?")]
        books = {"KXNFLGAME-1": book(no_bids=[[40, 500]])}
        (quote,), _ = KS.price_rungs(model, line, markets, books, cfg)
        assert quote.fair == pytest.approx(line.fair_win_prob)
        assert quote.fair != pytest.approx(model.prob_margin_over(0.0))

    def test_and_says_so_on_the_quote(self, setup):
        cfg, line, model = setup
        markets = [FakeMarket("KXNFLGAME-1", "Will the Chiefs beat the Broncos?")]
        books = {"KXNFLGAME-1": book(no_bids=[[40, 500]])}
        (quote,), _ = KS.price_rungs(model, line, markets, books, cfg)
        assert "priced_from_the_sharp_moneyline" in " ".join(quote.flags)

    def test_it_describes_itself_as_a_moneyline(self, setup):
        cfg, line, model = setup
        markets = [FakeMarket("KXNFLGAME-1", "Will the Chiefs beat the Broncos?")]
        books = {"KXNFLGAME-1": book(no_bids=[[40, 500]])}
        (quote,), _ = KS.price_rungs(model, line, markets, books, cfg)
        assert "to win" in quote.describe()
        assert "Kansas City Chiefs" in quote.describe()

    def test_a_real_ladder_rung_still_uses_the_model(self, setup):
        cfg, line, model = setup
        markets = [FakeMarket("T", "Chiefs to beat the Broncos by more than 9.5")]
        books = {"T": book(no_bids=[[60, 500]])}
        (quote,), _ = KS.price_rungs(model, line, markets, books, cfg)
        assert quote.fair == pytest.approx(model.prob_margin_over(9.5))
        assert "priced_from_the_sharp_moneyline" not in " ".join(quote.flags)


class TestATeamNamedMarket:
    """
    Kalshi's KXNFLGAME lists each fixture ONCE PER TEAM. The event names
    both, the market's own subtitle is just a team name -- no verb, no
    number, nothing a word list would recognise. Requiring a winner word
    refused all thirty-two markets on a live Week 3 board after the join
    had already succeeded on all sixteen games.

    The structure is the signal: the subtitle naming exactly one of the
    two teams IS the side the contract pays on.
    """

    @pytest.fixture
    def setup(self):
        cfg = Config()
        priors = L.PriorSet.load()
        (line,) = KS.game_lines([odds_event()])
        model = L.fit(line.spread, line.fair_win_prob,
                      "americanfootball_nfl", priors)
        return cfg, line, model

    def market(self, team):
        return FakeMarket(
            "KXNFLGAME-26SEP17DENKC-X",
            "Denver Broncos at Kansas City Chiefs",
            subtitle=team,
        )

    def test_the_favourite_s_contract_is_read(self):
        threshold, about_favourite, problem = KS.parse_rung(
            self.market("Kansas City Chiefs"),
            "Kansas City Chiefs", "Denver Broncos",
        )
        assert (threshold, about_favourite, problem) == (0.0, True, "")

    def test_the_underdog_s_contract_is_read_as_the_other_side(self):
        threshold, about_favourite, problem = KS.parse_rung(
            self.market("Denver Broncos"),
            "Kansas City Chiefs", "Denver Broncos",
        )
        assert (threshold, about_favourite, problem) == (0.0, False, "")

    def test_the_underdog_is_priced_at_the_complement(self, setup):
        """
        Half these contracts pay on the UNDERDOG. Pricing them with the
        favourite's probability would make every single one look like a
        gift -- a 30% shot quoted as a 70% one.
        """
        cfg, line, model = setup
        markets = [self.market("Denver Broncos")]
        books = {markets[0].ticker: book(no_bids=[[60, 500]])}
        (quote,), _ = KS.price_rungs(model, line, markets, books, cfg)
        assert quote.fair == pytest.approx(1.0 - line.fair_win_prob)
        assert quote.favourite == "Denver Broncos"

    def test_the_favourite_is_priced_at_the_sharp_number(self, setup):
        cfg, line, model = setup
        markets = [self.market("Kansas City Chiefs")]
        books = {markets[0].ticker: book(no_bids=[[40, 500]])}
        (quote,), _ = KS.price_rungs(model, line, markets, books, cfg)
        assert quote.fair == pytest.approx(line.fair_win_prob)

    def test_the_two_sides_are_complements_of_each_other(self, setup):
        # The internal check: one game, two contracts, probabilities that
        # sum to one. If they ever do not, a side has been mislabelled.
        cfg, line, model = setup
        markets = [self.market("Kansas City Chiefs"),
                   self.market("Denver Broncos")]
        books = {m.ticker: book(no_bids=[[50, 500]]) for m in markets}
        quotes, _ = KS.price_rungs(model, line, markets, books, cfg)
        assert len(quotes) == 2
        assert sum(q.fair for q in quotes) == pytest.approx(1.0)

    def test_a_subtitle_naming_both_teams_is_not_a_side(self):
        # That is the event, not one team's contract.
        threshold, _fav, problem = KS.parse_rung(
            FakeMarket("T", "x", subtitle="Broncos at Chiefs"),
            "Kansas City Chiefs", "Denver Broncos",
        )
        assert threshold is None

    def test_a_subtitle_carrying_a_number_is_not_a_moneyline(self):
        # "Chiefs by 7+" is a margin rung and must go down the other
        # path, or the spread would be silently ignored.
        threshold, _fav, _problem = KS.parse_rung(
            FakeMarket("T", "Broncos at Chiefs", subtitle="Chiefs by 7"),
            "Kansas City Chiefs", "Denver Broncos",
        )
        assert threshold != 0.0

    def test_a_margin_rung_on_the_underdog_flips_its_threshold(self, setup):
        # One distribution read at a different point: the underdog
        # covering by 3 is the favourite's margin below -3.
        cfg, line, model = setup
        threshold, about_favourite, _ = KS.parse_rung(
            FakeMarket("T", "Broncos at Chiefs", subtitle="Broncos by 3"),
            "Kansas City Chiefs", "Denver Broncos",
        )
        assert about_favourite is False
