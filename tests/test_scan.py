"""End-to-end: payload in, ranked opportunities out."""

from datetime import timedelta

import pytest

from betedge import scan as S
from betedge.scan import scan
from conftest import NOW, FakeClient, book, event_payload, outcome


def nfl_game(dk_home_price=2.10, pin=(1.95, 1.95), commence_hours=10, event_id="evt1"):
    """A game line: Pinnacle at `pin`, DraftKings offering `dk_home_price`."""
    return event_payload(
        event_id=event_id,
        commence=NOW + timedelta(hours=commence_hours),
        bookmakers=[
            book("pinnacle", {"h2h": [
                outcome("Kansas City Chiefs", pin[0]),
                outcome("Denver Broncos", pin[1]),
            ]}),
            book("draftkings", {"h2h": [
                outcome("Kansas City Chiefs", dk_home_price),
                outcome("Denver Broncos", 1.80),
            ]}),
        ],
    )


def mlb_prop(dk_over=2.00, pin=(1.87, 1.95), market="pitcher_strikeouts"):
    return event_payload(
        event_id="mlb1",
        sport="baseball_mlb",
        commence=NOW + timedelta(hours=6),
        home="Los Angeles Dodgers",
        away="San Diego Padres",
        bookmakers=[
            book("pinnacle", {market: [
                outcome("Over", pin[0], point=6.5, description="Yoshinobu Yamamoto"),
                outcome("Under", pin[1], point=6.5, description="Yoshinobu Yamamoto"),
            ]}),
            book("draftkings", {market: [
                outcome("Over", dk_over, point=6.5, description="Yoshinobu Yamamoto"),
                outcome("Under", 1.80, point=6.5, description="Yoshinobu Yamamoto"),
            ]}),
        ],
    )


class TestParsing:
    def test_flattens_every_book_and_market(self):
        meta, quotes = S.parse_event_odds(nfl_game())
        assert meta["home_team"] == "Kansas City Chiefs"
        assert {q.book for q in quotes} == {"pinnacle", "draftkings"}
        assert len(quotes) == 4

    def test_player_name_comes_from_description(self):
        _meta, quotes = S.parse_event_odds(mlb_prop())
        assert all(q.selection == "Yoshinobu Yamamoto" for q in quotes)
        assert {q.side for q in quotes} == {"Over", "Under"}

    def test_nonsense_prices_are_dropped(self):
        payload = event_payload(bookmakers=[
            book("pinnacle", {"h2h": [
                outcome("A", 1.0), outcome("B", 0), outcome("C", None), outcome("D", 2.0),
            ]})
        ])
        _meta, quotes = S.parse_event_odds(payload)
        assert [q.side for q in quotes] == ["D"]

    def test_a_malformed_book_does_not_kill_the_event(self):
        payload = event_payload(bookmakers=[
            {"key": "broken"},
            book("pinnacle", {"h2h": [outcome("A", 1.95), outcome("B", 1.95)]}),
        ])
        _meta, quotes = S.parse_event_odds(payload)
        assert len(quotes) == 2


class TestGrouping:
    def test_spread_sides_pair_on_the_magnitude_of_the_handicap(self):
        q1 = S.Quote("pinnacle", "spreads", "A", "A", -2.5, 1.9, None)
        q2 = S.Quote("pinnacle", "spreads", "B", "B", 2.5, 1.9, None)
        assert S.group_key(q1) == S.group_key(q2)

    def test_different_handicaps_never_pair(self):
        q1 = S.Quote("draftkings", "spreads", "A", "A", -2.5, 1.9, None)
        q2 = S.Quote("pinnacle", "spreads", "A", "A", -3.5, 1.9, None)
        assert S.group_key(q1) != S.group_key(q2)

    def test_props_group_by_player_and_line(self):
        a = S.Quote("pinnacle", "player_points", "LeBron", "Over", 25.5, 1.9, None)
        b = S.Quote("pinnacle", "player_points", "LeBron", "Under", 25.5, 1.9, None)
        c = S.Quote("pinnacle", "player_points", "Curry", "Over", 25.5, 1.9, None)
        assert S.group_key(a) == S.group_key(b) != S.group_key(c)

    def test_game_totals_do_not_carry_a_player(self):
        q = S.Quote("pinnacle", "totals", "Over", "Over", 45.5, 1.9, None)
        assert S.group_key(q) == ("totals", None, 45.5)

    def test_one_sided_markets_are_not_returned(self):
        quotes = [S.Quote("pinnacle", "totals", "Over", "Over", 45.5, 1.9, None)]
        assert S.group_sharp_markets(quotes, "pinnacle") == {}


class TestEvaluate:
    def test_a_real_edge_is_flagged(self, cfg, now):
        meta, quotes = S.parse_event_odds(nfl_game(dk_home_price=2.10))
        opps = S.evaluate_event(meta, quotes, cfg, now=now)
        assert len(opps) == 1
        o = opps[0]
        assert o.selection == "Kansas City Chiefs"
        assert o.ev == pytest.approx(0.05, abs=0.002)
        assert o.recommended_stake > 0
        assert not o.suspect

    def test_no_edge_is_not_flagged(self, cfg, now):
        meta, quotes = S.parse_event_odds(nfl_game(dk_home_price=1.90))
        rej = {}
        assert S.evaluate_event(meta, quotes, cfg, now=now, rejections=rej) == []
        assert rej.get("below_min_ev")

    def test_an_implausible_edge_is_marked_suspect_and_not_staked(self, cfg, now):
        meta, quotes = S.parse_event_odds(nfl_game(dk_home_price=2.60))
        o = S.evaluate_event(meta, quotes, cfg, now=now)[0]
        assert o.suspect and o.recommended_stake == 0
        assert any("implausible" in f for f in o.flags)

    def test_an_event_starting_too_soon_is_skipped(self, cfg, now):
        meta, quotes = S.parse_event_odds(nfl_game(commence_hours=0.01))
        rej = {}
        assert S.evaluate_event(meta, quotes, cfg, now=now, rejections=rej) == []
        assert rej.get("too_close_to_start")

    def test_an_event_too_far_out_is_skipped(self, cfg, now):
        meta, quotes = S.parse_event_odds(nfl_game(commence_hours=200))
        rej = {}
        assert S.evaluate_event(meta, quotes, cfg, now=now, rejections=rej) == []
        assert rej.get("too_far_out")

    def test_a_malformed_overround_is_skipped(self, cfg, now):
        # Both sides at 1.30 is a 54% book: not a real market.
        meta, quotes = S.parse_event_odds(nfl_game(dk_home_price=2.10, pin=(1.30, 1.30)))
        rej = {}
        assert S.evaluate_event(meta, quotes, cfg, now=now, rejections=rej) == []
        assert rej.get("sharp_overround_out_of_bounds")

    def test_a_soft_line_the_sharp_book_does_not_price_is_skipped(self, cfg, now):
        payload = event_payload(
            commence=NOW + timedelta(hours=10),
            bookmakers=[
                book("pinnacle", {"totals": [
                    outcome("Over", 1.95, point=45.5), outcome("Under", 1.95, point=45.5)]}),
                book("draftkings", {"totals": [
                    outcome("Over", 2.20, point=47.5), outcome("Under", 1.70, point=47.5)]}),
            ],
        )
        meta, quotes = S.parse_event_odds(payload)
        rej = {}
        assert S.evaluate_event(meta, quotes, cfg, now=now, rejections=rej) == []
        assert rej.get("no_matching_sharp_line")

    def test_a_stale_sharp_quote_is_suspect(self, cfg, now):
        payload = event_payload(
            commence=NOW + timedelta(hours=10),
            bookmakers=[
                book("pinnacle", {"h2h": [
                    outcome("Kansas City Chiefs", 1.95), outcome("Denver Broncos", 1.95)]},
                    last_update=NOW - timedelta(minutes=90)),
                book("draftkings", {"h2h": [
                    outcome("Kansas City Chiefs", 2.10), outcome("Denver Broncos", 1.80)]}),
            ],
        )
        meta, quotes = S.parse_event_odds(payload)
        o = S.evaluate_event(meta, quotes, cfg, now=now)[0]
        assert o.suspect and any("sharp_quote" in f for f in o.flags)


class TestLiquidityBar:
    def test_a_mainline_edge_clears_the_base_bar(self, cfg, now):
        meta, quotes = S.parse_event_odds(nfl_game(dk_home_price=2.06))
        opps = S.evaluate_event(meta, quotes, cfg, now=now)
        assert opps and opps[0].market_tier == "mainline"
        assert opps[0].liquidity > 0.85

    def test_the_same_edge_on_a_thin_prop_does_not(self, cfg, now):
        """
        The behaviour change this feature exists for. A +2.1% edge on a
        5% prop market used to be flagged on a flat 2% bar; the fair
        probability there is not precise enough to support it.
        """
        meta, quotes = S.parse_event_odds(mlb_prop(dk_over=2.00))
        rej = {}
        opps = S.evaluate_event(meta, quotes, cfg, now=now, rejections=rej)
        assert opps == []
        assert rej.get("below_liquidity_bar")

    def test_a_bigger_edge_on_the_same_prop_still_gets_through(self, cfg, now):
        meta, quotes = S.parse_event_odds(mlb_prop(dk_over=2.30))
        opps = S.evaluate_event(meta, quotes, cfg, now=now)
        assert len(opps) == 1
        assert opps[0].ev > opps[0].required_ev
        assert opps[0].market_tier == "primary_prop"

    def test_turning_the_penalty_off_restores_the_flat_bar(self, cfg, now):
        cfg.model.liquidity_ev_penalty = 0.0
        meta, quotes = S.parse_event_odds(mlb_prop(dk_over=2.00))
        assert S.evaluate_event(meta, quotes, cfg, now=now)

    def test_the_min_liquidity_floor_rejects_outright(self, cfg, now):
        cfg.model.min_liquidity = 0.9
        meta, quotes = S.parse_event_odds(mlb_prop(dk_over=2.60))
        rej = {}
        assert S.evaluate_event(meta, quotes, cfg, now=now, rejections=rej) == []
        assert rej.get("market_too_thin")

    def test_ranking_prefers_a_deep_market_over_a_bigger_thin_edge(self, cfg, now):
        """
        A +4.7% strikeout prop (liquidity 0.58) against a +3.0% NFL side
        (liquidity 0.99). Both clear their bars, so both are real
        candidates; the question is only which one the list opens with.
        Discounted, the prop is worth 0.027 and the side 0.030, so the side
        leads -- you can get real money down on it and the number behind it
        is one Pinnacle will defend.
        """
        client = FakeClient(
            bulk_odds={"americanfootball_nfl": [nfl_game(dk_home_price=2.06)]},
            events_by_sport={"baseball_mlb": [
                {"id": "mlb1", "commence_time": (NOW + timedelta(hours=6)).isoformat()}]},
            event_odds={("baseball_mlb", "mlb1"): mlb_prop(dk_over=2.05)},
        )
        cfg.core_sports = ["americanfootball_nfl"]
        cfg.sports = ["baseball_mlb"]
        cfg.prop_markets = {"baseball_mlb": ["pitcher_strikeouts"]}
        result = scan(cfg, client, now=now)

        assert len(result.opportunities) == 2
        top, second = result.opportunities
        assert top.market == "h2h", "the deep market should rank first"
        assert top.ev < second.ev, "even though its raw EV is lower"
        assert top.edge_score > second.edge_score
        # Both are genuinely playable -- this is a ranking decision, not a
        # rejection. Ranking on raw EV would have opened with the prop.
        assert all(o.ev > o.required_ev for o in result.opportunities)


class TestOrchestration:
    def test_cheap_core_pass_runs_before_expensive_props(self, cfg, now):
        """
        Ordering matters: whichever pass runs second is the one a tight
        budget truncates, and it should not be the cheap deep one.
        """
        client = FakeClient(
            bulk_odds={"americanfootball_nfl": [nfl_game()]},
            events_by_sport={"baseball_mlb": [
                {"id": "mlb1", "commence_time": (NOW + timedelta(hours=6)).isoformat()}]},
            event_odds={("baseball_mlb", "mlb1"): mlb_prop()},
        )
        cfg.core_sports = ["americanfootball_nfl"]
        cfg.sports = ["baseball_mlb"]
        scan(cfg, client, now=now)
        assert [c[0] for c in client.order] == ["core", "prop"]

    def test_wildcards_expand_against_the_live_sport_list(self, cfg, now):
        client = FakeClient(
            sports_list=[{"key": "tennis_atp_china_open"}, {"key": "tennis_wta_wuhan"},
                         {"key": "baseball_mlb"}],
            bulk_odds={"tennis_atp_china_open": [], "tennis_wta_wuhan": []},
        )
        cfg.core_sports = ["tennis_*"]
        result = scan(cfg, client, now=now)
        assert "tennis_atp_china_open" in result.sports
        assert "tennis_wta_wuhan" in result.sports
        assert "baseball_mlb" not in result.sports

    def test_events_outside_the_window_are_never_paid_for(self, cfg, now):
        client = FakeClient(events_by_sport={"baseball_mlb": [
            {"id": "soon", "commence_time": (NOW + timedelta(minutes=1)).isoformat()},
            {"id": "far", "commence_time": (NOW + timedelta(days=9)).isoformat()},
        ]})
        cfg.sports = ["baseball_mlb"]
        scan(cfg, client, now=now)
        assert client.calls["event_odds"] == 0

    def test_one_failing_sport_does_not_kill_the_scan(self, cfg, now):
        class Flaky(FakeClient):
            def odds(self, sport, **kw):
                if sport == "bad":
                    raise RuntimeError("boom")
                return super().odds(sport, **kw)

        client = Flaky(bulk_odds={"americanfootball_nfl": [nfl_game()]})
        cfg.core_sports = ["bad", "americanfootball_nfl"]
        result = scan(cfg, client, now=now)
        assert result.opportunities
        assert any("boom" in e for e in result.errors)

    def test_max_events_caps_the_prop_spend(self, cfg, now):
        events = [{"id": f"e{i}", "commence_time": (NOW + timedelta(hours=6)).isoformat()}
                  for i in range(10)]
        client = FakeClient(events_by_sport={"baseball_mlb": events})
        cfg.sports = ["baseball_mlb"]
        scan(cfg, client, now=now, max_events_per_sport=3)
        assert client.calls["event_odds"] == 3


class TestPostProcessing:
    def _opp(self, ev, liquidity, market="h2h", selection="A", stake=20.0):
        return S.Opportunity(
            sport="s", event_id="e", commence_time=NOW, home_team="H", away_team="A",
            market=market, selection=selection, line=None, side=selection,
            sharp_book="pinnacle", sharp_price_taken_side=1.95,
            sharp_price_other_side=1.95, sharp_overround=0.025, fair_prob=0.5,
            fair_price=2.0, devig_method="worst_case", devig_spread=0.001,
            fair_prob_by_method={}, soft_book="draftkings", soft_price=2.1, ev=ev,
            ev_range=(ev, ev), kelly_fraction=0.05, recommended_stake=stake,
            market_tier="mainline", liquidity=liquidity, required_ev=0.02,
            edge_score=ev * liquidity, sharp_last_update=None, soft_last_update=None,
            scanned_at=NOW,
        )

    def test_best_per_selection_keeps_the_best_edge_score(self):
        opps = [self._opp(0.08, 0.3), self._opp(0.04, 1.0)]
        kept = S.best_per_selection(opps)
        assert len(kept) == 1
        assert kept[0].liquidity == 1.0, "keeps the trustworthy one, not the loud one"

    def test_exposure_cap_scales_stakes_down(self, cfg):
        cfg.bankroll.amount = 1000
        cfg.bankroll.max_total_exposure_fraction = 0.10   # 100 total
        opps = [self._opp(0.05, 1.0, selection=f"S{i}", stake=50.0) for i in range(6)]
        S.cap_exposure(opps, cfg)
        assert S.total_exposure(opps) <= 100 + cfg.bankroll.round_to

    def test_exposure_cap_leaves_a_small_board_alone(self, cfg):
        cfg.bankroll.amount = 1000
        opps = [self._opp(0.05, 1.0, stake=20.0)]
        S.cap_exposure(opps, cfg)
        assert opps[0].recommended_stake == 20.0
