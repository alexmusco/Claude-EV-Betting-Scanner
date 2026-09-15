"""Client behaviour around quota accounting and market rejection."""

import pytest

from betedge.oddsapi import CreditBudgetExceeded, OddsApiClient, Quota


class TestQuota:
    def test_headers_are_read(self):
        q = Quota()
        q.update({"x-requests-remaining": "19950",
                  "x-requests-used": "50", "x-requests-last": "3"})
        assert (q.remaining, q.used, q.last_cost, q.spent_this_session) == (19950, 50, 3, 3)

    def test_session_spend_accumulates(self):
        q = Quota()
        for _ in range(3):
            q.update({"x-requests-last": "4"})
        assert q.spent_this_session == 12

    def test_zero_remaining_is_recorded_not_discarded(self):
        """`or` treated a remaining count of 0 as absent, so the stale value
        survived at exactly the moment the floor needed to trip."""
        q = Quota(remaining=500)
        q.update({"x-requests-remaining": "0"})
        assert q.remaining == 0

    def test_missing_headers_leave_the_previous_value_alone(self):
        q = Quota(remaining=100, used=5)
        q.update({})
        assert q.remaining == 100 and q.used == 5

    def test_junk_headers_are_ignored(self):
        q = Quota(remaining=100)
        q.update({"x-requests-remaining": "not-a-number"})
        assert q.remaining == 100


class TestBudgetGuards:
    def test_per_scan_ceiling_stops_further_calls(self):
        c = OddsApiClient(api_key="k", max_credits_per_scan=10)
        c.quota.spent_this_session = 10
        with pytest.raises(CreditBudgetExceeded, match="scan budget"):
            c._check_budget()

    def test_remaining_floor_stops_further_calls(self):
        c = OddsApiClient(api_key="k", min_credits_remaining=100)
        c.quota.remaining = 100
        with pytest.raises(CreditBudgetExceeded, match="credits left"):
            c._check_budget()

    def test_a_healthy_client_passes(self):
        c = OddsApiClient(api_key="k", max_credits_per_scan=600,
                          min_credits_remaining=100)
        c.quota.spent_this_session = 5
        c.quota.remaining = 19000
        c._check_budget()


class TestMarketRejection:
    def test_offending_market_keys_are_identified_from_the_error(self):
        msg = ("422: {'message': \"Unknown market. The market key "
               "'player_fantasy_pts' is not valid\"}")
        found = OddsApiClient._markets_named_in(
            msg, ["player_pass_yds", "player_fantasy_pts"]
        )
        assert found == {"player_fantasy_pts"}

    def test_an_unrelated_error_names_no_markets(self):
        found = OddsApiClient._markets_named_in("500 Internal Server Error",
                                                ["player_pass_yds"])
        assert found == set()

    def test_a_bad_key_is_dropped_and_the_call_retried(self, monkeypatch):
        """
        One unsupported key used to 422 the whole event -- and since the
        market list is identical for every event in the sport, the whole
        slate. A rejected call is not billed, so retrying without it is free.
        """
        c = OddsApiClient(api_key="k")
        attempts = []

        def fake_get(path, params, billed=True):
            markets = params["markets"].split(",")
            attempts.append(markets)
            if "bad_market" in markets:
                raise __import__("betedge.oddsapi", fromlist=["OddsApiError"]).OddsApiError(
                    "422: Unknown market. The market key 'bad_market' is not valid"
                )
            return {"id": "e1", "bookmakers": []}

        monkeypatch.setattr(c, "_get", fake_get)
        result = c.event_odds("nfl", "e1", ["player_pass_yds", "bad_market"], ["pinnacle"])

        assert result is not None
        assert attempts == [["player_pass_yds", "bad_market"], ["player_pass_yds"]]
        assert "bad_market" in c._bad_markets

    def test_a_known_bad_key_is_not_requested_again(self, monkeypatch):
        c = OddsApiClient(api_key="k")
        c._bad_markets.add("bad_market")
        seen = []

        def fake_get(path, params, billed=True):
            seen.append(params["markets"])
            return {"id": "e1", "bookmakers": []}

        monkeypatch.setattr(c, "_get", fake_get)
        c.event_odds("nfl", "e1", ["player_pass_yds", "bad_market"], ["pinnacle"])
        assert seen == ["player_pass_yds"]

    def test_all_markets_bad_returns_none_without_calling(self, monkeypatch):
        c = OddsApiClient(api_key="k")
        c._bad_markets.add("bad_market")
        monkeypatch.setattr(c, "_get", lambda *a, **k: pytest.fail("should not call"))
        assert c.event_odds("nfl", "e1", ["bad_market"], ["pinnacle"]) is None
