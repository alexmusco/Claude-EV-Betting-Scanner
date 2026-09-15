"""Shared fixtures: a fake API client and payload builders.

Nothing in the suite touches the network or spends a credit.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from betedge.config import Config
from betedge.oddsapi import Quota

NOW = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)


def iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def outcome(name, price, point=None, description=None):
    o = {"name": name, "price": price}
    if point is not None:
        o["point"] = point
    if description is not None:
        o["description"] = description
    return o


def book(key, markets, last_update=None):
    return {
        "key": key,
        "last_update": iso(last_update or NOW),
        "markets": [
            {"key": mk, "last_update": iso(last_update or NOW), "outcomes": outs}
            for mk, outs in markets.items()
        ],
    }


def event_payload(
    event_id="evt1",
    sport="americanfootball_nfl",
    commence=None,
    home="Kansas City Chiefs",
    away="Denver Broncos",
    bookmakers=None,
):
    return {
        "id": event_id,
        "sport_key": sport,
        "commence_time": iso(commence or (NOW + timedelta(hours=10))),
        "home_team": home,
        "away_team": away,
        "bookmakers": bookmakers or [],
    }


class FakeClient:
    """Stands in for OddsApiClient. Counts calls so cost can be asserted."""

    def __init__(self, *, events_by_sport=None, event_odds=None,
                 bulk_odds=None, sports_list=None, remaining=20000):
        self._events = events_by_sport or {}
        self._event_odds = event_odds or {}
        self._bulk = bulk_odds or {}
        self._sports = sports_list or []
        self.quota = Quota(remaining=remaining)
        self.max_credits_per_scan = None
        self.calls = {"sports": 0, "events": 0, "odds": 0, "event_odds": 0}
        self.order = []

    def sports(self, all_sports=False):
        self.calls["sports"] += 1
        return self._sports

    def events(self, sport):
        self.calls["events"] += 1
        return self._events.get(sport, [])

    def odds(self, sport, markets=("h2h",), bookmakers=("pinnacle",), odds_format="decimal"):
        self.calls["odds"] += 1
        self.order.append(("core", sport))
        self.quota.spent_this_session += len(markets)
        return self._bulk.get(sport, [])

    def event_odds(self, sport, event_id, markets, bookmakers, odds_format="decimal"):
        self.calls["event_odds"] += 1
        self.order.append(("prop", sport, event_id))
        self.quota.spent_this_session += len(markets)
        return self._event_odds.get((sport, event_id))

    def probe_quota(self):
        self.calls["sports"] += 1
        return self.quota


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.books.soft = ["draftkings"]
    c.sports = []
    c.core_sports = []
    c.database = str(tmp_path / "test.db")
    c.reports_dir = str(tmp_path / "reports")
    return c


@pytest.fixture
def now():
    return NOW
