"""Shared fixtures: a fake API client and payload builders.

Nothing in the suite touches the network or spends a credit.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from betedge.config import Config
from betedge.oddsapi import Quota

# Anchored to import time, not a fixed date.
#
# Every fixture offset below is relative to NOW, and the tests that drive
# the engine directly pass `now=NOW` explicitly, so relative timing stays
# exactly as deterministic as a hardcoded date would make it.
#
# The CLI path is why this cannot be frozen. `betedge daily` calls
# datetime.now() internally, while the fixtures stamp each quote's
# last_update at NOW. With a fixed NOW those two drift apart as the wall
# clock advances, and once the gap passes max_soft_staleness_minutes every
# quote is stale, every opportunity is flagged suspect, and the CLI tests
# fail for reasons that have nothing to do with the code. That is a suite
# that rots on a timer.
NOW = datetime.now(timezone.utc).replace(microsecond=0)


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


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """
    Enforce the suite's promise rather than assume it.

    "No network, no credits" was true by construction while every caller
    took an injected client. Roster refresh broke that: `parlay scan` can
    now fetch a roster feed on its own, and a test that forgot to turn it
    off quietly made a real HTTP request -- which passed, slowly, and would
    fail on a machine with no route out. Blocking requests at the source
    turns that from a silent dependency into an immediate, obvious failure.
    """
    import requests

    def blocked(*args, **kwargs):
        raise AssertionError(
            f"a test tried to reach the network: {args[:2]}. Inject a fake "
            "opener or client instead."
        )

    monkeypatch.setattr(requests, "get", blocked)
    monkeypatch.setattr(requests, "post", blocked)
    monkeypatch.setattr(requests.Session, "request", blocked)


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
