"""
The Odds API v4 client.

Two things this does that the old R code did not:

1. Tracks credits. Every response carries x-requests-remaining,
   x-requests-used and x-requests-last. The client reads them, so it knows
   the real cost of each call rather than guessing, and refuses to keep
   going once a per-scan budget or a remaining-credit floor is hit.

2. Sends `bookmakers` and never `regions`. Cost is markets x regions, and
   ten bookmakers count as one region equivalent, so naming three books
   keeps the multiplier at 1. Sending `regions=us,eu` as well -- as the old
   scripts did -- doubles the bill and returns nothing extra.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import requests

log = logging.getLogger(__name__)


class OddsApiError(RuntimeError):
    pass


class CreditBudgetExceeded(OddsApiError):
    """Raised when a scan would spend more than it is allowed to."""


@dataclass
class Quota:
    remaining: int | None = None
    used: int | None = None
    last_cost: int | None = None
    spent_this_session: int = 0

    def update(self, headers) -> None:
        def _int(name):
            v = headers.get(name)
            try:
                return int(float(v))
            except (TypeError, ValueError):
                return None

        self.remaining = _int("x-requests-remaining") or self.remaining
        used = _int("x-requests-used")
        if used is not None:
            self.used = used
        cost = _int("x-requests-last")
        if cost is not None:
            self.last_cost = cost
            self.spent_this_session += cost


@dataclass
class OddsApiClient:
    api_key: str
    base_url: str = "https://api.the-odds-api.com/v4"
    timeout: float = 20.0
    max_retries: int = 3
    cache_seconds: int = 0
    cache_dir: Path = field(default_factory=lambda: Path("data/cache"))
    max_credits_per_scan: int | None = None
    min_credits_remaining: int = 0
    quota: Quota = field(default_factory=Quota)
    session: requests.Session = field(default_factory=requests.Session)

    # ---------------------------------------------------------------- core

    def _get(self, path: str, params: dict[str, Any], billed: bool = True) -> Any:
        params = {k: v for k, v in params.items() if v is not None}
        params["apiKey"] = self.api_key
        url = f"{self.base_url}{path}"

        cached = self._cache_read(url, params)
        if cached is not None:
            log.debug("cache hit %s", path)
            return cached

        if billed:
            self._check_budget()

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(2 ** attempt)
                continue

            self.quota.update(resp.headers)

            if resp.status_code == 200:
                data = resp.json()
                self._cache_write(url, params, data)
                return data

            if resp.status_code == 401:
                raise OddsApiError("401 Unauthorized - the API key is wrong or revoked.")
            if resp.status_code == 422:
                # Usually an unknown market/sport key for this event. Not
                # retryable and not fatal to a scan.
                raise OddsApiError(f"422 from {path}: {resp.text[:300]}")
            if resp.status_code == 429:
                sleep = 2 ** attempt * 2
                log.warning("rate limited, sleeping %ss", sleep)
                time.sleep(sleep)
                continue
            if 500 <= resp.status_code < 600:
                time.sleep(2 ** attempt)
                last_exc = OddsApiError(f"{resp.status_code} from {path}")
                continue

            raise OddsApiError(f"{resp.status_code} from {path}: {resp.text[:300]}")

        raise OddsApiError(f"giving up on {path} after {self.max_retries} tries: {last_exc}")

    def _check_budget(self) -> None:
        if (
            self.max_credits_per_scan is not None
            and self.quota.spent_this_session >= self.max_credits_per_scan
        ):
            raise CreditBudgetExceeded(
                f"scan budget of {self.max_credits_per_scan} credits is spent "
                f"({self.quota.spent_this_session} used). Raise "
                "api.max_credits_per_scan or narrow the market list."
            )
        if (
            self.quota.remaining is not None
            and self.quota.remaining <= self.min_credits_remaining
        ):
            raise CreditBudgetExceeded(
                f"only {self.quota.remaining} credits left, which is at or below "
                f"the floor of {self.min_credits_remaining}. Stopping so you keep "
                "a reserve."
            )

    # --------------------------------------------------------------- cache

    def _cache_path(self, url: str, params: dict) -> Path:
        safe = {k: v for k, v in params.items() if k != "apiKey"}
        digest = hashlib.sha256(
            (url + json.dumps(safe, sort_keys=True)).encode()
        ).hexdigest()[:24]
        return self.cache_dir / f"{digest}.json"

    def _cache_read(self, url: str, params: dict) -> Any | None:
        if self.cache_seconds <= 0:
            return None
        p = self._cache_path(url, params)
        if not p.exists():
            return None
        if time.time() - p.stat().st_mtime > self.cache_seconds:
            return None
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def _cache_write(self, url: str, params: dict, data: Any) -> None:
        if self.cache_seconds <= 0:
            return
        p = self._cache_path(url, params)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data))

    # ------------------------------------------------------------ endpoints

    def sports(self, all_sports: bool = False) -> list[dict]:
        """Free. Every sport key the API currently serves."""
        return self._get("/sports", {"all": "true" if all_sports else None}, billed=False)

    def events(self, sport: str) -> list[dict]:
        """
        Free. Upcoming events for a sport: id, commence_time, teams.

        Always start a prop scan here rather than with /odds -- the event
        list costs nothing, and you need the ids anyway.
        """
        return self._get(f"/sports/{sport}/events", {}, billed=False)

    def odds(
        self,
        sport: str,
        markets: Sequence[str] = ("h2h",),
        bookmakers: Sequence[str] = ("pinnacle",),
        odds_format: str = "decimal",
    ) -> list[dict]:
        """
        Bulk game-level odds for a whole sport. Costs markets x 1 region
        equivalent. Use for h2h/spreads/totals; player props are not
        available here.
        """
        return self._get(
            f"/sports/{sport}/odds",
            {
                "markets": ",".join(markets),
                "bookmakers": ",".join(bookmakers),
                "oddsFormat": odds_format,
                "dateFormat": "iso",
            },
        )

    def event_odds(
        self,
        sport: str,
        event_id: str,
        markets: Sequence[str],
        bookmakers: Sequence[str],
        odds_format: str = "decimal",
    ) -> dict | None:
        """
        Player props for one event. Costs [markets returned] x 1.

        Returns None rather than raising when the event has no such markets,
        which is common and not an error worth aborting a scan over.
        """
        try:
            return self._get(
                f"/sports/{sport}/events/{event_id}/odds",
                {
                    "markets": ",".join(markets),
                    "bookmakers": ",".join(bookmakers),
                    "oddsFormat": odds_format,
                    "dateFormat": "iso",
                },
            )
        except CreditBudgetExceeded:
            raise
        except OddsApiError as exc:
            log.warning("no odds for event %s: %s", event_id, exc)
            return None

    # ------------------------------------------------------------- helpers

    def probe_quota(self) -> Quota:
        """
        Cheapest possible quota check: one free /sports call, whose response
        still carries the quota headers.
        """
        self.sports()
        return self.quota
