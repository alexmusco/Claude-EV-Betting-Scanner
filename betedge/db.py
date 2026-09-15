"""
SQLite store: scans, flagged opportunities, placed bets, closing lines.

Why log opportunities you did not bet
-------------------------------------
Results on bets you placed are a tiny, self-selected sample. Logging every
flagged opportunity lets you ask the question that actually matters: does
the model's EV estimate predict closing-line value? If your +3% bets close
at +3% on average, the model works and a losing month is variance. If they
close at 0%, the model is finding stale prices rather than mispriced ones,
and no amount of good luck will save it. That distinction takes hundreds of
settled bets to see in profit and loss, but only dozens to see in CLV.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .scan import Opportunity, ScanResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at        TEXT NOT NULL,
    finished_at       TEXT NOT NULL,
    sports            TEXT NOT NULL,
    events_scanned    INTEGER NOT NULL DEFAULT 0,
    quotes_seen       INTEGER NOT NULL DEFAULT 0,
    markets_paired    INTEGER NOT NULL DEFAULT 0,
    n_flagged         INTEGER NOT NULL DEFAULT 0,
    n_suspect         INTEGER NOT NULL DEFAULT 0,
    credits_spent     INTEGER NOT NULL DEFAULT 0,
    credits_remaining INTEGER,
    rejections        TEXT,
    errors            TEXT
);

CREATE TABLE IF NOT EXISTS opportunities (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id                 INTEGER NOT NULL REFERENCES scans(id),
    scanned_at              TEXT NOT NULL,
    sport                   TEXT NOT NULL,
    event_id                TEXT NOT NULL,
    commence_time           TEXT NOT NULL,
    home_team               TEXT,
    away_team               TEXT,
    market                  TEXT NOT NULL,
    selection               TEXT NOT NULL,
    line                    REAL,
    side                    TEXT NOT NULL,
    sharp_book              TEXT NOT NULL,
    sharp_price_taken_side  REAL NOT NULL,
    sharp_price_other_side  REAL NOT NULL,
    sharp_overround         REAL NOT NULL,
    fair_prob               REAL NOT NULL,
    fair_price              REAL NOT NULL,
    devig_method            TEXT NOT NULL,
    devig_spread            REAL NOT NULL,
    fair_prob_by_method     TEXT,
    soft_book               TEXT NOT NULL,
    soft_price              REAL NOT NULL,
    american_price          REAL,
    ev                      REAL NOT NULL,
    ev_min                  REAL,
    ev_max                  REAL,
    kelly_fraction          REAL,
    recommended_stake       REAL,
    sharp_last_update       TEXT,
    soft_last_update        TEXT,
    suspect                 INTEGER NOT NULL DEFAULT 0,
    flags                   TEXT
);

CREATE INDEX IF NOT EXISTS idx_opp_event   ON opportunities(event_id);
CREATE INDEX IF NOT EXISTS idx_opp_scan    ON opportunities(scan_id);
CREATE INDEX IF NOT EXISTS idx_opp_time    ON opportunities(commence_time);

CREATE TABLE IF NOT EXISTS bets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    opportunity_id  INTEGER REFERENCES opportunities(id),
    placed_at       TEXT NOT NULL,
    sport           TEXT,
    event_id        TEXT,
    commence_time   TEXT,
    matchup         TEXT,
    market          TEXT,
    selection       TEXT,
    line            REAL,
    side            TEXT,
    book            TEXT NOT NULL,
    price           REAL NOT NULL,
    stake           REAL NOT NULL,
    ev_at_bet       REAL,
    fair_prob_at_bet REAL,
    status          TEXT NOT NULL DEFAULT 'pending',
    settled_at      TEXT,
    pnl             REAL,
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_bets_status ON bets(status);

CREATE TABLE IF NOT EXISTS closing_lines (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    opportunity_id      INTEGER REFERENCES opportunities(id),
    bet_id              INTEGER REFERENCES bets(id),
    captured_at         TEXT NOT NULL,
    sharp_price_taken   REAL,
    sharp_price_other   REAL,
    fair_prob_close     REAL,
    soft_price_close    REAL,
    clv_ev              REAL,   -- EV of your price against the closing fair prob
    clv_prob_points     REAL,   -- closing fair prob minus fair prob at bet time
    clv_price_pct       REAL    -- how much better your price was than the close
);
"""

VALID_STATUS = {"pending", "won", "lost", "push", "void", "half_won", "half_lost"}


def parse_timestamp(value) -> datetime | None:
    """Tolerant ISO parse. Timestamps arrive as "...Z" from the API and as
    "...+00:00" from datetime.isoformat; both must round-trip."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def tx(self):
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ------------------------------------------------------------- writing

    def record_scan(self, result: ScanResult) -> int:
        with self.tx() as c:
            cur = c.execute(
                """INSERT INTO scans (started_at, finished_at, sports, events_scanned,
                       quotes_seen, markets_paired, n_flagged, n_suspect,
                       credits_spent, credits_remaining, rejections, errors)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    result.started_at.isoformat(),
                    result.finished_at.isoformat(),
                    ",".join(result.sports),
                    result.events_scanned,
                    result.quotes_seen,
                    result.sharp_markets_paired,
                    len(result.opportunities),
                    len(result.suspect),
                    result.credits_spent,
                    result.credits_remaining,
                    "; ".join(f"{k}={v}" for k, v in sorted(result.rejections.items())),
                    " | ".join(result.errors) or None,
                ),
            )
            scan_id = cur.lastrowid
            for opp in result.opportunities:
                self._insert_opportunity(c, scan_id, opp)
        return scan_id

    @staticmethod
    def _insert_opportunity(c: sqlite3.Connection, scan_id: int, opp: Opportunity) -> int:
        row = opp.to_row()
        cur = c.execute(
            """INSERT INTO opportunities (
                   scan_id, scanned_at, sport, event_id, commence_time, home_team,
                   away_team, market, selection, line, side, sharp_book,
                   sharp_price_taken_side, sharp_price_other_side, sharp_overround,
                   fair_prob, fair_price, devig_method, devig_spread,
                   fair_prob_by_method, soft_book, soft_price, american_price, ev,
                   ev_min, ev_max, kelly_fraction, recommended_stake,
                   sharp_last_update, soft_last_update, suspect, flags)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                scan_id, row["scanned_at"], row["sport"], row["event_id"],
                row["commence_time"], row["home_team"], row["away_team"],
                row["market"], row["selection"], row["line"], row["side"],
                row["sharp_book"], row["sharp_price_taken_side"],
                row["sharp_price_other_side"], row["sharp_overround"],
                row["fair_prob"], row["fair_price"], row["devig_method"],
                row["devig_spread"], row["fair_prob_by_method"], row["soft_book"],
                row["soft_price"], row["american_price"], row["ev"], row["ev_min"],
                row["ev_max"], row["kelly_fraction"], row["recommended_stake"],
                row["sharp_last_update"], row["soft_last_update"],
                int(row["suspect"]), row["flags"],
            ),
        )
        return cur.lastrowid

    def place_bet(
        self,
        opportunity_id: int | None = None,
        *,
        stake: float,
        price: float | None = None,
        book: str | None = None,
        notes: str | None = None,
        placed_at: datetime | None = None,
        **manual: Any,
    ) -> int:
        """
        Log a bet. Either point at a flagged opportunity (everything is
        copied from it) or pass the fields directly for a bet the scanner
        did not surface.
        """
        placed_at = placed_at or datetime.now(timezone.utc)
        fields = dict(manual)

        if opportunity_id is not None:
            opp = self.get_opportunity(opportunity_id)
            if opp is None:
                raise ValueError(f"no opportunity with id {opportunity_id}")
            fields.setdefault("sport", opp["sport"])
            fields.setdefault("event_id", opp["event_id"])
            fields.setdefault("commence_time", opp["commence_time"])
            fields.setdefault("matchup", f"{opp['away_team']} @ {opp['home_team']}")
            fields.setdefault("market", opp["market"])
            fields.setdefault("selection", opp["selection"])
            fields.setdefault("line", opp["line"])
            fields.setdefault("side", opp["side"])
            book = book or opp["soft_book"]
            price = price if price is not None else opp["soft_price"]
            # EV is recomputed at the price you actually got, which is often
            # worse than the price that was showing when the scan ran.
            fields.setdefault("fair_prob_at_bet", opp["fair_prob"])
            fields.setdefault("ev_at_bet", opp["fair_prob"] * price - 1.0)

        if price is None or book is None:
            raise ValueError("a bet needs at least a book and a price")

        with self.tx() as c:
            cur = c.execute(
                """INSERT INTO bets (opportunity_id, placed_at, sport, event_id,
                       commence_time, matchup, market, selection, line, side, book,
                       price, stake, ev_at_bet, fair_prob_at_bet, status, notes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'pending', ?)""",
                (
                    opportunity_id, placed_at.isoformat(), fields.get("sport"),
                    fields.get("event_id"), fields.get("commence_time"),
                    fields.get("matchup"), fields.get("market"),
                    fields.get("selection"), fields.get("line"), fields.get("side"),
                    book, price, stake, fields.get("ev_at_bet"),
                    fields.get("fair_prob_at_bet"), notes,
                ),
            )
            return cur.lastrowid

    def settle_bet(
        self, bet_id: int, status: str, settled_at: datetime | None = None
    ) -> float:
        """Mark a bet won/lost/push/void and compute its profit or loss."""
        status = status.lower()
        if status not in VALID_STATUS:
            raise ValueError(f"status must be one of {sorted(VALID_STATUS)}")
        bet = self.get_bet(bet_id)
        if bet is None:
            raise ValueError(f"no bet with id {bet_id}")

        stake, price = bet["stake"], bet["price"]
        pnl = {
            "won": stake * (price - 1.0),
            "lost": -stake,
            "push": 0.0,
            "void": 0.0,
            "half_won": stake * (price - 1.0) / 2.0,
            "half_lost": -stake / 2.0,
            "pending": None,
        }[status]

        with self.tx() as c:
            c.execute(
                "UPDATE bets SET status=?, settled_at=?, pnl=? WHERE id=?",
                (
                    status,
                    (settled_at or datetime.now(timezone.utc)).isoformat(),
                    pnl,
                    bet_id,
                ),
            )
        return pnl if pnl is not None else 0.0

    def record_closing_line(
        self,
        *,
        opportunity_id: int | None,
        bet_id: int | None,
        sharp_price_taken: float,
        sharp_price_other: float,
        fair_prob_close: float,
        price_taken: float,
        fair_prob_at_bet: float | None = None,
        soft_price_close: float | None = None,
        captured_at: datetime | None = None,
    ) -> int:
        clv_ev = fair_prob_close * price_taken - 1.0
        clv_prob_points = (
            fair_prob_close - fair_prob_at_bet if fair_prob_at_bet is not None else None
        )
        clv_price_pct = (
            price_taken / (1.0 / fair_prob_close) - 1.0 if fair_prob_close > 0 else None
        )
        with self.tx() as c:
            cur = c.execute(
                """INSERT INTO closing_lines (opportunity_id, bet_id, captured_at,
                       sharp_price_taken, sharp_price_other, fair_prob_close,
                       soft_price_close, clv_ev, clv_prob_points, clv_price_pct)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    opportunity_id, bet_id,
                    (captured_at or datetime.now(timezone.utc)).isoformat(),
                    sharp_price_taken, sharp_price_other, fair_prob_close,
                    soft_price_close, clv_ev, clv_prob_points, clv_price_pct,
                ),
            )
            return cur.lastrowid

    # ------------------------------------------------------------- reading

    def get_opportunity(self, opp_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM opportunities WHERE id=?", (opp_id,)
        ).fetchone()

    def get_bet(self, bet_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM bets WHERE id=?", (bet_id,)).fetchone()

    def latest_scan_id(self) -> int | None:
        row = self.conn.execute("SELECT MAX(id) AS id FROM scans").fetchone()
        return row["id"] if row else None

    def opportunities_for_scan(self, scan_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM opportunities WHERE scan_id=? ORDER BY ev DESC", (scan_id,)
        ).fetchall()

    def open_bets(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM bets WHERE status='pending' ORDER BY commence_time"
        ).fetchall()

    def all_bets(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM bets ORDER BY placed_at").fetchall()

    def pending_closing_capture(self, now: datetime | None = None) -> list[sqlite3.Row]:
        """
        Bets whose event has started but which have no closing line yet.

        Filtering happens in Python rather than SQL because timestamps reach
        the database in more than one ISO spelling ("...Z" from the API,
        "...+00:00" from datetime.isoformat) and those do not compare
        correctly as strings.
        """
        now = now or datetime.now(timezone.utc)
        rows = self.conn.execute(
            """SELECT b.* FROM bets b
               LEFT JOIN closing_lines c ON c.bet_id = b.id
               WHERE c.id IS NULL AND b.commence_time IS NOT NULL"""
        ).fetchall()
        out = []
        for r in rows:
            ts = parse_timestamp(r["commence_time"])
            if ts is not None and ts <= now:
                out.append(r)
        return out

    # ----------------------------------------------------------- analytics

    def summary(self) -> dict[str, Any]:
        """Headline performance numbers over all settled bets."""
        rows = self.conn.execute(
            "SELECT * FROM bets WHERE status NOT IN ('pending')"
        ).fetchall()
        settled = [r for r in rows if r["status"] not in ("void",)]
        staked = sum(r["stake"] for r in settled)
        pnl = sum(r["pnl"] or 0.0 for r in settled)
        decided = [r for r in settled if r["status"] in ("won", "lost")]
        wins = [r for r in decided if r["status"] == "won"]

        clv_rows = self.conn.execute(
            "SELECT clv_ev FROM closing_lines WHERE clv_ev IS NOT NULL"
        ).fetchall()
        clv = [r["clv_ev"] for r in clv_rows]

        pending = self.conn.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(stake),0) s FROM bets WHERE status='pending'"
        ).fetchone()

        ev_rows = self.conn.execute(
            "SELECT ev_at_bet, stake FROM bets WHERE ev_at_bet IS NOT NULL"
        ).fetchall()
        expected = sum((r["ev_at_bet"] or 0) * r["stake"] for r in ev_rows)

        return {
            "bets_settled": len(settled),
            "bets_pending": pending["n"],
            "stake_pending": pending["s"],
            "total_staked": staked,
            "total_pnl": pnl,
            "roi": (pnl / staked) if staked else None,
            "win_rate": (len(wins) / len(decided)) if decided else None,
            "expected_pnl": expected,
            "pnl_vs_expected": pnl - expected if ev_rows else None,
            "avg_clv": (sum(clv) / len(clv)) if clv else None,
            "clv_positive_rate": (
                sum(1 for x in clv if x > 0) / len(clv) if clv else None
            ),
            "clv_sample": len(clv),
        }

    def breakdown(self, column: str) -> list[dict[str, Any]]:
        """Settled performance grouped by any bet column (sport, market, book)."""
        allowed = {"sport", "market", "book", "side", "selection"}
        if column not in allowed:
            raise ValueError(f"column must be one of {sorted(allowed)}")
        rows = self.conn.execute(
            f"""SELECT {column} AS k,
                       COUNT(*) AS n,
                       SUM(stake) AS staked,
                       SUM(COALESCE(pnl,0)) AS pnl,
                       AVG(ev_at_bet) AS avg_ev
                FROM bets WHERE status NOT IN ('pending','void')
                GROUP BY {column} ORDER BY pnl DESC"""
        ).fetchall()
        return [
            {
                "key": r["k"],
                "n": r["n"],
                "staked": r["staked"],
                "pnl": r["pnl"],
                "roi": (r["pnl"] / r["staked"]) if r["staked"] else None,
                "avg_ev": r["avg_ev"],
            }
            for r in rows
        ]
