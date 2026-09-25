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
from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

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
    market_tier             TEXT,
    liquidity               REAL,
    required_ev             REAL,
    edge_score              REAL,
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

-- Every billed API call, so the budget planner can answer "what did I
-- spend today?". The provider's x-requests-remaining header is the ground
-- truth for what is LEFT; this is the local record of where it went.
CREATE TABLE IF NOT EXISTS credit_spend (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    at          TEXT NOT NULL,
    command     TEXT,
    detail      TEXT,
    credits     INTEGER NOT NULL,
    remaining   INTEGER
);

CREATE INDEX IF NOT EXISTS idx_spend_at ON credit_spend(at);

-- ---------------------------------------------------------------------
-- MULTI-LEG TICKETS
--
-- Mirrors opportunities/bets deliberately. `parlay_tickets` holds every
-- ticket the optimizer GENERATED, bet or not, for the same reason
-- `opportunities` holds every flagged single: results on the tickets you
-- actually entered are a tiny self-selected sample, and the question
-- worth answering is whether the modelled EV predicts anything at all.
-- That needs the ones you passed on as much as the ones you took.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS parlay_tickets (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id                 INTEGER REFERENCES scans(id),
    created_at              TEXT NOT NULL,
    product                 TEXT NOT NULL,
    book                    TEXT,
    kind                    TEXT,
    n_legs                  INTEGER NOT NULL,
    sports                  TEXT,
    event_ids               TEXT,
    commence_time           TEXT,
    same_game               INTEGER NOT NULL DEFAULT 1,
    joint_prob              REAL NOT NULL,
    joint_prob_se           REAL,
    joint_prob_independent  REAL,
    hit_distribution        TEXT,
    ev                      REAL NOT NULL,
    ev_se                   REAL,
    ev_independent          REAL,
    payout_all_hit          REAL,
    variance                REAL,
    ev_per_variance         REAL,
    kelly_fraction          REAL,
    log_optimal_fraction    REAL,
    recommended_stake       REAL,
    correlation_summary     TEXT,
    correlation_all_prior   INTEGER NOT NULL DEFAULT 0,
    draws                   INTEGER,
    suspect                 INTEGER NOT NULL DEFAULT 0,
    flags                   TEXT
);

CREATE INDEX IF NOT EXISTS idx_pt_scan ON parlay_tickets(scan_id);
CREATE INDEX IF NOT EXISTS idx_pt_time ON parlay_tickets(commence_time);

CREATE TABLE IF NOT EXISTS parlay_legs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id           INTEGER NOT NULL REFERENCES parlay_tickets(id),
    leg_index           INTEGER NOT NULL,
    sport               TEXT,
    event_id            TEXT,
    commence_time       TEXT,
    matchup             TEXT,
    market              TEXT NOT NULL,
    selection           TEXT NOT NULL,
    side                TEXT NOT NULL,
    line                REAL,
    team                TEXT,
    book                TEXT,
    book_price          REAL,
    fair_prob           REAL NOT NULL,
    push_prob           REAL NOT NULL DEFAULT 0,
    hit_prob            REAL NOT NULL,
    sharp_price_taken   REAL,
    sharp_price_other   REAL,
    sharp_overround     REAL,
    devig_spread        REAL,
    sharp_line          REAL,
    line_source         TEXT,
    push_source         TEXT,
    market_tier         TEXT,
    liquidity           REAL,
    flags               TEXT,
    -- Closing-line capture writes these back. A ticket's CLV is only
    -- meaningful once every leg has one, which is why they live on the leg.
    fair_prob_close     REAL,
    push_prob_close     REAL,
    closed_at           TEXT
);

CREATE INDEX IF NOT EXISTS idx_pl_ticket ON parlay_legs(ticket_id);
CREATE INDEX IF NOT EXISTS idx_pl_event  ON parlay_legs(event_id);

CREATE TABLE IF NOT EXISTS parlay_bets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id       INTEGER REFERENCES parlay_tickets(id),
    placed_at       TEXT NOT NULL,
    book            TEXT NOT NULL,
    product         TEXT NOT NULL,
    n_legs          INTEGER NOT NULL,
    stake           REAL NOT NULL,
    ev_at_bet       REAL,
    joint_prob_at_bet REAL,
    payout_all_hit  REAL,
    status          TEXT NOT NULL DEFAULT 'pending',
    legs_hit        INTEGER,
    legs_void       INTEGER,
    settled_at      TEXT,
    pnl             REAL,
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_pb_status ON parlay_bets(status);

-- Ticket-level closing line: every leg re-priced at the close and the
-- joint probability recomputed through the same copula. Ticket CLV is the
-- only fast read on whether any of this works -- profit and loss on
-- multi-leg tickets is so noisy that a hundred of them tell you nothing.
CREATE TABLE IF NOT EXISTS parlay_closing_lines (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id               INTEGER REFERENCES parlay_tickets(id),
    parlay_bet_id           INTEGER REFERENCES parlay_bets(id),
    captured_at             TEXT NOT NULL,
    legs_captured           INTEGER NOT NULL,
    n_legs                  INTEGER NOT NULL,
    joint_prob_close        REAL,
    joint_prob_at_bet       REAL,
    ev_close                REAL,
    ev_at_bet               REAL,
    clv_ev                  REAL,   -- EV of the entry against the closing joint prob
    clv_prob_points         REAL    -- closing joint prob minus the modelled one
);

CREATE INDEX IF NOT EXISTS idx_pcl_ticket ON parlay_closing_lines(ticket_id);

-- Who plays for whom. Populated from the game logs already being fitted,
-- from a published roster feed, and from the user's own override file.
--
-- Rows are stored as a SNAPSHOT per (sport, source) and replaced wholesale
-- on refresh, never accumulated. A traded player who left a row behind on
-- his old club would read as one name on two teams, which the resolver
-- treats as two different players sharing a name and refuses to answer for
-- -- turning a correct update into a silent loss of coverage.
CREATE TABLE IF NOT EXISTS rosters (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sport       TEXT NOT NULL,
    player      TEXT NOT NULL,
    player_key  TEXT NOT NULL,
    exact_key   TEXT NOT NULL,
    team        TEXT NOT NULL,
    position    TEXT,
    source      TEXT NOT NULL,
    as_of       TEXT,
    fetched_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_roster_key    ON rosters(sport, player_key);
CREATE INDEX IF NOT EXISTS idx_roster_source ON rosters(sport, source);

-- Pairwise correlations fitted from game logs the user supplied. The
-- sample size is stored because it is what decides whether the number is
-- used at all -- an estimate from 12 joint observations is not an
-- estimate, and the report must be able to say which is which.
CREATE TABLE IF NOT EXISTS correlation_estimates (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    sport           TEXT NOT NULL,
    market_a        TEXT NOT NULL,
    market_b        TEXT NOT NULL,
    relation        TEXT NOT NULL,
    rho             REAL NOT NULL,
    spearman        REAL,
    n_observations  INTEGER NOT NULL,
    n_games         INTEGER,
    fitted_at       TEXT NOT NULL,
    note            TEXT,
    UNIQUE(sport, market_a, market_b, relation)
);

-- What has already been sent to a phone. The whole point of a scheduled
-- scan is that each run is a fresh process with no recollection of the
-- last, so "have I already said this?" has to be answered from disk.
CREATE TABLE IF NOT EXISTS notifications (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint     TEXT NOT NULL,
    sent_at         TEXT NOT NULL,
    provider        TEXT,
    ev              REAL,
    price           REAL,
    opportunity_id  INTEGER REFERENCES opportunities(id),
    title           TEXT,
    ok              INTEGER NOT NULL DEFAULT 1,
    error           TEXT
);

CREATE INDEX IF NOT EXISTS idx_notifications_fp
    ON notifications(fingerprint, sent_at);

-- Every prop line this tool has ever seen, with the de-vigged fair
-- probability of THAT EXACT LINE at that moment.
--
-- Why a time series and not just the latest
-- -----------------------------------------
-- A pick'em book posts a projection and leaves it. The sharp market
-- reprices continuously. So the edge on a pick'em pick is not a property
-- of the line -- it is a property of the line AND THE CLOCK: the same
-- "Deebo Samuel Over 3.5" is a coin flip at noon and a gift twenty
-- minutes after his team-mate is ruled out.
--
-- A scan that looks once can only ever see the coin flip. It has no way
-- to know whether it is early or late, because it has nothing to compare
-- against. This table is that comparison.
--
-- `fair_prob` is the whole point. It is the de-vigged probability of the
-- PICK'EM BOOK'S line, not the sharp book's -- so a move in this column
-- with `line` unchanged means exactly one thing: the market repriced and
-- the pick'em book has not yet followed. That is the entire signal.
--
-- Recording is free. The scan already fetches all of this and throws it
-- away; the only cost is disk.
CREATE TABLE IF NOT EXISTS prop_lines (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    observed_at     TEXT NOT NULL,
    sport           TEXT NOT NULL,
    event_id        TEXT NOT NULL,
    commence_time   TEXT,
    market          TEXT NOT NULL,
    selection       TEXT NOT NULL,
    side            TEXT NOT NULL,
    book            TEXT NOT NULL,
    line            REAL,
    book_price      REAL,
    fair_prob       REAL NOT NULL,
    push_prob       REAL,
    sharp_line      REAL,
    sharp_price     REAL,
    line_source     TEXT
);

CREATE INDEX IF NOT EXISTS idx_prop_lines_key
    ON prop_lines(sport, event_id, market, selection, side, book, line);
CREATE INDEX IF NOT EXISTS idx_prop_lines_time ON prop_lines(observed_at);

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


@dataclass(frozen=True)
class LineMove:
    """
    A pick'em line that stood still while the market walked away from it.

    `drift` is the change in the de-vigged probability of THE BOOK'S OWN
    LINE. Positive means the posted side got more likely since it was
    first seen -- the value is on that side. Negative means it got less
    likely, and the value is on the other one. Both are edges; only the
    direction of the bet differs, which is why this refuses to report an
    absolute number and make you work out the sign yourself.
    """

    sport: str
    event_id: str
    market: str
    selection: str
    side: str
    book: str
    line: float | None
    first_seen: datetime | None
    last_seen: datetime | None
    fair_first: float
    fair_last: float
    sharp_line_first: float | None
    sharp_line_last: float | None
    commence_time: str | None = None
    observations: int = 2

    @property
    def drift(self) -> float:
        return self.fair_last - self.fair_first

    @property
    def value_side(self) -> str:
        """The side to actually take, which is not always the posted one."""
        if self.drift >= 0:
            return self.side
        return {"over": "Under", "under": "Over"}.get(
            (self.side or "").lower(), f"not {self.side}")

    @property
    def sharp_line_moved(self) -> bool:
        """
        Whether the SHARP book moved its number too.

        The strongest form of the signal. A fair probability drifting on
        a static sharp line can be ordinary price noise; the sharp book
        physically relocating its line while the pick'em book sits still
        is news that one venue has priced and the other has not.
        """
        if self.sharp_line_first is None or self.sharp_line_last is None:
            return False
        return abs(self.sharp_line_last - self.sharp_line_first) >= 0.5

    @property
    def minutes(self) -> float | None:
        if not self.first_seen or not self.last_seen:
            return None
        return (self.last_seen - self.first_seen).total_seconds() / 60.0

    def describe(self) -> str:
        line = "" if self.line is None else f" {self.line:g}"
        return f"{self.selection} {self.side}{line}"


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        # Migration runs BEFORE the schema script. SCHEMA creates indexes
        # over columns that an older database may not have yet, and CREATE
        # INDEX on a missing column is a hard error -- so the columns have to
        # exist first. On a new database _migrate finds no tables and does
        # nothing, and the schema script builds everything.
        self._migrate()
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def _migrate(self) -> None:
        """
        Add columns introduced after a database was first created.

        CREATE TABLE IF NOT EXISTS silently does nothing to an existing
        table, so a database carrying real bet history would otherwise be
        missing every column added since. Each ALTER is additive and
        nullable, so old rows stay valid and no data is rewritten.

        PRAGMA table_info on a table that does not exist returns no rows,
        which is how a brand-new database is recognised and skipped -- an
        empty result means "absent", not "has no columns".
        """
        wanted = {
            "opportunities": {
                "market_tier": "TEXT",
                "liquidity": "REAL",
                "required_ev": "REAL",
                "edge_score": "REAL",
            },
        }
        for table, columns in wanted.items():
            have = {
                r["name"]
                for r in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if not have:
                # No rows means the table does not exist yet, not that it has
                # no columns. The schema script below will create it complete.
                continue
            for name, decl in columns.items():
                if name not in have:
                    self.conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {name} {decl}"
                    )

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
                # Stamp the row id back onto the object so the report can
                # print the number the `bet` command actually takes.
                opp.db_id = self._insert_opportunity(c, scan_id, opp)
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
                   market_tier, liquidity, required_ev, edge_score,
                   sharp_last_update, soft_last_update, suspect, flags)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                row["market_tier"], row["liquidity"], row["required_ev"],
                row["edge_score"],
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
                # Say what the likely cause is. The number people type
                # here is copied off a scan table, and that table used to
                # print a row NUMBER when a scan was not persisted --
                # indistinguishable from an id until this line rejected
                # it, long after the output had scrolled away.
                newest = self.conn.execute(
                    "SELECT MAX(id) FROM opportunities"
                ).fetchone()[0]
                hint = (f" The highest id on record is {newest}."
                        if newest else " No opportunities have been saved yet.")
                raise ValueError(
                    f"no opportunity with id {opportunity_id}.{hint} If you "
                    "copied it from a scan's # column, check it was a real "
                    "id and not a parenthesised row number -- `betedge show "
                    "--ids` lists the saved ones. To log it anyway, pass the "
                    "details directly: `betedge bet --stake N --price P "
                    "--book B --selection ... --side ... --line ...`."
                )
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

    # ------------------------------------------------- multi-leg tickets

    def record_parlay_tickets(
        self, tickets, scan_id: int | None = None, draws: int | None = None
    ) -> list[int]:
        """
        Store every generated ticket and its legs.

        Tickets that will never be bet are stored too. That is the whole
        point: the modelled EV of the ones you passed on is testable
        against their closing lines exactly as the ones you took are, and
        without them the only sample is the one you selected.
        """
        ids: list[int] = []
        with self.tx() as c:
            for ticket in tickets:
                row = ticket.to_row()
                row["draws"] = draws
                cur = c.execute(
                    """INSERT INTO parlay_tickets (
                           scan_id, created_at, product, book, kind, n_legs,
                           sports, event_ids, commence_time, same_game,
                           joint_prob, joint_prob_se, joint_prob_independent,
                           hit_distribution, ev, ev_se, ev_independent,
                           payout_all_hit, variance, ev_per_variance,
                           kelly_fraction, log_optimal_fraction,
                           recommended_stake, correlation_summary,
                           correlation_all_prior, draws, suspect, flags)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        scan_id, row["created_at"], row["product"], row["book"],
                        row["kind"], row["n_legs"], row["sports"], row["event_ids"],
                        row["commence_time"], row["same_game"], row["joint_prob"],
                        row["joint_prob_se"], row["joint_prob_independent"],
                        row["hit_distribution"], row["ev"], row["ev_se"],
                        row["ev_independent"], row["payout_all_hit"], row["variance"],
                        row["ev_per_variance"], row["kelly_fraction"],
                        row["log_optimal_fraction"], row["recommended_stake"],
                        row["correlation_summary"], row["correlation_all_prior"],
                        row["draws"], row["suspect"], row["flags"],
                    ),
                )
                ticket_id = cur.lastrowid
                ticket.db_id = ticket_id
                ids.append(ticket_id)
                for index, leg in enumerate(ticket.legs):
                    c.execute(
                        """INSERT INTO parlay_legs (
                               ticket_id, leg_index, sport, event_id, commence_time,
                               matchup, market, selection, side, line, team, book,
                               book_price, fair_prob, push_prob, hit_prob,
                               sharp_price_taken, sharp_price_other, sharp_overround,
                               devig_spread, sharp_line, line_source, push_source,
                               market_tier, liquidity, flags)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            ticket_id, index, leg.sport, leg.event_id,
                            leg.commence_time.isoformat(), leg.matchup, leg.market,
                            leg.selection, leg.side, leg.line, leg.team, leg.book,
                            leg.book_price, leg.fair_prob, leg.push_prob, leg.hit_prob,
                            leg.sharp_price_taken, leg.sharp_price_other,
                            leg.sharp_overround, leg.devig_spread, leg.sharp_line,
                            leg.line_source, leg.push_source, leg.market_tier,
                            leg.liquidity, ",".join(leg.flags),
                        ),
                    )
        return ids

    def place_parlay_bet(
        self,
        ticket_id: int,
        *,
        stake: float,
        book: str | None = None,
        notes: str | None = None,
        placed_at: datetime | None = None,
    ) -> int:
        """Log an entry you actually placed against a generated ticket."""
        ticket = self.get_parlay_ticket(ticket_id)
        if ticket is None:
            raise ValueError(f"no parlay ticket with id {ticket_id}")
        with self.tx() as c:
            cur = c.execute(
                """INSERT INTO parlay_bets (ticket_id, placed_at, book, product,
                       n_legs, stake, ev_at_bet, joint_prob_at_bet,
                       payout_all_hit, status, notes)
                   VALUES (?,?,?,?,?,?,?,?,?, 'pending', ?)""",
                (
                    ticket_id,
                    (placed_at or datetime.now(timezone.utc)).isoformat(),
                    book or ticket["book"], ticket["product"], ticket["n_legs"],
                    stake, ticket["ev"], ticket["joint_prob"],
                    ticket["payout_all_hit"], notes,
                ),
            )
            return cur.lastrowid

    def settle_parlay_bet(
        self,
        bet_id: int,
        legs_hit: int,
        legs_void: int = 0,
        settled_at: datetime | None = None,
        payouts_path: str | None = None,
    ) -> float:
        """
        Settle an entry from how many legs actually landed.

        A multi-leg entry has no won/lost: it has a number of legs that
        hit, which the payout structure turns into a return. So settlement
        takes the count and re-reads the structure, rather than asking the
        user to work out what they were paid.
        """
        from .parlay import PayoutTable

        bet = self.get_parlay_bet(bet_id)
        if bet is None:
            raise ValueError(f"no parlay bet with id {bet_id}")
        n_legs = bet["n_legs"]
        if not 0 <= legs_hit <= n_legs:
            raise ValueError(f"legs_hit must be between 0 and {n_legs}")
        if not 0 <= legs_void <= n_legs or legs_hit + legs_void > n_legs:
            raise ValueError("legs_hit and legs_void cannot exceed the leg count")

        # The user's own table, if they configured one: settling against
        # the shipped ladder when they edited theirs would book the wrong
        # profit on every entry.
        table = PayoutTable.load(payouts_path)
        product = table.products.get(bet["product"])
        if product is not None:
            multiple = float(product.multiple_grid(n_legs)[legs_void, legs_hit])
        elif legs_hit + legs_void == n_legs:
            # A priced parlay is not in the table; all legs home pays the
            # price that was recorded when the entry was logged.
            multiple = float(bet["payout_all_hit"] or 0.0)
        else:
            multiple = 0.0

        pnl = bet["stake"] * (multiple - 1.0)
        status = "won" if multiple > 1.0 else ("push" if multiple == 1.0 else "lost")
        with self.tx() as c:
            c.execute(
                """UPDATE parlay_bets SET status=?, legs_hit=?, legs_void=?,
                       settled_at=?, pnl=? WHERE id=?""",
                (
                    status, legs_hit, legs_void,
                    (settled_at or datetime.now(timezone.utc)).isoformat(),
                    pnl, bet_id,
                ),
            )
        return pnl

    def record_parlay_leg_close(
        self,
        leg_id: int,
        fair_prob_close: float,
        push_prob_close: float = 0.0,
        captured_at: datetime | None = None,
    ) -> None:
        with self.tx() as c:
            c.execute(
                """UPDATE parlay_legs SET fair_prob_close=?, push_prob_close=?,
                       closed_at=? WHERE id=?""",
                (
                    fair_prob_close, push_prob_close,
                    (captured_at or datetime.now(timezone.utc)).isoformat(),
                    leg_id,
                ),
            )

    def record_parlay_closing_line(
        self,
        *,
        ticket_id: int,
        parlay_bet_id: int | None,
        legs_captured: int,
        n_legs: int,
        joint_prob_close: float | None,
        joint_prob_at_bet: float | None,
        ev_close: float | None,
        ev_at_bet: float | None,
        captured_at: datetime | None = None,
    ) -> int:
        clv_ev = ev_close
        clv_prob_points = (
            joint_prob_close - joint_prob_at_bet
            if joint_prob_close is not None and joint_prob_at_bet is not None
            else None
        )
        with self.tx() as c:
            cur = c.execute(
                """INSERT INTO parlay_closing_lines (ticket_id, parlay_bet_id,
                       captured_at, legs_captured, n_legs, joint_prob_close,
                       joint_prob_at_bet, ev_close, ev_at_bet, clv_ev,
                       clv_prob_points)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    ticket_id, parlay_bet_id,
                    (captured_at or datetime.now(timezone.utc)).isoformat(),
                    legs_captured, n_legs, joint_prob_close, joint_prob_at_bet,
                    ev_close, ev_at_bet, clv_ev, clv_prob_points,
                ),
            )
            return cur.lastrowid

    def save_correlation_estimates(self, estimates) -> int:
        """
        Upsert fitted correlations. A refit replaces the previous number
        for that (sport, market pair, relation) rather than accumulating
        rows, so a lookup never has to choose between two answers.
        """
        written = 0
        with self.tx() as c:
            for e in estimates:
                c.execute(
                    """INSERT INTO correlation_estimates (sport, market_a, market_b,
                           relation, rho, spearman, n_observations, n_games,
                           fitted_at, note)
                       VALUES (?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(sport, market_a, market_b, relation) DO UPDATE SET
                           rho=excluded.rho, spearman=excluded.spearman,
                           n_observations=excluded.n_observations,
                           n_games=excluded.n_games, fitted_at=excluded.fitted_at,
                           note=excluded.note""",
                    (
                        e.sport, e.market_a, e.market_b, e.relation, e.rho,
                        e.spearman, e.n_observations, e.n_games,
                        (e.fitted_at or datetime.now(timezone.utc)).isoformat(),
                        e.note,
                    ),
                )
                written += 1
        return written

    # -------------------------------------------------------------- rosters

    def replace_rosters(self, sport: str, source: str, entries) -> int:
        """
        Swap in a fresh snapshot for one (sport, source), atomically.

        Delete-then-insert rather than upsert, because a roster feed is a
        statement about the whole league at a moment, not a stream of
        corrections. Upserting would leave a traded player on both clubs.
        """
        rows = list(entries)
        with self.tx() as c:
            c.execute(
                "DELETE FROM rosters WHERE sport=? AND source=?", (sport, source)
            )
            c.executemany(
                """INSERT INTO rosters (sport, player, player_key, exact_key,
                       team, position, source, as_of, fetched_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        e.sport or sport, e.player, e.player_key, e.exact_key,
                        e.team, e.position, e.source,
                        e.as_of.isoformat() if e.as_of else None,
                        (e.fetched_at or datetime.now(timezone.utc)).isoformat(),
                    )
                    for e in rows
                ],
            )
        return len(rows)

    def roster_rows(self, sport: str | None = None) -> list[sqlite3.Row]:
        if sport:
            return self.conn.execute(
                "SELECT * FROM rosters WHERE sport=?", (sport,)
            ).fetchall()
        return self.conn.execute("SELECT * FROM rosters").fetchall()

    def roster_freshness(self) -> dict[tuple[str, str], dict[str, Any]]:
        """
        Per (sport, source): how many players, and when it was last pulled.

        This is what the refresh timer reads. It asks when the snapshot was
        FETCHED, not how old the players are, because those are different
        questions -- a feed pulled an hour ago is current even if it is
        reporting a roster that has not changed in a month.
        """
        out: dict[tuple[str, str], dict[str, Any]] = {}
        for r in self.conn.execute(
            """SELECT sport, source, COUNT(*) AS n, MAX(fetched_at) AS last
               FROM rosters GROUP BY sport, source"""
        ).fetchall():
            out[(r["sport"], r["source"])] = {
                "players": r["n"],
                "fetched_at": parse_timestamp(r["last"]),
            }
        return out

    def correlation_estimates(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM correlation_estimates ORDER BY sport, market_a, market_b"
        ).fetchall()

    def get_parlay_ticket(self, ticket_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM parlay_tickets WHERE id=?", (ticket_id,)
        ).fetchone()

    def get_parlay_bet(self, bet_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM parlay_bets WHERE id=?", (bet_id,)
        ).fetchone()

    def parlay_legs(self, ticket_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM parlay_legs WHERE ticket_id=? ORDER BY leg_index",
            (ticket_id,),
        ).fetchall()

    def latest_parlay_tickets(self, limit: int = 20) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM parlay_tickets ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    def open_parlay_bets(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM parlay_bets WHERE status='pending' ORDER BY placed_at"
        ).fetchall()

    def parlay_tickets_for_closing(self) -> list[sqlite3.Row]:
        """
        Every ticket with at least one leg not yet priced at the close.

        Which of those legs are actually due is decided per leg by the
        caller, since a ticket can span games hours apart and the first
        leg's market is pulled long before the last one's.

        Tickets that were never bet are included on purpose -- their
        closing lines are what turns "did the bets win" into "does the
        model predict anything", and they cost the same call as the bet
        ones on the same event.
        """
        return self.conn.execute(
            """SELECT t.* FROM parlay_tickets t
               WHERE EXISTS (SELECT 1 FROM parlay_legs l
                             WHERE l.ticket_id = t.id AND l.closed_at IS NULL)"""
        ).fetchall()

    def parlay_summary(self) -> dict[str, Any]:
        """Headline numbers for multi-leg entries, kept separate from
        single bets because the two have completely different variance."""
        settled = self.conn.execute(
            "SELECT * FROM parlay_bets WHERE status NOT IN ('pending')"
        ).fetchall()
        staked = sum(r["stake"] for r in settled)
        pnl = sum(r["pnl"] or 0.0 for r in settled)
        pending = self.conn.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(stake),0) s FROM parlay_bets "
            "WHERE status='pending'"
        ).fetchone()
        clv = [
            r["clv_ev"] for r in self.conn.execute(
                "SELECT clv_ev FROM parlay_closing_lines WHERE clv_ev IS NOT NULL"
            ).fetchall()
        ]
        generated = self.conn.execute(
            "SELECT COUNT(*) n FROM parlay_tickets"
        ).fetchone()["n"]
        return {
            "tickets_generated": generated,
            "entries_settled": len(settled),
            "entries_pending": pending["n"],
            "stake_pending": pending["s"],
            "total_staked": staked,
            "total_pnl": pnl,
            "roi": (pnl / staked) if staked else None,
            "avg_ticket_clv": (sum(clv) / len(clv)) if clv else None,
            "clv_sample": len(clv),
        }

    def record_spend(
        self,
        credits: int,
        command: str | None = None,
        detail: str | None = None,
        remaining: int | None = None,
        at: datetime | None = None,
    ) -> int | None:
        """Log what a command cost. Zero-cost runs are not worth a row."""
        if credits <= 0:
            return None
        with self.tx() as c:
            cur = c.execute(
                """INSERT INTO credit_spend (at, command, detail, credits, remaining)
                   VALUES (?,?,?,?,?)""",
                (
                    (at or datetime.now(timezone.utc)).isoformat(),
                    command,
                    detail,
                    int(credits),
                    remaining,
                ),
            )
            return cur.lastrowid

    def spend_between(self, start: datetime, end: datetime) -> int:
        """
        Credits logged in a half-open interval.

        Summed in Python rather than SQL because timestamps reach the
        database in more than one ISO spelling and those do not compare
        correctly as strings -- the same reason pending_closing_capture
        filters in Python.
        """
        total = 0
        for r in self.conn.execute(
            "SELECT at, credits FROM credit_spend"
        ).fetchall():
            ts = parse_timestamp(r["at"])
            if ts is not None and start <= ts < end:
                total += r["credits"] or 0
        return total

    def spend_by_day(self, limit: int = 30) -> list[dict[str, Any]]:
        """Recent daily totals, newest first, for the budget report."""
        buckets: dict[str, int] = {}
        for r in self.conn.execute(
            "SELECT at, credits FROM credit_spend"
        ).fetchall():
            ts = parse_timestamp(r["at"])
            if ts is None:
                continue
            key = ts.strftime("%Y-%m-%d")
            buckets[key] = buckets.get(key, 0) + (r["credits"] or 0)
        rows = sorted(buckets.items(), reverse=True)[:limit]
        return [{"day": d, "credits": n} for d, n in rows]

    # ------------------------------------------------------------- reading

    def get_opportunity(self, opp_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM opportunities WHERE id=?", (opp_id,)
        ).fetchone()

    # -------------------------------------------------------------
    # Duplicate detection
    #
    # A bet logged twice is worse than a bet not logged at all. A
    # missing row makes the sample smaller; a doubled row makes the
    # P&L, the ROI and the realisation ratio all wrong, in a way that
    # looks entirely plausible afterwards and can never be spotted from
    # the numbers themselves. And the moment it is most likely to
    # happen is exactly this one: pasting a batch of bets copied off a
    # phone, where a re-run or an overlapping screenshot means doing
    # half of them twice.
    #
    # Identity deliberately EXCLUDES price, stake and status. Betting
    # the same prop twice at different prices is the thing worth
    # warning about, not the thing to permit silently -- if it really
    # was two positions, `--allow-duplicate` says so out loud.
    # -------------------------------------------------------------

    #: What makes two bets "the same bet".
    IDENTITY_FIELDS = ("sport", "market", "selection", "side", "line", "book")

    @staticmethod
    def bet_identity(row) -> tuple:
        """A normalised identity tuple, so case and spacing cannot hide a dupe."""
        get = row.get if isinstance(row, dict) else (lambda k, d=None: row[k]
                                                    if k in row.keys() else d)

        def norm(value):
            if value is None or value == "":
                return None
            if isinstance(value, (int, float)):
                return float(value)
            return " ".join(str(value).strip().lower().split())

        return tuple(norm(get(f)) for f in Database.IDENTITY_FIELDS)

    def matching_bets(self, **fields) -> list[sqlite3.Row]:
        """
        Every already-logged bet with the same identity as `fields`.

        Voided bets are excluded: a void is a bet that did not happen,
        so re-logging the same selection afterwards is legitimate.
        """
        wanted = self.bet_identity(fields)
        rows = self.conn.execute(
            "SELECT * FROM bets WHERE status != 'void'"
        ).fetchall()
        return [r for r in rows if self.bet_identity(r) == wanted]

    def duplicate_groups(self) -> list[list[sqlite3.Row]]:
        """
        Every set of two-or-more bets in the ledger sharing an identity.

        For auditing what is already there, rather than guarding what is
        about to go in.
        """
        buckets: dict[tuple, list] = {}
        for row in self.conn.execute(
            "SELECT * FROM bets WHERE status != 'void' ORDER BY id"
        ).fetchall():
            buckets.setdefault(self.bet_identity(row), []).append(row)
        return [g for g in buckets.values() if len(g) > 1]

    # -------------------------------------------------------------
    # Prop line history, and what moved
    # -------------------------------------------------------------

    def record_prop_lines(self, legs, sport: str,
                          observed_at: datetime | None = None) -> int:
        """
        Bank every leg a scan built, whether or not it cleared a filter.

        ESPECIALLY the ones that did not clear. A leg at 51% is worthless
        today and is the entire point tomorrow: it is the baseline that
        makes a later 68% legible as a MOVE rather than just a number.
        Recording only the winners would throw away the half of the data
        that gives the other half its meaning.
        """
        observed_at = observed_at or datetime.now(timezone.utc)
        stamp = observed_at.isoformat()
        rows = []
        for leg in legs:
            fair = getattr(leg, "fair_prob", None)
            if fair is None:
                continue
            commence = getattr(leg, "commence_time", None)
            rows.append((
                stamp, sport,
                str(getattr(leg, "event_id", "") or ""),
                commence.isoformat() if hasattr(commence, "isoformat")
                else (str(commence) if commence else None),
                str(getattr(leg, "market", "") or ""),
                str(getattr(leg, "selection", "") or ""),
                str(getattr(leg, "side", "") or ""),
                str(getattr(leg, "book", "") or ""),
                getattr(leg, "line", None),
                getattr(leg, "book_price", None),
                float(fair),
                getattr(leg, "push_prob", None),
                getattr(leg, "sharp_line", None),
                getattr(leg, "sharp_price_taken", None),
                getattr(leg, "line_source", None),
            ))
        if not rows:
            return 0
        with self.tx() as c:
            c.executemany(
                "INSERT INTO prop_lines (observed_at, sport, event_id, "
                "commence_time, market, selection, side, book, line, "
                "book_price, fair_prob, push_prob, sharp_line, sharp_price, "
                "line_source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        return len(rows)

    def line_movements(self, min_drift: float = 0.04,
                       max_age_hours: float = 48.0,
                       sport: str | None = None,
                       books: Sequence[str] | None = None,
                       now: datetime | None = None) -> list["LineMove"]:
        """
        Where the market moved and the pick'em book did not.

        Grouped by the book's OWN LINE as well as the selection, which is
        the crux: if the book moved its number, the two observations
        belong to different groups and no move is reported. That is
        correct -- a book that has repriced is a book that has caught up,
        and there is no edge left in it. Only a line held flat while the
        fair probability underneath it walked away is worth a phone
        buzzing.

        Returns the moves sorted by size, largest first. Reports both
        directions: fair probability falling on a line means the value is
        on the OTHER side of it.
        """
        now = now or datetime.now(timezone.utc)
        cutoff = (now - timedelta(hours=max_age_hours)).isoformat()
        sql = ("SELECT * FROM prop_lines WHERE observed_at >= ? "
               "ORDER BY observed_at")
        rows = self.conn.execute(sql, (cutoff,)).fetchall()

        wanted_books = {b.lower() for b in books} if books else None
        groups: dict[tuple, list] = {}
        for row in rows:
            if sport and row["sport"] != sport:
                continue
            if wanted_books and (row["book"] or "").lower() not in wanted_books:
                continue
            key = (row["sport"], row["event_id"], row["market"],
                   row["selection"], row["side"], row["book"], row["line"])
            groups.setdefault(key, []).append(row)

        moves = []
        for key, observations in groups.items():
            if len(observations) < 2:
                continue
            first, last = observations[0], observations[-1]
            drift = last["fair_prob"] - first["fair_prob"]
            if abs(drift) < min_drift:
                continue
            moves.append(LineMove(
                sport=key[0], event_id=key[1], market=key[2],
                selection=key[3], side=key[4], book=key[5], line=key[6],
                first_seen=parse_timestamp(first["observed_at"]),
                last_seen=parse_timestamp(last["observed_at"]),
                fair_first=first["fair_prob"],
                fair_last=last["fair_prob"],
                sharp_line_first=first["sharp_line"],
                sharp_line_last=last["sharp_line"],
                commence_time=last["commence_time"],
                observations=len(observations),
            ))
        moves.sort(key=lambda m: abs(m.drift), reverse=True)
        return moves

    def prop_line_coverage(self, hours: float = 48.0,
                           now: datetime | None = None) -> dict:
        """
        Whether there is enough history to say anything yet.

        A single observation of a line proves nothing about whether a
        book is slow; it takes two. This reports how much of the stored
        history is actually comparable, so "no moves found" can be told
        apart from "nothing has been looked at twice".
        """
        now = now or datetime.now(timezone.utc)
        cutoff = (now - timedelta(hours=hours)).isoformat()
        rows = self.conn.execute(
            "SELECT sport, event_id, market, selection, side, book, line, "
            "COUNT(*) n, MIN(observed_at) lo, MAX(observed_at) hi "
            "FROM prop_lines WHERE observed_at >= ? "
            "GROUP BY sport, event_id, market, selection, side, book, line",
            (cutoff,)).fetchall()
        repeated = [r for r in rows if r["n"] > 1]
        span = 0.0
        for r in repeated:
            lo, hi = parse_timestamp(r["lo"]), parse_timestamp(r["hi"])
            if lo and hi:
                span = max(span, (hi - lo).total_seconds() / 3600.0)
        return {
            "lines": len(rows),
            "seen_twice": len(repeated),
            "observations": sum(r["n"] for r in rows),
            "widest_span_hours": span,
        }

    def get_bet(self, bet_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM bets WHERE id=?", (bet_id,)).fetchone()

    def last_notification(self, fingerprint: str) -> sqlite3.Row | None:
        """The most recent SUCCESSFUL send for this bet, if any."""
        return self.conn.execute(
            "SELECT * FROM notifications WHERE fingerprint=? AND ok=1 "
            "ORDER BY sent_at DESC LIMIT 1",
            (fingerprint,),
        ).fetchone()

    def should_notify(
        self,
        fingerprint: str,
        price: float | None,
        now: datetime,
        resend_after_hours: float = 12.0,
        resend_on_price_gain: float = 0.05,
    ) -> tuple[bool, str]:
        """
        Whether this bet is worth buzzing about, and why.

        Three ways to earn a notification: never sent, sent long enough
        ago that it was plausibly missed, or the price has moved far
        enough in your favour that it is materially a better bet than the
        one already described. Anything else is the same bet again, and
        repeating it is how a person learns to ignore the alert that
        matters.
        """
        previous = self.last_notification(fingerprint)
        if previous is None:
            return True, "new"

        sent_at = parse_timestamp(previous["sent_at"])
        if sent_at is not None:
            age = (now - sent_at).total_seconds() / 3600.0
            if age >= resend_after_hours:
                return True, f"last sent {age:.0f}h ago"

        old_price = previous["price"]
        if price and old_price and old_price > 0:
            gain = (price - old_price) / old_price
            if gain >= resend_on_price_gain:
                return True, f"price improved {gain:+.1%}"
        return False, "already sent"

    def record_notification(
        self,
        fingerprint: str,
        now: datetime,
        provider: str | None = None,
        ev: float | None = None,
        price: float | None = None,
        opportunity_id: int | None = None,
        title: str | None = None,
        ok: bool = True,
        error: str | None = None,
    ) -> int:
        """
        Log a send, successful or not.

        Failures are recorded too, and deliberately do NOT count as
        having been sent -- otherwise a broken phone would silently
        suppress the bet on the next run, which is the worst possible
        moment to go quiet.
        """
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO notifications (fingerprint, sent_at, provider, "
                "ev, price, opportunity_id, title, ok, error) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (fingerprint, now.isoformat(), provider, ev, price,
                 opportunity_id, title, 1 if ok else 0, error),
            )
        return cursor.lastrowid

    def recent_notifications(self, limit: int = 20) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM notifications ORDER BY sent_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def delete_bet(self, bet_id: int) -> sqlite3.Row | None:
        """
        Remove a bet from the ledger entirely, with its closing lines.

        Settling a bet you never placed is not the same as never having
        placed it: a phantom win inflates realised P&L and, worse, the
        `realised / modelled` ratio that is supposed to answer whether the
        model's claimed edge shows up. There is no soft-delete here on
        purpose -- a row kept "for the record" is a row some later query
        will count.

        Returns the row as it was, so the caller can say what it removed,
        or None if there was nothing with that id.
        """
        row = self.get_bet(bet_id)
        if row is None:
            return None
        with self.conn:
            self.conn.execute("DELETE FROM closing_lines WHERE bet_id=?", (bet_id,))
            self.conn.execute("DELETE FROM bets WHERE id=?", (bet_id,))
        return row

    def latest_scan_id(self) -> int | None:
        row = self.conn.execute("SELECT MAX(id) AS id FROM scans").fetchone()
        return row["id"] if row else None

    def opportunities_for_scan(self, scan_id: int) -> list[sqlite3.Row]:
        """
        One scan's opportunities, ranked the same way the scan itself ranked
        them -- by liquidity-discounted edge, not raw EV. Ordering by ev here
        put the thin markets back on top and quietly contradicted the
        shortlist `betedge daily` had already printed.

        COALESCE covers rows written before edge_score existed.
        """
        return self.conn.execute(
            "SELECT * FROM opportunities WHERE scan_id=? "
            "ORDER BY COALESCE(edge_score, ev) DESC",
            (scan_id,),
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

    def compare_strategies(self):
        """
        Single bets against multi-leg entries, on identical metrics.

        Single bets and multi-leg entries have always been tracked in
        separate tables, which is what makes this possible at all --  but
        their summaries grew different field names, so nothing could read
        them together. This puts both through one shape.

        Void bets are excluded from both: a returned stake is not a result
        either way, and counting them would dilute every rate with
        outcomes that never happened.
        """
        from . import performance

        # Every single bet, however it was logged. Whether the scanner
        # printed a row for it or you typed it in afterwards is a
        # bookkeeping detail, not a strategy: the comparison here is
        # one-leg betting against multi-leg betting, and splitting the
        # ledger by how a bet reached it answers a question nobody asked
        # while making both halves too small to say anything.
        #
        # What DOES depend on it is `realised / modelled` -- a hand-logged
        # bet carries no EV -- and that is handled where the ratio is
        # computed, by matching its numerator to the bets that had one.
        singles = self.conn.execute(
            "SELECT stake, pnl, ev_at_bet FROM bets "
            "WHERE status NOT IN ('pending','void')"
        ).fetchall()
        single_pending = self.conn.execute(
            "SELECT id FROM bets WHERE status='pending'"
        ).fetchall()
        single_clv = [
            r["clv_ev"] for r in self.conn.execute(
                "SELECT clv_ev FROM closing_lines "
                "WHERE clv_ev IS NOT NULL AND bet_id IS NOT NULL"
            ).fetchall()
        ]

        parlays = self.conn.execute(
            "SELECT stake, pnl, ev_at_bet FROM parlay_bets "
            "WHERE status NOT IN ('pending','void')"
        ).fetchall()
        parlay_pending = self.conn.execute(
            "SELECT id FROM parlay_bets WHERE status='pending'"
        ).fetchall()
        # Only tickets actually entered. Closing lines are captured for
        # tickets that were never bet too -- that is the model-quality
        # signal, and belongs in the parlay report rather than in a
        # comparison of what the two strategies EARNED.
        parlay_clv = [
            r["clv_ev"] for r in self.conn.execute(
                "SELECT clv_ev FROM parlay_closing_lines "
                "WHERE clv_ev IS NOT NULL AND parlay_bet_id IS NOT NULL"
            ).fetchall()
        ]

        return performance.Comparison([
            performance.summarise(
                "single bets", singles, single_pending, single_clv
            ),
            performance.summarise(
                "multi-leg", parlays, parlay_pending, parlay_clv
            ),
        ])

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
