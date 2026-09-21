"""Storage, settlement arithmetic, the credit ledger, and migration."""

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from betedge.db import Database


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "t.db")
    yield d
    d.close()


class TestBets:
    def _bet(self, db, price=2.0, stake=100.0):
        return db.place_bet(stake=stake, price=price, book="draftkings",
                            selection="Test", side="Over", line=1.5)

    @pytest.mark.parametrize("status,expected", [
        ("won", 100.0), ("lost", -100.0), ("push", 0.0), ("void", 0.0),
        ("half_won", 50.0), ("half_lost", -50.0),
    ])
    def test_settlement_arithmetic(self, db, status, expected):
        bet_id = self._bet(db)
        assert db.settle_bet(bet_id, status) == pytest.approx(expected)
        assert db.get_bet(bet_id)["pnl"] == pytest.approx(expected)

    def test_an_unknown_status_is_rejected(self, db):
        with pytest.raises(ValueError, match="status must be one of"):
            db.settle_bet(self._bet(db), "maybe")

    def test_a_bet_needs_a_price_and_a_book(self, db):
        with pytest.raises(ValueError, match="book and a price"):
            db.place_bet(stake=50)

    def test_a_bet_on_an_unknown_opportunity_is_rejected(self, db):
        with pytest.raises(ValueError, match="no opportunity"):
            db.place_bet(opportunity_id=999, stake=50)

    def test_summary_is_empty_but_valid_with_no_bets(self, db):
        s = db.summary()
        assert s["bets_settled"] == 0 and s["roi"] is None

    def test_summary_tracks_roi_and_win_rate(self, db):
        db.settle_bet(self._bet(db, price=2.0), "won")
        db.settle_bet(self._bet(db, price=2.0), "lost")
        s = db.summary()
        assert s["bets_settled"] == 2
        assert s["total_staked"] == pytest.approx(200)
        assert s["total_pnl"] == pytest.approx(0)
        assert s["win_rate"] == pytest.approx(0.5)


class TestCreditLedger:
    def test_spend_is_recorded_and_totalled(self, db):
        db.record_spend(40, "scan", "nfl", remaining=19960)
        db.record_spend(12, "close", remaining=19948)
        now = datetime.now(timezone.utc)
        assert db.spend_between(now - timedelta(hours=1), now + timedelta(hours=1)) == 52

    def test_a_free_call_writes_no_row(self, db):
        assert db.record_spend(0, "quota") is None
        assert db.spend_by_day() == []

    def test_spend_outside_the_window_is_excluded(self, db):
        old = datetime.now(timezone.utc) - timedelta(days=40)
        db.record_spend(500, "scan", at=old)
        db.record_spend(10, "scan")
        now = datetime.now(timezone.utc)
        assert db.spend_between(now - timedelta(days=1), now + timedelta(days=1)) == 10

    def test_both_iso_spellings_parse(self, db):
        """Timestamps arrive as "...Z" from the API and "+00:00" from
        isoformat; string comparison would silently drop one of them."""
        now = datetime.now(timezone.utc)
        db.conn.execute(
            "INSERT INTO credit_spend (at, credits) VALUES (?,?)",
            (now.isoformat().replace("+00:00", "Z"), 7),
        )
        db.conn.commit()
        assert db.spend_between(now - timedelta(hours=1), now + timedelta(hours=1)) == 7

    def test_daily_buckets(self, db):
        db.record_spend(10, "scan")
        db.record_spend(5, "scan")
        db.record_spend(99, "scan", at=datetime.now(timezone.utc) - timedelta(days=2))
        by_day = db.spend_by_day()
        assert by_day[0]["credits"] == 15, "newest first"
        assert len(by_day) == 2


class TestMigration:
    def test_columns_are_added_to_a_database_from_an_older_version(self, tmp_path):
        """A real database has bet history in it. Losing that to a schema
        change is not an acceptable upgrade path."""
        path = tmp_path / "old.db"
        c = sqlite3.connect(path)
        # The real previous schema, not a stripped-down stand-in: the
        # index definitions in SCHEMA reference these columns.
        c.executescript("""
            CREATE TABLE opportunities (
              id INTEGER PRIMARY KEY AUTOINCREMENT, scan_id INTEGER,
              scanned_at TEXT, sport TEXT, event_id TEXT, commence_time TEXT,
              home_team TEXT, away_team TEXT, market TEXT, selection TEXT,
              line REAL, side TEXT, sharp_book TEXT, sharp_price_taken_side REAL,
              sharp_price_other_side REAL, sharp_overround REAL, fair_prob REAL,
              fair_price REAL, devig_method TEXT, devig_spread REAL,
              fair_prob_by_method TEXT, soft_book TEXT, soft_price REAL,
              american_price REAL, ev REAL, ev_min REAL, ev_max REAL,
              kelly_fraction REAL, recommended_stake REAL, sharp_last_update TEXT,
              soft_last_update TEXT, suspect INTEGER, flags TEXT);
            CREATE TABLE bets (
              id INTEGER PRIMARY KEY AUTOINCREMENT, opportunity_id INTEGER,
              placed_at TEXT, sport TEXT, event_id TEXT, commence_time TEXT,
              matchup TEXT, market TEXT, selection TEXT, line REAL, side TEXT,
              book TEXT, price REAL, stake REAL, ev_at_bet REAL,
              fair_prob_at_bet REAL, status TEXT DEFAULT 'pending',
              settled_at TEXT, pnl REAL, notes TEXT);
            INSERT INTO bets (placed_at, book, price, stake, status, pnl, selection)
              VALUES ('2026-09-01T00:00:00+00:00','draftkings',2.1,50,'won',55.0,'Keep Me');
        """)
        c.commit()
        c.close()

        db = Database(path)
        cols = {r[1] for r in db.conn.execute("PRAGMA table_info(opportunities)")}
        assert {"market_tier", "liquidity", "required_ev", "edge_score"} <= cols
        rows = db.conn.execute("SELECT selection, pnl FROM bets").fetchall()
        assert rows[0]["selection"] == "Keep Me" and rows[0]["pnl"] == 55.0
        db.close()

    def test_opening_twice_is_harmless(self, tmp_path):
        Database(tmp_path / "a.db").close()
        d = Database(tmp_path / "a.db")
        d.close()

    def test_a_brand_new_database_gets_every_table(self, tmp_path):
        d = Database(tmp_path / "new.db")
        tables = {
            r[0] for r in d.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"scans", "opportunities", "bets", "closing_lines",
                "credit_spend"} <= tables
        d.close()


class TestClosingLines:
    def test_clv_is_positive_when_you_beat_the_close(self, db):
        bet_id = db.place_bet(stake=100, price=2.20, book="draftkings",
                              selection="X", side="Over")
        db.record_closing_line(
            opportunity_id=None, bet_id=bet_id, sharp_price_taken=2.05,
            sharp_price_other=1.85, fair_prob_close=0.50, price_taken=2.20,
            fair_prob_at_bet=0.48,
        )
        row = db.conn.execute("SELECT * FROM closing_lines").fetchone()
        assert row["clv_ev"] == pytest.approx(0.10)
        assert row["clv_price_pct"] == pytest.approx(0.10)

    def test_clv_is_negative_when_the_close_moves_against_you(self, db):
        bet_id = db.place_bet(stake=100, price=1.90, book="draftkings",
                              selection="X", side="Over")
        db.record_closing_line(
            opportunity_id=None, bet_id=bet_id, sharp_price_taken=2.10,
            sharp_price_other=1.80, fair_prob_close=0.48, price_taken=1.90,
        )
        assert db.conn.execute("SELECT clv_ev FROM closing_lines").fetchone()[0] < 0


class TestOrdering:
    def test_a_scan_is_re_read_in_the_order_it_was_ranked(self, db):
        """`betedge show` must not contradict the shortlist `daily` printed."""
        db.conn.execute(
            "INSERT INTO scans (started_at, finished_at, sports) VALUES ('a','b','s')"
        )
        for ev, edge, sel in [(0.08, 0.024, "thin"), (0.03, 0.030, "deep")]:
            db.conn.execute(
                "INSERT INTO opportunities (scan_id, scanned_at, sport, event_id,"
                " commence_time, market, selection, side, sharp_book,"
                " sharp_price_taken_side, sharp_price_other_side, sharp_overround,"
                " fair_prob, fair_price, devig_method, devig_spread, soft_book,"
                " soft_price, ev, edge_score, suspect)"
                " VALUES (1,'t','s','e','c','h2h',?,'X','pinnacle',1.9,1.9,0.02,"
                "0.5,2.0,'worst_case',0.001,'draftkings',2.1,?,?,0)",
                (sel, ev, edge),
            )
        db.conn.commit()
        rows = db.opportunities_for_scan(1)
        assert [r["selection"] for r in rows] == ["deep", "thin"]

    def test_rows_without_an_edge_score_fall_back_to_ev(self, db):
        db.conn.execute(
            "INSERT INTO scans (started_at, finished_at, sports) VALUES ('a','b','s')"
        )
        for ev, sel in [(0.02, "small"), (0.09, "big")]:
            db.conn.execute(
                "INSERT INTO opportunities (scan_id, scanned_at, sport, event_id,"
                " commence_time, market, selection, side, sharp_book,"
                " sharp_price_taken_side, sharp_price_other_side, sharp_overround,"
                " fair_prob, fair_price, devig_method, devig_spread, soft_book,"
                " soft_price, ev, suspect)"
                " VALUES (1,'t','s','e','c','h2h',?,'X','pinnacle',1.9,1.9,0.02,"
                "0.5,2.0,'worst_case',0.001,'draftkings',2.1,?,0)",
                (sel, ev),
            )
        db.conn.commit()
        assert [r["selection"] for r in db.opportunities_for_scan(1)] == ["big", "small"]


class TestDuplicateBets:
    """
    A bet logged twice is worse than a bet not logged at all.

    A missing row only makes the sample smaller. A doubled row makes the
    P&L, the ROI and the realisation ratio all wrong, plausibly, in a way
    that can never be spotted from the numbers themselves afterwards.
    """

    def bet(self, db, **over):
        fields = dict(
            sport="americanfootball_nfl", market="player_receptions",
            selection="Rashee Rice", side="Under", line=4.5,
            book="draftkings", price=2.08, stake=10.0,
        )
        fields.update(over)
        return db.place_bet(**fields)

    def test_the_same_bet_is_found(self, tmp_path):
        db = Database(tmp_path / "t.db")
        self.bet(db)
        assert len(db.matching_bets(
            sport="americanfootball_nfl", market="player_receptions",
            selection="Rashee Rice", side="Under", line=4.5,
            book="draftkings")) == 1

    def test_case_and_spacing_cannot_hide_a_duplicate(self, tmp_path):
        # Copied off a phone by hand, the same bet arrives spelled a
        # dozen ways. Identity is normalised so none of them slip past.
        db = Database(tmp_path / "t.db")
        self.bet(db)
        assert db.matching_bets(
            sport="americanfootball_NFL", market="player_receptions",
            selection="  rashee   rice ", side="under", line=4.5,
            book="DraftKings")

    def test_a_different_line_is_a_different_bet(self, tmp_path):
        db = Database(tmp_path / "t.db")
        self.bet(db)
        assert not db.matching_bets(
            sport="americanfootball_nfl", market="player_receptions",
            selection="Rashee Rice", side="Under", line=5.5,
            book="draftkings")

    def test_a_different_book_is_a_different_bet(self, tmp_path):
        db = Database(tmp_path / "t.db")
        self.bet(db)
        assert not db.matching_bets(
            sport="americanfootball_nfl", market="player_receptions",
            selection="Rashee Rice", side="Under", line=4.5, book="fanduel")

    def test_price_and_stake_are_NOT_part_of_identity(self, tmp_path):
        """
        Betting the same prop twice at a different price is exactly the
        thing worth warning about, so a changed price must not make it
        look like a new bet.
        """
        db = Database(tmp_path / "t.db")
        self.bet(db, price=2.08, stake=10.0)
        assert db.matching_bets(
            sport="americanfootball_nfl", market="player_receptions",
            selection="Rashee Rice", side="Under", line=4.5,
            book="draftkings")

    def test_a_voided_bet_does_not_block_relogging(self, tmp_path):
        # A void is a bet that did not happen.
        db = Database(tmp_path / "t.db")
        bet_id = self.bet(db)
        db.settle_bet(bet_id, "void")
        assert not db.matching_bets(
            sport="americanfootball_nfl", market="player_receptions",
            selection="Rashee Rice", side="Under", line=4.5,
            book="draftkings")

    def test_duplicate_groups_finds_what_is_already_there(self, tmp_path):
        db = Database(tmp_path / "t.db")
        self.bet(db)
        self.bet(db)
        self.bet(db, selection="George Holani", market="player_anytime_td",
                 side="Yes", line=None)
        groups = db.duplicate_groups()
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_a_clean_ledger_has_no_groups(self, tmp_path):
        db = Database(tmp_path / "t.db")
        self.bet(db)
        self.bet(db, selection="Deebo Samuel")
        assert db.duplicate_groups() == []
