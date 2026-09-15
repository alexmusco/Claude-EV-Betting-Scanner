"""
The parlay commands end to end, plus persistence and closing lines.

No network and no credits: every payload comes from `fixtures`, and the
client is conftest's FakeClient, which counts its calls so the credit cost
of a command can be asserted rather than hoped for.
"""

import sys
from datetime import timedelta
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from conftest import NOW, FakeClient
from fixtures import KC_STACK, make_leg, parlay_config, prop_event

from betedge import cli, correlation as C, parlay as P
from betedge.closing import capture_parlay_closing_lines
from betedge.db import Database


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """A config file on disk, an NFL event, and a fake client behind build_client."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "books": {"sharp": "pinnacle", "soft": ["draftkings"]},
        "sports": ["americanfootball_nfl"],
        "core_sports": [],
        "prop_markets": {"americanfootball_nfl": [
            "player_pass_yds", "player_reception_yds", "player_rush_yds",
        ]},
        "database": str(tmp_path / "t.db"),
        "reports_dir": str(tmp_path / "reports"),
        "budget": {"enabled": False},
        "parlay": {
            "products": ["underdog_standard"],
            "draws": 20000, "search_draws": 4000, "beam_width": 8,
        },
    }))
    client = FakeClient(
        events_by_sport={"americanfootball_nfl": [
            {"id": "kc1", "commence_time": (NOW + timedelta(hours=8)).isoformat()}
        ]},
        event_odds={("americanfootball_nfl", "kc1"): prop_event()},
        remaining=18000,
    )
    monkeypatch.setattr(cli, "build_client", lambda cfg: client)
    return cfg_path, client, tmp_path


def run(argv):
    return cli.main(argv)


class TestParlayScanCommand:
    def test_a_full_run_succeeds(self, wired, capsys):
        cfg_path, _client, _tmp = wired
        assert run(["--config", str(cfg_path), "parlay", "scan"]) == 0
        out = capsys.readouterr().out
        assert "legs built" in out
        assert "Ranked by expected value" in out
        assert "ticket(s) logged" in out

    def test_it_shows_both_rankings(self, wired, capsys):
        cfg_path, _client, _tmp = wired
        run(["--config", str(cfg_path), "parlay", "scan"])
        out = capsys.readouterr().out
        assert "Ranked by expected value\n" in out
        assert "per unit of variance" in out

    def test_every_ticket_shows_its_independent_ev_alongside(self, wired, capsys):
        cfg_path, _client, _tmp = wired
        run(["--config", str(cfg_path), "parlay", "scan"])
        out = capsys.readouterr().out
        assert "indep" in out

    def test_it_warns_that_the_payout_table_is_unverified(self, wired, capsys):
        cfg_path, _client, _tmp = wired
        run(["--config", str(cfg_path), "parlay", "scan"])
        out = capsys.readouterr().out
        assert "not been verified" in out
        assert "underdog_standard" in out

    def test_it_only_warns_about_the_products_it_used(self, wired, capsys):
        # Naming the whole table on every run trains the reader to skip a
        # line that matters.
        cfg_path, _client, _tmp = wired
        run(["--config", str(cfg_path), "parlay", "scan"])
        note = capsys.readouterr().out.split("NOTE:")[-1]
        assert "underdog_standard" in note
        assert "prizepicks_power" not in note

    def test_a_book_priced_product_needs_no_payout_warning(self, wired, capsys):
        cfg_path, _client, _tmp = wired
        run(["--config", str(cfg_path), "parlay", "scan", "--no-report",
             "--products", "draftkings_parlay"])
        assert "not been verified" not in capsys.readouterr().out

    def test_it_writes_a_report(self, wired):
        cfg_path, _client, tmp_path = wired
        run(["--config", str(cfg_path), "parlay", "scan"])
        reports = list((tmp_path / "reports").glob("parlay_*.html"))
        assert reports
        html = reports[0].read_text()
        assert "independent" in html
        assert "Correlation used, pair by pair" in html

    def test_the_report_names_the_correlation_source_per_pair(self, wired):
        cfg_path, _client, tmp_path = wired
        run(["--config", str(cfg_path), "parlay", "scan"])
        html = list((tmp_path / "reports").glob("parlay_*.html"))[0].read_text()
        assert "src-prior" in html or "src-default" in html

    def test_no_report_is_respected(self, wired, tmp_path):
        cfg_path, _client, tmp_path = wired
        run(["--config", str(cfg_path), "parlay", "scan", "--no-report"])
        assert not list((tmp_path / "reports").glob("parlay_*.html"))

    def test_tickets_land_in_the_database_bet_or_not(self, wired):
        cfg_path, _client, tmp_path = wired
        run(["--config", str(cfg_path), "parlay", "scan"])
        db = Database(tmp_path / "t.db")
        tickets = db.latest_parlay_tickets()
        assert tickets
        # Not one of them has been bet; they are stored anyway, because
        # their closing lines are the evidence the model works.
        assert db.open_parlay_bets() == []
        legs = db.parlay_legs(tickets[0]["id"])
        assert len(legs) == tickets[0]["n_legs"]
        assert all(l["fair_prob"] > 0 for l in legs)
        db.close()

    def test_the_stored_ticket_keeps_both_ev_numbers(self, wired):
        cfg_path, _client, tmp_path = wired
        run(["--config", str(cfg_path), "parlay", "scan"])
        db = Database(tmp_path / "t.db")
        row = db.latest_parlay_tickets()[0]
        assert row["ev"] is not None
        assert row["ev_independent"] is not None
        assert row["joint_prob_se"] is not None
        assert row["correlation_summary"]
        db.close()

    def test_it_spends_only_what_the_slate_costs(self, wired):
        cfg_path, client, _tmp = wired
        run(["--config", str(cfg_path), "parlay", "scan"])
        # One free event list, one billed per-event call, three markets.
        assert client.calls["events"] == 1
        assert client.calls["event_odds"] == 1
        assert client.quota.spent_this_session == 3

    def test_max_events_caps_the_spend(self, wired):
        cfg_path, client, _tmp = wired
        run(["--config", str(cfg_path), "parlay", "scan", "--max-events", "0"])
        assert client.calls["event_odds"] == 0

    def test_a_credit_budget_stops_it_early(self, wired, capsys):
        cfg_path, client, _tmp = wired
        client.max_credits_per_scan = 0

        def refuse(*a, **k):
            from betedge.oddsapi import CreditBudgetExceeded
            raise CreditBudgetExceeded("scan budget of 0 credits is spent")

        client.event_odds = refuse
        assert run(["--config", str(cfg_path), "parlay", "scan"]) == 0
        assert "budget" in capsys.readouterr().out.lower()

    def test_an_excluded_sport_is_never_scanned(self, wired, capsys):
        cfg_path, client, _tmp = wired
        run(["--config", str(cfg_path), "parlay", "scan",
             "--sports", "basketball_ncaab"])
        assert client.calls["events"] == 0
        assert client.quota.spent_this_session == 0

    def test_an_unknown_product_fails_clearly(self, wired, capsys):
        cfg_path, _client, _tmp = wired
        assert run(["--config", str(cfg_path), "parlay", "scan",
                    "--products", "nonsense"]) == 1
        assert "unknown payout product" in capsys.readouterr().err

    def test_a_roster_resolves_team_mates_from_opponents(self, wired, tmp_path, capsys):
        cfg_path, _client, tmp_path = wired
        roster = tmp_path / "roster.csv"
        roster.write_text(
            "player,team\n" + "\n".join(
                f"{player},Kansas City Chiefs" for _m, player, _l, _p in KC_STACK
            ) + "\n"
        )
        run(["--config", str(cfg_path), "parlay", "scan", "--no-report",
             "--rosters", str(roster)])
        out = capsys.readouterr().out
        # With teams known the same-game blend priors stop being used, so
        # the disclosure about them disappears.
        assert "leg_teams_unknown" not in out
        db = Database(tmp_path / "t.db")
        ticket = db.latest_parlay_tickets()[0]
        assert all(l["team"] == "Kansas City Chiefs"
                   for l in db.parlay_legs(ticket["id"]))
        db.close()

    def test_without_a_roster_the_blend_is_disclosed(self, wired, capsys):
        cfg_path, _client, _tmp = wired
        run(["--config", str(cfg_path), "parlay", "scan", "--no-report"])
        assert "leg_teams_unknown" in capsys.readouterr().out


class TestVerifyPayoutsCommand:
    def test_it_prints_every_shipped_structure(self, wired, capsys):
        cfg_path, _client, _tmp = wired
        assert run(["--config", str(cfg_path), "parlay", "verify-payouts"]) == 0
        out = capsys.readouterr().out
        for key in ("underdog_standard", "underdog_flex",
                    "prizepicks_power", "prizepicks_flex"):
            assert key in out

    def test_it_prints_the_actual_multipliers(self, wired, capsys):
        cfg_path, _client, _tmp = wired
        run(["--config", str(cfg_path), "parlay", "verify-payouts"])
        out = capsys.readouterr().out
        assert "2/2: 3x" in out
        assert "5/5: 20x" in out

    def test_it_says_loudly_what_is_unverified(self, wired, capsys):
        cfg_path, _client, _tmp = wired
        run(["--config", str(cfg_path), "parlay", "verify-payouts"])
        out = capsys.readouterr().out
        assert "NOT VERIFIED" in out
        assert "CHECK THESE BEFORE TRUSTING ANY EV NUMBER" in out

    def test_it_states_the_break_even_each_structure_needs(self, wired, capsys):
        cfg_path, _client, _tmp = wired
        run(["--config", str(cfg_path), "parlay", "verify-payouts"])
        assert "needs 54.9% a leg" in capsys.readouterr().out

    def test_it_costs_nothing(self, wired):
        cfg_path, client, _tmp = wired
        run(["--config", str(cfg_path), "parlay", "verify-payouts"])
        assert client.quota.spent_this_session == 0

    def test_a_verified_table_says_so(self, wired, tmp_path, capsys):
        cfg_path, _client, tmp_path = wired
        table = tmp_path / "mine.yaml"
        table.write_text(
            "meta: {last_verified_by_user: 2026-09-15}\n"
            "products:\n  mine:\n    book: underdog\n    kind: pickem\n"
            "    verified: true\n    payouts:\n      2: [0, 0, 3.0]\n"
        )
        cfg = yaml.safe_load(cfg_path.read_text())
        cfg["parlay"]["payouts_path"] = str(table)
        cfg_path.write_text(yaml.safe_dump(cfg))
        run(["--config", str(cfg_path), "parlay", "verify-payouts"])
        out = capsys.readouterr().out
        assert "Every product is marked verified." in out
        assert "2026-09-15" in out


class TestCoverageCommand:
    def test_it_reports_per_sport(self, wired, capsys):
        cfg_path, _client, _tmp = wired
        assert run(["--config", str(cfg_path), "parlay", "coverage",
                    "--sports", "americanfootball_nfl"]) == 0
        out = capsys.readouterr().out
        assert "americanfootball_nfl" in out
        assert "same line" in out

    def test_it_explains_every_exclusion(self, wired, capsys):
        cfg_path, _client, _tmp = wired
        run(["--config", str(cfg_path), "parlay", "coverage",
             "--sports", "americanfootball_nfl"])
        out = capsys.readouterr().out
        assert "Deliberately excluded" in out
        assert "tennis_*" in out
        assert "mma_mixed_martial_arts" in out
        assert "soccer_*" in out

    def test_it_recommends_what_it_measured(self, wired, capsys):
        cfg_path, _client, _tmp = wired
        run(["--config", str(cfg_path), "parlay", "coverage",
             "--sports", "americanfootball_nfl"])
        assert "Usable today" in capsys.readouterr().out

    def test_it_is_capped_to_a_couple_of_events(self, wired):
        cfg_path, client, _tmp = wired
        run(["--config", str(cfg_path), "parlay", "coverage",
             "--sports", "americanfootball_nfl", "--max-events", "1"])
        assert client.calls["event_odds"] == 1


class TestCorrelationsCommand:
    def test_it_says_what_to_do_when_nothing_is_fitted(self, wired, capsys):
        cfg_path, _client, _tmp = wired
        assert run(["--config", str(cfg_path), "parlay", "correlations"]) == 0
        out = capsys.readouterr().out
        assert "No fitted correlations" in out
        assert "hypothesis rather than an estimate" in out

    def test_it_fits_from_a_game_log(self, wired, tmp_path, capsys):
        cfg_path, _client, tmp_path = wired
        logs = tmp_path / "logs.csv"
        rows = ["game_id,sport,player,team,market,value"]
        for i in range(200):
            rows.append(f"g{i},americanfootball_nfl,QB,KC,player_pass_yds,{250 + i}")
            rows.append(
                f"g{i},americanfootball_nfl,WR,KC,player_reception_yds,{60 + i // 2}"
            )
        logs.write_text("\n".join(rows) + "\n")

        assert run(["--config", str(cfg_path), "parlay", "correlations",
                    "--from", str(logs)]) == 0
        out = capsys.readouterr().out
        assert "fitted" in out
        assert "same_team" in out
        assert "used" in out

    def test_a_thin_sample_is_marked_unused(self, wired, tmp_path, capsys):
        cfg_path, _client, tmp_path = wired
        logs = tmp_path / "logs.csv"
        rows = ["game_id,sport,player,team,market,value"]
        for i in range(8):
            rows.append(f"g{i},americanfootball_nfl,QB,KC,player_pass_yds,{250 + i}")
            rows.append(f"g{i},americanfootball_nfl,WR,KC,player_reception_yds,{60 + i}")
        logs.write_text("\n".join(rows) + "\n")
        run(["--config", str(cfg_path), "parlay", "correlations",
             "--from", str(logs)])
        assert "no (<100)" in capsys.readouterr().out

    def test_a_fitted_value_then_reaches_a_scan(self, wired, tmp_path):
        cfg_path, _client, tmp_path = wired
        db = Database(tmp_path / "t.db")
        db.save_correlation_estimates([C.Estimate(
            sport="americanfootball_nfl", market_a="player_pass_yds",
            market_b="player_reception_yds", relation=C.SAME_GAME,
            rho=0.4, spearman=0.38, n_observations=800, n_games=800,
        )])
        db.close()
        run(["--config", str(cfg_path), "parlay", "scan", "--no-report"])
        db = Database(tmp_path / "t.db")
        summaries = [r["correlation_summary"] for r in db.latest_parlay_tickets()]
        assert any("empirical" in (s or "") for s in summaries)
        db.close()

    def test_it_costs_nothing(self, wired):
        cfg_path, client, _tmp = wired
        run(["--config", str(cfg_path), "parlay", "correlations"])
        assert client.quota.spent_this_session == 0


class TestBetAndSettle:
    def logged_ticket(self, cfg_path, tmp_path):
        run(["--config", str(cfg_path), "parlay", "scan", "--no-report"])
        db = Database(tmp_path / "t.db")
        ticket = db.latest_parlay_tickets()[0]
        db.close()
        return ticket

    def test_an_entry_can_be_logged_and_settled(self, wired, capsys):
        cfg_path, _client, tmp_path = wired
        ticket = self.logged_ticket(cfg_path, tmp_path)
        assert run(["--config", str(cfg_path), "parlay", "bet",
                    str(ticket["id"]), "--stake", "20"]) == 0
        db = Database(tmp_path / "t.db")
        bet = db.open_parlay_bets()[0]
        assert bet["stake"] == 20
        assert bet["ev_at_bet"] is not None
        db.close()

        assert run(["--config", str(cfg_path), "parlay", "settle",
                    str(bet["id"]), "--hit", str(bet["n_legs"])]) == 0
        assert "won" in capsys.readouterr().out

    def test_settlement_pays_what_the_structure_says(self, tmp_path):
        db = Database(tmp_path / "t.db")
        db.conn.execute(
            """INSERT INTO parlay_tickets (created_at, product, book, kind,
                   n_legs, joint_prob, ev, payout_all_hit)
               VALUES ('2026-09-15T00:00:00+00:00','underdog_standard',
                       'underdog','pickem',3,0.2,0.1,6.0)"""
        )
        db.conn.commit()
        bet_id = db.place_parlay_bet(1, stake=10)
        assert db.settle_parlay_bet(bet_id, legs_hit=3) == pytest.approx(50.0)
        db.close()

    def test_a_miss_loses_the_stake(self, tmp_path):
        db = Database(tmp_path / "t.db")
        db.conn.execute(
            """INSERT INTO parlay_tickets (created_at, product, book, kind,
                   n_legs, joint_prob, ev, payout_all_hit)
               VALUES ('2026-09-15T00:00:00+00:00','underdog_standard',
                       'underdog','pickem',3,0.2,0.1,6.0)"""
        )
        db.conn.commit()
        bet_id = db.place_parlay_bet(1, stake=10)
        assert db.settle_parlay_bet(bet_id, legs_hit=2) == pytest.approx(-10.0)
        assert db.get_parlay_bet(bet_id)["status"] == "lost"
        db.close()

    def test_a_pushed_leg_settles_on_the_reduced_table(self, tmp_path):
        # Three legs, one void, two winners: a 3-pick becomes a 2-pick and
        # pays 3x, not 6x and not nothing.
        db = Database(tmp_path / "t.db")
        db.conn.execute(
            """INSERT INTO parlay_tickets (created_at, product, book, kind,
                   n_legs, joint_prob, ev, payout_all_hit)
               VALUES ('2026-09-15T00:00:00+00:00','underdog_standard',
                       'underdog','pickem',3,0.2,0.1,6.0)"""
        )
        db.conn.commit()
        bet_id = db.place_parlay_bet(1, stake=10)
        assert db.settle_parlay_bet(bet_id, legs_hit=2, legs_void=1) == pytest.approx(20.0)
        db.close()

    def test_a_flex_near_miss_still_pays(self, tmp_path):
        db = Database(tmp_path / "t.db")
        db.conn.execute(
            """INSERT INTO parlay_tickets (created_at, product, book, kind,
                   n_legs, joint_prob, ev, payout_all_hit)
               VALUES ('2026-09-15T00:00:00+00:00','underdog_flex',
                       'underdog','pickem',3,0.2,0.1,2.25)"""
        )
        db.conn.commit()
        bet_id = db.place_parlay_bet(1, stake=100)
        assert db.settle_parlay_bet(bet_id, legs_hit=2) == pytest.approx(25.0)
        db.close()

    def test_an_impossible_settlement_is_refused(self, tmp_path):
        db = Database(tmp_path / "t.db")
        db.conn.execute(
            """INSERT INTO parlay_tickets (created_at, product, book, kind,
                   n_legs, joint_prob, ev, payout_all_hit)
               VALUES ('2026-09-15T00:00:00+00:00','underdog_standard',
                       'underdog','pickem',3,0.2,0.1,6.0)"""
        )
        db.conn.commit()
        bet_id = db.place_parlay_bet(1, stake=10)
        with pytest.raises(ValueError):
            db.settle_parlay_bet(bet_id, legs_hit=4)
        with pytest.raises(ValueError):
            db.settle_parlay_bet(bet_id, legs_hit=2, legs_void=2)
        db.close()

    def test_betting_a_ticket_that_does_not_exist_is_refused(self, tmp_path):
        db = Database(tmp_path / "t.db")
        with pytest.raises(ValueError, match="no parlay ticket"):
            db.place_parlay_bet(999, stake=10)
        db.close()

    def test_the_summary_counts_generated_tickets_not_just_bet_ones(
        self, wired, tmp_path
    ):
        cfg_path, _client, tmp_path = wired
        run(["--config", str(cfg_path), "parlay", "scan", "--no-report"])
        db = Database(tmp_path / "t.db")
        s = db.parlay_summary()
        assert s["tickets_generated"] > 0
        assert s["entries_settled"] == 0
        db.close()


class TestParlayClosingLines:
    def scanned(self, cfg_path, tmp_path):
        run(["--config", str(cfg_path), "parlay", "scan", "--no-report"])
        return Database(tmp_path / "t.db")

    def test_legs_are_repriced_and_the_ticket_joint_recomputed(self, wired):
        cfg_path, client, tmp_path = wired
        db = self.scanned(cfg_path, tmp_path)
        cfg = cli.Config.load(cfg_path)

        # Pinnacle at the close: every prop has drifted against us.
        closing = prop_event(spec=[
            (m, p, l, (1.95, 1.95)) for m, p, l, _q in KC_STACK
        ])
        client._event_odds[("americanfootball_nfl", "kc1")] = closing

        stats = capture_parlay_closing_lines(
            cfg, client, db, window_minutes=600, now=NOW + timedelta(hours=7.5)
        )
        assert stats["legs_captured"] > 0
        assert stats["tickets_closed"] > 0

        rows = db.conn.execute("SELECT * FROM parlay_closing_lines").fetchall()
        assert rows
        row = rows[0]
        assert row["legs_captured"] == row["n_legs"]
        # Every leg closed at a coin flip, so the joint must have fallen.
        assert row["joint_prob_close"] < row["joint_prob_at_bet"]
        assert row["clv_ev"] is not None
        assert row["clv_prob_points"] < 0
        db.close()

    def test_tickets_that_were_never_bet_are_captured_too(self, wired):
        cfg_path, client, tmp_path = wired
        db = self.scanned(cfg_path, tmp_path)
        cfg = cli.Config.load(cfg_path)
        assert db.open_parlay_bets() == []
        stats = capture_parlay_closing_lines(
            cfg, client, db, window_minutes=600, now=NOW + timedelta(hours=7.5)
        )
        assert stats["tickets_closed"] > 0
        db.close()

    def test_one_call_serves_every_ticket_on_a_game(self, wired):
        cfg_path, client, tmp_path = wired
        db = self.scanned(cfg_path, tmp_path)
        cfg = cli.Config.load(cfg_path)
        assert len(db.latest_parlay_tickets()) > 1
        before = client.calls["event_odds"]
        capture_parlay_closing_lines(
            cfg, client, db, window_minutes=600, now=NOW + timedelta(hours=7.5)
        )
        assert client.calls["event_odds"] - before == 1
        db.close()

    def test_nothing_is_captured_outside_the_window(self, wired):
        cfg_path, client, tmp_path = wired
        db = self.scanned(cfg_path, tmp_path)
        cfg = cli.Config.load(cfg_path)
        stats = capture_parlay_closing_lines(
            cfg, client, db, window_minutes=5, now=NOW
        )
        assert stats["legs_captured"] == 0
        assert stats["tickets_closed"] == 0
        db.close()

    def test_a_ticket_is_only_closed_once(self, wired):
        cfg_path, client, tmp_path = wired
        db = self.scanned(cfg_path, tmp_path)
        cfg = cli.Config.load(cfg_path)
        kwargs = dict(window_minutes=600, now=NOW + timedelta(hours=7.5))
        first = capture_parlay_closing_lines(cfg, client, db, **kwargs)
        second = capture_parlay_closing_lines(cfg, client, db, **kwargs)
        assert first["tickets_closed"] > 0
        assert second["tickets_closed"] == 0
        db.close()

    def test_a_missing_market_is_counted_not_fatal(self, wired):
        cfg_path, client, tmp_path = wired
        db = self.scanned(cfg_path, tmp_path)
        cfg = cli.Config.load(cfg_path)
        client._event_odds[("americanfootball_nfl", "kc1")] = None
        stats = capture_parlay_closing_lines(
            cfg, client, db, window_minutes=600, now=NOW + timedelta(hours=7.5)
        )
        assert stats["not_found"] > 0
        assert stats["legs_captured"] == 0
        db.close()

    def test_the_close_command_covers_tickets_too(self, wired, capsys):
        cfg_path, _client, tmp_path = wired
        self.scanned(cfg_path, tmp_path).close()
        assert run(["--config", str(cfg_path), "close", "--window", "600"]) == 0
        assert "Tickets:" in capsys.readouterr().out

    def test_the_close_command_can_skip_them(self, wired, capsys):
        cfg_path, _client, tmp_path = wired
        self.scanned(cfg_path, tmp_path).close()
        run(["--config", str(cfg_path), "close", "--window", "600", "--no-parlays"])
        assert "Tickets:" not in capsys.readouterr().out
