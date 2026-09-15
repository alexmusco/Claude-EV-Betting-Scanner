"""The `daily` workflow end to end, with no network and no credits spent."""

from datetime import timedelta

import pytest

from betedge import cli
from betedge.db import Database
from conftest import NOW, FakeClient
from test_scan import mlb_prop, nfl_game


@pytest.fixture
def wired(monkeypatch, cfg, tmp_path):
    """A config file on disk and a fake client behind build_client."""
    import yaml

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "books": {"sharp": "pinnacle", "soft": ["draftkings"]},
        "sports": ["baseball_mlb"],
        "core_sports": ["americanfootball_nfl"],
        "prop_markets": {"baseball_mlb": ["pitcher_strikeouts"]},
        "database": str(tmp_path / "t.db"),
        "reports_dir": str(tmp_path / "reports"),
        "budget": {"monthly_credits": 20000, "reserve": 800, "daily_burst": 2.0},
    }))

    client = FakeClient(
        bulk_odds={"americanfootball_nfl": [nfl_game(dk_home_price=2.10)]},
        events_by_sport={"baseball_mlb": [
            {"id": "mlb1", "commence_time": (NOW + timedelta(hours=6)).isoformat()}]},
        event_odds={("baseball_mlb", "mlb1"): mlb_prop(dk_over=2.30)},
        remaining=18000,
    )
    monkeypatch.setattr(cli, "build_client", lambda cfg: client)
    return cfg_path, client, tmp_path


def run(argv):
    return cli.main(argv)


class TestDaily:
    def test_a_full_run_succeeds_and_reports(self, wired, capsys):
        cfg_path, client, tmp_path = wired
        assert run(["--config", str(cfg_path), "daily", "--no-close"]) == 0
        out = capsys.readouterr().out
        assert "Remaining" in out
        assert "Scanning with up to" in out
        assert "Scan #1" in out
        assert "Kansas City Chiefs" in out

    def test_it_writes_a_report_file(self, wired):
        cfg_path, _client, tmp_path = wired
        run(["--config", str(cfg_path), "daily", "--no-close"])
        assert list((tmp_path / "reports").glob("scan_*.html"))

    def test_no_report_flag_is_respected(self, wired):
        cfg_path, _client, tmp_path = wired
        run(["--config", str(cfg_path), "daily", "--no-close", "--no-report"])
        assert not (tmp_path / "reports").exists() or \
            not list((tmp_path / "reports").glob("scan_*.html"))

    def test_spending_is_written_to_the_ledger(self, wired):
        cfg_path, client, tmp_path = wired
        run(["--config", str(cfg_path), "daily", "--no-close"])
        db = Database(tmp_path / "t.db")
        assert db.spend_by_day()[0]["credits"] == client.quota.spent_this_session > 0
        db.close()

    def test_the_scan_is_capped_by_the_daily_allowance(self, wired):
        """The point of the budget: the per-scan ceiling is set from the
        plan, not from the static config value."""
        cfg_path, client, _ = wired
        run(["--config", str(cfg_path), "daily", "--no-close"])
        assert client.max_credits_per_scan is not None
        assert client.max_credits_per_scan < 20000

    def test_an_exhausted_allowance_skips_scanning(self, wired, capsys):
        cfg_path, client, tmp_path = wired
        db = Database(tmp_path / "t.db")
        db.record_spend(5000, "scan")      # today's allowance already gone
        db.close()
        assert run(["--config", str(cfg_path), "daily", "--no-close"]) == 0
        out = capsys.readouterr().out
        assert "No credits left in today's allowance" in out
        assert client.calls["odds"] == 0, "must not spend after the allowance is gone"

    def test_below_the_reserve_nothing_is_scanned(self, wired, capsys):
        cfg_path, client, _ = wired
        client.quota.remaining = 300       # under the 800 reserve
        assert run(["--config", str(cfg_path), "daily", "--no-close"]) == 0
        assert client.calls["odds"] == 0

    def test_bankroll_override_changes_the_stakes(self, wired, capsys):
        cfg_path, _client, _ = wired
        run(["--config", str(cfg_path), "daily", "--no-close", "--bankroll", "10000"])
        big = capsys.readouterr().out
        run(["--config", str(cfg_path), "daily", "--no-close", "--bankroll", "500"])
        small = capsys.readouterr().out
        assert big != small

    def test_an_unreachable_api_fails_cleanly(self, monkeypatch, wired, capsys):
        cfg_path, client, _ = wired

        def boom():
            raise RuntimeError("network down")
        monkeypatch.setattr(client, "probe_quota", boom)
        assert run(["--config", str(cfg_path), "daily"]) == 1
        assert "cannot reach the API" in capsys.readouterr().err


class TestBudgetCommand:
    def test_it_prints_the_plan_and_history(self, wired, capsys):
        cfg_path, _client, tmp_path = wired
        db = Database(tmp_path / "t.db")
        db.record_spend(120, "scan")
        db.close()
        assert run(["--config", str(cfg_path), "budget"]) == 0
        out = capsys.readouterr().out
        assert "Remaining" in out and "120" in out

    def test_it_works_with_no_history(self, wired, capsys):
        cfg_path, _client, _ = wired
        assert run(["--config", str(cfg_path), "budget"]) == 0
        assert "No spending logged yet" in capsys.readouterr().out


class TestBetLifecycle:
    def test_scan_then_bet_then_settle(self, wired, capsys):
        cfg_path, _client, tmp_path = wired
        run(["--config", str(cfg_path), "daily", "--no-close"])
        capsys.readouterr()

        db = Database(tmp_path / "t.db")
        opp_id = db.conn.execute(
            "SELECT id FROM opportunities WHERE suspect=0 ORDER BY edge_score DESC"
        ).fetchone()["id"]
        db.close()

        assert run(["--config", str(cfg_path), "bet", str(opp_id), "--stake", "50"]) == 0
        assert "Logged bet #1" in capsys.readouterr().out

        assert run(["--config", str(cfg_path), "settle", "1", "won"]) == 0
        assert "settled won" in capsys.readouterr().out

        assert run(["--config", str(cfg_path), "report"]) == 0
        assert "Betting performance" in capsys.readouterr().out

    def test_export_to_csv(self, wired, tmp_path, capsys):
        cfg_path, _client, _ = wired
        run(["--config", str(cfg_path), "daily", "--no-close"])
        db = Database(tmp_path / "t.db")
        db.place_bet(stake=50, price=2.1, book="draftkings", selection="X", side="Over")
        db.close()
        capsys.readouterr()
        out_csv = tmp_path / "bets.csv"
        assert run(["--config", str(cfg_path), "export", str(out_csv)]) == 0
        assert out_csv.exists() and "draftkings" in out_csv.read_text()


class TestShownIdsAreUsable:
    def test_shortlist_ids_are_the_ones_bet_accepts(self, wired, capsys):
        """
        The numbers printed have to be the numbers you type. Numbering rows
        1..n while `betedge bet` keyed on database ids meant the obvious
        action placed a different bet than the one on screen.
        """
        cfg_path, _client, tmp_path = wired
        run(["--config", str(cfg_path), "daily", "--no-close"])
        out = capsys.readouterr().out

        db = Database(tmp_path / "t.db")
        rows = db.conn.execute(
            "SELECT id, selection FROM opportunities WHERE suspect=0"
        ).fetchall()
        db.close()

        shortlist = out.split("id      EV")[1]
        for r in rows:
            assert f"{r['id']:>3}  " in shortlist, f"id {r['id']} not shown"

        first_id = rows[0]["id"]
        assert run(["--config", str(cfg_path), "bet", str(first_id),
                    "--stake", "25"]) == 0
        assert rows[0]["selection"].split()[0] in capsys.readouterr().out

    def test_scan_command_shows_ids_too(self, wired, capsys):
        cfg_path, _client, tmp_path = wired
        run(["--config", str(cfg_path), "scan", "--no-report"])
        out = capsys.readouterr().out
        db = Database(tmp_path / "t.db")
        ids = [r["id"] for r in db.conn.execute(
            "SELECT id FROM opportunities WHERE suspect=0")]
        db.close()
        assert ids and all(str(i) in out for i in ids)


class TestQuota:
    def test_it_projects_what_the_plan_affords(self, wired, capsys):
        cfg_path, _client, _ = wired
        assert run(["--config", str(cfg_path), "quota"]) == 0
        out = capsys.readouterr().out
        assert "Credits remaining" in out
        assert "Game-level sweep" in out
        assert "runs a day" in out

    def test_it_spends_nothing(self, wired):
        """/sports and /events are unbilled, so this must stay free."""
        cfg_path, client, _ = wired
        run(["--config", str(cfg_path), "quota"])
        assert client.quota.spent_this_session == 0


class TestTrackerExport:
    def test_a_missing_template_says_what_to_do(self, wired, tmp_path, capsys):
        cfg_path, _client, _ = wired
        db = Database(tmp_path / "t.db")
        db.place_bet(stake=50, price=2.1, book="draftkings", selection="X", side="Over")
        db.close()
        assert run(["--config", str(cfg_path), "export",
                    str(tmp_path / "out.xlsx")]) == 1
        err = capsys.readouterr().err
        assert "--template" in err and ".csv" in err

    def test_csv_export_needs_no_template(self, wired, tmp_path):
        cfg_path, _client, _ = wired
        db = Database(tmp_path / "t.db")
        db.place_bet(stake=50, price=2.1, book="draftkings", selection="X", side="Over")
        db.close()
        out = tmp_path / "bets.csv"
        assert run(["--config", str(cfg_path), "export", str(out)]) == 0
        assert out.exists()
