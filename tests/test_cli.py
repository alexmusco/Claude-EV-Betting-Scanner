"""The `daily` workflow end to end, with no network and no credits spent."""

from datetime import datetime, timedelta, timezone

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


class TestExclusionsSurviveOverrides:
    def test_a_cli_sports_override_cannot_reach_a_blocked_sport(self, wired, capsys):
        """`--sports basketball_ncaab` is the obvious way to bypass a config
        list, so the guard has to sit below the override, not in it."""
        cfg_path, client, _ = wired
        run(["--config", str(cfg_path), "scan", "--no-report",
             "--sports", "basketball_ncaab"])
        assert client.calls["event_odds"] == 0

    def test_a_cli_core_sports_override_cannot_either(self, wired, capsys):
        cfg_path, client, _ = wired
        run(["--config", str(cfg_path), "scan", "--no-report", "--no-props",
             "--core-sports", "americanfootball_ncaaf"])
        assert client.calls["odds"] == 0
        assert client.quota.spent_this_session == 0

    def test_quota_marks_a_blocked_sport_rather_than_pricing_it(self, wired, capsys, tmp_path):
        import yaml
        cfg_file = wired[0]
        raw = yaml.safe_load(cfg_file.read_text())
        raw["core_sports"] = ["americanfootball_ncaaf", "baseball_mlb"]
        cfg_file.write_text(yaml.safe_dump(raw))
        run(["--config", str(cfg_file), "quota"])
        out = capsys.readouterr().out
        assert "americanfootball_ncaaf" in out and "excluded" in out


class TestShow:
    def test_it_states_how_old_the_scan_is(self, wired, capsys):
        cfg_path, _client, _ = wired
        run(["--config", str(cfg_path), "daily", "--no-close"])
        capsys.readouterr()
        run(["--config", str(cfg_path), "show"])
        out = capsys.readouterr().out
        assert "Scan #1" in out and "ago" in out

    def test_started_events_are_dropped_by_default(self, wired, capsys, tmp_path):
        """With an hourly cron most of a scan can be stale by the time you
        look; showing a bet whose game kicked off invites a bad click."""
        cfg_path, _client, _ = wired
        run(["--config", str(cfg_path), "daily", "--no-close"])
        capsys.readouterr()

        db = Database(tmp_path / "t.db")
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        db.conn.execute("UPDATE opportunities SET commence_time=?", (past,))
        db.conn.commit()
        db.close()

        run(["--config", str(cfg_path), "show"])
        out = capsys.readouterr().out
        assert "Nothing from this scan is still playable" in out
        assert "already started" in out

    def test_all_flag_shows_them_anyway(self, wired, capsys, tmp_path):
        cfg_path, _client, _ = wired
        run(["--config", str(cfg_path), "daily", "--no-close"])
        db = Database(tmp_path / "t.db")
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        db.conn.execute("UPDATE opportunities SET commence_time=?", (past,))
        db.conn.commit()
        db.close()
        capsys.readouterr()
        run(["--config", str(cfg_path), "show", "--all"])
        assert "Kansas City Chiefs" in capsys.readouterr().out

    def test_it_shows_the_liquidity_column(self, wired, capsys):
        cfg_path, _client, _ = wired
        run(["--config", str(cfg_path), "daily", "--no-close"])
        capsys.readouterr()
        run(["--config", str(cfg_path), "show"])
        out = capsys.readouterr().out
        assert "liq" in out and "deep" in out

    def test_no_scans_yet_points_at_the_right_command(self, wired, capsys):
        cfg_path, _client, _ = wired
        assert run(["--config", str(cfg_path), "show"]) == 1
        assert "betedge daily" in capsys.readouterr().out


class TestMarketIsAlwaysShown:
    def test_show_names_the_stat(self, wired, capsys):
        cfg_path, _client, _ = wired
        run(["--config", str(cfg_path), "daily", "--no-close"])
        capsys.readouterr()
        # The fixture's prop carries an implausible edge and is flagged
        # suspect, so it is behind --include-suspect. The stat has to be
        # named there too -- suspect rows are the ones you look at hardest.
        run(["--config", str(cfg_path), "show", "--include-suspect"])
        assert "Pitcher Ks" in capsys.readouterr().out

    def test_the_bet_confirmation_names_the_stat(self, wired, capsys, tmp_path):
        """The worst place to omit it: the line confirming what you logged."""
        cfg_path, _client, _ = wired
        run(["--config", str(cfg_path), "daily", "--no-close"])
        capsys.readouterr()
        db = Database(tmp_path / "t.db")
        opp = db.conn.execute(
            "SELECT id FROM opportunities WHERE market='pitcher_strikeouts' "
            "AND suspect=0 LIMIT 1"
        ).fetchone()
        db.close()
        if opp is None:
            pytest.skip("no prop opportunity in this fixture run")
        run(["--config", str(cfg_path), "bet", str(opp["id"]), "--stake", "20"])
        assert "Pitcher Ks" in capsys.readouterr().out

    def test_diagnose_names_the_stat(self, wired, capsys):
        cfg_path, _client, _ = wired
        run(["--config", str(cfg_path), "diagnose"])
        assert "Pitcher Ks" in capsys.readouterr().out


class TestAmericanPrices:
    def test_show_displays_american(self, wired, capsys):
        cfg_path, _client, _ = wired
        run(["--config", str(cfg_path), "daily", "--no-close"])
        capsys.readouterr()
        run(["--config", str(cfg_path), "show"])
        out = capsys.readouterr().out
        assert "+110" in out, "2.10 should read as +110"

    def test_a_bet_can_be_logged_at_an_american_price(self, wired, capsys, tmp_path):
        """`--price -110` used to be read as a decimal price of -110."""
        cfg_path, _client, _ = wired
        run(["--config", str(cfg_path), "daily", "--no-close"])
        capsys.readouterr()
        db = Database(tmp_path / "t.db")
        opp = db.conn.execute(
            "SELECT id FROM opportunities WHERE suspect=0 LIMIT 1"
        ).fetchone()
        db.close()
        assert run(["--config", str(cfg_path), "bet", str(opp["id"]),
                    "--stake", "15", "--price", "+122"]) == 0
        assert "+122" in capsys.readouterr().out

        db = Database(tmp_path / "t.db")
        stored = db.conn.execute("SELECT price FROM bets").fetchone()["price"]
        db.close()
        assert stored == pytest.approx(2.22), "stored as decimal for the maths"

    def test_a_negative_american_price_is_not_read_as_decimal(self, wired, tmp_path, capsys):
        cfg_path, _client, _ = wired
        run(["--config", str(cfg_path), "daily", "--no-close"])
        db = Database(tmp_path / "t.db")
        opp = db.conn.execute(
            "SELECT id FROM opportunities WHERE suspect=0 LIMIT 1"
        ).fetchone()
        db.close()
        capsys.readouterr()
        run(["--config", str(cfg_path), "bet", str(opp["id"]),
             "--stake", "10", "--price", "-110"])
        db = Database(tmp_path / "t.db")
        stored = db.conn.execute("SELECT price FROM bets").fetchone()["price"]
        db.close()
        assert stored == pytest.approx(1.909, abs=1e-3)

    def test_decimal_input_still_works(self, wired, tmp_path, capsys):
        cfg_path, _client, _ = wired
        run(["--config", str(cfg_path), "daily", "--no-close"])
        db = Database(tmp_path / "t.db")
        opp = db.conn.execute(
            "SELECT id FROM opportunities WHERE suspect=0 LIMIT 1"
        ).fetchone()
        db.close()
        capsys.readouterr()
        run(["--config", str(cfg_path), "bet", str(opp["id"]),
             "--stake", "10", "--price", "2.05"])
        db = Database(tmp_path / "t.db")
        stored = db.conn.execute("SELECT price FROM bets").fetchone()["price"]
        db.close()
        assert stored == pytest.approx(2.05)


class TestCompareCommand:
    def seed(self, tmp_path):
        from betedge.db import Database

        from test_performance import seed_opportunity

        db = Database(tmp_path / "t.db")
        opp = seed_opportunity(db)
        for i in range(14):
            bid = db.place_bet(opp, stake=50, price=1.91, book="draftkings")
            db.settle_bet(bid, "won" if i % 2 == 0 else "lost")
            db.record_closing_line(
                opportunity_id=opp, bet_id=bid, sharp_price_taken=1.9,
                sharp_price_other=1.9, fair_prob_close=0.54, price_taken=1.91,
            )
        db.conn.execute(
            """INSERT INTO parlay_tickets (created_at, product, book, kind,
                   n_legs, joint_prob, ev, payout_all_hit)
               VALUES ('2026-09-15T00:00:00+00:00','underdog_standard',
                       'underdog','pickem',2,0.36,0.08,3.0)"""
        )
        db.conn.commit()
        for i in range(14):
            pb = db.place_parlay_bet(1, stake=20)
            db.settle_parlay_bet(pb, legs_hit=2 if i % 3 == 0 else 1)
        db.close()

    def test_it_puts_both_strategies_on_the_same_metrics(self, wired, capsys):
        cfg_path, _client, tmp_path = wired
        self.seed(tmp_path)
        assert cli.main(["--config", str(cfg_path), "compare"]) == 0
        out = capsys.readouterr().out
        assert "Strategy comparison" in out
        assert "single bets" in out and "multi-leg" in out
        for row in ("ROI", "settled bets", "avg closing-line value",
                    "realised / modelled"):
            assert row in out

    def test_bets_you_picked_yourself_get_their_own_column(self, wired, capsys):
        cfg_path, _client, tmp_path = wired
        self.seed(tmp_path)
        from betedge.db import Database

        db = Database(tmp_path / "t.db")
        mine = db.place_bet(stake=10, price=2.22, book="draftkings",
                            selection="Max Fried", side="Under", line=4.5)
        db.settle_bet(mine, "lost")
        db.close()
        cli.main(["--config", str(cfg_path), "compare"])
        out = capsys.readouterr().out
        assert "your own picks" in out

    def test_it_shows_an_interval_not_just_a_point(self, wired, capsys):
        cfg_path, _client, tmp_path = wired
        self.seed(tmp_path)
        cli.main(["--config", str(cfg_path), "compare"])
        out = capsys.readouterr().out
        assert "ROI 90% interval" in out
        assert " to " in out

    def test_it_says_how_many_bets_a_verdict_would_need(self, wired, capsys):
        cfg_path, _client, tmp_path = wired
        self.seed(tmp_path)
        cli.main(["--config", str(cfg_path), "compare"])
        assert "bets needed for +/-5% ROI" in capsys.readouterr().out

    def test_it_declines_to_call_a_small_sample(self, wired, capsys):
        cfg_path, _client, tmp_path = wired
        self.seed(tmp_path)
        cli.main(["--config", str(cfg_path), "compare"])
        # Whitespace collapsed first: the verdict is wrapped to the
        # terminal, so any phrase long enough to be worth asserting on is
        # long enough to be split across two lines.
        out = " ".join(capsys.readouterr().out.split())
        assert "not evidence of anything yet" in out or "Too few" in out

    def test_an_empty_log_does_not_crash(self, wired, capsys):
        cfg_path, _client, _tmp = wired
        assert cli.main(["--config", str(cfg_path), "compare"]) == 0
        assert "Nothing settled yet" in capsys.readouterr().out

    def test_it_costs_nothing(self, wired):
        cfg_path, client, tmp_path = wired
        self.seed(tmp_path)
        cli.main(["--config", str(cfg_path), "compare"])
        assert client.quota.spent_this_session == 0

    def test_the_html_report_carries_the_comparison(self, wired, tmp_path):
        cfg_path, _client, tmp_path = wired
        self.seed(tmp_path)
        cli.main(["--config", str(cfg_path), "report"])
        html = (tmp_path / "reports" / "performance.html").read_text()
        assert "Which strategy is working" in html
        assert "ROI 90% interval" in html
        assert "Realised / modelled" in html


class TestEventIdWarning:
    """
    The warning is about losing closing-line value, which can only be lost
    on a bet whose event has not closed yet.
    """

    def test_a_settled_bet_is_not_warned_about(self, wired, capsys):
        cfg_path, _client, _tmp = wired
        cli.main(["--config", str(cfg_path), "bet", "--stake", "10",
                  "--price", "1.91", "--book", "draftkings", "--settle", "lost",
                  "--selection", "Max Fried", "--side", "Under", "--line", "4.5"])
        out = capsys.readouterr().out
        assert "settled lost" in out
        assert "no event id" not in out

    def test_a_pending_bet_still_is(self, wired, capsys):
        cfg_path, _client, _tmp = wired
        cli.main(["--config", str(cfg_path), "bet", "--stake", "10",
                  "--price", "1.91", "--book", "draftkings",
                  "--selection", "Max Fried", "--side", "Under", "--line", "4.5"])
        out = capsys.readouterr().out
        assert "no event id" in out
        assert "betedge bet <id>" in out

    def test_a_pending_bet_with_an_event_id_is_not_warned_about(self, wired, capsys):
        cfg_path, _client, _tmp = wired
        cli.main(["--config", str(cfg_path), "bet", "--stake", "10",
                  "--price", "1.91", "--book", "draftkings",
                  "--event-id", "evt1",
                  "--selection", "Max Fried", "--side", "Under", "--line", "4.5"])
        assert "no event id" not in capsys.readouterr().out


class TestDeleteCommand:
    """
    A bet that was never placed is worse than a missing one: it does not
    just add noise, it moves `realised / modelled`, which is the one number
    in the comparison that claims to say whether an edge is real.
    """

    def place(self, cfg_path, selection, settle=None):
        argv = ["--config", str(cfg_path), "bet", "--stake", "10",
                "--price", "1.91", "--book", "draftkings",
                "--selection", selection, "--side", "Under", "--line", "4.5"]
        if settle:
            argv += ["--settle", settle]
        cli.main(argv)

    def test_it_lists_what_it_will_remove_before_removing_it(
        self, wired, capsys, monkeypatch
    ):
        cfg_path, _client, tmp_path = wired
        self.place(cfg_path, "Max Fried", settle="lost")
        capsys.readouterr()
        monkeypatch.setattr("builtins.input", lambda _prompt: "y")
        assert cli.main(["--config", str(cfg_path), "delete", "1"]) == 0
        out = capsys.readouterr().out
        assert "About to delete" in out
        assert "Max Fried" in out
        db = Database(tmp_path / "t.db")
        assert db.get_bet(1) is None
        db.close()

    def test_declining_the_prompt_leaves_the_bet_alone(
        self, wired, capsys, monkeypatch
    ):
        cfg_path, _client, tmp_path = wired
        self.place(cfg_path, "Max Fried", settle="lost")
        monkeypatch.setattr("builtins.input", lambda _prompt: "")
        assert cli.main(["--config", str(cfg_path), "delete", "1"]) == 1
        assert "Left alone" in capsys.readouterr().out
        db = Database(tmp_path / "t.db")
        assert db.get_bet(1) is not None
        db.close()

    def test_yes_skips_the_prompt(self, wired, capsys, monkeypatch):
        cfg_path, _client, tmp_path = wired
        self.place(cfg_path, "Max Fried", settle="lost")

        def no_input(_prompt):  # pragma: no cover - must never run
            raise AssertionError("--yes should not prompt")

        monkeypatch.setattr("builtins.input", no_input)
        assert cli.main(["--config", str(cfg_path), "delete", "1", "--yes"]) == 0
        db = Database(tmp_path / "t.db")
        assert db.get_bet(1) is None
        db.close()

    def test_an_unknown_id_is_named_and_nothing_else_is_touched(
        self, wired, capsys
    ):
        cfg_path, _client, tmp_path = wired
        self.place(cfg_path, "Max Fried", settle="lost")
        capsys.readouterr()
        assert cli.main(["--config", str(cfg_path), "delete", "1", "99",
                         "--yes"]) == 0
        out = capsys.readouterr().out
        assert "No bet with id: 99" in out
        db = Database(tmp_path / "t.db")
        assert db.get_bet(1) is None
        db.close()

    def test_deleting_only_unknown_ids_changes_nothing(self, wired, capsys):
        cfg_path, _client, tmp_path = wired
        self.place(cfg_path, "Max Fried", settle="lost")
        capsys.readouterr()
        assert cli.main(["--config", str(cfg_path), "delete", "99", "--yes"]) == 1
        db = Database(tmp_path / "t.db")
        assert db.get_bet(1) is not None
        db.close()

    def test_the_deleted_bets_leave_the_comparison(self, wired, capsys):
        cfg_path, _client, tmp_path = wired
        self.place(cfg_path, "Max Fried", settle="lost")
        self.place(cfg_path, "Zack Wheeler", settle="won")
        capsys.readouterr()
        cli.main(["--config", str(cfg_path), "delete", "1", "2", "--yes"])
        capsys.readouterr()
        cli.main(["--config", str(cfg_path), "compare"])
        out = capsys.readouterr().out
        assert "Nothing settled" in out or "settled bets" in out
        db = Database(tmp_path / "t.db")
        assert db.conn.execute("SELECT COUNT(*) c FROM bets").fetchone()["c"] == 0
        db.close()

    def test_a_closing_line_goes_with_the_bet(self, wired, tmp_path):
        cfg_path, _client, tmp_path = wired
        self.place(cfg_path, "Max Fried")
        db = Database(tmp_path / "t.db")
        db.conn.execute(
            "INSERT INTO closing_lines (bet_id, captured_at, clv_ev) "
            "VALUES (1, '2026-09-15T00:00:00+00:00', 0.02)"
        )
        db.conn.commit()
        assert db.delete_bet(1) is not None
        left = db.conn.execute(
            "SELECT COUNT(*) c FROM closing_lines WHERE bet_id=1"
        ).fetchone()["c"]
        assert left == 0
        db.close()
