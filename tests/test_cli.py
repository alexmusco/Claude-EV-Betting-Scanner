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
        # Spend more than the WHOLE cycle holds, not a fixed 5,000.
        #
        # A day's allowance is the credits left spread over the days
        # left, so it GROWS as the cycle runs down: 5,000 exhausted it
        # in the first week and stopped exhausting it in the last, and
        # this test duly passed for three weeks a month. The code was
        # right both times. Anything above the cycle total is over the
        # allowance on every date, since the allowance is capped at
        # what is actually there.
        cfg_path, client, tmp_path = wired
        db = Database(tmp_path / "t.db")
        db.record_spend(25000, "scan")     # > the 20,000 monthly total
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

    def test_there_are_two_columns_however_a_bet_was_logged(self, wired, capsys):
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
        assert "your own picks" not in out
        header = next(l for l in out.splitlines() if "single bets" in l)
        assert header.count("single bets") == 1
        assert "multi-leg" in header

    def test_the_realisation_ratio_says_how_many_bets_it_covers(
        self, wired, capsys
    ):
        # Hand-logged bets sit in the settled count and not in the ratio,
        # so without this row the ratio silently reads as covering
        # everything above it.
        cfg_path, _client, tmp_path = wired
        self.seed(tmp_path)
        from betedge.db import Database

        db = Database(tmp_path / "t.db")
        db.settle_bet(
            db.place_bet(stake=10, price=2.22, book="draftkings"), "lost"
        )
        db.close()
        cli.main(["--config", str(cfg_path), "compare"])
        out = capsys.readouterr().out
        assert "...over how many bets" in out

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


class TestNotifyInDaily:
    """
    The scheduled-run path: scan, store, push. The scan has already cost
    credits by the time notification happens, so nothing there may lose
    it -- a dead phone reports and the run still succeeds.
    """

    def enable(self, tmp_path, **overrides):
        import yaml

        cfg_path = tmp_path / "config.yaml"
        raw = yaml.safe_load(cfg_path.read_text())
        raw["notify"] = {
            "enabled": True, "provider": "ntfy", "ntfy_topic": "t",
            "quiet_start": "", "quiet_end": "", "min_ev": 0.0,
            "min_minutes_to_start": 0.0, **overrides,
        }
        cfg_path.write_text(yaml.safe_dump(raw))
        return cfg_path

    def test_a_dry_run_says_what_it_would_send(self, wired, capsys):
        cfg_path, _client, tmp_path = wired
        self.enable(tmp_path)
        cli.main(["--config", str(cfg_path), "daily", "--dry-run-notify",
                  "--no-report"])
        assert "[dry run]" in capsys.readouterr().out

    def test_no_notify_skips_it_entirely(self, wired, capsys):
        cfg_path, _client, tmp_path = wired
        self.enable(tmp_path)
        cli.main(["--config", str(cfg_path), "daily", "--no-notify",
                  "--no-report"])
        out = capsys.readouterr().out
        assert "[dry run]" not in out
        assert "Notified" not in out

    def test_quiet_hours_are_reported_not_silent(self, wired, capsys):
        # A run that sends nothing because of the hour must say so, or it
        # is indistinguishable from a broken notifier.
        #
        # The window is built AROUND the clock rather than written down,
        # because `daily` reads the real time. A fixed "00:00 to 23:59"
        # covers every minute but one, and this test duly failed once,
        # at 23:59. An hour either side of now is quiet whenever the
        # suite runs, and still goes through the real wrapping logic.
        cfg_path, _client, tmp_path = wired
        now = datetime.now(timezone.utc).astimezone()
        self.enable(
            tmp_path,
            quiet_start=(now - timedelta(hours=1)).strftime("%H:%M"),
            quiet_end=(now + timedelta(hours=1)).strftime("%H:%M"),
        )
        cli.main(["--config", str(cfg_path), "daily", "--no-report"])
        assert "Quiet hours" in capsys.readouterr().out

    def test_a_failing_phone_does_not_fail_the_scan(self, wired, capsys,
                                                    monkeypatch):
        cfg_path, _client, tmp_path = wired
        self.enable(tmp_path)

        def explode(*a, **kw):
            raise RuntimeError("phone is off")

        monkeypatch.setattr("betedge.notify.send", explode)
        assert cli.main(["--config", str(cfg_path), "daily",
                         "--no-report"]) == 0
        out = capsys.readouterr().out
        assert "Scan #" in out                 # the scan survived
        assert "FAILED" in out and "phone is off" in out
        assert "try again" in out

    def test_no_bet_is_ever_pushed_twice(self, wired, capsys, monkeypatch):
        """
        Asserted against the ledger rather than a send count, because a
        second scan may legitimately surface a bet the first did not.
        The invariant is per-bet: one successful notification each.
        """
        from betedge.db import Database

        cfg_path, _client, tmp_path = wired
        self.enable(tmp_path)
        monkeypatch.setattr("betedge.notify.send", lambda m, c, **kw: "ntfy")
        cli.main(["--config", str(cfg_path), "daily", "--no-report"])
        capsys.readouterr()
        cli.main(["--config", str(cfg_path), "daily", "--no-report"])
        assert "already sent" in capsys.readouterr().out

        db = Database(tmp_path / "t.db")
        rows = db.conn.execute(
            "SELECT fingerprint, COUNT(*) n FROM notifications "
            "WHERE ok=1 GROUP BY fingerprint HAVING n > 1"
        ).fetchall()
        db.close()
        assert rows == [], "a bet was notified more than once"

    def test_a_suspect_bet_is_never_pushed(self, wired, capsys, monkeypatch):
        """
        The terminal shows a +17% implausible edge because it is
        interesting to look at, next to the flag saying why it is not
        staked. A phone notification carries neither the flag nor the
        context -- it reads exactly like a recommendation.
        """
        from betedge.db import Database

        cfg_path, _client, tmp_path = wired
        self.enable(tmp_path)
        monkeypatch.setattr("betedge.notify.send", lambda m, c, **kw: "ntfy")
        cli.main(["--config", str(cfg_path), "daily", "--no-report"])

        db = Database(tmp_path / "t.db")
        suspect_ids = {
            r["id"] for r in db.conn.execute(
                "SELECT id FROM opportunities WHERE suspect=1"
            ).fetchall()
        }
        notified = {
            r["opportunity_id"] for r in db.conn.execute(
                "SELECT opportunity_id FROM notifications WHERE ok=1"
            ).fetchall()
        }
        db.close()
        assert suspect_ids, "fixture should produce at least one suspect row"
        assert not (suspect_ids & notified)

    def test_notifications_are_off_unless_asked_for(self, wired, capsys,
                                                    monkeypatch):
        cfg_path, _client, _tmp = wired
        monkeypatch.setattr("betedge.notify.send",
                            lambda *a, **kw: pytest.fail("should not send"))
        cli.main(["--config", str(cfg_path), "daily", "--no-report"])


class TestNotifyCommands:
    def test_test_refuses_while_disabled_unless_forced(self, wired, capsys):
        cfg_path, _client, _tmp = wired
        assert cli.main(["--config", str(cfg_path), "notify", "test"]) == 1
        assert "disabled" in capsys.readouterr().out

    def test_a_send_failure_is_reported_not_raised(self, wired, capsys,
                                                   monkeypatch):
        cfg_path, _client, _tmp = wired

        def explode(*a, **kw):
            raise RuntimeError("no topic")

        monkeypatch.setattr("betedge.notify.send", explode)
        assert cli.main(["--config", str(cfg_path), "notify", "test",
                         "--force"]) == 1
        assert "Could not send" in capsys.readouterr().out

    def test_the_log_is_empty_before_anything_is_sent(self, wired, capsys):
        cfg_path, _client, _tmp = wired
        cli.main(["--config", str(cfg_path), "notify", "log"])
        assert "Nothing sent yet" in capsys.readouterr().out


class TestKalshiSubcommandsParse:
    """
    Every kalshi subcommand reaches its handler THROUGH THE PARSER.

    The tests below call the handlers directly, which is the wrong end:
    a handler can be perfect and the command still not exist, and that
    is exactly what a user hits -- `invalid choice: 'raw'` -- while the
    suite stays green. These assert the wiring itself.
    """

    def parse(self, argv):
        return cli.build_parser().parse_args(argv)

    def test_raw_is_wired_to_its_handler(self):
        args = self.parse(["kalshi", "raw", "KXNFLGAME-X-DET"])
        assert args.func is cli.cmd_kalshi_raw
        assert args.ticker == "KXNFLGAME-X-DET"

    def test_scan_and_calibrate_are_too(self):
        assert self.parse(["kalshi", "scan"]).func is cli.cmd_kalshi_scan
        assert self.parse(["kalshi", "calibrate"]).func \
            is cli.cmd_kalshi_calibrate

    def test_every_kalshi_subcommand_has_a_handler(self):
        # Catches a subparser added without set_defaults, which parses
        # fine and then fails at dispatch.
        parser = cli.build_parser()
        kalshi = [a for a in parser._subparsers._group_actions[0].choices.items()
                  if a[0] == "kalshi"][0][1]
        names = list(kalshi._subparsers._group_actions[0].choices)
        assert set(names) >= {"scan", "raw", "calibrate"}
        for name in names:
            assert self.parse(["kalshi", name] + (
                ["T"] if name == "raw" else [])).func is not None


class TestKalshiRaw:
    """
    The escape hatch: print what the exchange actually sends.

    It exists because five scans in a row failed on a guess about an
    undocumented payload, and every one of those would have been a
    two-minute fix with the real thing in hand.
    """

    def stub(self, monkeypatch, market_fails=False):
        from betedge import kalshi

        calls = []

        class Stub:
            requests_made = 0

            def __init__(self, **kwargs):
                pass

            def raw(self, path, params=None):
                calls.append(path)
                Stub.requests_made += 1
                if path.endswith("/orderbook"):
                    return {"orderbook": {"yes": [[55, 10]], "no": [[42, 25]]}}
                if path.startswith("/markets/"):
                    if market_fails:
                        raise kalshi.KalshiError("404 not a market")
                    return {"market": {"ticker": "M"}}
                return {"markets": [{"ticker": "EV-DET"}, {"ticker": "EV-BUF"}]}

        monkeypatch.setattr("betedge.kalshi.MarketData", Stub)
        return calls

    def args(self, ticker="M"):
        import types

        return types.SimpleNamespace(ticker=ticker, path=None, limit=100000,
                                     base_url=None)

    def cfg(self):
        import types

        return types.SimpleNamespace(
            kalshi=types.SimpleNamespace(base_url="x"))

    def test_a_market_ticker_dumps_the_market_and_its_book(self, monkeypatch,
                                                           capsys):
        calls = self.stub(monkeypatch)
        assert cli.cmd_kalshi_raw(self.cfg(), self.args()) == 0
        assert calls == ["/markets/M", "/markets/M/orderbook"]

    def test_an_event_ticker_resolves_to_a_market(self, monkeypatch, capsys):
        """
        A Kalshi URL gives the EVENT ticker. Demanding the market ticker
        would put the operator back in the guessing business that this
        command exists to end.
        """
        calls = self.stub(monkeypatch, market_fails=True)
        assert cli.cmd_kalshi_raw(self.cfg(), self.args("EV")) == 0
        assert calls == ["/markets/EV", "/events/EV", "/markets/EV-DET/orderbook"]
        out = capsys.readouterr().out
        assert "is an EVENT holding 2 market(s)" in out

    def test_a_ticker_that_is_neither_says_so(self, monkeypatch, capsys):
        from betedge import kalshi

        class Stub:
            requests_made = 0

            def __init__(self, **kwargs):
                pass

            def raw(self, path, params=None):
                raise kalshi.KalshiError("404")

        monkeypatch.setattr("betedge.kalshi.MarketData", Stub)
        assert cli.cmd_kalshi_raw(self.cfg(), self.args("NOPE")) == 1
        assert "neither a market nor an event" in capsys.readouterr().out


class TestNotifyMoves:
    """
    The one alert here that is genuinely time-critical.

    A bet at +4% is still +4% in an hour. A line that is stale because
    news broke twenty minutes ago is worth nothing the moment the book
    updates it, so the message pushes what CHANGED and how fast.
    """

    T0 = datetime(2026, 9, 27, 16, 0, tzinfo=timezone.utc)

    def leg(self, selection, line=4.5, fair=0.50, sharp_line=4.5,
            commence="2026-09-27T20:00:00Z"):
        import types

        return types.SimpleNamespace(
            event_id="mia-sf", market="player_receptions",
            selection=selection, side="Over", line=line, book="prizepicks",
            book_price=None, fair_prob=fair, push_prob=0.0,
            sharp_line=sharp_line, sharp_price_taken=1.95,
            line_source="exact", commence_time=commence)

    def moves(self, tmp_path, first, second, now=None):
        from betedge.db import Database

        db = Database(tmp_path / "t.db")
        db.record_prop_lines(first, "americanfootball_nfl", observed_at=self.T0)
        db.record_prop_lines(second, "americanfootball_nfl",
                             observed_at=self.T0 + timedelta(minutes=20))
        now = now or self.T0 + timedelta(minutes=25)
        return db, db.line_movements(min_drift=0.04, now=now)

    def cfg(self, **over):
        import types

        fields = dict(
            enabled=True, provider="ntfy", quiet_start="", quiet_end="",
            min_drift=0.10, max_messages=4, min_minutes_to_start=15.0,
            resend_after_hours=12.0, resend_on_price_gain=0.03)
        fields.update(over)
        return types.SimpleNamespace(notify=types.SimpleNamespace(**fields))

    def capture(self, monkeypatch):
        sent = []
        monkeypatch.setattr("betedge.notify.send",
                            lambda m, c, **kw: (sent.append(m), "ntfy")[1])
        return sent

    def test_a_big_drift_is_pushed(self, tmp_path, monkeypatch):
        sent = self.capture(monkeypatch)
        db, moves = self.moves(tmp_path, [self.leg("A", fair=0.505)],
                               [self.leg("A", fair=0.671, sharp_line=6.5)])
        stats = cli.notify_moves(self.cfg(), db, moves,
                                 now=self.T0 + timedelta(minutes=25))
        assert stats["sent"] == 1
        assert "Jauan" not in sent[0].body
        assert "A Over 4.5" in sent[0].body
        assert "take Over" in sent[0].body
        assert "50% -> 67%" in sent[0].body

    def test_the_strong_form_says_the_sharp_book_moved(self, tmp_path,
                                                       monkeypatch):
        sent = self.capture(monkeypatch)
        db, moves = self.moves(tmp_path, [self.leg("A", fair=0.50)],
                               [self.leg("A", fair=0.68, sharp_line=6.5)])
        cli.notify_moves(self.cfg(), db, moves,
                         now=self.T0 + timedelta(minutes=25))
        assert "sharp book moved 4.5 -> 6.5" in sent[0].body
        assert sent[0].priority == 5

    def test_a_small_drift_is_below_the_push_bar(self, tmp_path, monkeypatch):
        # Listed in the terminal at 4%, not worth a phone going off.
        sent = self.capture(monkeypatch)
        db, moves = self.moves(tmp_path, [self.leg("A", fair=0.512)],
                               [self.leg("A", fair=0.560)])
        stats = cli.notify_moves(self.cfg(), db, moves,
                                 now=self.T0 + timedelta(minutes=25))
        assert stats["sent"] == 0 and stats["below_bar"] == 1
        assert sent == []

    def test_the_same_line_does_not_buzz_twice(self, tmp_path, monkeypatch):
        self.capture(monkeypatch)
        db, moves = self.moves(tmp_path, [self.leg("A", fair=0.50)],
                               [self.leg("A", fair=0.68, sharp_line=6.5)])
        now = self.T0 + timedelta(minutes=25)
        assert cli.notify_moves(self.cfg(), db, moves, now=now)["sent"] == 1
        again = cli.notify_moves(self.cfg(), db, moves,
                                 now=now + timedelta(minutes=1))
        assert again["sent"] == 0 and again["suppressed"] == 1

    def test_a_game_about_to_start_is_not_pushed(self, tmp_path, monkeypatch):
        # A stale line you cannot reach in time is not an edge.
        self.capture(monkeypatch)
        db, moves = self.moves(
            tmp_path,
            [self.leg("A", fair=0.50, commence="2026-09-27T16:30:00Z")],
            [self.leg("A", fair=0.68, sharp_line=6.5,
                      commence="2026-09-27T16:30:00Z")])
        stats = cli.notify_moves(self.cfg(), db, moves,
                                 now=self.T0 + timedelta(minutes=25))
        assert stats["sent"] == 0 and stats["too_soon"] == 1

    def test_quiet_hours_hold_it_rather_than_dropping_it(self, tmp_path,
                                                         monkeypatch):
        self.capture(monkeypatch)
        db, moves = self.moves(tmp_path, [self.leg("A", fair=0.50)],
                               [self.leg("A", fair=0.68)])
        now = (self.T0 + timedelta(minutes=25)).astimezone()
        cfg = self.cfg(
            quiet_start=(now - timedelta(hours=1)).strftime("%H:%M"),
            quiet_end=(now + timedelta(hours=1)).strftime("%H:%M"))
        stats = cli.notify_moves(cfg, db, moves,
                                 now=self.T0 + timedelta(minutes=25))
        assert stats["quiet"] and stats["sent"] == 0

    def test_disabled_notifications_do_nothing(self, tmp_path, monkeypatch):
        self.capture(monkeypatch)
        db, moves = self.moves(tmp_path, [self.leg("A", fair=0.50)],
                               [self.leg("A", fair=0.68)])
        assert cli.notify_moves(self.cfg(enabled=False), db, moves,
                                now=self.T0 + timedelta(minutes=25))["sent"] == 0

    def test_a_dead_phone_does_not_raise(self, tmp_path, monkeypatch):
        def explode(*a, **kw):
            raise RuntimeError("phone is off")

        monkeypatch.setattr("betedge.notify.send", explode)
        db, moves = self.moves(tmp_path, [self.leg("A", fair=0.50)],
                               [self.leg("A", fair=0.68)])
        stats = cli.notify_moves(self.cfg(), db, moves,
                                 now=self.T0 + timedelta(minutes=25))
        assert stats["errors"] == 1 and stats["sent"] == 0


class TestSampleNotifications:
    """
    Rehearsing the real alert, not a stand-in for it.

    A test that hand-writes its own text proves only that a phone can
    receive text -- which is exactly how a notification reading
    "Stake 5" passed its own test for weeks.
    """

    def test_a_sample_bet_goes_through_the_real_formatter(self):
        message = cli._sample_message("bet")
        assert "Deebo Samuel Over 3.5" in message.body
        assert "draftkings" in message.body
        assert "+110" in message.body

    def test_a_sample_move_carries_the_drift_and_the_side(self):
        message = cli._sample_message("move")
        assert "Jauan Jennings Over 4.5" in message.body
        assert "take Over" in message.body
        assert "50% -> 67%" in message.body
        assert "sharp book moved 4.5 -> 6.5" in message.body

    def test_a_sample_move_breaks_through(self):
        assert cli._sample_message("move").priority == 5

    def test_every_sample_says_TEST_on_the_first_body_line(self):
        """
        On the FIRST BODY LINE, not only in the title.

        A title is an HTTP header and can be dropped in transit -- the
        whole reason the bet moved into the body. A fake alert arriving
        without its label is a fake alert someone acts on.
        """
        for kind in ("bet", "move"):
            first = cli._sample_message(kind).body.split("\n")[0]
            assert "TEST" in first, kind

    def test_the_plain_sample_is_unchanged(self):
        message = cli._sample_message("plain")
        assert "title channel OK" in message.title

    def test_an_unknown_kind_falls_back_to_plain(self):
        assert "title channel OK" in cli._sample_message("nonsense").title
