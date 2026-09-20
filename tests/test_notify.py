"""
Phone notifications.

The tests that matter are the suppression ones. A scan running every
fifteen minutes finds the same bet every fifteen minutes, and twelve
buzzes for one bet trains a person to ignore the thirteenth -- which is
the one that mattered. Nothing here touches the network.
"""

from datetime import datetime, timedelta, timezone

import pytest

from betedge import notify as N
from betedge.config import NotifyConfig
from betedge.db import Database

NOW = datetime(2026, 9, 16, 15, 0, tzinfo=timezone.utc)


def opp(**kw):
    base = {
        "id": 7, "sport": "baseball_mlb", "event_id": "evt1",
        "market": "pitcher_strikeouts", "selection": "Zack Wheeler",
        "side": "Under", "line": 6.5, "book": "draftkings",
        "soft_price": 1.91, "ev": 0.037, "recommended_stake": 15.0,
        "suspect": 0,
        "matchup": "Phillies @ Nationals",
        "commence_time": "2026-09-16T18:00:00+00:00",
    }
    base.update(kw)
    return base


class FakeSession:
    def __init__(self, fail=False):
        self.posts = []
        self.fail = fail

    def post(self, url, **kw):
        self.posts.append((url, kw))
        if self.fail:
            raise RuntimeError("phone is off")
        return _Ok()


class _Ok:
    def raise_for_status(self):
        return None


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------


class TestFingerprint:
    def test_the_same_bet_has_the_same_identity(self):
        assert N.opportunity_fingerprint(opp()) == \
            N.opportunity_fingerprint(opp())

    def test_price_is_deliberately_not_part_of_it(self):
        # A line ticking from -110 to -108 is the same bet. Including the
        # price would make every tick a new notification, which is the
        # failure this module exists to prevent.
        assert N.opportunity_fingerprint(opp(soft_price=1.91)) == \
            N.opportunity_fingerprint(opp(soft_price=1.95))

    def test_nor_is_the_recommended_stake(self):
        assert N.opportunity_fingerprint(opp(recommended_stake=5)) == \
            N.opportunity_fingerprint(opp(recommended_stake=50))

    @pytest.mark.parametrize("field,value", [
        ("selection", "Max Fried"), ("side", "Over"), ("line", 7.5),
        ("book", "fanduel"), ("event_id", "evt2"),
        ("market", "pitcher_outs"),
    ])
    def test_a_different_bet_is_a_different_identity(self, field, value):
        assert N.opportunity_fingerprint(opp()) != \
            N.opportunity_fingerprint(opp(**{field: value}))

    def test_case_and_spacing_do_not_make_a_new_bet(self):
        assert N.opportunity_fingerprint(opp(selection="  zack wheeler ")) == \
            N.opportunity_fingerprint(opp(selection="Zack Wheeler"))


# --------------------------------------------------------------------------
# Suppression
# --------------------------------------------------------------------------


class TestSuppression:
    @pytest.fixture
    def db(self, tmp_path):
        d = Database(tmp_path / "t.db")
        yield d
        d.close()

    def test_a_new_bet_is_sent(self, db):
        wanted, why = db.should_notify("fp1", 1.91, NOW)
        assert wanted and why == "new"

    def test_the_same_bet_is_not_sent_twice(self, db):
        db.record_notification("fp1", NOW, price=1.91)
        wanted, why = db.should_notify("fp1", 1.91, NOW + timedelta(minutes=15))
        assert not wanted
        assert why == "already sent"

    def test_a_materially_better_price_is_worth_saying(self, db):
        db.record_notification("fp1", NOW, price=1.91)
        wanted, why = db.should_notify(
            "fp1", 2.10, NOW + timedelta(minutes=15), resend_on_price_gain=0.05
        )
        assert wanted
        assert "price improved" in why

    def test_a_trivial_price_move_is_not(self, db):
        db.record_notification("fp1", NOW, price=1.91)
        wanted, _ = db.should_notify("fp1", 1.93, NOW + timedelta(minutes=15))
        assert not wanted

    def test_a_worse_price_is_certainly_not(self, db):
        db.record_notification("fp1", NOW, price=1.91)
        wanted, _ = db.should_notify("fp1", 1.70, NOW + timedelta(minutes=15))
        assert not wanted

    def test_it_is_resent_once_enough_time_has_passed(self, db):
        db.record_notification("fp1", NOW, price=1.91)
        wanted, why = db.should_notify(
            "fp1", 1.91, NOW + timedelta(hours=13), resend_after_hours=12
        )
        assert wanted
        assert "13h ago" in why

    def test_a_failed_send_does_not_count_as_sent(self, db):
        # The worst possible moment to go quiet is right after the phone
        # broke, so a failure must not suppress the next attempt.
        db.record_notification("fp1", NOW, price=1.91, ok=False,
                               error="phone is off")
        wanted, why = db.should_notify("fp1", 1.91, NOW + timedelta(minutes=15))
        assert wanted and why == "new"

    def test_failures_are_still_recorded_for_the_log(self, db):
        db.record_notification("fp1", NOW, ok=False, error="phone is off")
        rows = db.recent_notifications()
        assert len(rows) == 1
        assert rows[0]["ok"] == 0
        assert "phone is off" in rows[0]["error"]


# --------------------------------------------------------------------------
# Quiet hours
# --------------------------------------------------------------------------


class TestQuietHours:
    def test_a_window_that_wraps_midnight_works(self):
        # The normal configuration. Treating it as start <= t < end would
        # make 23:00-08:00 silently do nothing at all.
        at = lambda h, m=0: datetime(2026, 9, 16, h, m)
        assert N.in_quiet_hours(at(23, 30), "23:00", "08:00")
        assert N.in_quiet_hours(at(3), "23:00", "08:00")
        assert N.in_quiet_hours(at(7, 59), "23:00", "08:00")
        assert not N.in_quiet_hours(at(8), "23:00", "08:00")
        assert not N.in_quiet_hours(at(15), "23:00", "08:00")

    def test_a_window_inside_one_day_also_works(self):
        at = lambda h: datetime(2026, 9, 16, h)
        assert N.in_quiet_hours(at(10), "09:00", "17:00")
        assert not N.in_quiet_hours(at(20), "09:00", "17:00")

    def test_an_empty_window_is_never_quiet(self):
        now = datetime(2026, 9, 16, 3)
        assert not N.in_quiet_hours(now, "", "")
        assert not N.in_quiet_hours(now, None, "08:00")

    def test_an_unparseable_window_is_never_quiet(self):
        # Failing open: a typo in the config must not silence the tool
        # permanently and invisibly.
        assert not N.in_quiet_hours(datetime(2026, 9, 16, 3), "nope", "08:00")

    def test_a_zero_length_window_is_never_quiet(self):
        assert not N.in_quiet_hours(datetime(2026, 9, 16, 3), "23:00", "23:00")


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------


class TestProviders:
    def test_ntfy_posts_the_body_and_a_title_header(self):
        session = FakeSession()
        N.send_ntfy(N.Message("A title", "A body", tags=["x"]), "topic123",
                    session=session)
        url, kw = session.posts[0]
        assert url == "https://ntfy.sh/topic123"
        assert kw["data"] == b"A body"
        assert kw["headers"]["X-Title"] == "A title"
        assert kw["headers"]["Tags"] == "x"

    def test_a_title_that_cannot_be_a_header_does_not_kill_the_send(self):
        """
        Headers are latin-1. One accented letter in a player's name would
        otherwise raise inside the HTTP client and take the whole
        notification with it -- the bet is worth more than the accent.
        """
        session = FakeSession()
        N.send_ntfy(N.Message("+4.2%  Nikola Joki\u0107 Over 24.5", "b"),
                    "t", session=session)
        title = session.posts[0][1]["headers"]["X-Title"]
        title.encode("latin-1")            # the point: this must not raise
        assert "Jokic" in title

    def test_a_newline_never_reaches_a_header(self):
        session = FakeSession()
        N.send_ntfy(N.Message("one\ntwo", "b"), "t", session=session)
        assert "\n" not in session.posts[0][1]["headers"]["X-Title"]

    def test_ntfy_without_a_topic_is_an_error(self):
        with pytest.raises(N.NotifyError, match="topic"):
            N.send_ntfy(N.Message("t", "b"), "")

    def test_pushover_needs_both_credentials(self):
        with pytest.raises(N.NotifyError, match="token and a user"):
            N.send_pushover(N.Message("t", "b"), "tok", "")

    def test_telegram_needs_both_credentials(self):
        with pytest.raises(N.NotifyError, match="bot token and a chat"):
            N.send_telegram(N.Message("t", "b"), "", "chat")

    def test_send_routes_to_the_configured_provider(self):
        session = FakeSession()
        cfg = NotifyConfig(provider="ntfy", ntfy_topic="abc")
        assert N.send(N.Message("t", "b"), cfg, session=session) == "ntfy"
        assert session.posts[0][0].endswith("/abc")

    def test_an_unknown_provider_says_what_is_valid(self):
        with pytest.raises(N.NotifyError, match="unknown notification"):
            N.send(N.Message("t", "b"), NotifyConfig(provider="carrier_pigeon"))

    def test_no_provider_at_all_is_an_error_not_a_silent_skip(self):
        with pytest.raises(N.NotifyError, match="no notification provider"):
            N.send(N.Message("t", "b"), NotifyConfig(provider=""))

    def test_the_environment_beats_the_config_file(self, monkeypatch):
        # A config file gets committed by accident; an env var does not.
        monkeypatch.setenv("BETEDGE_NTFY_TOPIC", "from-env")
        session = FakeSession()
        cfg = NotifyConfig(provider="ntfy", ntfy_topic="from-file")
        N.send(N.Message("t", "b"), cfg, session=session)
        assert session.posts[0][0].endswith("/from-env")

    def test_a_custom_ntfy_server_is_honoured(self):
        session = FakeSession()
        N.send_ntfy(N.Message("t", "b"), "abc",
                    server="https://ntfy.example.com/", session=session)
        assert session.posts[0][0] == "https://ntfy.example.com/abc"


# --------------------------------------------------------------------------
# What the message says
# --------------------------------------------------------------------------


class TestMessageContent:
    def test_everything_needed_to_place_it_is_there(self):
        message = N.format_opportunity(opp(), stake=15)
        text = message.as_text()
        for needed in ("Zack Wheeler", "Under", "6.5", "draftkings", "15"):
            assert needed in text

    def test_the_edge_leads_because_it_decides_whether_to_look(self):
        assert N.format_opportunity(opp(ev=0.037)).title.startswith("+3.7%")

    def test_it_carries_the_command_to_log_the_bet(self):
        # So the bet can be recorded without hunting for the id after.
        body = N.format_opportunity(opp(), stake=15).body
        assert "bet bet 7 --stake 15" in body

    def test_a_big_edge_gets_a_higher_priority(self):
        assert N.format_opportunity(opp(ev=0.08)).priority > \
            N.format_opportunity(opp(ev=0.031)).priority

    def test_a_digest_names_the_count_and_the_best(self):
        message = N.format_digest(9, 0.061)
        assert "9 bet(s)" in message.title
        assert "+6.1%" in message.title

    def test_a_digest_LISTS_the_bets_rather_than_sending_you_to_a_terminal(self):
        """
        The whole point is acting from a lock screen. A digest reading
        "Run `bet show` for the list" tells a phone to go and ask a
        laptop what it already knows, which is the one thing a
        notification must never do.
        """
        from betedge import report as R

        rows = [
            ({"selection": "Tyler Glasnow", "side": "Over", "line": 6.5,
              "soft_price": 2.62, "book": "draftkings", "ev": 0.061}, 8),
            ({"selection": "Chris Olave", "side": "Under", "line": 5.5,
              "soft_price": 2.09, "book": "draftkings", "ev": 0.042}, 5),
        ]
        message = N.format_digest(2, 0.061, rows=rows, american=R.american)
        assert "Tyler Glasnow Over 6.5" in message.body
        assert "Chris Olave Under 5.5" in message.body
        assert "+162" in message.body          # the price, not just the edge
        assert "bet show" not in message.body  # never again

    def test_a_digest_with_no_rows_still_says_something_useful(self):
        message = N.format_digest(9, 0.061)
        assert "9 bet(s)" in message.body
