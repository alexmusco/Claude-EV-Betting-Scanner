"""
Scan profiles: named bundles of config overrides.

The property that matters most is that nothing moves silently. A profile
can reach the staking fractions and the guard thresholds, so `apply_profile`
returns every change it made and the CLI prints them — a scan running under
settings nobody stated is the same class of failure as a stale roster.
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from betedge.config import Config, load_profiles

SHIPPED = Path(__file__).resolve().parent.parent / "betedge" / "data" / "profiles.yaml"


def config_with(tmp_path, **raw) -> tuple[Config, Path]:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(raw))
    return Config.load(p), p


class TestShippedProfiles:
    def test_the_file_loads(self):
        assert "nfl-week" in load_profiles()

    def test_every_shipped_profile_explains_itself(self):
        for name, spec in load_profiles().items():
            assert spec.get("description"), f"{name} has no description"

    def test_every_shipped_profile_applies_cleanly(self):
        # A profile naming a setting that does not exist raises, so this
        # catches a typo in the shipped file before a user hits it.
        for name in load_profiles():
            Config().apply_profile(name)

    def test_nfl_week_widens_the_window_past_a_thursday_game(self):
        # A Thursday 8:15pm ET kickoff is ~57 hours out on the Tuesday
        # before it, so a 48-hour window excludes it entirely.
        cfg = Config.load(Path(__file__).resolve().parent.parent / "config.yaml")
        assert cfg.prop_window_hours("americanfootball_nfl") < 57
        cfg.apply_profile("nfl-week")
        assert cfg.prop_window_hours("americanfootball_nfl") > 57

    def test_nfl_week_uses_every_prop_market(self):
        from betedge.markets import markets_for

        cfg = Config.load(Path(__file__).resolve().parent.parent / "config.yaml")
        assert len(cfg.markets_for_sport("americanfootball_nfl")) < len(
            markets_for("americanfootball_nfl")
        )
        cfg.apply_profile("nfl-week")
        assert cfg.markets_for_sport("americanfootball_nfl") == markets_for(
            "americanfootball_nfl"
        )

    def test_nfl_week_scans_only_football(self):
        cfg = Config()
        cfg.sports = ["baseball_mlb", "americanfootball_nfl"]
        cfg.apply_profile("nfl-week")
        assert cfg.sports == ["americanfootball_nfl"]

    def test_no_shipped_profile_loosens_a_staking_cap(self):
        """
        A profile is allowed to touch staking, but none of the shipped ones
        should — a preset that quietly raised the Kelly fraction would be
        the worst possible thing to hide behind a convenience flag.
        """
        base = Config()
        for name in load_profiles():
            cfg = Config()
            cfg.apply_profile(name)
            assert cfg.parlay.kelly_multiplier <= base.parlay.kelly_multiplier
            assert cfg.parlay.max_ticket_fraction <= base.parlay.max_ticket_fraction
            assert (
                cfg.parlay.max_game_exposure_fraction
                <= base.parlay.max_game_exposure_fraction
            )
            assert cfg.bankroll.max_fraction <= base.bankroll.max_fraction

    def test_no_shipped_profile_weakens_a_guard(self):
        base = Config()
        for name in load_profiles():
            cfg = Config()
            cfg.apply_profile(name)
            assert cfg.parlay.max_plausible_ev <= base.parlay.max_plausible_ev
            assert cfg.parlay.min_ev >= base.parlay.min_ev
            assert cfg.parlay.allow_line_interpolation is False
            assert cfg.model.max_devig_spread <= base.model.max_devig_spread


class TestApplying:
    def test_it_reports_what_it_changed(self, tmp_path):
        cfg, _p = config_with(tmp_path, profiles={
            "t": {"parlay": {"max_legs": 3}},
        })
        changes = cfg.apply_profile("t")
        assert len(changes) == 1
        assert changes[0].path == "parlay.max_legs"
        assert changes[0].before == 5
        assert changes[0].after == 3
        assert "5 -> 3" in changes[0].describe()

    def test_a_setting_already_at_that_value_is_not_reported(self, tmp_path):
        cfg, _p = config_with(tmp_path, profiles={
            "t": {"parlay": {"max_legs": 5}},
        })
        assert cfg.apply_profile("t") == []

    def test_lists_are_replaced(self, tmp_path):
        cfg, _p = config_with(tmp_path, sports=["a", "b"], profiles={
            "t": {"sports": ["c"]},
        })
        cfg.apply_profile("t")
        assert cfg.sports == ["c"]

    def test_per_sport_maps_are_merged_not_replaced(self, tmp_path):
        cfg, _p = config_with(
            tmp_path,
            prop_windows={"baseball_mlb": 8, "americanfootball_nfl": 48},
            profiles={"t": {"prop_windows": {"americanfootball_nfl": 72}}},
        )
        cfg.apply_profile("t")
        assert cfg.prop_windows == {"baseball_mlb": 8, "americanfootball_nfl": 72}

    def test_a_null_removes_an_override(self, tmp_path):
        from betedge.markets import markets_for

        cfg, _p = config_with(
            tmp_path,
            prop_markets={"americanfootball_nfl": ["player_pass_yds"]},
            profiles={"t": {"prop_markets": {"americanfootball_nfl": None}}},
        )
        changes = cfg.apply_profile("t")
        assert cfg.markets_for_sport("americanfootball_nfl") == markets_for(
            "americanfootball_nfl"
        )
        assert "full registry list" in changes[0].describe()

    def test_removing_an_override_that_was_not_there_changes_nothing(self, tmp_path):
        cfg, _p = config_with(tmp_path, profiles={
            "t": {"prop_markets": {"americanfootball_nfl": None}},
        })
        assert cfg.apply_profile("t") == []

    def test_several_sections_at_once(self, tmp_path):
        cfg, _p = config_with(tmp_path, profiles={
            "t": {"model": {"min_ev": 0.05}, "bankroll": {"amount": 500}},
        })
        paths = {c.path for c in cfg.apply_profile("t")}
        assert paths == {"model.min_ev", "bankroll.amount"}
        assert cfg.model.min_ev == 0.05
        assert cfg.bankroll.amount == 500

    def test_a_description_is_not_treated_as_a_setting(self, tmp_path):
        cfg, _p = config_with(tmp_path, profiles={
            "t": {"description": "hello", "parlay": {"max_legs": 3}},
        })
        assert [c.path for c in cfg.apply_profile("t")] == ["parlay.max_legs"]


class TestProfileErrors:
    def test_an_unknown_profile_lists_what_exists(self, tmp_path):
        cfg, _p = config_with(tmp_path, profiles={"real": {}})
        with pytest.raises(ValueError, match="real"):
            cfg.apply_profile("imaginary")

    def test_a_typo_in_a_setting_is_an_error_not_a_shrug(self, tmp_path):
        # A silently ignored typo is indistinguishable from a setting that
        # did not work, which is the worst of both.
        cfg, _p = config_with(tmp_path, profiles={
            "t": {"parlay": {"max_leggs": 3}},
        })
        with pytest.raises(ValueError, match="max_leggs"):
            cfg.apply_profile("t")

    def test_the_error_names_the_valid_settings(self, tmp_path):
        cfg, _p = config_with(tmp_path, profiles={"t": {"parlay": {"nope": 1}}})
        with pytest.raises(ValueError, match="max_legs"):
            cfg.apply_profile("t")

    def test_a_section_a_profile_may_not_touch_is_refused(self, tmp_path):
        cfg, _p = config_with(tmp_path, profiles={"t": {"database": "/tmp/x.db"}})
        with pytest.raises(ValueError, match="cannot"):
            cfg.apply_profile("t")


class TestUserProfiles:
    def test_a_user_profile_is_available_alongside_the_shipped_ones(self, tmp_path):
        cfg, _p = config_with(tmp_path, profiles={"mine": {"parlay": {"max_legs": 2}}})
        assert "mine" in cfg.profiles
        assert "nfl-week" in cfg.profiles

    def test_a_user_profile_replaces_a_shipped_one_of_the_same_name(self, tmp_path):
        cfg, _p = config_with(tmp_path, profiles={
            "nfl-week": {"parlay": {"max_legs": 2}},
        })
        changes = cfg.apply_profile("nfl-week")
        # The shipped version widens the window; this one does not exist
        # any more, so nothing touches prop_windows.
        assert [c.path for c in changes] == ["parlay.max_legs"]

    def test_no_profiles_anywhere_is_not_an_error(self, tmp_path):
        cfg, _p = config_with(tmp_path)
        assert cfg.profiles
        assert load_profiles(tmp_path / "nothing.yaml")
