"""
Measuring the margin model against finished games.

The test that matters is the last class: a model priced from a MEASURED
sigma must reproduce the frequencies the games actually produced. That is
the only check here that could have caught the bias in the per-game fit,
and it is the reason this module exists.
"""

import csv
import math
import random
import statistics

import pytest

from betedge import calibrate as C
from betedge import ladder as L


def games(n=2000, sigma=13.0, spread=6.5, seed=7, lumps=None):
    """Synthetic games with a known sigma, for testing the measurement."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        margin = round(rng.gauss(spread, sigma))
        if lumps and rng.random() < lumps.get("share", 0.0):
            margin = lumps["at"]
        out.append(C.Game(season=2020, margin=int(margin), spread=spread))
    return out


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


class TestReading:
    def write(self, tmp_path, rows):
        path = tmp_path / "games.csv"
        fields = ["season", "game_type", "result", "spread_line",
                  "home_moneyline", "away_moneyline"]
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return path

    def test_it_reads_a_finished_game(self, tmp_path):
        path = self.write(tmp_path, [
            {"season": 2023, "game_type": "REG", "result": -1,
             "spread_line": 4, "home_moneyline": -200,
             "away_moneyline": 170},
        ])
        (game,) = C.read_nflverse_games(path)
        assert game.margin == -1
        assert game.spread == 4
        assert game.residual == -5

    def test_an_unplayed_game_is_skipped(self, tmp_path):
        path = self.write(tmp_path, [
            {"season": 2026, "game_type": "REG", "result": "",
             "spread_line": 3},
        ])
        assert C.read_nflverse_games(path) == []

    def test_a_game_with_no_closing_spread_is_skipped(self, tmp_path):
        # The residual is the measurement; without a line there is none.
        path = self.write(tmp_path, [
            {"season": 2001, "game_type": "REG", "result": 7,
             "spread_line": ""},
        ])
        assert C.read_nflverse_games(path) == []

    def test_playoffs_are_excluded_by_default(self, tmp_path):
        # Neutral site, rested, single elimination: a different
        # distribution, and there are seven thousand regular-season
        # games without muddying them.
        path = self.write(tmp_path, [
            {"season": 2023, "game_type": "POST", "result": 3,
             "spread_line": 1},
        ])
        assert C.read_nflverse_games(path) == []
        assert len(C.read_nflverse_games(path, regular_season_only=False)) == 1

    def test_a_missing_moneyline_is_not_fatal(self, tmp_path):
        path = self.write(tmp_path, [
            {"season": 2003, "game_type": "REG", "result": 7,
             "spread_line": 3, "home_moneyline": "", "away_moneyline": ""},
        ])
        (game,) = C.read_nflverse_games(path)
        assert game.home_moneyline is None


# --------------------------------------------------------------------------
# The measurements
# --------------------------------------------------------------------------


class TestMeasurement:
    def test_it_recovers_a_sigma_it_was_given(self):
        sigma, bias = C.measure_sigma(games(n=6000, sigma=13.0))
        assert sigma == pytest.approx(13.0, abs=0.5)
        assert abs(bias) < 0.5

    def test_it_recovers_a_different_sigma_too(self):
        sigma, _bias = C.measure_sigma(games(n=6000, sigma=7.0, seed=3))
        assert sigma == pytest.approx(7.0, abs=0.4)

    def test_a_line_that_is_off_centre_shows_up_as_bias(self):
        # A sharp closing line should be centred. If this is not near
        # zero on real data, the sign convention is wrong somewhere.
        skewed = [C.Game(season=2020, margin=g.margin + 6, spread=g.spread)
                  for g in games(n=3000)]
        _sigma, bias = C.measure_sigma(skewed)
        assert bias == pytest.approx(6.0, abs=0.5)
        assert not C.Calibration(sport="x", bias=bias, sigma=13).line_is_unbiased

    def test_too_few_games_is_an_error_not_a_number(self):
        with pytest.raises(ValueError, match="not enough games"):
            C.measure_sigma(games(n=1))

    def test_it_finds_a_lump_that_was_put_there(self):
        # 12% of games forced onto a margin of 3.
        lumpy = games(n=8000, lumps={"at": 3, "share": 0.12})
        sigma, _ = C.measure_sigma(lumpy)
        bumps = C.measure_key_numbers(lumpy, sigma)
        assert 3.0 in bumps
        assert bumps[3.0] > 1.0

    def test_it_does_not_invent_lumps_that_are_not_there(self):
        smooth = games(n=8000)
        bumps = C.measure_key_numbers(smooth, 13.0)
        # A few may scrape the threshold by chance; none should be large.
        assert all(abs(b) < 1.0 for b in bumps.values())

    def test_the_far_tail_cannot_produce_a_key_number(self):
        """
        The guard that was added after the first run. Out past thirty
        points the smooth expectation is a fraction of a game, so a
        handful of blowouts yields a '+150% key number' that is pure
        sampling noise -- and it would then be applied to every ladder.
        """
        lumpy = games(n=8000, lumps={"at": 45, "share": 0.004})
        bumps = C.measure_key_numbers(lumpy, 13.0)
        assert 45.0 not in bumps

    def test_a_negative_bump_is_kept(self):
        # Nine points is genuinely rare in the NFL. A model that knows
        # only about the peaks overprices the gaps between them.
        lumpy = games(n=8000, lumps={"at": 3, "share": 0.15})
        bumps = C.measure_key_numbers(lumpy, 13.0)
        assert any(b < 0 for b in bumps.values())

    def test_sigma_must_be_positive_to_measure_lumps_against(self):
        with pytest.raises(ValueError, match="positive"):
            C.measure_key_numbers(games(n=100), 0.0)

    def test_sigma_by_spread_reports_each_band(self):
        mixed = games(n=3000, spread=2.0) + games(n=3000, spread=11.0, seed=9)
        bands = C.measure_sigma_by_spread(mixed)
        assert "0-3" in bands and "10-14" in bands


# --------------------------------------------------------------------------
# The whole thing
# --------------------------------------------------------------------------


class TestCalibration:
    def test_it_reports_what_it_measured(self):
        result = C.calibrate(games(n=4000))
        assert result.games == 4000
        assert result.sigma == pytest.approx(13.0, abs=0.5)
        assert result.seasons == (2020, 2020)

    def test_nothing_to_calibrate_is_an_error(self):
        with pytest.raises(ValueError, match="no games"):
            C.calibrate([])

    def test_the_yaml_it_writes_loads_back(self, tmp_path):
        result = C.calibrate(games(n=4000), sport="americanfootball_nfl")
        path = tmp_path / "priors.yaml"
        path.write_text("sports:\n" + C.to_yaml_block(result))
        priors = L.PriorSet.load(path)
        prior = priors.get("americanfootball_nfl")
        assert prior.verified is True
        assert prior.sigma_measured == pytest.approx(result.sigma)

    def test_a_measured_prior_prices_the_ladder(self, tmp_path):
        result = C.calibrate(games(n=4000))
        path = tmp_path / "priors.yaml"
        path.write_text("sports:\n" + C.to_yaml_block(result))
        priors = L.PriorSet.load(path)
        model = L.fit(6.5, 0.70, "americanfootball_nfl", priors)
        assert model.sigma_source == "measured"
        assert model.sigma == pytest.approx(result.sigma)
        assert "margin_priors_unverified" not in model.flags

    def test_a_wildly_different_moneyline_is_flagged_not_believed(self, tmp_path):
        # The moneyline is a check now, not the source. A large
        # disagreement usually means a stale price or a mismatched
        # event rather than an unusual game.
        result = C.calibrate(games(n=4000))
        path = tmp_path / "priors.yaml"
        path.write_text("sports:\n" + C.to_yaml_block(result))
        priors = L.PriorSet.load(path)
        model = L.fit(6.5, 0.95, "americanfootball_nfl", priors)
        assert model.sigma_source == "measured"
        assert any("moneyline_implies_sigma" in f for f in model.flags)


class TestTheShippedNflPriors:
    """The real ones, measured from nflverse."""

    @pytest.fixture
    def nfl(self):
        return L.PriorSet.load().get("americanfootball_nfl")

    def test_it_is_measured_not_recollected(self, nfl):
        assert nfl.verified is True
        assert nfl.sigma_measured

    def test_sigma_is_where_football_actually_is(self, nfl):
        # Wrong by two points either way and every outer rung is wrong.
        assert 12.5 < nfl.sigma_measured < 14.0

    def test_three_and_seven_are_the_big_lumps(self, nfl):
        assert nfl.key_numbers[3.0] > 1.0
        assert nfl.key_numbers[7.0] > 0.5
        assert nfl.key_numbers[3.0] > nfl.key_numbers[7.0]

    def test_the_numbers_i_guessed_wrong_are_gone(self, nfl):
        # Written from recollection the file claimed 6 and 4. Measured,
        # they sit slightly BELOW a smooth fit and are not key numbers.
        assert nfl.key_numbers.get(6.0, 0) <= 0
        assert nfl.key_numbers.get(4.0, 0) <= 0

    def test_every_lump_is_reachable_by_threes_and_sevens(self, nfl):
        """
        The structural check, and the reason to believe the measurement
        rather than merely accept it: every POSITIVE bump is a score
        football can actually produce from touchdowns and field goals,
        and the troughs are the numbers between them.
        """
        reachable = {7 * a + 3 * b for a in range(8) for b in range(12)}
        for margin, bump in nfl.key_numbers.items():
            if bump > 0:
                assert int(margin) in reachable, margin

    def test_a_model_from_them_is_still_a_distribution(self, nfl):
        priors = L.PriorSet.load()
        model = L.fit(6.5, 0.70, "americanfootball_nfl", priors)
        assert sum(model.pmf().values()) == pytest.approx(1.0, abs=1e-9)
        previous = 1.0
        for threshold in [x * 0.5 for x in range(-40, 61)]:
            p = model.prob_margin_over(threshold)
            assert p <= previous + 1e-12
            previous = p


class TestComparingWithWhatIsLoaded:
    """
    A calibration run that always ends "paste this in" hands you work
    that is already done, and after the second time you stop reading the
    output. So it has to know when there is nothing to say.
    """

    @pytest.fixture
    def measured(self):
        return C.calibrate(games(n=4000))

    def test_a_matching_file_produces_no_differences(self, measured, tmp_path):
        path = tmp_path / "priors.yaml"
        path.write_text("sports:\n" + C.to_yaml_block(measured))
        prior = L.PriorSet.load(path).get("americanfootball_nfl")
        assert C.compare_with(measured, prior) == []

    def test_the_shipped_nfl_priors_match_a_fresh_measurement(self):
        """
        The end-to-end check: re-measuring from nflverse must reproduce
        what is committed. If this ever fails, either the upstream data
        moved or the committed numbers were edited by hand.
        """
        import os

        source = "/home/user/nflverse/nfldata/data/games.csv"
        if not os.path.exists(source):
            pytest.skip("nflverse checkout not present")
        fresh = C.calibrate(C.read_nflverse_games(source))
        prior = L.PriorSet.load().get("americanfootball_nfl")
        assert C.compare_with(fresh, prior) == []

    def test_a_changed_sigma_is_reported(self, measured, tmp_path):
        path = tmp_path / "priors.yaml"
        path.write_text(
            ("sports:\n" + C.to_yaml_block(measured))
            .replace(f"sigma_measured: {measured.sigma:g}",
                     "sigma_measured: 9.5")
        )
        prior = L.PriorSet.load(path).get("americanfootball_nfl")
        differences = C.compare_with(measured, prior)
        assert any("sigma" in d for d in differences)

    def test_an_unverified_file_is_reported(self, measured, tmp_path):
        path = tmp_path / "priors.yaml"
        path.write_text(("sports:\n" + C.to_yaml_block(measured))
                        .replace("verified: true", "verified: false"))
        prior = L.PriorSet.load(path).get("americanfootball_nfl")
        assert any("unverified" in d for d in C.compare_with(measured, prior))

    def test_a_missing_sport_is_reported(self, measured):
        assert C.compare_with(measured, None) == [
            "americanfootball_nfl is not in the priors file at all"
        ]

    def test_a_new_key_number_is_named(self, measured, tmp_path):
        path = tmp_path / "priors.yaml"
        body = "sports:\n" + C.to_yaml_block(measured)
        margin = sorted(measured.key_numbers)[0]
        bump = measured.key_numbers[margin]
        path.write_text(body.replace(f"      {int(margin)}: {bump:g}\n", "", 1))
        prior = L.PriorSet.load(path).get("americanfootball_nfl")
        differences = C.compare_with(measured, prior)
        assert any(f"margin {margin:g}: new" in d for d in differences)
