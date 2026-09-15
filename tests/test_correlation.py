"""
Where the correlation numbers come from, and whether the tool is honest
about it.

The tests that matter most here are not the arithmetic ones. They are the
ones asserting that a prior is never reported as a measurement, and that
taking a leg Under flips the sign of its correlations -- because getting
either wrong produces a number that looks exactly as authoritative as a
correct one.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from betedge import correlation as C
from betedge.db import Database
from fixtures import make_leg


@pytest.fixture
def priors():
    return C.PriorSet.load()


class TestPriorFile:
    def test_the_shipped_file_loads(self, priors):
        assert priors.by_sport
        assert set(priors.defaults) >= set(C.RELATIONS)

    def test_every_shipped_sport_has_entries(self, priors):
        for sport in ("americanfootball_nfl", "basketball_nba",
                      "baseball_mlb", "icehockey_nhl"):
            assert priors.entries_for(sport), f"{sport} has no priors"

    def test_every_prior_is_a_legal_correlation(self, priors):
        for entries in priors.by_sport.values():
            for e in entries:
                assert -1.0 < e.rho < 1.0
                assert e.relation in C.RELATIONS

    def test_every_prior_carries_its_reasoning(self, priors):
        # These are judgements, and a judgement without a stated reason is
        # indistinguishable from a number someone made up.
        for sport, entries in priors.by_sport.items():
            for e in entries:
                assert e.why, f"{sport} {e.markets} {e.relation} has no rationale"

    def test_a_quarterback_and_his_receiver_are_positively_correlated(self, priors):
        rho, source, _why = priors.lookup(
            "americanfootball_nfl", "player_pass_yds",
            "player_reception_yds", C.SAME_TEAM,
        )
        assert rho > 0.3
        assert source == C.SOURCE_PRIOR

    def test_two_backs_in_one_backfield_are_negatively_correlated(self, priors):
        rho, source, _why = priors.lookup(
            "americanfootball_nfl", "player_rush_yds",
            "player_rush_yds", C.SAME_TEAM,
        )
        assert rho < 0
        assert source == C.SOURCE_PRIOR

    def test_market_order_does_not_matter(self, priors):
        a = priors.lookup("americanfootball_nfl", "player_pass_yds",
                          "player_reception_yds", C.SAME_TEAM)
        b = priors.lookup("americanfootball_nfl", "player_reception_yds",
                          "player_pass_yds", C.SAME_TEAM)
        assert a == b

    def test_a_wildcard_covers_anything_against_the_game_total(self, priors):
        rho, source, _why = priors.lookup(
            "basketball_nba", "player_steals", "totals", C.SAME_GAME
        )
        assert rho > 0
        assert source == C.SOURCE_PRIOR

    def test_a_named_pair_beats_the_wildcard(self, priors):
        named, _s, _w = priors.lookup(
            "americanfootball_nfl", "player_pass_yds", "totals", C.SAME_GAME
        )
        wild, _s2, _w2 = priors.lookup(
            "americanfootball_nfl", "player_tackles_assists", "totals", C.SAME_GAME
        )
        assert named != wild
        assert named > wild

    def test_an_unknown_sport_falls_back_to_the_defaults(self, priors):
        rho, source, _why = priors.lookup(
            "curling_whatever", "player_points", "player_points", C.SAME_TEAM
        )
        assert source == C.SOURCE_DEFAULT
        assert rho == priors.defaults[C.SAME_TEAM]

    def test_different_games_default_to_no_correlation(self, priors):
        rho, source, _why = priors.lookup(
            "americanfootball_nfl", "player_pass_yds",
            "player_reception_yds", C.CROSS_GAME,
        )
        assert rho == 0.0
        assert source == C.SOURCE_DEFAULT

    def test_a_malformed_prior_file_is_rejected(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "defaults: {}\npriors:\n  nfl:\n    - markets: [a, b, c]\n"
            "      relation: same_team\n      rho: 0.2\n"
        )
        with pytest.raises(ValueError, match="exactly two markets"):
            C.PriorSet.load(bad)

    def test_an_out_of_range_rho_is_rejected(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "priors:\n  nfl:\n    - markets: [a, b]\n"
            "      relation: same_team\n      rho: 1.4\n"
        )
        with pytest.raises(ValueError, match="inside"):
            C.PriorSet.load(bad)

    def test_an_unknown_relation_is_rejected(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "priors:\n  nfl:\n    - markets: [a, b]\n"
            "      relation: telepathy\n      rho: 0.2\n"
        )
        with pytest.raises(ValueError, match="unknown relation"):
            C.PriorSet.load(bad)


class TestRelations:
    def test_two_games_are_never_related(self):
        a = make_leg(event_id="one")
        b = make_leg(event_id="two", selection="Somebody Else")
        assert C.relation_between(a, b) == C.CROSS_GAME

    def test_one_player_twice_is_the_same_player(self):
        a = make_leg(market="player_pass_yds")
        b = make_leg(market="player_pass_tds")
        assert C.relation_between(a, b) == C.SAME_PLAYER

    def test_player_names_match_case_insensitively(self):
        a = make_leg(selection="Patrick Mahomes")
        b = make_leg(selection="patrick mahomes", market="player_pass_tds")
        assert C.relation_between(a, b) == C.SAME_PLAYER

    def test_known_teams_separate_team_mates_from_opponents(self):
        a = make_leg(selection="A", team="Chiefs")
        b = make_leg(selection="B", team="Chiefs")
        c = make_leg(selection="C", team="Broncos")
        assert C.relation_between(a, b) == C.SAME_TEAM
        assert C.relation_between(a, c) == C.OPPOSING_TEAM

    def test_without_a_roster_it_refuses_to_guess(self):
        # The API does not say which team a player is on, and guessing
        # would put the wrong SIGN on the most important priors there are.
        a = make_leg(selection="A", team=None)
        b = make_leg(selection="B", team=None)
        assert C.relation_between(a, b) == C.SAME_GAME

    def test_a_half_known_roster_still_refuses(self):
        a = make_leg(selection="A", team="Chiefs")
        b = make_leg(selection="B", team=None)
        assert C.relation_between(a, b) == C.SAME_GAME


class TestSideSign:
    def test_over_and_yes_keep_the_stated_direction(self):
        assert C.side_sign("Over") == 1
        assert C.side_sign("Yes") == 1

    def test_under_and_no_reverse_it(self):
        assert C.side_sign("Under") == -1
        assert C.side_sign("No") == -1

    def test_a_competitor_named_side_has_no_orientation_to_flip(self):
        assert C.side_sign("Kansas City Chiefs") == 1
        assert C.side_sign(None) == 1


class TestAssembly:
    def test_a_two_leg_matrix_is_symmetric_with_a_unit_diagonal(self, priors):
        legs = [make_leg(selection="A", team="KC", market="player_pass_yds"),
                make_leg(selection="B", team="KC", market="player_reception_yds")]
        m = C.assemble(legs, priors).matrix
        assert np.allclose(np.diag(m), 1.0)
        assert np.allclose(m, m.T)

    def test_the_prior_is_applied_at_its_stated_sign_for_two_overs(self, priors):
        legs = [make_leg(selection="A", team="KC", market="player_pass_yds"),
                make_leg(selection="B", team="KC", market="player_reception_yds")]
        built = C.assemble(legs, priors)
        assert built.matrix[0, 1] == pytest.approx(0.45)
        assert built.pairs[0].source == C.SOURCE_PRIOR

    def test_taking_one_leg_under_flips_the_sign(self, priors):
        legs = [make_leg(selection="A", team="KC", market="player_pass_yds"),
                make_leg(selection="B", team="KC",
                         market="player_reception_yds", side="Under")]
        built = C.assemble(legs, priors)
        assert built.matrix[0, 1] == pytest.approx(-0.45)
        assert built.pairs[0].sign == -1
        assert built.pairs[0].base_rho == pytest.approx(0.45)

    def test_both_legs_under_restores_the_positive_sign(self, priors):
        legs = [make_leg(selection="A", team="KC", market="player_pass_yds",
                         side="Under"),
                make_leg(selection="B", team="KC",
                         market="player_reception_yds", side="Under")]
        assert C.assemble(legs, priors).matrix[0, 1] == pytest.approx(0.45)

    def test_correlations_are_clamped_below_one(self, priors, tmp_path):
        extreme = tmp_path / "p.yaml"
        extreme.write_text(
            "priors:\n  americanfootball_nfl:\n    - markets: [a, b]\n"
            "      relation: same_team\n      rho: 0.99\n      why: test\n"
        )
        legs = [make_leg(selection="A", team="KC", market="a"),
                make_leg(selection="B", team="KC", market="b")]
        built = C.assemble(legs, C.PriorSet.load(extreme), max_abs_rho=0.95)
        assert built.matrix[0, 1] == pytest.approx(0.95)

    def test_an_inconsistent_set_of_priors_is_projected(self, priors, tmp_path):
        impossible = tmp_path / "p.yaml"
        impossible.write_text(
            "priors:\n  americanfootball_nfl:\n"
            "    - {markets: [a, b], relation: same_team, rho: 0.95, why: t}\n"
            "    - {markets: [b, c], relation: same_team, rho: 0.95, why: t}\n"
            "    - {markets: [a, c], relation: same_team, rho: -0.95, why: t}\n"
        )
        legs = [make_leg(selection=n, team="KC", market=m)
                for n, m in (("A", "a"), ("B", "b"), ("C", "c"))]
        built = C.assemble(legs, C.PriorSet.load(impossible))
        assert built.projected
        assert "not positive semi-definite" in built.psd.note

    def test_it_reports_when_nothing_rests_on_data(self, priors):
        legs = [make_leg(selection="A", team="KC", market="player_pass_yds"),
                make_leg(selection="B", team="KC", market="player_reception_yds")]
        built = C.assemble(legs, priors)
        assert built.all_prior
        assert not built.any_measured
        assert "prior" in built.summary()

    def test_a_well_evidenced_estimate_displaces_the_prior(self, priors):
        store = C.EstimateStore([{
            "sport": "americanfootball_nfl",
            "market_a": "player_pass_yds",
            "market_b": "player_reception_yds",
            "relation": C.SAME_TEAM,
            "rho": 0.61, "spearman": 0.6,
            "n_observations": 500, "n_games": 500,
            "fitted_at": "2026-09-01T00:00:00+00:00", "note": "",
        }])
        legs = [make_leg(selection="A", team="KC", market="player_pass_yds"),
                make_leg(selection="B", team="KC", market="player_reception_yds")]
        built = C.assemble(legs, priors, store, min_sample=100)
        assert built.matrix[0, 1] == pytest.approx(0.61)
        assert built.pairs[0].source == C.SOURCE_EMPIRICAL
        assert built.pairs[0].sample_size == 500
        assert built.any_measured and not built.all_prior
        assert "measured" in built.pairs[0].describe()

    def test_a_thin_estimate_does_not(self, priors):
        store = C.EstimateStore([{
            "sport": "americanfootball_nfl",
            "market_a": "player_pass_yds",
            "market_b": "player_reception_yds",
            "relation": C.SAME_TEAM,
            "rho": 0.61, "spearman": 0.6,
            "n_observations": 12, "n_games": 6,
            "fitted_at": "2026-09-01T00:00:00+00:00", "note": "",
        }])
        legs = [make_leg(selection="A", team="KC", market="player_pass_yds"),
                make_leg(selection="B", team="KC", market="player_reception_yds")]
        built = C.assemble(legs, priors, store, min_sample=100)
        assert built.matrix[0, 1] == pytest.approx(0.45)
        assert built.pairs[0].source == C.SOURCE_PRIOR
        assert built.all_prior

    def test_the_most_negative_pair_is_findable(self, priors):
        legs = [
            make_leg(selection="A", team="KC", market="player_rush_yds"),
            make_leg(selection="B", team="KC", market="player_rush_yds"),
            make_leg(selection="C", team="KC", market="player_pass_yds"),
        ]
        built = C.assemble(legs, priors)
        assert built.most_negative.rho < 0
        assert built.strongest is not None


class TestSpearman:
    def test_a_perfect_monotone_relationship_scores_one(self):
        assert C.spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(1.0)

    def test_a_perfect_inverse_scores_minus_one(self):
        assert C.spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == pytest.approx(-1.0)

    def test_it_is_immune_to_a_monotone_transform(self):
        # The reason the fit goes through ranks: one 200-yard game should
        # not dominate the estimate.
        xs = [1, 2, 3, 4, 5, 6]
        ys = [2, 4, 6, 8, 10, 12]
        assert C.spearman(xs, ys) == pytest.approx(
            C.spearman(xs, [y ** 3 for y in ys])
        )

    def test_a_constant_column_has_no_correlation(self):
        assert C.spearman([1, 2, 3, 4], [5, 5, 5, 5]) == 0.0

    def test_too_few_points_returns_zero(self):
        assert C.spearman([1, 2], [3, 4]) == 0.0

    def test_ties_are_averaged(self):
        assert abs(C.spearman([1, 1, 2, 3], [1, 1, 2, 3])) == pytest.approx(1.0)


class TestSpearmanToLatent:
    def test_the_endpoints(self):
        assert C.spearman_to_latent(0.0) == 0.0
        assert C.spearman_to_latent(1.0) == pytest.approx(1.0)
        assert C.spearman_to_latent(-1.0) == pytest.approx(-1.0)

    def test_it_matches_the_identity(self):
        for rho_s in (-0.7, -0.2, 0.3, 0.85):
            assert C.spearman_to_latent(rho_s) == pytest.approx(
                2 * math.sin(math.pi * rho_s / 6)
            )

    def test_the_latent_correlation_exceeds_the_rank_one(self):
        assert C.spearman_to_latent(0.5) > 0.5


class TestGameLogFitting:
    def write_logs(self, tmp_path, rows, header=None):
        p = tmp_path / "logs.csv"
        header = header or "game_id,sport,player,team,market,value"
        p.write_text(header + "\n" + "\n".join(rows) + "\n")
        return p

    def test_it_reads_a_well_formed_log(self, tmp_path):
        p = self.write_logs(tmp_path, [
            "g1,americanfootball_nfl,Mahomes,KC,player_pass_yds,300",
            "g1,americanfootball_nfl,Kelce,KC,player_reception_yds,90",
        ])
        rows = C.read_game_logs(p)
        assert len(rows) == 2
        assert rows[0].market == "player_pass_yds"

    def test_a_missing_column_is_named(self, tmp_path):
        p = self.write_logs(tmp_path, ["g1,Mahomes,KC,player_pass_yds,300"],
                            header="game_id,player,team,market,value")
        with pytest.raises(ValueError, match="sport"):
            C.read_game_logs(p)

    def test_unparseable_values_are_skipped_not_fatal(self, tmp_path):
        p = self.write_logs(tmp_path, [
            "g1,nfl,Mahomes,KC,player_pass_yds,not_a_number",
            "g1,nfl,Kelce,KC,player_reception_yds,90",
        ])
        assert len(C.read_game_logs(p)) == 1

    def synthetic(self, n_games=200, rho=0.8, seed=1):
        """Correlated team mates, so the fit has a known right answer."""
        rng = np.random.default_rng(seed)
        draws = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n_games)
        rows = []
        for i, (a, b) in enumerate(draws):
            rows.append(C.GameLogRow(f"g{i}", "americanfootball_nfl", "QB", "KC",
                                     "player_pass_yds", 250 + 40 * a))
            rows.append(C.GameLogRow(f"g{i}", "americanfootball_nfl", "WR", "KC",
                                     "player_reception_yds", 60 + 20 * b))
        return rows

    def test_it_recovers_a_known_correlation(self):
        fitted = C.fit_correlations(self.synthetic(n_games=2000, rho=0.8))
        pair = [e for e in fitted if e.relation == C.SAME_TEAM][0]
        assert pair.rho == pytest.approx(0.8, abs=0.05)
        assert pair.n_observations == 2000
        assert pair.n_games == 2000

    def test_it_recovers_a_negative_one(self):
        fitted = C.fit_correlations(self.synthetic(n_games=2000, rho=-0.5))
        pair = [e for e in fitted if e.relation == C.SAME_TEAM][0]
        assert pair.rho == pytest.approx(-0.5, abs=0.05)

    def test_opposing_players_are_bucketed_separately(self):
        rows = [
            C.GameLogRow("g1", "nfl", "A", "KC", "player_pass_yds", 300),
            C.GameLogRow("g1", "nfl", "B", "KC", "player_reception_yds", 90),
            C.GameLogRow("g1", "nfl", "C", "DEN", "player_reception_yds", 40),
        ] * 5
        relations = {e.relation for e in C.fit_correlations(rows)}
        assert C.SAME_TEAM in relations
        assert C.OPPOSING_TEAM in relations

    def test_one_player_two_stats_is_the_same_player_bucket(self):
        rows = []
        for i in range(50):
            rows += [
                C.GameLogRow(f"g{i}", "nfl", "A", "KC", "player_pass_yds", 200 + i),
                C.GameLogRow(f"g{i}", "nfl", "A", "KC", "player_pass_tds", i % 4),
            ]
        assert any(e.relation == C.SAME_PLAYER for e in C.fit_correlations(rows))

    def test_the_same_number_is_never_paired_with_itself(self):
        rows = [C.GameLogRow("g1", "nfl", "A", "KC", "player_pass_yds", 300)] * 2
        assert C.fit_correlations(rows) == []

    def test_a_same_market_bucket_counts_pairs_not_rows(self):
        # Both orderings go in so the estimate is symmetric, but the
        # honest sample size is the number of distinct pairs.
        rows = []
        for i in range(40):
            rows += [
                C.GameLogRow(f"g{i}", "nfl", "A", "KC", "player_rush_yds", 60 + i),
                C.GameLogRow(f"g{i}", "nfl", "B", "KC", "player_rush_yds", 90 - i),
            ]
        pair = [e for e in C.fit_correlations(rows)
                if e.market_a == e.market_b == "player_rush_yds"][0]
        assert pair.n_observations == 40

    def test_a_same_market_fit_is_order_symmetric(self):
        rows, flipped = [], []
        rng = np.random.default_rng(3)
        for i in range(60):
            x, y = rng.normal(), rng.normal()
            rows += [
                C.GameLogRow(f"g{i}", "nfl", "A", "KC", "player_rush_yds", x),
                C.GameLogRow(f"g{i}", "nfl", "B", "KC", "player_rush_yds", y),
            ]
            flipped += [
                C.GameLogRow(f"g{i}", "nfl", "B", "KC", "player_rush_yds", y),
                C.GameLogRow(f"g{i}", "nfl", "A", "KC", "player_rush_yds", x),
            ]
        a = C.fit_correlations(rows)[0].rho
        b = C.fit_correlations(flipped)[0].rho
        assert a == pytest.approx(b)

    def test_market_filtering_is_respected(self):
        rows = self.synthetic(n_games=200)
        fitted = C.fit_correlations(rows, markets=["player_pass_yds"])
        assert fitted == []

    def test_the_per_bucket_cap_is_honoured(self):
        fitted = C.fit_correlations(self.synthetic(n_games=500),
                                    max_pairs_per_bucket=100)
        assert fitted[0].n_observations == 100


class TestEstimateStore:
    def test_a_round_trip_through_the_database(self, tmp_path):
        db = Database(tmp_path / "t.db")
        estimates = [C.Estimate(
            sport="americanfootball_nfl", market_a="player_pass_yds",
            market_b="player_reception_yds", relation=C.SAME_TEAM,
            rho=0.5, spearman=0.48, n_observations=300, n_games=300,
        )]
        assert db.save_correlation_estimates(estimates) == 1
        store = C.EstimateStore.from_db(db)
        found = store.lookup("americanfootball_nfl", "player_reception_yds",
                             "player_pass_yds", C.SAME_TEAM)
        assert found.rho == pytest.approx(0.5)
        assert found.n_observations == 300
        db.close()

    def test_a_refit_replaces_rather_than_duplicates(self, tmp_path):
        db = Database(tmp_path / "t.db")
        def est(rho, n):
            return C.Estimate("nfl", "a", "b", C.SAME_TEAM, rho, rho, n, n)
        db.save_correlation_estimates([est(0.3, 100)])
        db.save_correlation_estimates([est(0.6, 900)])
        rows = db.correlation_estimates()
        assert len(rows) == 1
        assert rows[0]["rho"] == pytest.approx(0.6)
        assert rows[0]["n_observations"] == 900
        db.close()

    def test_an_absent_pair_returns_nothing(self):
        assert C.EstimateStore().lookup("nfl", "a", "b", C.SAME_TEAM) is None
