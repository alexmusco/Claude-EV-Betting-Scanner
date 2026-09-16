"""
`betedge rt` end to end: contracts file -> Rotten Tomatoes page -> price
-> EV -> stake.

The film page is the real Resident Evil fixture, served from the cache
directory so no request is made. Nothing here touches the network.
"""

import gzip
import textwrap
from pathlib import Path

import pytest
import yaml

from betedge import cli

FIXTURE = Path(__file__).parent / "fixtures" / "rt_resident_evil.html.gz"


@pytest.fixture
def page() -> str:
    with gzip.open(FIXTURE, "rt", encoding="utf-8", errors="replace") as fh:
        return fh.read()


@pytest.fixture
def wired(tmp_path, page):
    """A config, a contracts file, and the film page already cached."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "database": str(tmp_path / "t.db"),
        "reports_dir": str(tmp_path / "reports"),
        "bankroll": {"amount": 1000.0},
    }))
    cache = tmp_path / "rt-cache"
    cache.mkdir()
    (cache / "m_resident_evil.html").write_text(page, encoding="utf-8")
    return cfg_path, tmp_path, cache


def contracts_file(tmp_path, **overrides) -> Path:
    entry = {
        "ticker": "KXRT-RE-T30",
        "slug": "resident_evil",
        "film": "Resident Evil",
        "threshold": 30,
        "direction": "above",
        "inclusive": True,
        "scope": "all_critics",
        "max_new_reviews": 40,
        "expected_new_reviews": 5,
        "verified": True,
    }
    entry.update(overrides)
    path = tmp_path / "contracts.yaml"
    path.write_text(yaml.safe_dump({"meta": {"version": 1},
                                    "contracts": [entry]}))
    return path


def run(argv):
    return cli.main(argv)


class TestContractsCommand:
    def test_it_reports_an_empty_shipped_file_helpfully(self, wired, capsys):
        cfg_path, _tmp, _cache = wired
        assert run(["--config", str(cfg_path), "rt", "contracts"]) == 0
        out = capsys.readouterr().out
        assert "No contracts defined yet" in out
        assert "--init" in out

    def test_it_prints_what_a_contract_settles_on(self, wired, capsys):
        cfg_path, tmp, _cache = wired
        path = contracts_file(tmp, verified=False)
        run(["--config", str(cfg_path), "rt", "contracts",
             "--contracts", str(path)])
        out = capsys.readouterr().out
        assert "KXRT-RE-T30" in out
        assert "all_critics" in out
        assert "at or above 30%" in out
        assert "NOT VERIFIED" in out

    def test_unverified_contracts_are_called_out(self, wired, capsys):
        cfg_path, tmp, _cache = wired
        path = contracts_file(tmp, verified=False)
        run(["--config", str(cfg_path), "rt", "contracts",
             "--contracts", str(path)])
        assert "NOTHING IS STAKED" in capsys.readouterr().out

    def test_a_duplicate_ticker_is_refused(self, wired, tmp_path):
        from betedge.tomatoes import ContractBook

        # Two rows for one market would be priced and staked twice.
        path = tmp_path / "dupe.yaml"
        path.write_text(yaml.safe_dump({"contracts": [
            {"ticker": "A", "threshold": 60},
            {"ticker": "A", "threshold": 70},
        ]}))
        with pytest.raises(ValueError, match="appears twice"):
            ContractBook.load(path)


class TestScanEndToEnd:
    def test_a_decided_contract_is_priced_and_staked(self, wired, capsys):
        # Resident Evil is 58/162 = 36%. With at most 40 more reviews it
        # cannot fall below 58/202 = 28.7% -> 29%... so a 30% threshold
        # is NOT decided. A 25% one is.
        cfg_path, tmp, cache = wired
        path = contracts_file(tmp, ticker="KXRT-RE-T25", threshold=25)
        run(["--config", str(cfg_path), "rt", "scan",
             "--contracts", str(path), "--cache", str(cache),
             "--price", "0.80"])
        out = capsys.readouterr().out
        assert "KXRT-RE-T25" in out
        assert "58/162" in out
        assert "decided_by_arithmetic" in out
        assert "position(s) worth taking" in out

    def test_the_fair_price_of_a_decided_contract_is_a_dollar(
        self, wired, capsys
    ):
        cfg_path, tmp, cache = wired
        path = contracts_file(tmp, ticker="KXRT-RE-T25", threshold=25)
        run(["--config", str(cfg_path), "rt", "scan",
             "--contracts", str(path), "--cache", str(cache),
             "--price", "0.80"])
        out = capsys.readouterr().out
        assert "100.0c" in out

    def test_a_contract_it_cannot_win_is_not_bet(self, wired, capsys):
        # 58/162 with 40 more reviews cannot reach 60%.
        cfg_path, tmp, cache = wired
        path = contracts_file(tmp, ticker="KXRT-RE-T60", threshold=60)
        run(["--config", str(cfg_path), "rt", "scan",
             "--contracts", str(path), "--cache", str(cache),
             "--price", "0.20"])
        out = capsys.readouterr().out
        assert "0.0c" in out
        assert "Nothing worth betting" in out

    def test_an_unverified_contract_is_priced_but_never_staked(
        self, wired, capsys
    ):
        cfg_path, tmp, cache = wired
        path = contracts_file(tmp, ticker="KXRT-RE-T25", threshold=25,
                              verified=False)
        run(["--config", str(cfg_path), "rt", "scan",
             "--contracts", str(path), "--cache", str(cache),
             "--price", "0.80"])
        out = capsys.readouterr().out
        assert "settlement_terms_unverified" in out
        assert "Nothing worth betting" in out
        assert "unverified, so staked" in out

    def test_the_wrong_tomatometer_is_refused_rather_than_compared(
        self, wired, capsys
    ):
        # The contract settles on Top Critics (30%) but we would be
        # scoring it against All Critics (36%). Six points apart.
        cfg_path, tmp, cache = wired
        path = contracts_file(tmp, scope="top_critics", threshold=25)
        run(["--config", str(cfg_path), "rt", "scan",
             "--contracts", str(path), "--cache", str(cache),
             "--price", "0.80"])
        out = capsys.readouterr().out
        # It uses the top-critics snapshot, so 13/43, and agrees.
        assert "13/43" in out

    def test_a_contract_with_no_slug_says_so_instead_of_guessing(
        self, wired, capsys
    ):
        cfg_path, tmp, cache = wired
        path = contracts_file(tmp, slug="")
        run(["--config", str(cfg_path), "rt", "scan",
             "--contracts", str(path), "--cache", str(cache),
             "--price", "0.5"])
        assert "no rotten tomatoes slug" in capsys.readouterr().out

    def test_an_unreadable_film_page_does_not_stop_the_scan(
        self, wired, capsys
    ):
        cfg_path, tmp, cache = wired
        (cache / "m_other_film.html").write_text("<html>nope</html>")
        path = contracts_file(tmp, slug="other_film")
        assert run(["--config", str(cfg_path), "rt", "scan",
                    "--contracts", str(path), "--cache", str(cache),
                    "--price", "0.5"]) == 0
        assert "score unreadable" in capsys.readouterr().out


class TestSnapshotCommand:
    def test_it_prints_both_tomatometers_with_counts(self, wired, capsys):
        cfg_path, _tmp, cache = wired
        assert run(["--config", str(cfg_path), "rt", "snapshot",
                    "resident_evil", "--cache", str(cache)]) == 0
        out = capsys.readouterr().out
        assert "Resident Evil" in out
        assert "58/162" in out
        assert "13/43" in out

    def test_an_unreadable_page_reports_rather_than_raising(
        self, wired, capsys
    ):
        cfg_path, _tmp, cache = wired
        (cache / "m_nope.html").write_text("<html>nope</html>")
        assert run(["--config", str(cfg_path), "rt", "snapshot", "nope",
                    "--cache", str(cache)]) == 1
        assert "Could not read /m/nope" in capsys.readouterr().out
