import json

import pytest

from ceilings import prereg


def test_freeze_and_check(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"b": 1, "a": [1, 2]}))
    digest = prereg.freeze(cfg, tmp_path)
    assert (tmp_path / "HASH").read_text().strip() == digest
    assert prereg.check_hash(tmp_path) == digest
    assert prereg.load_frozen(tmp_path) == {"a": [1, 2], "b": 1}


def test_any_change_breaks_the_hash(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"margin": 0.05}))
    prereg.freeze(cfg, tmp_path)
    frozen = tmp_path / "config_frozen.json"
    frozen.write_text(frozen.read_text().replace("0.05", "0.06"))
    with pytest.raises(prereg.PreregHashMismatch):
        prereg.check_hash(tmp_path)


def test_unfrozen_raises_and_current_hash_falls_back(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(prereg.PreregHashMismatch):
        prereg.check_hash(tmp_path)
    assert len(prereg.current_hash(tmp_path)) == 64


def test_log_deviation_appends_row(tmp_path):
    (tmp_path / "DEVIATIONS.md").write_text("| h |\n")
    prereg.log_deviation("H6", "k=3", "k=4", "floor too short", "none", tmp_path)
    assert "k=3 | k=4" in (tmp_path / "DEVIATIONS.md").read_text()
