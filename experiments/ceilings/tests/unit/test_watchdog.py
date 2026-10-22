"""Watchdog kill rules, with made-up samples (no Docker, no Ollama)."""

import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "watchdog", Path(__file__).parents[2] / "scripts" / "watchdog.py")
wd = importlib.util.module_from_spec(spec)
sys.modules["watchdog"] = wd  # dataclasses look the module up while the file loads
spec.loader.exec_module(wd)


def _mon(**kw):
    return wd.Monitor(wd.Limits(**{"ollama_cpu": 300, "container_cpu": 150, "sustain": 60,
                                   "max_seconds": 3600, **kw}), start=0.0)


def test_short_spikes_are_tolerated():
    m = _mon()
    for t in range(0, 60, 5):
        assert m.check(wd.Sample(t, 400, 50, "ok")) is None
    assert m.check(wd.Sample(65, 100, 50, "ok")) is None  # back under: the clock resets
    assert m.check(wd.Sample(70, 400, 50, "ok")) is None


def test_sustained_ollama_cpu_kills():
    m = _mon()
    assert m.check(wd.Sample(0, 400, 50, "ok")) is None
    assert "ollama cpu" in m.check(wd.Sample(60, 400, 50, "ok"))


def test_sustained_container_cpu_kills():
    m = _mon(sustain=10)
    m.check(wd.Sample(0, 0, 160, "ok"))
    assert "container cpu" in m.check(wd.Sample(10, 0, 160, "ok"))


def test_thermal_warning_kills_immediately():
    assert "thermal" in _mon().check(wd.Sample(0, 0, 0, "CPU_Speed_Limit = 80"))


def test_wall_clock_kills():
    assert "wall clock" in _mon(max_seconds=100).check(wd.Sample(101, 0, 0, "ok"))


@pytest.mark.parametrize("out,expected", [
    ("Note: No thermal warning level has been recorded\n"
     "Note: No performance warning level has been recorded\n", "ok"),
    ("CPU_Scheduler_Limit \t= 100\nCPU_Available_CPUs \t= 10\nCPU_Speed_Limit \t= 100\n", "ok"),
    ("CPU_Speed_Limit \t= 70\n", "CPU_Speed_Limit = 70"),
    ("Thermal warning level set to 1.\n", "Thermal warning level set to 1."),
])
def test_thermal_parsing(monkeypatch, out, expected):
    monkeypatch.setattr(wd.sys, "platform", "darwin")
    monkeypatch.setattr(wd.subprocess, "run",
                        lambda *a, **k: type("R", (), {"stdout": out})())
    assert wd.thermal_state() == expected
