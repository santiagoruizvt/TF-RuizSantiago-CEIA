"""Tests de las particiones por sujeto y del protocolo de evaluación común."""

from pathlib import Path

import numpy as np
import pytest

from rppg.eval.protocol import compare_hr, summarize_windows
from rppg.eval.splits import assert_disjoint, load_splits, make_splits
from rppg.utils.config import apply_overrides, load_yaml

ROOT = Path(__file__).resolve().parents[1]
EVAL_CFG = load_yaml(ROOT / "configs" / "eval.yaml")


# --- particiones -------------------------------------------------------------


def test_committed_splits_are_disjoint_and_cover_ubfc():
    """Requerimiento del plan: ningún sujeto compartido entre train, val y test."""
    splits = load_splits(ROOT / "configs" / "splits.yaml")  # load_splits ya verifica
    all_subjects = splits["train"] + splits["val"] + splits["test"]
    assert len(all_subjects) == len(set(all_subjects)) == 42
    assert len(splits["test"]) == 9 and len(splits["val"]) == 5


def test_make_splits_is_reproducible_and_disjoint():
    subjects = [f"subject{i}" for i in range(1, 21)]
    a = make_splits(subjects, n_test=4, n_val=3, seed=7)
    assert a == make_splits(subjects, n_test=4, n_val=3, seed=7)
    assert a != make_splits(subjects, n_test=4, n_val=3, seed=8)
    assert_disjoint(a)
    assert sorted(a["train"] + a["val"] + a["test"], key=lambda s: int(s[7:])) == subjects


def test_assert_disjoint_detects_leak():
    with pytest.raises(ValueError, match="subject3"):
        assert_disjoint({"train": ["subject1", "subject3"], "val": [], "test": ["subject3"]})


def test_make_splits_requires_training_subjects():
    with pytest.raises(ValueError):
        make_splits(["a", "b", "c"], n_test=2, n_val=1, seed=0)


# --- protocolo ---------------------------------------------------------------


def test_compare_hr_identical_signals_gives_zero_error():
    t = np.cumsum(np.random.default_rng(0).uniform(0.025, 0.04, 1500))
    ppg = np.sin(2 * np.pi * 1.3 * t)
    windows = compare_hr("s1", t, ppg, t, ppg, EVAL_CFG)
    assert windows["valid"].all()
    summary = summarize_windows(windows)
    assert summary.loc["ALL", "mae"] == pytest.approx(0, abs=1e-9)
    assert windows["hr_ref"].median() == pytest.approx(78, abs=1)


def test_compare_hr_truncates_to_common_windows_and_summarizes_per_subject():
    t = np.arange(1200) / 30.0
    ref = np.sin(2 * np.pi * 1.0 * t)
    est = np.sin(2 * np.pi * 1.1 * t[:900])  # más corta y 6 bpm más rápida
    windows = compare_hr("s2", t[:900], est, t, ref, EVAL_CFG)
    summary = summarize_windows(windows)
    assert len(windows) == 21  # ventanas de 10 s con paso de 1 s en 30 s
    assert summary.loc["s2", "mae"] == pytest.approx(6, abs=1)
    assert list(summary.index) == ["ALL", "s2"]


def test_apply_overrides_parses_yaml_and_rejects_unknown_keys():
    cfg = {"train": {"epochs": 30, "lr": 0.001}, "run_name": "x"}
    apply_overrides(cfg, ["train.epochs=5", "train.lr=1e-4", "run_name=smoke"])
    assert cfg == {"train": {"epochs": 5, "lr": 1e-4}, "run_name": "smoke"}
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["train.epoch=5"])
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["train.epochs"])
