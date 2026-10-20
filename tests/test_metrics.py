import numpy as np
import pytest

from rppg.eval.metrics import mae, pearson, rmse, summarize


def test_known_values():
    est = np.array([70.0, 82.0, 95.0, 60.0])
    ref = np.array([72.0, 80.0, 90.0, 60.0])
    assert mae(est, ref) == pytest.approx(9 / 4)
    assert rmse(est, ref) == pytest.approx(np.sqrt(33 / 4))
    assert pearson(est, ref) == pytest.approx(np.corrcoef(est, ref)[0, 1])


def test_nan_pairs_are_dropped():
    est = np.array([70.0, np.nan, 90.0, 100.0])
    ref = np.array([70.0, 80.0, np.nan, 101.0])
    s = summarize(est, ref)
    assert s["n"] == 2
    assert s["mae"] == pytest.approx(0.5)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mae(np.zeros(3), np.zeros(4))


def test_pearson_constant_input_is_nan():
    assert np.isnan(pearson(np.ones(5), np.arange(5.0)))
