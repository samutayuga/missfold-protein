import numpy as np
import pytest

from misfold.sweep import DEFAULT_PH_VALUES, ph_sweep


def test_default_grid_spans_full_ph_range():
    assert DEFAULT_PH_VALUES[0] == 0.0 and DEFAULT_PH_VALUES[-1] == 14.0


def test_returns_one_point_per_ph():
    result = ph_sweep("HKHPHDH", 1.2, 50, ph_values=[2.0, 7.0, 12.0], replicas=2, seed=0)
    assert list(result.ph) == [2.0, 7.0, 12.0]
    assert result.mean_rg.shape == result.std_rg.shape == (3,)
    assert np.all(result.mean_rg > 0)
    assert np.all(result.std_rg >= 0)


def test_reproducible_with_seed():
    a = ph_sweep("HKHPHDH", 1.2, 50, ph_values=[7.0], replicas=2, seed=3)
    b = ph_sweep("HKHPHDH", 1.2, 50, ph_values=[7.0], replicas=2, seed=3)
    assert a.mean_rg[0] == b.mean_rg[0]


def test_charged_chain_swells_at_extreme_ph():
    """Salt bridges hold KDKD... compact at pH 7; acid or base breaks them and it swells."""
    result = ph_sweep("KDKDHHKDKDHHKD", 1.2, 300, ph_values=[2.0, 7.0, 12.5], replicas=6, seed=0)
    acid, neutral, base = result.mean_rg
    assert acid > neutral + 0.15
    assert base > neutral + 0.15


def test_rejects_zero_replicas():
    with pytest.raises(ValueError):
        ph_sweep("HKH", 1.2, 50, ph_values=[7.0], replicas=0)
