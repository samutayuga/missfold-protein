import time

import numpy as np
import pytest

from misfold.energy import energy, interaction_matrix
from misfold.engine import metropolis_accept, reference_energy, simulate
from misfold.lattice import is_valid_chain

SEQ = "HKHPHDHPKHHDH"


def test_accept_downhill_always():
    rng = np.random.default_rng(0)
    assert all(metropolis_accept(-0.5, 0.1, rng) for _ in range(100))


def test_accept_uphill_rate_matches_boltzmann():
    rng = np.random.default_rng(0)
    rate = np.mean([metropolis_accept(1.0, 1.0, rng) for _ in range(20_000)])
    assert rate == pytest.approx(np.exp(-1.0), abs=0.02)


def test_history_length_and_energy_moves():
    r = simulate(SEQ, 1.2, 7.0, 300, np.random.default_rng(0))
    assert len(r.energy_history) == 300  # one entry per sweep
    assert r.attempted == 300 * len(SEQ) and r.accepted > 0
    assert min(r.energy_history) < 0  # chain actually folds
    assert r.energy_history[-1] == pytest.approx(energy(r.coords, interaction_matrix(SEQ, 7.0)))


@pytest.mark.parametrize("t, ph", [(0.1, 7.0), (5.0, 7.0), (1.2, 0.0), (1.2, 14.0)])
def test_chain_valid_at_all_conditions(t, ph):
    r = simulate("HKPDHHKPDHPKHHDPHKDH", t, ph, 1000, np.random.default_rng(3))
    assert is_valid_chain(r.coords)


def test_deterministic_with_seed():
    a = simulate("HKHDPH", 1.0, 7.0, 200, np.random.default_rng(7))
    b = simulate("HKHDPH", 1.0, 7.0, 200, np.random.default_rng(7))
    assert a.energy_history == b.energy_history


def test_reference_energy_deterministic_and_low():
    e = reference_energy(SEQ)
    assert e == reference_energy(SEQ)
    assert e <= -3.0


def test_low_ph_raises_reference_energy():  # salt bridges lost
    assert reference_energy(SEQ, ph=2.0) > reference_energy(SEQ, ph=7.0)


def test_max_length_is_interactive():
    seq = "HKPD" * 5
    t0 = time.perf_counter()
    simulate(seq, 1.2, 7.0, 1000, np.random.default_rng(0))
    reference_energy(seq)
    assert time.perf_counter() - t0 < 10.0


def test_default_conditions_fold():  # reduced temperature units: T=1.2 is below folding
    finals = [simulate(SEQ, 1.2, 7.0, 300, np.random.default_rng(s)).energy_history[-1]
              for s in range(5)]
    assert np.mean(finals) <= -3.0


def test_uphill_acceptance_tracks_temperature():
    cold = simulate(SEQ, 0.1, 7.0, 300, np.random.default_rng(0))
    hot = simulate(SEQ, 4.5, 7.0, 300, np.random.default_rng(0))
    assert cold.uphill_attempted > 0 and hot.uphill_attempted > 0
    assert cold.uphill_acceptance < 0.05 < hot.uphill_acceptance


def test_reference_energy_budget_scales_with_length():
    # A fixed proposal budget left long chains far above their optimum
    # (30-mer: -13.2 vs -18.1 from a 10x longer anneal).
    assert reference_energy((SEQ * 3)[:30]) <= -17.0
