import numpy as np
import pytest

from misfold.diagnostics import Status, classify
from misfold.engine import SimulationResult, reference_energy, simulate


def _r(final, uphill_acc, uphill_att=100):
    return SimulationResult(np.zeros((3, 3), int), [final], 50, 100, uphill_acc, uphill_att)


def test_native():
    assert classify(_r(-4.0, 10), -4.5, 1.2, 7.0, "HKHPHDH").status is Status.NATIVE


def test_denatured():
    assert classify(_r(-0.5, 60), -4.5, 4.5, 7.0, "HKHPHDH").status is Status.DENATURED


def test_trapped():
    assert classify(_r(-1.0, 5), -4.5, 0.2, 7.0, "HKHPHDH").status is Status.TRAPPED


def test_low_temperature_neutral_moves_do_not_count_as_denatured():
    # Plenty of accepted moves overall, but almost no uphill ones: cold, not hot.
    r = SimulationResult(np.zeros((3, 3), int), [-1.0], 90, 100, 0, 40)
    assert classify(r, -4.5, 0.1, 7.0, "HKHPHDH").status is Status.TRAPPED


def test_no_core():
    assert classify(_r(0.0, 50), 0.0, 1.2, 7.0, "HKHPHDH").status is None


def test_message_contains_numbers():
    m = classify(_r(-1.0, 5), -4.5, 0.2, 7.0, "HKHPHDH").message
    assert "-1.00" in m and "-4.50" in m and "0.20" in m and "7.00" in m


def test_acid_denaturation_message():
    m = classify(_r(-0.5, 60), -4.5, 1.2, 2.0, "KDKDHH").message
    assert "acid" in m.lower()


def test_base_denaturation_message():
    m = classify(_r(-0.5, 60), -4.5, 1.2, 12.0, "KDKDHH").message
    assert "base" in m.lower()


@pytest.mark.parametrize(
    "seq, t, ph, expected",
    [
        ("HKHPHDHPKHHDH", 0.2, 7.0, Status.TRAPPED),
        ("HKHPHDHPKHHDH", 1.2, 7.0, Status.NATIVE),
        ("HKHPHDHPKHHDH", 4.5, 7.0, Status.DENATURED),
        ("KDKDHHKDKDHHKD", 1.2, 7.0, Status.NATIVE),
        ("KDKDHHKDKDHHKD", 1.2, 2.0, Status.DENATURED),
        ("HKHPHDHPKHHDHHKHPHDH", 1.2, 7.0, Status.NATIVE),  # MAX_LENGTH still folds
    ],
)
def test_calibration(seq, t, ph, expected):
    ref = reference_energy(seq)
    hits = sum(
        classify(simulate(seq, t, ph, 300, np.random.default_rng(s)), ref, t, ph, seq).status
        is expected
        for s in range(10)
    )
    assert hits >= 6


@pytest.mark.parametrize("ph, word", [(1.0, "acid"), (13.0, "base")])
def test_no_charged_residues_never_blames_ph(ph, word):
    m = classify(_r(-0.5, 60), -4.5, 4.5, ph, "HHPHPHPHPHHPH").message
    assert word not in m.lower() and "thermal" in m.lower()


@pytest.mark.parametrize("final, up, ref", [(-4.0, 10, -4.5), (-0.5, 60, -4.5), (-1.0, 5, -4.5), (0.0, 5, 0.0)])
def test_messages_have_no_emoji(final, up, ref):
    # Icons come from Material Symbols in the UI, not emoji in the text.
    m = classify(_r(final, up), ref, 1.2, 7.0, "HKHPHDH").message
    assert all(ord(ch) < 0x2100 for ch in m)
