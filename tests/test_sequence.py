import numpy as np
import pytest

from misfold.sequence import SequenceError, net_charge, residue_charges, validate_sequence


@pytest.mark.parametrize("seq", ["HKHPHDHPKHHDH", "HHPHPHPHPHHPH"])
def test_valid_sequences_pass(seq):
    assert validate_sequence(seq) == seq


def test_normalises_case_and_whitespace():
    assert validate_sequence("  hkd ph ") == "HKDPH"


@pytest.mark.parametrize("raw", ["HHXP", "HH-P", "123"])
def test_rejects_invalid_characters(raw):
    with pytest.raises(SequenceError, match="only H, P, K and D"):
        validate_sequence(raw)


@pytest.mark.parametrize("raw", ["", "HP", "H" * 61])
def test_rejects_bad_length(raw):
    with pytest.raises(SequenceError, match="between 3 and 60"):
        validate_sequence(raw)


def test_boundary_lengths_accepted():
    assert len(validate_sequence("HPH")) == 3
    assert len(validate_sequence("H" * 60)) == 60


def test_charges_at_neutral_and_pka():
    q = residue_charges("HPKD", 7.0)
    assert q[0] == q[1] == 0.0
    assert q[2] == pytest.approx(1.0, abs=1e-3)
    assert q[3] == pytest.approx(-1.0, abs=1e-3)
    assert residue_charges("D", 3.9)[0] == pytest.approx(-0.5)
    assert residue_charges("K", 10.5)[0] == pytest.approx(0.5)


@pytest.mark.parametrize("ph", [0.0, 14.0])
def test_charges_bounded_at_extremes(ph):
    q = residue_charges("KD", ph)
    assert np.all(np.isfinite(q))
    assert 0 <= q[0] <= 1
    assert -1 <= q[1] <= 0


def test_net_charge_sign_flips_with_ph():
    assert net_charge("KKDD", 2.0) > 0 > net_charge("KKDD", 12.0)
