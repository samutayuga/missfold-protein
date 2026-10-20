import numpy as np
import pytest

from misfold.energy import contact_pairs, energy, interaction_matrix, radius_of_gyration

SQUARE = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])  # only 0–3 in contact


def test_straight_chain_has_no_contacts():
    c = np.array([[i, 0, 0] for i in range(5)])
    assert contact_pairs(c) == []
    assert energy(c, interaction_matrix("HHHHH", 7.0)) == 0.0


def test_hh_contact():
    assert contact_pairs(SQUARE) == [(0, 3)]
    assert energy(SQUARE, interaction_matrix("HPPH", 7.0)) == pytest.approx(-1.0)


@pytest.mark.parametrize("seq", ["HPPP", "PPPH", "PPPP", "KPPH", "DPPP"])
def test_non_hh_non_charged_pair_is_plus_point_one(seq):
    assert energy(SQUARE, interaction_matrix(seq, 7.0)) == pytest.approx(0.1)


def test_salt_bridge_at_neutral_ph():
    assert energy(SQUARE, interaction_matrix("KPPD", 7.0)) == pytest.approx(-0.5, abs=1e-3)


def test_salt_bridge_lost_at_low_ph():
    assert energy(SQUARE, interaction_matrix("KPPD", 2.0)) == pytest.approx(0.0, abs=0.01)


def test_like_charges_repel():
    assert energy(SQUARE, interaction_matrix("KPPK", 2.0)) == pytest.approx(0.5, abs=1e-3)


def test_diagonal_is_not_contact():
    assert contact_pairs(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]])) == []


def test_radius_of_gyration():
    assert radius_of_gyration(np.array([[0, 0, 0], [2, 0, 0]])) == pytest.approx(1.0)
