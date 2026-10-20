"""Contact energy on the cubic lattice, with pH-dependent charge interactions."""

import numpy as np

from misfold.sequence import residue_charges

HH_EPSILON = -1.0
OTHER_EPSILON = 0.1
COULOMB_K = 0.5


def interaction_matrix(seq: str, ph: float) -> np.ndarray:
    """eps[i, j] is the energy contributed when residues i and j are in contact."""
    residues = np.array(list(seq))
    eps = np.full((len(seq), len(seq)), OTHER_EPSILON)

    hydrophobic = residues == "H"
    eps[np.outer(hydrophobic, hydrophobic)] = HH_EPSILON

    charged = np.isin(residues, ["K", "D"])
    both_charged = np.outer(charged, charged)
    q = residue_charges(seq, ph)
    eps[both_charged] = (COULOMB_K * np.outer(q, q))[both_charged]
    return eps


def _contact_mask(coords: np.ndarray) -> np.ndarray:
    manhattan = np.abs(coords[:, None, :] - coords[None, :, :]).sum(axis=-1)
    return np.triu(manhattan == 1, k=2)


def contact_pairs(coords: np.ndarray) -> list[tuple[int, int]]:
    i, j = np.nonzero(_contact_mask(coords))
    return [(int(a), int(b)) for a, b in zip(i, j)]


def energy(coords: np.ndarray, eps: np.ndarray) -> float:
    return float(eps[_contact_mask(coords)].sum())


def radius_of_gyration(coords: np.ndarray) -> float:
    centred = coords - coords.mean(axis=0)
    return float(np.sqrt((centred**2).sum(axis=1).mean()))
