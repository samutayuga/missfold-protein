"""Sequence validation and pH-dependent residue charges."""

import numpy as np

MIN_LENGTH = 3
MAX_LENGTH = 20  # beyond ~20 residues the local move set cannot fold within 1000 sweeps
ALPHABET = "HPKD"
PKA_ACID = 3.9
PKA_BASE = 10.5
NEUTRAL_PH = 7.0


class SequenceError(ValueError):
    """Raised when a user-supplied sequence cannot be simulated."""


def validate_sequence(raw: str) -> str:
    seq = "".join(raw.split()).upper()
    if any(ch not in ALPHABET for ch in seq):
        raise SequenceError("Sequence may contain only H, P, K and D.")
    if not MIN_LENGTH <= len(seq) <= MAX_LENGTH:
        raise SequenceError(
            f"Sequence length must be between {MIN_LENGTH} and {MAX_LENGTH} residues "
            f"(got {len(seq)})."
        )
    return seq


def residue_charges(seq: str, ph: float) -> np.ndarray:
    residues = np.array(list(seq))
    charges = np.zeros(len(seq))
    charges[residues == "D"] = -1.0 / (1.0 + 10.0 ** (PKA_ACID - ph))
    charges[residues == "K"] = 1.0 / (1.0 + 10.0 ** (ph - PKA_BASE))
    return charges


def net_charge(seq: str, ph: float) -> float:
    return float(residue_charges(seq, ph).sum())
