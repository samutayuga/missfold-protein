"""Classify a finished simulation from its data, not from the input settings."""

from dataclasses import dataclass
from enum import Enum

from misfold.engine import SimulationResult
from misfold.sequence import NEUTRAL_PH

NATIVE_FRACTION = 0.8
DENATURED_UPHILL = 0.25  # share of energy-raising moves accepted
PH_DENATURATION_OFFSET = 3.0


class Status(Enum):
    TRAPPED = "Severely Misfolded (Kinetic Trap)"
    DENATURED = "Denatured"
    NATIVE = "Stable Native Fold"


@dataclass(frozen=True)
class Diagnosis:
    status: Status | None  # None: no stabilising contact is possible
    message: str


def classify(
    result: SimulationResult, native_energy: float, temperature: float, ph: float
) -> Diagnosis:
    """native_energy is the annealed optimum at neutral pH, the native-condition benchmark."""
    final = result.energy_history[-1]
    conditions = f"T = {temperature:.2f}, pH = {ph:.2f}"
    energies = f"final energy {final:.2f} vs. native-condition optimum {native_energy:.2f}"

    if native_energy >= 0:
        return Diagnosis(
            None,
            f"ℹ️ **No stable fold possible.** This sequence cannot form any stabilising "
            f"contact (no hydrophobic core or salt bridge), so folding cannot be assessed "
            f"({conditions}; {energies}).",
        )
    if final <= NATIVE_FRACTION * native_energy:
        return Diagnosis(
            Status.NATIVE,
            f"✅ **Status: {Status.NATIVE.value}**. The simulation safely traversed the free "
            f"energy landscape down to a thermodynamic minimum ({conditions}; {energies}).",
        )
    if result.uphill_acceptance >= DENATURED_UPHILL:
        if ph - NEUTRAL_PH <= -PH_DENATURATION_OFFSET:
            cause = (
                "Acid denaturation: acidic residues lose their charge, salt bridges break "
                "and basic residues repel each other"
            )
        elif ph - NEUTRAL_PH >= PH_DENATURATION_OFFSET:
            cause = (
                "Base denaturation: basic residues lose their charge, salt bridges break "
                "and acidic residues repel each other"
            )
        else:
            cause = "Thermal noise is too high for the non-covalent contacts to stabilise"
        return Diagnosis(
            Status.DENATURED,
            f"⚠️ **Status: {Status.DENATURED.value}**. {cause}, leaving the chain as an "
            f"open random coil ({conditions}; {energies}).",
        )
    return Diagnosis(
        Status.TRAPPED,
        f"🛑 **Status: {Status.TRAPPED.value}**. The chain lacked the kinetic energy to "
        f"overcome structural energy barriers and is stuck in an irregular, high-energy "
        f"configuration ({conditions}; {energies}).",
    )
