"""Metropolis-Hastings folding simulation and annealing reference."""

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from misfold.energy import energy, interaction_matrix
from misfold.lattice import propose_move, straight_chain
from misfold.sequence import NEUTRAL_PH

ANNEAL_T_START = 2.0
ANNEAL_T_END = 0.1


@dataclass(frozen=True)
class SimulationResult:
    coords: np.ndarray
    energy_history: list[float]
    accepted: int
    attempted: int

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.attempted if self.attempted else 0.0


def metropolis_accept(delta_e: float, temperature: float, rng: np.random.Generator) -> bool:
    if delta_e < 0:
        return True
    return bool(rng.random() < math.exp(-delta_e / temperature))


def _run(
    eps: np.ndarray, n: int, temperatures: Iterable[float], rng: np.random.Generator
) -> tuple[SimulationResult, float]:
    """Run one proposal per temperature; return the result and the lowest energy seen."""
    coords = straight_chain(n)
    current = energy(coords, eps)
    lowest = current
    history: list[float] = []
    accepted = 0
    for temperature in temperatures:
        proposed = propose_move(coords, rng)
        if proposed is not None:
            proposed_energy = energy(proposed, eps)
            if metropolis_accept(proposed_energy - current, temperature, rng):
                coords, current = proposed, proposed_energy
                accepted += 1
                lowest = min(lowest, current)
        history.append(current)
    return SimulationResult(coords, history, accepted, len(history)), lowest


def simulate(
    seq: str, temperature: float, ph: float, steps: int, rng: np.random.Generator
) -> SimulationResult:
    result, _ = _run(interaction_matrix(seq, ph), len(seq), [temperature] * steps, rng)
    return result


def reference_energy(
    seq: str, ph: float = NEUTRAL_PH, steps: int = 20_000, seed: int = 0
) -> float:
    schedule = np.geomspace(ANNEAL_T_START, ANNEAL_T_END, steps)
    _, lowest = _run(interaction_matrix(seq, ph), len(seq), schedule, np.random.default_rng(seed))
    return lowest
