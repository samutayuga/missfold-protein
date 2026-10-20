"""Metropolis-Hastings folding simulation and annealing reference."""

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from misfold.energy import energy, interaction_matrix
from misfold.lattice import propose_move, straight_chain
from misfold.sequence import NEUTRAL_PH

# Slider temperatures are in reduced units: acceptance uses T / TEMPERATURE_SCALE, so
# with contact energies of order 1 the fold is stable around T = 1.2 and melts above ~3.
TEMPERATURE_SCALE = 6.0
# Annealing schedule, in effective (already scaled) temperature.
ANNEAL_T_START = 2.0
ANNEAL_T_END = 0.1
REFERENCE_SWEEPS = 1500


@dataclass(frozen=True)
class SimulationResult:
    coords: np.ndarray
    energy_history: list[float]
    accepted: int
    attempted: int
    uphill_accepted: int = 0
    uphill_attempted: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.attempted if self.attempted else 0.0

    @property
    def uphill_acceptance(self) -> float:
        """Share of energy-raising proposals accepted: a direct measure of thermal agitation."""
        return self.uphill_accepted / self.uphill_attempted if self.uphill_attempted else 0.0


def metropolis_accept(delta_e: float, temperature: float, rng: np.random.Generator) -> bool:
    if delta_e < 0:
        return True
    return bool(rng.random() < math.exp(-delta_e / temperature))


def _run(
    eps: np.ndarray,
    n: int,
    temperatures: Iterable[float],
    rng: np.random.Generator,
    proposals_per_step: int = 1,
) -> tuple[SimulationResult, float]:
    """Run proposals_per_step proposals per temperature, recording the energy after each
    step; return the result and the lowest energy seen."""
    coords = straight_chain(n)
    current = energy(coords, eps)
    lowest = current
    history: list[float] = []
    accepted = attempted = uphill_accepted = uphill_attempted = 0
    for temperature in temperatures:
        for _ in range(proposals_per_step):
            attempted += 1
            proposed = propose_move(coords, rng)
            if proposed is None:
                continue
            proposed_energy = energy(proposed, eps)
            delta_e = proposed_energy - current
            uphill = delta_e > 0
            uphill_attempted += uphill
            if metropolis_accept(delta_e, temperature, rng):
                coords, current = proposed, proposed_energy
                accepted += 1
                uphill_accepted += uphill
                lowest = min(lowest, current)
        history.append(current)
    result = SimulationResult(
        coords, history, accepted, attempted, uphill_accepted, uphill_attempted
    )
    return result, lowest


def simulate(
    seq: str, temperature: float, ph: float, steps: int, rng: np.random.Generator
) -> SimulationResult:
    """Each step is one Monte Carlo sweep: len(seq) proposals."""
    effective = temperature / TEMPERATURE_SCALE
    result, _ = _run(
        interaction_matrix(seq, ph), len(seq), [effective] * steps, rng, proposals_per_step=len(seq)
    )
    return result


def reference_energy(
    seq: str, ph: float = NEUTRAL_PH, sweeps: int = REFERENCE_SWEEPS, seed: int = 0
) -> float:
    """Simulated annealing optimum; the budget is in sweeps so long chains get as many
    proposals per residue as short ones."""
    schedule = np.geomspace(ANNEAL_T_START, ANNEAL_T_END, sweeps)
    _, lowest = _run(
        interaction_matrix(seq, ph), len(seq), schedule, np.random.default_rng(seed),
        proposals_per_step=len(seq),
    )
    return lowest
