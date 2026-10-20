"""pH sweep: how the chain's size (radius of gyration) responds to acidity."""

from dataclasses import dataclass

import numpy as np

from misfold.energy import radius_of_gyration
from misfold.engine import simulate

DEFAULT_PH_VALUES = np.arange(0.0, 14.5, 1.0)
DEFAULT_REPLICAS = 6


@dataclass
class SweepResult:
    ph: np.ndarray
    mean_rg: np.ndarray
    std_rg: np.ndarray


def ph_sweep(
    seq: str,
    temperature: float,
    steps: int,
    ph_values=DEFAULT_PH_VALUES,
    replicas: int = DEFAULT_REPLICAS,
    seed: int | None = None,
) -> SweepResult:
    """Run independent simulations at each pH and average the final radius of gyration.

    A single run's final state is noisy, so each pH gets `replicas` runs; the spread
    becomes the error bar.
    """
    if replicas < 1:
        raise ValueError("replicas must be at least 1")
    rng = np.random.default_rng(seed)
    ph = np.asarray(ph_values, dtype=float)
    sizes = np.array([
        [radius_of_gyration(simulate(seq, temperature, p, steps, rng).coords) for _ in range(replicas)]
        for p in ph
    ])
    return SweepResult(ph=ph, mean_rg=sizes.mean(axis=1), std_rg=sizes.std(axis=1))
