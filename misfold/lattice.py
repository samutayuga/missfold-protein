"""Self-avoiding chain on the cubic lattice and its local move set.

Moves: end move, corner flip, crankshaft. Each proposal picks one bead and
returns a new chain, or None when that bead has no legal move.
"""

import numpy as np

_NEIGHBOUR_STEPS = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
)


def straight_chain(n: int) -> np.ndarray:
    coords = np.zeros((n, 3), dtype=np.int64)
    coords[:, 0] = np.arange(n)
    return coords


def is_valid_chain(coords: np.ndarray) -> bool:
    bonds = np.abs(np.diff(coords, axis=0)).sum(axis=1)
    unique_sites = len({tuple(p) for p in coords.tolist()})
    return bool(np.all(bonds == 1)) and unique_sites == len(coords)


def _end_moves(coords: np.ndarray, i: int, occupied: set) -> list[np.ndarray]:
    anchor = coords[1] if i == 0 else coords[-2]
    candidates = []
    for target in anchor + _NEIGHBOUR_STEPS:
        if tuple(target) not in occupied:
            new = coords.copy()
            new[i] = target
            candidates.append(new)
    return candidates


def _corner_flips(coords: np.ndarray, i: int, occupied: set) -> list[np.ndarray]:
    prev, here, nxt = coords[i - 1], coords[i], coords[i + 1]
    if np.abs(nxt - prev).sum() != 2 or np.array_equal(nxt - here, here - prev):
        return []  # straight segment, not a corner
    target = prev + nxt - here
    if tuple(target) in occupied:
        return []
    new = coords.copy()
    new[i] = target
    return [new]


def _crankshafts(coords: np.ndarray, i: int, occupied: set) -> list[np.ndarray]:
    if not 0 < i < len(coords) - 2:
        return []
    start, end = coords[i - 1], coords[i + 2]
    axis = end - start
    if np.abs(axis).sum() != 1:
        return []  # beads i-1 and i+2 must be lattice neighbours to form a U
    offset = coords[i] - start
    if not np.array_equal(coords[i + 1] - end, offset):
        return []
    candidates = []
    for step in _NEIGHBOUR_STEPS:
        # Rotations of the offset about the axis: any unit step orthogonal to it.
        if np.dot(step, axis) != 0 or np.array_equal(step, offset):
            continue
        a, b = start + step, end + step
        if tuple(a) in occupied or tuple(b) in occupied:
            continue
        new = coords.copy()
        new[i], new[i + 1] = a, b
        candidates.append(new)
    return candidates


def propose_move(coords: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    n = len(coords)
    i = int(rng.integers(n))
    occupied = {tuple(p) for p in coords.tolist()}
    if i in (0, n - 1):
        candidates = _end_moves(coords, i, occupied)
    else:
        candidates = _corner_flips(coords, i, occupied) + _crankshafts(coords, i, occupied)
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]
