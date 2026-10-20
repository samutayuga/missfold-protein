import numpy as np

from misfold.lattice import is_valid_chain, propose_move, straight_chain


def test_straight_chain_valid():
    assert is_valid_chain(straight_chain(13))


def test_overlap_and_broken_bond_invalid():
    assert not is_valid_chain(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]))
    assert not is_valid_chain(np.array([[0, 0, 0], [2, 0, 0]]))


def test_straight_chain_can_move():  # reference code bug regression
    rng = np.random.default_rng(0)
    c = straight_chain(13)
    assert any(propose_move(c, rng) is not None for _ in range(50))


def test_proposals_always_valid_and_input_untouched():
    rng = np.random.default_rng(1)
    c = straight_chain(10)
    for _ in range(2000):
        before = c.copy()
        new = propose_move(c, rng)
        assert np.array_equal(c, before)
        if new is not None:
            assert is_valid_chain(new)
            c = new


def test_corner_flip_target():
    c = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    rng = np.random.default_rng(2)
    moved = [
        new
        for new in (propose_move(c, rng) for _ in range(300))
        if new is not None and not np.array_equal(new[1], c[1])
    ]
    assert moved and all(np.array_equal(m[1], [0, 1, 0]) for m in moved)


def test_crankshaft_moves_pair():
    # U-shape: 0 and 3 are lattice neighbours, so beads 1 and 2 may rotate together.
    c = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [2, 0, 0]])
    rng = np.random.default_rng(4)
    seen = set()
    for _ in range(500):
        new = propose_move(c, rng)
        if new is not None and not np.array_equal(new[1], c[1]) and not np.array_equal(new[2], c[2]):
            assert is_valid_chain(new)
            seen.add((tuple(new[1]), tuple(new[2])))
    assert ((0, 0, 1), (1, 0, 1)) in seen and ((0, 0, -1), (1, 0, -1)) in seen
