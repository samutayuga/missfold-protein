# 3D Protein Misfolding Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Streamlit app that runs a 3D lattice Metropolis Monte Carlo folding simulation under user-set temperature and pH, then shows the final conformation, the energy trace, and a diagnosis taken from the simulation data.

**Architecture:** Pure-Python package `misfold/` (sequence validation and charge, lattice moves, energy, engine, diagnostics, plotting) with no Streamlit imports, so it can be unit-tested. A thin `app.py` wires sidebar inputs to the package and renders the results. Randomness always comes in as a `numpy.random.Generator`, so tests are deterministic. pH enters only through the residue charges, which feed one per-run interaction matrix.

**Tech Stack:** Python 3.10+ (local: 3.12), Streamlit, NumPy, Matplotlib (`mplot3d`), pytest.

**Spec:** `docs/specs/protein_misfolding_spec.md`, plus the pH extension agreed with the user on 2026-10-20 (see "Extensions" below).

## Global Constraints

- Python `>=3.10`; dependencies limited to `streamlit`, `numpy`, `matplotlib` (+ `pytest` for dev).
- Alphabet: `H` (hydrophobic), `P` (polar), `K` (basic, lysine-like), `D` (acidic, aspartate-like).
- Charges (Henderson–Hasselbalch): `q_D(pH) = -1 / (1 + 10**(3.9 - pH))`, `q_K(pH) = +1 / (1 + 10**(pH - 10.5))`; `H` and `P` have charge 0.
- Contact energy (pairs `j > i+1`, lattice distance exactly 1): `-1.0` if both `H`; `0.5 · q_i · q_j` if both in {K, D}; otherwise `+0.1`.
- Metropolis: accept if `ΔE < 0`, else accept with probability `exp(-ΔE / T)`.
- Default sequence `"HKHPHDHPKHHDH"`.
- Temperature slider `0.1`–`5.0`, default `1.2`, step `0.1`. pH slider `0.0`–`14.0`, default `7.0`, step `0.5`. Steps slider `50`–`1000`, default `300`, step `50`. Sidebar order: sequence, temperature, pH, steps, button.
- Button label `"Run 3D Simulation"`. Page title `"3D Protein Misfolding Simulator"`, layout `"wide"`. Header `"🧬 3D Protein Misfolding & Energy Landscape Simulator"`.
- Colors: H `#FF4B4B`, P `#1C83E1`, K `#AB63FA`, D `#FFA15A`, backbone `#D3D3D3`, energy line `#00CC96`.
- Diagnosis labels exactly: `Severely Misfolded (Kinetic Trap)`, `Denatured`, `Stable Native Fold`.
- Energy y-axis label `"Free Energy (ΔG)"`, x-axis `"Simulation Step / Time"`.

## Extensions Beyond the Spec (agreed with user)

- **pH via charged residues.** `K`/`D` letters are added, with pH-dependent charges as above. At pH 7, D–K salt bridges stabilise the fold (`-0.5` per contact). At low pH, D loses its charge, so the salt bridges disappear and K–K contacts repel. At high pH, K loses its charge in the same way. Acid or base denaturation is reported under the existing `Denatured` label; no new category.
- **New default sequence** `"HKHPHDHPKHHDH"`, so pH has a visible effect without the user typing anything. The spec's `"HHPHPHPHPHHPH"` still works.

## Deviations From Spec §5 Reference Code (deliberate)

1. **Moves.** In the reference, a bead moves by a random vector from {-1,0,1}³ and the move is kept only if both bonds stay length 1. From the straight starting chain, the only position that meets that is the bead's current one, so the chain never changes and the energy trace is flat. The ends (indices 0 and n-1) never move either. Replace this with standard lattice moves: end move, corner flip and crankshaft (Task 3).
2. **Self-avoidance.** The reference lets two beads sit on the same site. Every proposed move must keep the chain self-avoiding.
3. **Diagnosis from data.** The reference picks the diagnosis from the temperature slider. Spec §4.2 says it must analyse the final output data. Classify from the final energy, measured against a reference minimum found by annealing **at neutral pH**, and from the acceptance rate (Task 5). The reference has to be at pH 7: a lattice chain can always avoid repulsive contacts by spreading out, so low pH shows up as lost stabilisation compared with the native-condition optimum. Comparing a run with the best energy at its own pH would hide that.
4. **Length cap.** Sequences are limited to 3–60 residues so a run stays interactive.

## Review Focus

1. **Sequence that cannot form any stabilising contact** (`"PPPPP"`, `"HPPPP"`, `"KKKKK"`): the neutral-pH reference energy is ≥ 0, so "Stable Native Fold" would mean nothing. Show an info message instead. Test in Task 5.
2. **pH at the slider extremes** (0.0, 14.0): charges stay within [-1, 0] / [0, 1] with no overflow or NaN, and the simulation runs. Tests in Task 1 and Task 7.
3. **Messy user input** (`" hkdph "`, `"HHXP"`, `""`): lowercase and whitespace are normalised; a bad character or bad length gives a clear sidebar error instead of a traceback. Tests in Task 1 and Task 7.
4. **Chain validity at every temperature/pH**: after any number of steps at T=0.1 or T=5.0 and pH 0 or 14, bonds are all length 1 and no two beads share a site. Property test in Task 4.
5. **Max length is still interactive**: a 60-residue run of 1000 steps plus the reference annealing finishes fast. Performance test in Task 4.

---

## File Structure

```
pyproject.toml            # deps, pytest config
app.py                    # Streamlit UI only
misfold/__init__.py
misfold/sequence.py       # validate_sequence, residue_charges, net_charge
misfold/energy.py         # interaction_matrix, energy, contacts, Rg
misfold/lattice.py        # initial chain, validity, move proposals
misfold/engine.py         # Metropolis loop, reference annealing
misfold/diagnostics.py    # Status, Diagnosis, classify
misfold/plots.py          # Figure builders
tests/test_sequence.py
tests/test_energy.py
tests/test_lattice.py
tests/test_engine.py
tests/test_diagnostics.py
tests/test_plots.py
tests/test_app.py
```

Shared types: `Coords` = `numpy.ndarray`, shape `(n, 3)`, dtype `int64`, integer lattice positions. `Interactions` = `numpy.ndarray`, shape `(n, n)`, float, symmetric: `eps[i, j]` is the energy contributed if `i` and `j` are in contact.

---

### Task 1: Project scaffold, sequence validation, residue charges

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `misfold/__init__.py`, `misfold/sequence.py`
- Test: `tests/test_sequence.py`

**Interfaces:**
- Produces: `MIN_LENGTH = 3`, `MAX_LENGTH = 60`, `ALPHABET = "HPKD"`, `PKA_ACID = 3.9`, `PKA_BASE = 10.5`, `NEUTRAL_PH = 7.0`; `class SequenceError(ValueError)`; `validate_sequence(raw: str) -> str`; `residue_charges(seq: str, ph: float) -> np.ndarray` (shape `(n,)`); `net_charge(seq: str, ph: float) -> float`.

- [ ] **Step 1: `git init`, write `pyproject.toml`** (project `misfold`, `requires-python = ">=3.10"`, deps `streamlit`, `numpy`, `matplotlib`; optional `dev = ["pytest"]`; `[tool.pytest.ini_options] testpaths = ["tests"]`), `.gitignore` (`.venv/`, `__pycache__/`, `.pytest_cache/`). Create `.venv` and run `pip install -e ".[dev]"`.

- [ ] **Step 2: Write failing tests**

```python
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

def test_charges_at_neutral_and_pka():
    q = residue_charges("HPKD", 7.0)
    assert q[0] == q[1] == 0.0
    assert q[2] == pytest.approx(1.0, abs=1e-3) and q[3] == pytest.approx(-1.0, abs=1e-3)
    assert residue_charges("D", 3.9)[0] == pytest.approx(-0.5)
    assert residue_charges("K", 10.5)[0] == pytest.approx(0.5)

@pytest.mark.parametrize("ph", [0.0, 14.0])
def test_charges_bounded_at_extremes(ph):
    q = residue_charges("KD", ph)
    assert np.all(np.isfinite(q)) and 0 <= q[0] <= 1 and -1 <= q[1] <= 0

def test_net_charge_sign_flips_with_ph():
    assert net_charge("KKDD", 2.0) > 0 > net_charge("KKDD", 12.0)
```

- [ ] **Step 3: Run** `pytest tests/test_sequence.py -v`. Expected: FAIL (import error).
- [ ] **Step 4: Implement.** In `validate_sequence`, remove all whitespace, uppercase, check the characters, then check the length. In `residue_charges`, use the Global Constraints formulas with NumPy.
- [ ] **Step 5: Run** `pytest tests/test_sequence.py -v`. Expected: all PASS.
- [ ] **Step 6: Commit** `feat: scaffold, sequence validation, pH-dependent charges`

---

### Task 2: Interaction matrix, energy, structure metrics

**Files:**
- Create: `misfold/energy.py`
- Test: `tests/test_energy.py`

**Interfaces:**
- Consumes: `residue_charges`.
- Produces: `HH_EPSILON = -1.0`, `OTHER_EPSILON = 0.1`, `COULOMB_K = 0.5`; `interaction_matrix(seq: str, ph: float) -> Interactions`; `contact_pairs(coords: Coords) -> list[tuple[int, int]]` (pairs `i < j`, `j > i+1`, Manhattan distance 1); `energy(coords: Coords, eps: Interactions) -> float`; `radius_of_gyration(coords: Coords) -> float`.

- [ ] **Step 1: Write failing tests**

```python
SQUARE = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]])   # only 0–3 in contact

def test_straight_chain_has_no_contacts():
    c = np.array([[i, 0, 0] for i in range(5)])
    assert contact_pairs(c) == [] and energy(c, interaction_matrix("HHHHH", 7.0)) == 0.0

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
    assert contact_pairs(np.array([[0,0,0],[1,0,0],[1,1,0],[2,1,0]])) == []

def test_radius_of_gyration():
    assert radius_of_gyration(np.array([[0,0,0],[2,0,0]])) == pytest.approx(1.0)
```

- [ ] **Step 2: Run** `pytest tests/test_energy.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Compute `contact_pairs` with NumPy: take the pairwise Manhattan distance matrix (`np.abs(c[:,None]-c[None]).sum(-1)`) and keep the upper triangle with offset 2 where the value is 1. `energy` sums `eps` over that mask. Build `interaction_matrix` with NumPy: start with `OTHER_EPSILON` everywhere, overwrite H–H pairs and charged–charged pairs (`COULOMB_K * np.outer(q, q)`). Do not loop over pairs in Python; Task 4's performance test depends on this.
- [ ] **Step 4: Run** `pytest tests/test_energy.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: pH-aware interaction matrix and lattice energy`

---

### Task 3: Lattice and self-avoiding move set

**Files:**
- Create: `misfold/lattice.py`
- Test: `tests/test_lattice.py`

**Interfaces:**
- Produces: `straight_chain(n: int) -> Coords` (bead `i` at `(i,0,0)`); `is_valid_chain(coords: Coords) -> bool` (every bond has Manhattan length 1 and all sites are unique); `propose_move(coords: Coords, rng: np.random.Generator) -> Coords | None`. It returns a **new** array, or `None` if the randomly chosen bead and move type give no legal target. It never changes its input.

Move rules (pick bead `i` uniformly from `0..n-1`; collect every legal target for that bead and choose one at random; `None` if there are none):
- **End move** (`i == 0` or `n-1`): move to any free lattice neighbour of its only bonded bead.
- **Corner flip** (`0 < i < n-1`, and `coords[i-1]`, `coords[i]`, `coords[i+1]` form a 90° corner): move to `coords[i-1] + coords[i+1] - coords[i]` if that site is free.
- **Crankshaft** (`0 < i < n-2`, and `i`, `i+1` form a U with `i-1`, `i+2`, i.e. `coords[i-1]` and `coords[i+2]` are lattice neighbours): rotate the pair 90° or 180° about the axis `i-1 → i+2`; both new sites must be free. Choose among the legal rotations at random.

- [ ] **Step 1: Write failing tests**

```python
def test_straight_chain_valid():
    assert is_valid_chain(straight_chain(13))

def test_overlap_and_broken_bond_invalid():
    assert not is_valid_chain(np.array([[0,0,0],[1,0,0],[0,0,0]]))
    assert not is_valid_chain(np.array([[0,0,0],[2,0,0]]))

def test_straight_chain_can_move():   # reference code bug regression
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
    c = np.array([[0,0,0],[1,0,0],[1,1,0]])
    rng = np.random.default_rng(2)
    moved = [new for new in (propose_move(c, rng) for _ in range(300))
             if new is not None and not np.array_equal(new[1], c[1])]
    assert moved and all(np.array_equal(m[1], [0, 1, 0]) for m in moved)
```

- [ ] **Step 2: Run** `pytest tests/test_lattice.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the three move types as private helpers that return lists of candidate coord arrays. Check site occupancy with a `set` of tuples.
- [ ] **Step 4: Run** `pytest tests/test_lattice.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: self-avoiding lattice move set`

---

### Task 4: Metropolis engine + reference annealing

**Files:**
- Create: `misfold/engine.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: `interaction_matrix`, `energy`, `straight_chain`, `propose_move`, `is_valid_chain`, `NEUTRAL_PH`.
- Produces:
  - `@dataclass(frozen=True) class SimulationResult: coords: Coords; energy_history: list[float]; accepted: int; attempted: int` plus `@property acceptance_rate -> float` (`accepted / attempted`, `0.0` if `attempted == 0`). `attempted` counts every step, including steps where `propose_move` returned `None`.
  - `metropolis_accept(delta_e: float, temperature: float, rng: np.random.Generator) -> bool`
  - `simulate(seq: str, temperature: float, ph: float, steps: int, rng: np.random.Generator) -> SimulationResult`: builds the interaction matrix once, starts from `straight_chain`, makes one proposal per step, and appends the current energy after every step, so `len(energy_history) == steps`.
  - `reference_energy(seq: str, ph: float = NEUTRAL_PH, steps: int = 20_000, seed: int = 0) -> float`: simulated annealing with T falling geometrically from `2.0` to `0.1`. Returns the lowest energy seen. Deterministic for a given set of arguments.

- [ ] **Step 1: Write failing tests**

```python
SEQ = "HKHPHDHPKHHDH"

def test_accept_downhill_always():
    rng = np.random.default_rng(0)
    assert all(metropolis_accept(-0.5, 0.1, rng) for _ in range(100))

def test_accept_uphill_rate_matches_boltzmann():
    rng = np.random.default_rng(0)
    rate = np.mean([metropolis_accept(1.0, 1.0, rng) for _ in range(20_000)])
    assert rate == pytest.approx(np.exp(-1.0), abs=0.02)

def test_history_length_and_energy_moves():
    r = simulate(SEQ, 1.2, 7.0, 300, np.random.default_rng(0))
    assert len(r.energy_history) == 300
    assert r.attempted == 300 and r.accepted > 0
    assert min(r.energy_history) < 0                     # chain actually folds
    assert r.energy_history[-1] == pytest.approx(energy(r.coords, interaction_matrix(SEQ, 7.0)))

@pytest.mark.parametrize("t, ph", [(0.1, 7.0), (5.0, 7.0), (1.2, 0.0), (1.2, 14.0)])
def test_chain_valid_at_all_conditions(t, ph):
    r = simulate("HKPDHHKPDHPKHHDPHKDH", t, ph, 1000, np.random.default_rng(3))
    assert is_valid_chain(r.coords)

def test_deterministic_with_seed():
    a = simulate("HKHDPH", 1.0, 7.0, 200, np.random.default_rng(7))
    b = simulate("HKHDPH", 1.0, 7.0, 200, np.random.default_rng(7))
    assert a.energy_history == b.energy_history

def test_reference_energy_deterministic_and_low():
    e = reference_energy(SEQ)
    assert e == reference_energy(SEQ) and e <= -3.0

def test_low_ph_raises_reference_energy():   # salt bridges lost
    assert reference_energy(SEQ, ph=2.0) > reference_energy(SEQ, ph=7.0)

def test_max_length_is_interactive():
    seq = "HKPD" * 15
    t0 = time.perf_counter()
    simulate(seq, 1.2, 7.0, 1000, np.random.default_rng(0)); reference_energy(seq)
    assert time.perf_counter() - t0 < 10.0
```

- [ ] **Step 2: Run** `pytest tests/test_engine.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** In `metropolis_accept`, return `True` when `delta_e < 0`, else `rng.random() < math.exp(-delta_e / temperature)`. Share one private `_run(eps, n, temperature_schedule, rng)` loop between `simulate` (constant schedule) and `reference_energy` (geometric schedule).
- [ ] **Step 4: Run** `pytest tests/test_engine.py -v`. Expected: PASS. If `e <= -3.0` fails, print what `reference_energy` actually returns, set the bound to that measured value (the test guards against regressions, not a known optimum), and note this in the commit message.
- [ ] **Step 5: Commit** `feat: Metropolis engine and annealing reference`

---

### Task 5: Data-driven diagnostics

**Files:**
- Create: `misfold/diagnostics.py`
- Test: `tests/test_diagnostics.py`

**Interfaces:**
- Consumes: `SimulationResult`, `simulate`, `reference_energy`.
- Produces:
  - `class Status(Enum)` with `TRAPPED = "Severely Misfolded (Kinetic Trap)"`, `DENATURED = "Denatured"`, `NATIVE = "Stable Native Fold"`.
  - `NATIVE_FRACTION = 0.8`, `DENATURED_ACCEPTANCE = 0.3` (tunable module constants).
  - `@dataclass(frozen=True) class Diagnosis: status: Status | None; message: str`. `status is None` means no stabilising contact is possible.
  - `classify(result: SimulationResult, native_energy: float, temperature: float, ph: float) -> Diagnosis`. `native_energy` is always `reference_energy(seq)` at pH 7 (see Deviation 3).

Rules, applied in order (`final = result.energy_history[-1]`):
1. `native_energy >= 0` → `status=None`, message says the sequence cannot form any stabilising contact.
2. `final <= NATIVE_FRACTION * native_energy` → `NATIVE`.
3. `result.acceptance_rate >= DENATURED_ACCEPTANCE` → `DENATURED`.
4. Otherwise → `TRAPPED`.

Messages follow the spec §5 wording, with emoji prefixes `🛑` / `⚠️` / `✅`, and include `temperature`, `ph`, the final energy and `native_energy`, each formatted `:.2f`. When `DENATURED` and `abs(ph - 7.0) >= 3`, the message credits pH ("acid"/"base" denaturation, lost salt bridges) instead of thermal noise.

- [ ] **Step 1: Write failing tests.** Unit tests build `SimulationResult` objects by hand:

```python
def _r(final, acc, att=100):
    return SimulationResult(np.zeros((3, 3), int), [final], acc, att)

def test_native(): assert classify(_r(-4.0, 10), -4.5, 1.2, 7.0).status is Status.NATIVE
def test_denatured(): assert classify(_r(-0.5, 60), -4.5, 4.5, 7.0).status is Status.DENATURED
def test_trapped(): assert classify(_r(-1.0, 5), -4.5, 0.2, 7.0).status is Status.TRAPPED
def test_no_core(): assert classify(_r(0.0, 50), 0.0, 1.2, 7.0).status is None

def test_message_contains_numbers():
    m = classify(_r(-1.0, 5), -4.5, 0.2, 7.0).message
    assert "-1.00" in m and "-4.50" in m and "0.20" in m and "7.00" in m

def test_acid_denaturation_message():
    m = classify(_r(-0.5, 60), -4.5, 1.2, 2.0).message
    assert "acid" in m.lower()
```

  Calibration test (pins the thresholds to real behaviour). `CHARGED = "KDKDHHKDKDHHKD"` is salt-bridge-rich, so pH has a strong effect on it:

```python
@pytest.mark.parametrize("seq, t, ph, expected", [
    ("HKHPHDHPKHHDH", 0.2, 7.0, Status.TRAPPED),
    ("HKHPHDHPKHHDH", 1.2, 7.0, Status.NATIVE),
    ("HKHPHDHPKHHDH", 4.5, 7.0, Status.DENATURED),
    ("KDKDHHKDKDHHKD", 1.2, 7.0, Status.NATIVE),
    ("KDKDHHKDKDHHKD", 1.2, 2.0, Status.DENATURED),
])
def test_calibration(seq, t, ph, expected):
    ref = reference_energy(seq)
    hits = sum(classify(simulate(seq, t, ph, 300, np.random.default_rng(s)), ref, t, ph).status
               is expected for s in range(10))
    assert hits >= 6
```

- [ ] **Step 2: Run** `pytest tests/test_diagnostics.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `classify`.**
- [ ] **Step 4: Run** `pytest tests/test_diagnostics.py -v`. Expected: PASS. If calibration fails, tune `NATIVE_FRACTION` / `DENATURED_ACCEPTANCE` first; you may also replace `CHARGED` with another sequence that has at least half its residues K/D. If nothing satisfies every row, stop and report the measured status counts per row. Do not weaken the test.
- [ ] **Step 5: Commit** `feat: data-driven folding diagnostics with pH`

---

### Task 6: Plot builders

**Files:**
- Create: `misfold/plots.py`
- Test: `tests/test_plots.py`

**Interfaces:**
- Produces: `RESIDUE_COLORS: dict[str, str]` (`H`, `P`, `K`, `D` → hex from Global Constraints), `BACKBONE_COLOR`, `ENERGY_COLOR`; `plot_structure(coords: Coords, seq: str, temperature: float, ph: float) -> matplotlib.figure.Figure` (figsize `(6, 6)`, 3D axes, backbone line + scatter with `s=150`, `edgecolors="black"`, `depthshade=False`, title `f"Final Simulated State (Temp: {temperature}, pH: {ph})"`, legend listing only the residue types present, equal box aspect so the lattice is not distorted); `plot_energy(history: list[float]) -> Figure` (figsize `(6, 5.3)`, linewidth 2, axis labels from Global Constraints, dashed grid `alpha=0.6`).

- [ ] **Step 1: Write failing tests**

```python
C = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]])

def test_structure_colors_and_title():
    ax = plot_structure(C, "HPKD", 1.2, 7.0).axes[0]
    assert ax.name == "3d" and ax.get_title() == "Final Simulated State (Temp: 1.2, pH: 7.0)"
    fc = [to_hex(c) for c in ax.collections[0].get_facecolors()]
    assert fc == [RESIDUE_COLORS[r].lower() for r in "HPKD"]

def test_legend_only_present_types():
    ax = plot_structure(C[:3], "HPH", 1.2, 7.0).axes[0]
    assert sorted(t.get_text() for t in ax.get_legend().get_texts()) == ["Hydrophobic (H)", "Polar (P)"]

def test_energy_labels():
    ax = plot_energy([0.0, -1.0, -2.0]).axes[0]
    assert ax.get_ylabel() == "Free Energy (ΔG)"
    assert ax.get_xlabel() == "Simulation Step / Time"
    assert list(ax.lines[0].get_ydata()) == [0.0, -1.0, -2.0]

def test_no_pyplot_figures_leak():
    plot_structure(C, "HPKD", 1.0, 7.0); plot_energy([0.0])
    assert plt.get_fignums() == []
```

  Legend labels: `Hydrophobic (H)`, `Polar (P)`, `Basic (K)`, `Acidic (D)`.

- [ ] **Step 2: Run** `pytest tests/test_plots.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** with `Figure()` and `fig.add_subplot(...)` (no `plt.figure`). Build the legend from proxy `Line2D` markers. Call `ax.set_box_aspect` with the axis ranges (minimum 1 per axis).
- [ ] **Step 4: Run** `pytest tests/test_plots.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: structure and energy plot builders`

---

### Task 7: Streamlit app

**Files:**
- Create: `app.py`, `README.md` (install and `streamlit run app.py`, one paragraph on the model including pH, the extensions and the deviations)
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: everything above.
- Layout per spec §4 plus the pH slider. Sidebar header `"Simulation Settings"`; text input label `"Amino Acid Sequence (H = Hydrophobic, P = Polar, K = Basic, D = Acidic)"`; sliders `"Simulation Temperature (T)"`, `"Solution pH"`, `"Simulation Steps"`; the button. On click: `validate_sequence` (on `SequenceError`, show `st.sidebar.error(str(e))` and stop); run `simulate` with `np.random.default_rng()` inside `st.spinner("Calculating thermodynamic trajectories...")`; get `reference_energy(seq)` (pH 7) through a `@st.cache_data` wrapper keyed on the sequence. Two columns: `"3D Backbone Conformation"` / `"Thermodynamic Energy Landscape"` via `st.pyplot(fig)`. Then a `"Diagnostic Evaluation"` subheader that maps the status to `st.error` (TRAPPED), `st.warning` (DENATURED), `st.success` (NATIVE) or `st.info` (None), followed by `st.caption(f"Net charge at pH {ph}: {net_charge(seq, ph):+.2f}")`.

- [ ] **Step 1: Write failing tests** with `streamlit.testing.v1.AppTest`:

```python
def _run(seq=None, temp=None, ph=None):
    at = AppTest.from_file("app.py", default_timeout=30).run()
    if seq is not None: at.sidebar.text_input[0].set_value(seq)
    if temp is not None: at.sidebar.slider[0].set_value(temp)
    if ph is not None: at.sidebar.slider[1].set_value(ph)
    at.sidebar.button[0].click().run()
    return at

def test_defaults_render():
    at = AppTest.from_file("app.py").run()
    assert at.sidebar.text_input[0].value == "HKHPHDHPKHHDH"
    assert [s.value for s in at.sidebar.slider] == [1.2, 7.0, 300]
    assert at.sidebar.button[0].label == "Run 3D Simulation"

def _diagnoses(at): return len(at.error) + len(at.warning) + len(at.success) + len(at.info)

def test_run_shows_diagnosis_and_charge():
    at = _run()
    assert not at.exception and _diagnoses(at) == 1
    assert "Net charge at pH 7.0" in at.caption[0].value

@pytest.mark.parametrize("ph", [0.0, 14.0])
def test_extreme_ph_runs(ph):
    at = _run(ph=ph)
    assert not at.exception and _diagnoses(at) == 1

def test_invalid_sequence_shows_error():
    at = _run(seq="HHXP")
    assert not at.exception and "only H, P, K and D" in at.sidebar.error[0].value

def test_no_core_sequence_shows_info():
    at = _run(seq="PPPPP")
    assert not at.exception and len(at.info) == 1
```

- [ ] **Step 2: Run** `pytest tests/test_app.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `app.py`.**
- [ ] **Step 4: Run** the full suite `pytest -v`. Expected: all PASS. Then `streamlit run app.py` and click Run at (T, pH) = (0.2, 7), (1.2, 7), (4.5, 7), (1.2, 2). Check that the chain is folded (not straight) at (1.2, 7), the energy trace is not flat, and (1.2, 2) gives a higher final energy than (1.2, 7).
- [ ] **Step 5: Commit** `feat: Streamlit dashboard with pH control`
