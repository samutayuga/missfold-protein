# 3D Protein Misfolding Simulator

Interactive Streamlit app that folds a toy protein on a 3D cubic lattice using
Metropolis Monte Carlo, under a chosen temperature and pH, then shows the final
conformation, the energy trace, and a diagnosis: *Stable Native Fold*,
*Severely Misfolded (Kinetic Trap)* or *Denatured*.

Spec: [`docs/specs/protein_misfolding_spec.md`](docs/specs/protein_misfolding_spec.md)

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/streamlit run app.py
.venv/bin/pytest            # test suite
```

## Model

- **Alphabet:** `H` hydrophobic, `P` polar, `K` basic (pKa 10.5), `D` acidic (pKa 3.9).
  Sequences are 3–20 residues: beyond about 20, the local move set below cannot
  reach the native fold within the app's step range, so every run would read as a trap.
- **Energy:** sum over non-bonded lattice contacts: `-1.0` for H–H, `0.5·qᵢ·qⱼ` for
  two charged residues, `+0.1` otherwise. Charges follow Henderson–Hasselbalch, so at
  pH 7 a K–D contact is a `-0.5` salt bridge; at low pH D is neutralised (bridges
  break, K–K repels), at high pH K is.
- **Moves:** end moves, corner flips and crankshaft rotations; the chain is always
  self-avoiding. The chain starts fully extended.
- **Steps:** one step is one Monte Carlo sweep (one proposal per residue).
- **Temperature:** reduced units. Acceptance is `exp(-ΔE / (T / 6))`, which puts the
  folding transition where the spec places it: roughly T < 0.5 traps, T ≈ 1.2 folds,
  T > 3 denatures.
- **Diagnosis** comes from the run's data, not the slider values. Each run is compared
  with the best energy found by simulated annealing at **pH 7** (1500 sweeps, cached
  per sequence):
  - final energy ≤ 80 % of that optimum → *Stable Native Fold*;
  - otherwise, if ≥ 25 % of energy-raising moves were accepted (the chain is
    thermally agitated) → *Denatured*, attributed to acid (base) when pH is ≥ 3 units
    below (above) neutral and the sequence contains D (K); otherwise to heat;
  - otherwise → *Severely Misfolded (Kinetic Trap)*.
  - Sequences that cannot form any stabilising contact get an info message instead.

## Differences from the spec's reference code

1. The reference move (random ±1 jump that must keep both bonds at length 1) can never
   move a bead of a straight chain, so its chain never changes. Replaced by the lattice
   moves above.
2. Self-avoidance is enforced.
3. The diagnosis is computed from simulation data instead of the temperature slider.
4. Temperature is in reduced units (see above), because with the spec's energies the
   literal formula only folds below T ≈ 0.3.
5. pH and the `K`/`D` residues are an extension; the default sequence is
   `HKHPHDHPKHHDH` so pH has a visible effect.

Note: hydrophobic-rich sequences such as the spec's original `HHPHPHPHPHHPH` have a
rugged landscape and usually end in a kinetic trap within 300 sweeps.
