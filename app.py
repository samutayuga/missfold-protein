"""Streamlit dashboard for the 3D protein misfolding simulator."""

from pathlib import Path

import numpy as np
import streamlit as st

from misfold.diagnostics import Status, classify
from misfold.engine import reference_energy, simulate
from misfold.plots import plot_energy, plot_size_vs_ph, plot_structure
from misfold.sequence import SequenceError, net_charge, validate_sequence
from misfold.sweep import ph_sweep

# Original icon rendered by scripts/make_icon.py (half helix, half unravelled chain).
ICON = str(Path(__file__).resolve().parent / "assets" / "icon.png")

# Alert widget and Material Symbols icon per diagnosis.
ALERTS = {
    Status.TRAPPED: (st.error, ":material/error:"),
    Status.DENATURED: (st.warning, ":material/warning:"),
    Status.NATIVE: (st.success, ":material/check_circle:"),
    None: (st.info, ":material/info:"),
}


@st.cache_data(show_spinner=False)
def native_energy(seq: str) -> float:
    """Annealed optimum at neutral pH: the benchmark every run is graded against."""
    return reference_energy(seq)


@st.cache_data(show_spinner=False)
def size_vs_ph(seq: str, temperature: float, steps: int):
    """Fixed seed so the same settings always draw the same curve."""
    return ph_sweep(seq, temperature, steps, seed=0)


def validated_sequence(raw: str) -> str:
    try:
        return validate_sequence(raw)
    except SequenceError as e:
        st.sidebar.error(str(e), icon=":material/block:")
        st.stop()


st.set_page_config(
    page_title="3D Protein Misfolding Simulator", page_icon=ICON, layout="wide"
)
st.logo(ICON, size="large")
icon_col, title_col = st.columns([1, 12], vertical_alignment="center")
icon_col.image(ICON, width=72)
title_col.title("3D Protein Misfolding & Energy Landscape Simulator")
st.write(
    "Explore how temperature and pH trap proteins in misfolded, toxic configurations "
    "or unfold them entirely."
)

st.sidebar.header(":material/tune: Simulation Settings")
raw_sequence = st.sidebar.text_input(
    "Amino Acid Sequence (H = Hydrophobic, P = Polar, K = Basic, D = Acidic)", "HKHPHDHPKHHDH"
)
temp = st.sidebar.slider("Simulation Temperature (T)", min_value=0.1, max_value=5.0, value=1.2, step=0.1)
ph = st.sidebar.slider("Solution pH", min_value=0.0, max_value=14.0, value=7.0, step=0.5)
steps = st.sidebar.slider("Simulation Steps", min_value=50, max_value=1000, value=300, step=50)

# Results live in session_state so a click in one tab does not wipe the other.
if st.sidebar.button("Run 3D Simulation", icon=":material/play_arrow:", type="primary"):
    sequence = validated_sequence(raw_sequence)
    with st.spinner("Calculating thermodynamic trajectories..."):
        result = simulate(sequence, temp, ph, steps, np.random.default_rng())
        reference = native_energy(sequence)
    st.session_state["simulation"] = (sequence, temp, ph, result, reference)

sim_tab, sweep_tab = st.tabs([":material/view_in_ar: Simulation", ":material/ssid_chart: pH vs Size"])

with sim_tab:
    if "simulation" not in st.session_state:
        st.markdown("Press **Run 3D Simulation** in the sidebar to fold the chain.")
    else:
        sequence, run_temp, run_ph, result, reference = st.session_state["simulation"]
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(":material/view_in_ar: 3D Backbone Conformation")
            st.pyplot(plot_structure(result.coords, sequence, run_temp, run_ph))
        with col2:
            st.subheader(":material/show_chart: Thermodynamic Energy Landscape")
            st.pyplot(plot_energy(result.energy_history))

        st.subheader(":material/biotech: Diagnostic Evaluation")
        diagnosis = classify(result, reference, run_temp, run_ph, sequence)
        alert, icon = ALERTS[diagnosis.status]
        alert(diagnosis.message, icon=icon)
        st.caption(f":material/bolt: Net charge at pH {run_ph}: {net_charge(sequence, run_ph):+.2f}")

with sweep_tab:
    st.markdown(
        "Particle size is the chain's **radius of gyration** $R_g$: small when folded into "
        "a compact core, large when unfolded. Each pH (0–14) is simulated several times at "
        "the sidebar temperature and step count; the band is ±1 standard deviation and the "
        "dotted line marks the sidebar pH."
    )
    if st.button("Run pH Sweep", icon=":material/ssid_chart:", type="primary"):
        sequence = validated_sequence(raw_sequence)
        with st.spinner("Sweeping pH 0–14..."):
            st.session_state["sweep"] = (sequence, temp, size_vs_ph(sequence, temp, steps))
    if "sweep" in st.session_state:
        sequence, run_temp, sweep = st.session_state["sweep"]
        st.subheader(f":material/straighten: Size vs pH — {sequence} at T = {run_temp}")
        st.pyplot(plot_size_vs_ph(sweep.ph, sweep.mean_rg, sweep.std_rg, ph))
