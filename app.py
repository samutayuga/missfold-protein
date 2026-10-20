"""Streamlit dashboard for the 3D protein misfolding simulator."""

import numpy as np
import streamlit as st

from misfold.diagnostics import Status, classify
from misfold.engine import reference_energy, simulate
from misfold.plots import plot_energy, plot_structure
from misfold.sequence import SequenceError, net_charge, validate_sequence

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


st.set_page_config(
    page_title="3D Protein Misfolding Simulator", page_icon=":material/genetics:", layout="wide"
)
st.title(":material/genetics: 3D Protein Misfolding & Energy Landscape Simulator")
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

if st.sidebar.button("Run 3D Simulation", icon=":material/play_arrow:", type="primary"):
    try:
        sequence = validate_sequence(raw_sequence)
    except SequenceError as e:
        st.sidebar.error(str(e), icon=":material/block:")
        st.stop()

    with st.spinner("Calculating thermodynamic trajectories..."):
        result = simulate(sequence, temp, ph, steps, np.random.default_rng())
        reference = native_energy(sequence)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(":material/view_in_ar: 3D Backbone Conformation")
        st.pyplot(plot_structure(result.coords, sequence, temp, ph))
    with col2:
        st.subheader(":material/show_chart: Thermodynamic Energy Landscape")
        st.pyplot(plot_energy(result.energy_history))

    st.subheader(":material/biotech: Diagnostic Evaluation")
    diagnosis = classify(result, reference, temp, ph, sequence)
    alert, icon = ALERTS[diagnosis.status]
    alert(diagnosis.message, icon=icon)
    st.caption(f":material/bolt: Net charge at pH {ph}: {net_charge(sequence, ph):+.2f}")
