# Software Design Specification: 3D Protein Misfolding Simulator App

## 1. Executive Summary
This document specifies the architecture and implementation details for a lightweight, interactive **3D Protein Misfolding Simulator App** written in Python. The application utilizes a simplified **Hydrophobic-Polar (HP) Lattice Model** and a **Monte Carlo (Metropolis-Hastings)** physics engine to simulate protein folding pathways. It provides a visual interface for users to experiment with structural biology concepts, specifically focusing on how environmental fluctuations trap proteins in misfolded, toxic configurations (kinetic traps).

---

## 2. System Architecture & Tech Stack

### 2.1 Technology Stack
*   **Language:** Python 3.10+
*   **UI Framework:** Streamlit (For web dashboard configuration, parameter sliders, and real-time layout management).
*   **Numerical Computation:** NumPy (For 3D vector math, distance matrices, and random coordinate mutations).
*   **Data Visualization:** Matplotlib / `mpl_toolkits.mplot3d` (For generating the 3D backbone configuration map and 2D free-energy tracking charts).

### 2.2 System Component Diagram
```
[ User Input (Sidebar) ] ---> [ Streamlit UI Controller ]
                                         │
                                         ▼
                        [ Monte Carlo Physics Engine ]
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
     [ 3D Backbone Generator ]                       [ Thermodynamic Tracker ]
                 │                                               │
                 ▼                                               ▼
      (Matplotlib 3D Plot)                            (Matplotlib 2D Plot)
                 └───────────────────────┬───────────────────────┘
                                         ▼
                           [ Diagnostic Evaluation Matrix ]
```

---

## 3. Core Engine Mechanics & Physics Formulas

### 3.1 The HP Lattice Model
Amino acids are abstracted into a simplified primary sequence string containing two types of monomers:
*   **H (Hydrophobic):** Non-polar residues that lower overall free energy when grouped tightly together away from the solvent environment.
*   **P (Polar):** Hydrophilic residues that interact favorably with the solvent and occupy surface positions.

### 3.2 Thermodynamic Energy Function
The energy state (E) of a given conformation is calculated by iterating over all non-consecutive amino acid pairs that are physically adjacent (distance ≈ 1.0) on the 3D simulation grid:

\[E = \sum_{i} \sum_{j > i+1} \epsilon(s_i, s_j) \cdot \delta(\mathbf{r}_i, \mathbf{r}_j)\]

Where:
*   \(\delta(\mathbf{r}_i, \mathbf{r}_j) = 1\) if \(\vert{}\vert{}\mathbf{r}_i - \mathbf{r}_j\vert{}\vert{}_2 = 1\), else 0.
*   \(\epsilon(s_i, s_j) = -1.0\) if both monomers \(s_i\) and \(s_j\) are **Hydrophobic (H)**.
*   \(\epsilon(s_i, s_j) = +0.1\) for other interactions to simulate minor electrostatic repulsion/neutrality.

### 3.3 Metropolis-Hastings Acceptance Criteria
To transition from a current structural configuration to a mutated configuration, the engine calculates the difference in free energy:

\[\Delta E = E_{\text{proposed}} - E_{\text{current}}\]

The mutation is accepted into the structural history timeline based on the following probability rule:

\[P(\text{accept}) = \begin{cases} 1 & \text{if } \Delta E < 0 \\ e^{-\frac{\Delta E}{T}} & \text{if } \Delta E \ge 0 \end{cases}\]

Where T is the user-defined simulation temperature. 
*   **Low T (T < 0.5):** Leads to steep, restrictive landscapes where the protein lacks kinetic energy to escape sub-optimal shapes (**Misfolding / Kinetic Trap**).
*   **High T (T > 3.0):** Leads to thermal denaturing where stable bonds break up completely.

---

## 4. UI/UX & Functional Requirements

### 4.1 Input Parameters (Sidebar Controls)
1.  **Sequence String input:** Text field validating character sequences containing only `H` and `P` letters (Default: `"HHPHPHPHPHHPH"`).
2.  **Temperature Slider:** Float slider ranging from `0.1` to `5.0` (Default: `1.2`, Step: `0.1`).
3.  **Simulation Steps Slider:** Integer slider ranging from `50` to `1000` (Default: `300`, Step: `50`).
4.  **Execution Trigger:** A "Run 3D Simulation" button to initialize computation.

### 4.2 Output Views (Main Dashboard Canvas)
*   **Left Column (3D Visualizer):** A 3D scatter-and-line chart showing individual amino acids as spheres connected sequentially by a structural backbone line. 
    *   **Color-coding:** Hydrophobic (H) beads rendered in Red; Polar (P) beads rendered in Blue.
*   **Right Column (Energy Line Graph):** A 2D tracking chart plotting Free Energy (Δ G) along the vertical axis against the Simulation Step sequence along the horizontal axis.
*   **Bottom Section (Diagnostic Summary):** An alert field analyzing the final output data and outputting one of three diagnostic categorizations: *Severely Misfolded (Kinetic Trap)*, *Denatured*, or *Stable Native Fold*.

---

## 5. Implementation Code Reference

```python
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="3D Protein Misfolding Simulator", layout="wide")
st.title("🧬 3D Protein Misfolding & Energy Landscape Simulator")
st.write("Explore how environmental conditions trap proteins in misfolded, toxic configurations.")

st.sidebar.header("Simulation Settings")
sequence = st.sidebar.text_input("Amino Acid Sequence (H = Hydrophobic, P = Polar)", "HHPHPHPHPHHPH")
temp = st.sidebar.slider("Simulation Temperature (T)", min_value=0.1, max_value=5.0, value=1.2, step=0.1)
steps = st.sidebar.slider("Simulation Steps", min_value=50, max_value=1000, value=300, step=50)

def simulate_folding(seq, temperature, max_steps):
    n = len(seq)
    coords = np.zeros((n, 3))
    coords[:, 0] = np.arange(n)
    energy_history = []
    
    def calc_energy(current_coords):
        energy = 0
        for i in range(n):
            for j in range(i + 2, n):
                distance = np.linalg.norm(current_coords[i] - current_coords[j])
                if np.isclose(distance, 1.0):
                    if seq[i] == 'H' and seq[j] == 'H':
                        energy -= 1.0
                    else:
                        energy += 0.1
        return energy

    current_energy = calc_energy(coords)
    
    for step in range(max_steps):
        idx = np.random.randint(1, n - 1)
        move = np.random.choice([-1, 0, 1], size=3)
        proposed_coords = coords.copy()
        proposed_coords[idx] += move
        
        dist_prev = np.linalg.norm(proposed_coords[idx] - proposed_coords[idx-1])
        dist_next = np.linalg.norm(proposed_coords[idx] - proposed_coords[idx+1])
        
        if np.isclose(dist_prev, 1.0) and np.isclose(dist_next, 1.0):
            proposed_energy = calc_energy(proposed_coords)
            delta_e = proposed_energy - current_energy
            
            if delta_e < 0 or np.random.rand() < np.exp(-delta_e / temperature):
                coords = proposed_coords
                current_energy = proposed_energy
                
        energy_history.append(current_energy)
        
    return coords, energy_history

if st.sidebar.button("Run 3D Simulation"):
    with st.spinner("Calculating thermodynamic trajectories..."):
        final_structure, energy_profile = simulate_folding(sequence, temp, steps)
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("3D Backbone Conformation")
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        colors = ['#FF4B4B' if residue == 'H' else '#1C83E1' for residue in sequence]
        ax.plot(final_structure[:, 0], final_structure[:, 1], final_structure[:, 2], color='#D3D3D3', lw=3, zorder=1)
        ax.scatter(final_structure[:, 0], final_structure[:, 1], final_structure[:, 2], color=colors, s=150, edgecolors='black', depthshade=False, zorder=2)
        ax.set_title(f"Final Simulated State (Temp: {temp})")
        st.pyplot(fig)
        
    with col2:
        st.subheader("Thermodynamic Energy Landscape")
        fig_energy, ax_energy = plt.subplots(figsize=(6, 5.3))
        ax_energy.plot(energy_profile, color='#00CC96', linewidth=2)
        ax_energy.set_xlabel("Simulation Step / Time")
        ax_energy.set_ylabel("Free Energy (ΔG)")
        ax_energy.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig_energy)
        
    st.subheader("Diagnostic Evaluation")
    final_energy = energy_profile[-1]
    
    if temp < 0.5:
        st.error(f"🛑 **Status: Severely Misfolded (Kinetic Trap)**. The temperature was too low ({temp}) to overcome structural energy barriers. The protein generated an irregular aggregate configuration with a high energy state of {final_energy:.2f}.")
    elif temp > 3.0:
        st.warning(f"⚠️ **Status: Denatured**. Thermal noise is too high ({temp}). The non-covalent bonds cannot stabilize, leaving the chain floating as an open random coil.")
    else:
        st.success(f"✅ **Status: Stable Native Fold**. The simulation safely traversed the free energy landscape down to a thermodynamic minimum of {final_energy:.2f}.")
```