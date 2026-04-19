"""
Irodov Electrodynamics Simulator
A Streamlit app to visualize electrostatics, magnetostatics,
and magnetodynamics problems from I.E. Irodov.
"""
import os
import json
import numpy as np
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from physics.electrostatics import (
    superposition_E, superposition_V,
    e_field_dipole, e_field_ring, e_field_infinite_line,
    make_grid_2d, make_grid_3d,
)
from physics.magnetodynamics import (
    b_field_circular_loop, b_field_infinite_wire, b_field_solenoid,
    b_field_uniform, solve_trajectory, cyclotron_radius, cyclotron_frequency,
)
from visualizations.field_plots import (
    plot_2d_field, plot_3d_field, plot_trajectory, plot_trajectory_analysis,
)
from problems.irodov_presets import get_preset_names, get_preset
from ai_parser import parse_problem, fallback_scene_from_text

st.set_page_config(
    page_title="Irodov EM Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
body, .stApp { background-color: #0D1117; color: #E6EDF3; }
.stSidebar { background-color: #161B22; }
.stButton>button {
    background: linear-gradient(135deg, #FF6B35, #FF4136);
    color: white; border: none; border-radius: 8px;
    padding: 0.5rem 1.5rem; font-weight: 600;
    transition: all 0.2s;
}
.stButton>button:hover { transform: scale(1.02); opacity: 0.9; }
.metric-card {
    background: #161B22; border: 1px solid #30363D;
    border-radius: 10px; padding: 1rem; margin: 0.4rem 0;
}
h1, h2, h3 { color: #E6EDF3 !important; }
.stTabs [data-baseweb="tab"] { color: #8B949E; }
.stTabs [aria-selected="true"] { color: #FF6B35 !important; border-bottom-color: #FF6B35 !important; }
.problem-box {
    background: #161B22; border-left: 4px solid #FF6B35;
    border-radius: 6px; padding: 1rem 1.2rem; margin: 0.5rem 0;
    font-style: italic; color: #C9D1D9;
}
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Irodov EM Simulator")
    st.markdown("---")

    mode = st.radio("Input Mode", ["📚 Preset Problems", "✏️ Custom / AI Parse"], index=0)

    scene_config = None
    preset_name = None

    if mode == "📚 Preset Problems":
        preset_name = st.selectbox("Select Problem", get_preset_names())
        if preset_name:
            scene_config = get_preset(preset_name)
            st.markdown(f'<div class="problem-box">{scene_config["description"]}</div>',
                        unsafe_allow_html=True)

    else:
        st.markdown("**Paste an Irodov problem:**")
        problem_text = st.text_area(
            "", height=160,
            placeholder="e.g. An electron moves with velocity v = 3×10⁶ m/s at 45° to a uniform field B = 0.1 T. Trace the path.",
        )

        api_key = st.text_input(
            "Gemini API Key", type="password",
            value=os.getenv("GEMINI_API_KEY", ""),
            help="Get a free key at https://aistudio.google.com/apikey",
        )

        col_ai, col_fb = st.columns(2)
        with col_ai:
            if st.button("🤖 AI Parse", use_container_width=True) and problem_text:
                if not api_key:
                    st.error("Enter a Gemini API key.")
                else:
                    with st.spinner("Parsing with Gemini..."):
                        try:
                            scene_config = parse_problem(problem_text, api_key)
                            st.session_state["scene_config"] = scene_config
                            st.success("Parsed!")
                        except Exception as exc:
                            st.error(f"AI parse failed: {exc}")
                            scene_config = fallback_scene_from_text(problem_text)
                            st.warning("Using rule-based fallback.")
        with col_fb:
            if st.button("⚙️ Quick Parse", use_container_width=True) and problem_text:
                scene_config = fallback_scene_from_text(problem_text)
                st.session_state["scene_config"] = scene_config
                st.info("Fallback parse applied.")

        if "scene_config" in st.session_state and scene_config is None:
            scene_config = st.session_state["scene_config"]

    st.markdown("---")

    # Show raw JSON of current scene
    if scene_config:
        with st.expander("🔍 Scene JSON"):
            st.json(scene_config)


# ─── Helper: build E/B field from scene objects ────────────────────────────

def build_e_field(objects, grid):
    """Dispatch object kinds to their E-field calculators."""
    E = np.zeros_like(grid, dtype=float)
    V = np.zeros(len(grid), dtype=float)
    for obj in objects:
        kind = obj.get("kind")
        if kind == "point_charge":
            from physics.electrostatics import e_field_point_charge, potential_point_charge
            E += e_field_point_charge(obj["q"], obj["position"], grid)
            V += potential_point_charge(obj["q"], obj["position"], grid)
        elif kind == "ring_charge":
            E += e_field_ring(obj["Q"], obj["R"], obj["axis"], obj["center"], grid)
        elif kind == "dipole":
            E += e_field_dipole(obj["p_vec"], obj["center"], grid)
        elif kind == "line_charge":
            E += e_field_infinite_line(obj["lambda"], obj["axis_point"], obj["axis_dir"], grid)
    return E, V


def build_b_field(objects, grid):
    """Dispatch object kinds to their B-field calculators."""
    B = np.zeros_like(grid, dtype=float)
    for obj in objects:
        kind = obj.get("kind")
        if kind == "circular_loop":
            B += b_field_circular_loop(obj["I"], obj["R"], obj["axis"], obj["center"], grid)
        elif kind == "infinite_wire":
            B += b_field_infinite_wire(obj["I"], obj["axis_point"], obj["axis_dir"], grid)
        elif kind == "solenoid":
            B += b_field_solenoid(
                obj["I"], obj["n_turns"], obj["length"],
                obj["R"], obj["axis"], obj["center"], grid,
            )
    return B


def extract_charges(objects):
    """Return (charges, positions) lists for point charges."""
    charges, positions = [], []
    for obj in objects:
        if obj.get("kind") == "point_charge":
            charges.append(obj["q"])
            positions.append(obj["position"])
    return charges, positions


# ─── Main content ──────────────────────────────────────────────────────────

st.markdown("# ⚡ Irodov Electrodynamics Simulator")

if scene_config is None:
    st.info("👈 Select a preset or paste a problem in the sidebar to begin.")
    st.markdown("""
    ### What this simulator can do
    - **Electrostatics**: E field & equipotential maps for point charges, rings, dipoles, line charges
    - **Magnetostatics**: B field from wires, loops, and solenoids
    - **Magnetodynamics**: Particle trajectory under Lorentz force (helical, cycloidal motion)
    - **AI Parsing**: Paste any Irodov problem text and let Gemini extract the physics scene

    ### Covered Irodov Chapters
    | Chapter | Topics |
    |---------|--------|
    | 3.1 | Constant electric field in vacuum |
    | 3.2 | Conductors and dielectrics |
    | 3.6 | Magnetic field in vacuum |
    | 3.4 / 3.5 | Electric current, charged particle motion |
    """)
    st.stop()

sim_type = scene_config.get("type", "electrostatics")
scene = scene_config.get("scene", {})

# ─── ELECTROSTATICS ────────────────────────────────────────────────────────
if sim_type == "electrostatics":
    objects = scene.get("objects", [])
    view = scene.get("view", {"xlim": [-0.3, 0.3], "ylim": [-0.3, 0.3], "z_slice": 0.0})
    xlim = view.get("xlim", [-0.3, 0.3])
    ylim = view.get("ylim", [-0.3, 0.3])
    z_val = view.get("z_slice", 0.0)

    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown("### ⚙️ Parameters")
        n_grid = st.slider("Grid resolution", 20, 60, 30, step=5)
        show_3d = st.toggle("3D view", value=False)

    grid2d, XX, YY = make_grid_2d(xlim, ylim, z_val=z_val, n=n_grid)
    E2d, V2d = build_e_field(objects, grid2d)
    charges, positions = extract_charges(objects)

    tab1, tab2 = st.tabs(["🗺️ Field Map (2D)", "🌐 3D Field"])

    with tab1:
        fig2d = plot_2d_field(
            grid2d, E2d, XX, YY,
            charges=charges if charges else None,
            positions=positions if positions else None,
            title=f"E Field – {preset_name or 'Custom'}",
            potential=V2d,
        )
        st.plotly_chart(fig2d, use_container_width=True)

        # Axis plot if requested
        if scene.get("axis_plot"):
            axis_range = scene.get("axis_range", xlim)
            z_axis = np.linspace(*axis_range, 300)
            axis_grid = np.stack([np.zeros_like(z_axis), np.zeros_like(z_axis), z_axis], axis=-1)
            E_axis, _ = build_e_field(objects, axis_grid)
            E_z = E_axis[:, 2]
            import plotly.graph_objects as go
            fig_ax = go.Figure()
            fig_ax.add_trace(go.Scatter(
                x=z_axis, y=E_z, mode="lines",
                line=dict(color="#FF6B35", width=2.5), name="|E_z| on axis",
            ))
            fig_ax.update_layout(
                title="E field along axis", xaxis_title="z (m)", yaxis_title="E_z (V/m)",
                paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
                font=dict(color="white"), height=320,
                xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
            )
            st.plotly_chart(fig_ax, use_container_width=True)

    with tab2:
        if show_3d:
            grid3d, *_ = make_grid_3d(xlim, ylim, [xlim[0], xlim[1]], n=12)
            E3d, _ = build_e_field(objects, grid3d)
            span = xlim[1] - xlim[0]
            # Build source markers for charges
            src_markers = []
            for obj in objects:
                if obj.get("kind") == "point_charge":
                    q_val = obj["q"]
                    src_markers.append({
                        "pos": obj["position"],
                        "label": f"q = {q_val:.2e} C",
                        "color": PALETTE["charge_pos"] if q_val > 0 else PALETTE["charge_neg"],
                    })
                elif obj.get("kind") == "ring_charge":
                    src_markers.append({
                        "pos": obj.get("center", [0, 0, 0]),
                        "label": f"Ring  Q={obj['Q']:.2e} C  R={obj['R']} m",
                        "color": "#FFD700",
                    })
                elif obj.get("kind") == "dipole":
                    src_markers.append({
                        "pos": obj.get("center", [0, 0, 0]),
                        "label": "Electric dipole",
                        "color": "#AA77FF",
                    })
            fig3d = plot_3d_field(
                grid3d, E3d,
                title=f"3D Electric Field – {preset_name or 'Custom'}",
                field_label="Electric Field E",
                field_unit="V/m",
                source_markers=src_markers or None,
                scale=span * 0.06,
            )
            st.plotly_chart(fig3d, use_container_width=True)
        else:
            st.info("Enable '3D view' in the sidebar panel to render the 3D field (slower).")

    # Metrics
    E_mag = np.linalg.norm(E2d, axis=-1)
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Max |E| (V/m)", f"{np.nanmax(E_mag):.3e}")
    c2.metric("Min |V| (V)", f"{np.nanmin(np.abs(V2d)):.3e}")
    c3.metric("Max |V| (V)", f"{np.nanmax(np.abs(V2d)):.3e}")


# ─── MAGNETOSTATICS ────────────────────────────────────────────────────────
elif sim_type == "magnetostatics":
    objects = scene.get("objects", [])
    view = scene.get("view", {"xlim": [-0.15, 0.15], "ylim": [-0.15, 0.15], "z_slice": 0.0})
    xlim = view.get("xlim", [-0.15, 0.15])
    ylim = view.get("ylim", [-0.15, 0.15])
    z_val = view.get("z_slice", 0.0)

    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown("### ⚙️ Parameters")
        n_grid = st.slider("Grid resolution", 15, 50, 25, step=5)
        show_3d = st.toggle("3D view", value=False)

    grid2d, XX, YY = make_grid_2d(xlim, ylim, z_val=z_val, n=n_grid)
    B2d = build_b_field(objects, grid2d)

    tab1, tab2 = st.tabs(["🗺️ B Field Map (2D)", "🌐 3D B Field"])

    with tab1:
        fig2d = plot_2d_field(
            grid2d, B2d, XX, YY,
            title=f"B Field – {preset_name or 'Custom'}",
        )
        st.plotly_chart(fig2d, use_container_width=True)

        if scene.get("axis_plot"):
            axis_range = scene.get("axis_range", xlim)
            z_axis = np.linspace(*axis_range, 300)
            axis_grid = np.stack([np.zeros_like(z_axis), np.zeros_like(z_axis), z_axis], axis=-1)
            B_axis = build_b_field(objects, axis_grid)
            B_z = B_axis[:, 2]
            import plotly.graph_objects as go
            fig_ax = go.Figure()
            fig_ax.add_trace(go.Scatter(
                x=z_axis, y=B_z, mode="lines",
                line=dict(color="#4ECDC4", width=2.5), name="B_z on axis",
            ))
            fig_ax.update_layout(
                title="B field along axis", xaxis_title="z (m)", yaxis_title="B_z (T)",
                paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
                font=dict(color="white"), height=320,
                xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
            )
            st.plotly_chart(fig_ax, use_container_width=True)

    with tab2:
        if show_3d:
            grid3d, *_ = make_grid_3d(xlim, ylim, [xlim[0], xlim[1]], n=10)
            B3d = build_b_field(objects, grid3d)
            span = xlim[1] - xlim[0]
            src_markers_b = []
            for obj in objects:
                if obj.get("kind") == "circular_loop":
                    src_markers_b.append({
                        "pos": obj.get("center", [0, 0, 0]),
                        "label": f"Loop  I={obj['I']} A  R={obj['R']} m",
                        "color": PALETTE["B_field"],
                    })
                elif obj.get("kind") == "solenoid":
                    src_markers_b.append({
                        "pos": obj.get("center", [0, 0, 0]),
                        "label": f"Solenoid  I={obj['I']} A  N={obj['n_turns']}",
                        "color": "#AA77FF",
                    })
            fig3d = plot_3d_field(
                grid3d, B3d,
                title=f"3D Magnetic Field – {preset_name or 'Custom'}",
                field_label="Magnetic Field B",
                field_unit="T",
                source_markers=src_markers_b or None,
                scale=span * 0.06,
            )
            st.plotly_chart(fig3d, use_container_width=True)
        else:
            st.info("Enable '3D view' to render the 3D field.")

    B_mag = np.linalg.norm(B2d, axis=-1)
    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric("Max |B| (T)", f"{np.nanmax(B_mag):.4e}")
    c2.metric("Mean |B| (T)", f"{np.nanmean(B_mag):.4e}")


# ─── MAGNETODYNAMICS ───────────────────────────────────────────────────────
elif sim_type == "magnetodynamics":
    particle = scene.get("particle", {})
    fields_cfg = scene.get("fields", {})
    t_span = scene.get("t_span", [0, 1e-8])

    q = particle.get("q", -1.6e-19)
    m = particle.get("m", 9.11e-31)
    r0 = np.array(particle.get("r0", [0, 0, 0]), dtype=float)
    v0 = np.array(particle.get("v0", [1e6, 0, 1e6]), dtype=float)

    E_vec = np.array(fields_cfg.get("E", [0, 0, 0]), dtype=float)
    B_vec = np.array(fields_cfg.get("B", [0, 0, 0.1]), dtype=float)
    B_mag = np.linalg.norm(B_vec)

    # Derive species label
    m_e, m_p = 9.11e-31, 1.67e-27
    if q < 0 and abs(m - m_e) / m_e < 0.05:
        species = "Electron"
    elif q > 0 and abs(m - m_p) / m_p < 0.05:
        species = "Proton"
    elif q < 0:
        species = "Negative charge"
    else:
        species = "Positive charge"

    # Sidebar controls
    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown("### ⚙️ Particle")
        q_sign = species
        st.markdown(f"**Species:** {q_sign}")
        st.markdown(f"**Charge:** `{q:.3e} C`")
        st.markdown(f"**Mass:** `{m:.3e} kg`")
        v0_mag = np.linalg.norm(v0)
        st.markdown(f"**|v₀|:** `{v0_mag:.3e} m/s`")

        st.markdown("### 🧲 Fields")
        st.markdown(f"**E:** `{E_vec}`")
        st.markdown(f"**B:** `{B_vec}`")
        n_steps = st.slider("Time steps", 500, 5000, 2000, step=500)

    # Uniform field functions
    def E_func(r):
        return E_vec

    def B_func(r):
        return B_vec

    with st.spinner("Integrating trajectory (RK45)..."):
        traj = solve_trajectory(q, m, r0, v0, E_func, B_func,
                                t_span=tuple(t_span), n_steps=n_steps)

    if not traj["success"]:
        st.warning(f"Solver warning: {traj['message']}")

    tab1, tab2 = st.tabs(["🌀 3D Trajectory", "📊 Analysis"])

    with tab1:
        title = f"Charged Particle Trajectory – {preset_name or 'Custom'}"
        fig_traj = plot_trajectory(
            traj,
            B_vec=B_vec,
            E_vec=E_vec,
            charge_sign=int(np.sign(q)),
            species=species,
            title=title,
        )
        st.plotly_chart(fig_traj, use_container_width=True)

    with tab2:
        fig_analysis = plot_trajectory_analysis(traj, m, species=species)
        st.plotly_chart(fig_analysis, use_container_width=True)

    # Physics metrics
    st.markdown("---")
    v_perp = np.linalg.norm(v0 - np.dot(v0, B_vec / (B_mag + 1e-30)) * B_vec / (B_mag + 1e-30))
    v_par = abs(np.dot(v0, B_vec / (B_mag + 1e-30)))

    c1, c2, c3, c4 = st.columns(4)
    if B_mag > 1e-20:
        r_c = cyclotron_radius(m, v_perp, q, B_mag)
        omega_c = cyclotron_frequency(q, m, B_mag)
        T_c = 2 * np.pi / omega_c
        pitch = v_par * T_c
        c1.metric("Cyclotron radius (m)", f"{r_c:.4e}")
        c2.metric("Cyclotron freq (MHz)", f"{omega_c / 1e6:.4f}")
        c3.metric("Period (ns)", f"{T_c * 1e9:.4f}")
        c4.metric("Helix pitch (m)", f"{pitch:.4e}")
    else:
        c1.metric("|v₀| (m/s)", f"{np.linalg.norm(v0):.3e}")
        c2.metric("KE₀ (eV)", f"{0.5 * m * np.linalg.norm(v0)**2 / 1.6e-19:.3f}")


else:
    st.error(f"Unknown simulation type: `{sim_type}`. Expected electrostatics, magnetostatics, or magnetodynamics.")

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#8B949E; font-size:0.8rem;">'
    "Irodov EM Simulator · Physics Engine: NumPy/SciPy · Viz: Plotly · AI: Gemini"
    "</p>",
    unsafe_allow_html=True,
)
