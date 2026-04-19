"""
Plotly-based visualization layer for EM fields and particle trajectories.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PALETTE = {
    "E_field": "#FF6B35",
    "B_field": "#4ECDC4",
    "trajectory": "#FFE66D",
    "charge_pos": "#FF4136",
    "charge_neg": "#0074D9",
    "neutral": "#AAAAAA",
    "background": "#0D1117",
    "surface": "#161B22",
}


def _normalize_vectors(U, V, W, scale=1.0):
    """Normalize arrow lengths for display."""
    mag = np.sqrt(U**2 + V**2 + W**2) + 1e-30
    max_mag = np.percentile(mag, 95)
    factor = scale / (max_mag + 1e-30)
    return U * factor, V * factor, W * factor


def plot_2d_field(
    grid2d, E, XX, YY,
    charges=None, positions=None,
    title="Electric Field",
    potential=None,
):
    """
    2D quiver + contour plot of an E (or B) field slice.
    grid2d : (N, 3)
    E      : (N, 3) field vectors
    XX, YY : meshgrid arrays (shape n×n)
    """
    n = XX.shape[0]
    Ex = E[:, 0].reshape(XX.shape)
    Ey = E[:, 1].reshape(XX.shape)
    mag = np.sqrt(Ex**2 + Ey**2) + 1e-30

    # Normalize arrows
    Ex_n = Ex / mag
    Ey_n = Ey / mag

    fig = go.Figure()

    # Field magnitude heatmap
    fig.add_trace(go.Heatmap(
        x=XX[0], y=YY[:, 0],
        z=np.log10(mag + 1),
        colorscale="Plasma",
        showscale=True,
        colorbar=dict(title=dict(text="log₁₀|E|", font=dict(color="white")), tickfont=dict(color="white")),
        opacity=0.6,
    ))

    # Equipotential contours
    if potential is not None:
        V = potential.reshape(XX.shape)
        fig.add_trace(go.Contour(
            x=XX[0], y=YY[:, 0], z=V,
            ncontours=15,
            contours=dict(coloring="none", showlabels=False),
            line=dict(color="rgba(255,255,255,0.4)", width=1),
            showscale=False,
        ))

    # Quiver arrows (subsample)
    step = max(1, n // 15)
    xs = XX[::step, ::step].ravel()
    ys = YY[::step, ::step].ravel()
    us = Ex_n[::step, ::step].ravel()
    vs = Ey_n[::step, ::step].ravel()

    arrow_scale = (XX[0, -1] - XX[0, 0]) / (n / step) * 0.4
    for x0, y0, u, v in zip(xs, ys, us, vs):
        fig.add_annotation(
            x=x0 + u * arrow_scale, y=y0 + v * arrow_scale,
            ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2, arrowsize=1, arrowwidth=1.2,
            arrowcolor=PALETTE["E_field"],
        )

    # Charge markers
    if charges and positions:
        for q, pos in zip(charges, positions):
            color = PALETTE["charge_pos"] if q > 0 else PALETTE["charge_neg"]
            symbol = "circle" if q > 0 else "x"
            fig.add_trace(go.Scatter(
                x=[pos[0]], y=[pos[1]],
                mode="markers+text",
                marker=dict(size=14, color=color, symbol=symbol,
                            line=dict(color="white", width=1)),
                text=[f"+{q:.1e}C" if q > 0 else f"{q:.1e}C"],
                textposition="top center",
                textfont=dict(color="white", size=11),
                showlegend=False,
            ))

    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=16)),
        paper_bgcolor=PALETTE["background"],
        plot_bgcolor=PALETTE["surface"],
        xaxis=dict(title="x (m)", color="white", gridcolor="#333"),
        yaxis=dict(title="y (m)", color="white", gridcolor="#333", scaleanchor="x"),
        margin=dict(l=40, r=40, t=60, b=40),
        height=520,
    )
    return fig


def plot_3d_field(
    grid, field,
    title="3D Field",
    field_label="Field",
    field_unit="V/m",
    source_markers=None,
    scale=1.0,
    max_arrows=400,
):
    """
    3D cone (arrow) plot of a vector field.
    grid          : (N, 3)
    field         : (N, 3)
    field_label   : display name shown in colorbar and legend (e.g. "Electric Field E")
    field_unit    : unit string for colorbar (e.g. "V/m" or "T")
    source_markers: list of dicts with keys 'pos', 'label', 'color' for source objects
    """
    # Subsample to keep rendering fast
    if len(grid) > max_arrows:
        idx = np.random.choice(len(grid), max_arrows, replace=False)
        grid = grid[idx]
        field = field[idx]

    mag = np.linalg.norm(field, axis=-1)
    max_mag = np.percentile(mag, 95) + 1e-30
    scale_factor = scale / max_mag

    # Hover text: position + magnitude for each arrow
    hover = [
        f"<b>Position</b><br>x={grid[i,0]:.3e} m<br>y={grid[i,1]:.3e} m<br>z={grid[i,2]:.3e} m"
        f"<br><b>|{field_label}|</b> = {mag[i]:.3e} {field_unit}"
        for i in range(len(grid))
    ]

    fig = go.Figure()

    # Field cones
    fig.add_trace(go.Cone(
        x=grid[:, 0], y=grid[:, 1], z=grid[:, 2],
        u=field[:, 0] * scale_factor,
        v=field[:, 1] * scale_factor,
        w=field[:, 2] * scale_factor,
        sizemode="absolute",
        sizeref=scale * 0.15,
        colorscale="Plasma",
        showscale=True,
        colorbar=dict(
            title=dict(
                text=f"|{field_label}|<br>({field_unit})",
                font=dict(color="white", size=12),
            ),
            tickfont=dict(color="white"),
            bgcolor="rgba(22,27,34,0.8)",
            bordercolor="#444",
        ),
        opacity=0.85,
        hovertext=hover,
        hoverinfo="text",
        name=f"{field_label} vectors",
        showlegend=True,
    ))

    # Source object markers (charges, loops, etc.)
    if source_markers:
        for src in source_markers:
            pos = src["pos"]
            fig.add_trace(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode="markers+text",
                marker=dict(size=10, color=src.get("color", "#FFFFFF"),
                            symbol="circle", line=dict(color="white", width=1)),
                text=[src.get("label", "source")],
                textposition="top center",
                textfont=dict(color="white", size=11),
                name=src.get("label", "Source"),
                showlegend=True,
            ))

    # Origin reference marker
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(size=5, color="white", symbol="cross"),
        name="Origin (0,0,0)",
        showlegend=True,
    ))

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>Cone direction = field direction · Cone color = field magnitude</sup>",
            font=dict(color="white", size=15),
        ),
        paper_bgcolor=PALETTE["background"],
        scene=dict(
            bgcolor=PALETTE["surface"],
            xaxis=dict(title="x (m)", color="white", gridcolor="#333", showbackground=True,
                       backgroundcolor=PALETTE["surface"]),
            yaxis=dict(title="y (m)", color="white", gridcolor="#333", showbackground=True,
                       backgroundcolor=PALETTE["surface"]),
            zaxis=dict(title="z (m)", color="white", gridcolor="#333", showbackground=True,
                       backgroundcolor=PALETTE["surface"]),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        legend=dict(
            font=dict(color="white", size=11),
            bgcolor="rgba(22,27,34,0.85)",
            bordercolor="#444",
            borderwidth=1,
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
        ),
        margin=dict(l=0, r=0, t=70, b=0),
        height=580,
    )
    return fig


def plot_trajectory(
    trajectory,
    B_vec=None,
    E_vec=None,
    charge_sign=1,
    species="Particle",
    title="Particle Trajectory",
):
    """
    3D trajectory plot for a charged particle with full legend, hover data,
    velocity arrow, B-field fan, and reference plane.

    trajectory : dict with 'r' (N,3), 'v' (N,3), 't' (N,)
    B_vec      : (3,) magnetic field vector
    E_vec      : (3,) electric field vector
    species    : display name e.g. "Electron", "Proton"
    """
    r = trajectory["r"]
    t = trajectory["t"]
    v = trajectory["v"]
    speed = np.linalg.norm(v, axis=-1)
    v0 = v[0]
    v0_mag = np.linalg.norm(v0)

    # Bounding span for sizing arrows/planes
    r_range = np.ptp(r, axis=0)
    span = float(np.max(r_range)) if np.max(r_range) > 1e-30 else 1.0
    center = r.mean(axis=0)

    fig = go.Figure()

    # ── 1. Trajectory line, colored by speed ──────────────────────────────
    hover_traj = [
        f"<b>t</b> = {t[i]:.3e} s<br>"
        f"<b>Position</b>: ({r[i,0]:.3e}, {r[i,1]:.3e}, {r[i,2]:.3e}) m<br>"
        f"<b>Speed</b>: {speed[i]:.3e} m/s"
        for i in range(len(r))
    ]
    fig.add_trace(go.Scatter3d(
        x=r[:, 0], y=r[:, 1], z=r[:, 2],
        mode="lines",
        line=dict(
            color=speed,
            colorscale="Viridis",
            width=5,
            colorbar=dict(
                title=dict(text=f"{species} Speed (m/s)", font=dict(color="white", size=11)),
                tickfont=dict(color="white", size=10),
                bgcolor="rgba(22,27,34,0.8)",
                bordercolor="#444",
                x=1.02,
            ),
        ),
        hovertext=hover_traj,
        hoverinfo="text",
        name=f"{species} path  [color = speed]",
        legendgroup="particle",
        legendgrouptitle=dict(text="PARTICLE", font=dict(color="#AAA", size=10)),
    ))

    # ── 2. Start marker ────────────────────────────────────────────────────
    fig.add_trace(go.Scatter3d(
        x=[r[0, 0]], y=[r[0, 1]], z=[r[0, 2]],
        mode="markers+text",
        text=["  t = 0  (start)"],
        textposition="middle right",
        textfont=dict(color="#00FF99", size=11),
        marker=dict(size=9, color="#00FF99", symbol="circle",
                    line=dict(color="white", width=1.5)),
        name="Start position  (t = 0)",
        legendgroup="particle",
        hovertext=f"<b>Start</b><br>r₀ = ({r[0,0]:.3e}, {r[0,1]:.3e}, {r[0,2]:.3e}) m",
        hoverinfo="text",
    ))

    # ── 3. End marker ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter3d(
        x=[r[-1, 0]], y=[r[-1, 1]], z=[r[-1, 2]],
        mode="markers+text",
        text=[f"  t = {t[-1]:.2e} s  (end)"],
        textposition="middle right",
        textfont=dict(color="#FF6B6B", size=11),
        marker=dict(size=9, color="#FF6B6B", symbol="square",
                    line=dict(color="white", width=1.5)),
        name=f"End position  (t = {t[-1]:.2e} s)",
        legendgroup="particle",
        hovertext=f"<b>End</b><br>r = ({r[-1,0]:.3e}, {r[-1,1]:.3e}, {r[-1,2]:.3e}) m",
        hoverinfo="text",
    ))

    # ── 4. Initial velocity arrow (cone) ───────────────────────────────────
    if v0_mag > 1e-30:
        arr_len = span * 0.18
        v0_hat = v0 / v0_mag
        fig.add_trace(go.Cone(
            x=[r[0, 0]], y=[r[0, 1]], z=[r[0, 2]],
            u=[v0_hat[0] * arr_len],
            v=[v0_hat[1] * arr_len],
            w=[v0_hat[2] * arr_len],
            sizemode="absolute",
            sizeref=arr_len * 0.5,
            colorscale=[[0, "#FFD700"], [1, "#FFD700"]],
            showscale=False,
            opacity=1.0,
            hovertext=f"<b>Initial velocity v₀</b><br>|v₀| = {v0_mag:.3e} m/s<br>"
                      f"Direction: ({v0_hat[0]:.2f}, {v0_hat[1]:.2f}, {v0_hat[2]:.2f})",
            hoverinfo="text",
            name=f"Initial velocity v₀  (|v₀| = {v0_mag:.3e} m/s)",
            legendgroup="fields",
            legendgrouptitle=dict(text="FIELDS", font=dict(color="#AAA", size=10)),
            showlegend=True,
        ))

    # ── 5. B field: fan of parallel arrows ────────────────────────────────
    if B_vec is not None:
        B_vec = np.array(B_vec, dtype=float)
        B_mag = np.linalg.norm(B_vec)
        if B_mag > 1e-20:
            B_hat = B_vec / B_mag
            arr_len = span * 0.20

            # Build a 3×3 fan of B arrows in the plane perpendicular to B
            perp1 = np.array([1., 0., 0.]) if abs(B_hat[0]) < 0.9 else np.array([0., 1., 0.])
            perp1 -= np.dot(perp1, B_hat) * B_hat
            perp1 /= np.linalg.norm(perp1)
            perp2 = np.cross(B_hat, perp1)

            offsets = [c1 * perp1 * span * 0.25 + c2 * perp2 * span * 0.25
                       for c1 in [-1, 0, 1] for c2 in [-1, 0, 1]]
            bx = [center[0] + o[0] for o in offsets]
            by = [center[1] + o[1] for o in offsets]
            bz = [center[2] + o[2] for o in offsets]

            fig.add_trace(go.Cone(
                x=bx, y=by, z=bz,
                u=[B_hat[0] * arr_len] * len(offsets),
                v=[B_hat[1] * arr_len] * len(offsets),
                w=[B_hat[2] * arr_len] * len(offsets),
                sizemode="absolute",
                sizeref=arr_len * 0.45,
                colorscale=[[0, PALETTE["B_field"]], [1, PALETTE["B_field"]]],
                showscale=False,
                opacity=0.7,
                hovertext=f"<b>Magnetic Field B</b><br>|B| = {B_mag:.4f} T<br>"
                          f"Direction: ({B_hat[0]:.2f}, {B_hat[1]:.2f}, {B_hat[2]:.2f})",
                hoverinfo="text",
                name=f"Magnetic field B  (|B| = {B_mag:.4f} T)",
                legendgroup="fields",
                showlegend=True,
            ))

    # ── 6. E field arrow (if non-zero) ────────────────────────────────────
    if E_vec is not None:
        E_vec = np.array(E_vec, dtype=float)
        E_mag = np.linalg.norm(E_vec)
        if E_mag > 1e-20:
            E_hat = E_vec / E_mag
            arr_len = span * 0.18
            fig.add_trace(go.Cone(
                x=[center[0] - span * 0.35],
                y=[center[1]],
                z=[center[2]],
                u=[E_hat[0] * arr_len],
                v=[E_hat[1] * arr_len],
                w=[E_hat[2] * arr_len],
                sizemode="absolute",
                sizeref=arr_len * 0.5,
                colorscale=[[0, PALETTE["E_field"]], [1, PALETTE["E_field"]]],
                showscale=False,
                opacity=0.9,
                hovertext=f"<b>Electric Field E</b><br>|E| = {E_mag:.4f} V/m<br>"
                          f"Direction: ({E_hat[0]:.2f}, {E_hat[1]:.2f}, {E_hat[2]:.2f})",
                hoverinfo="text",
                name=f"Electric field E  (|E| = {E_mag:.2e} V/m)",
                legendgroup="fields",
                showlegend=True,
            ))

    # ── 7. Reference plane (XY-plane at z=r[0,2]) ─────────────────────────
    plane_half = span * 0.55
    px = np.array([-plane_half, plane_half, plane_half, -plane_half])
    py = np.array([-plane_half, -plane_half, plane_half, plane_half])
    pz = np.full(4, r[0, 2])
    fig.add_trace(go.Mesh3d(
        x=px + r[0, 0], y=py + r[0, 1], z=pz,
        i=[0], j=[1], k=[2],
        opacity=0.06,
        color="#FFFFFF",
        hoverinfo="skip",
        name="XY reference plane",
        legendgroup="reference",
        legendgrouptitle=dict(text="REFERENCE", font=dict(color="#AAA", size=10)),
        showlegend=True,
    ))
    fig.add_trace(go.Mesh3d(
        x=px + r[0, 0], y=py + r[0, 1], z=pz,
        i=[0], j=[2], k=[3],
        opacity=0.06,
        color="#FFFFFF",
        hoverinfo="skip",
        showlegend=False,
    ))

    # ── 8. Layout ──────────────────────────────────────────────────────────
    charge_label = "−" if charge_sign < 0 else "+"
    subtitle = (
        f"{species}  ({charge_label})  ·  "
        f"Path color = instantaneous speed  ·  "
        f"Hover for values at each point"
    )
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>{subtitle}</sup>",
            font=dict(color="white", size=15),
        ),
        paper_bgcolor=PALETTE["background"],
        scene=dict(
            bgcolor=PALETTE["surface"],
            xaxis=dict(
                title="x  (metres)", color="white",
                gridcolor="#2A2A2A", showbackground=True,
                backgroundcolor=PALETTE["surface"],
            ),
            yaxis=dict(
                title="y  (metres)", color="white",
                gridcolor="#2A2A2A", showbackground=True,
                backgroundcolor=PALETTE["surface"],
            ),
            zaxis=dict(
                title="z  (metres)", color="white",
                gridcolor="#2A2A2A", showbackground=True,
                backgroundcolor=PALETTE["surface"],
            ),
            aspectmode="data",
            camera=dict(eye=dict(x=1.4, y=1.0, z=0.8)),
        ),
        legend=dict(
            font=dict(color="white", size=11),
            bgcolor="rgba(22,27,34,0.9)",
            bordercolor="#555",
            borderwidth=1,
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
            groupclick="toggleitem",
        ),
        margin=dict(l=0, r=0, t=80, b=0),
        height=620,
    )
    return fig


def plot_trajectory_analysis(trajectory, m, species="Particle"):
    """Four-panel analysis: speed, KE, XY path, XZ path — all with units."""
    from physics.magnetodynamics import kinetic_energy

    t = trajectory["t"]
    v = trajectory["v"]
    r = trajectory["r"]
    KE = kinetic_energy(m, v)
    KE_eV = KE / 1.6e-19
    speed = np.linalg.norm(v, axis=-1)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Speed  |v|  vs  Time",
            "Kinetic Energy  vs  Time",
            "Trajectory  —  XY Plane  (top view)",
            "Trajectory  —  XZ Plane  (side view)",
        ],
        horizontal_spacing=0.12,
        vertical_spacing=0.16,
    )

    # Speed vs time
    fig.add_trace(go.Scatter(
        x=t, y=speed, mode="lines",
        line=dict(color=PALETTE["E_field"], width=2),
        name="Speed",
        hovertemplate="t = %{x:.3e} s<br>|v| = %{y:.3e} m/s<extra></extra>",
    ), row=1, col=1)

    # KE vs time (eV)
    fig.add_trace(go.Scatter(
        x=t, y=KE_eV, mode="lines",
        line=dict(color=PALETTE["B_field"], width=2),
        name="KE (eV)",
        hovertemplate="t = %{x:.3e} s<br>KE = %{y:.3e} eV<extra></extra>",
    ), row=1, col=2)

    # XY projection
    fig.add_trace(go.Scatter(
        x=r[:, 0], y=r[:, 1], mode="lines",
        line=dict(color=PALETTE["trajectory"], width=2),
        name="XY path",
        hovertemplate="x = %{x:.3e} m<br>y = %{y:.3e} m<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[r[0, 0]], y=[r[0, 1]], mode="markers",
        marker=dict(size=9, color="#00FF99", symbol="circle"),
        name="Start (XY)", showlegend=False,
        hovertemplate="Start<extra></extra>",
    ), row=2, col=1)

    # XZ projection
    fig.add_trace(go.Scatter(
        x=r[:, 0], y=r[:, 2], mode="lines",
        line=dict(color="#C77DFF", width=2),
        name="XZ path",
        hovertemplate="x = %{x:.3e} m<br>z = %{y:.3e} m<extra></extra>",
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=[r[0, 0]], y=[r[0, 2]], mode="markers",
        marker=dict(size=9, color="#00FF99", symbol="circle"),
        name="Start (XZ)", showlegend=False,
        hovertemplate="Start<extra></extra>",
    ), row=2, col=2)

    # Axis labels
    fig.update_xaxes(title_text="Time  (s)",       row=1, col=1)
    fig.update_yaxes(title_text="Speed  (m/s)",    row=1, col=1)
    fig.update_xaxes(title_text="Time  (s)",       row=1, col=2)
    fig.update_yaxes(title_text="Kinetic Energy  (eV)", row=1, col=2)
    fig.update_xaxes(title_text="x  (m)",          row=2, col=1)
    fig.update_yaxes(title_text="y  (m)",          row=2, col=1, scaleanchor="x3", scaleratio=1)
    fig.update_xaxes(title_text="x  (m)",          row=2, col=2)
    fig.update_yaxes(title_text="z  (m)",          row=2, col=2)

    fig.update_layout(
        paper_bgcolor=PALETTE["background"],
        plot_bgcolor=PALETTE["surface"],
        font=dict(color="white", size=11),
        showlegend=False,
        height=540,
        margin=dict(l=60, r=20, t=70, b=50),
    )
    for axis in fig.layout:
        if "axis" in str(axis):
            try:
                fig.layout[axis].update(gridcolor="#2A2A2A", color="white",
                                        zerolinecolor="#555")
            except Exception:
                pass
    return fig
