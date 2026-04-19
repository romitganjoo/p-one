"""
Magnetodynamics engine.
Handles B fields (Biot-Savart), Lorentz force, and particle trajectory solving.
"""
import numpy as np
from scipy.integrate import solve_ivp

MU_0 = 4 * np.pi * 1e-7  # permeability of free space


# ─── Magnetic field sources ────────────────────────────────────────────────

def b_field_infinite_wire(I, axis_point, axis_dir, grid):
    """
    B field from an infinite straight wire (exact: Ampere's law).
    I         : current in Amperes
    axis_point: any point on the wire (3,)
    axis_dir  : direction of current flow (3,)
    grid      : (N, 3)
    """
    axis_dir = np.array(axis_dir, dtype=float)
    axis_dir /= np.linalg.norm(axis_dir)
    axis_point = np.array(axis_point, dtype=float)
    r = grid - axis_point
    parallel = np.sum(r * axis_dir, axis=-1, keepdims=True) * axis_dir
    perp = r - parallel
    perp_dist = np.linalg.norm(perp, axis=-1, keepdims=True)
    perp_dist = np.where(perp_dist < 1e-12, 1e-12, perp_dist)
    perp_hat = perp / perp_dist
    B_hat = np.cross(axis_dir, perp_hat)
    magnitude = MU_0 * I / (2 * np.pi * perp_dist)
    return magnitude * B_hat


def b_field_circular_loop(I, R, axis, center, grid, n_segments=300):
    """
    B field from a circular current loop (Biot-Savart, numerical).
    I          : current (A)
    R          : loop radius (m)
    axis       : (3,) unit vector along loop normal
    center     : (3,) loop center
    grid       : (N, 3)
    """
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    center = np.array(center, dtype=float)

    perp1 = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    perp1 -= np.dot(perp1, axis) * axis
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(axis, perp1)

    phi = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    dphi = 2 * np.pi / n_segments

    B = np.zeros_like(grid, dtype=float)
    for angle in phi:
        # Position on loop
        src = center + R * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
        # dl element (tangent to loop × dphi × R)
        dl = R * dphi * (-np.sin(angle) * perp1 + np.cos(angle) * perp2)
        r = grid - src
        r_mag = np.linalg.norm(r, axis=-1, keepdims=True)
        r_mag = np.where(r_mag < 1e-12, 1e-12, r_mag)
        dB = (MU_0 * I / (4 * np.pi)) * np.cross(dl, r) / r_mag ** 3
        B += dB
    return B


def b_field_solenoid(I, n_turns, length, R, axis, center, grid):
    """
    B field from a finite solenoid (superposition of circular loops).
    n_turns: integer number of turns
    length : solenoid length (m)
    """
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    center = np.array(center, dtype=float)
    positions = np.linspace(-length / 2, length / 2, n_turns)
    B = np.zeros_like(grid, dtype=float)
    for z in positions:
        loop_center = center + z * axis
        B += b_field_circular_loop(I, R, axis, loop_center, grid, n_segments=100)
    return B


def b_field_uniform(B_vec, grid):
    """Uniform magnetic field everywhere."""
    return np.tile(np.array(B_vec, dtype=float), (len(grid), 1))


# ─── Particle dynamics ─────────────────────────────────────────────────────

def lorentz_force(q, v, E, B):
    """F = q(E + v × B). All vectors are (3,)."""
    return q * (E + np.cross(v, B))


def _equations_of_motion(t, state, q, m, E_func, B_func):
    """
    ODE system for a charged particle.
    state = [x, y, z, vx, vy, vz]
    E_func(r) -> (3,)  electric field at position r
    B_func(r) -> (3,)  magnetic field at position r
    """
    r = state[:3]
    v = state[3:]
    E = E_func(r)
    B = B_func(r)
    F = lorentz_force(q, v, E, B)
    a = F / m
    return np.concatenate([v, a])


def solve_trajectory(q, m, r0, v0, E_func, B_func, t_span, n_steps=2000):
    """
    Integrate particle motion under Lorentz force.
    Returns dict with 't', 'r' (N,3), 'v' (N,3) arrays.
    """
    state0 = np.concatenate([r0, v0])
    t_eval = np.linspace(t_span[0], t_span[1], n_steps)
    sol = solve_ivp(
        _equations_of_motion,
        t_span,
        state0,
        t_eval=t_eval,
        args=(q, m, E_func, B_func),
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
        dense_output=False,
    )
    return {
        "t": sol.t,
        "r": sol.y[:3].T,
        "v": sol.y[3:].T,
        "success": sol.success,
        "message": sol.message,
    }


def kinetic_energy(m, v_array):
    """Kinetic energy at each time step. v_array: (N, 3)."""
    return 0.5 * m * np.sum(v_array ** 2, axis=-1)


def cyclotron_radius(m, v_perp, q, B_mag):
    """r = mv⊥/(|q|B)"""
    return m * v_perp / (abs(q) * B_mag)


def cyclotron_frequency(q, m, B_mag):
    """ω = |q|B/m  (rad/s)"""
    return abs(q) * B_mag / m
