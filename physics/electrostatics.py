"""
Electrostatics engine.
Handles E field and potential from point charges, line charges,
ring charges, and electric dipoles via superposition.
"""
import numpy as np

K_E = 8.9875517923e9  # Coulomb constant (N·m²/C²)
EPSILON_0 = 8.854187817e-12


def _safe_norm(r, min_dist=1e-12):
    """Euclidean norm with a floor to avoid division by zero."""
    return np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), min_dist)


def e_field_point_charge(q, pos, grid):
    """
    Electric field at every grid point due to a single point charge.
    q   : charge in Coulombs
    pos : (3,) source position
    grid: (N, 3) evaluation points
    Returns (N, 3) E vectors in V/m.
    """
    r = grid - np.array(pos)
    dist = _safe_norm(r)
    return K_E * q * r / dist ** 3


def potential_point_charge(q, pos, grid):
    """Electric potential (V) at every grid point from a point charge."""
    r = grid - np.array(pos)
    dist = np.linalg.norm(r, axis=-1)
    dist = np.where(dist < 1e-12, 1e-12, dist)
    return K_E * q / dist


def superposition_E(charges, positions, grid):
    """
    Total E field at grid points from multiple point charges.
    charges  : list of floats
    positions: list of (3,) arrays
    grid     : (N, 3)
    Returns (N, 3).
    """
    E = np.zeros_like(grid, dtype=float)
    for q, pos in zip(charges, positions):
        E += e_field_point_charge(q, pos, grid)
    return E


def superposition_V(charges, positions, grid):
    """Total electric potential at grid points."""
    V = np.zeros(len(grid), dtype=float)
    for q, pos in zip(charges, positions):
        V += potential_point_charge(q, pos, grid)
    return V


def e_field_dipole(p_vec, center, grid):
    """
    E field of an ideal electric dipole.
    p_vec  : dipole moment vector (3,) in C·m
    center : (3,) dipole position
    grid   : (N, 3)
    """
    r = grid - np.array(center)
    r_mag = _safe_norm(r)
    r_hat = r / r_mag
    p = np.array(p_vec)
    p_dot_r = np.sum(p * r_hat, axis=-1, keepdims=True)
    E = K_E / r_mag ** 3 * (3 * p_dot_r * r_hat - p)
    return E


def e_field_ring(Q, R, axis, center, grid, n_segments=200):
    """
    E field from a uniformly charged ring via numerical integration.
    Q          : total charge (C)
    R          : ring radius (m)
    axis       : (3,) unit vector along ring axis
    center     : (3,) ring center
    grid       : (N, 3)
    n_segments : integration resolution
    """
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    center = np.array(center, dtype=float)
    dq = Q / n_segments
    phi = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)

    # Build two orthogonal vectors perpendicular to axis
    perp1 = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    perp1 -= np.dot(perp1, axis) * axis
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(axis, perp1)

    E = np.zeros_like(grid, dtype=float)
    for angle in phi:
        src = center + R * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
        E += e_field_point_charge(dq, src, grid)
    return E


def e_field_infinite_line(lam, axis_point, axis_dir, grid):
    """
    E field from an infinite line charge (exact analytic formula).
    lam      : linear charge density (C/m)
    axis_point: any point on the line (3,)
    axis_dir  : direction of the line (3,)
    grid      : (N, 3)
    """
    axis_dir = np.array(axis_dir, dtype=float)
    axis_dir /= np.linalg.norm(axis_dir)
    axis_point = np.array(axis_point, dtype=float)
    r = grid - axis_point
    # Component perpendicular to line
    parallel = np.sum(r * axis_dir, axis=-1, keepdims=True) * axis_dir
    perp = r - parallel
    perp_dist = np.linalg.norm(perp, axis=-1, keepdims=True)
    perp_dist = np.where(perp_dist < 1e-12, 1e-12, perp_dist)
    perp_hat = perp / perp_dist
    magnitude = lam / (2 * np.pi * EPSILON_0 * perp_dist)
    return magnitude * perp_hat


def make_grid_2d(xlim, ylim, z_val=0.0, n=30):
    """Create a 2D evaluation grid (slice at z=z_val)."""
    x = np.linspace(*xlim, n)
    y = np.linspace(*ylim, n)
    XX, YY = np.meshgrid(x, y)
    ZZ = np.full_like(XX, z_val)
    grid = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=-1)
    return grid, XX, YY


def make_grid_3d(xlim, ylim, zlim, n=10):
    """Create a 3D evaluation grid."""
    x = np.linspace(*xlim, n)
    y = np.linspace(*ylim, n)
    z = np.linspace(*zlim, n)
    XX, YY, ZZ = np.meshgrid(x, y, z)
    grid = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=-1)
    return grid, XX, YY, ZZ
