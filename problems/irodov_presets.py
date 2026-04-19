"""
Hardcoded Irodov problem configurations.
Each entry maps to a physics scene description.
Schema keys are consumed by the Streamlit UI and solver pipeline.
"""

PRESETS = {
    "3.3 – Electric field on axis of charged ring": {
        "type": "electrostatics",
        "description": (
            "A thin wire ring of radius R = 10 cm carries a charge q = 0.8 µC. "
            "Find the electric field strength on the axis of the ring as a function "
            "of distance x from its centre."
        ),
        "scene": {
            "objects": [
                {
                    "kind": "ring_charge",
                    "Q": 0.8e-6,
                    "R": 0.10,
                    "axis": [0, 0, 1],
                    "center": [0, 0, 0],
                }
            ],
            "view": {"xlim": [-0.3, 0.3], "ylim": [-0.3, 0.3], "z_slice": 0.0},
            "axis_plot": True,
            "axis_range": [-0.3, 0.3],
        },
    },

    "3.9 – Field of electric dipole": {
        "type": "electrostatics",
        "description": (
            "Find the electric field strength E and potential φ of an electric "
            "dipole with moment p = 3.6 pC·m at a point distant r = 10 cm from "
            "the dipole, in a direction making an angle θ = 30° with the dipole axis."
        ),
        "scene": {
            "objects": [
                {
                    "kind": "dipole",
                    "p_vec": [0, 0, 3.6e-12],
                    "center": [0, 0, 0],
                }
            ],
            "view": {"xlim": [-0.25, 0.25], "ylim": [-0.25, 0.25], "z_slice": 0.0},
        },
    },

    "3.24 – Two equal opposite charges": {
        "type": "electrostatics",
        "description": (
            "Two point charges q = +1 µC and −q = −1 µC are separated by a "
            "distance l = 20 cm. Find the electric field and potential at "
            "points along the perpendicular bisector and the line joining them."
        ),
        "scene": {
            "objects": [
                {"kind": "point_charge", "q": 1e-6, "position": [-0.10, 0, 0]},
                {"kind": "point_charge", "q": -1e-6, "position": [0.10, 0, 0]},
            ],
            "view": {"xlim": [-0.40, 0.40], "ylim": [-0.40, 0.40], "z_slice": 0.0},
        },
    },

    "3.274 – Electron in uniform B field (helix)": {
        "type": "magnetodynamics",
        "description": (
            "An electron enters a uniform magnetic field B = 0.1 T (along z-axis) "
            "with velocity v = 3×10⁶ m/s directed at 45° to B. "
            "Trace the helical trajectory."
        ),
        "scene": {
            "particle": {
                "q": -1.6e-19,
                "m": 9.11e-31,
                "r0": [0, 0, 0],
                "v0": [3e6 * 0.7071, 0.0, 3e6 * 0.7071],
            },
            "fields": {
                "E": [0, 0, 0],
                "B": [0, 0, 0.1],
                "type": "uniform",
            },
            "t_span": [0, 2e-9],
        },
    },

    "3.290 – Proton in crossed E and B fields": {
        "type": "magnetodynamics",
        "description": (
            "A proton moves in crossed electric E = 1 kV/m (x-direction) and "
            "magnetic B = 10 mT (z-direction) fields, starting from rest. "
            "Plot the cycloidal trajectory."
        ),
        "scene": {
            "particle": {
                "q": 1.6e-19,
                "m": 1.67e-27,
                "r0": [0, 0, 0],
                "v0": [0, 0, 0],
            },
            "fields": {
                "E": [1000, 0, 0],
                "B": [0, 0, 0.01],
                "type": "uniform",
            },
            "t_span": [0, 2e-6],
        },
    },

    "3.247 – Magnetic field on axis of circular loop": {
        "type": "magnetostatics",
        "description": (
            "A circular loop of radius R = 5 cm carries a current I = 10 A. "
            "Find the magnetic field on the axis of the loop."
        ),
        "scene": {
            "objects": [
                {
                    "kind": "circular_loop",
                    "I": 10.0,
                    "R": 0.05,
                    "axis": [0, 0, 1],
                    "center": [0, 0, 0],
                }
            ],
            "view": {"xlim": [-0.15, 0.15], "ylim": [-0.15, 0.15], "z_slice": 0.0},
            "axis_plot": True,
            "axis_range": [-0.15, 0.15],
        },
    },

    "3.281 – Solenoid B field": {
        "type": "magnetostatics",
        "description": (
            "A solenoid of length l = 20 cm, radius R = 2 cm with n = 500 turns/m "
            "carries current I = 2 A. Visualize the B field inside and outside."
        ),
        "scene": {
            "objects": [
                {
                    "kind": "solenoid",
                    "I": 2.0,
                    "n_turns": 20,
                    "length": 0.20,
                    "R": 0.02,
                    "axis": [0, 0, 1],
                    "center": [0, 0, 0],
                }
            ],
            "view": {"xlim": [-0.10, 0.10], "ylim": [-0.10, 0.10], "z_slice": 0.0},
        },
    },
}


def get_preset_names():
    return list(PRESETS.keys())


def get_preset(name):
    return PRESETS.get(name)
