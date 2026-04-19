"""
AI Parser using Google Gemini.
Converts Irodov problem text into a structured physics scene JSON.
"""
import os
import json
import re
import google.generativeai as genai

SYSTEM_PROMPT = """You are a physics problem parser specialized in electrodynamics and magnetodynamics.

Given a physics problem, extract all physical entities and return ONLY a valid JSON object with this schema:

{
  "type": "<electrostatics|magnetostatics|magnetodynamics>",
  "description": "<brief summary>",
  "scene": {
    "objects": [
      {
        "kind": "<point_charge|ring_charge|line_charge|dipole|circular_loop|solenoid|infinite_wire>",
        "q_or_I": <numeric value in SI>,
        ... (other relevant numeric fields like R, length, axis, center, p_vec)
      }
    ],
    "particle": {
      "q": <charge in C>,
      "m": <mass in kg>,
      "r0": [x, y, z],
      "v0": [vx, vy, vz]
    },
    "fields": {
      "E": [Ex, Ey, Ez],
      "B": [Bx, By, Bz],
      "type": "uniform"
    },
    "t_span": [t_start, t_end],
    "view": {
      "xlim": [xmin, xmax],
      "ylim": [ymin, ymax],
      "z_slice": 0.0
    }
  }
}

Rules:
- Use SI units exclusively (meters, Coulombs, Teslas, Volts/meter, seconds).
- Omit sections that are not relevant (e.g., omit "particle" for static problems).
- If the problem involves a moving charged particle in fields, include "particle" and "fields".
- Set "t_span" to a physically reasonable simulation window (enough cycles or traversals).
- For "view", choose limits that encompass all objects with some margin.
- If a value is not given, use a physically reasonable default.
- Output ONLY the JSON, no prose, no markdown fences.
"""

COMMON_CONSTANTS = """
Useful constants (already in SI):
- e = 1.6e-19 C (elementary charge)
- m_e = 9.11e-31 kg (electron mass)
- m_p = 1.67e-27 kg (proton mass)
- epsilon_0 = 8.85e-12 C²/(N·m²)
- mu_0 = 1.26e-6 T·m/A
"""


def parse_problem(problem_text: str, api_key: str) -> dict:
    """
    Send problem text to Gemini and return a parsed physics scene dict.
    Raises ValueError if parsing fails.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=SYSTEM_PROMPT + COMMON_CONSTANTS,
    )

    response = model.generate_content(problem_text)
    raw = response.text.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        scene = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gemini returned invalid JSON:\n{raw}\n\nError: {exc}") from exc

    return scene


def fallback_scene_from_text(problem_text: str) -> dict:
    """
    Minimal rule-based fallback if AI is unavailable.
    Detects keywords to build a basic scene config.
    """
    text = problem_text.lower()
    scene = {"type": "electrostatics", "description": problem_text[:120], "scene": {}}

    if any(k in text for k in ["magnetic", "solenoid", "ampere", "current", "tesla"]):
        scene["type"] = "magnetostatics"
    if any(k in text for k in ["velocity", "moving charge", "lorentz", "trajectory", "helix"]):
        scene["type"] = "magnetodynamics"
        scene["scene"]["particle"] = {
            "q": -1.6e-19, "m": 9.11e-31,
            "r0": [0, 0, 0], "v0": [1e6, 0, 1e6],
        }
        scene["scene"]["fields"] = {"E": [0, 0, 0], "B": [0, 0, 0.1], "type": "uniform"}
        scene["scene"]["t_span"] = [0, 5e-9]
    else:
        scene["scene"]["objects"] = [
            {"kind": "point_charge", "q": 1e-6, "position": [-0.1, 0, 0]},
            {"kind": "point_charge", "q": -1e-6, "position": [0.1, 0, 0]},
        ]
        scene["scene"]["view"] = {"xlim": [-0.3, 0.3], "ylim": [-0.3, 0.3], "z_slice": 0.0}

    return scene
