# Irodov Electrodynamics Simulator

A Streamlit-based physics simulator for visualizing electrostatics, magnetostatics, and magnetodynamics problems from **I.E. Irodov's Problems in General Physics**.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Electrostatics**: E field & equipotential maps for point charges, rings, dipoles, line charges
- **Magnetostatics**: B field from wires, loops, and solenoids (Biot-Savart)
- **Magnetodynamics**: Particle trajectory under Lorentz force (helical, cycloidal motion)
- **AI Parsing**: Paste any Irodov problem text and let Gemini extract the physics scene
- **Interactive 3D**: Rotate, zoom, hover for values at each point

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up Gemini API key (optional, for AI parsing)
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Run the app
streamlit run app.py --server.port 8501
```

## Project Structure

```
p-one/
├── app.py                        # Streamlit UI (main entry point)
├── physics/
│   ├── electrostatics.py         # E fields, potential, ring/dipole/line charge
│   └── magnetodynamics.py        # B fields (Biot-Savart), Lorentz force, RK45
├── visualizations/
│   └── field_plots.py            # Plotly 2D/3D field and trajectory plots
├── problems/
│   └── irodov_presets.py         # 7 preset Irodov problems
├── ai_parser.py                  # Gemini parser: problem text → scene JSON
└── requirements.txt
```

## Preset Problems

| Problem | Type | Description |
|---------|------|-------------|
| 3.3 | Electrostatics | E field on axis of charged ring |
| 3.9 | Electrostatics | Field of electric dipole |
| 3.24 | Electrostatics | Two equal opposite charges |
| 3.247 | Magnetostatics | B field on axis of circular loop |
| 3.281 | Magnetostatics | Solenoid B field |
| 3.274 | Magnetodynamics | Electron helix in uniform B field |
| 3.290 | Magnetodynamics | Proton cycloid in crossed E and B fields |

## Tech Stack

- **Physics Engine**: NumPy, SciPy (RK45 ODE solver)
- **Visualization**: Plotly (interactive 3D)
- **UI**: Streamlit
- **AI**: Google Gemini API

## License

MIT
