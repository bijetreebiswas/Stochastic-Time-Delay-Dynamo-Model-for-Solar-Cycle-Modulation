# Stochastic Time‑Delay Dynamo Model for Solar Cycle Modulation

This repository contains a Python implementation of a low‑order, stochastically forced time‑delay dynamo model used to study the origin of long‑term (supradecadal) modulation in the solar magnetic activity cycle. The code reproduces the main result of the paper *“On the Origin of Long‑term Modulation in the Sun’s Magnetic Activity Cycle”* (Saha et al., ApJL 2025) – namely, that **stochastic fluctuations are necessary to generate statistically significant power in the supradecadal band** (periods > 50 years), while nonlinearity alone is insufficient.

## Features

- **Time‑delay dynamo equations** (poloidal field `A` and toroidal field `B_φ`) with a smooth, erf‑based quenching function.
- **Additive white noise** in the poloidal field equation to mimic the scatter in sunspot tilt angles.
- **AR(1) surrogate significance testing** to establish whether observed low‑frequency power is real or merely due to red noise.
- **Power spectrum computation** (Welch’s method) and visualisation.
- **Cycle period statistics** (mean and standard deviation) for both deterministic and stochastic runs.

## Requirements

- Python 3.8 or higher
- `numpy`
- `scipy`
- `matplotlib`
- `tqdm` (optional, for progress bars)

## Results
<img width="1713" height="705" alt="image" src="https://github.com/user-attachments/assets/a58d9880-3672-4955-b0d6-0c061cdfe73f" />
Deterministic cycle period: 152.5 ± 0.9 years
Stochastic cycle period: 152.4 ± 1.1 years
