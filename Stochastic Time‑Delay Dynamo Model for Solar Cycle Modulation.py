# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 09:47:56 2026

@author: bijet
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
from scipy.special import erf

# ============================================================================
# Parameters
# ============================================================================
tau = 10.0
omega_over_L = 1.0
alpha_BL = 0.3
T_l = 5.0
T_0 = 5.0
B_lo = 0.2
B_up = 1.0
d_lo = 0.1
d_up = 0.1
sigma_noise = 0.1          # increased for clearer modulation
dt = 0.01
T_max = 50000.0            # long enough to resolve low frequencies
N_steps = int(T_max / dt)

# ----------------------------------------------------------------------------
def f_quench(B):
    """Nonlinear quenching function from the paper."""
    x_lo = (B**2 - B_lo**2) / (d_lo**2)
    x_up = (B**2 - B_up**2) / (d_up**2)
    x_lo = np.clip(x_lo, -20, 20)
    x_up = np.clip(x_up, -20, 20)
    return (1 + erf(x_lo)) * (1 - erf(x_up))

# ----------------------------------------------------------------------------
def solve_dde(noise_on=True):
    """Integrate the DDE system with optional white noise."""
    t = np.zeros(N_steps + 1)
    A = np.zeros(N_steps + 1)
    B = np.zeros(N_steps + 1)
    t[0] = 0.0
    A[0] = 0.1
    B[0] = 0.1

    for i in range(N_steps):
        t[i+1] = t[i] + dt

        # B(t - T_l)
        t_l = t[i+1] - T_l
        if t_l <= 0:
            B_lag = B[0]
        else:
            idx = int(np.floor(t_l / dt))
            idx = min(idx, i)
            if idx >= 0:
                t1 = t[idx]
                t2 = t[idx+1]
                B1 = B[idx]
                B2 = B[idx+1]
                if t2 != t1:
                    B_lag = B1 + (B2 - B1) * (t_l - t1) / (t2 - t1)
                else:
                    B_lag = B1
            else:
                B_lag = B[0]

        # A(t - T_0)
        t_0 = t[i+1] - T_0
        if t_0 <= 0:
            A_lag = A[0]
        else:
            idx = int(np.floor(t_0 / dt))
            idx = min(idx, i)
            if idx >= 0:
                t1 = t[idx]
                t2 = t[idx+1]
                A1 = A[idx]
                A2 = A[idx+1]
                if t2 != t1:
                    A_lag = A1 + (A2 - A1) * (t_0 - t1) / (t2 - t1)
                else:
                    A_lag = A1
            else:
                A_lag = A[0]

        dA = alpha_BL * f_quench(B_lag) * B_lag - A[i] / tau
        dB = omega_over_L * A_lag - B[i] / tau

        if noise_on:
            dA += sigma_noise * np.sqrt(dt) * np.random.randn()

        A[i+1] = A[i] + dA * dt
        B[i+1] = B[i] + dB * dt

        # Prevent runaway
        if abs(A[i+1]) > 10:
            A[i+1] = np.sign(A[i+1]) * 10
        if abs(B[i+1]) > 10:
            B[i+1] = np.sign(B[i+1]) * 10

    return t, B          # we only need B for analysis, A is not used later

# ----------------------------------------------------------------------------
def ar1_surrogates(B, t, dt, n_surr=200):
    """Generate AR(1) surrogates and return 95% power spectrum threshold."""
    # Remove initial transient
    transient = 5000
    idx_start = int(transient / dt)
    B_trim = B[idx_start:]

    # Fit AR(1) model
    a = np.corrcoef(B_trim[:-1], B_trim[1:])[0, 1]
    resid = B_trim[1:] - a * B_trim[:-1]
    var_e = np.var(resid)

    fs = 1.0 / dt
    P_surr = []
    for _ in range(n_surr):
        x = np.zeros(len(B_trim))
        x[0] = B_trim[0]
        for i in range(1, len(x)):
            x[i] = a * x[i-1] + np.sqrt(var_e) * np.random.randn()
        f, P = welch(x, fs=fs, nperseg=min(4096, len(x)//4))
        P_surr.append(P)
    P_surr = np.array(P_surr)
    P_95 = np.percentile(P_surr, 95, axis=0)
    return f, P_95

# ----------------------------------------------------------------------------
def power_spectrum(B, t, dt):
    """Compute power spectral density of B after initial transient."""
    transient = 5000
    idx_start = int(transient / dt)
    B_trim = B[idx_start:]
    fs = 1.0 / dt
    f, P = welch(B_trim, fs=fs, nperseg=min(4096, len(B_trim)//4))
    return f, P

# ----------------------------------------------------------------------------
# Run simulations
# ----------------------------------------------------------------------------
print("Running deterministic simulation...")
np.random.seed(42)
t_det, B_det = solve_dde(noise_on=False)

print("Running stochastic simulation...")
t_stoc, B_stoc = solve_dde(noise_on=True)

# ----------------------------------------------------------------------------
# Compute power spectra
# ----------------------------------------------------------------------------
f_det, P_det = power_spectrum(B_det, t_det, dt)
f_stoc, P_stoc = power_spectrum(B_stoc, t_stoc, dt)

print("Computing AR(1) surrogate significance...")
f_sig, P_95 = ar1_surrogates(B_stoc, t_stoc, dt, n_surr=200)

# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.loglog(f_det, P_det, 'k-', lw=1.5, label='Deterministic')
plt.xlabel('Frequency [years⁻¹]')
plt.ylabel('Power Spectral Density')
plt.title('Deterministic: No supradecadal peaks')
plt.grid(True, alpha=0.3)
plt.xlim(1e-3, 1e0)
plt.legend()

plt.subplot(1, 2, 2)
plt.loglog(f_stoc, P_stoc, 'b-', lw=1.5, label='Stochastic')
plt.loglog(f_sig, P_95, 'r--', lw=1.5, label='95% AR(1) surrogate')
plt.xlabel('Frequency [years⁻¹]')
plt.ylabel('Power Spectral Density')
plt.title('Stochastic: Broad peaks in supradecadal band')
plt.grid(True, alpha=0.3)
plt.xlim(1e-3, 1e0)
plt.legend()

plt.tight_layout()
plt.savefig('dynamo_spectra_ar1.png', dpi=150)
plt.show()

# ----------------------------------------------------------------------------
# Print cycle statistics
# ----------------------------------------------------------------------------
def cycle_periods(B, t, dt):
    transient = 5000
    idx_start = int(transient / dt)
    B_smooth = np.convolve(B[idx_start:], np.ones(10)/10, mode='same')
    peaks, _ = find_peaks(B_smooth, distance=100/dt)
    peak_times = t[idx_start:][peaks]
    return np.diff(peak_times)

if len(t_det) > 0:
    periods_det = cycle_periods(B_det, t_det, dt)
    if len(periods_det) > 0:
        print(f"Deterministic cycle period: {np.mean(periods_det):.1f} ± {np.std(periods_det):.1f} years")
    periods_stoc = cycle_periods(B_stoc, t_stoc, dt)
    if len(periods_stoc) > 0:
        print(f"Stochastic cycle period: {np.mean(periods_stoc):.1f} ± {np.std(periods_stoc):.1f} years")