"""
Plot EKF replay of bioreactor_data.csv: OD with EKF estimate, doubling time,
and temperature. Time range: 0–9 hours.

Usage:
    python analysis/plot_ekf_replay.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the EKF replay and CSV loader from the tuning GUI
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hardware_testing.ekf_tuning_gui import load_csv, run_ekf_replay

# --- Load data ---
CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'src', 'bioreactor_data', 'bioreactor_data.csv',
)

times, measurements, pump_events, original, voltage_cols, df = load_csv(
    CSV_PATH, od_channel='Eyespy_sct_V')

result = run_ekf_replay(times, measurements, pump_events, Q_growth_rate=5e-12)

# --- Time range: 0–9 hours ---
t_hours = times / 3600.0
mask = t_hours <= 9.0
t_h = t_hours[mask]

raw_od = measurements[mask]
ekf_od = result['od_est'][mask]
od_std = result['od_std'][mask]

dt_min = result['doubling_time_s'][mask] / 60.0
dt_plot = np.where(np.isfinite(dt_min), dt_min, np.nan)
dt_std_min = result['dt_std'][mask] / 60.0

temp = df['temperature_C'].values.astype(float)[mask]

pump_times = t_h[pump_events[mask]]

# --- Plot ---
fig, (ax_od, ax_dt, ax_temp) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.subplots_adjust(hspace=0.12, top=0.95, bottom=0.08, left=0.10, right=0.97)

# OD plot
ax_od.plot(t_h, raw_od, '.', color='#cccccc', markersize=2, label='Raw OD', zorder=1)
ax_od.plot(t_h, ekf_od, '-', color='#2196F3', linewidth=1.2, label='EKF OD est', zorder=3)
ax_od.fill_between(t_h, ekf_od - od_std, ekf_od + od_std,
                    color='#2196F3', alpha=0.15, zorder=2)
for pt in pump_times:
    ax_od.axvline(pt, color='#FF5722', alpha=0.3, linewidth=0.8)
ax_od.set_ylabel('OD (V)')
ax_od.legend(loc='upper left', fontsize=9)
ax_od.grid(True, alpha=0.3)
ax_od.set_title('EKF Replay — bioreactor_data.csv (0–9 hrs)')

# Doubling time plot
ax_dt.plot(t_h, dt_plot, '-', color='#9C27B0', linewidth=1.2, label='Doubling time')
with np.errstate(invalid='ignore'):
    dt_upper = np.where(np.isfinite(dt_plot) & np.isfinite(dt_std_min),
                        dt_plot + dt_std_min, np.nan)
    dt_lower = np.where(np.isfinite(dt_plot) & np.isfinite(dt_std_min),
                        np.maximum(dt_plot - dt_std_min, 0), np.nan)
ax_dt.fill_between(t_h, dt_lower, dt_upper, color='#9C27B0', alpha=0.15)
for pt in pump_times:
    ax_dt.axvline(pt, color='#FF5722', alpha=0.3, linewidth=0.8)
ax_dt.set_ylabel('Doubling time (min)')
# Auto-scale y
finite_vals = dt_plot[np.isfinite(dt_plot)]
if len(finite_vals) > 0:
    p5, p95 = np.percentile(finite_vals, [5, 95])
    margin = (p95 - p5) * 0.15
    ax_dt.set_ylim(max(0, p5 - margin), p95 + margin)
ax_dt.legend(loc='upper left', fontsize=9)
ax_dt.grid(True, alpha=0.3)

# Temperature plot
ax_temp.plot(t_h, temp, '-', color='#E65100', linewidth=1.2, label='Temperature')
for pt in pump_times:
    ax_temp.axvline(pt, color='#FF5722', alpha=0.3, linewidth=0.8)
ax_temp.set_ylabel('Temperature (°C)')
ax_temp.set_xlabel('Elapsed time (hours)')
ax_temp.legend(loc='upper left', fontsize=9)
ax_temp.grid(True, alpha=0.3)

plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
            'ekf_replay.png'), dpi=150)
plt.show()
