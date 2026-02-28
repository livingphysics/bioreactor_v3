"""
EKF Tuning GUI — Replay historical bioreactor CSV data through the Extended
Kalman Filter with adjustable parameters.

Loads a bioreactor CSV, detects pump events, and replays the OD measurements
through the same EKF math used in src/utils.py (Hoffmann et al. 2017).
Log-scale sliders let you sweep R, Q, pump distrust settings, etc. and
instantly see the effect on OD estimate, growth rate, and doubling time.

Usage:
    python hardware_testing/ekf_tuning_gui.py
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Standalone EKF replay — identical math to src/utils.py lines 1095-1211
# ---------------------------------------------------------------------------

def run_ekf_replay(
    times,                  # 1-D array of elapsed_time (seconds)
    measurements,           # 1-D array of raw OD voltage (z_k)
    pump_events,            # 1-D bool array — True on the cycle a pump fires
    R=0.001,
    Q_growth_rate=5e-13,
    initial_growth_rate=1.0,
    initial_P_r=0.0005**2,
    pump_distrust_cycles=10,
    pump_distrust_P_od=None,
):
    """Replay measurements through the EKF. Returns dict of arrays."""

    n = len(times)
    if pump_distrust_P_od is None:
        pump_distrust_P_od = 10.0 * R
    initial_P_od = R

    # Output arrays
    od_est = np.full(n, np.nan)
    growth_rate = np.full(n, np.nan)
    doubling_time_s = np.full(n, np.nan)
    od_std = np.full(n, np.nan)
    r_std = np.full(n, np.nan)
    dt_std = np.full(n, np.nan)

    # --- Initialize on first measurement ---
    z0 = measurements[0]
    x = np.array([z0, initial_growth_rate])
    P = np.array([
        [initial_P_od, 0.0],
        [0.0, initial_P_r],
    ])
    distrust_counter = 0
    last_time = times[0]

    od_est[0] = z0
    growth_rate[0] = initial_growth_rate
    od_std[0] = np.sqrt(initial_P_od)
    r_std[0] = np.sqrt(initial_P_r)
    doubling_time_s[0] = np.inf
    dt_std[0] = np.inf

    I2 = np.eye(2)
    H_vec = np.array([1.0, 0.0])

    for i in range(1, n):
        z_k = measurements[i]
        if np.isnan(z_k):
            # carry forward
            od_est[i] = od_est[i - 1]
            growth_rate[i] = growth_rate[i - 1]
            doubling_time_s[i] = doubling_time_s[i - 1]
            od_std[i] = od_std[i - 1]
            r_std[i] = r_std[i - 1]
            dt_std[i] = dt_std[i - 1]
            continue

        dt_cycle = times[i] - last_time
        if dt_cycle <= 0:
            dt_cycle = 1.0
        last_time = times[i]

        # If a pump fires on this cycle, start the distrust counter
        if pump_events[i]:
            distrust_counter = pump_distrust_cycles

        od_k, r_k = x[0], x[1]

        # --- Predict ---
        x_pred = np.array([od_k * r_k, r_k])
        F = np.array([
            [r_k, od_k],
            [0.0, 1.0],
        ])
        Q_mat = np.array([
            [0.0, 0.0],
            [0.0, Q_growth_rate],
        ])
        P_pred = F @ P @ F.T + Q_mat

        # --- Pump distrust ---
        currently_pumping = pump_events[i]
        if currently_pumping or distrust_counter > 0:
            P_pred[0, 0] = pump_distrust_P_od
            P_pred[0, 1] = 0.0
            P_pred[1, 0] = 0.0
            x_pred[0] = z_k  # reset OD to raw measurement
            if not currently_pumping:
                distrust_counter -= 1

        # --- Update ---
        y = z_k - x_pred[0]
        S = P_pred[0, 0] + R
        K = P_pred[:, 0] / S
        x_updated = x_pred + K * y
        KH = np.outer(K, H_vec)
        P_updated = (I2 - KH) @ P_pred

        # --- Secondary reset: innovation > 5σ ---
        innovation_threshold = 5.0 * np.sqrt(R)
        if abs(z_k - x_pred[0]) > innovation_threshold:
            P_updated[0, 1] = 0.0
            P_updated[1, 0] = 0.0
            P_updated[0, 0] = (x_updated[0] - z_k) ** 2

        # --- Store ---
        x = x_updated
        P = P_updated

        od_est[i] = x[0]
        growth_rate[i] = x[1]
        od_std[i] = np.sqrt(P[0, 0])
        r_std[i] = np.sqrt(P[1, 1])

        # Doubling time
        r_est = x[1]
        if r_est > 1.0 and dt_cycle > 0:
            ln_r = np.log(r_est)
            doubling_time_s[i] = dt_cycle * np.log(2.0) / ln_r
            dt_std[i] = dt_cycle * np.log(2.0) * r_std[i] / (r_est * ln_r ** 2)
        else:
            doubling_time_s[i] = np.inf
            dt_std[i] = np.inf

    return {
        'od_est': od_est,
        'growth_rate': growth_rate,
        'doubling_time_s': doubling_time_s,
        'od_std': od_std,
        'r_std': r_std,
        'dt_std': dt_std,
    }


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_csv(path, od_channel='Eyespy_sct_V'):
    """Load a bioreactor CSV and return times, measurements, pump events, and
    original EKF estimates (if present)."""
    df = pd.read_csv(path)

    times = df['elapsed_time'].values.astype(float)
    measurements = df[od_channel].values.astype(float)

    # Detect pump events: cumulative pump_inflow_time_s increases
    if 'pump_inflow_time_s' in df.columns:
        pump_cum = df['pump_inflow_time_s'].values.astype(float)
        pump_events = np.zeros(len(pump_cum), dtype=bool)
        pump_events[1:] = np.diff(pump_cum) > 0
    else:
        pump_events = np.zeros(len(times), dtype=bool)

    # Original EKF estimates for comparison
    original = {}
    for col in ['ekf_od_est', 'ekf_growth_rate', 'ekf_doubling_time_s']:
        if col in df.columns:
            original[col] = pd.to_numeric(df[col], errors='coerce').values

    # Voltage columns for channel selector
    voltage_cols = [c for c in df.columns if c.endswith('_V')]

    return times, measurements, pump_events, original, voltage_cols, df


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'src', 'bioreactor_data', 'ekf_bioreactor_data.csv',
)


class EKFTuningGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EKF Tuning — Bioreactor Data Replay")
        self.root.geometry("1400x1000")

        self._debounce_id = None

        # Data state
        self.times = None
        self.measurements = None
        self.pump_events = None
        self.original = {}
        self.voltage_cols = []

        # --- Top bar: file selector + OD channel ---
        top = tk.Frame(root)
        top.pack(fill='x', padx=8, pady=4)

        tk.Label(top, text="CSV File:", font=("Arial", 10)).pack(side='left')
        self.file_var = tk.StringVar(value=DEFAULT_CSV)
        tk.Entry(top, textvariable=self.file_var, width=60, font=("Arial", 9)).pack(side='left', padx=4)
        tk.Button(top, text="Browse...", command=self._browse_file).pack(side='left')

        tk.Label(top, text="  OD Channel:", font=("Arial", 10)).pack(side='left', padx=(16, 0))
        self.channel_var = tk.StringVar(value='Eyespy_sct_V')
        self.channel_combo = ttk.Combobox(
            top, textvariable=self.channel_var, state='readonly', width=18,
        )
        self.channel_combo.pack(side='left', padx=4)
        self.channel_combo.bind('<<ComboboxSelected>>', lambda e: self._load_and_run())

        tk.Button(top, text="Load", command=self._load_and_run,
                  bg="#4CAF50", fg="white", font=("Arial", 10, "bold")).pack(side='left', padx=8)

        # --- Parameter controls (left panel) ---
        controls_frame = tk.Frame(root, width=300)
        controls_frame.pack(side='left', fill='y', padx=8, pady=4)
        controls_frame.pack_propagate(False)

        tk.Label(controls_frame, text="EKF Parameters",
                 font=("Arial", 12, "bold")).pack(pady=(4, 8))

        self.sliders = {}
        self._make_log_slider(controls_frame, 'R', -5.0, -0.5, -3.0,
                              "Measurement noise variance")
        self._make_log_slider(controls_frame, 'Q_growth_rate', -16.0, -8.0, -12.3,
                              "Growth rate process noise")
        self._make_linear_slider(controls_frame, 'pump_distrust_cycles', 0, 50, 10,
                                 "Distrust cycles after pump")
        self._make_log_slider(controls_frame, 'pump_distrust_P_od', -5.0, 0.0, -2.0,
                              "P[0,0] during distrust")
        self._make_log_slider(controls_frame, 'initial_P_r', -10.0, -3.0, -6.6,
                              "Initial growth rate covariance")

        tk.Frame(controls_frame, height=2, bg="gray").pack(fill='x', pady=8)

        tk.Button(controls_frame, text="Reset Defaults", command=self._reset_defaults,
                  font=("Arial", 10)).pack(pady=4)

        # Current values display
        self.value_text = tk.Text(controls_frame, height=8, width=34, font=("Courier", 9),
                                  state='disabled', bg='#f0f0f0')
        self.value_text.pack(pady=8, padx=4)

        # --- Matplotlib figure (right panel) ---
        fig_frame = tk.Frame(root)
        fig_frame.pack(side='right', fill='both', expand=True, padx=4, pady=4)

        self.fig = Figure(figsize=(10, 9), dpi=100)
        self.ax_od, self.ax_r, self.ax_dph, self.ax_dt = self.fig.subplots(4, 1, sharex=True)
        self.fig.subplots_adjust(hspace=0.15, top=0.97, bottom=0.05, left=0.08, right=0.97)

        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, fig_frame)
        toolbar.update()
        toolbar.pack(fill='x')

        # Load default file
        self._load_and_run()

    # --- Slider helpers ---

    def _make_log_slider(self, parent, name, lo, hi, default, description):
        frame = tk.Frame(parent)
        frame.pack(fill='x', padx=4, pady=2)
        tk.Label(frame, text=name, font=("Arial", 10, "bold"), anchor='w').pack(fill='x')
        tk.Label(frame, text=description, font=("Arial", 8), fg="gray", anchor='w').pack(fill='x')

        var = tk.DoubleVar(value=default)
        slider = tk.Scale(
            frame, from_=lo, to=hi, resolution=0.1, orient='horizontal',
            variable=var, showvalue=False, length=250,
            command=lambda v: self._on_slider_change(),
        )
        slider.pack(fill='x')

        val_label = tk.Label(frame, text=f"{10**default:.2e}", font=("Courier", 9))
        val_label.pack()

        self.sliders[name] = {'var': var, 'label': val_label, 'type': 'log',
                              'default': default, 'slider': slider}

    def _make_linear_slider(self, parent, name, lo, hi, default, description):
        frame = tk.Frame(parent)
        frame.pack(fill='x', padx=4, pady=2)
        tk.Label(frame, text=name, font=("Arial", 10, "bold"), anchor='w').pack(fill='x')
        tk.Label(frame, text=description, font=("Arial", 8), fg="gray", anchor='w').pack(fill='x')

        var = tk.IntVar(value=default)
        slider = tk.Scale(
            frame, from_=lo, to=hi, orient='horizontal',
            variable=var, showvalue=False, length=250,
            command=lambda v: self._on_slider_change(),
        )
        slider.pack(fill='x')

        val_label = tk.Label(frame, text=str(default), font=("Courier", 9))
        val_label.pack()

        self.sliders[name] = {'var': var, 'label': val_label, 'type': 'linear',
                              'default': default, 'slider': slider}

    def _get_param(self, name):
        info = self.sliders[name]
        raw = info['var'].get()
        if info['type'] == 'log':
            return 10 ** raw
        return int(raw)

    def _update_value_labels(self):
        lines = []
        for name, info in self.sliders.items():
            val = self._get_param(name)
            if info['type'] == 'log':
                text = f"{val:.3e}"
            else:
                text = str(val)
            info['label'].config(text=text)
            lines.append(f"{name:>24s} = {text}")

        self.value_text.config(state='normal')
        self.value_text.delete('1.0', 'end')
        self.value_text.insert('1.0', '\n'.join(lines))
        self.value_text.config(state='disabled')

    def _reset_defaults(self):
        for info in self.sliders.values():
            info['var'].set(info['default'])
        self._run_and_plot()

    # --- File loading ---

    def _browse_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=os.path.dirname(self.file_var.get()),
        )
        if path:
            self.file_var.set(path)
            self._load_and_run()

    def _load_and_run(self):
        path = self.file_var.get()
        if not os.path.isfile(path):
            return

        channel = self.channel_var.get()
        self.times, self.measurements, self.pump_events, self.original, \
            self.voltage_cols, _ = load_csv(path, od_channel=channel)

        # Update channel dropdown
        self.channel_combo['values'] = self.voltage_cols
        if channel not in self.voltage_cols and self.voltage_cols:
            self.channel_var.set(self.voltage_cols[0])
            self.times, self.measurements, self.pump_events, self.original, \
                self.voltage_cols, _ = load_csv(path, od_channel=self.voltage_cols[0])

        self._run_and_plot()

    # --- Debounced slider callback ---

    def _on_slider_change(self):
        if self._debounce_id is not None:
            self.root.after_cancel(self._debounce_id)
        self._debounce_id = self.root.after(200, self._run_and_plot)

    # --- Core: run EKF replay and update plots ---

    def _run_and_plot(self):
        self._debounce_id = None
        self._update_value_labels()

        if self.times is None:
            return

        result = run_ekf_replay(
            self.times, self.measurements, self.pump_events,
            R=self._get_param('R'),
            Q_growth_rate=self._get_param('Q_growth_rate'),
            pump_distrust_cycles=self._get_param('pump_distrust_cycles'),
            pump_distrust_P_od=self._get_param('pump_distrust_P_od'),
            initial_P_r=self._get_param('initial_P_r'),
        )

        t_min = self.times / 60.0  # convert to minutes

        # Pump event x-positions
        pump_times = t_min[self.pump_events]

        # --- OD plot ---
        ax = self.ax_od
        ax.clear()
        ax.plot(t_min, self.measurements, '.', color='#cccccc', markersize=2, label='Raw OD', zorder=1)
        ax.plot(t_min, result['od_est'], '-', color='#2196F3', linewidth=1.2, label='EKF OD est', zorder=3)
        # ±1σ band
        upper = result['od_est'] + result['od_std']
        lower = result['od_est'] - result['od_std']
        ax.fill_between(t_min, lower, upper, color='#2196F3', alpha=0.15, zorder=2)
        # Original estimate
        if 'ekf_od_est' in self.original:
            ax.plot(t_min, self.original['ekf_od_est'], '--', color='#999999',
                    linewidth=0.8, label='Original est', zorder=2)
        for pt in pump_times:
            ax.axvline(pt, color='#FF5722', alpha=0.3, linewidth=0.8)
        ax.set_ylabel('OD (V)')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- Growth rate plot ---
        ax = self.ax_r
        ax.clear()
        ax.plot(t_min, result['growth_rate'], '-', color='#4CAF50', linewidth=1.2, label='Growth rate (r)')
        upper = result['growth_rate'] + result['r_std']
        lower = result['growth_rate'] - result['r_std']
        ax.fill_between(t_min, lower, upper, color='#4CAF50', alpha=0.15)
        if 'ekf_growth_rate' in self.original:
            ax.plot(t_min, self.original['ekf_growth_rate'], '--', color='#999999',
                    linewidth=0.8, label='Original')
        ax.axhline(1.0, color='black', linewidth=0.5, linestyle=':')
        for pt in pump_times:
            ax.axvline(pt, color='#FF5722', alpha=0.3, linewidth=0.8)
        ax.set_ylabel('Growth rate (r)')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- Doublings per hour plot ---
        ax = self.ax_dph
        ax.clear()
        dt_s = result['doubling_time_s']
        dt_std_s = result['dt_std']
        # doublings/hr = 3600 / doubling_time_s
        dph = np.where((dt_s > 0) & np.isfinite(dt_s), 3600.0 / dt_s, np.nan)
        # σ_dph = 3600 * σ_dt / dt^2  (error propagation of 1/x)
        dph_std = np.where((dt_s > 0) & np.isfinite(dt_s) & np.isfinite(dt_std_s),
                           3600.0 * dt_std_s / (dt_s ** 2), np.nan)
        ax.plot(t_min, dph, '-', color='#FF9800', linewidth=1.2, label='Doublings/hr')
        dph_upper = np.where(np.isfinite(dph + dph_std), dph + dph_std, np.nan)
        dph_lower = np.where(np.isfinite(dph - dph_std), np.maximum(dph - dph_std, 0), np.nan)
        ax.fill_between(t_min, dph_lower, dph_upper, color='#FF9800', alpha=0.15)
        for pt in pump_times:
            ax.axvline(pt, color='#FF5722', alpha=0.3, linewidth=0.8)
        ax.set_ylabel('Doublings/hr')
        # Auto-scale y
        finite_dph = np.concatenate([dph, dph_upper])
        finite_dph = finite_dph[np.isfinite(finite_dph)]
        if len(finite_dph) > 0:
            p5, p95 = np.percentile(finite_dph, [5, 95])
            margin = (p95 - p5) * 0.15
            ax.set_ylim(max(0, p5 - margin), p95 + margin)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- Doubling time plot ---
        ax = self.ax_dt
        ax.clear()
        dt_min = result['doubling_time_s'] / 60.0
        # Replace inf with NaN so matplotlib skips gaps instead of clipping
        dt_plot = np.where(np.isfinite(dt_min), dt_min, np.nan)
        ax.plot(t_min, dt_plot, '-', color='#9C27B0', linewidth=1.2, label='Doubling time')
        # ±1σ band
        dt_std_min = result['dt_std'] / 60.0
        dt_upper = np.where(np.isfinite(dt_min + dt_std_min), dt_min + dt_std_min, np.nan)
        dt_lower = np.where(np.isfinite(dt_min - dt_std_min), np.maximum(dt_min - dt_std_min, 0), np.nan)
        ax.fill_between(t_min, dt_lower, dt_upper, color='#9C27B0', alpha=0.15)
        if 'ekf_doubling_time_s' in self.original:
            orig_dt = self.original['ekf_doubling_time_s'] / 60.0
            orig_plot = np.where(np.isfinite(orig_dt), orig_dt, np.nan)
            ax.plot(t_min, orig_plot, '--', color='#999999', linewidth=0.8, label='Original')
        for pt in pump_times:
            ax.axvline(pt, color='#FF5722', alpha=0.3, linewidth=0.8)
        ax.set_ylabel('Doubling time (min)')
        ax.set_xlabel('Elapsed time (min)')
        # Auto-scale y to finite data (including band) with some padding
        all_dt = np.concatenate([dt_plot, dt_upper])
        finite_vals = all_dt[np.isfinite(all_dt)]
        if len(finite_vals) > 0:
            p5, p95 = np.percentile(finite_vals, [5, 95])
            margin = (p95 - p5) * 0.15
            ax.set_ylim(max(0, p5 - margin), p95 + margin)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

        self.canvas.draw_idle()


def main():
    root = tk.Tk()
    EKFTuningGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
