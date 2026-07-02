"""
Heater Range Sweep

Automated sweep that steps the peltier through its full range:
    h100, h90, h80, ..., h10, 0, c10, c20, ..., c100   (21 steps)

Each step holds for a fixed duration (default 10 minutes) so the temperature
can stabilise. Temperature is plotted live with the current setting overlaid
on the plot and shown in a status read-out.

Use "Start Sweep" to begin and "Abort" to stop early; closing the window
also stops the peltier cleanly.
"""

import csv
import os
import sys
import time
from collections import deque
from datetime import datetime

import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# Allow imports from src/
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from src import Bioreactor, Config
from src.io import get_temperature, set_peltier_power, stop_peltier


SAMPLE_PERIOD_MS = 1000           # poll temperature at 1 Hz
DEFAULT_STEP_DURATION_S = 10 * 60  # 10 minutes per step


def build_sweep_schedule(max_heat=100, max_cool=100, step=10):
    """Heating max_heat% -> step%, off, cooling step% -> max_cool%. Returns list of (direction, duty)."""
    heat = [('heat', d) for d in range(max_heat, 0, -step)] if max_heat > 0 else []
    zero = [('heat', 0)]
    cool = [('cool', d) for d in range(step, max_cool + 1, step)] if max_cool > 0 else []
    return heat + zero + cool


def step_label(direction, duty):
    if duty == 0:
        return "0"
    return f"{'h' if direction == 'heat' else 'c'}{duty}"


class HeaterSweepGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Heater Range Sweep")
        self.root.geometry("1100x720")

        # Initialise only the components needed
        override = {k: False for k in Config.INIT_COMPONENTS}
        override['temp_sensor'] = True
        override['peltier_driver'] = True
        Config.INIT_COMPONENTS = override

        # The sweep writes its own CSV via a save dialog (see _save/export below),
        # so disable the auto data file to avoid an empty header-only CSV per launch.
        Config.DATA_LOGGING = False

        try:
            self.bio = Bioreactor(Config)
        except Exception as e:
            messagebox.showerror("Init Error", f"Failed to initialize bioreactor:\n{e}")
            root.destroy()
            return
        if not self.bio.is_component_initialized('temp_sensor'):
            messagebox.showerror("Init Error", "DS18B20 sensor failed to initialize.")
            root.destroy()
            return
        if not self.bio.is_component_initialized('peltier_driver'):
            messagebox.showerror("Init Error", "Peltier driver failed to initialize.")
            root.destroy()
            return

        self.schedule = build_sweep_schedule(70, 100)
        self.step_duration = DEFAULT_STEP_DURATION_S

        self._t0 = None              # set when sweep starts (or first sample arrives)
        self._step_started = None    # wall-clock time the current step started
        self._step_idx = -1
        self._running = False

        self._times = deque()
        self._temps = deque()
        self._settings = deque()     # parallel: (label, direction, duty) per sample
        self._step_marks = []        # list of (elapsed_s, label) for vlines
        self._vlines = []
        self._vtexts = []

        self._make_widgets()
        # Start polling immediately so the user can see ambient temp before starting
        self._t0 = time.time()
        self._tick()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ------------------------------------------------------------------ UI

    def _make_widgets(self):
        # Settings row: duration + range entries + dynamic schedule summary
        top1 = tk.Frame(self.root)
        top1.pack(fill='x', padx=10, pady=(8, 2))

        tk.Label(top1, text="Step duration (s):", font=("Helvetica", 10)).pack(side='left')
        self.duration_var = tk.IntVar(value=self.step_duration)
        tk.Spinbox(top1, from_=10, to=3600, increment=10,
                   textvariable=self.duration_var, width=7).pack(side='left', padx=(4, 16))

        tk.Label(top1, text="Max H (%):", font=("Helvetica", 10)).pack(side='left')
        self.max_h_var = tk.StringVar(value="70")
        self.max_h_var.trace_add('write', lambda *a: self._update_info())
        tk.Entry(top1, textvariable=self.max_h_var, width=5,
                 font=("Helvetica", 10)).pack(side='left', padx=(4, 12))

        tk.Label(top1, text="Max C (%):", font=("Helvetica", 10)).pack(side='left')
        self.max_c_var = tk.StringVar(value="100")
        self.max_c_var.trace_add('write', lambda *a: self._update_info())
        tk.Entry(top1, textvariable=self.max_c_var, width=5,
                 font=("Helvetica", 10)).pack(side='left', padx=(4, 12))

        self.info_var = tk.StringVar(value="")
        tk.Label(top1, textvariable=self.info_var,
                 font=("Helvetica", 9), fg='#555').pack(side='left', padx=8)

        # Action row: Start / Abort / Export
        top2 = tk.Frame(self.root)
        top2.pack(fill='x', padx=10, pady=(2, 8))

        self.start_btn = tk.Button(top2, text="Start Sweep", font=("Helvetica", 11, "bold"),
                                   bg='#0a6', fg='white', width=14,
                                   command=self._start)
        self.start_btn.pack(side='left', padx=4)
        self.stop_btn = tk.Button(top2, text="Abort", font=("Helvetica", 11, "bold"),
                                  bg='#c00', fg='white', width=10, state='disabled',
                                  command=self._abort)
        self.stop_btn.pack(side='left', padx=4)
        self.export_btn = tk.Button(top2, text="Export CSV", font=("Helvetica", 11),
                                    width=12, command=self._export_csv)
        self.export_btn.pack(side='left', padx=4)

        self._update_info()

        # Status row
        status = tk.Frame(self.root)
        status.pack(fill='x', padx=10, pady=(2, 6))

        self.temp_var = tk.StringVar(value="-- °C")
        tk.Label(status, textvariable=self.temp_var, font=("Helvetica", 26, "bold"),
                 fg="#0a6").pack(side='left', padx=(0, 24))

        self.setting_var = tk.StringVar(value="setting: idle")
        tk.Label(status, textvariable=self.setting_var,
                 font=("Helvetica", 16, "bold")).pack(side='left')

        self.progress_var = tk.StringVar(value="")
        tk.Label(status, textvariable=self.progress_var,
                 font=("Helvetica", 11), fg='#555').pack(side='left', padx=12)

        # Plot
        plot_frame = tk.Frame(self.root)
        plot_frame.pack(fill='both', expand=True, padx=4, pady=4)

        self.fig = Figure(figsize=(9, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Temperature (°C)")
        self.ax.grid(True, alpha=0.3)
        self.line, = self.ax.plot([], [], color='#0a6', lw=1.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        toolbar.pack(fill='x')

    # ------------------------------------------------------------------ sweep control

    def _parse_max(self, var, name):
        """Validate an entry box value: integer in [0, 100]. Shows an error and returns None on failure."""
        raw = var.get().strip()
        try:
            v = int(raw)
        except ValueError:
            messagebox.showerror(
                "Invalid input",
                f"{name} must be an integer 0–100 (got {raw!r}).",
            )
            return None
        if not (0 <= v <= 100):
            messagebox.showerror(
                "Out of range",
                f"{name} must be between 0 and 100 (got {v}).",
            )
            return None
        return v

    def _update_info(self):
        """Recompute the dynamic schedule summary when max H / max C change."""
        try:
            max_h = int(self.max_h_var.get())
            max_c = int(self.max_c_var.get())
        except (ValueError, tk.TclError):
            self.info_var.set("(enter integers 0–100)")
            return
        if not (0 <= max_h <= 100 and 0 <= max_c <= 100):
            self.info_var.set("(values must be 0–100)")
            return
        sched = build_sweep_schedule(max_h, max_c)
        if not sched:
            self.info_var.set("(no steps)")
            return
        first = step_label(*sched[0])
        last = step_label(*sched[-1])
        self.info_var.set(f"({len(sched)} steps: {first} … {last})")

    def _start(self):
        if self._running:
            return
        try:
            self.step_duration = max(10, int(self.duration_var.get()))
        except (TypeError, ValueError):
            self.step_duration = DEFAULT_STEP_DURATION_S

        max_h = self._parse_max(self.max_h_var, "Max H")
        if max_h is None:
            return
        max_c = self._parse_max(self.max_c_var, "Max C")
        if max_c is None:
            return
        self.schedule = build_sweep_schedule(max_h, max_c)
        if max_h == 0 and max_c == 0:
            messagebox.showerror("Empty schedule", "Both Max H and Max C are 0 — nothing to sweep.")
            return

        # Reset plot/data so the sweep starts at t=0
        self._t0 = time.time()
        self._times.clear()
        self._temps.clear()
        self._settings.clear()
        self._step_marks.clear()
        self._step_idx = -1
        self._running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self._advance_step()

    def _abort(self):
        self._running = False
        self._step_idx = -1
        try:
            stop_peltier(self.bio)
        except Exception:
            pass
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.setting_var.set("setting: aborted")
        self.progress_var.set("")

    def _advance_step(self):
        if not self._running:
            return
        self._step_idx += 1
        if self._step_idx >= len(self.schedule):
            stop_peltier(self.bio)
            self._running = False
            self.setting_var.set("setting: complete")
            self.progress_var.set(f"finished {len(self.schedule)} steps")
            self.start_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            return

        direction, duty = self.schedule[self._step_idx]
        if duty == 0:
            stop_peltier(self.bio)
        else:
            set_peltier_power(self.bio, duty, forward=direction)

        self._step_started = time.time()
        label = step_label(direction, duty)
        elapsed_s = self._step_started - self._t0
        self._step_marks.append((elapsed_s, label))
        self.bio.logger.info(
            f"Sweep step {self._step_idx + 1}/{len(self.schedule)}: {label} "
            f"({self.step_duration}s)"
        )

    # ------------------------------------------------------------------ polling

    def _tick(self):
        try:
            temp = get_temperature(self.bio)
        except Exception as e:
            self.bio.logger.error(f"Temperature read failed: {e}")
            temp = float('nan')

        now = time.time()
        elapsed = now - self._t0
        self._times.append(elapsed)
        self._temps.append(temp)

        # Record the active setting alongside this sample for CSV export
        if self._running and 0 <= self._step_idx < len(self.schedule):
            direction, duty = self.schedule[self._step_idx]
            self._settings.append((step_label(direction, duty), direction, duty))
        else:
            self._settings.append(('', '', 0))

        if temp == temp:
            self.temp_var.set(f"{temp:.2f} °C")
        else:
            self.temp_var.set("-- °C")

        # Update sweep progress and advance step when its duration elapses
        if self._running and self._step_started is not None:
            step_elapsed = now - self._step_started
            remaining = max(0.0, self.step_duration - step_elapsed)
            direction, duty = self.schedule[self._step_idx]
            mins, secs = divmod(int(remaining), 60)
            self.setting_var.set(f"setting: {step_label(direction, duty)}")
            self.progress_var.set(
                f"step {self._step_idx + 1}/{len(self.schedule)}   "
                f"{mins:d}:{secs:02d} remaining"
            )
            if step_elapsed >= self.step_duration:
                self._advance_step()

        self._refresh_plot()
        self.root.after(SAMPLE_PERIOD_MS, self._tick)

    def _refresh_plot(self):
        if not self._times:
            self.canvas.draw_idle()
            return
        self.line.set_data(list(self._times), list(self._temps))

        # Redraw step boundary markers
        for vl in self._vlines:
            vl.remove()
        for txt in self._vtexts:
            txt.remove()
        self._vlines.clear()
        self._vtexts.clear()

        valid = [v for v in self._temps if v == v]
        if valid:
            ymin, ymax = min(valid), max(valid)
            pad = max(0.5, 0.1 * (ymax - ymin))
            self.ax.set_ylim(ymin - pad, ymax + pad)
            for t_mark, label in self._step_marks:
                self._vlines.append(
                    self.ax.axvline(t_mark, color='#888', lw=0.5, alpha=0.5)
                )
                self._vtexts.append(
                    self.ax.text(t_mark, ymax + pad * 0.4, label,
                                 fontsize=7, rotation=90,
                                 va='top', ha='left', color='#444')
                )
        self.ax.set_xlim(0, max(self._times[-1], 1))
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ export

    def _export_csv(self):
        if not self._times:
            messagebox.showinfo("Export CSV", "No data to export yet.")
            return

        start_dt = datetime.fromtimestamp(self._t0)
        default_name = f"heater_sweep_{start_dt.strftime('%Y%m%d_%H%M%S')}.csv"
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export sweep data",
        )
        if not path:
            return

        # Snapshot the deques so concurrent ticks don't shift indices during write
        times = list(self._times)
        temps = list(self._temps)
        settings = list(self._settings)
        n = min(len(times), len(temps), len(settings))

        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'elapsed_s', 'temperature_C',
                    'step_label', 'direction', 'duty_percent',
                ])
                for i in range(n):
                    elapsed = times[i]
                    temp = temps[i]
                    label, direction, duty = settings[i]
                    abs_t = datetime.fromtimestamp(self._t0 + elapsed).isoformat(
                        timespec='milliseconds'
                    )
                    writer.writerow([
                        abs_t,
                        f"{elapsed:.3f}",
                        f"{temp:.4f}" if temp == temp else "",
                        label,
                        direction,
                        duty,
                    ])
            self.bio.logger.info(f"Exported {n} samples to {path}")
            messagebox.showinfo("Export CSV", f"Exported {n} samples to:\n{path}")
        except Exception as e:
            self.bio.logger.error(f"CSV export failed: {e}")
            messagebox.showerror("Export CSV", f"Failed to export:\n{e}")

    # ------------------------------------------------------------------ shutdown

    def on_closing(self):
        self._running = False
        try:
            stop_peltier(self.bio)
        except Exception:
            pass
        try:
            self.bio.finish()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = HeaterSweepGUI(root)
    root.mainloop()
