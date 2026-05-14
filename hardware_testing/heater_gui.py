"""
Heater Control GUI

Live temperature plot from a DS18B20 sensor with interactive controls for
the peltier module: a heat/cool direction switch and a 0-100% PWM slider.

Changes to the slider or direction switch are applied to the peltier driver
immediately. The "ALL OFF" button stops PWM output and zeroes the slider.
"""

import os
import sys
import time
from collections import deque

import tkinter as tk
from tkinter import messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# Allow imports from src/
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from src import Bioreactor, Config
from src.io import get_temperature, set_peltier_power, stop_peltier
from src.utils import temperature_pid_controller


SAMPLE_PERIOD_MS = 1000          # poll temperature at 1 Hz
PLOT_WINDOW_SECONDS = 30 * 60    # keep the most recent 30 minutes


class HeaterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Heater Control")
        self.root.geometry("1000x780")

        # Initialise only the components needed for heater control
        override = {k: False for k in Config.INIT_COMPONENTS}
        override['temp_sensor'] = True
        override['peltier_driver'] = True
        Config.INIT_COMPONENTS = override

        try:
            self.bio = Bioreactor(Config)
        except Exception as e:
            messagebox.showerror("Init Error", f"Failed to initialize bioreactor:\n{e}")
            root.destroy()
            return

        if not self.bio.is_component_initialized('temp_sensor'):
            messagebox.showerror("Init Error",
                "DS18B20 temperature sensor failed to initialize.\n\n"
                "Check that the 1-Wire interface is enabled and a sensor is connected.")
            root.destroy()
            return
        if not self.bio.is_component_initialized('peltier_driver'):
            messagebox.showerror("Init Error",
                "Peltier driver failed to initialize.\n\n"
                "Check PELTIER_PWM_PIN / PELTIER_DIR_PIN in Config.")
            root.destroy()
            return

        self._t0 = time.time()
        self._times = deque()
        self._temps = deque()
        self._pid_active = False

        self._make_widgets()
        self._apply_setting()  # push initial state (0% / heat) to the driver
        self._tick()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ------------------------------------------------------------------ UI

    def _make_widgets(self):
        # Left column — controls
        left = tk.Frame(self.root, width=320)
        left.pack(side='left', fill='y', padx=10, pady=10)
        left.pack_propagate(False)

        tk.Label(left, text="Heater Control", font=("Helvetica", 14, "bold")).pack(pady=(0, 12))

        # Live temperature read-out
        self.temp_var = tk.StringVar(value="-- °C")
        tk.Label(left, textvariable=self.temp_var, font=("Helvetica", 28, "bold"),
                 fg="#0a6").pack(pady=(0, 14))

        # Direction switch (radio styled as buttons)
        tk.Label(left, text="Direction", font=("Helvetica", 11, "bold"), anchor='w').pack(fill='x')
        self.direction_var = tk.StringVar(value='heat')
        dir_frame = tk.Frame(left)
        dir_frame.pack(fill='x', pady=(2, 14))
        self.dir_radios = []
        heat_rb = tk.Radiobutton(dir_frame, text="HEAT", variable=self.direction_var, value='heat',
                       font=("Helvetica", 11, "bold"), indicatoron=False, height=2,
                       command=self._apply_setting,
                       selectcolor='#e66', bg='#fee')
        heat_rb.pack(side='left', expand=True, fill='x', padx=2)
        self.dir_radios.append(heat_rb)
        cool_rb = tk.Radiobutton(dir_frame, text="COOL", variable=self.direction_var, value='cool',
                       font=("Helvetica", 11, "bold"), indicatoron=False, height=2,
                       command=self._apply_setting,
                       selectcolor='#5ae', bg='#eef')
        cool_rb.pack(side='left', expand=True, fill='x', padx=2)
        self.dir_radios.append(cool_rb)

        # PWM slider
        tk.Label(left, text="PWM duty cycle", font=("Helvetica", 11, "bold"), anchor='w').pack(fill='x')
        self.duty_var = tk.DoubleVar(value=0.0)
        self.duty_label_var = tk.StringVar(value="0 %")
        tk.Label(left, textvariable=self.duty_label_var, font=("Courier", 14)).pack()
        self.duty_slider = tk.Scale(
            left, from_=0, to=100, resolution=1, orient='horizontal',
            variable=self.duty_var, showvalue=False, length=280,
            command=lambda v: self._on_slider_change(),
        )
        self.duty_slider.pack(fill='x', pady=(0, 14))

        # Off button
        tk.Button(left, text="ALL OFF", font=("Helvetica", 11, "bold"),
                  bg='#c00', fg='white', height=2,
                  command=self._all_off).pack(fill='x', pady=4)

        # ---------------- PID controller ----------------
        tk.Frame(left, height=2, bg='#ccc').pack(fill='x', pady=(12, 8))
        tk.Label(left, text="PID controller", font=("Helvetica", 11, "bold"),
                 anchor='w').pack(fill='x')

        sp_frame = tk.Frame(left)
        sp_frame.pack(fill='x', pady=(4, 4))
        tk.Label(sp_frame, text="Setpoint (°C):", font=("Helvetica", 10)).pack(side='left')
        self.setpoint_var = tk.StringVar(value="30.0")
        tk.Entry(sp_frame, textvariable=self.setpoint_var, width=8,
                 font=("Helvetica", 11)).pack(side='left', padx=4)

        self.pid_btn = tk.Button(left, text="Start PID", font=("Helvetica", 11, "bold"),
                                 bg='#06c', fg='white', height=2,
                                 command=self._toggle_pid)
        self.pid_btn.pack(fill='x', pady=4)

        self.pid_status_var = tk.StringVar(value="PID: off")
        tk.Label(left, textvariable=self.pid_status_var, font=("Helvetica", 9),
                 fg='#555', anchor='w', justify='left').pack(fill='x')

        # Right column — plot
        right = tk.Frame(self.root)
        right.pack(side='right', fill='both', expand=True, padx=4, pady=4)

        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Temperature (°C)")
        self.ax.grid(True, alpha=0.3)
        self.line, = self.ax.plot([], [], color='#0a6', lw=1.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, right)
        toolbar.update()
        toolbar.pack(fill='x')

    # ------------------------------------------------------------------ events

    def _on_slider_change(self):
        self.duty_label_var.set(f"{int(self.duty_var.get())} %")
        self._apply_setting()

    def _apply_setting(self):
        duty = float(self.duty_var.get())
        direction = self.direction_var.get()
        if duty <= 0:
            stop_peltier(self.bio)
        else:
            set_peltier_power(self.bio, duty, forward=direction)

    def _all_off(self):
        if self._pid_active:
            self._stop_pid()
        self.duty_var.set(0)
        self.duty_label_var.set("0 %")
        stop_peltier(self.bio)

    # ------------------------------------------------------------------ PID

    def _toggle_pid(self):
        if self._pid_active:
            self._stop_pid()
        else:
            self._start_pid()

    def _start_pid(self):
        try:
            sp = float(self.setpoint_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid setpoint",
                f"Setpoint must be a number (got {self.setpoint_var.get()!r}).",
            )
            return

        # Reset PID state on the bioreactor so the integral starts fresh
        for attr in ('_temp_integral', '_temp_last_error',
                     '_temp_last_time', '_temp_last_derivative'):
            if hasattr(self.bio, attr):
                delattr(self.bio, attr)

        self._pid_active = True
        self.pid_btn.config(text="Stop PID", bg='#c00')
        self.duty_slider.config(state='disabled')
        for rb in self.dir_radios:
            rb.config(state='disabled')
        self.pid_status_var.set(f"PID: running (target {sp:.2f} °C)")

    def _stop_pid(self):
        self._pid_active = False
        self.pid_btn.config(text="Start PID", bg='#06c')
        self.duty_slider.config(state='normal')
        for rb in self.dir_radios:
            rb.config(state='normal')
        try:
            stop_peltier(self.bio)
        except Exception:
            pass
        self.pid_status_var.set("PID: off")

    def _pid_step(self, temp):
        try:
            sp = float(self.setpoint_var.get())
        except ValueError:
            self.pid_status_var.set("PID: invalid setpoint")
            return
        if temp != temp:  # NaN — skip
            self.pid_status_var.set(f"PID: target {sp:.2f} °C  (no temp reading)")
            return
        try:
            temperature_pid_controller(self.bio, setpoint=sp, current_temp=temp)
        except Exception as e:
            self.bio.logger.error(f"PID call failed: {e}")
            self.pid_status_var.set(f"PID: error ({e})")
            return

        # Reflect what the PID is doing on the (disabled) slider + status label
        state = self.bio.peltier_driver.get_state() if self.bio.peltier_driver else None
        if state is not None:
            duty, _ = state
            duty_int = int(round(duty))
            self.duty_var.set(duty_int)
            self.duty_label_var.set(f"{duty_int} %")
            self.pid_status_var.set(
                f"PID: target {sp:.2f} °C   duty {duty_int} %"
            )

    # ------------------------------------------------------------------ polling

    def _tick(self):
        try:
            temp = get_temperature(self.bio)
        except Exception as e:
            self.bio.logger.error(f"Temperature read failed: {e}")
            temp = float('nan')

        t = time.time() - self._t0
        self._times.append(t)
        self._temps.append(temp)

        # Drop samples outside the plot window
        while self._times and self._times[0] < t - PLOT_WINDOW_SECONDS:
            self._times.popleft()
            self._temps.popleft()

        if temp == temp:  # filter NaN
            self.temp_var.set(f"{temp:.2f} °C")
        else:
            self.temp_var.set("-- °C")

        # PID step (if active): drive peltier toward setpoint
        if self._pid_active:
            self._pid_step(temp)

        self.line.set_data(list(self._times), list(self._temps))
        valid = [v for v in self._temps if v == v]
        if valid:
            ymin, ymax = min(valid), max(valid)
            pad = max(0.5, 0.1 * (ymax - ymin))
            self.ax.set_ylim(ymin - pad, ymax + pad)
        if self._times:
            self.ax.set_xlim(max(0, self._times[0]), max(self._times[-1], 1))
        self.canvas.draw_idle()

        self.root.after(SAMPLE_PERIOD_MS, self._tick)

    # ------------------------------------------------------------------ shutdown

    def on_closing(self):
        self._pid_active = False
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
    app = HeaterGUI(root)
    root.mainloop()
