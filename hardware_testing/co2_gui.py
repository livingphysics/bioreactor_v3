"""
Live CO2 + O2 GUI

Streams CO2 readings from a Senseair K33 (I2C, via sensair_k33) and O2 readings
from an Atlas Scientific O2 sensor (I2C, via atlas_o2), displaying both as large
numeric readouts plus scrolling strip charts.

The charts are drawn on plain Tkinter canvases (no matplotlib), so the only hard
dependencies are tkinter and smbus2 — the same set sensair_k33.py already needs.
The O2 sensor is optional: if the atlas_i2c library or the sensor is missing,
CO2 keeps streaming and O2 shows the error.

Features:
  - Live CO2 (ppm) and O2 (%) readouts with min / max / mean over the window
  - Adjustable I2C bus, sensor addresses and sample interval
  - I2C bus scan to confirm which sensors are present
  - Optional CSV logging to bioreactor_data/<timestamp>_co2_o2_data.csv

Usage:
    python hardware_testing/co2_gui.py [bus_num]
"""

import csv
import os
import queue
import sys
import threading
import time
import tkinter as tk
from collections import deque
from datetime import datetime
from tkinter import messagebox, ttk

# Allow running as `python hardware_testing/co2_gui.py` from the repo root
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from atlas_o2 import AtlasO2, AtlasO2Error
from atlas_o2 import DEFAULT_I2C_ADDRESS as O2_DEFAULT_ADDRESS
from sensair_k33 import (
    DEFAULT_I2C_ADDRESS as K33_DEFAULT_ADDRESS,
    SenseairK33,
    SenseairK33Error,
    scan_i2c_bus,
)


# Number of samples retained for the charts / statistics
MAX_POINTS = 600

# Chart geometry
PLOT_WIDTH = 640
PLOT_HEIGHT = 190
PLOT_MARGIN_LEFT = 60
PLOT_MARGIN_RIGHT = 15
PLOT_MARGIN_TOP = 15
PLOT_MARGIN_BOTTOM = 30


class SensorChart(tk.Canvas):
    """Scrolling strip chart of one sensor signal against elapsed time."""

    def __init__(self, master, axis_label="value", colour="#2196F3",
                 min_span=100.0, value_fmt="{:.0f}", **kwargs):
        super().__init__(master, width=PLOT_WIDTH, height=PLOT_HEIGHT,
                         bg="white", highlightthickness=1,
                         highlightbackground="#cccccc", **kwargs)
        self.axis_label = axis_label
        self.colour = colour
        self.min_span = min_span  # Minimum y span, so a flat trace isn't amplified noise
        self.value_fmt = value_fmt
        self.bind("<Configure>", lambda event: self.redraw())
        self.samples = []  # list of (elapsed_seconds, value)

    def update_samples(self, samples):
        self.samples = samples
        self.redraw()

    def redraw(self):
        self.delete("all")

        width = self.winfo_width() or PLOT_WIDTH
        height = self.winfo_height() or PLOT_HEIGHT
        x0 = PLOT_MARGIN_LEFT
        y0 = PLOT_MARGIN_TOP
        x1 = width - PLOT_MARGIN_RIGHT
        y1 = height - PLOT_MARGIN_BOTTOM

        if x1 <= x0 or y1 <= y0:
            return

        # Plot frame
        self.create_rectangle(x0, y0, x1, y1, outline="#888888")

        if not self.samples:
            self.create_text((x0 + x1) / 2, (y0 + y1) / 2,
                             text=f"Waiting for {self.axis_label} readings...",
                             fill="#888888", font=("Arial", 11))
            return

        times = [s[0] for s in self.samples]
        values = [s[1] for s in self.samples]

        y_min, y_max = self._y_range(values)
        t_min = times[0]
        t_max = times[-1]
        t_span = max(t_max - t_min, 1.0)

        # Horizontal gridlines and y labels
        for i in range(5):
            frac = i / 4.0
            y = y1 - frac * (y1 - y0)
            value = y_min + frac * (y_max - y_min)
            if i not in (0, 4):
                self.create_line(x0, y, x1, y, fill="#e8e8e8")
            self.create_text(x0 - 6, y, text=self.value_fmt.format(value), anchor="e",
                             fill="#555555", font=("Arial", 8))

        # X axis labels (seconds since the run started)
        self.create_text(x0, y1 + 12, text=f"{t_min:.0f}s", anchor="w",
                         fill="#555555", font=("Arial", 8))
        self.create_text(x1, y1 + 12, text=f"{t_max:.0f}s", anchor="e",
                         fill="#555555", font=("Arial", 8))
        self.create_text((x0 + x1) / 2, y1 + 12, text="elapsed time",
                         fill="#555555", font=("Arial", 8))

        # Y axis title
        self.create_text(12, (y0 + y1) / 2, text=self.axis_label, angle=90,
                         fill="#555555", font=("Arial", 9))

        # Trace
        points = []
        for t, value in self.samples:
            x = x0 + (t - t_min) / t_span * (x1 - x0)
            y = y1 - (value - y_min) / (y_max - y_min) * (y1 - y0)
            points.extend([x, y])

        if len(points) >= 4:
            self.create_line(*points, fill=self.colour, width=2)

        # Marker on the most recent sample
        self.create_oval(points[-2] - 3, points[-1] - 3,
                         points[-2] + 3, points[-1] + 3,
                         fill=self.colour, outline="")

    def _y_range(self, values):
        """Auto-scaled y limits with padding and a minimum span."""
        low = min(values)
        high = max(values)
        span = high - low
        if span < self.min_span:
            centre = (high + low) / 2.0
            low = centre - self.min_span / 2.0
            high = centre + self.min_span / 2.0
        else:
            pad = span * 0.1
            low -= pad
            high += pad
        return low, high


class GasMonitorGUI:
    def __init__(self, root, bus_num=1, co2_addr=K33_DEFAULT_ADDRESS,
                 o2_addr=O2_DEFAULT_ADDRESS):
        self.root = root
        self.root.title("CO2 + O2 Monitor (I2C)")
        self.root.geometry("760x840")

        # Sampling state
        self.reader_thread = None
        self.stop_event = threading.Event()
        self.sample_queue = queue.Queue()
        self.co2_samples = deque(maxlen=MAX_POINTS)
        self.o2_samples = deque(maxlen=MAX_POINTS)
        self.start_time = None
        self.error_count = 0

        # CSV logging state
        self.csv_file = None
        self.csv_writer = None
        self.csv_path = None

        self.bus_var = tk.StringVar(value=str(bus_num))
        self.co2_addr_var = tk.StringVar(value=f"0x{co2_addr:02X}")
        self.o2_addr_var = tk.StringVar(value=f"0x{o2_addr:02X}")
        self.interval_var = tk.StringVar(value="2.0")
        self.co2_enabled_var = tk.BooleanVar(value=True)
        self.o2_enabled_var = tk.BooleanVar(value=True)
        self.log_csv_var = tk.BooleanVar(value=False)

        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(200, self.poll_samples)

    # ------------------------------------------------------------------ UI

    def create_widgets(self):
        title_label = tk.Label(self.root, text="CO2 + O2 Monitor",
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(10, 2))

        self.status_label = tk.Label(self.root, text="Idle — press Start",
                                     fg="orange", font=("Arial", 10))
        self.status_label.pack(pady=(0, 8))

        # Sensor settings
        settings_frame = tk.Frame(self.root)
        settings_frame.pack(pady=4)

        tk.Label(settings_frame, text="Bus:", font=("Arial", 10)).grid(row=0, column=0, sticky="e", padx=(5, 2))
        self.bus_entry = tk.Entry(settings_frame, textvariable=self.bus_var, width=4)
        self.bus_entry.grid(row=0, column=1, sticky="w")

        tk.Label(settings_frame, text="Interval (s):", font=("Arial", 10)).grid(row=0, column=2, sticky="e", padx=(12, 2))
        self.interval_combo = ttk.Combobox(settings_frame, textvariable=self.interval_var,
                                           values=["0.5", "1.0", "2.0", "5.0", "10.0"],
                                           width=5, state="readonly")
        self.interval_combo.grid(row=0, column=3, sticky="w")

        self.co2_check = tk.Checkbutton(settings_frame, text="CO2 (K33) @",
                                        variable=self.co2_enabled_var, font=("Arial", 10))
        self.co2_check.grid(row=1, column=0, columnspan=2, sticky="e", pady=(6, 0))
        self.co2_addr_entry = tk.Entry(settings_frame, textvariable=self.co2_addr_var, width=6)
        self.co2_addr_entry.grid(row=1, column=2, sticky="w", pady=(6, 0))

        self.o2_check = tk.Checkbutton(settings_frame, text="O2 (Atlas) @",
                                       variable=self.o2_enabled_var, font=("Arial", 10))
        self.o2_check.grid(row=1, column=3, sticky="e", padx=(12, 2), pady=(6, 0))
        self.o2_addr_entry = tk.Entry(settings_frame, textvariable=self.o2_addr_var, width=6)
        self.o2_addr_entry.grid(row=1, column=4, sticky="w", pady=(6, 0))

        # Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=8)

        self.start_button = tk.Button(button_frame, text="Start", command=self.start_reading,
                                      font=("Arial", 11, "bold"), bg="#4CAF50", fg="white",
                                      width=10)
        self.start_button.pack(side="left", padx=5)

        self.stop_button = tk.Button(button_frame, text="Stop", command=self.stop_reading,
                                     font=("Arial", 11, "bold"), bg="#f44336", fg="white",
                                     width=10, state="disabled")
        self.stop_button.pack(side="left", padx=5)

        self.scan_button = tk.Button(button_frame, text="Scan I2C Bus", command=self.scan_bus,
                                     font=("Arial", 11), bg="#2196F3", fg="white", width=12)
        self.scan_button.pack(side="left", padx=5)

        self.clear_button = tk.Button(button_frame, text="Clear", command=self.clear_data,
                                      font=("Arial", 11), width=8)
        self.clear_button.pack(side="left", padx=5)

        self.log_check = tk.Checkbutton(self.root, text="Log readings to CSV",
                                        variable=self.log_csv_var, font=("Arial", 9))
        self.log_check.pack()

        # Current values, side by side
        readout_frame = tk.Frame(self.root)
        readout_frame.pack(pady=(8, 4), fill="x")
        readout_frame.columnconfigure(0, weight=1)
        readout_frame.columnconfigure(1, weight=1)

        self.co2_value_label = tk.Label(readout_frame, text="--- ppm",
                                        font=("Arial", 30, "bold"), fg="#333333")
        self.co2_value_label.grid(row=0, column=0)
        self.co2_stats_label = tk.Label(readout_frame,
                                        text="min --  |  max --  |  mean --  |  n = 0",
                                        font=("Arial", 9), fg="#555555")
        self.co2_stats_label.grid(row=1, column=0)

        self.o2_value_label = tk.Label(readout_frame, text="--- %",
                                       font=("Arial", 30, "bold"), fg="#333333")
        self.o2_value_label.grid(row=0, column=1)
        self.o2_stats_label = tk.Label(readout_frame,
                                       text="min --  |  max --  |  mean --  |  n = 0",
                                       font=("Arial", 9), fg="#555555")
        self.o2_stats_label.grid(row=1, column=1)

        # Charts
        self.co2_chart = SensorChart(self.root, axis_label="CO2 (ppm)", colour="#2196F3",
                                     min_span=100.0, value_fmt="{:.0f}")
        self.co2_chart.pack(fill="both", expand=True, padx=12, pady=(6, 4))

        self.o2_chart = SensorChart(self.root, axis_label="O2 (%)", colour="#4CAF50",
                                    min_span=1.0, value_fmt="{:.2f}")
        self.o2_chart.pack(fill="both", expand=True, padx=12, pady=(4, 12))

    # ------------------------------------------------------------- sampling

    @staticmethod
    def parse_address(text):
        text = text.strip()
        return int(text, 16) if text.lower().startswith("0x") else int(text)

    def parse_settings(self):
        """Read bus / addresses / interval from the entry fields."""
        bus_num = int(self.bus_var.get().strip())
        co2_addr = self.parse_address(self.co2_addr_var.get())
        o2_addr = self.parse_address(self.o2_addr_var.get())
        interval = float(self.interval_var.get())
        return bus_num, co2_addr, o2_addr, interval

    def start_reading(self):
        try:
            bus_num, co2_addr, o2_addr, interval = self.parse_settings()
        except ValueError:
            messagebox.showerror("Invalid settings",
                                 "Bus must be an integer, addresses an integer or 0x hex value.")
            return

        if not self.co2_enabled_var.get() and not self.o2_enabled_var.get():
            messagebox.showerror("No sensors selected", "Enable CO2, O2, or both.")
            return

        if self.log_csv_var.get() and self.csv_writer is None:
            try:
                self.open_csv()
            except OSError as e:
                messagebox.showerror("CSV error", f"Could not open log file: {e}")
                return

        self.stop_event.clear()
        self.error_count = 0
        if self.start_time is None:
            self.start_time = time.time()

        co2_sensor = (SenseairK33(bus_num=bus_num, i2c_addr=co2_addr)
                      if self.co2_enabled_var.get() else None)
        o2_sensor = AtlasO2(i2c_addr=o2_addr) if self.o2_enabled_var.get() else None

        self.reader_thread = threading.Thread(
            target=self.reader_loop, args=(co2_sensor, o2_sensor, interval), daemon=True)
        self.reader_thread.start()

        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.set_widgets_state("disabled")

        active = []
        if co2_sensor is not None:
            active.append(f"CO2 0x{co2_addr:02X}")
        if o2_sensor is not None:
            active.append(f"O2 0x{o2_addr:02X}")
        self.status_label.config(text=f"Reading {' + '.join(active)} on bus {bus_num}...",
                                 fg="green")

    def stop_reading(self):
        self.stop_event.set()
        if self.reader_thread is not None:
            self.reader_thread.join(timeout=5.0)
            self.reader_thread = None
        self.close_csv()

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.set_widgets_state("normal")
        self.status_label.config(text="Stopped", fg="orange")

    def reader_loop(self, co2_sensor, o2_sensor, interval):
        """Background thread: read both sensors and hand each cycle to the UI."""
        while not self.stop_event.is_set():
            record = {"timestamp": time.time(), "co2": None, "o2": None, "errors": []}

            if co2_sensor is not None:
                try:
                    record["co2"] = co2_sensor.read_co2()
                except SenseairK33Error as e:
                    record["errors"].append(f"CO2: {e}")
                except Exception as e:
                    record["errors"].append(f"CO2: unexpected error: {e}")

            if o2_sensor is not None and not self.stop_event.is_set():
                try:
                    record["o2"] = o2_sensor.read_o2()
                except AtlasO2Error as e:
                    record["errors"].append(f"O2: {e}")
                except Exception as e:
                    record["errors"].append(f"O2: unexpected error: {e}")

            self.sample_queue.put(record)

            # Wait on the event so Stop responds quickly at long intervals
            self.stop_event.wait(interval)

    def poll_samples(self):
        """Main thread: drain the queue, update readouts, charts and CSV."""
        new_co2 = False
        new_o2 = False

        while True:
            try:
                record = self.sample_queue.get_nowait()
            except queue.Empty:
                break

            timestamp = record["timestamp"]
            elapsed = timestamp - (self.start_time or timestamp)

            if record["co2"] is not None:
                self.co2_samples.append((elapsed, float(record["co2"])))
                new_co2 = True
                self.co2_value_label.config(text=f"{record['co2']} ppm", fg="#333333")
            elif self.co2_enabled_var.get():
                self.co2_value_label.config(fg="#f44336")

            if record["o2"] is not None:
                self.o2_samples.append((elapsed, float(record["o2"])))
                new_o2 = True
                self.o2_value_label.config(text=f"{record['o2']:.2f} %", fg="#333333")
            elif self.o2_enabled_var.get():
                self.o2_value_label.config(fg="#f44336")

            self.write_csv_row(timestamp, elapsed, record["co2"], record["o2"])

            if record["errors"]:
                self.error_count += 1
                self.status_label.config(
                    text=f"Error ({self.error_count}): {'; '.join(record['errors'])}",
                    fg="red")
            else:
                self.error_count = 0
                if self.stop_button["state"] == "normal":
                    self.status_label.config(text="Reading...", fg="green")

        if new_co2:
            self.co2_chart.update_samples(list(self.co2_samples))
        if new_o2:
            self.o2_chart.update_samples(list(self.o2_samples))
        if new_co2 or new_o2:
            self.update_stats()

        self.root.after(200, self.poll_samples)

    def update_stats(self):
        self.co2_stats_label.config(
            text=self.format_stats([s[1] for s in self.co2_samples], "{:.0f}"))
        self.o2_stats_label.config(
            text=self.format_stats([s[1] for s in self.o2_samples], "{:.2f}"))

    @staticmethod
    def format_stats(values, fmt):
        if not values:
            return "min --  |  max --  |  mean --  |  n = 0"
        mean = sum(values) / len(values)
        return (f"min {fmt.format(min(values))}  |  max {fmt.format(max(values))}  |  "
                f"mean {fmt.format(mean)}  |  n = {len(values)}")

    def clear_data(self):
        self.co2_samples.clear()
        self.o2_samples.clear()
        self.start_time = time.time() if self.reader_thread is not None else None
        self.co2_chart.update_samples([])
        self.o2_chart.update_samples([])
        self.co2_value_label.config(text="--- ppm", fg="#333333")
        self.o2_value_label.config(text="--- %", fg="#333333")
        self.update_stats()

    def scan_bus(self):
        try:
            bus_num, co2_addr, o2_addr, _ = self.parse_settings()
        except ValueError:
            messagebox.showerror("Invalid settings", "Bus must be an integer.")
            return

        try:
            devices = scan_i2c_bus(bus_num, verbose=False)
        except SenseairK33Error as e:
            messagebox.showerror("Scan failed", str(e))
            return

        found = ", ".join(devices) if devices else "none"
        missing = []
        if self.co2_enabled_var.get() and hex(co2_addr) not in devices:
            missing.append(f"CO2 0x{co2_addr:02X}")
        if self.o2_enabled_var.get() and hex(o2_addr) not in devices:
            missing.append(f"O2 0x{o2_addr:02X}")

        if missing:
            self.status_label.config(
                text=f"Not found on bus {bus_num}: {', '.join(missing)} — devices: {found}",
                fg="red")
        else:
            self.status_label.config(
                text=f"All enabled sensors found on bus {bus_num}", fg="green")
        messagebox.showinfo("I2C scan", f"Bus {bus_num} devices: {found}")

    # ---------------------------------------------------------- CSV logging

    def open_csv(self):
        data_dir = os.path.join(script_dir, "bioreactor_data")
        os.makedirs(data_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(data_dir, f"{stamp}_co2_o2_data.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["timestamp", "elapsed_s", "co2_ppm", "o2_percent"])

    def write_csv_row(self, timestamp, elapsed, co2_ppm, o2_percent):
        if self.csv_writer is None:
            return
        self.csv_writer.writerow([
            datetime.fromtimestamp(timestamp).isoformat(timespec="seconds"),
            f"{elapsed:.2f}",
            "" if co2_ppm is None else co2_ppm,
            "" if o2_percent is None else f"{o2_percent:.2f}",
        ])
        self.csv_file.flush()

    def close_csv(self):
        if self.csv_file is not None:
            self.csv_file.close()
            print(f"Gas log written to {self.csv_path}")
        self.csv_file = None
        self.csv_writer = None

    # -------------------------------------------------------------- helpers

    def set_widgets_state(self, state):
        self.bus_entry.config(state=state)
        self.co2_addr_entry.config(state=state)
        self.o2_addr_entry.config(state=state)
        self.co2_check.config(state=state)
        self.o2_check.config(state=state)
        self.interval_combo.config(state="readonly" if state == "normal" else "disabled")
        self.scan_button.config(state=state)
        self.log_check.config(state=state)

    def on_close(self):
        self.stop_event.set()
        if self.reader_thread is not None:
            self.reader_thread.join(timeout=5.0)
        self.close_csv()
        self.root.destroy()


def main():
    bus_num = 1
    if len(sys.argv) > 1:
        try:
            bus_num = int(sys.argv[1])
        except ValueError:
            print(f"Invalid bus number: {sys.argv[1]}, using default: 1")

    root = tk.Tk()
    GasMonitorGUI(root, bus_num=bus_num)
    root.mainloop()


if __name__ == "__main__":
    main()
