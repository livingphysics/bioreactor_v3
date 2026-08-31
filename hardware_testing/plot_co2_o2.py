#!/usr/bin/env python3
"""
Quick plotter for CO2 / O2 log files written by co2_gui.py.

Reads a CSV with columns timestamp, elapsed_s, co2_ppm [, o2_percent] and plots
CO2 on the left axis and O2 (when present) on the right axis against elapsed
time. Blank cells (a sensor that failed for one cycle) are skipped per channel.

Usage:
    python hardware_testing/plot_co2_o2.py                 # most recent log
    python hardware_testing/plot_co2_o2.py <csv_path>      # a specific file
    python hardware_testing/plot_co2_o2.py <csv_path> --save-only

Options:
    --save-only    Write the PNG next to the CSV without opening a window
"""

import csv
import glob
import os
import sys

import matplotlib
import matplotlib.pyplot as plt

# Default location for logs written by co2_gui.py
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bioreactor_data")

CO2_COLOUR = "#2196F3"
O2_COLOUR = "#4CAF50"


def find_latest_csv(data_dir=DATA_DIR):
    """Return the most recently modified CO2/O2 log in data_dir."""
    candidates = glob.glob(os.path.join(data_dir, "*_co2_data.csv"))
    candidates += glob.glob(os.path.join(data_dir, "*_co2_o2_data.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CO2/O2 CSV files found in {data_dir}")
    return max(candidates, key=os.path.getmtime)


def load_csv(csv_path):
    """
    Load a CO2/O2 log.

    Returns:
        dict with 'co2' and 'o2' entries, each a (times_minutes, values) tuple.
        A channel with no data has empty lists.
    """
    co2_t, co2_v, o2_t, o2_v = [], [], [], []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if "elapsed_s" not in (reader.fieldnames or []):
            raise ValueError(
                f"{csv_path} does not look like a co2_gui.py log "
                f"(columns: {reader.fieldnames})"
            )

        for row in reader:
            try:
                minutes = float(row["elapsed_s"]) / 60.0
            except (TypeError, ValueError):
                continue  # Skip malformed / partially written rows

            co2 = (row.get("co2_ppm") or "").strip()
            if co2:
                try:
                    co2_v.append(float(co2))
                    co2_t.append(minutes)
                except ValueError:
                    pass

            o2 = (row.get("o2_percent") or "").strip()
            if o2:
                try:
                    o2_v.append(float(o2))
                    o2_t.append(minutes)
                except ValueError:
                    pass

    return {"co2": (co2_t, co2_v), "o2": (o2_t, o2_v)}


def summarise(name, values, unit, fmt="{:.1f}"):
    """Print min / max / mean / n for one channel."""
    if not values:
        print(f"{name}: no data")
        return
    mean = sum(values) / len(values)
    print(f"{name}: n = {len(values)}, min {fmt.format(min(values))} {unit}, "
          f"max {fmt.format(max(values))} {unit}, mean {fmt.format(mean)} {unit}, "
          f"first {fmt.format(values[0])} -> last {fmt.format(values[-1])} {unit}")


def plot(csv_path, show=True):
    """Plot one CO2/O2 log and save a PNG next to it. Returns the PNG path."""
    data = load_csv(csv_path)
    co2_t, co2_v = data["co2"]
    o2_t, o2_v = data["o2"]

    if not co2_v and not o2_v:
        raise ValueError(f"No usable CO2 or O2 readings in {csv_path}")

    print(f"Plotting {csv_path}")
    summarise("CO2", co2_v, "ppm", "{:.0f}")
    summarise("O2", o2_v, "%", "{:.2f}")

    fig, ax_co2 = plt.subplots(figsize=(10, 5))

    if co2_v:
        ax_co2.plot(co2_t, co2_v, color=CO2_COLOUR, linewidth=1.5, label="CO2")
    ax_co2.set_xlabel("Elapsed time (min)")
    ax_co2.set_ylabel("CO2 (ppm)", color=CO2_COLOUR)
    ax_co2.tick_params(axis="y", labelcolor=CO2_COLOUR)
    ax_co2.grid(True, alpha=0.3)

    if o2_v:
        ax_o2 = ax_co2.twinx()
        ax_o2.plot(o2_t, o2_v, color=O2_COLOUR, linewidth=1.5, label="O2")
        ax_o2.set_ylabel("O2 (%)", color=O2_COLOUR)
        ax_o2.tick_params(axis="y", labelcolor=O2_COLOUR)

    fig.suptitle(os.path.basename(csv_path))
    fig.tight_layout()

    png_path = os.path.splitext(csv_path)[0] + ".png"
    fig.savefig(png_path, dpi=150)
    print(f"Saved {png_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return png_path


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    show = "--save-only" not in sys.argv[1:]

    # Without a display, saving is the only thing that can work
    if show and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("No display detected — saving PNG only.")
        show = False
    if not show:
        matplotlib.use("Agg")

    try:
        csv_path = args[0] if args else find_latest_csv()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    if not os.path.isfile(csv_path):
        print(f"Error: no such file: {csv_path}")
        return 1

    try:
        plot(csv_path, show=show)
    except (ValueError, OSError) as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
