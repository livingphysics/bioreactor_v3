#!/usr/bin/env python3
"""
One-point calibration for the Atlas Scientific EZO-O2 sensor (I2C mode).

The EZO-O2 ships two-point factory calibrated (0% and 20.95%); that data is
permanent. After ~12 months a single-point recalibration to atmospheric O2 is
all that is normally needed — leave the sensor in open, still air until the
reading is stable, then run this script.

Calibration value by altitude (datasheet):
    sea level              20.95 %
    1,000 ft   (305 m)     20.1 %
    5,000 ft (1,524 m)     17.3 %
   10,000 ft (3,048 m)     14.3 %

Commands used (EZO-O2 datasheet, I2C mode):
    Cal,nn.nn   one-point calibration to nn.nn %O2   (1300 ms)
    Cal,0       zero-point calibration                (1300 ms)
    Cal,clear   delete custom calibration data        (300 ms)
    Cal,?       calibration state: 0 / 1 / 2 points   (300 ms)

Usage:
    python hardware_testing/calibrate_o2.py                 # cal to 20.95 %
    python hardware_testing/calibrate_o2.py 20.1            # cal to 20.1 %
    python hardware_testing/calibrate_o2.py --status        # query only
    python hardware_testing/calibrate_o2.py --zero          # 0% O2 point
    python hardware_testing/calibrate_o2.py --clear         # erase custom cal
    python hardware_testing/calibrate_o2.py --addr 0x6C -y  # skip the prompt
"""

import argparse
import os
import sys
import time

# Allow running as `python hardware_testing/calibrate_o2.py` from the repo root
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from atlas_o2 import DEFAULT_I2C_ADDRESS, AtlasO2, AtlasO2Error

# Processing delays in ms, from the datasheet's I2C calibration page
CAL_DELAY_MS = 1300
CAL_QUERY_DELAY_MS = 300

ATMOSPHERIC_O2 = 20.95

CAL_STATE = {
    "0": "not calibrated (factory calibration only)",
    "1": "single point calibration",
    "2": "two point calibration",
}


def cal_status(sensor):
    """Return the sensor's calibration state as a human-readable string."""
    response = sensor.command("Cal,?", processing_delay=CAL_QUERY_DELAY_MS)
    points = response.split(",")[-1].strip()  # "?Cal,1" -> "1"
    return f"{response} — {CAL_STATE.get(points, 'unknown')}"


def show_readings(sensor, count=5, interval=1.0):
    """Print a few readings so the operator can see whether O2 has settled."""
    values = []
    for i in range(count):
        try:
            value = sensor.read_o2()
            values.append(value)
            print(f"  reading {i + 1}/{count}: {value:.2f} %")
        except AtlasO2Error as e:
            print(f"  reading {i + 1}/{count}: error: {e}")
        if i < count - 1:
            time.sleep(interval)

    if len(values) >= 2:
        print(f"  spread over {len(values)} readings: {max(values) - min(values):.2f} %")
    return values


def main():
    parser = argparse.ArgumentParser(
        description="One-point calibration of the Atlas EZO-O2 sensor over I2C")
    parser.add_argument("percent", nargs="?", type=float, default=ATMOSPHERIC_O2,
                        help=f"O2 %% to calibrate to (default: {ATMOSPHERIC_O2})")
    parser.add_argument("--addr", default=hex(DEFAULT_I2C_ADDRESS),
                        help=f"I2C address (default: {hex(DEFAULT_I2C_ADDRESS)})")
    parser.add_argument("--zero", action="store_true",
                        help="Calibrate the zero point (sensor in 0%% O2) instead")
    parser.add_argument("--clear", action="store_true",
                        help="Delete custom calibration data and exit")
    parser.add_argument("--status", action="store_true",
                        help="Report calibration state and exit")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Do not prompt for confirmation")
    args = parser.parse_args()

    try:
        addr = int(args.addr, 16) if args.addr.lower().startswith("0x") else int(args.addr)
    except ValueError:
        print(f"Error: invalid address: {args.addr}")
        return 1

    sensor = AtlasO2(i2c_addr=addr)
    print(f"Atlas EZO-O2 at 0x{addr:02X}")

    try:
        print(f"Calibration state: {cal_status(sensor)}")

        if args.status:
            return 0

        if args.clear:
            if not args.yes and input("Delete custom calibration data? [y/N] ").strip().lower() != "y":
                print("Aborted.")
                return 1
            sensor.command("Cal,clear", processing_delay=CAL_QUERY_DELAY_MS)
            print("Custom calibration cleared (factory calibration is unaffected).")
            print(f"Calibration state: {cal_status(sensor)}")
            return 0

        if args.zero:
            command = "Cal,0"
            description = "zero point (0 % O2)"
        else:
            command = f"Cal,{args.percent:.2f}"
            description = f"{args.percent:.2f} % O2"

        print(f"\nCurrent readings before calibrating to {description}:")
        show_readings(sensor)

        if not args.yes:
            print(f"\nThe sensor must be sitting in {description} and reading steadily.")
            if input(f"Send {command}? [y/N] ").strip().lower() != "y":
                print("Aborted — nothing was written to the sensor.")
                return 1

        sensor.command(command, processing_delay=CAL_DELAY_MS)
        print(f"{command} accepted.")
        print(f"Calibration state: {cal_status(sensor)}")

        print("\nReadings after calibration:")
        show_readings(sensor, count=3)

    except AtlasO2Error as e:
        print(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nAborted.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
