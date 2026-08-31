#!/usr/bin/env python3
"""
I2C reliability soak test for the CO2 (K33) + O2 (Atlas EZO) sensors.

Reads both sensors on a fixed interval and reports how many reads failed,
alongside the number of kernel-level "controller timed out" messages the Pi's
I2C controller logged during the run. Use it to compare bus settings — e.g.
before and after changing dtparam=i2c_arm_baudrate in /boot/firmware/config.txt.

Usage:
    python hardware_testing/i2c_soak.py [duration_s] [interval_s]

Example (6 minutes at 5s, the default):
    /home/david/bioreactor/bin/python hardware_testing/i2c_soak.py
"""

import os
import subprocess
import sys
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from atlas_o2 import AtlasO2, AtlasO2Error
from sensair_k33 import SenseairK33, SenseairK33Error


def kernel_timeouts():
    """Count 'controller timed out' lines currently in the kernel log."""
    try:
        out = subprocess.run(["dmesg"], capture_output=True, text=True).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    return out.count("controller timed out")


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 360.0
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0

    o2 = AtlasO2(i2c_addr=0x6C)
    k33 = SenseairK33(bus_num=1, i2c_addr=0x68)

    base = kernel_timeouts()
    cycles = co2_fail = o2_fail = 0
    last_error = None
    end = time.time() + duration

    print(f"Soaking for {duration:.0f}s at {interval:.1f}s intervals...")
    while time.time() < end:
        cycles += 1
        try:
            k33.read_co2()
        except SenseairK33Error as e:
            co2_fail += 1
            last_error = f"CO2: {e}"
        try:
            o2.read_o2()
        except AtlasO2Error as e:
            o2_fail += 1
            last_error = f"O2: {e}"
        time.sleep(interval)

    timeouts = kernel_timeouts() - base
    print(f"cycles={cycles} co2_failures={co2_fail} o2_failures={o2_fail} "
          f"kernel_timeouts=+{timeouts}")
    if cycles:
        print(f"failure rate: CO2 {100 * co2_fail / cycles:.1f}%, "
              f"O2 {100 * o2_fail / cycles:.1f}%")
    if last_error:
        print(f"last error: {last_error}")


if __name__ == "__main__":
    main()
