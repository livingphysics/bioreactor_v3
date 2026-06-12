#!/usr/bin/env python3
"""
Pump cleaning cycle: flush the lines with ethanol, then water, then empty them.

You are prompted to change fluids between stages. All pumps run in their
configured (forward-flow) direction at 0.2 ml/s, which is 200 steps/s -- a Tic
target velocity of 2,000,000 at the default steps_per_ml of 10,000,000
(velocity = 8 * int(0.2 * 10,000,000 / 8) = 2,000,000; the Tic unit is steps per
10,000 s, so that is 200 steps/s).

Sequence:
    1. prompt: put all inlet lines in ETHANOL
    2. run   15 min   (push ethanol through every pump)
    3. pause 15 min   (soak, pumps off)
    4. prompt: move all inlet lines to WATER
    5. run   15 min   (rinse)
    6. pause 15 min   (soak, pumps off)
    7. prompt: disconnect the inlet lines (leave them in air)
    8. run    5 min   (empty the lines)

The Bioreactor class drives the pumps, but with data logging turned OFF
(config.DATA_LOGGING = False), so NO CSV or results files are written. (The
operational log, bioreactor.log, is still written -- that's a log, not data.)

Usage:
    python hardware_testing/clean_pumps.py
    python hardware_testing/clean_pumps.py --flush-min 10 --empty-min 3
"""

import argparse
import os
import sys
import time

# Add the repo root to the path so we can import the src package (this file
# lives in hardware_testing/), exactly like the other hardware_testing scripts.
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)

try:
    from src import Bioreactor, Config
    from src.io import change_pump, stop_all_pumps
except ImportError as e:
    print(
        "Could not import the bioreactor package. On the rig you need a local "
        "src/config.py (copy src/config_default.py to src/config.py and set your "
        f"real pump serial numbers).\nOriginal error: {e}",
        file=sys.stderr,
    )
    raise

CLEAN_SPEED_ML_S = 0.2  # 0.2 ml/s = 200 steps/s (Tic velocity 2,000,000 at default steps_per_ml)


def run_all_pumps(reactor, ml_per_sec: float) -> None:
    """Run every configured pump in its configured (forward-flow) direction."""
    for name in reactor.pumps:
        # No direction override: uses each pump's config direction so fluid flows
        # the normal forward way regardless of how that pump's motor is wired.
        change_pump(reactor, name, ml_per_sec=ml_per_sec)


def wait(minutes: float, label: str) -> None:
    """Sleep for `minutes`, printing remaining time each minute (Ctrl-C aborts)."""
    remaining = int(round(minutes * 60))
    print(f"  {label}: {minutes:g} min")
    while remaining > 0:
        step = min(60, remaining)
        time.sleep(step)
        remaining -= step
        if remaining > 0:
            print(f"    {label}: {remaining // 60}:{remaining % 60:02d} remaining")
    print(f"    {label}: done")


def prompt(message: str, no_prompt: bool) -> None:
    print("\n" + "=" * 70)
    print(message)
    print("=" * 70)
    if not no_prompt:
        try:
            input("Press Enter to continue (Ctrl-C to abort)... ")
        except EOFError:
            pass


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Clean the pumps with ethanol then water, then empty them.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--speed", type=float, default=CLEAN_SPEED_ML_S,
                   help="Pump speed in ml/s (0.2 = 200 steps/s).")
    p.add_argument("--flush-min", type=float, default=15.0,
                   help="Run minutes per flush stage (ethanol and water).")
    p.add_argument("--soak-min", type=float, default=15.0,
                   help="Pause minutes per soak stage (pumps off).")
    p.add_argument("--empty-min", type=float, default=5.0,
                   help="Run minutes for the final empty.")
    p.add_argument("--no-prompt", action="store_true",
                   help="Don't wait for Enter at the fluid-change prompts.")
    args = p.parse_args(argv)

    config = Config()
    # Only the pumps; no sensors, and crucially NO data file / results package.
    config.INIT_COMPONENTS = {k: False for k in Config.INIT_COMPONENTS}
    config.INIT_COMPONENTS['pumps'] = True
    config.DATA_LOGGING = False
    config.RESULTS_PACKAGE = False

    if not getattr(config, 'PUMPS', None):
        print("Config.PUMPS is empty; nothing to clean.", file=sys.stderr)
        return 1

    print("Pump cleaning cycle")
    print(f"  pumps: {list(config.PUMPS.keys())}")
    print(f"  speed: {args.speed} ml/s | flush {args.flush_min:g} min | "
          f"soak {args.soak_min:g} min | empty {args.empty_min:g} min")

    # Prompt for ethanol BEFORE constructing the Bioreactor, because pump init
    # briefly runs each pump (a Tic smoke test) -- nicer to have ethanol loaded.
    prompt("STEP 1/3: Place ALL pump inlet lines in ETHANOL (outlets to waste).",
           args.no_prompt)

    with Bioreactor(config) as reactor:
        if not reactor.is_component_initialized('pumps') or not getattr(reactor, 'pumps', None):
            print("ERROR: no pumps initialized (Tics not found?).", file=sys.stderr)
            return 1
        try:
            # --- Ethanol ---
            print("\nEthanol flush: running all pumps forward.")
            run_all_pumps(reactor, args.speed)
            wait(args.flush_min, "ethanol flush")
            stop_all_pumps(reactor)
            wait(args.soak_min, "ethanol soak (pumps off)")

            # --- Water ---
            prompt("STEP 2/3: Move ALL pump inlet lines to WATER.", args.no_prompt)
            print("\nWater rinse: running all pumps forward.")
            run_all_pumps(reactor, args.speed)
            wait(args.flush_min, "water rinse")
            stop_all_pumps(reactor)
            wait(args.soak_min, "water soak (pumps off)")

            # --- Empty ---
            prompt("STEP 3/3: DISCONNECT the inlet lines (leave them in air) to empty.",
                   args.no_prompt)
            print("\nEmptying: running all pumps forward to clear the lines.")
            run_all_pumps(reactor, args.speed)
            wait(args.empty_min, "emptying")
            stop_all_pumps(reactor)
        except KeyboardInterrupt:
            print("\nAborted; stopping pumps.")
        finally:
            stop_all_pumps(reactor)

    print("\nCleaning cycle complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
