"""
Run a peltier excitation schedule and record a characterization dataset.

Drives the peltier through a schedule (loaded from a CSV, or generated on the
fly) while recording, once per sample period, the following columns via
measure_and_record_sensors:

    temperature_C      bath temperature (DS18B20)
    ambient_temp_C     ambient temperature (PCT2075)
    peltier_current_A  peltier supply current (INA228, unsigned magnitude)
    peltier_duty       commanded duty cycle 0-100 %
    peltier_forward    direction flag (1.0 = cool, 0.0 = heat)

(The CSV also carries empty ekf_* columns that measure_and_record always emits;
they stay blank because no OD sensor is enabled.)

Usage:
    # generate a fresh 3 h heat+cool schedule and run it (schedule is saved too):
    python run_peltier_schedule.py

    # run an existing schedule file:
    python run_peltier_schedule.py my_schedule.csv

    # generate with options and run:
    python run_peltier_schedule.py --scope both --hours 3 --seed 42

    # validate a schedule without touching hardware:
    python run_peltier_schedule.py my_schedule.csv --dry-run

Temperature-limit response (--on-limit, default 'adapt'):
    adapt : do NOT abort. Cap the offending direction's duty at the level that
            caused the excursion (so nothing that hot/cold is sampled again),
            hold 0 duty for --rest-minutes (default 5) to recover, then RESAMPLE
            the rest of the run from the uniform distribution with the tightened
            bounds. Repeats (tightening further) on subsequent excursions.
    abort : zero the peltier and stop the run (the original behaviour).

Safety (always): the run aborts if the bath temperature reads NaN (sensor
dropout / a value outside 0-100 C) for ~30 s straight, or on Ctrl-C. Either way
the peltier is zeroed and the data file is closed cleanly.

NOTE: only one process may own the peltier GPIO. Close heater_gui.py (or any
other controller) before starting this run.
"""

import argparse
import os
import sys
import time
from datetime import datetime

# Allow imports from src/ and from this directory (peltier_schedule)
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))  # repo root -> `src`
sys.path.insert(0, _here)                   # this dir -> `peltier_schedule`

from src import Bioreactor, Config
from src.io import set_peltier_power, stop_peltier
from src.utils import measure_and_record_sensors
from peltier_schedule import (
    generate_schedule, load_schedule, write_schedule, summarize,
    infer_scope, tighten_cap, DEFAULT_MAX_HEAT, DEFAULT_MAX_COOL, DEFAULT_STEP,
)


def _apply_step(bio, duty, direction):
    """Command the peltier for one schedule step (duty<=0 stops it)."""
    if duty <= 0:
        stop_peltier(bio)
    else:
        set_peltier_power(bio, duty, forward=direction)


def _sample_hold(bio, hold_s, sample_period_s, t0, have_temp, temp_min, temp_max,
                 nan_state, check_limits=True):
    """Record for hold_s seconds. Returns (status, last_temp).

    status is 'ok' | 'hot' | 'cold' | 'nan'. nan_state is a mutable [count, max]
    so the consecutive-NaN watchdog carries across segments. When check_limits is
    False (e.g. during the recovery rest) temperature limits are not evaluated,
    but the NaN watchdog still runs.
    """
    seg_end = time.time() + hold_s
    last_temp = float('nan')
    while time.time() < seg_end:
        data = measure_and_record_sensors(bio, elapsed=time.time() - t0)
        temp = data.get('temperature', float('nan'))
        last_temp = temp
        if have_temp:
            if temp != temp:  # NaN: read fault or reading outside 0-100 C
                nan_state[0] += 1
                if nan_state[0] >= nan_state[1]:
                    return 'nan', temp
            else:
                nan_state[0] = 0
                if check_limits and temp > temp_max:
                    return 'hot', temp
                if check_limits and temp < temp_min:
                    return 'cold', temp
        remaining = seg_end - time.time()
        if remaining <= 0:
            break
        time.sleep(min(sample_period_s, remaining))
    return 'ok', last_temp


def run(steps, bio, sample_period_s, temp_min, temp_max, *, on_limit='adapt',
        scope='both', min_hold_s=60.0, max_hold_s=300.0, seed=None,
        max_heat=DEFAULT_MAX_HEAT, max_cool=DEFAULT_MAX_COOL, rest_s=300.0, t0=None):
    """Drive `steps` on `bio`, recording every sample_period_s. Returns exit code.

    On a temperature-limit excursion with on_limit='adapt', tightens the offending
    direction's cap, rests at 0 duty for rest_s, and resamples the remaining
    duration with the new bounds instead of aborting.
    """
    have_temp = bio.is_component_initialized('temp_sensor')
    if not have_temp:
        print("WARNING: temp_sensor not initialized — temperature safety monitoring is DISABLED.")
    # get_temperature() returns NaN both on read faults and for readings outside
    # 0-100 C, so a genuinely out-of-range bath reads NaN. A sustained run of NaN
    # samples (~30 s) is treated as a fault and aborts, so the watchdog can't be blinded.
    max_nan_samples = max(3, int(round(30.0 / max(sample_period_s, 0.1))))
    nan_state = [0, max_nan_samples]
    t0 = t0 if t0 is not None else time.time()
    total_s = sum(s['hold_s'] for s in steps)     # nominal run length; kept across resamples
    cur_max_heat, cur_max_cool = max_heat, max_cool
    rebounds = 0

    print(f"Running ~{total_s/3600:.2f} h, sampling every {sample_period_s:.1f} s. "
          f"On temp-limit: {on_limit}.")
    print(f"Data -> {bio.out_file_path}")

    i = 0
    step_no = 0
    while i < len(steps):
        step = steps[i]
        duty, direction, hold = step['duty'], step['direction'], step['hold_s']
        _apply_step(bio, duty, direction)
        step_no += 1
        label = 'off' if duty <= 0 else f"{direction} {duty:.0f}%"
        print(f"[{step_no}] {label:<12} hold {hold:5.0f}s  (elapsed {(time.time()-t0)/60:5.1f} min)")

        status, temp = _sample_hold(bio, hold, sample_period_s, t0, have_temp,
                                    temp_min, temp_max, nan_state, check_limits=True)

        if status == 'nan':
            print(f"\nSAFETY: no valid bath temperature for ~{max_nan_samples * sample_period_s:.0f} s "
                  f"— aborting run.")
            return 2

        if status in ('hot', 'cold'):
            if on_limit == 'abort':
                print(f"\nSAFETY CUTOFF: bath {temp:.1f} °C outside "
                      f"[{temp_min:.0f}, {temp_max:.0f}] °C — aborting run.")
                return 2

            # --- adaptive response: tighten the offending cap, rest, resample ---
            rebounds += 1
            if status == 'hot':
                prev, cur_max_heat = cur_max_heat, tighten_cap(
                    cur_max_heat, duty, direction == 'heat', DEFAULT_STEP)
                print(f"\nSAFETY: bath {temp:.1f} °C > {temp_max:.0f} °C at {label}. "
                      f"Capping HEAT {prev:.0f}%→{cur_max_heat:.0f}%; resting 0 duty for {rest_s/60:.0f} min.")
            else:
                prev, cur_max_cool = cur_max_cool, tighten_cap(
                    cur_max_cool, duty, direction == 'cool', DEFAULT_STEP)
                print(f"\nSAFETY: bath {temp:.1f} °C < {temp_min:.0f} °C at {label}. "
                      f"Capping COOL {prev:.0f}%→{cur_max_cool:.0f}%; resting 0 duty for {rest_s/60:.0f} min.")
            try:
                bio.logger.info(f"Adaptive rebound #{rebounds}: max_heat={cur_max_heat}, "
                                f"max_cool={cur_max_cool}")
            except Exception:
                pass

            # Recovery: hold 0 duty for rest_s (recorded; NaN watchdog on, limits off)
            _apply_step(bio, 0, direction)
            rstatus, _ = _sample_hold(bio, rest_s, sample_period_s, t0, have_temp,
                                      temp_min, temp_max, nan_state, check_limits=False)
            if rstatus == 'nan':
                print("\nSAFETY: no valid bath temperature during recovery rest — aborting run.")
                return 2

            # Resample the remaining duration with the tightened bounds
            remaining_s = total_s - (time.time() - t0)
            if remaining_s < min_hold_s:
                print("Remaining time below one hold; ending run after rest.")
                break
            try:
                steps, _ = generate_schedule(
                    scope=scope, total_s=remaining_s,
                    min_hold_s=min_hold_s, max_hold_s=max_hold_s,
                    max_heat=cur_max_heat, max_cool=cur_max_cool,
                    seed=(seed + rebounds) if seed is not None else None,
                )
            except ValueError as e:
                print(f"Adaptive bounds left nothing safe to sample ({e}); ending run.")
                break
            print(f"Resampling {len(steps)} steps for the remaining {remaining_s/60:.0f} min "
                  f"(heat≤{cur_max_heat:.0f}%, cool≤{cur_max_cool:.0f}%).")
            i = 0
            continue

        i += 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('schedule', nargs='?', default=None,
                    help="schedule CSV to run (omit to generate a new one)")
    ap.add_argument('--scope', choices=['heat', 'cool', 'both'], default='both',
                    help="direction scope when generating (default both)")
    ap.add_argument('--hours', type=float, default=3.0, help="duration when generating (default 3)")
    ap.add_argument('--min-hold', type=float, default=60.0, help="min hold seconds (default 60)")
    ap.add_argument('--max-hold', type=float, default=300.0, help="max hold seconds (default 300)")
    ap.add_argument('--seed', type=int, default=None, help="RNG seed when generating")
    ap.add_argument('--sample-period', type=float, default=2.0, help="seconds between samples (default 2)")
    ap.add_argument('--temp-max', type=float, default=60.0, help="upper bath-temp limit (default 60)")
    ap.add_argument('--temp-min', type=float, default=2.0, help="lower bath-temp limit (default 2)")
    ap.add_argument('--on-limit', choices=['adapt', 'abort'], default='adapt',
                    help="response when bath temp exceeds a limit: 'adapt' (cap that direction, "
                         "rest at 0 duty for --rest-minutes, resample with new bounds) or 'abort' "
                         "(default adapt)")
    ap.add_argument('--rest-minutes', type=float, default=5.0,
                    help="minutes to hold 0 duty after an adaptive cap (default 5)")
    ap.add_argument('--dry-run', action='store_true',
                    help="validate/generate the schedule and print it; do NOT touch hardware")
    args = ap.parse_args()

    # --- Obtain the schedule ---------------------------------------------------
    generated_meta = None
    if args.schedule:
        steps = load_schedule(args.schedule)
        print(f"Loaded {len(steps)} steps from {args.schedule}")
    else:
        steps, generated_meta = generate_schedule(
            scope=args.scope, total_s=args.hours * 3600.0,
            min_hold_s=args.min_hold, max_hold_s=args.max_hold, seed=args.seed,
        )
        print(f"Generated schedule: {generated_meta['ladder']}")
    print("  " + summarize(steps))

    if args.dry_run:
        print("\n-- dry run: initial schedule steps --")
        t = 0.0
        for i, s in enumerate(steps, 1):
            lbl = 'off' if s['duty'] <= 0 else f"{s['direction']} {s['duty']:.0f}%"
            print(f"  [{i:>2}] t+{t/60:6.1f} min  {lbl:<12} {s['hold_s']:5.0f}s")
            t += s['hold_s']
        print("(dry run — no hardware touched; adaptive rebounding only happens at runtime "
              "when a real temperature excursion occurs)")
        return 0

    # --- Configure a data-logging bioreactor with just the needed sensors ------
    override = {k: False for k in Config.INIT_COMPONENTS}
    override['temp_sensor'] = True
    override['ambient_temp'] = True
    override['peltier_current'] = True
    override['peltier_driver'] = True
    Config.INIT_COMPONENTS = override
    Config.DATA_LOGGING = True
    Config.USE_TIMESTAMPED_FILENAME = True
    Config.RESULTS_PACKAGE = False
    Config.DATA_OUT_FILE = 'peltier_characterization.csv'

    bio = Bioreactor(Config)
    if not bio.is_component_initialized('peltier_driver'):
        print("ERROR: peltier driver failed to initialize — is heater_gui.py or another "
              "process holding the peltier GPIO? Close it and retry.")
        bio.finish()
        return 1
    for name in ('ambient_temp', 'peltier_current', 'temp_sensor'):
        if not bio.is_component_initialized(name):
            print(f"WARNING: '{name}' did not initialize; its column will be blank/NaN.")

    # Save the exact schedule that will run, alongside the dataset
    try:
        sched_path = os.path.splitext(bio.out_file_path)[0] + '_schedule.csv'
        write_schedule(sched_path, steps, generated_meta)
        print(f"Schedule saved to {sched_path}")
    except Exception as e:
        print(f"(could not save schedule copy: {e})")

    effective_scope = args.scope if not args.schedule else infer_scope(steps)

    rc = 0
    try:
        rc = run(steps, bio, args.sample_period, args.temp_min, args.temp_max,
                 on_limit=args.on_limit, scope=effective_scope,
                 min_hold_s=args.min_hold, max_hold_s=args.max_hold, seed=args.seed,
                 rest_s=args.rest_minutes * 60.0)
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl-C).")
        rc = 130
    finally:
        try:
            stop_peltier(bio)
        except Exception:
            pass
        bio.finish()
        print(f"Peltier off. Data written to {bio.out_file_path}")
    return rc


if __name__ == '__main__':
    sys.exit(main())
