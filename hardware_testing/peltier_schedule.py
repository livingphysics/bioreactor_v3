"""
Generate / load a peltier excitation schedule for characterization data collection.

A *schedule* is an ordered list of steps, each ``(duty, direction, hold_s)``:
duty cycles are drawn from a discrete ladder, each level is held for a
uniform-random time in [min_hold, max_hold] (default 1-5 min), and successive
transitions deliberately alternate between *small* (adjacent-level) and *large*
(across-range) jumps so the resulting dataset contains both kinds of step.

Direction scope:
  - 'heat' : duty 0-70 %       (heat only)
  - 'cool' : duty 0, 50-100 %  (cool only; cooling is ineffective <50 % on this rig)
  - 'both' : a signed ladder spanning heat and cool, with the cooling dead-zone honoured

Duty caps mirror src/config.py (PELTIER_MAX_DUTY_HEAT=70, PELTIER_MAX_DUTY_COOL=100,
PELTIER_MIN_DUTY_COOL=50).

The schedule file is a CSV; lines starting with '#' are metadata comments and are
ignored on load:

    # peltier characterization schedule
    # scope=both total_s=10800 seed=42 ...
    duty,direction,hold_s
    0,heat,143.2
    30,cool,205.7
    ...

Used by run_peltier_schedule.py (standalone) and heater_gui.py ("Load Schedule").
"""

import csv
import random
from datetime import datetime

# Duty-level caps (%) — keep in sync with src/config.py peltier limits
DEFAULT_MAX_HEAT = 70
DEFAULT_MAX_COOL = 100
DEFAULT_MIN_COOL = 50
DEFAULT_STEP = 10  # duty-level granularity (%)

# peltier_forward encoding recorded by measure_and_record_sensors:
#   direction 'cool' -> peltier_forward = 1.0
#   direction 'heat' -> peltier_forward = 0.0


def build_ladder(scope='both', max_heat=DEFAULT_MAX_HEAT, max_cool=DEFAULT_MAX_COOL,
                 min_cool=DEFAULT_MIN_COOL, step=DEFAULT_STEP):
    """Return an ordered signed-duty ladder.

    Negative values are heat magnitudes, positive values are cool magnitudes,
    0 is 'off'. e.g. scope='both' -> [-70,-60,...,-10, 0, 50,60,...,100].
    """
    # Coerce to int so callers may pass float caps (e.g. an adaptive cap derived
    # from a float step duty); duties live on the `step`-multiple grid regardless.
    step = int(step)
    max_heat = int(max_heat)
    max_cool = int(max_cool)
    min_cool = int(min_cool)
    heat = list(range(step, max_heat + 1, step))                 # 10..70
    cool = list(range(max(min_cool, step), max_cool + 1, step))  # 50..100
    ladder = [0]
    if scope in ('heat', 'both'):
        ladder = [-h for h in reversed(heat)] + ladder
    if scope in ('cool', 'both'):
        ladder = ladder + cool
    if len(ladder) < 2:
        raise ValueError(f"scope={scope!r} produced too few duty levels: {ladder}")
    return ladder


def signed_to_command(v):
    """Map a signed ladder value to (duty, direction). 0 -> off (direction 'heat')."""
    if v < 0:
        return abs(v), 'heat'
    if v > 0:
        return v, 'cool'
    return 0, 'heat'


def infer_scope(steps):
    """Infer 'heat'/'cool'/'both' from the directions present in a step list."""
    dirs = {s['direction'] for s in steps if s['duty'] > 0}
    if dirs == {'heat'}:
        return 'heat'
    if dirs == {'cool'}:
        return 'cool'
    return 'both'


def tighten_cap(prev_cap, offending_duty, is_offending_dir, step=DEFAULT_STEP):
    """Return a strictly-lower duty cap for a direction after a temperature excursion.

    Honours "set the current duty cycle as the maximum" when the excursion happened
    while driving that direction, but always tightens by at least one ``step`` so a
    repeat excursion at the same level can't stall progress. Never returns below 0.
    """
    if is_offending_dir and offending_duty > 0:
        cap = offending_duty if offending_duty < prev_cap else prev_cap - step
    else:
        cap = prev_cap - step
    return max(0, cap)


def generate_schedule(scope='both', total_s=3 * 3600, min_hold_s=60.0, max_hold_s=300.0,
                      max_heat=DEFAULT_MAX_HEAT, max_cool=DEFAULT_MAX_COOL,
                      min_cool=DEFAULT_MIN_COOL, step=DEFAULT_STEP, seed=None,
                      small_frac=0.5):
    """Generate an APRBS-style schedule.

    Returns (steps, meta) where steps is a list of {'duty','direction','hold_s'}.
    Transitions are ~small_frac 'small' moves (<=2 ladder steps) and the rest
    'large' moves (>= n//3 ladder steps), guaranteeing both nearby and far-apart
    duty changes. The run starts from 'off'.
    """
    rng = random.Random(seed)
    ladder = build_ladder(scope, max_heat, max_cool, min_cool, step)
    n = len(ladder)
    far_thresh = max(3, n // 3)

    idx = ladder.index(0)  # start off
    steps = []
    t = 0.0
    n_small = n_large = 0
    while t < total_s:
        hold = rng.uniform(min_hold_s, max_hold_s)
        remaining = total_s - t
        # Trim the final step toward total_s, but never below min_hold_s: if the
        # remaining budget is smaller than min_hold_s, keep the full draw and let
        # the run overshoot slightly rather than emit a sub-minimum hold.
        if hold > remaining and remaining >= min_hold_s:
            hold = remaining
        duty, direction = signed_to_command(ladder[idx])
        steps.append({'duty': float(duty), 'direction': direction, 'hold_s': round(hold, 1)})
        t += hold

        # Pick the next level: small or large move (guaranteeing a mix)
        want_small = rng.random() < small_frac
        if want_small:
            candidates = [j for j in range(n) if j != idx and abs(j - idx) <= 2]
        else:
            candidates = [j for j in range(n) if j != idx and abs(j - idx) >= far_thresh]
        if not candidates:                          # edge fallback: any other level
            candidates = [j for j in range(n) if j != idx]
        nxt = rng.choice(candidates)
        if abs(nxt - idx) <= 2:
            n_small += 1
        else:
            n_large += 1
        idx = nxt

    meta = {
        'scope': scope, 'total_s': int(total_s),
        'min_hold_s': min_hold_s, 'max_hold_s': max_hold_s,
        'max_heat': max_heat, 'max_cool': max_cool, 'min_cool': min_cool, 'step': step,
        'seed': seed, 'n_steps': len(steps),
        'sum_hold_s': round(sum(s['hold_s'] for s in steps), 1),
        'n_small_moves': n_small, 'n_large_moves': n_large,
        'ladder': ladder,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    return steps, meta


def write_schedule(path, steps, meta=None):
    """Write steps to a CSV schedule file, with optional '#' metadata comments."""
    with open(path, 'w', newline='') as f:
        if meta:
            f.write("# peltier characterization schedule\n")
            f.write("# " + "  ".join(f"{k}={v}" for k, v in meta.items() if k != 'ladder') + "\n")
            f.write(f"# ladder={meta.get('ladder')}\n")
        w = csv.writer(f)
        w.writerow(['duty', 'direction', 'hold_s'])
        for s in steps:
            w.writerow([s['duty'], s['direction'], s['hold_s']])


def load_schedule(path, max_heat=DEFAULT_MAX_HEAT, max_cool=DEFAULT_MAX_COOL):
    """Load a schedule CSV (ignoring '#'-comment lines). Returns list of step dicts.

    Validates each step: a 'heat'/'cool' direction, a positive hold, and a duty in
    [0, cap] where the cap is the peltier safety limit for that direction
    (``max_heat`` for heat, ``max_cool`` for cool). Raises ValueError on a
    malformed or unsafe file — so a hand-edited or foreign schedule can never
    drive the peltier past its documented limits. Also raises if the file parses
    to zero steps (e.g. a header with no data rows).
    """
    with open(path, newline='') as f:
        data_lines = [ln for ln in f if not ln.lstrip().startswith('#')]
    if not data_lines:
        raise ValueError(f"Schedule file {path!r} has no data rows")
    reader = csv.DictReader(data_lines)
    required = {'duty', 'direction', 'hold_s'}
    if not required.issubset(set(reader.fieldnames or [])):
        raise ValueError(f"Schedule {path!r} must have columns {sorted(required)}, got {reader.fieldnames}")
    steps = []
    for i, r in enumerate(reader, 1):
        try:
            duty = float(r['duty'])
            direction = (r.get('direction') or 'heat').strip().lower()
            hold_s = float(r['hold_s'])
        except (TypeError, ValueError) as e:
            raise ValueError(f"Schedule {path!r} row {i}: bad numeric value ({e})") from None
        if duty < 0:
            raise ValueError(f"Schedule {path!r} row {i}: negative duty {duty}")
        if direction not in ('heat', 'cool'):
            raise ValueError(f"Schedule {path!r} row {i}: direction must be 'heat'/'cool', got {direction!r}")
        cap = max_heat if direction == 'heat' else max_cool
        if duty > cap:
            raise ValueError(
                f"Schedule {path!r} row {i}: {direction} duty {duty:.0f}% exceeds the "
                f"{cap:.0f}% safety cap — refusing to load (fix the schedule, or raise the cap "
                f"via load_schedule(max_{'heat' if direction == 'heat' else 'cool'}=...)).")
        if hold_s <= 0:
            raise ValueError(f"Schedule {path!r} row {i}: hold_s must be > 0, got {hold_s}")
        steps.append({'duty': duty, 'direction': direction, 'hold_s': hold_s})
    if not steps:
        raise ValueError(f"Schedule file {path!r} has column headers but no data rows")
    return steps


def summarize(steps):
    """Return a short human-readable summary string for a list of steps."""
    if not steps:
        return "empty schedule"
    holds = [s['hold_s'] for s in steps]
    total = sum(holds)
    duties = sorted({(s['direction'], s['duty']) for s in steps})
    # transition magnitudes between consecutive duty commands (signed)
    def signed(s):
        return -s['duty'] if s['direction'] == 'heat' else s['duty']
    jumps = [abs(signed(steps[i]) - signed(steps[i - 1])) for i in range(1, len(steps))]
    return (
        f"{len(steps)} steps, total {total/3600:.2f} h "
        f"({total:.0f} s); hold min/mean/max = {min(holds):.0f}/{total/len(steps):.0f}/{max(holds):.0f} s; "
        f"levels used = {len(duties)}; "
        f"transition |Δduty| min/mean/max = {min(jumps):.0f}/{sum(jumps)/len(jumps):.0f}/{max(jumps):.0f} %"
    )


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="Generate a peltier characterization schedule CSV.")
    ap.add_argument('output', help="output schedule .csv path")
    ap.add_argument('--scope', choices=['heat', 'cool', 'both'], default='both')
    ap.add_argument('--hours', type=float, default=3.0, help="approximate total duration (default 3)")
    ap.add_argument('--min-hold', type=float, default=60.0, help="min hold seconds (default 60)")
    ap.add_argument('--max-hold', type=float, default=300.0, help="max hold seconds (default 300)")
    ap.add_argument('--seed', type=int, default=None, help="RNG seed for a reproducible schedule")
    args = ap.parse_args()

    steps, meta = generate_schedule(
        scope=args.scope, total_s=args.hours * 3600.0,
        min_hold_s=args.min_hold, max_hold_s=args.max_hold, seed=args.seed,
    )
    write_schedule(args.output, steps, meta)
    print(f"Wrote {args.output}")
    print(f"  ladder: {meta['ladder']}")
    print(f"  {summarize(steps)}")
    print(f"  small/large moves: {meta['n_small_moves']}/{meta['n_large_moves']}")
