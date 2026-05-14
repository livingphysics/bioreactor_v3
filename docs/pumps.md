# Pump System

This document describes how to configure and control the peristaltic pumps in
`bioreactor_v3`. Pumps are driven by Pololu Tic stepper-motor controllers over
USB using the `ticlib` Python library.

- Hardware driver: Pololu Tic (USB)
- Library: `ticlib.TicUSB`
- Init function: `src/components.py::init_pumps`
- Control functions: `src/io.py::change_pump`, `stop_pump`, `stop_all_pumps`
- Coordinated flow helpers: `src/utils.py::balanced_flow`,
  `independent_flow`, `chemostat_mode`, `turbidostat_ekf_mode`

## 1. Prerequisites

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   The relevant package for pumps is `ticlib`.
2. Plug each Tic controller in via USB and confirm it is enumerated.
3. Disable the Tic command timeout in `ticgui` (or via the Tic Control Center).
   Otherwise the Tic will halt the motor a few seconds after the last command,
   which breaks `balanced_flow` and continuous chemostat modes.
4. Read each Tic's serial number from `ticgui` (or `ticcmd --list`); you will
   need it for the config.

## 2. Configuration

Pumps are declared in the `Config.PUMPS` dictionary in
`src/config_default.py`. Each entry maps a pump name to its Tic settings:

```python
PUMPS = {
    'inflow': {
        'serial': '00473498',     # Tic USB serial number
        'step_mode': 2,            # 0..3 microstep setting
        'current_limit': 32,       # Tic current-limit code
        'direction': 'forward',    # 'forward' or 'reverse'
        'steps_per_ml': 10000000.0 # per-pump calibration
    },
    'outflow': {
        'serial': '00473497',
        'step_mode': 2,
        'current_limit': 32,
        'direction': 'forward',
        'steps_per_ml': 10000000.0,
    },
}
```

You must also enable pump initialization in `INIT_COMPONENTS`:

```python
INIT_COMPONENTS = {
    ...,
    'pumps': True,
}
```

Notes:
- `direction` flips the velocity sign in `change_pump`. Use it to make the
  motor move liquid the right way for how your tubing is routed.
- `steps_per_ml` is per-pump. Calibrate by running the pump for a known time
  and weighing/measuring the dispensed volume — see §6.
- The pair `'inflow'`/`'outflow'` is special-cased by `balanced_flow` and
  `independent_flow` for chemostat/turbidostat modes. Other names work for
  manual control but require explicit pairing.

### Overriding config in your script

```python
from src import Config
config = Config()
config.INIT_COMPONENTS['pumps'] = True
config.PUMPS = {
    'media_in':  {'serial': '00473498', 'step_mode': 2, 'current_limit': 32,
                  'direction': 'forward', 'steps_per_ml': 9_500_000.0},
    'waste_out': {'serial': '00473497', 'step_mode': 2, 'current_limit': 32,
                  'direction': 'forward', 'steps_per_ml': 9_700_000.0},
}
```

## 3. What happens at initialization

When the bioreactor starts and `INIT_COMPONENTS['pumps']` is True, for every
configured pump `init_pumps` does the following:

1. `TicUSB(serial_number=<serial>)`
2. `energize()` → `exit_safe_start()`
3. `set_step_mode(...)`, `set_current_limit(...)`
4. **Smoke test**: 3 seconds at velocity `2_000_000`, then `velocity = 0` and
   `deenergize()`.

If any pump fails to initialize it is skipped and the others continue.
After init, the bioreactor exposes:

- `bioreactor.pumps` — `dict[name, TicUSB]`
- `bioreactor.pump_configs` — `dict[name, settings]`
- `bioreactor.pump_direction` — `dict[name, 'forward'|'reverse']`

## 4. Basic control: `change_pump` / `stop_pump` / `stop_all_pumps`

Import from `src.io`:

```python
from src.io import change_pump, stop_pump, stop_all_pumps

# Run inflow at 1.5 ml/s in its configured direction
change_pump(reactor, 'inflow', ml_per_sec=1.5)

# Override direction at the call site
change_pump(reactor, 'outflow', ml_per_sec=2.0, direction='reverse')

# Stop a single pump
stop_pump(reactor, 'inflow')

# Stop everything (safe to call on shutdown)
stop_all_pumps(reactor)
```

Behavior:
- `ml_per_sec` must be ≥ 0. Negative rates raise `ValueError`; use `direction`
  to reverse.
- The conversion is `steps_per_sec = 8 * int(ml_per_sec * steps_per_ml / 8)`.
- Setting `ml_per_sec=0` calls `pump.deenergize()` (the motor goes limp).
- Any non-zero rate calls `energize()` → `exit_safe_start()` →
  `set_target_velocity(...)`.

## 5. Coordinated flow patterns

These helpers live in `src/utils.py` and assume an `inflow`/`outflow` pair (or
a `<base>_in[flow]`/`<base>_out[flow]` pair). They are designed to be passed
to `Bioreactor.run(jobs)` as scheduled jobs.

### `balanced_flow(reactor, pump_name, ml_per_sec, duration=None)`

Sets *both* pumps in the pair to the same `ml_per_sec`. If `duration` is
given, both pumps run for that duration and then stop; otherwise they run
continuously (this is the chemostat default and **requires the Tic command
timeout to be disabled**).

```python
from src.utils import balanced_flow
balanced_flow(reactor, 'inflow', ml_per_sec=2.0)            # continuous
balanced_flow(reactor, 'inflow', ml_per_sec=2.0, duration=5.0)  # 5 s pulse
```

### `independent_flow(reactor, pump_name, ml_per_sec, duration, converse_duration=None)`

Runs the primary pump for `duration` seconds, *then* the converse pump for
`converse_duration` seconds (defaults to `duration`). Used by the turbidostat
to dilute then drain — the outflow is typically run slightly longer
(`converse_duration = duration * 1.1`) to avoid overfilling.

The function spawns a daemon thread, sets `bioreactor.pumping_active = True`
while it runs, and clears it when done. Other jobs (e.g. the EKF) check
`pumping_active` to inflate measurement uncertainty during dilutions.

### `chemostat_mode(reactor, pump_name, flow_rate_ml_s, temp_setpoint=None, ...)`

A job-friendly wrapper that calls `balanced_flow` plus an optional PID
temperature controller.

### `turbidostat_ekf_mode(reactor, od_setpoint, pump_name='inflow', flow_rate_ml_s=2.0, pump_duration=5.0, ...)`

Reads the most recent OD row from the run's CSV, runs an EKF, and triggers
`independent_flow` when estimated OD exceeds `od_setpoint`. After each
dilution event it inflates `P[0,0]` for `pump_distrust_cycles` cycles to
absorb the OD discontinuity.

## 6. Calibrating `steps_per_ml`

Because the helper uses `steps_per_sec = 8 * int(ml_per_sec * steps_per_ml / 8)`,
larger `steps_per_ml` means more steps for the same commanded volume. To
calibrate a pump:

1. Place the inlet in a graduated cylinder (or weigh a beaker).
2. Run the pump at a known commanded rate for a known time:
   ```python
   change_pump(reactor, 'inflow', ml_per_sec=1.0)
   time.sleep(60.0)
   stop_pump(reactor, 'inflow')
   ```
3. Measure the actual volume `V_actual` (ml) dispensed.
4. Update `steps_per_ml`:
   `new_steps_per_ml = old_steps_per_ml * (60.0 / V_actual)`
5. Update the matching entry in `Config.PUMPS` and re-run.

Repeat per pump — peristaltic tubing varies enough that you should not assume
two pumps share a calibration.

## 7. Manual / hardware testing

For one-off tests outside the `Bioreactor` lifecycle, see the scripts in
`hardware_testing/`. None of them are pump-specific yet, but they show the
same pattern of opening a device, doing one thing, and closing.

For a quick sanity check inside an example run, copy the pump self-test from
`examples/example_usage.py`:

```python
if reactor.is_component_initialized('pumps') and reactor.pumps:
    name = 'inflow' if 'inflow' in reactor.pumps else next(iter(reactor.pumps))
    change_pump(reactor, name, ml_per_sec=2.0, direction='forward')
    time.sleep(2.0)
    change_pump(reactor, name, ml_per_sec=2.0, direction='reverse')
    time.sleep(2.0)
    change_pump(reactor, name, ml_per_sec=0.0)
```

## 8. Putting it together — minimal script

```python
import time
from src import Bioreactor, Config
from src.io import change_pump, stop_all_pumps

config = Config()
config.INIT_COMPONENTS = {k: False for k in config.INIT_COMPONENTS}
config.INIT_COMPONENTS['pumps'] = True
config.PUMPS = {
    'inflow':  {'serial': '00473498', 'step_mode': 2, 'current_limit': 32,
                'direction': 'forward', 'steps_per_ml': 10_000_000.0},
    'outflow': {'serial': '00473497', 'step_mode': 2, 'current_limit': 32,
                'direction': 'forward', 'steps_per_ml': 10_000_000.0},
}

with Bioreactor(config) as reactor:
    try:
        change_pump(reactor, 'inflow',  ml_per_sec=1.0)
        change_pump(reactor, 'outflow', ml_per_sec=1.0)
        time.sleep(30.0)
    finally:
        stop_all_pumps(reactor)
```

## 9. Troubleshooting

- **"No pumps configured"** — `Config.PUMPS` is empty or
  `INIT_COMPONENTS['pumps']` is False.
- **Pump initialization failed for serial X** — Tic is not enumerated. Check
  USB cable, `lsusb`, or that another process (e.g. an open `ticgui`) holds
  it. Try unplugging and replugging.
- **Pump runs for a few seconds then stops** — Tic command timeout is on.
  Disable it via `ticgui` (Settings → Command timeout: disabled) and save to
  the device.
- **Pump runs the wrong way** — flip `direction` in `Config.PUMPS`, or pass
  `direction=` to `change_pump`.
- **Volume delivered does not match commanded ml/s** — recalibrate
  `steps_per_ml` per §6.
- **`change_pump(... 0.0)` and motor still spins** — should not happen;
  zero-velocity calls `deenergize()`. If you see this, check that
  `ml_per_sec` rounded to 0 in `8 * int(ml_per_sec * steps_per_ml / 8)` and
  that no other job is re-issuing a non-zero rate.
