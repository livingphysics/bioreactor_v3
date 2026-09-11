# bioreactor_v3

Hardware drivers and control routines for the Living Physics bioreactor.

This is the **lowest layer** of a three-repo system, and the source of truth for
anything that touches a bus:

| Repo | Role |
|---|---|
| **`bioreactor_v3`** (this repo) | drivers, sensor/actuator I/O, control routines. Used directly by scripts, and as a git submodule by the API. |
| [`bioreactor_api`](https://github.com/livingphysics/bioreactor_api) | FastAPI server on the Pi — REST endpoints, the 1 Hz run loop, safety cutoffs, background samplers. Pins this repo as a submodule. |
| [`bioreactor_server`](https://github.com/livingphysics/bioreactor_server) | remote dashboard on a droplet. Pure proxy onto the Pi API. |

Two ways to use it: write a **script** against `Bioreactor` + `run(jobs)` (see
`examples/`), or let the **API** drive it (the API supplies its own control loop
and bus locking, and calls into `io.py` / `utils.py` here).

## Layout

```
src/
├── bioreactor.py       # Bioreactor class: init, component registry, job scheduler, CSV logging
├── components.py       # COMPONENT_REGISTRY + init_* for each piece of hardware
├── io.py               # the verb layer — every read/write to a sensor or actuator
├── utils.py            # composite routines: PID, profiles, chemostat/turbidostat modes
├── optics.py           # resolves VOLTAGE_SOURCES + OD_MEASUREMENTS into bioreactor.optics
├── config_default.py   # the Config class, with defaults for every driver setting
├── config.py           # GITIGNORED, per-rig — see "Configuration" below
└── bioreactor_data/    # run output: CSV + a copy of the script + run_config.json

docs/
├── optics.md           # voltage sources & OD measurements — read before touching optics
└── pumps.md            # pump config, calibration, and the coordinated flow modes

examples/               # runnable scripts: chemostat, 12/12 algae, EKF turbidostat
hardware_testing/       # ~28 standalone bench utilities and Tk GUIs (heater_gui, od_gui, ...)
analysis/               # offline plotting of recorded runs
test_optics.py          # the repo's test suite (optics config resolution)
plot_csv_data.py        # plot a run CSV
```

## Configuration

`src/config.py` is **gitignored and per-rig**, but `src/__init__.py` does
`from .config import Config` — so **a fresh clone cannot import `src` until you
create it**:

```bash
cp src/config_default.py src/config.py   # then edit for your rig
```

When this repo is used as the API's submodule, that file is instead a symlink to
the API's `config.py`, so both layers read one file:

```bash
ln -sfn ../../config.py bioreactor-api/bioreactor_v3/src/config.py
```

`INIT_COMPONENTS` decides what comes up. Anything `False` (or whose init raises)
is simply absent — `is_component_initialized(name)` returns False and the drivers
skip it rather than failing:

```python
# config_default.py defaults — a rig's config.py enables what it actually has
INIT_COMPONENTS = {
    'i2c': True, 'temp_sensor': True, 'peltier_driver': True, 'stirrer': True,
    'led': True, 'ring_light': True, 'optical_density': True,
    'eyespy_adc': False, 'co2_sensor': False, 'o2_sensor': False,
    'ambient_temp': False, 'peltier_current': False, 'pumps': False, 'relays': False,
}
```

Settings worth knowing before you drive hardware:

| Setting | Default | Why it matters |
|---|---|---|
| `PELTIER_MAX_DUTY_HEAT` | 70.0 | hardware safety ceiling for heating |
| `PELTIER_MAX_DUTY_COOL` | 100.0 | ceiling for cooling |
| `PELTIER_MIN_DUTY_COOL` | 50.0 | cooling is ineffective below this, so any non-zero cooling demand jumps straight here |
| `PELTIER_DIR_INVERTED` | False | **wiring-specific.** Wrong value makes a call for heat cool the bath, and the loop runs away from setpoint. Verify empirically. |
| `PUMPS[*].steps_per_ml` | — | **calibrate per pump** (`docs/pumps.md` §6) |

> The API's `config.example.py` is a **standalone** `class Config` — it does not
> inherit from `config_default.py`, and currently omits 15 settings defined here
> (including `PELTIER_MIN_DUTY_COOL`). Omitted settings fall back to whatever
> default the call site passes to `getattr`, which is not always the value in this
> file. When adding a setting here, add it to that template too.

## Install

```bash
pip install -r requirements.txt
```

Most of it (`lgpio`, `adafruit-circuitpython-*`, `pi5neo`, `smbus2`, `ticlib`)
only works on a Raspberry Pi with I2C/GPIO enabled. `numpy` and `matplotlib` are
needed anywhere — including by `temperature_pid_controller`, so the API layer
needs this file installed too, not just its own `requirements.txt`.

## Using it from a script

`Bioreactor(config)` initializes every enabled component, opens a run CSV, and
gives you a context manager. `run(jobs)` takes `(function, period_s, duration_s)`
tuples and starts each on its own daemon thread; each function is called as
`func(bioreactor, elapsed=<seconds since start>)`. `True` means "continuous" for
period and "indefinite" for duration.

```python
from functools import partial
from src import Bioreactor, Config
from src.utils import measure_and_record_sensors, temperature_pid_controller

with Bioreactor(Config()) as reactor:
    jobs = [
        (measure_and_record_sensors, 10, True),                                  # log a CSV row every 10 s
        (partial(temperature_pid_controller, setpoint=30.0), 5, True),           # hold 30 °C
    ]
    reactor.run(jobs)        # non-blocking
    time.sleep(3600)
```

Output lands in `src/bioreactor_data/<timestamp>/` — the CSV, a copy of the
script that produced it, and `run_config.json`. See `examples/` for complete
scripts (chemostat, 12/12 light cycling, EKF turbidostat).

> **`run()` does not serialize bus access.** Jobs run on parallel threads with no
> locking, which is fine for a script whose jobs read different hardware, but is
> why the API layer wraps every call in its own `HARDWARE_LOCK`. If you write
> multiple jobs that touch the same I2C device, serialize them yourself.

## Temperature control

`utils.temperature_pid_controller(bioreactor, setpoint, current_temp=None, kp=12.0, ki=0.015, kd=0.0)`
is the PID used by both the scripts and the API.

```
error  = setpoint - current_temp
output = kp*error + ki*∫error·dt + kd*d(error)/dt      # derivative EMA-filtered, alpha=0.7
```

`output > 0` → **heat** at `min(|output|, PELTIER_MAX_DUTY_HEAT)`;
`output ≤ 0` → **cool** at `clamp(|output|, PELTIER_MIN_DUTY_COOL, PELTIER_MAX_DUTY_COOL)`.

Things to know before you tune it:

- **State lives on the `Bioreactor` instance**, not in a PID object:
  `_temp_integral`, `_temp_last_error`, `_temp_last_time`, `_temp_last_derivative`.
  Delete those four attributes to reset the loop between runs — nothing does it
  for you except the API's `start_pid` / `start_program`.
- **`dt` is the gap since the last call**, measured from `_temp_last_time`. If you
  stop calling the PID for a while and then resume, the first call integrates
  `error × <the whole gap>`. Pass an explicit `dt=` if your job can be suspended.
- **There is no anti-windup.** The integral accumulates unclamped (deliberately —
  "pure PID") while the output saturates at the duty ceilings, so sustained
  saturation winds it up with nothing to bleed it off but overshoot.
- **There is no deadband**, and cooling has a floor. With
  `PELTIER_MIN_DUTY_COOL = 50`, an `output` of −0.01 commands 50 % cooling.

`utils.temperature_profile(...)` runs a `[(duration_s, setpoint), ...]` sequence
over the same PID, holding the last setpoint indefinitely.

## Other control routines

In `utils.py`, all usable as `run()` jobs via `functools.partial`:

- `measure_and_record_sensors` — read everything enabled, write one CSV row
- `relay_schedule`, `ring_light_cycle` — timed relay / illumination patterns
- `balanced_flow`, `independent_flow` — coordinated inflow/outflow
- `chemostat_mode`, `chemostat_duty_mode`, `chemostat_schedule` — continuous culture
- `turbidostat_ekf_mode` — OD-setpoint turbidostat with an EKF state estimate

`docs/pumps.md` covers all the flow modes in detail.

## Optics

Two layers between the ADCs and anything that consumes "optical density": named
**voltage sources** (an ADS1115 channel or an eyespy board) and the four canonical
**OD measurements** (`OD_45` / `OD_ref` / `OD_90` / `OD_135`), each fed by one
source. Both resolve once at construction into `bioreactor.optics`.

Read **[`docs/optics.md`](docs/optics.md)** before changing `VOLTAGE_SOURCES` or
`OD_MEASUREMENTS` — it covers naming rules, shorthand forms, how CSV columns are
chosen, and how legacy configs keep working.

## Bench utilities

`hardware_testing/` holds ~28 standalone scripts for bringing up and checking
hardware, independent of any run:

| Script | Purpose |
|---|---|
| `heater_gui.py` | Tk GUI: live temp/current plot, manual duty, PID, schedule runner. The API's run loop is a port of this. |
| `od_gui.py`, `eyespy_adc.py` | optical readout |
| `co2_gui.py`, `sensair_k33.py`, `atlas_o2.py` | gas sensors |
| `pump_calibration.py`, `clean_pumps.py` | pump `steps_per_ml` calibration and cleaning |
| `relay_gui.py`, `actuate_relays.py` | relay control |
| `peltier_schedule.py`, `heater_sweep.py` | open-loop peltier characterization |
| `i2c_soak.py` | bus stability soak test |

## Tests

Hardware-free `unittest` suite, run from the repo root:

```bash
python3 -m unittest test_optics -v
```

28 tests covering optics config resolution and the CSV label/column behaviour
that depends on it. It never imports the ADS1115/eyespy drivers and creates a
temporary `src/config.py` if one is missing, so it runs on a fresh clone off a
Pi. The 14 integration tests need `numpy` (via `src/utils.py`) and **skip**
rather than fail without it — `pip install numpy` to run the full set.

That is currently the whole suite. The control routines in `utils.py` (PID,
chemostat modes) and the drivers in `io.py` are untested.
