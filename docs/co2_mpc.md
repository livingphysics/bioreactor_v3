# CO₂ control with timed valve pulses

`src/co2_mpc.py` contains the hardware-independent model and controller.
`src/co2_control.py` runs that same controller in a worker with injected sensor and
valve functions. The API and `examples/co2_mpc.py` use this worker. No controller
starts automatically at boot or after a fault. Concentrations are **ppm**:
5% = 50,000 ppm; 9.5% = 95,000 ppm.

## Model and optimization

A commanded pulse has effective duration `max(0, duration - valve_dead_s)`.
It adds `gain_ppm_per_s * effective_duration` ppm to a pending mixing compartment
after `delay_s`. Mixing is first order with time constant `mixing_s`; the measured
compartment leaks exponentially toward `ambient_ppm` at `leak_per_s`.
The controller retains issued pulses, including those still in transit, and
corrects its decaying baseline from fresh observations. It therefore does not
interpret a delayed response as a reason to keep opening the valve.

At each eligible fresh sample it predicts multiple future pulse moves, minimizes
tracking error (overshoot costs four times as much), and penalizes gas use. A
bounded two-sweep coordinate search chooses among zero and discrete pulse lengths.
Only the first pulse is executed; later choices are recomputed from new data.
This is approximate discrete MPC, without a global-optimum or robust-stability
guarantee. It follows the standard finite-horizon/receding-horizon formulation
described in the [do-mpc theory documentation](https://www.do-mpc.com/en/v4.1.0/theory_mpc.html),
but uses a small NumPy-free optimizer rather than adding a solver to the Pi.

The ceiling check uses extra gain on future/pending doses and a concentration
margin. This is conditional on model accuracy; a closed valve cannot remove gas
already injected. A lower target is reached only through leakage/consumption.
Concentration-dependent flow, gas uptake, circulation changes and regulator
changes can invalidate a model fitted at one operating condition.

Before each pulse, a second ceiling check adds the measured concentration to
an upper estimate of all gas still in transit, plus the proposed dose. It uses
`gain_safety_factor`, `delay_safety_factor` and `mixing_safety_factor`, with no
credit for future leakage and no subtraction of valve dead time. Delay and
mixing factors default to 2; gain must be set from an empirical upper bound.
This bound assumes the model's path fractions and the configured uncertainty
limits cover the real response; it is not protection against arbitrary model
errors, unrecorded doses or sensor faults. Broad uncertainty can slow tracking.
Do not infer validity merely because independent one-parameter simulations pass:
test combined gain/delay/loss errors as well.

### Optional second delayed mixing path

`GasModel` also accepts `slow_fraction` (0–1), `slow_delay_s` and `slow_mixing_s`.
An injection splits between the original path and this second path; both feed
the same measured compartment with the same loss coefficient. This can describe
a rapid rise followed by slower gas arrival. It does not identify the physical
cause of a plateau. The default fraction is zero, preserving existing profiles.

The horizon and restart settling guard cover **both** enabled paths. Long horizons
can use `prediction_step_s` and `planning_interval_s` for coarser prediction and
future valve scheduling while retaining the original sensor cadence and freshness
limit. Prediction step must lie between sensor cadence and minimum pulse interval;
planning interval cannot be shorter than the minimum pulse interval. The solver
still limits prediction points to 1,000 and planned moves to 60.

`tracking_time_s` optionally discounts tracking error farther into the future;
zero retains equal weighting. This helps a long transport horizon avoid delaying
near-term correction. The concentration ceiling is checked over the **entire**
horizon regardless of that discount. Set these values through simulation and
independent rig validation; no new profile is enabled automatically.

The controller also faults if fresh timestamped values remain within
`stuck_tolerance_ppm` (default 30 ppm) for `stuck_window_s` (600 s) while issued
gas predicts a rise of at least `stuck_expected_rise_ppm` (1,000 ppm). This detects
some frozen-value failures, not every possible sensor failure. Incorrect models
or valve failures can trigger the same fault and require investigation.

## Configuration

Keep the regulator, tubing, circulation and reactor contents consistent with the
conditions used to identify the model. Validate pulse response, valve-off decay
and minimum reliable pulse duration before enabling unrestricted control.

### Control profile

Both config templates default to `CO2_MPC = None`. Supply a dictionary shaped like:

```python
CO2_MPC = {
    'validated': False,  # change only after rig-specific validation
    'model': {
        'gain_ppm_per_s': 18000,  # ILLUSTRATIVE, not a calibration
        'delay_s': 90,
        'mixing_s': 60,
        'leak_per_s': 0.00015,
        'ambient_ppm': 420,
        'valve_dead_s': 0,
    },
    'settings': {
        'target_ppm': 50000, 'max_ppm': 95000, 'margin_ppm': 5000,
        'min_pulse_s': 0.25, 'max_pulse_s': 1.0, 'pulse_quantum_s': 0.05,
        'min_interval_s': 60, 'sample_s': 5, 'stale_s': 20,
        'horizon_s': 1800, 'gain_safety_factor': 1.5,
    },
}
```

The prediction horizon must cover delay + four mixing constants + one pulse
interval. Its bounded solver permits up to 1,000 samples and 60 future moves.
`min_pulse_s` must exceed the estimated valve dead time. The API checks the MPC
maximum pulse, minimum interval and concentration ceiling against `RELAY_SAFETY['CO2']`.
It never relaxes these existing guards. The standalone script uses the MPC limits;
its config must name the CO₂ relay `CO2` and enable the correct sensor type.

## Run standalone

Stop the API and other valve writers first. Copy the normal rig config to
`src/config.py`, including the fitted `CO2_MPC`, Senseair configuration and
`RELAYS['CO2']` GPIO pin. From the driver repository root:

```sh
python -m examples.co2_mpc --target 50000 --duration 3600 --log co2-run.jsonl
```

Use `--duration 0` to run until stopped when the profile permits indefinite control.
The API equivalent is `{"target_percent":2,"duration_s":0}`. Status reports
`indefinite: true` and `remaining_s: null`; all concentration, sensor, pulse and
ownership checks still apply. A restart does not automatically resume control.

Only I²C, the CO₂ sensor and relays initialize; pumps/Peltier are not initialized.
Ctrl-C, SIGTERM, expiry, invalid/stale readings, timing overruns and exceptions
stop the worker and de-energize the valve. GPIO write failures become faults.
Software cannot guarantee closure after power loss, process kill, or failed
relay hardware; the gas valve must be normally closed when de-energized.

## API and programs

With a validated profile installed and service restarted:

- `POST /api/co2/control`: `{"target_percent":2,"duration_s":3600}`.
- `POST /api/co2/stop`: stop and de-energize the valve.
- `GET /api/co2/controller`: active state, owner, fault, target and last forecast.
- `/api/state` also includes `co2_control`; detailed decisions go to the API log.

Without persistent dose history, API startup/manual dosing/stopping requires each
enabled path's delay + five mixing constants before restarting control, because
the prior pending-gas state is unknown. With persistence, compatible confirmed
doses are restored and the blanket restart wait is unnecessary. Live
setpoint changes by the same owner preserve that state. Manual ON is rejected
while MPC owns the valve; immediate OFF always stops it. A timed OFF that would
reopen the CO₂ valve later is rejected. API shutdown and run-stop also close it.
GPIO closure does not share the I²C lock, so sensor timeouts cannot delay it.

Paste/upload this JSON in the dashboard's existing Program panel:

```json
{
  "name": "CO2 at 2 percent",
  "duration": "1h",
  "tracks": [{"name": "CO2", "steps": [{"co2": {"percent": 2}}]}]
}
```

`{"co2":false}` stops the program's CO₂ controller. Sequential CO₂ steps
change its target. Percent objects use 2 for 2%; legacy numeric steps use ppm. A CO₂ track and a `relay` track named `CO2` cannot coexist;
they own the same device. End-of-track, program stop/completion/abort and sensor
fault close the valve. The existing temperature supervision still applies to
API programs. For a gas-only rig use the standalone script or direct CO₂ endpoint.

## Bounded trials with a provisional model

An explicitly supervised commissioning trial may retain `validated: false` and
add `"trial": {"enabled": true, "target_max_ppm": 10000, "max_duration_s": 7200}`
to a rig-specific profile. These are example trial limits, not a calibration.
The regular model, concentration ceiling, pulse, freshness and restart-settling
checks still apply. Supply a finite duration to the direct API endpoint or
standalone script. A target above the trial maximum, an excessive/absent duration,
or an indefinite program start is refused by default. Updating a timed trial
with another positive duration cannot extend its original deadline. Status distinguishes `trial_mode` from `model_validated`
and reports `remaining_s` and `restart_wait_s`.

Every fresh measurement corrects the observer's concentration estimate before
the next pulse is chosen. Previously commanded pulses remain in the prediction
until their delayed effect arrives. Optional uncertainty learning updates pulse
gain within a fixed safety bound; delay and leakage are not refitted online.
Assess independent responses before marking the model validated.

### Explicit indefinite operation

Validated profiles allow indefinite control. For a provisional profile, enable it
explicitly with `CO2_MPC['trial']['allow_indefinite'] = True`; this does **not** mark
the model validated or change its target, concentration, pulse or sensor limits.
`max_duration_s` still caps positive durations. Zero (or the internal/program
`None` duration) then removes the deadline. Changing an active run to zero retains
its observer and pending doses; changing from indefinite to a positive duration
starts a timer. Indefinite control continues after browser disconnection, stops on
Stop, shutdown or a latched fault, and is not automatically resumed after restart.

## Brief measurement gaps

The shared API/standalone worker inhibits dosing on missing, invalid or stale
measurements and on sensor I/O exceptions. It keeps its observer, pending doses,
owner and original deadline. `settings.missing_timeout_s` defaults to 30 seconds
from the last valid acquisition. An outage reaching that limit latches a fault
and closes the valve; starting a controller still requires a valid fresh sample.

After a gap, `settings.recovery_samples` (default 2, minimum 2) distinct fresh
acquisitions are required before dosing resumes. Cached repeats do not count.
Status exports `measurement_paused`, `recovery_samples` and `last_valid_age_s`.
Concentration limits, frozen-response detection and valve/solver failures retain
their stop behavior. The loop never uses a replacement or interpolated reading
to authorize a pulse. API supervisors must also tolerate brief sensor gaps,
otherwise they can stop the worker before its recovery policy takes effect.

## Optional response uncertainty and online gain learning

Add an `uncertainty` object **alongside** `model` and `settings` in a rig profile
(or the `CO2_MPC` dictionary). Omit it to retain the original single-model planner.
The shared worker applies it in standalone scripts, API control and API programs;
there is no separate front-end algorithm or new start command.

For a profile with nominal gain 20,000 ppm/s, an illustrative uncertainty
configuration is:

```json
"uncertainty": {
  "gain_min_ppm_per_s": 10000,
  "gain_max_ppm_per_s": 35000,
  "kinetics_factor": 1.5,
  "risk_weight": 0.5,
  "learning_rate": 0.3,
  "min_response_ppm": 200,
  "max_relative_rmse": 0.25
}
```

These numbers are illustrative, **not generic sensor defaults**.
They do not enable or validate a profile. Retain the rig's existing trial bounds,
pulse limits, safety factors, measurement timeout and external supervisor.

### Planning

The planner evaluates nine operating scenarios: low/current/high gain crossed
with fast/nominal/slow kinetics. Delay and mixing multiply by1/factor,1,factor;
leakage divides by the same factor. Scenario predictions anchor to the latest
fresh measurement, accounting for issued gas and its future arrival. The cost is
`(1-risk_weight)*mean(scenario costs) + risk_weight*worst scenario cost`.
Overshoot still costs four times undershoot. Only the first discrete move executes.
Status includes `scenario_peak_min_ppm`, `scenario_peak_max_ppm` and `scenario_count`
in the last eligible plan; during cooldown that plan is historical.

The fixed safety gain remains `model.gain_ppm_per_s * settings.gain_safety_factor`.
Learning never changes it, the no-leak pending-gas budget, or the delay/mixing
safety factors. The operating gain maximum cannot exceed this gain; the kinetics
factor cannot exceed the safety factors. The horizon must cover the slowest
operating scenario plus one planned move. The original conservative forecast
check is retained and all operating scenario forecasts must also stay below the
configured planner ceiling. These finite scenarios are not a proof against all
combinations of pulse gains, time constants, disturbances or sensor errors.

### Learning from isolated responses (default)

After a confirmed pulse, the engine captures the last fresh measured baseline
and actual electrical duration. It fits one response amplitude with the existing
kinetics, allowing for previously issued doses. It waits through
`model.settling_s() * kinetics_factor`, covering four mixing constants of the slow
operating scenario. A new pulse cancels an unfinished fit window. Cached readings
never count twice; a gap longer than `stale_s`, fewer than12 samples, unresolved
rise, or poor relative residual error prevents an update. At most2,048 observations
are retained per fit window. Brief missing measurements still follow the worker's
existing pause/recovery rules; control is not restarted to obtain a fit.

An accepted fit updates the current gain by `learning_rate` and the estimated
amplitude of that issued pulse. The observer is re-anchored without a state jump.
The operating range can expand to include80–120% of a fitted gain, but never
contracts within a run. Its maximum is capped at the fixed safety gain. An
otherwise acceptable estimate **above** that fixed gain faults the worker and
closes the valve. Stopping cannot remove already injected gas.

Estimates remain conditional on the fixed kinetics and baseline. Unmodelled gas
arrival, external disturbances and changing uptake can bias them. Learning does
not write config files, change a deadline or mark a model validated. Estimates
reset at a new run unless persistent dose history is enabled. `/api/co2/controller`
and standalone JSONL logs expose a
`response_uncertainty` object with current gain/range, immutable safety gain,
accepted-update count and the last fit's state, gain, RMSE and reason.

## Concentration-dependent loss and overlapping doses

The shared worker selects `NonlinearPulseMPC` in `src/co2_nonlinear.py` when
`model.loss_exponent` differs from 1 or `uncertainty.learning_mode` is `"window"`.
Direct Python callers should use `make_controller(model, settings, uncertainty)`
from `src.co2_mpc` to select the appropriate engine. Existing profiles retain the
original engine and isolated-response learner without configuration changes.

The optional loss law uses excess concentration `x = C - ambient_ppm`:

```
loss(C) = leak_per_s * loss_reference_ppm * (x / loss_reference_ppm) ** loss_exponent
dC/dt = arriving gas - loss(C)
```

For concentrations below ambient the loss reverses sign. `loss_exponent` defaults
to 1 (exponential decay) and must be between 1 and 3; `loss_reference_ppm` defaults
to 50,000 and is a reference **excess** concentration. `leak_per_s` is the fractional
loss rate at that reference. Fit these parameters from valve-off data covering
the intended concentration range. This describes net gas loss, including possible
uptake; it does not identify leakage separately from other processes. Extrapolation
above the measured range remains provisional.

For the nonlinear model, a gain learner must use window mode. Add these fields
to the existing `uncertainty` object (other required gain bounds still apply):

```python
'learning_mode': 'window',
'learning_window_s': 2700,
'learning_interval_s': 300,
```

The window must cover `model.settling_s() * kinetics_factor`; the update interval
must not exceed the window. The engine retains at most 10,000 observations and
prunes doses only after their mixing tails and learning windows have passed.
Every fit integrates measured loss over a rolling window and fits a common gain:

```
measured concentration change + integrated loss = gain * arrived valve seconds
```

Arrived valve seconds include **all** overlapping doses and residual arrival from
doses before the window. A new pulse therefore does not cancel learning. Fits
still require resolved excitation, at least 12 points and acceptable residuals.
A measurement gap longer than `stale_s` clears the learning window; the worker's
existing missing-reading pause, recovery and timeout policy remains in force.
Successive windows overlap, so accepted-update counts are not independent trials.

An accepted gain is smoothed by `learning_rate`. The operating range includes
the configured range and 80–120% of the current estimate/latest fit. It can recover
from earlier extreme estimates but cannot shrink inside the configured range.
The fixed safety gain is never learned or reduced; a resolved fit above it faults
control. Delay, mixing and loss parameters remain fixed. A wrong loss model can
bias the estimated gain or trigger this fault, so inspect held-out responses too.

Prediction integrates mixing-reservoir arrival and nonlinear decay instead of
adding independent impulse responses. Nine operating scenarios guide tracking;
a separate fixed-upper-gain forecast also checks the ceiling. The immediate
pending-gas check still assumes **no future loss** and uses inflated delay/mixing.
Learning cannot bypass either check. Status includes `learning_mode`, the loss
parameters, window-fit details and `safety_predicted_peak_ppm` when a plan is made.

### Dosing capacity and high setpoints

Maximum average delivery is approximately
`gain * (max_pulse_s - valve_dead_s) / min_interval_s` ppm/s. Compare this with
the fitted loss at the requested target. Capacity can be increased by shortening
the minimum interval while retaining a reliably resolved pulse length. Revalidate
overlapping responses and keep `planning_interval_s >= min_interval_s`; API relay
guards still apply. Capacity alone does not guarantee the target is reachable.

A target close to the upper limit may be blocked by the pulse's size, unarrived
gas, or forecast uncertainty even when nominal delivery exceeds loss. Raising
the target cap does not remove those limits. Establish high-concentration loss
and gain bounds and test the resulting controller before increasing a deployed
target cap. Do not disable the no-loss pending-gas guard merely to reach a target.

## Persistent dose history

Set `CO2_MPC_STATE_PATH` to a writable file on persistent local storage, for example:

```python
CO2_MPC_STATE_PATH = '/home/david/bioreactor-state/co2-controller-state.json'
```

The API config template sets this to `co2-controller-state.json` beside the rig's
resolved `config.py`. API and standalone must use **the same config/path** on the
same Pi. The driver default is `None`, preserving the previous behavior. Keep the
history outside Git; do not delete it to bypass a wait. Only one process can own
the file: an exclusive lock also prevents simultaneous API/standalone writers.
Stop the API before starting standalone control, as for other hardware ownership.

The journal stores confirmed pulse starts, actual durations and operating gain
estimates. After a service/process restart within the same Pi boot, a new Start
command reconstructs their remaining arrival and minimum-dose cooldown before
using a **new fresh measurement**. The pending-gas upper bound includes restored
doses. A clean stop no longer invents a full settling interval. The target,
deadline and active-run state are **not** resumed automatically, and unfinished
learning windows are discarded. A process crash between completed pulses can
restore the last confirmed journal without treating those doses as zero.

An in-progress marker is atomically written and fsynced before valve ON. The
confirmed pulse is saved after OFF; disk writes are outside the energized interval.
Write failures inhibit further injections. Failed or interrupted writes never
turn a missing dose into an assumed zero dose. Status exposes `dose_history`
(enabled, recovery reason, stored dose count, pending marker, storage error),
alongside the usual `restart_wait_s`.

A settling wait remains appropriate when the history is missing, corrupt,
incompatible with the model/limits/hardware, or indicates an interrupted pulse.
Manual API doses are conservatively marked untracked and require settling from
confirmed closure; they cannot silently bypass the journal. The initial install
therefore has one normal settling wait. A saved fallback deadline survives later
service restarts, so repeated restarts within the same boot do not reset that wait.

Timestamps use the Linux boot ID and monotonic clock, avoiding wall-clock/NTP
changes. A **Pi reboot** changes that clock, so it deliberately falls back to a
settling wait after confirmed closure. This implementation eliminates unnecessary
service-restart waits, not the reboot guard. Unsupported boot-clock identification
also uses that fallback.

All valve writers must participate. If a different program/version, disabled
persistence or physical intervention could have introduced unrecorded gas, archive
or invalidate the old journal and allow the full settling wait. Do not reuse an
old journal when re-enabling persistence after such operations. Changing models,
kinetic/pulse/safety settings or the recorded sensor/relay configuration causes
fallback automatically; changing only the target does not.
