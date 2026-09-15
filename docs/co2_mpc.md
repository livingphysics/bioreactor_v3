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

## Identify the rig before enabling control

1. Keep flow, stirring, tubing, gas regulator and reactor volume fixed. Start from
   a stable baseline. Stop other writers to the CO₂ valve.
2. Use the API repository's `tools/co2_response.py` for one guarded pulse. It logs
   baseline, injection time and the subsequent rise/decay to CSV. It never repeats
   an injection automatically and never retries an uncertain POST. The Pi's API
   owns closure if the measurement client disconnects.
3. Observe through the peak and a clearly resolved decline. Repeat at several
   pulse lengths, including the proposed minimum. A single response cannot prove
   the smallest reliable solenoid pulse. Repeatability matters as well as mean gain.
4. Fit offline: `python -m tools.fit_co2 response.csv --output profile.json`.
   This needs NumPy. The fit includes any pre-existing mixing at recording start,
   assumes external CO₂ is 420 ppm unless `--ambient` is supplied, and reports
   unresolved decay/pulse-resolution warnings. `dose_s` must be actual energized
   duration for precise calibration; the API measurement utility records requested
   durations, so account for closure latency when assessing short pulses.
5. Validate on separate pulses and near the intended operating range. Review the
   fit, residuals, minimum pulse, gain margin and predicted overshoot before setting
   `validated: true`. The fitter deliberately never does this automatically.

## Configuration

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

Only I²C, the CO₂ sensor and relays initialize; pumps/Peltier are not initialized.
Ctrl-C, SIGTERM, expiry, invalid/stale readings, timing overruns and exceptions
stop the worker and de-energize the valve. GPIO write failures become faults.
Software cannot guarantee closure after power loss, process kill, or failed
relay hardware; the gas valve must be normally closed when de-energized.

## API and programs

With a validated profile installed and service restarted:

- `POST /api/co2/control`: `{"target_ppm":50000,"duration_s":3600}`.
- `POST /api/co2/stop`: stop and de-energize the valve.
- `GET /api/co2/controller`: active state, owner, fault, target and last forecast.
- `/api/state` also includes `co2_control`; detailed decisions go to the API log.

After API startup/manual dosing/stopping, wait delay + five mixing constants
before restarting control, because the prior pending-gas state is unknown. Live
setpoint changes by the same owner preserve that state. Manual ON is rejected
while MPC owns the valve; immediate OFF always stops it. A timed OFF that would
reopen the CO₂ valve later is rejected. API shutdown and run-stop also close it.
GPIO closure does not share the I²C lock, so sensor timeouts cannot delay it.

Paste/upload this JSON in the dashboard's existing Program panel:

```json
{
  "name": "CO2 at 5 percent",
  "duration": "1h",
  "tracks": [{"name": "CO2", "steps": [{"co2": 50000}]}]
}
```

`{"co2":false}` stops the program's CO₂ controller. Sequential numeric CO₂ steps
change its target. A CO₂ track and a `relay` track named `CO2` cannot coexist;
they own the same device. End-of-track, program stop/completion/abort and sensor
fault close the valve. The existing temperature supervision still applies to
API programs. For a gas-only rig use the standalone script or direct CO₂ endpoint.

## Checks

`python -m unittest test_co2_mpc` exercises delayed/leaky closed-loop tracking,
pending-dose suppression, quantization, cooldown, bad samples and worker teardown.
`test_co2_fit` checks fitting against synthetic known parameters (NumPy required).
The API repository tests cover program conflict/stop semantics and authenticated
simulation endpoints. These establish software behavior, not a calibrated rig.
