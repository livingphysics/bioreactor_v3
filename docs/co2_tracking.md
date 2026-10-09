# Delayed CO2 tracking

A follower targets `master(t - delay_s)`. A delay covering the prediction horizon
makes the complete future reference known without forecasting algae behavior.
`ReferenceTrajectory` uses the controller's monotonic clock; convert source UTC
acquisition times once using a fixed UTC/monotonic anchor. Never compare two
machines' monotonic clocks. Reject clock jumps and gaps rather than extrapolating.

`PulseMPC.set_reference(trajectory)` enables per-horizon targets in the linear and
nonlinear planners. A flat reference preserves the constant-target behavior.
Concentration, pending-gas, minimum-pulse, spacing and uncertainty guards remain
in force. `CO2Control.start(..., owner='tracking', reference_provider=provider)`
refreshes the immutable trajectory before each observation/planning tick and
rechecks source readiness after planning and disk writes. The provider raises
`ReferenceUnavailable` if its source is stale or its history incomplete. Dosing
pauses while the local observer and dose history continue updating. A long local
sensor outage still faults normally. Reference bounds are checked independently
of any average-error offset. The provider must enforce any additional trial cap.

Average correction integrates the time-weighted **follower minus contemporaneous
reference error** through ordinary reference changes, and bounds the offset at
each requested concentration. A source outage clears its averaging window and
freezes its offset until a complete fresh window is available. Manual scalar
setpoint changes retain their previous reset behavior. A new run resets learning.
In reference mode `mean_ppm` is null; `mean_error_ppm` is the meaningful statistic.

For hardware operation use the API adapter documented in bioreactor_api. A
standalone caller can use the same worker/provider contract, with exclusive
hardware ownership and the existing dose journal. Neither creation of a reference
nor loading a journal starts control automatically.

## Offline replay

Run from this repository using a hardware-free Python environment with numpy.
Matplotlib is optional for figures.

```sh
python tools/co2_tracking_replay.py \
  --trace master-history.json --profile follower-profile.json \
  --delays-min 15 30 61 120 --hours 24 --initial-ppm 420 \
  --output replay-results
```

Trace input is API history JSON (`points` with `t` in UTC milliseconds and `co2`
in ppm); a bare points list also works. The profile is the `CO2_MPC` object, or a
snapshot containing `profile`. Missing readings remain gaps. The minimum delay
for complete preview is the configured prediction horizon; shorter candidates
are reported as insufficient rather than silently using unknown future data.

The simulator uses independent one-second reservoir/loss integration, with
five-second observations and a configurable planning cadence (default 300 s).
That slower cadence is an exploratory feasibility test, not production-loop
validation. Outputs include the declared-delay reference, simulated follower,
no-input baseline, pulse count, mean bias, MAE, RMSE, sampled 95th-percentile and
maximum absolute error, coverage, and rising/falling MAE. Errors use the declared
alignment; no best-fit time shift is applied. Time averages exclude measurement
gaps over 60 seconds. No N2 actuator is implemented.

## Commissioning limits

A delay cannot overcome a coarse dose or a slow passive decline. On 2026-10-09,
72 hours of the master algae trace spanned 498–1053 ppm. The existing follower
profile was fitted at much higher concentrations: its 0.125-second minimum pulse
represents 7250 nominal ppm of injected gas before mixing/loss. The first 24-hour
replay with that model selected no pulses (about 479 ppm RMSE), rather than produce
large overshoots. This is a model simulation, not a new injection test or a
validated low-concentration model. Recommission gas delivery and sensor agreement
at the observed range before enabling tracking actuation. If passive decline is
insufficient, separately calibrate a flush actuator; N2 also affects O2.
