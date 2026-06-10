"""
Yeast chemostat with a 24 h batch phase, then duty-cycled dilution.

Modeled on examples/yeast_ekf.py, but instead of the EKF turbidostat it runs the
duty-cycle chemostat (src/utils.py::chemostat_schedule):

    1. measure_and_record_sensors  - logs OD/temperature every 10 s. Its built-in
       standalone EKF keeps estimating OD, growth rate and doubling time; the
       chemostat duty modes deliberately do NOT claim the EKF, so it keeps
       tracking right through the dilution phase.
    2. temperature_pid_controller  - holds 30 C as a SEPARATE job (recommended,
       so the chemostat duty-cycle timing stays exact).
    3. chemostat_schedule          - dilution profile over time:
         - 0 dilution for 24 h (batch growth to density), then
         - duty 0.5 at 0.00174 ml/s held indefinitely.

Requirements (set in your src/config.py, same as yeast_ekf.py):
    - INIT_COMPONENTS['pumps'] = True
    - PUMPS has both an 'inflow' and an 'outflow' entry (the outflow pump is
      inferred from the inflow and run slightly longer for overfill protection).
    - Pumps calibrated (steps_per_ml) so 0.00174 ml/s is accurate -- see
      hardware_testing/pump_calibration.py.
"""

import time
import sys
import os
from functools import partial

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import Bioreactor, Config
from src.utils import *
from src.io import *

# Load default config
config = Config()

# Override some settings in the configuration
config.LOG_TO_TERMINAL = True  # Print logs to terminal (default: True)
config.LOG_FILE = 'bioreactor.log'  # Also log to file
config.USE_TIMESTAMPED_FILENAME: bool = False

# ─ Chemostat dilution settings ───────────────────────────────────────────────
# "Dilution rate" here is the pump speed while it is actually running (ml/s); the
# duty cycle is the fraction of each period the pump is on. The chemostat pulses
# the pump at DILUTION_FLOW_ML_S for `duty * period` seconds out of every period,
# so the time-averaged dilution is `duty * DILUTION_FLOW_ML_S`.
BATCH_SECONDS = 24 * 3600      # 24 h batch (no-dilution) phase
DILUTION_FLOW_ML_S = 0.00174   # pump speed while diluting (ml/s)
DILUTION_DUTY = 0.5            # on-fraction of each duty period
# Mean dilution once running: 0.5 * 0.00174 = 0.00087 ml/s  (~3.13 ml/h).
# (If you instead want the *mean* dilution to be 0.00174 ml/s, set
#  DILUTION_FLOW_ML_S = 0.00348 with DILUTION_DUTY = 0.5.)
#
# Note on very low flow rates: each "on" pulse lasts duty * period seconds (the
# default period is 1.0 s, so 0.5 s here). At slow speeds that is only a handful
# of motor microsteps per pulse; if you want larger, smoother pulses, pass a
# bigger `period=` (e.g. period=10.0) to chemostat_schedule below.

# Initialize bioreactor
with Bioreactor(config) as reactor:
    # Check if components are initialized
    if reactor.is_component_initialized('temp_sensor'):
        print("Temperature sensors are ready!")

    # Read all eyespy boards in a single call
    if reactor.is_component_initialized('eyespy_adc'):
        eyespy_readings = read_all_eyespy_boards(reactor)
        print(f"Eyespy readings: {eyespy_readings}")

    # Pre-job initialization check: ring light test
    if reactor.is_component_initialized('ring_light') and hasattr(reactor, 'ring_light_driver'):
        try:
            reactor.logger.info("Ring light initialization check: turning red for 2 seconds")
            reactor.ring_light_driver.set_color((255, 0, 0))  # Red
            time.sleep(2.0)
            reactor.ring_light_driver.off()
            reactor.logger.info("Ring light initialization check complete")
        except Exception as e:
            reactor.logger.error(f"Ring light initialization check failed: {e}")

    # Pre-job initialization check: pump test (forward and backward)
    if reactor.is_component_initialized('pumps') and hasattr(reactor, 'pumps'):
        try:
            # Use 'inflow' pump if available, otherwise use the first available pump
            pump_name = 'inflow' if 'inflow' in reactor.pumps else list(reactor.pumps.keys())[0] if reactor.pumps else None

            if pump_name:
                reactor.logger.info(f"Pump initialization check: running {pump_name} forward for 2 seconds at 2 ml/s")
                change_pump(reactor, pump_name, ml_per_sec=2.0, direction='forward')
                time.sleep(2.0)

                reactor.logger.info(f"Pump initialization check: running {pump_name} backward for 2 seconds at 2 ml/s")
                change_pump(reactor, pump_name, ml_per_sec=2.0, direction='reverse')
                time.sleep(2.0)

                # Stop the pump
                change_pump(reactor, pump_name, ml_per_sec=0.0)
                reactor.logger.info(f"Pump initialization check complete for {pump_name}")
            else:
                reactor.logger.warning("No pumps available for initialization check")
        except Exception as e:
            reactor.logger.error(f"Pump initialization check failed: {e}")

    # Start scheduled jobs
    # Format: (function, frequency_seconds, duration)
    # frequency: time between calls in seconds, or True for continuous
    # duration: how long to run in seconds, or True for indefinite
    jobs = [
        # Measure and record sensors every 10 s with IR LED at 15%. The standalone
        # EKF (OD estimate, growth rate, doubling time) runs inside this job and
        # keeps tracking through the chemostat phase.
        (partial(measure_and_record_sensors, led_power=15.0), 10, True),

        # Hold 30 C. Kept as a SEPARATE job (temp_setpoint left out of the
        # chemostat) so the duty-cycle timing stays exact.
        (partial(temperature_pid_controller, setpoint=30.0, kp=12.0, ki=0.015, kd=0.0), 5, True),

        # Chemostat dilution schedule: no dilution for 24 h, then duty 0.5 at
        # 0.00174 ml/s held indefinitely (None duration = hold the last step).
        # Must run continuously (True) -- each call consumes exactly one period.
        (partial(chemostat_schedule,
                 schedule=[
                     (BATCH_SECONDS, 0.0),            # 24 h batch: pumps off
                     (12*3600,  0.25),  # then 0.5 duty 
                     (None,          DILUTION_DUTY),  # then 0.5 duty forever
                 ],
                 flow_rate_ml_s=DILUTION_FLOW_ML_S),
         True, True),
    ]

    reactor.run(jobs)
    print("Started scheduled jobs. Press Ctrl+C to stop.")
    print(f"Batch phase: {BATCH_SECONDS / 3600:.0f} h with no dilution, then "
          f"duty {DILUTION_DUTY} at {DILUTION_FLOW_ML_S} ml/s "
          f"(mean {DILUTION_DUTY * DILUTION_FLOW_ML_S:.5f} ml/s).")

    # Keep the program running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping bioreactor...")
        reactor.finish()
