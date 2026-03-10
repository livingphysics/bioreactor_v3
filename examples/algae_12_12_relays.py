"""
12 hour light/dark cycle for growing algae with relay schedule.

- 20% illumination, 20s measurements, 25C PID
- Relays 3 & 4: open for 60s at start, then 2 cycles of 5 days open /
  5 days closed (20 days total), then closed indefinitely.

Relay schedule timeline (relay_3 and relay_4):
  0s  – 60s    : ON  (initial flush)
  60s – 5d     : OFF (cycle 1 closed)
  5d  – 10d    : ON  (cycle 1 open)
  10d – 15d    : OFF (cycle 2 closed)
  15d – 20d    : ON  (cycle 2 open)
  20d –  ...   : OFF (indefinitely)
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
config.LOG_TO_TERMINAL = True
config.LOG_FILE = 'bioreactor.log'
config.USE_TIMESTAMPED_FILENAME = False

# Enable relays
config.INIT_COMPONENTS['relays'] = True

# Time constants
DAY = 86400  # seconds in a day

# Relay schedule for relay_3 and relay_4:
#   60s ON, then 5 days ON, 5 days OFF, 5 days ON, 5 days OFF, then OFF forever
relay_steps = [
    (60,      'relay_3', True),   # initial 60s flush
    (5*DAY,   'relay_3', False),  # cycle 1: closed 5 days
    (5*DAY,   'relay_3', True),   # cycle 1: open 5 days
    (5*DAY,   'relay_3', False),  # cycle 2: closed 5 days
    (5*DAY,   'relay_3', True),   # cycle 2: open 5 days
    (None,    'relay_3', False),  # stay closed indefinitely

    (60,      'relay_4', True),
    (5*DAY,   'relay_4', False),
    (5*DAY,   'relay_4', True),
    (5*DAY,   'relay_4', False),
    (5*DAY,   'relay_4', True),
    (None,    'relay_4', False),
]

# Initialize bioreactor
with Bioreactor(config) as reactor:
    jobs = [
        # Measure and record sensors every 20 seconds
        (partial(measure_and_record_sensors, led_power=15.0), 20, True),

        # Temperature PID controller - maintains 25.0C
        (partial(temperature_pid_controller, setpoint=25.0, kp=12.0, ki=0.015, kd=0.0), 5, True),

        # Ring light 12h on / 12h off cycle
        (partial(ring_light_cycle, color=(50, 50, 50), on_time=43200.0, off_time=43200.0), 1, True),

        # Relay schedule - check every 10 seconds
        (partial(relay_schedule, schedule=relay_steps), 10, True),
    ]

    reactor.run(jobs)
    print("Started scheduled jobs. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping bioreactor...")
        reactor.finish()
