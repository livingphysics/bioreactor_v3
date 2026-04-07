"""
12 hour light dark cycle for growing algae 20% illumination, 20s measurements

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


# Initialize bioreactor
with Bioreactor(config) as reactor:
    # Single run functions

    # Start scheduled jobs
    # Format: (function, frequency_seconds, duration)
    # frequency: time between calls in seconds, or True for continuous
    # duration: how long to run in seconds, or True for indefinite
    jobs = [
        # Measure and record sensors every 20 seconds
        (partial(measure_and_record_sensors, led_power=15.0), 20, True),  # Read sensors and record to CSV every 5 seconds
        
        # Temperature PID controller - maintains temperature at 25.0°C
        # Run PID controller every 5 seconds
        (partial(temperature_pid_controller, setpoint=25.0, kp=12.0, ki=0.015, kd=0.0), 5, True),
        
        # Ring light cycle - turns on at (50,50,50) for 12h, then off for 12h, repeating
        # Check every 1 second to update state
        (partial(ring_light_cycle, color=(50, 50, 50), on_time=43200.0, off_time=43200.0), 1, True),
        
    ]
    
    reactor.run(jobs)
    print("Started scheduled jobs. Press Ctrl+C to stop.")
    
    # Keep the program running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping bioreactor...")
        reactor.finish()
    
    # Your bioreactor code here...

