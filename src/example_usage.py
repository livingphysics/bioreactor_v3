"""
Example usage of the modular bioreactor system.

This shows how to:
1. Configure which components to use
2. Initialize the bioreactor
3. Use the components
4. Schedule recurring jobs
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
config.INIT_COMPONENTS = {
    'i2c': True,
    'temp_sensor': True,
    'peltier_driver': True,
    'stirrer': True,
    'led': True,  # Enable LED PWM control
    'ring_light': True,  # Enable ring light (neopixel)
    'optical_density': False,  # Enable optical density sensor (ADS1115)
    'eyespy_adc': True,  # Enable eyespy ADC boards
}

config.LOG_TO_TERMINAL = True  # Print logs to terminal (default: True)
config.LOG_FILE = 'bioreactor.log'  # Also log to file

config.USE_TIMESTAMPED_FILENAME: bool = False 

# Initialize bioreactor
with Bioreactor(config) as reactor:
    # Check if components are initialized
    if reactor.is_component_initialized('temp_sensor'):
        print("Temperature sensors are ready!")
        # Use sensors via reactor.temp_sensors array
    
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

    # Start scheduled jobs
    # Format: (function, frequency_seconds, duration)
    # frequency: time between calls in seconds, or True for continuous
    # duration: how long to run in seconds, or True for indefinite
    jobs = [
        # Measure and record sensors every 5 seconds
        (partial(measure_and_record_sensors, led_power=8.0), 2, True),  # Read sensors and record to CSV every 5 seconds
        
        # Temperature PID controller - maintains temperature at 37.0°C
        # Run PID controller every 5 seconds
        (partial(temperature_pid_controller, setpoint=37.0, kp=12.0, ki=0.015, kd=0.0), 5, True),
        
        # Ring light cycle - turns on at (50,50,50) for 60s, then off for 60s, repeating
        # Check every 1 second to update state
        (partial(ring_light_cycle, color=(50, 50, 50), on_time=60.0, off_time=60.0), 1, True),

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

