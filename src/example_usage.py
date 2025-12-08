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
    'relays': True,
    'co2_sensor': True,
    'co2_sensor_2': True,  # Enable second CO2 sensor on /dev/ttyUSB1
    'o2_sensor': True,  # Enable O2 sensor for plotting
    'i2c': True,
    'temp_sensor': True,
    'peltier_driver': True,
    'stirrer': True,
    'led': True,  # Enable LED PWM control
    'optical_density': True,  # Enable optical density sensor (ADS1115)
}

config.RELAY_PINS = {
    'pump_1': 6,
    'co2_solenoid': 13,
    'dump_valve': 19,
    'relay_4': 26,
}

config.LOG_TO_TERMINAL = True  # Print logs to terminal (default: True)
config.LOG_FILE = 'bioreactor.log'  # Also log to file
# Set LOG_TO_TERMINAL = False to only log to file

config.USE_TIMESTAMPED_FILENAME: bool = False 

# Initialize bioreactor
with Bioreactor(config) as reactor:
    # Check if components are initialized
    if reactor.is_component_initialized('relays'):
        print("Relays are ready!")
        # Use relays via reactor.relays dict
    
    if reactor.is_component_initialized('co2_sensor'):
        print("CO2 sensor is ready!")
        # Use sensor via reactor.co2_sensor
    
    if reactor.is_component_initialized('temp_sensor'):
        print("Temperature sensors are ready!")
        # Use sensors via reactor.temp_sensors array

    # Start scheduled jobs
    # Format: (function, frequency_seconds, duration)
    # frequency: time between calls in seconds, or True for continuous
    # duration: how long to run in seconds, or True for indefinite
    jobs = [
        # Measure, record, and plot sensors every 5 seconds
        (partial(measure_and_plot_sensors, led_power=80.0), 5, True),  # Read sensors and update plot every 5 seconds
        
        # Temperature PID controller - maintains temperature at 37.0°C
        # Run PID controller every 1 second
        (partial(temperature_pid_controller, setpoint=37.0, kp=12.0, ki=0.015, kd=0.0), 5, True),
        
        # Pressurize chamber (no CO2 injection)
        (partial(pressurize_and_inject_co2, pressurize_duration=10.0, co2_duration=0.0), 180, True),  # Pressurize every 3 minutes
        
        # Delayed CO2 injection - wait 120s, then inject for 10s (one-time job)
        (partial(inject_co2_delayed, delay_seconds=120.0, injection_duration_seconds=10.0), True, 150),  # One-time injection after 120s delay

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

