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

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import Bioreactor, Config
from src.utils import actuate_pump1_relay, read_sensors_and_plot

# Option 1: Use default config
config = Config()

# Option 2: Customize which components to initialize
config.INIT_COMPONENTS = {
    'relays': True,
    'co2_sensor': True,
    'o2_sensor': True,  # Enable O2 sensor for plotting
    'i2c': False,
}

# Option 3: Customize component settings
config.RELAY_PINS = [6, 13, 19, 26]
config.RELAY_NAMES = ['pump_1', 'co2_solenoid', 'dump_valve', 'relay_4']
config.CO2_SENSOR_ADDRESS = 105

# Option 4: Control logging output
config.LOG_TO_TERMINAL = True  # Print logs to terminal (default: True)
config.LOG_FILE = 'bioreactor.log'  # Also log to file
# Set LOG_TO_TERMINAL = False to only log to file

# Initialize bioreactor
with Bioreactor(config) as reactor:
    # Check if components are initialized
    if reactor.is_component_initialized('relays'):
        print("Relays are ready!")
        # Use relays via reactor.relays dict
    
    if reactor.is_component_initialized('co2_sensor'):
        print("CO2 sensor is ready!")
        # Use sensor via reactor.co2_sensor
    
    # Start scheduled jobs
    # Format: (function, frequency_seconds, duration)
    # frequency: time between calls in seconds, or True for continuous
    # duration: how long to run in seconds, or True for indefinite
    jobs = [
        (actuate_pump1_relay, 300, True),  # Run every 5 minutes (300s) indefinitely
        (read_sensors_and_plot, 5, True),  # Read sensors and update plot every 5 seconds
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

