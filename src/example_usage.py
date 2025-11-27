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
from src.utils import *
from src.io import *

# Load default config
config = Config()

# Override some settings in the configuration
config.INIT_COMPONENTS = {
    'relays': False,
    'co2_sensor': False,
    'co2_sensor_2': False,  # Enable second CO2 sensor on /dev/ttyUSB1
    'o2_sensor': False,  # Enable O2 sensor for plotting
    'i2c': False,
    'temp_sensor': False,
    'peltier_driver': True,
    'stirrer': True,
}

config.RELAY_PINS = [6, 13, 19, 26]
config.RELAY_NAMES = ['pump_1', 'co2_solenoid', 'dump_valve', 'relay_4']

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
        # Run pump_1 every 3 minutes for 15 seconds (pass duration argument)
        # (read_sensors_and_plot, 5, True),  # Read sensors and update plot every 5 seconds

    ]
    if reactor.is_component_initialized('temp_sensor'):
        print(f"Temperature: {get_temperature(reactor, 0)}")
    
    # Use RelayController for clean API (recommended)
    if reactor.is_component_initialized('relays'):
        print(f"pump_1: {'on' if reactor.relay_controller.get_state('pump_1') else 'off'}")
        # Other examples:
        # reactor.relay_controller.on('pump_1')      # Turn relay ON
        # reactor.relay_controller.off('pump_1')     # Turn relay OFF
        # reactor.relay_controller.all_on()          # Turn all relays ON
        # reactor.relay_controller.get_all_states()   # Get all relay states
    
    if reactor.is_component_initialized('peltier_driver'):
        set_peltier_power(reactor, 25, 'heat')
        time.sleep(10)
        set_peltier_power(reactor, 25, 'cool')
        time.sleep(10)
        print("Peltier set to 25% duty (heat direction)")
        stop_peltier(reactor)
    
    if reactor.is_component_initialized('stirrer'):
        set_stirrer_speed(reactor, 50)
        print("Stirrer running at 50% duty")
        time.sleep(5)
        stop_stirrer(reactor)
    
    # You can also call functions directly (not as scheduled jobs):
    # flush_tank(reactor, 30)  # Flush tank once with 30s valve open
    # inject_co2_delayed(reactor, 300, 30)  # Wait 5 min (300s), inject CO2 for 30s (one-time)
    
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

