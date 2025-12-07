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
        (measure_and_plot_sensors, 5, True),  # Read sensors and update plot every 5 seconds
        
        # Temperature PID controller - maintains temperature at 37.0°C
        # Run PID controller every 1 second
        (partial(temperature_pid_controller, setpoint=37.0, kp=3.0, ki=0.3, kd=0.0), 1, True),

    ]
    # if reactor.is_component_initialized('temp_sensor'):
    #     print(f"Temperature: {get_temperature(reactor, 0)}")
    
    # Use RelayController for clean API (recommended)
    # if reactor.is_component_initialized('relays'):
    #     print(f"pump_1: {'on' if reactor.relay_controller.get_state('pump_1') else 'off'}")
        # Other examples:
        # reactor.relay_controller.on('pump_1')      # Turn relay ON
        # reactor.relay_controller.off('pump_1')     # Turn relay OFF
        # reactor.relay_controller.all_on()          # Turn all relays ON
        # reactor.relay_controller.get_all_states()   # Get all relay states
    
    # if reactor.is_component_initialized('peltier_driver'):
    #     set_peltier_power(reactor, 25, 'heat')
    #     time.sleep(10)
    #     set_peltier_power(reactor, 25, 'cool')
    #     time.sleep(10)
    #     print("Peltier set to 25% duty (heat direction)")
    #     stop_peltier(reactor)
    
    # if reactor.is_component_initialized('stirrer'):
    #     set_stirrer_speed(reactor, 50)
    #     print("Stirrer running at 50% duty")
    #     time.sleep(5)
    #     stop_stirrer(reactor)
    
    # # Optical density measurement example
    # if reactor.is_component_initialized('led') and reactor.is_component_initialized('optical_density'):
    #     print("Taking OD measurement...")
    #     # Measure OD with LED at 50% power, averaging for 2 seconds on single channel
    #     od_voltage = measure_od(reactor, led_power=30.0, averaging_duration=2.0, channel_name='Trx')
    #     if od_voltage is not None:
    #         print(f"OD measurement (Trx channel): {od_voltage:.4f}V")
        
    #     # Example: Measure all channels at once
    #     print("Taking OD measurement on all channels...")
    #     all_od_results = measure_od(reactor, led_power=30.0, averaging_duration=2.0, channel_name='all')
    #     if all_od_results is not None:
    #         print("OD measurements (all channels):")
    #         for channel, voltage in all_od_results.items():
    #             print(f"  {channel}: {voltage:.4f}V")
        
    #     # Example: Read voltage from channels without LED (baseline)
    #     ref_voltage = read_voltage(reactor, 'Ref')
    #     sct_voltage = read_voltage(reactor, 'Sct')
    #     if ref_voltage is not None:
    #         print(f"Reference channel voltage (LED off): {ref_voltage:.4f}V")
    #     if sct_voltage is not None:
    #         print(f"Scatter channel voltage (LED off): {sct_voltage:.4f}V")
    
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

