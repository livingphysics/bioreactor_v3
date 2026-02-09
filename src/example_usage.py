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

config.LOG_TO_TERMINAL = True  # Print logs to terminal (default: True)
config.LOG_FILE = 'bioreactor.log'  # Also log to file

config.USE_TIMESTAMPED_FILENAME: bool = False 

# Initialize bioreactor (script_path is copied into results package when RESULTS_PACKAGE is True)
with Bioreactor(config, script_path=os.path.abspath(__file__)) as reactor:
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
        # Measure and record sensors every 5 seconds
        (partial(measure_and_record_sensors, led_power=15.0), 20, True),  # Read sensors and record to CSV every 5 seconds
        
        # Temperature PID controller - maintains temperature at 37.0°C
        # Run PID controller every 5 seconds
        (partial(temperature_pid_controller, setpoint=25.0, kp=12.0, ki=0.015, kd=0.0), 5, True),
        
        # Ring light cycle - turns on at (50,50,50) for 60s, then off for 60s, repeating
        # Check every 1 second to update state
        (partial(ring_light_cycle, color=(50, 50, 50), on_time=60.0, off_time=60.0), 1, True),
        
        # Balanced flow - maintains balanced inflow/outflow for chemostat mode
        # Sets both inflow and outflow pumps to the same flow rate (2 ml/s)
        # Run every 10 seconds to maintain flow rate
        # (partial(balanced_flow, pump_name='inflow', ml_per_sec=2.0), 10, True),

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

