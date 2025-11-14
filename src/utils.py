"""
Utility functions for bioreactor operations.
These functions are designed to be used with bioreactor.run() for scheduled tasks.
"""

import time
import logging
from collections import deque
import matplotlib.pyplot as plt

logger = logging.getLogger("Bioreactor.Utils")

# Global variables for plotting (shared across calls)
_plot_initialized = False
_fig = None
_ax1 = None
_ax2 = None
_co2_data = deque(maxlen=1000)
_o2_data = deque(maxlen=1000)
_time_data = deque(maxlen=1000)
_start_time = None


def actuate_relay_timed(bioreactor, relay_name, duration_seconds, elapsed=None):
    """
    Generic function to actuate any relay for a specified duration.
    
    Args:
        bioreactor: Bioreactor instance
        relay_name: Name of the relay to actuate
        duration_seconds: How long to keep relay ON (in seconds)
        elapsed: Time elapsed since job started (optional)
    """
    if not bioreactor.is_component_initialized('relays'):
        bioreactor.logger.warning("Relays not initialized")
        return
    
    if not hasattr(bioreactor, 'relays') or relay_name not in bioreactor.relays:
        bioreactor.logger.warning(f"Relay '{relay_name}' not found")
        return
    
    try:
        import lgpio
        
        relay_info = bioreactor.relays[relay_name]
        gpio_chip = relay_info['chip']
        pin = relay_info['pin']
        
        # Turn relay ON
        lgpio.gpio_write(gpio_chip, pin, 0)
        bioreactor.logger.info(f"{relay_name} relay turned ON")
        
        # Wait for specified duration
        time.sleep(duration_seconds)
        
        # Turn relay OFF
        lgpio.gpio_write(gpio_chip, pin, 1)
        bioreactor.logger.info(f"{relay_name} relay turned OFF ({duration_seconds}s elapsed)")
        
    except Exception as e:
        bioreactor.logger.error(f"Error actuating {relay_name} relay: {e}")


def actuate_pump1_relay(bioreactor, elapsed=None):
    """
    Actuate pump1 relay ON for 10 seconds.
    Designed to run every 5 minutes (300 seconds).
    
    Uses the general actuate_relay_timed function.
    
    Args:
        bioreactor: Bioreactor instance
        elapsed: Time elapsed since job started (optional, provided by run())
    """
    actuate_relay_timed(bioreactor, 'pump_1', 10, elapsed)


def create_inject_co2_job(delay_seconds, injection_duration_seconds):
    """
    Create an inject_co2_delayed function with specific delay and injection duration for use with bioreactor.run().
    
    Args:
        delay_seconds: Time to wait before starting CO2 injection
        injection_duration_seconds: Duration to keep CO2 solenoid ON
        
    Returns:
        function: A function that can be used with bioreactor.run()
        
    Example:
        jobs = [
            (create_inject_co2_job(300, 30), True, 330),  # Wait 5 min, inject 30s
        ]
    """
    def inject_co2_job(bioreactor, elapsed=None):
        inject_co2_delayed(bioreactor, delay_seconds, injection_duration_seconds, elapsed)
    
    return inject_co2_job


def inject_co2_delayed(bioreactor, delay_seconds, injection_duration_seconds, elapsed=None):
    """
    Wait for specified delay, then inject CO2 for specified duration by turning on co2_solenoid relay.
    This is a one-time job that completes after the injection.
    
    Sequence:
    1. Wait for delay_seconds
    2. Turn ON co2_solenoid relay for injection_duration_seconds
    3. Turn OFF co2_solenoid relay
    4. Job completes
    
    Args:
        bioreactor: Bioreactor instance
        delay_seconds: Time to wait before starting CO2 injection (in seconds)
        injection_duration_seconds: Duration to keep CO2 solenoid ON (in seconds)
        elapsed: Time elapsed since job started (optional, provided by run())
    """
    if not bioreactor.is_component_initialized('relays'):
        bioreactor.logger.warning("Relays not initialized, cannot inject CO2")
        return
    
    if not hasattr(bioreactor, 'relays') or 'co2_solenoid' not in bioreactor.relays:
        bioreactor.logger.warning("co2_solenoid relay not found")
        return
    
    try:
        import lgpio
        
        relay_info = bioreactor.relays['co2_solenoid']
        gpio_chip = relay_info['chip']
        relay_pin = relay_info['pin']
        
        # Step 1: Wait for specified delay
        bioreactor.logger.info(f"Waiting {delay_seconds} seconds before CO2 injection...")
        time.sleep(delay_seconds)
        
        # Step 2: Turn ON co2_solenoid for specified duration
        bioreactor.logger.info(f"Starting CO2 injection ({injection_duration_seconds} seconds)...")
        lgpio.gpio_write(gpio_chip, relay_pin, 0)  # 0 = ON
        bioreactor.logger.info("co2_solenoid turned ON")
        
        time.sleep(injection_duration_seconds)  # Inject for specified duration
        
        # Step 3: Turn OFF co2_solenoid
        lgpio.gpio_write(gpio_chip, relay_pin, 1)  # 1 = OFF
        bioreactor.logger.info("co2_solenoid turned OFF - CO2 injection complete")
        
    except Exception as e:
        bioreactor.logger.error(f"Error during CO2 injection: {e}")
        # Try to turn off relay in case of error
        try:
            import lgpio
            gpio_chip = bioreactor.relays['co2_solenoid']['chip']
            relay_pin = bioreactor.relays['co2_solenoid']['pin']
            lgpio.gpio_write(gpio_chip, relay_pin, 1)  # Turn off
            bioreactor.logger.info("Emergency: co2_solenoid turned OFF")
        except:
            pass


def create_flush_tank_job(duration_seconds):
    """
    Create a flush_tank function with a specific duration for use with bioreactor.run().
    
    Args:
        duration_seconds: Duration to keep valve open during flush
        
    Returns:
        function: A function that can be used with bioreactor.run()
        
    Example:
        jobs = [
            (create_flush_tank_job(30), 3600, True),  # Flush every hour for 30s
        ]
    """
    def flush_tank_job(bioreactor, elapsed=None):
        flush_tank(bioreactor, duration_seconds, elapsed)
    
    return flush_tank_job


def flush_tank(bioreactor, duration_seconds, elapsed=None):
    """
    Flush tank by running pump_1 and opening dump_valve for specified duration,
    then closing valve and continuing pump for additional 20 seconds.
    
    Sequence:
    1. Turn ON pump_1
    2. Turn ON dump_valve (valve opens)
    3. Wait for duration_seconds
    4. Turn OFF dump_valve (valve closes)
    5. Continue running pump_1 for additional 20 seconds
    6. Turn OFF pump_1
    
    Args:
        bioreactor: Bioreactor instance
        duration_seconds: Duration to keep valve open (in seconds)
        elapsed: Time elapsed since job started (optional, provided by run())
    """
    if not bioreactor.is_component_initialized('relays'):
        bioreactor.logger.warning("Relays not initialized, cannot flush tank")
        return
    
    if not hasattr(bioreactor, 'relays'):
        bioreactor.logger.warning("Relays not found")
        return
    
    pump_relay = 'pump_1'
    valve_relay = 'dump_valve'
    
    if pump_relay not in bioreactor.relays:
        bioreactor.logger.warning(f"Relay '{pump_relay}' not found")
        return
    
    if valve_relay not in bioreactor.relays:
        bioreactor.logger.warning(f"Relay '{valve_relay}' not found")
        return
    
    try:
        import lgpio
        
        pump_info = bioreactor.relays[pump_relay]
        valve_info = bioreactor.relays[valve_relay]
        gpio_chip = pump_info['chip']  # Both should use same chip
        pump_pin = pump_info['pin']
        valve_pin = valve_info['pin']
        
        bioreactor.logger.info(f"Starting tank flush: pump ON, valve opening for {duration_seconds}s")
        
        # Step 1: Turn ON pump_1
        lgpio.gpio_write(gpio_chip, pump_pin, 0)  # 0 = ON
        bioreactor.logger.info(f"{pump_relay} turned ON")
        
        # Step 2: Turn ON dump_valve (valve opens)
        lgpio.gpio_write(gpio_chip, valve_pin, 0)  # 0 = ON
        bioreactor.logger.info(f"{valve_relay} turned ON (valve open)")
        
        # Step 3: Wait for specified duration
        time.sleep(duration_seconds)
        bioreactor.logger.info(f"Valve open duration ({duration_seconds}s) completed")
        
        # Step 4: Turn OFF dump_valve (valve closes)
        lgpio.gpio_write(gpio_chip, valve_pin, 1)  # 1 = OFF
        bioreactor.logger.info(f"{valve_relay} turned OFF (valve closed)")
        
        # Step 5: Continue running pump_1 for additional 20 seconds
        bioreactor.logger.info("Continuing pump for additional 20 seconds")
        time.sleep(20)
        
        # Step 6: Turn OFF pump_1
        lgpio.gpio_write(gpio_chip, pump_pin, 1)  # 1 = OFF
        bioreactor.logger.info(f"{pump_relay} turned OFF - Tank flush complete")
        
    except Exception as e:
        bioreactor.logger.error(f"Error during tank flush: {e}")
        # Try to turn off both relays in case of error
        try:
            import lgpio
            gpio_chip = bioreactor.relays[pump_relay]['chip']
            lgpio.gpio_write(gpio_chip, pump_info['pin'], 1)  # Turn off pump
            lgpio.gpio_write(gpio_chip, valve_info['pin'], 1)  # Turn off valve
            bioreactor.logger.info("Emergency: Both relays turned OFF")
        except:
            pass


def read_sensors_and_plot(bioreactor, elapsed=None):
    """
    Read CO2 and O2 sensors and update live plot with interactive relay control buttons.
    Designed to run periodically (e.g., every 1-5 seconds).
    
    Args:
        bioreactor: Bioreactor instance
        elapsed: Time elapsed since job started (optional, provided by run())
    """
    global _plot_initialized, _fig, _ax1, _ax2, _co2_data, _o2_data, _time_data, _start_time
    global _bioreactor_ref, _relay_buttons
    
    # Initialize plot on first call
    if not _plot_initialized:
        try:
            _fig, (_ax1, _ax2) = plt.subplots(2, 1, figsize=(12, 8))
            _fig.suptitle('Live CO2 and O2 Monitoring')
            
            # CO2 subplot (top)
            _ax1.set_title('CO2 Concentration')
            _ax1.set_ylabel('CO2 (ppm)')
            _ax1.set_ylim(300, 100000)  # Fixed scale
            _ax1.grid(True, alpha=0.3)
            
            # O2 subplot (bottom)
            _ax2.set_title('O2 Concentration')
            _ax2.set_ylabel('O2 (%)')
            _ax2.set_xlabel('Time (seconds)')
            _ax2.set_ylim(15, 25)  # Fixed scale
            _ax2.grid(True, alpha=0.3)
            
            # No buttons in plot window anymore - they're in separate window
            plt.ion()  # Turn on interactive mode
            plt.show(block=False)
            
            _start_time = time.time()
            _plot_initialized = True
            bioreactor.logger.info("Plot initialized for sensor monitoring")
        except Exception as e:
            bioreactor.logger.error(f"Error initializing plot: {e}")
            return
    
    try:
        co2_value = None
        o2_value = None
        
        # Read O2 sensor if available
        if bioreactor.is_component_initialized('o2_sensor') and hasattr(bioreactor, 'o2_sensor'):
            try:
                from atlas_i2c import commands
                o2_reading = bioreactor.o2_sensor.query(commands.READ)
                o2_value = float(o2_reading.data.decode().replace('%', '').strip())
            except Exception as e:
                bioreactor.logger.warning(f"Error reading O2 sensor: {e}")
        
        # Read CO2 sensor from serial if available
        if bioreactor.is_component_initialized('co2_sensor') and hasattr(bioreactor, 'co2_sensor'):
            try:
                # CO2 sensor is now a serial.Serial object
                bioreactor.co2_sensor.write(b"\xFE\x44\x00\x08\x02\x9F\x25")
                time.sleep(0.1)
                resp = bioreactor.co2_sensor.read(7)
                if len(resp) >= 5:
                    high = resp[3]
                    low = resp[4]
                    co2_value = 10 * ((high * 256) + low)
            except Exception as e:
                bioreactor.logger.warning(f"Error reading CO2 sensor: {e}")
        
        # Add data to arrays
        current_time = time.time()
        if _start_time is None:
            _start_time = current_time
        
        if co2_value is not None:
            _co2_data.append(co2_value)
        if o2_value is not None:
            _o2_data.append(o2_value)
        _time_data.append(current_time)
        
        # Update plots if we have data
        if len(_time_data) > 1:
            # Convert time to relative seconds
            time_relative = [(t - _start_time) for t in _time_data]
            
            # Update CO2 plot
            _ax1.clear()
            _ax1.set_title('CO2 Concentration')
            _ax1.set_ylabel('CO2 (ppm)')
            _ax1.set_ylim(300, 100000)
            _ax1.grid(True, alpha=0.3)
            if len(_co2_data) > 0:
                _ax1.plot(time_relative[-len(_co2_data):], list(_co2_data), 'b-', linewidth=2, label='CO2')
            _ax1.legend()
            
            # Update O2 plot
            _ax2.clear()
            _ax2.set_title('O2 Concentration')
            _ax2.set_ylabel('O2 (%)')
            _ax2.set_xlabel('Time (seconds)')
            _ax2.set_ylim(15, 25)
            _ax2.grid(True, alpha=0.3)
            if len(_o2_data) > 0:
                _ax2.plot(time_relative[-len(_o2_data):], list(_o2_data), 'r-', linewidth=2, label='O2')
            _ax2.legend()
            
            # Refresh plot
            plt.draw()
            plt.pause(0.01)
        
        # Log readings
        readings = []
        if co2_value is not None:
            readings.append(f"CO2: {co2_value:.1f} ppm")
        if o2_value is not None:
            readings.append(f"O2: {o2_value:.1f}%")
        
        if readings:
            bioreactor.logger.info(", ".join(readings))
        
        # Write to CSV if available
        if hasattr(bioreactor, 'writer') and hasattr(bioreactor, 'fieldnames'):
            try:
                # Use sensor labels from config if available, otherwise use generic names
                row = {'time': current_time}
                
                # Get sensor label keys from config if available
                if hasattr(bioreactor.cfg, 'SENSOR_LABELS'):
                    co2_key = None
                    o2_key = None
                    for key, label in bioreactor.cfg.SENSOR_LABELS.items():
                        if 'co2' in key.lower():
                            co2_key = label
                        if 'o2' in key.lower():
                            o2_key = label
                    
                    if co2_value is not None:
                        row[co2_key if co2_key else 'CO2_ppm'] = co2_value
                    if o2_value is not None:
                        row[o2_key if o2_key else 'O2_percent'] = o2_value
                else:
                    # Use generic names if no config labels
                    if co2_value is not None:
                        row['CO2_ppm'] = co2_value
                    if o2_value is not None:
                        row['O2_percent'] = o2_value
                
                # Only write if row has data and matches fieldnames
                if row and all(k in bioreactor.fieldnames for k in row.keys()):
                    bioreactor.writer.writerow(row)
                    if hasattr(bioreactor, 'out_file'):
                        bioreactor.out_file.flush()
            except Exception as e:
                bioreactor.logger.warning(f"Error writing to CSV: {e}")
                
    except Exception as e:
        bioreactor.logger.error(f"Error in read_sensors_and_plot: {e}")

