"""
Composite utility functions for bioreactor operations.
These functions combine multiple operations or provide higher-level convenience wrappers.
These functions are designed to be used with bioreactor.run() for scheduled tasks.
"""

import time
import logging
from typing import Union, Optional
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("Bioreactor.Utils")

# Global storage for plotting data
_plot_data = {
    'time': deque(maxlen=1000),
    'co2': deque(maxlen=1000),
    'co2_2': deque(maxlen=1000),
    'o2': deque(maxlen=1000),
    'temperature': deque(maxlen=1000),
    'od_trx': deque(maxlen=1000),
    'od_sct': deque(maxlen=1000),
}

# Global figure and axes for plotting
_plot_fig = None
_plot_axes = None

# Global storage for CO2 stabilization
_co2_duration = 0.5  # Global CO2 injection duration (updated by stabilize_co2)


def measure_and_plot_sensors(bioreactor, elapsed: Optional[float] = None, led_power: float = 30.0, averaging_duration: float = 0.5):
    """
    Measure, record, and plot sensor data from OD (Trx, Sct), Temperature, CO2, and O2.
    
    This composite function:
    1. Reads all sensor values
    2. Writes data to CSV file
    3. Updates live plots in 3 subplots:
       - Subplot 1: CO2 and CO2_2 (both sensors)
       - Subplot 2: O2 and Temperature
       - Subplot 3: OD voltages (Trx and Sct)
    
    Args:
        bioreactor: Bioreactor instance
        elapsed: Elapsed time in seconds (if None, uses time since start)
        led_power: LED power percentage for OD measurements (default: 30.0)
        averaging_duration: Duration in seconds for averaging OD measurements (default: 0.5)
        
    Returns:
        dict: Dictionary with all sensor readings
    """
    global _plot_fig, _plot_axes
    
    # Import IO functions
    from .io import get_temperature, read_voltage, measure_od, read_co2, read_co2_2, read_o2
    
    # Get elapsed time
    if elapsed is None:
        if not hasattr(bioreactor, '_start_time'):
            bioreactor._start_time = time.time()
        elapsed = time.time() - bioreactor._start_time
    
    # Read sensors
    sensor_data = {'time': elapsed}
    
    # Read CO2 (first sensor)
    co2_value = read_co2(bioreactor)
    if co2_value is not None:
        sensor_data['co2'] = co2_value
        _plot_data['co2'].append(co2_value)
    else:
        sensor_data['co2'] = float('nan')
        _plot_data['co2'].append(float('nan'))
    
    # Read CO2 (second sensor)
    co2_2_value = read_co2_2(bioreactor)
    if co2_2_value is not None:
        sensor_data['co2_2'] = co2_2_value
        _plot_data['co2_2'].append(co2_2_value)
    else:
        sensor_data['co2_2'] = float('nan')
        _plot_data['co2_2'].append(float('nan'))
    
    # Read O2
    o2_value = read_o2(bioreactor)
    if o2_value is not None:
        sensor_data['o2'] = o2_value
        _plot_data['o2'].append(o2_value)
    else:
        sensor_data['o2'] = float('nan')
        _plot_data['o2'].append(float('nan'))
    
    # Read Temperature
    temp_value = get_temperature(bioreactor, sensor_index=0)
    if not np.isnan(temp_value):
        sensor_data['temperature'] = temp_value
        _plot_data['temperature'].append(temp_value)
    else:
        sensor_data['temperature'] = float('nan')
        _plot_data['temperature'].append(float('nan'))
    
    # Read OD channels (Trx, Sct, and Ref)
    if bioreactor.is_component_initialized('led') and bioreactor.is_component_initialized('optical_density'):
        # Measure OD with LED on
        od_results = measure_od(bioreactor, led_power=led_power, averaging_duration=averaging_duration, channel_name='all')
        if od_results:
            trx_voltage = od_results.get('Trx', None)
            sct_voltage = od_results.get('Sct', None)
            ref_voltage = od_results.get('Ref', None)
            
            if trx_voltage is not None:
                sensor_data['od_trx'] = trx_voltage
                _plot_data['od_trx'].append(trx_voltage)
            else:
                sensor_data['od_trx'] = float('nan')
                _plot_data['od_trx'].append(float('nan'))
            
            if sct_voltage is not None:
                sensor_data['od_sct'] = sct_voltage
                _plot_data['od_sct'].append(sct_voltage)
            else:
                sensor_data['od_sct'] = float('nan')
                _plot_data['od_sct'].append(float('nan'))
            
            # Record Ref voltage in CSV but don't plot it
            if ref_voltage is not None:
                sensor_data['od_ref'] = ref_voltage
            else:
                sensor_data['od_ref'] = float('nan')
        else:
            sensor_data['od_trx'] = float('nan')
            sensor_data['od_sct'] = float('nan')
            sensor_data['od_ref'] = float('nan')
            _plot_data['od_trx'].append(float('nan'))
            _plot_data['od_sct'].append(float('nan'))
    else:
        # Try reading without LED if OD sensor is available but LED is not
        if bioreactor.is_component_initialized('optical_density'):
            trx_voltage = read_voltage(bioreactor, 'Trx')
            sct_voltage = read_voltage(bioreactor, 'Sct')
            ref_voltage = read_voltage(bioreactor, 'Ref')
            
            sensor_data['od_trx'] = trx_voltage if trx_voltage is not None else float('nan')
            sensor_data['od_sct'] = sct_voltage if sct_voltage is not None else float('nan')
            sensor_data['od_ref'] = ref_voltage if ref_voltage is not None else float('nan')
            _plot_data['od_trx'].append(sensor_data['od_trx'])
            _plot_data['od_sct'].append(sensor_data['od_sct'])
        else:
            sensor_data['od_trx'] = float('nan')
            sensor_data['od_sct'] = float('nan')
            sensor_data['od_ref'] = float('nan')
            _plot_data['od_trx'].append(float('nan'))
            _plot_data['od_sct'].append(float('nan'))
    
    # Add time to plot data
    _plot_data['time'].append(elapsed)
    
    # Write to CSV
    if hasattr(bioreactor, 'writer') and bioreactor.writer:
        # Map to config labels if available
        config = getattr(bioreactor, 'cfg', None)
        if config and hasattr(config, 'SENSOR_LABELS'):
            csv_row = {
                'time': elapsed,
                config.SENSOR_LABELS.get('co2', 'CO2_ppm'): sensor_data['co2'],
                config.SENSOR_LABELS.get('co2_2', 'CO2_2_ppm'): sensor_data.get('co2_2', float('nan')),
                config.SENSOR_LABELS.get('o2', 'O2_percent'): sensor_data['o2'],
                config.SENSOR_LABELS.get('temperature', 'temperature_C'): sensor_data['temperature'],
            }
            # Add OD data using config labels
            if 'od_trx' in sensor_data:
                csv_row[config.SENSOR_LABELS.get('od_trx', 'OD_Trx_V')] = sensor_data['od_trx']
            if 'od_sct' in sensor_data:
                csv_row[config.SENSOR_LABELS.get('od_sct', 'OD_Sct_V')] = sensor_data['od_sct']
            if 'od_ref' in sensor_data:
                csv_row[config.SENSOR_LABELS.get('od_ref', 'OD_Ref_V')] = sensor_data['od_ref']
        else:
            csv_row = sensor_data
        
        try:
            # Only write fields that exist in fieldnames to avoid errors
            if hasattr(bioreactor, 'fieldnames'):
                filtered_row = {k: v for k, v in csv_row.items() if k in bioreactor.fieldnames}
                bioreactor.writer.writerow(filtered_row)
            else:
                bioreactor.writer.writerow(csv_row)
            if hasattr(bioreactor, 'out_file'):
                bioreactor.out_file.flush()
        except Exception as e:
            bioreactor.logger.error(f"Error writing to CSV: {e}")
    
    # Update plots
    if len(_plot_data['time']) > 1:
        # Initialize figure if needed
        if _plot_fig is None:
            _plot_fig, _plot_axes = plt.subplots(2, 2, figsize=(14, 10))
            _plot_fig.suptitle('Live Sensor Monitoring', fontsize=14)
            plt.ion()  # Turn on interactive mode
            plt.show(block=False)
        
        # Top left: CO2 (both sensors)
        ax1 = _plot_axes[0, 0]
        ax1.clear()
        ax1.set_title('CO2 Concentration')
        ax1.set_ylabel('CO2 (ppm)')
        ax1.grid(True, alpha=0.3)
        if len(_plot_data['co2']) > 0:
            ax1.plot(list(_plot_data['time']), list(_plot_data['co2']), 'b-', linewidth=2, label='CO2')
        if len(_plot_data['co2_2']) > 0:
            ax1.plot(list(_plot_data['time']), list(_plot_data['co2_2']), 'r--', linewidth=2, label='CO2_2')
        ax1.legend()
        
        # Top right: O2
        ax2 = _plot_axes[0, 1]
        ax2.clear()
        ax2.set_title('O2 Concentration')
        ax2.set_ylabel('O2 (%)')
        ax2.grid(True, alpha=0.3)
        if len(_plot_data['o2']) > 0:
            ax2.plot(list(_plot_data['time']), list(_plot_data['o2']), 'r-', linewidth=2, label='O2')
        ax2.legend()
        
        # Bottom left: Temperature
        ax3 = _plot_axes[1, 0]
        ax3.clear()
        ax3.set_title('Temperature')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Temperature (°C)')
        ax3.grid(True, alpha=0.3)
        if len(_plot_data['temperature']) > 0:
            ax3.plot(list(_plot_data['time']), list(_plot_data['temperature']), 'g-', linewidth=2, label='Temperature')
        ax3.legend()
        
        # Bottom right: OD voltages
        ax4 = _plot_axes[1, 1]
        ax4.clear()
        ax4.set_title('Optical Density Voltages')
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Voltage (V)')
        ax4.grid(True, alpha=0.3)
        if len(_plot_data['od_trx']) > 0:
            ax4.plot(list(_plot_data['time']), list(_plot_data['od_trx']), 'm-', linewidth=2, label='Trx')
        if len(_plot_data['od_sct']) > 0:
            ax4.plot(list(_plot_data['time']), list(_plot_data['od_sct']), 'c-', linewidth=2, label='Sct')
        ax4.legend()
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)  # Small pause to update display
    
    bioreactor.logger.info(
        f"Sensor readings - CO2: {sensor_data.get('co2', 'N/A'):.1f} ppm, "
        f"CO2_2: {sensor_data.get('co2_2', 'N/A'):.1f} ppm, "
        f"O2: {sensor_data.get('o2', 'N/A'):.2f}%, "
        f"Temp: {sensor_data.get('temperature', 'N/A'):.2f}°C, "
        f"OD Trx: {sensor_data.get('od_trx', 'N/A'):.4f}V, "
        f"OD Sct: {sensor_data.get('od_sct', 'N/A'):.4f}V"
    )
    
    return sensor_data


def temperature_pid_controller(
    bioreactor,
    setpoint: float,
    current_temp: Optional[float] = None,
    kp: float = 12.0,
    ki: float = 0.015,
    kd: float = 0.0,
    dt: Optional[float] = None,
    elapsed: Optional[float] = None,
    sensor_index: int = 0,
    max_duty: float = 70.0,
    derivative_alpha: float = 0.7
) -> None:
    """
    Pure PID controller to maintain bioreactor temperature at setpoint by modulating peltier power.
    
    This composite function:
    1. Reads current temperature (or uses provided value)
    2. Calculates PID output based on error (setpoint - current_temp)
    3. Modulates peltier power and direction based on PID output
    
    Args:
        bioreactor: Bioreactor instance
        setpoint: Desired temperature (°C)
        current_temp: Measured temperature (°C). If None, reads from temperature sensor.
        kp: Proportional gain (default: 5.0)
        ki: Integral gain (default: 0.3)
        kd: Derivative gain (default: 2.0)
        dt: Time elapsed since last call (s). If None, uses elapsed parameter or estimates.
        elapsed: Elapsed time since start (s). Used to estimate dt if dt is None.
        sensor_index: Index of temperature sensor to read (default: 0)
        max_duty: Maximum duty cycle percentage (default: 70.0, hardware safety limit)
        derivative_alpha: Derivative filter coefficient (default: 0.7, 0-1, higher = less filtering)
        
    Note:
        PID state (_temp_integral, _temp_last_error, _temp_last_time) is stored on bioreactor instance.
        Initialize these values before first call if needed, or they will be auto-initialized.
        
    Example usage as a job:
        from functools import partial
        from src.utils import temperature_pid_controller
        
        # Create a partial function with setpoint=37.0°C
        pid_job = partial(temperature_pid_controller, setpoint=37.0, kp=5.0, ki=0.5, kd=0.0)
        
        # Add to jobs list
        jobs = [
            (pid_job, 1, True),  # Run PID controller every 1 second
        ]
        reactor.run(jobs)
    """
    from .io import get_temperature, set_peltier_power
    
    # Initialize PID state if not present
    if not hasattr(bioreactor, '_temp_integral'):
        bioreactor._temp_integral = 0.0
    if not hasattr(bioreactor, '_temp_last_error'):
        bioreactor._temp_last_error = 0.0
    if not hasattr(bioreactor, '_temp_last_time'):
        bioreactor._temp_last_time = None
    if not hasattr(bioreactor, '_temp_last_derivative'):
        bioreactor._temp_last_derivative = 0.0
    
    # Get current temperature if not provided
    if current_temp is None:
        current_temp = get_temperature(bioreactor, sensor_index=sensor_index)
    
    # Calculate error
    error = setpoint - current_temp
    
    # Calculate dt (time since last call)
    if dt is None:
        current_time = elapsed if elapsed is not None else time.time()
        if bioreactor._temp_last_time is not None:
            dt = current_time - bioreactor._temp_last_time
        else:
            dt = 1.0  # Default to 1 second for first call
        bioreactor._temp_last_time = current_time
    else:
        # Update last_time if elapsed is provided
        if elapsed is not None:
            bioreactor._temp_last_time = elapsed
    
    # Only update PID if error is not NaN
    if not np.isnan(error) and not np.isnan(current_temp):
        # Update integral term (pure PID - no clamping)
        bioreactor._temp_integral += error * dt
        
        # Calculate derivative term with low-pass filtering to reduce noise sensitivity
        raw_derivative = (error - bioreactor._temp_last_error) / dt if dt > 0 else 0.0
        # Apply exponential moving average filter to derivative
        derivative = derivative_alpha * bioreactor._temp_last_derivative + (1 - derivative_alpha) * raw_derivative
        bioreactor._temp_last_derivative = derivative
        
        # Calculate PID output (pure PID formula)
        output = kp * error + ki * bioreactor._temp_integral + kd * derivative
        
        # Convert output to duty cycle (0-100) and clamp to max_duty (hardware safety limit)
        duty = max(0, min(max_duty, abs(output)))
        
        # Determine direction based on PID output:
        # error = setpoint - current_temp
        # If error > 0 (too cold), output > 0, we need to HEAT
        # If error < 0 (too hot), output < 0, we need to COOL
        direction = 'heat' if output > 0 else 'cool'
        
        # Apply peltier control
        if bioreactor.is_component_initialized('peltier_driver'):
            if duty > 0:
                set_peltier_power(bioreactor, duty, forward=direction)
            else:
                # Turn off peltier when duty is 0
                from .io import stop_peltier
                stop_peltier(bioreactor)
            
            bioreactor.logger.info(
                f"Temperature PID: setpoint={setpoint:.2f}°C, "
                f"current={current_temp:.2f}°C, "
                f"error={error:.2f}°C, "
                f"output={output:.2f}, "
                f"duty={duty:.1f}%, "
                f"direction={direction}, "
                f"integral={bioreactor._temp_integral:.2f}"
            )
        else:
            bioreactor.logger.warning("Peltier driver not initialized; PID controller cannot modulate temperature.")
        
        # Store error for next iteration
        bioreactor._temp_last_error = error
    else:
        # Skip peltier update if error or temperature is NaN
        bioreactor.logger.warning(
            f"Temperature PID: NaN detected, skipping update. "
            f"setpoint={setpoint:.2f}°C, current_temp={current_temp}"
        )


def pressurize_and_inject_co2(
    bioreactor,
    pressurize_duration: float = 10.0,
    pause: float = 30.0,
    co2_duration: Union[float, str] = 0.1,
    elapsed: Optional[float] = None
) -> None:
    """
    Pressurize with pump_1, wait, then inject CO2.
    
    Sequence:
    1. Turn ON pump_1 relay for pressurize_duration seconds
    2. Turn OFF pump_1 relay
    3. Wait pause seconds
    4. Turn ON co2_solenoid relay for co2_duration seconds
    5. Turn OFF co2_solenoid relay
    
    Args:
        bioreactor: Bioreactor instance
        pressurize_duration: Duration to run pump_1 (default: 10.0 seconds)
        pause: Wait time between pump and CO2 injection (default: 30.0 seconds)
        co2_duration: Duration for CO2 injection in seconds, or "auto" to calculate based on CO2_2 reading (default: 0.1 seconds)
                      When "auto", calculates duration as CO2_2_value / 100000.0
        elapsed: Time elapsed since job started (optional)
    """
    if not bioreactor.is_component_initialized('relays') or not hasattr(bioreactor, 'relay_controller') or bioreactor.relay_controller is None:
        bioreactor.logger.warning("Relays not initialized or RelayController not available")
        return
    
    # Calculate CO2 duration if "auto"
    actual_co2_duration = co2_duration
    if co2_duration == "auto":
        from .io import read_co2_2
        co2_2_value = read_co2_2(bioreactor)
        if co2_2_value is not None and not np.isnan(co2_2_value):
            actual_co2_duration = co2_2_value / 100000.0
            bioreactor.logger.info(f"Auto CO2 duration: CO2_2={co2_2_value:.1f} ppm, calculated duration={actual_co2_duration:.3f}s")
        else:
            bioreactor.logger.warning("CO2_2 reading unavailable for auto calculation, using default 0.1s")
            actual_co2_duration = 0.1
    
    try:
        # Pressurize
        bioreactor.logger.info(f"Pressurizing: pump_1 ON for {pressurize_duration}s")
        bioreactor.relay_controller.on('pump_1')
        time.sleep(pressurize_duration)
        bioreactor.relay_controller.off('pump_1')
        
        # Wait before CO2 injection
        bioreactor.logger.info(f"Waiting {pause}s before CO2 injection...")
        time.sleep(pause)
        
        # Inject CO2
        bioreactor.logger.info(f"Injecting CO2: co2_solenoid ON for {actual_co2_duration:.3f}s")
        bioreactor.relay_controller.on('co2_solenoid')
        time.sleep(actual_co2_duration)
        bioreactor.relay_controller.off('co2_solenoid')
        bioreactor.logger.info("Pressurize and inject complete")
        
    except Exception as e:
        bioreactor.logger.error(f"Error in pressurize_and_inject_co2: {e}")
        # Emergency shutdown
        try:
            bioreactor.relay_controller.off('pump_1')
            bioreactor.relay_controller.off('co2_solenoid')
        except:
            pass


def inject_co2_delayed(
    bioreactor,
    delay_seconds: float = 0.0,
    injection_duration_seconds: float = 1.0,
    elapsed: Optional[float] = None
) -> None:
    """
    Wait for specified delay, then inject CO2 for specified duration.
    This is a one-time job that completes after the injection.
    
    Sequence:
    1. Wait for delay_seconds
    2. Turn ON co2_solenoid relay for injection_duration_seconds
    3. Turn OFF co2_solenoid relay
    4. Job completes
    
    Args:
        bioreactor: Bioreactor instance
        delay_seconds: Time to wait before starting CO2 injection (default: 0.0 seconds)
        injection_duration_seconds: Duration to keep CO2 solenoid ON (default: 1.0 seconds)
        elapsed: Time elapsed since job started (optional)
    """
    # Track if injection has already been executed (one-time job)
    if not hasattr(bioreactor, '_co2_injection_executed'):
        bioreactor._co2_injection_executed = False
    
    if bioreactor._co2_injection_executed:
        return  # Already executed, skip
    
    if not bioreactor.is_component_initialized('relays') or not hasattr(bioreactor, 'relay_controller') or bioreactor.relay_controller is None:
        bioreactor.logger.warning("Relays not initialized or RelayController not available")
        return
    
    try:
        # Mark as executed before starting to prevent duplicate runs
        bioreactor._co2_injection_executed = True
        
        # Wait for specified delay
        if delay_seconds > 0:
            bioreactor.logger.info(f"Waiting {delay_seconds}s before CO2 injection...")
            time.sleep(delay_seconds)
        
        # Inject CO2
        bioreactor.logger.info(f"Starting CO2 injection ({injection_duration_seconds}s)...")
        bioreactor.relay_controller.on('co2_solenoid')
        time.sleep(injection_duration_seconds)
        bioreactor.relay_controller.off('co2_solenoid')
        bioreactor.logger.info("CO2 injection complete")
        
    except Exception as e:
        bioreactor.logger.error(f"Error during CO2 injection: {e}")
        # Emergency shutdown
        try:
            bioreactor.relay_controller.off('co2_solenoid')
        except:
            pass


def flush_tank(
    bioreactor,
    duration_seconds: float,
    elapsed: Optional[float] = None
) -> None:
    """
    Flush tank by running pump_1 and opening dump_valve for specified duration,
    then closing valve and continuing pump for the same duration.
    
    Sequence:
    1. Turn ON pump_1
    2. Turn ON dump_valve (valve opens)
    3. Wait for duration_seconds
    4. Turn OFF dump_valve (valve closes)
    5. Continue running pump_1 for additional duration_seconds
    6. Turn OFF pump_1
    
    Args:
        bioreactor: Bioreactor instance
        duration_seconds: Duration to keep valve open and continue pump after closing (in seconds)
        elapsed: Time elapsed since job started (optional)
    """
    if not bioreactor.is_component_initialized('relays') or not hasattr(bioreactor, 'relay_controller') or bioreactor.relay_controller is None:
        bioreactor.logger.warning("Relays not initialized or RelayController not available")
        return
    
    try:
        bioreactor.logger.info(f"Starting tank flush: pump ON, valve opening for {duration_seconds}s")
        
        # Step 1: Turn ON pump_1
        bioreactor.relay_controller.on('pump_1')
        bioreactor.logger.info("pump_1 turned ON")
        
        # Step 2: Turn ON dump_valve (valve opens)
        bioreactor.relay_controller.on('dump_valve')
        bioreactor.logger.info("dump_valve turned ON (valve open)")
        
        # Step 3: Wait for specified duration
        time.sleep(duration_seconds)
        bioreactor.logger.info(f"Valve open duration ({duration_seconds}s) completed")
        
        # Step 4: Turn OFF dump_valve (valve closes)
        bioreactor.relay_controller.off('dump_valve')
        bioreactor.logger.info("dump_valve turned OFF (valve closed)")
        
        # Step 5: Continue running pump_1 for additional duration_seconds
        bioreactor.logger.info(f"Continuing pump for additional {duration_seconds} seconds")
        time.sleep(duration_seconds)
        
        # Step 6: Turn OFF pump_1
        bioreactor.relay_controller.off('pump_1')
        bioreactor.logger.info("pump_1 turned OFF - Tank flush complete")
        
    except Exception as e:
        bioreactor.logger.error(f"Error during tank flush: {e}")
        # Emergency shutdown - turn off both relays
        try:
            bioreactor.relay_controller.off('pump_1')
            bioreactor.relay_controller.off('dump_valve')
            bioreactor.logger.info("Emergency: Both relays turned OFF")
        except:
            pass


def stabilize_co2(
    bioreactor,
    pressurize_duration: float = 10.0,
    pause: float = 30.0,
    co2_duration: Optional[float] = None,
    elapsed: Optional[float] = None
) -> None:
    """
    Stabilize CO2 by constantly pressurizing and injecting CO2, 
    adjusting injection duration based on CO2 trend slope.
    
    Always pressurizes and injects CO2, then calculates the slope of CO2 readings
    from measure_and_plot_sensors data and adjusts the injection duration accordingly.
    
    Sequence:
    1. Run pressurize_and_inject_co2 with current CO2 duration
    2. Use CO2_2 data from measure_and_plot_sensors (stored in _plot_data)
    3. Calculate linear fit slope from last 70 CO2 data points
    4. Update CO2 duration for future use based on slope adjustment
    
    Args:
        bioreactor: Bioreactor instance
        pressurize_duration: Fixed duration to run pump_1 (default: 10.0 seconds)
        pause: Wait time between pump and CO2 injection (default: 30.0 seconds)
        co2_duration: Initial CO2 injection duration (default: uses global _co2_duration or 0.5s)
        elapsed: Time elapsed since job started (optional)
    """
    global _co2_duration
    
    if not bioreactor.is_component_initialized('relays') or not hasattr(bioreactor, 'relay_controller') or bioreactor.relay_controller is None:
        bioreactor.logger.warning("Relays not initialized or RelayController not available")
        return
    
    # Use global co2_duration if not provided
    if co2_duration is None:
        co2_duration = _co2_duration if _co2_duration >= 0 else 0.1
    
    # Always pressurize and inject with current CO2 duration
    pressurize_and_inject_co2(bioreactor, pressurize_duration, pause, co2_duration, elapsed)
    
    # Use CO2_2 data directly from measure_and_plot_sensors (stored in _plot_data)
    # Get the last 70 points (or all if fewer) for linear fit
    if len(_plot_data['co2_2']) < 2:
        bioreactor.logger.warning("Not enough CO2_2 data from measure_and_plot_sensors (need at least 2 points)")
        return
    
    # Ensure co2_2 and time arrays have the same length
    co2_list = list(_plot_data['co2_2'])
    time_list = list(_plot_data['time'])
    min_len = min(len(co2_list), len(time_list))
    
    if min_len < 2:
        bioreactor.logger.warning("Not enough data points (need at least 2 points)")
        return
    
    # Take the last n_points from both arrays, ensuring they're aligned
    n_points = min(20, min_len)
    co2_values = np.array(co2_list[-n_points:])
    time_values = np.array(time_list[-n_points:])
    
    # Ensure arrays have the same length
    if len(co2_values) != len(time_values):
        # Take the minimum length to ensure alignment
        min_array_len = min(len(co2_values), len(time_values))
        co2_values = co2_values[-min_array_len:]
        time_values = time_values[-min_array_len:]
    
    # Filter out NaN values
    valid_mask = ~np.isnan(co2_values)
    if np.sum(valid_mask) < 2:
        bioreactor.logger.warning("Not enough valid CO2_2 data points (need at least 2 non-NaN values)")
        return
    
    # Apply mask to both arrays (they should be the same length now)
    co2_values = co2_values[valid_mask]
    time_values = time_values[valid_mask]
    
    # Perform linear regression: y = slope * x + intercept
    if len(time_values) > 1 and len(co2_values) > 1 and len(time_values) == len(co2_values):
        # Linear fit: [slope, intercept]
        coeffs = np.polyfit(time_values, co2_values, 1)
        slope = coeffs[0]  # ppm per second
        
        bioreactor.logger.info(f"CO2 slope: {slope:.2f} ppm/s (from {len(co2_values)} points)")
        
        # Adjust CO2 duration based on slope
        if abs(slope) < 3:  # Essentially zero (avoid floating point issues)
            # Slope is zero, don't change duration
            bioreactor.logger.info("CO2 slope is zero, keeping current CO2 duration")
        else:
            # Calculate adjustment: more aggressive for positive slopes (CO2 increasing)
            if slope > 0:
                # Positive slope: CO2 increasing, reduce injection more aggressively
                # Use smaller divisor (50 instead of 100) for faster response
                adjustment = -slope / 2000.0
            else:
                # Negative slope: CO2 decreasing, increase injection less aggressively
                adjustment = -slope / 2000.0
            
            # Apply adjustment to CO2 duration
            new_co2_duration = _co2_duration + adjustment
            
            # If CO2 duration would go to zero or negative, keep it at zero
            if new_co2_duration <= 0:
                _co2_duration = 0.0
                bioreactor.logger.info(f"CO2 duration set to zero (slope adjustment would have made it negative)")
            else:
                # Update CO2 duration for future use
                _co2_duration = new_co2_duration
                bioreactor.logger.info(f"Updated CO2 duration to {_co2_duration:.3f}s for future injections (change: {adjustment:.3f}s, slope: {slope:.2f} ppm/s)")
    else:
        bioreactor.logger.warning("Insufficient data for linear fit")

