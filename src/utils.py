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
    'temperature': deque(maxlen=1000),
    'od_trx': deque(maxlen=1000),
    'od_sct': deque(maxlen=1000),
}

# Global figure and axes for plotting
_plot_fig = None
_plot_axes = None


def measure_and_plot_sensors(bioreactor, elapsed: Optional[float] = None, led_power: float = 30.0, averaging_duration: float = 0.5):
    """
    Measure, record, and plot sensor data from OD (Trx, Sct) and Temperature.
    
    This composite function:
    1. Reads all sensor values
    2. Writes data to CSV file
    3. Updates live plots in 2 subplots:
       - Subplot 1: Temperature
       - Subplot 2: OD voltages (Trx and Sct)
    
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
    from .io import get_temperature, read_voltage, measure_od
    
    # Get elapsed time
    if elapsed is None:
        if not hasattr(bioreactor, '_start_time'):
            bioreactor._start_time = time.time()
        elapsed = time.time() - bioreactor._start_time
    
    # Read sensors
    sensor_data = {'time': elapsed}
    
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
            _plot_fig, _plot_axes = plt.subplots(1, 2, figsize=(14, 5))
            _plot_fig.suptitle('Live Sensor Monitoring', fontsize=14)
            plt.ion()  # Turn on interactive mode
            plt.show(block=False)
        
        # Left: Temperature
        ax1 = _plot_axes[0]
        ax1.clear()
        ax1.set_title('Temperature')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Temperature (°C)')
        ax1.grid(True, alpha=0.3)
        if len(_plot_data['temperature']) > 0:
            ax1.plot(list(_plot_data['time']), list(_plot_data['temperature']), 'g-', linewidth=2, label='Temperature')
        ax1.legend()
        
        # Right: OD voltages
        ax2 = _plot_axes[1]
        ax2.clear()
        ax2.set_title('Optical Density Voltages')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Voltage (V)')
        ax2.grid(True, alpha=0.3)
        if len(_plot_data['od_trx']) > 0:
            ax2.plot(list(_plot_data['time']), list(_plot_data['od_trx']), 'm-', linewidth=2, label='Trx')
        if len(_plot_data['od_sct']) > 0:
            ax2.plot(list(_plot_data['time']), list(_plot_data['od_sct']), 'c-', linewidth=2, label='Sct')
        ax2.legend()
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)  # Small pause to update display
    
    bioreactor.logger.info(
        f"Sensor readings - Temp: {sensor_data.get('temperature', 'N/A'):.2f}°C, "
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

