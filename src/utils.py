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
# OD channels will be dynamically added based on config
_plot_data = {
    'time': deque(maxlen=1000),
    'temperature': deque(maxlen=1000),
}
PLOT_DATA_MAXLEN = 1000

# Global figure and axes for plotting
_plot_fig = None
_plot_axes = None


def measure_and_plot_sensors(bioreactor, elapsed: Optional[float] = None, led_power: float = 30.0, averaging_duration: float = 0.5):
    """
    Measure, record, and plot sensor data from OD channels and Temperature.
    
    This composite function:
    1. Reads all sensor values (dynamically based on config)
    2. Writes data to CSV file (field names from config)
    3. Updates live plots dynamically based on configured OD channels
    
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
    
    # Get config
    config = getattr(bioreactor, 'cfg', None)
    
    # Get OD channel names from config (keys of OD_ADC_CHANNELS dict)
    od_channel_names = []
    if config and hasattr(config, 'OD_ADC_CHANNELS'):
        od_channel_names = list(config.OD_ADC_CHANNELS.keys())
    elif hasattr(bioreactor, 'od_channels'):
        # Fallback: use channel names from initialized od_channels
        od_channel_names = list(bioreactor.od_channels.keys())
    
    # Initialize plot data storage for OD channels dynamically
    for ch_name in od_channel_names:
        plot_key = f"od_{ch_name.lower()}"
        if plot_key not in _plot_data:
            _plot_data[plot_key] = deque(maxlen=PLOT_DATA_MAXLEN)
    
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
    
    # Read OD channels dynamically based on config
    if bioreactor.is_component_initialized('led') and bioreactor.is_component_initialized('optical_density'):
        # Measure OD with LED on
        od_results = measure_od(bioreactor, led_power=led_power, averaging_duration=averaging_duration, channel_name='all')
        if od_results and od_channel_names:
            for ch_name in od_channel_names:
                plot_key = f"od_{ch_name.lower()}"
                od_value = od_results.get(ch_name, None)
                if od_value is not None:
                    sensor_data[plot_key] = od_value
                    _plot_data[plot_key].append(od_value)
                else:
                    sensor_data[plot_key] = float('nan')
                    _plot_data[plot_key].append(float('nan'))
        else:
            # No results, set all to NaN
            for ch_name in od_channel_names:
                plot_key = f"od_{ch_name.lower()}"
                sensor_data[plot_key] = float('nan')
                _plot_data[plot_key].append(float('nan'))
    else:
        # Try reading without LED if OD sensor is available but LED is not
        if bioreactor.is_component_initialized('optical_density') and od_channel_names:
            for ch_name in od_channel_names:
                plot_key = f"od_{ch_name.lower()}"
                od_value = read_voltage(bioreactor, ch_name)
                sensor_data[plot_key] = od_value if od_value is not None else float('nan')
                _plot_data[plot_key].append(sensor_data[plot_key])
        else:
            # No OD available, set all to NaN
            for ch_name in od_channel_names:
                plot_key = f"od_{ch_name.lower()}"
                sensor_data[plot_key] = float('nan')
                _plot_data[plot_key].append(float('nan'))
    
    # Add time to plot data
    _plot_data['time'].append(elapsed)
    
    # Write to CSV
    if hasattr(bioreactor, 'writer') and bioreactor.writer:
        csv_row = {'time': elapsed}
        
        # Add temperature with config label if available
        if config and hasattr(config, 'SENSOR_LABELS'):
            temp_label = config.SENSOR_LABELS.get('temperature', 'temperature_C')
            csv_row[temp_label] = sensor_data['temperature']
        else:
            csv_row['temperature'] = sensor_data['temperature']
        
        # Add OD data dynamically using config labels or auto-generate
        for ch_name in od_channel_names:
            plot_key = f"od_{ch_name.lower()}"
            if plot_key in sensor_data:
                # Try to get label from SENSOR_LABELS first
                if config and hasattr(config, 'SENSOR_LABELS'):
                    # Try multiple possible label keys
                    label = (config.SENSOR_LABELS.get(plot_key) or 
                            config.SENSOR_LABELS.get(f"od_{ch_name}") or
                            config.SENSOR_LABELS.get(f"od_{ch_name.lower()}") or
                            config.SENSOR_LABELS.get(f"od_{ch_name.upper()}") or
                            f"OD_{ch_name}_V")
                else:
                    # Auto-generate label
                    label = f"OD_{ch_name}_V"
                csv_row[label] = sensor_data[plot_key]
        
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
        
        # Right: OD voltages (dynamically plot all configured channels)
        ax2 = _plot_axes[1]
        ax2.clear()
        ax2.set_title('Optical Density Voltages')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Voltage (V)')
        ax2.grid(True, alpha=0.3)
        
        # Plot colors for different channels
        colors = ['m', 'c', 'b', 'r', 'g', 'y', 'k']
        for idx, ch_name in enumerate(od_channel_names):
            plot_key = f"od_{ch_name.lower()}"
            if plot_key in _plot_data and len(_plot_data[plot_key]) > 0:
                color = colors[idx % len(colors)]
                ax2.plot(list(_plot_data['time']), list(_plot_data[plot_key]), 
                        f'{color}-', linewidth=2, label=ch_name)
        ax2.legend()
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)  # Small pause to update display
    
    # Build log message dynamically
    log_parts = [f"Temp: {sensor_data.get('temperature', 'N/A'):.2f}°C"]
    for ch_name in od_channel_names:
        plot_key = f"od_{ch_name.lower()}"
        if plot_key in sensor_data:
            log_parts.append(f"OD {ch_name}: {sensor_data.get(plot_key, 'N/A'):.4f}V")
    bioreactor.logger.info(f"Sensor readings - {', '.join(log_parts)}")
    
    return sensor_data


def measure_and_record_sensors(bioreactor, elapsed: Optional[float] = None, led_power: float = 30.0, averaging_duration: float = 0.5):
    """
    Measure and record sensor data from OD channels and Temperature to CSV file (no plotting).
    
    This function:
    1. Reads all sensor values (dynamically based on config)
    2. Writes data to CSV file (field names from config)
    
    Args:
        bioreactor: Bioreactor instance
        elapsed: Elapsed time in seconds (if None, uses time since start)
        led_power: LED power percentage for OD measurements (default: 30.0)
        averaging_duration: Duration in seconds for averaging OD measurements (default: 0.5)
        
    Returns:
        dict: Dictionary with all sensor readings
    """
    # Import IO functions
    from .io import get_temperature, read_voltage, measure_od, read_all_eyespy_boards, read_eyespy_voltage, read_eyespy_adc
    
    # Get elapsed time
    if elapsed is None:
        if not hasattr(bioreactor, '_start_time'):
            bioreactor._start_time = time.time()
        elapsed = time.time() - bioreactor._start_time
    
    # Get config
    config = getattr(bioreactor, 'cfg', None)
    
    # Get OD channel names from config (keys of OD_ADC_CHANNELS dict)
    od_channel_names = []
    if config and hasattr(config, 'OD_ADC_CHANNELS'):
        od_channel_names = list(config.OD_ADC_CHANNELS.keys())
    elif hasattr(bioreactor, 'od_channels'):
        # Fallback: use channel names from initialized od_channels
        od_channel_names = list(bioreactor.od_channels.keys())
    
    # Read sensors
    sensor_data = {'time': elapsed}
    
    # Read Temperature
    temp_value = get_temperature(bioreactor, sensor_index=0)
    if not np.isnan(temp_value):
        sensor_data['temperature'] = temp_value
    else:
        sensor_data['temperature'] = float('nan')
    
    # Read OD channels and/or eyespy with LED on if LED is initialized
    # measure_od() handles turning LED on, taking readings, and turning LED off
    # It works with OD only, eyespy only, or both
    led_initialized = bioreactor.is_component_initialized('led')
    od_initialized = bioreactor.is_component_initialized('optical_density')
    eyespy_initialized = bioreactor.is_component_initialized('eyespy_adc')
    
    if led_initialized and (od_initialized or eyespy_initialized):
        # Measure with LED on (reads OD channels and/or eyespy if initialized)
        od_results = measure_od(bioreactor, led_power=led_power, averaging_duration=averaging_duration, channel_name='all')
        if od_results:
            # Extract OD channel readings (if OD is initialized)
            if od_initialized and od_channel_names:
                for ch_name in od_channel_names:
                    plot_key = f"od_{ch_name.lower()}"
                    od_value = od_results.get(ch_name, None)
                    if od_value is not None:
                        sensor_data[plot_key] = od_value
                    else:
                        sensor_data[plot_key] = float('nan')
            elif od_channel_names:
                # OD channels requested but not initialized - set to NaN
                for ch_name in od_channel_names:
                    plot_key = f"od_{ch_name.lower()}"
                    sensor_data[plot_key] = float('nan')
            
            # Extract eyespy readings from od_results (averaged voltages with LED on)
            if eyespy_initialized and hasattr(bioreactor, 'eyespy_boards'):
                for board_name in bioreactor.eyespy_boards.keys():
                    eyespy_voltage = od_results.get(board_name, None)
                    if eyespy_voltage is not None:
                        # Store the averaged voltage from measure_od (LED was on during measurement)
                        sensor_data[f"eyespy_{board_name}_voltage"] = eyespy_voltage
                        # Also get raw value for completeness (single reading after LED is off)
                        # Note: This raw value is NOT used to recalculate voltage - the averaged voltage above is used
                        raw_value = read_eyespy_adc(bioreactor, board_name)
                        sensor_data[f"eyespy_{board_name}_raw"] = raw_value if raw_value is not None else float('nan')
                    else:
                        sensor_data[f"eyespy_{board_name}_voltage"] = float('nan')
                        sensor_data[f"eyespy_{board_name}_raw"] = float('nan')
        else:
            # No results, set all to NaN
            if od_channel_names:
                for ch_name in od_channel_names:
                    plot_key = f"od_{ch_name.lower()}"
                    sensor_data[plot_key] = float('nan')
            # Also set eyespy to NaN if initialized
            if eyespy_initialized and hasattr(bioreactor, 'eyespy_boards'):
                for board_name in bioreactor.eyespy_boards.keys():
                    sensor_data[f"eyespy_{board_name}_voltage"] = float('nan')
                    sensor_data[f"eyespy_{board_name}_raw"] = float('nan')
    else:
        # Try reading without LED if OD sensor is available but LED is not
        if bioreactor.is_component_initialized('optical_density') and od_channel_names:
            for ch_name in od_channel_names:
                plot_key = f"od_{ch_name.lower()}"
                od_value = read_voltage(bioreactor, ch_name)
                sensor_data[plot_key] = od_value if od_value is not None else float('nan')
        else:
            # No OD available, set all to NaN
            for ch_name in od_channel_names:
                plot_key = f"od_{ch_name.lower()}"
                sensor_data[plot_key] = float('nan')
    
    # Eyespy ADC readings when LED is not initialized (read separately without LED)
    # Note: If LED is initialized, eyespy should have been read above via measure_od()
    if eyespy_initialized and not led_initialized:
        eyespy_readings = read_all_eyespy_boards(bioreactor)
        if eyespy_readings:
            for board_name, raw_value in eyespy_readings.items():
                if raw_value is not None:
                    # Store raw value
                    sensor_data[f"eyespy_{board_name}_raw"] = raw_value
                    # Also get voltage (single reading, LED off)
                    voltage = read_eyespy_voltage(bioreactor, board_name)
                    if voltage is not None:
                        sensor_data[f"eyespy_{board_name}_voltage"] = voltage
                    else:
                        sensor_data[f"eyespy_{board_name}_voltage"] = float('nan')
                else:
                    sensor_data[f"eyespy_{board_name}_raw"] = float('nan')
                    sensor_data[f"eyespy_{board_name}_voltage"] = float('nan')
    
    # Write to CSV
    if hasattr(bioreactor, 'writer') and bioreactor.writer:
        csv_row = {'time': elapsed}
        
        # Add temperature with config label if available
        if config and hasattr(config, 'SENSOR_LABELS'):
            temp_label = config.SENSOR_LABELS.get('temperature', 'temperature_C')
            csv_row[temp_label] = sensor_data['temperature']
        else:
            csv_row['temperature'] = sensor_data['temperature']
        
        # Add OD data dynamically using config labels or auto-generate
        for ch_name in od_channel_names:
            plot_key = f"od_{ch_name.lower()}"
            if plot_key in sensor_data:
                # Try to get label from SENSOR_LABELS first
                if config and hasattr(config, 'SENSOR_LABELS'):
                    # Try multiple possible label keys
                    label = (config.SENSOR_LABELS.get(plot_key) or 
                            config.SENSOR_LABELS.get(f"od_{ch_name}") or
                            config.SENSOR_LABELS.get(f"od_{ch_name.lower()}") or
                            config.SENSOR_LABELS.get(f"od_{ch_name.upper()}") or
                            f"OD_{ch_name}_V")
                else:
                    # Auto-generate label
                    label = f"OD_{ch_name}_V"
                csv_row[label] = sensor_data[plot_key]
        
        # Add eyespy ADC data dynamically
        if bioreactor.is_component_initialized('eyespy_adc') and hasattr(bioreactor, 'eyespy_boards'):
            for board_name in bioreactor.eyespy_boards.keys():
                raw_key = f"eyespy_{board_name}_raw"
                voltage_key = f"eyespy_{board_name}_voltage"
                
                # Get labels from config or auto-generate
                if config and hasattr(config, 'SENSOR_LABELS'):
                    raw_label = config.SENSOR_LABELS.get(raw_key, f"Eyespy_{board_name}_raw")
                    voltage_label = config.SENSOR_LABELS.get(voltage_key, f"Eyespy_{board_name}_V")
                else:
                    raw_label = f"Eyespy_{board_name}_raw"
                    voltage_label = f"Eyespy_{board_name}_V"
                
                # Write raw value if available
                if raw_key in sensor_data:
                    csv_row[raw_label] = sensor_data[raw_key]
                
                # Write voltage value - this should be the averaged voltage from measure_od (with LED on)
                if voltage_key in sensor_data:
                    voltage_value = sensor_data[voltage_key]
                    csv_row[voltage_label] = voltage_value
                    # Debug: verify we're writing the correct averaged value
                    if not np.isnan(voltage_value):
                        bioreactor.logger.debug(f"Writing eyespy {board_name} voltage to CSV: {voltage_value:.4f}V (label: {voltage_label})")
        
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
    
    # Build log message dynamically
    log_parts = [f"Temp: {sensor_data.get('temperature', 'N/A'):.2f}°C"]
    for ch_name in od_channel_names:
        plot_key = f"od_{ch_name.lower()}"
        if plot_key in sensor_data:
            log_parts.append(f"OD {ch_name}: {sensor_data.get(plot_key, 'N/A'):.4f}V")
    
    # Add eyespy readings to log
    if bioreactor.is_component_initialized('eyespy_adc') and hasattr(bioreactor, 'eyespy_boards'):
        for board_name in bioreactor.eyespy_boards.keys():
            voltage_key = f"eyespy_{board_name}_voltage"
            if voltage_key in sensor_data:
                voltage = sensor_data[voltage_key]
                if not np.isnan(voltage):
                    log_parts.append(f"Eyespy {board_name}: {voltage:.4f}V")
    
    bioreactor.logger.info(f"Sensor readings - {', '.join(log_parts)}")
    
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


def ring_light_cycle(
    bioreactor,
    color: tuple = (50, 50, 50),
    on_time: float = 60.0,
    off_time: float = 60.0,
    elapsed: Optional[float] = None
) -> None:
    """
    Cycle ring light on and off in a loop.
    
    This function starts by turning the ring light on at a specified color,
    then alternates between on and off for specified durations, repeating continuously.
    Starts with ring light ON first.
    
    Args:
        bioreactor: Bioreactor instance
        color: RGB tuple (r, g, b) with values 0-255 (default: (50, 50, 50))
        on_time: Duration in seconds to keep ring light on (default: 60.0)
        off_time: Duration in seconds to keep ring light off (default: 60.0)
        elapsed: Elapsed time since start (s). Used internally for timing.
        
    Note:
        State (_ring_light_state, _ring_light_last_switch_time) is stored on bioreactor instance.
        The function automatically initializes state on first call, starting with ring light ON.
        
    Example usage as a job:
        from functools import partial
        from src.utils import ring_light_cycle
        
        # Create a partial function with custom color and timing
        ring_light_job = partial(ring_light_cycle, color=(100, 100, 100), on_time=30.0, off_time=30.0)
        
        # Add to jobs list - call frequently (every 1 second) to check timing
        jobs = [
            (ring_light_job, 1, True),  # Check every 1 second
        ]
        reactor.run(jobs)
    """
    from .io import set_ring_light, turn_off_ring_light
    
    if not bioreactor.is_component_initialized('ring_light'):
        bioreactor.logger.warning("Ring light not initialized; skipping cycle")
        return
    
    # Initialize state if not present - start with ring light ON
    if not hasattr(bioreactor, '_ring_light_state'):
        bioreactor._ring_light_state = 'on'  # Start with ring light on
        bioreactor._ring_light_last_switch_time = None
        # Turn ring light on immediately on first call
        if set_ring_light(bioreactor, color):
            bioreactor.logger.info(
                f"Ring light cycle started: turned ON with color={color}, will stay on for {on_time}s"
            )
    
    # Get current time
    if elapsed is None:
        if not hasattr(bioreactor, '_ring_light_start_time'):
            bioreactor._ring_light_start_time = time.time()
        current_time = time.time() - bioreactor._ring_light_start_time
    else:
        current_time = elapsed
    
    # Initialize last switch time on first call
    if bioreactor._ring_light_last_switch_time is None:
        bioreactor._ring_light_last_switch_time = current_time
    
    # Calculate time since last state switch
    time_since_switch = current_time - bioreactor._ring_light_last_switch_time
    
    # Determine if we need to switch state
    if bioreactor._ring_light_state == 'on':
        # Currently on - check if we should turn off
        if time_since_switch >= on_time:
            # Turn ring light off
            turn_off_ring_light(bioreactor)
            bioreactor._ring_light_state = 'off'
            bioreactor._ring_light_last_switch_time = current_time
            bioreactor.logger.info(
                f"Ring light turned OFF, will stay off for {off_time}s"
            )
    else:  # state == 'off'
        # Currently off - check if we should turn on
        if time_since_switch >= off_time:
            # Turn ring light on
            if set_ring_light(bioreactor, color):
                bioreactor._ring_light_state = 'on'
                bioreactor._ring_light_last_switch_time = current_time
                bioreactor.logger.info(
                    f"Ring light turned ON: color={color}, will stay on for {on_time}s"
                )

