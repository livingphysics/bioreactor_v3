"""
Composite utility functions for bioreactor operations.
These functions combine multiple operations or provide higher-level convenience wrappers.
These functions are designed to be used with bioreactor.run() for scheduled tasks.
"""

import time
import logging
import threading
from typing import Union, Optional
from datetime import datetime
import numpy as np

logger = logging.getLogger("Bioreactor.Utils")


def _standalone_ekf_update(bioreactor, sensor_data, elapsed):
    """Run a standalone EKF update to estimate OD and growth rate.

    This is called by measure_and_record_sensors when no turbidostat EKF job
    is running (i.e. bioreactor.ekf_estimates does not already exist).  It
    uses the same maths as turbidostat_ekf_mode but without any pump logic.
    """
    # --- Pick the OD channel from sensor_data ---
    eyespy_init = bioreactor.is_component_initialized('eyespy_adc')
    od_init = bioreactor.is_component_initialized('optical_density')

    if eyespy_init:
        z_k = sensor_data.get('eyespy_sct_voltage', float('nan'))
    elif od_init:
        z_k = sensor_data.get('od_135', float('nan'))
    else:
        return  # no OD sensor available

    if np.isnan(z_k):
        return  # no valid reading this cycle

    # --- EKF defaults ---
    R = 0.003
    Q_growth_rate = 5e-12
    initial_P_r = 0.0005 ** 2
    pump_distrust_P_od = 10.0 * R

    # --- Initialize on first call ---
    if not getattr(bioreactor, '_ekf_initialized', False):
        bioreactor._ekf_state = np.array([z_k, 1.0])
        bioreactor._ekf_P = np.array([
            [R, 0.0],
            [0.0, initial_P_r],
        ])
        bioreactor._ekf_pump_distrust_counter = 0
        bioreactor._ekf_last_time = elapsed if elapsed is not None else time.time()
        bioreactor._ekf_initialized = True
        bioreactor.logger.info(
            f"Standalone EKF initialized: OD={z_k:.4f}, r=1.0"
        )
        return

    # --- Compute dt_cycle ---
    current_time = elapsed if elapsed is not None else time.time()
    dt_cycle = current_time - bioreactor._ekf_last_time
    if dt_cycle <= 0:
        dt_cycle = 1.0
    bioreactor._ekf_last_time = current_time

    # --- EKF Predict ---
    od_k = bioreactor._ekf_state[0]
    r_k = bioreactor._ekf_state[1]
    P = bioreactor._ekf_P

    x_pred = np.array([od_k * r_k, r_k])

    F = np.array([
        [r_k, od_k],
        [0.0, 1.0],
    ])

    Q = np.array([
        [0.0, 0.0],
        [0.0, Q_growth_rate],
    ])

    P_pred = F @ P @ F.T + Q

    # --- Pump distrust (honours counter set by other jobs) ---
    currently_pumping = getattr(bioreactor, 'pumping_active', False)
    if currently_pumping or getattr(bioreactor, '_ekf_pump_distrust_counter', 0) > 0:
        P_pred[0, 0] = pump_distrust_P_od
        P_pred[0, 1] = 0.0
        P_pred[1, 0] = 0.0
        x_pred[0] = z_k
        if not currently_pumping:
            bioreactor._ekf_pump_distrust_counter -= 1

    # --- EKF Update (H = [1, 0]) ---
    y = z_k - x_pred[0]
    S = P_pred[0, 0] + R
    K = P_pred[:, 0] / S
    x_updated = x_pred + K * y

    KH = np.outer(K, np.array([1.0, 0.0]))
    P_updated = (np.eye(2) - KH) @ P_pred

    # Secondary 5-sigma reset
    innovation_threshold = 5.0 * np.sqrt(R)
    if abs(z_k - x_pred[0]) > innovation_threshold:
        P_updated[0, 1] = 0.0
        P_updated[1, 0] = 0.0
        P_updated[0, 0] = (x_updated[0] - z_k) ** 2

    bioreactor._ekf_state = x_updated
    bioreactor._ekf_P = P_updated

    od_est = x_updated[0]
    r_est = x_updated[1]

    # --- Doubling time ---
    od_std = np.sqrt(P_updated[0, 0])
    r_std = np.sqrt(P_updated[1, 1])
    if r_est > 1.0 and dt_cycle > 0:
        ln_r = np.log(r_est)
        doubling_time = dt_cycle * np.log(2.0) / ln_r
        doubling_time_std = dt_cycle * np.log(2.0) * r_std / (r_est * ln_r ** 2)
    else:
        doubling_time = float('inf')
        doubling_time_std = float('inf')

    bioreactor.ekf_estimates = {
        'ekf_od_est': od_est,
        'ekf_growth_rate': r_est,
        'ekf_doubling_time_s': doubling_time,
        'ekf_od_std': od_std,
        'ekf_growth_rate_std': r_std,
        'ekf_doubling_time_std_s': doubling_time_std,
    }


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
    from .io import get_temperature, read_voltage, measure_od, read_all_eyespy_boards, read_eyespy_voltage, read_eyespy_adc, read_co2, read_o2, get_peltier_state, get_ring_light_color
    
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
    sensor_data = {'elapsed_time': elapsed}
    
    # Read Temperature only if temp_sensor is initialized
    if bioreactor.is_component_initialized('temp_sensor'):
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
    
    # Read CO2 sensor if initialized
    if bioreactor.is_component_initialized('co2_sensor'):
        co2_value = read_co2(bioreactor)
        if co2_value is not None:
            # read_co2 already returns value multiplied by 10 to get PPM
            sensor_data['co2'] = co2_value
        else:
            sensor_data['co2'] = float('nan')
    else:
        sensor_data['co2'] = float('nan')
    
    # Read O2 sensor if initialized
    if bioreactor.is_component_initialized('o2_sensor'):
        o2_value = read_o2(bioreactor)
        if o2_value is not None:
            sensor_data['o2'] = o2_value
        else:
            sensor_data['o2'] = float('nan')
    else:
        sensor_data['o2'] = float('nan')
    
    # Current peltier duty and direction (from driver state, not a sensor)
    if bioreactor.is_component_initialized('peltier_driver'):
        peltier_state = get_peltier_state(bioreactor)
        if peltier_state is not None:
            duty, forward = peltier_state
            sensor_data['peltier_duty'] = duty
            sensor_data['peltier_forward'] = 1.0 if forward else 0.0
        else:
            sensor_data['peltier_duty'] = float('nan')
            sensor_data['peltier_forward'] = float('nan')
    else:
        sensor_data['peltier_duty'] = float('nan')
        sensor_data['peltier_forward'] = float('nan')
    
    # Current ring light color (R, G, B 0-255)
    if bioreactor.is_component_initialized('ring_light'):
        ring_color = get_ring_light_color(bioreactor)
        if ring_color is not None:
            sensor_data['ring_light_R'] = float(ring_color[0])
            sensor_data['ring_light_G'] = float(ring_color[1])
            sensor_data['ring_light_B'] = float(ring_color[2])
        else:
            sensor_data['ring_light_R'] = float('nan')
            sensor_data['ring_light_G'] = float('nan')
            sensor_data['ring_light_B'] = float('nan')
    else:
        sensor_data['ring_light_R'] = float('nan')
        sensor_data['ring_light_G'] = float('nan')
        sensor_data['ring_light_B'] = float('nan')
    
    # Write to CSV
    if hasattr(bioreactor, 'writer') and bioreactor.writer:
        # Get current timestamp
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        csv_row = {
            'time': current_time,  # Actual timestamp
            'elapsed_time': elapsed  # Elapsed seconds since start
        }
        
        # Add temperature with config label if temp_sensor is initialized
        if bioreactor.is_component_initialized('temp_sensor') and 'temperature' in sensor_data:
            if config and hasattr(config, 'SENSOR_LABELS'):
                temp_label = config.SENSOR_LABELS.get('temperature', 'temperature_C')
            else:
                temp_label = 'temperature_C'
            csv_row[temp_label] = sensor_data['temperature']
        
        # Add OD data dynamically using config labels or auto-generate (only if optical_density is initialized)
        if bioreactor.is_component_initialized('optical_density'):
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
        
        # Add CO2 data if sensor is initialized
        if bioreactor.is_component_initialized('co2_sensor') and 'co2' in sensor_data:
            # Get label from config or auto-generate
            if config and hasattr(config, 'SENSOR_LABELS'):
                co2_label = config.SENSOR_LABELS.get('co2', 'CO2_ppm')
            else:
                co2_label = 'CO2_ppm'
            csv_row[co2_label] = sensor_data['co2']
        
        # Add O2 data if sensor is initialized
        if bioreactor.is_component_initialized('o2_sensor') and 'o2' in sensor_data:
            # Get label from config or auto-generate
            if config and hasattr(config, 'SENSOR_LABELS'):
                o2_label = config.SENSOR_LABELS.get('o2', 'O2_percent')
            else:
                o2_label = 'O2_percent'
            csv_row[o2_label] = sensor_data['o2']
        
        # Add peltier state if peltier_driver is initialized
        if bioreactor.is_component_initialized('peltier_driver'):
            if config and hasattr(config, 'SENSOR_LABELS'):
                duty_label = config.SENSOR_LABELS.get('peltier_duty', 'peltier_duty')
                fwd_label = config.SENSOR_LABELS.get('peltier_forward', 'peltier_forward')
            else:
                duty_label, fwd_label = 'peltier_duty', 'peltier_forward'
            csv_row[duty_label] = sensor_data['peltier_duty']
            csv_row[fwd_label] = sensor_data['peltier_forward']
        
        # Add ring light color if ring_light is initialized
        if bioreactor.is_component_initialized('ring_light'):
            for ch in ('R', 'G', 'B'):
                key = f'ring_light_{ch}'
                if config and hasattr(config, 'SENSOR_LABELS'):
                    label = config.SENSOR_LABELS.get(key, key)
                else:
                    label = key
                csv_row[label] = sensor_data[key]

        # Add cumulative pump run times if tracked
        if hasattr(bioreactor, 'pump_run_times') and bioreactor.pump_run_times:
            for pname, total_time in bioreactor.pump_run_times.items():
                csv_row[f"pump_{pname}_time_s"] = total_time

        # Standalone EKF: compute estimates if no turbidostat EKF job is running
        if not getattr(bioreactor, '_turbidostat_ekf_active', False):
            _standalone_ekf_update(bioreactor, sensor_data, elapsed)

        # Add EKF estimates if available (from turbidostat or standalone)
        if hasattr(bioreactor, 'ekf_estimates'):
            for key, value in bioreactor.ekf_estimates.items():
                csv_row[key] = value

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

    # Build log message dynamically (only include initialized sensors)
    log_parts = []

    # Add temperature to log only if temp_sensor is initialized
    if bioreactor.is_component_initialized('temp_sensor') and 'temperature' in sensor_data:
        temp_value = sensor_data.get('temperature', float('nan'))
        if not np.isnan(temp_value):
            log_parts.append(f"Temp: {temp_value:.2f}°C")
    
    # Add OD channels to log only if optical_density is initialized
    if bioreactor.is_component_initialized('optical_density'):
        for ch_name in od_channel_names:
            plot_key = f"od_{ch_name.lower()}"
            if plot_key in sensor_data:
                od_value = sensor_data.get(plot_key, float('nan'))
                if not np.isnan(od_value):
                    log_parts.append(f"OD {ch_name}: {od_value:.4f}V")
    
    # Add eyespy readings to log
    if bioreactor.is_component_initialized('eyespy_adc') and hasattr(bioreactor, 'eyespy_boards'):
        for board_name in bioreactor.eyespy_boards.keys():
            voltage_key = f"eyespy_{board_name}_voltage"
            if voltage_key in sensor_data:
                voltage = sensor_data[voltage_key]
                if not np.isnan(voltage):
                    log_parts.append(f"Eyespy {board_name}: {voltage:.4f}V")
    
    # Add CO2 reading to log
    if bioreactor.is_component_initialized('co2_sensor') and 'co2' in sensor_data:
        co2_value = sensor_data['co2']
        if not np.isnan(co2_value):
            # Value is already in PPM (multiplied by 10 in read_co2)
            log_parts.append(f"CO2: {co2_value:.0f} ppm")
    
    # Add O2 reading to log
    if bioreactor.is_component_initialized('o2_sensor') and 'o2' in sensor_data:
        o2_value = sensor_data['o2']
        if not np.isnan(o2_value):
            log_parts.append(f"O2: {o2_value:.2f}%")
    
    # Add peltier state to log
    if bioreactor.is_component_initialized('peltier_driver') and 'peltier_duty' in sensor_data:
        duty = sensor_data.get('peltier_duty', float('nan'))
        fwd = sensor_data.get('peltier_forward', 1.0)
        if not np.isnan(duty):
            dir_str = 'fwd' if fwd == 1.0 else 'rev'
            log_parts.append(f"Peltier: {duty:.1f}% {dir_str}")
    
    # Add ring light color to log
    if bioreactor.is_component_initialized('ring_light') and 'ring_light_R' in sensor_data:
        r, g, b = sensor_data.get('ring_light_R', 0), sensor_data.get('ring_light_G', 0), sensor_data.get('ring_light_B', 0)
        if not (np.isnan(r) or np.isnan(g) or np.isnan(b)):
            log_parts.append(f"Ring: ({int(r)},{int(g)},{int(b)})")
    
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
    max_duty_heat: Optional[float] = None,
    max_duty_cool: Optional[float] = None,
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
        kp: Proportional gain (default: 12.0)
        ki: Integral gain (default: 0.015)
        kd: Derivative gain (default: 0.0)
        dt: Time elapsed since last call (s). If None, uses elapsed parameter or estimates.
        elapsed: Elapsed time since start (s). Used to estimate dt if dt is None.
        sensor_index: Index of temperature sensor to read (default: 0)
        max_duty_heat: Max duty for heating (0-100). None = use config.PELTIER_MAX_DUTY_HEAT (default 70)
        max_duty_cool: Max duty for cooling (0-100). None = use config.PELTIER_MAX_DUTY_COOL (default 70)
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
        
        # Determine direction based on PID output:
        # error = setpoint - current_temp
        # If error > 0 (too cold), output > 0, we need to HEAT
        # If error < 0 (too hot), output < 0, we need to COOL
        direction = 'heat' if output > 0 else 'cool'
        
        # Resolve max duty from config if not explicitly provided
        config = getattr(bioreactor, 'cfg', None)
        limit_heat = max_duty_heat if max_duty_heat is not None else (getattr(config, 'PELTIER_MAX_DUTY_HEAT', 70.0) if config else 70.0)
        limit_cool = max_duty_cool if max_duty_cool is not None else (getattr(config, 'PELTIER_MAX_DUTY_COOL', 70.0) if config else 70.0)
        max_duty = limit_heat if direction == 'heat' else limit_cool
        
        # Convert output to duty cycle (0-100) and clamp to max_duty (hardware safety limit)
        duty = max(0, min(max_duty, abs(output)))
        
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


def temperature_profile(
    bioreactor,
    profile: list,
    kp: float = 12.0,
    ki: float = 0.015,
    kd: float = 0.0,
    sensor_index: int = 0,
    max_duty_heat: Optional[float] = None,
    max_duty_cool: Optional[float] = None,
    elapsed: Optional[float] = None,
) -> None:
    """
    Run a temperature profile that changes setpoint over time.

    The profile is a list of (duration_seconds, setpoint_celsius) tuples
    executed in order. The last setpoint is held indefinitely after all
    steps complete.

    Example:
        # 30°C for 3 hours, then 25°C
        partial(temperature_profile, profile=[(3*3600, 30.0), (None, 25.0)])

    Args:
        bioreactor: Bioreactor instance
        profile: List of (duration, setpoint) tuples. duration is in seconds.
                 Use None for the last duration to hold indefinitely.
        kp: Proportional gain (default: 12.0)
        ki: Integral gain (default: 0.015)
        kd: Derivative gain (default: 0.0)
        sensor_index: Temperature sensor index (default: 0)
        max_duty_heat: Max duty for heating. None = use config.PELTIER_MAX_DUTY_HEAT
        max_duty_cool: Max duty for cooling. None = use config.PELTIER_MAX_DUTY_COOL
        elapsed: Elapsed time in seconds (passed by bioreactor.run scheduler)
    """
    if elapsed is None or not profile:
        return

    # Determine which profile step we're in
    t = 0.0
    setpoint = profile[-1][1]  # default to last step
    for duration, sp in profile:
        if duration is None:
            setpoint = sp
            break
        if elapsed < t + duration:
            setpoint = sp
            break
        t += duration
        setpoint = sp  # carry forward in case we're past all steps

    temperature_pid_controller(
        bioreactor,
        setpoint=setpoint,
        kp=kp,
        ki=ki,
        kd=kd,
        elapsed=elapsed,
        sensor_index=sensor_index,
        max_duty_heat=max_duty_heat,
        max_duty_cool=max_duty_cool,
    )


def ring_light_cycle(
    bioreactor,
    color: tuple = (50, 50, 50),
    on_time: float = 60.0,
    off_time: float = 60.0,
    start_on: bool = True,
    elapsed: Optional[float] = None
) -> None:
    """
    Cycle ring light on and off in a loop.
    
    Alternates between on (at the specified color) and off for the given durations.
    Can start with the light on first or off first via start_on.
    
    Args:
        bioreactor: Bioreactor instance
        color: RGB tuple (r, g, b) with values 0-255 (default: (50, 50, 50))
        on_time: Duration in seconds to keep ring light on (default: 60.0)
        off_time: Duration in seconds to keep ring light off (default: 60.0)
        start_on: If True (default), start with light ON first; if False, start with light OFF first.
        elapsed: Elapsed time since start (s). Used internally for timing.
        
    Note:
        State (_ring_light_state, _ring_light_last_switch_time) is stored on bioreactor instance.
        The function automatically initializes state on first call.
        
    Example usage as a job:
        from functools import partial
        from src.utils import ring_light_cycle
        
        # Start with light on first (default)
        ring_light_job = partial(ring_light_cycle, color=(100, 100, 100), on_time=30.0, off_time=30.0)
        # Start with light off first
        ring_light_job = partial(ring_light_cycle, color=(100, 100, 100), on_time=30.0, off_time=30.0, start_on=False)
        
        jobs = [(ring_light_job, 1, True)]
        reactor.run(jobs)
    """
    from .io import set_ring_light, turn_off_ring_light
    
    if not bioreactor.is_component_initialized('ring_light'):
        bioreactor.logger.warning("Ring light not initialized; skipping cycle")
        return
    
    # Initialize state if not present
    if not hasattr(bioreactor, '_ring_light_state'):
        bioreactor._ring_light_last_switch_time = None
        if start_on:
            bioreactor._ring_light_state = 'on'
            if set_ring_light(bioreactor, color):
                bioreactor.logger.info(
                    f"Ring light cycle started: turned ON with color={color}, will stay on for {on_time}s"
                )
        else:
            bioreactor._ring_light_state = 'off'
            turn_off_ring_light(bioreactor)
            bioreactor.logger.info(
                f"Ring light cycle started: turned OFF first, will stay off for {off_time}s"
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


def balanced_flow(bioreactor, pump_name: str, ml_per_sec: float, elapsed: Optional[float] = None, duration: Optional[float] = None) -> None:
    """
    !!! Use ticgui to disable command timeout. !!!
    Set balanced flow: for a given pump, set its flow and automatically set the
    converse pump (inflow/outflow pair) to the same volumetric rate in the opposite direction.
    
    This is used for chemostat mode where inflow and outflow must be balanced.
    
    Args:
        bioreactor: Bioreactor instance
        pump_name: Name of the pump (e.g., 'inflow' or 'outflow')
                  If pump_name is 'inflow', sets both 'inflow' and 'outflow' to the same rate.
                  If pump_name is 'outflow', sets both 'outflow' and 'inflow' to the same rate.
        ml_per_sec: Desired flow rate in ml/sec (>= 0)
        elapsed: Elapsed time (unused, for compatibility with job functions)
        duration: Optional duration in seconds to run the pumps. If provided, pumps will run
                 for this duration and then stop. Must be less than the job frequency.
                 If None, pumps run continuously.
    """
    from .io import change_pump
    
    if not bioreactor.is_component_initialized('pumps'):
        bioreactor.logger.warning("Pumps not initialized; cannot set balanced flow")
        return
    
    if not hasattr(bioreactor, 'pumps'):
        bioreactor.logger.warning("Pumps not available")
        return
    
    # Determine the converse pump name
    # Default assumption: if 'inflow' exists, 'outflow' is the converse, and vice versa
    if pump_name == 'inflow':
        converse_name = 'outflow'
    elif pump_name == 'outflow':
        converse_name = 'inflow'
    else:
        # Try to infer from name patterns
        if pump_name.endswith('_in') or pump_name.endswith('_inflow'):
            # Remove suffix and add outflow suffix
            base = pump_name.rsplit('_', 1)[0] if '_' in pump_name else pump_name
            converse_name = f"{base}_out" if not base.endswith('out') else f"{base}_outflow"
        elif pump_name.endswith('_out') or pump_name.endswith('_outflow'):
            base = pump_name.rsplit('_', 1)[0] if '_' in pump_name else pump_name
            converse_name = f"{base}_in" if not base.endswith('in') else f"{base}_inflow"
        else:
            bioreactor.logger.warning(
                f"Cannot determine converse pump for '{pump_name}'. "
                f"Setting only the specified pump. Available pumps: {list(bioreactor.pumps.keys())}"
            )
            try:
                change_pump(bioreactor, pump_name, ml_per_sec)
                if duration is not None and duration > 0:
                    time.sleep(duration)
                    change_pump(bioreactor, pump_name, 0.0)
                    bioreactor.logger.info(f"Pump {pump_name} stopped after {duration:.2f} seconds")
            except Exception as e:
                bioreactor.logger.error(f"Error setting pump {pump_name}: {e}")
            return
    
    # Check if both pumps exist
    if pump_name not in bioreactor.pumps:
        bioreactor.logger.error(f"Pump '{pump_name}' not found. Available: {list(bioreactor.pumps.keys())}")
        return
    
    if converse_name not in bioreactor.pumps:
        bioreactor.logger.warning(
            f"Converse pump '{converse_name}' not found. "
            f"Setting only '{pump_name}'. Available pumps: {list(bioreactor.pumps.keys())}"
        )
        try:
            change_pump(bioreactor, pump_name, ml_per_sec)
            if duration is not None and duration > 0:
                time.sleep(duration)
                change_pump(bioreactor, pump_name, 0.0)
                bioreactor.logger.info(f"Pump {pump_name} stopped after {duration:.2f} seconds")
        except Exception as e:
            bioreactor.logger.error(f"Error setting pump {pump_name}: {e}")
        return
    
    # Set both pumps to the same rate
    try:
        change_pump(bioreactor, pump_name, ml_per_sec)
        change_pump(bioreactor, converse_name, ml_per_sec)
        
        if duration is not None:
            if duration <= 0:
                bioreactor.logger.warning(f"Duration must be positive, got {duration}. Ignoring duration parameter.")
            else:
                bioreactor.logger.info(
                    f"Balanced flow: {pump_name} and {converse_name} set to {ml_per_sec:.4f} ml/sec for {duration:.2f} seconds"
                )
                time.sleep(duration)
                # Stop both pumps after duration
                change_pump(bioreactor, pump_name, 0.0)
                change_pump(bioreactor, converse_name, 0.0)
                bioreactor.logger.info(
                    f"Balanced flow: {pump_name} and {converse_name} stopped after {duration:.2f} seconds"
                )
        else:
            bioreactor.logger.info(
                f"Balanced flow: {pump_name} and {converse_name} set to {ml_per_sec:.4f} ml/sec (continuous)"
            )
    except Exception as e:
        bioreactor.logger.error(f"Failed to set balanced flow: {e}")


def chemostat_mode(
    bioreactor,
    pump_name: str,
    flow_rate_ml_s: float,
    temp_setpoint: Optional[float] = None,
    kp: float = 12.0,
    ki: float = 0.015,
    kd: float = 0.0,
    dt: Optional[float] = None,
    elapsed: Optional[float] = None,
    sensor_index: int = 0,
    max_duty_heat: Optional[float] = None,
    max_duty_cool: Optional[float] = None,
    flow_freq: float = 1.0,
    temp_freq: float = 1.0,
) -> None:
    """
    Run the bioreactor in chemostat mode:
    - Balanced flow on the specified pump (inflow and outflow at same rate)
    - Optional PID temperature control
    
    This function is designed to be called as a job in bioreactor.run().
    It sets balanced flow every flow_freq seconds and optionally controls temperature.
    
    Args:
        bioreactor: Bioreactor instance
        pump_name: Name of the pump to use for balanced flow (e.g., 'inflow' or 'outflow')
        flow_rate_ml_s: Inflow/outflow rate (ml/sec)
        temp_setpoint: Optional desired temperature (°C). If None, only flow control is active.
        kp: Proportional gain for PID (default: 12.0)
        ki: Integral gain for PID (default: 0.015)
        kd: Derivative gain for PID (default: 0.0)
        dt: Time step for PID loop (s). If None, uses temp_freq.
        elapsed: Elapsed time since start (s). Used internally.
        sensor_index: Index of temperature sensor to read (default: 0)
        max_duty_heat: Max duty for heating. None = use config.PELTIER_MAX_DUTY_HEAT
        max_duty_cool: Max duty for cooling. None = use config.PELTIER_MAX_DUTY_COOL
        flow_freq: Frequency (s) for balanced flow updates (default: 1.0)
        temp_freq: Frequency (s) for temperature PID updates (default: 1.0)
    """
    # Set balanced flow
    balanced_flow(bioreactor, pump_name, flow_rate_ml_s, elapsed)
    
    # Optional temperature control
    if temp_setpoint is not None:
        temperature_pid_controller(
            bioreactor,
            setpoint=temp_setpoint,
            kp=kp,
            ki=ki,
            kd=kd,
            dt=dt if dt is not None else temp_freq,
            elapsed=elapsed,
            sensor_index=sensor_index,
            max_duty_heat=max_duty_heat,
            max_duty_cool=max_duty_cool,
        )


def independent_flow(
    bioreactor,
    pump_name: str,
    ml_per_sec: float,
    duration: float,
    converse_duration: Optional[float] = None,
    elapsed: Optional[float] = None,
) -> None:
    """
    Run a pump pair (inflow/outflow) sequentially at the same flow rate but with independent durations.

    Like balanced_flow, this identifies the converse pump automatically from the
    pump_name. The primary pump (inflow) runs first for its full duration, then the
    converse pump (outflow) runs for its full duration.

    Args:
        bioreactor: Bioreactor instance
        pump_name: Name of the primary pump (e.g., 'inflow')
        ml_per_sec: Flow rate in ml/sec for both pumps
        duration: Duration in seconds for the primary pump
        converse_duration: Duration in seconds for the converse pump.
                          If None, defaults to the same as duration (balanced).
        elapsed: Elapsed time (unused, for compatibility with job functions)
    """
    from .io import change_pump

    if converse_duration is None:
        converse_duration = duration

    if not bioreactor.is_component_initialized('pumps'):
        bioreactor.logger.warning("Pumps not initialized; cannot set independent flow")
        return

    if not hasattr(bioreactor, 'pumps'):
        bioreactor.logger.warning("Pumps not available")
        return

    # Determine the converse pump name
    if pump_name == 'inflow':
        converse_name = 'outflow'
    elif pump_name == 'outflow':
        converse_name = 'inflow'
    else:
        if pump_name.endswith('_in') or pump_name.endswith('_inflow'):
            base = pump_name.rsplit('_', 1)[0] if '_' in pump_name else pump_name
            converse_name = f"{base}_out" if not base.endswith('out') else f"{base}_outflow"
        elif pump_name.endswith('_out') or pump_name.endswith('_outflow'):
            base = pump_name.rsplit('_', 1)[0] if '_' in pump_name else pump_name
            converse_name = f"{base}_in" if not base.endswith('in') else f"{base}_inflow"
        else:
            bioreactor.logger.warning(
                f"Cannot determine converse pump for '{pump_name}'. "
                f"Running only the specified pump. Available pumps: {list(bioreactor.pumps.keys())}"
            )
            try:
                change_pump(bioreactor, pump_name, ml_per_sec)
                if duration > 0:
                    time.sleep(duration)
                    change_pump(bioreactor, pump_name, 0.0)
                    bioreactor.logger.info(f"Pump {pump_name} stopped after {duration:.2f} seconds")
            except Exception as e:
                bioreactor.logger.error(f"Error setting pump {pump_name}: {e}")
            return

    # Check if both pumps exist
    if pump_name not in bioreactor.pumps:
        bioreactor.logger.error(f"Pump '{pump_name}' not found. Available: {list(bioreactor.pumps.keys())}")
        return

    if converse_name not in bioreactor.pumps:
        bioreactor.logger.warning(
            f"Converse pump '{converse_name}' not found. "
            f"Running only '{pump_name}'. Available pumps: {list(bioreactor.pumps.keys())}"
        )
        try:
            change_pump(bioreactor, pump_name, ml_per_sec)
            if duration > 0:
                time.sleep(duration)
                change_pump(bioreactor, pump_name, 0.0)
                bioreactor.logger.info(f"Pump {pump_name} stopped after {duration:.2f} seconds")
        except Exception as e:
            bioreactor.logger.error(f"Error setting pump {pump_name}: {e}")
        return

    if duration <= 0 and converse_duration <= 0:
        bioreactor.logger.warning("Both durations must be positive. Ignoring.")
        return

    def _run_pumps():
        try:
            bioreactor.pumping_active = True
            bioreactor.logger.info(
                f"Independent flow: {pump_name}={duration:.2f}s then {converse_name}={converse_duration:.2f}s "
                f"at {ml_per_sec:.4f} ml/sec"
            )

            # Run inflow first
            change_pump(bioreactor, pump_name, ml_per_sec)
            if duration > 0:
                time.sleep(duration)
            change_pump(bioreactor, pump_name, 0.0)

            # Then run outflow
            change_pump(bioreactor, converse_name, ml_per_sec)
            if converse_duration > 0:
                time.sleep(converse_duration)
            change_pump(bioreactor, converse_name, 0.0)

            bioreactor.logger.info(
                f"Independent flow: {pump_name} and {converse_name} complete"
            )
        except Exception as e:
            bioreactor.logger.error(f"Failed to set independent flow: {e}")
        finally:
            bioreactor.pumping_active = False

    pump_thread = threading.Thread(target=_run_pumps, daemon=True)
    pump_thread.start()


def _read_last_csv_row(csv_path: str) -> Optional[dict]:
    """Read the last data row from a CSV file efficiently.

    Opens a separate read handle and seeks to the last 4KB of the file,
    then parses the final complete row. Safe to call alongside an open
    write handle as long as the writer calls flush() after each write.

    Returns:
        dict mapping column names to string values, or None if no data row exists.
    """
    import csv as _csv
    try:
        with open(csv_path, 'r') as f:
            f.seek(0, 2)  # seek to end
            file_size = f.tell()
            # Read last 4KB (or whole file if smaller)
            seek_pos = max(0, file_size - 4096)
            f.seek(seek_pos)
            tail = f.read()

        lines = tail.strip().split('\n')
        if len(lines) < 2:
            # Need at least header + 1 data row if we're at start of file
            # Re-read from beginning to get the header
            with open(csv_path, 'r') as f:
                lines = f.read().strip().split('\n')
            if len(lines) < 2:
                return None

        # If we seeked into the middle of the file, the first line is likely
        # a partial row. We need the header from the start of the file.
        with open(csv_path, 'r') as f:
            header_line = f.readline().strip()

        reader = _csv.DictReader([header_line, lines[-1]])
        for row in reader:
            return dict(row)
        return None
    except (OSError, IOError):
        return None


def turbidostat_ekf_mode(
    bioreactor,
    od_setpoint: float,
    pump_name: str = 'inflow',
    flow_rate_ml_s: float = 2.0,
    od_channel: str = 'OD_135_V',
    R: float = 0.001,
    Q_growth_rate: float = 5e-12,
    initial_growth_rate: float = 1.0,
    initial_P_od: Optional[float] = None,
    initial_P_r: float = 0.0005**2,
    pump_distrust_cycles: int = 10,
    pump_distrust_P_od: Optional[float] = None,
    pump_duration: float = 5.0,
    temp_setpoint: Optional[float] = None,
    kp: float = 12.0,
    ki: float = 0.015,
    kd: float = 0.0,
    dt: Optional[float] = None,
    sensor_index: int = 0,
    max_duty_heat: Optional[float] = None,
    max_duty_cool: Optional[float] = None,
    elapsed: Optional[float] = None,
) -> None:
    """
    Turbidostat mode using an Extended Kalman Filter (Hoffmann et al. 2017).

    Reads the latest OD measurement from the CSV file, filters it through
    an EKF to estimate true OD and growth rate, and triggers dilution
    (independent_flow) when the estimated OD exceeds the setpoint.

    When diluting, the outflow pump runs for 1.1x the inflow pump duration
    to ensure the vessel does not overfill.

    The EKF state vector is [OD, r] where r is the per-cycle multiplicative
    growth rate (r=1 means no growth). Doubling time is computed as
    dt_cycle * ln(2) / ln(r).

    Args:
        bioreactor: Bioreactor instance
        od_setpoint: Target OD voltage. Pump triggers when estimated OD exceeds this.
        pump_name: Name of the inflow pump for independent_flow (default: 'inflow')
        flow_rate_ml_s: Flow rate in ml/sec for dilution events (default: 2.0)
        od_channel: CSV column name for the OD reading (default: 'OD_135_V')
        R: Measurement noise variance (default: 0.001)
        Q_growth_rate: Process noise variance for growth rate state (default: 5e-13)
        initial_growth_rate: Initial growth rate estimate (default: 1.0, no growth)
        initial_P_od: Initial OD covariance. If None, defaults to R.
        initial_P_r: Initial growth rate covariance (default: 0.0005^2)
        pump_distrust_cycles: Number of cycles after pumping to inflate OD uncertainty (default: 10)
        pump_distrust_P_od: Inflated P[0,0] value during distrust. If None, defaults to 10*R.
        pump_duration: Duration of the inflow pump per dilution event in seconds (default: 5.0).
                      The outflow pump runs for 1.1x this duration.
        temp_setpoint: Optional temperature setpoint for PID control (default: None)
        kp: Proportional gain for temperature PID (default: 12.0)
        ki: Integral gain for temperature PID (default: 0.015)
        kd: Derivative gain for temperature PID (default: 0.0)
        dt: Time step for temperature PID. If None, auto-computed.
        sensor_index: Temperature sensor index (default: 0)
        max_duty_heat: Max duty for heating. None = use config.PELTIER_MAX_DUTY_HEAT
        max_duty_cool: Max duty for cooling. None = use config.PELTIER_MAX_DUTY_COOL
        elapsed: Elapsed time in seconds (passed by bioreactor.run scheduler)
    """
    # --- Read latest OD from CSV ---
    if not hasattr(bioreactor, 'out_file_path'):
        bioreactor.logger.warning("Turbidostat: no out_file_path on bioreactor")
        return

    row = _read_last_csv_row(bioreactor.out_file_path)
    if row is None or od_channel not in row:
        bioreactor.logger.debug("Turbidostat: no OD data yet, skipping cycle")
        return

    try:
        z_k = float(row[od_channel])
    except (ValueError, TypeError):
        bioreactor.logger.debug(f"Turbidostat: could not parse OD value '{row.get(od_channel)}'")
        return

    if np.isnan(z_k):
        bioreactor.logger.debug("Turbidostat: OD reading is NaN, skipping cycle")
        return

    # --- Defaults for optional covariance params ---
    if initial_P_od is None:
        initial_P_od = R
    if pump_distrust_P_od is None:
        pump_distrust_P_od = 10.0 * R

    # --- Initialize EKF state on first valid reading ---
    if not getattr(bioreactor, '_ekf_initialized', False):
        bioreactor._ekf_state = np.array([z_k, initial_growth_rate])
        bioreactor._ekf_P = np.array([
            [initial_P_od, 0.0],
            [0.0, initial_P_r],
        ])
        bioreactor._ekf_pump_distrust_counter = 0
        bioreactor._ekf_last_time = elapsed if elapsed is not None else time.time()
        bioreactor._ekf_initialized = True
        bioreactor.logger.info(
            f"Turbidostat EKF initialized: OD={z_k:.4f}, r={initial_growth_rate:.4f}, "
            f"setpoint={od_setpoint:.4f}"
        )
        return

    # --- Compute dt_cycle ---
    current_time = elapsed if elapsed is not None else time.time()
    dt_cycle = current_time - bioreactor._ekf_last_time
    if dt_cycle <= 0:
        dt_cycle = 1.0  # fallback to avoid division by zero
    bioreactor._ekf_last_time = current_time

    # --- EKF Predict (Hoffmann et al. 2017, Eqs 4-8) ---
    od_k = bioreactor._ekf_state[0]
    r_k = bioreactor._ekf_state[1]
    P = bioreactor._ekf_P

    # Predicted state: OD grows multiplicatively by r each cycle
    x_pred = np.array([od_k * r_k, r_k])

    # Jacobian of the state transition
    F = np.array([
        [r_k, od_k],
        [0.0, 1.0],
    ])

    # Process noise: no noise on OD prediction, only on growth rate
    Q = np.array([
        [0.0, 0.0],
        [0.0, Q_growth_rate],
    ])

    P_pred = F @ P @ F.T + Q

    # --- Pump distrust: reset OD covariance during and after dilution ---
    # Per Hoffmann et al. 2017: reset P[0,0] and zero off-diagonal terms so that
    # K[0] ≈ 1 (OD re-converges quickly) but K[1] ≈ 0 (r is protected from
    # dilution artifacts).
    currently_pumping = getattr(bioreactor, 'pumping_active', False)
    if currently_pumping or bioreactor._ekf_pump_distrust_counter > 0:
        P_pred[0, 0] = pump_distrust_P_od
        P_pred[0, 1] = 0.0
        P_pred[1, 0] = 0.0
        x_pred[0] = z_k          # reset OD estimate to raw measurement
        if not currently_pumping:
            bioreactor._ekf_pump_distrust_counter -= 1

    # --- EKF Update (direct observation: H = [1, 0]) ---
    # Innovation
    y = z_k - x_pred[0]

    # Innovation covariance
    S = P_pred[0, 0] + R

    # Kalman gain
    K = P_pred[:, 0] / S

    # Updated state
    x_updated = x_pred + K * y

    # Updated covariance: P = (I - K @ H) @ P_pred
    KH = np.outer(K, np.array([1.0, 0.0]))
    P_updated = (np.eye(2) - KH) @ P_pred

    # Secondary reset: if innovation exceeds 5σ, distrust the estimate
    innovation_threshold = 5.0 * np.sqrt(R)
    if abs(z_k - x_pred[0]) > innovation_threshold:
        P_updated[0, 1] = 0.0
        P_updated[1, 0] = 0.0
        P_updated[0, 0] = (x_updated[0] - z_k) ** 2

    # Store updated state
    bioreactor._ekf_state = x_updated
    bioreactor._ekf_P = P_updated

    od_est = x_updated[0]
    r_est = x_updated[1]

    # --- Compute doubling time and its uncertainty ---
    # T = dt_cycle * ln(2) / ln(r)
    # σ_T = |dT/dr| * σ_r = dt_cycle * ln(2) * σ_r / (r * ln(r)^2)
    od_std = np.sqrt(P_updated[0, 0])
    r_std = np.sqrt(P_updated[1, 1])
    if r_est > 1.0 and dt_cycle > 0:
        ln_r = np.log(r_est)
        doubling_time = dt_cycle * np.log(2.0) / ln_r
        doubling_time_std = dt_cycle * np.log(2.0) * r_std / (r_est * ln_r ** 2)
    else:
        doubling_time = float('inf')
        doubling_time_std = float('inf')

    # Flag so standalone EKF in measure_and_record_sensors knows to skip
    bioreactor._turbidostat_ekf_active = True

    # Store estimates so measure_and_record_sensors can write them to CSV
    bioreactor.ekf_estimates = {
        'ekf_od_est': od_est,
        'ekf_growth_rate': r_est,
        'ekf_doubling_time_s': doubling_time,
        'ekf_od_std': od_std,
        'ekf_growth_rate_std': r_std,
        'ekf_doubling_time_std_s': doubling_time_std,
    }

    # --- Pump decision ---
    pumped = False
    if od_est > od_setpoint:
        if currently_pumping:
            bioreactor.logger.debug(
                f"Turbidostat: OD_est={od_est:.4f} > setpoint={od_setpoint:.4f}, "
                f"but pumps already running — skipping"
            )
        else:
            converse_duration = pump_duration * 1.1
            bioreactor.logger.info(
                f"Turbidostat: OD_est={od_est:.4f} > setpoint={od_setpoint:.4f}, "
                f"pumping {pump_name}={pump_duration:.1f}s, outflow={converse_duration:.1f}s"
            )
            independent_flow(
                bioreactor, pump_name, flow_rate_ml_s,
                duration=pump_duration, converse_duration=converse_duration,
            )
            pumped = True
            bioreactor._ekf_pump_distrust_counter = pump_distrust_cycles

            # Track pump run times
            if hasattr(bioreactor, 'pump_run_times'):
                if pump_name in bioreactor.pump_run_times:
                    bioreactor.pump_run_times[pump_name] += pump_duration
                # Converse pump runs for 1.1x duration
                if pump_name == 'inflow':
                    converse = 'outflow'
                elif pump_name == 'outflow':
                    converse = 'inflow'
                else:
                    converse = None
                if converse and converse in bioreactor.pump_run_times:
                    bioreactor.pump_run_times[converse] += converse_duration

    # --- Optional temperature PID ---
    if temp_setpoint is not None:
        temperature_pid_controller(
            bioreactor,
            setpoint=temp_setpoint,
            kp=kp,
            ki=ki,
            kd=kd,
            dt=dt if dt is not None else dt_cycle,
            elapsed=elapsed,
            sensor_index=sensor_index,
            max_duty_heat=max_duty_heat,
            max_duty_cool=max_duty_cool,
        )

    # --- Log ---
    dt_str = f"{doubling_time:.1f}s" if doubling_time != float('inf') else "inf"
    if pumped:
        pump_str = " [PUMPED]"
    elif currently_pumping:
        pump_str = " [PUMPING]"
    else:
        pump_str = ""
    bioreactor.logger.info(
        f"Turbidostat: meas={z_k:.4f} est={od_est:.4f} r={r_est:.6f} "
        f"Td={dt_str} sp={od_setpoint:.4f}{pump_str}"
    )

