"""
Input/Output functions for bioreactor.
These are not intended to be used directly by the user, but rather to be used by the bioreactor class.
"""

import logging
import math
import time
from typing import Optional, Dict, Union

logger = logging.getLogger("Bioreactor.IO")


class PeltierDriver:
    """PWM/DIR controller for the peltier module using lgpio."""

    def __init__(self, bioreactor, gpio_chip, pwm_pin: int, dir_pin: int, frequency: int):
        self.bioreactor = bioreactor
        self._gpio_chip = gpio_chip
        self._pwm_pin = pwm_pin
        self._dir_pin = dir_pin
        self._frequency = frequency
        self._last_duty = 0.0
        self._last_forward = True

    def set(self, duty_cycle: float, forward: bool = True) -> bool:
        """
        Set the PWM duty cycle and direction.

        Args:
            duty_cycle: Target duty cycle (0-100)
            forward: Direction flag (True=forward/heat, False=reverse/cool)
        """
        try:
            import lgpio
        except Exception as e:
            self.bioreactor.logger.error(f"Peltier driver requires lgpio: {e}")
            return False

        try:
            duty = max(0.0, min(100.0, float(duty_cycle)))
        except (TypeError, ValueError):
            raise ValueError("Duty cycle must be numeric between 0 and 100") from None

        try:
            lgpio.gpio_write(self._gpio_chip, self._dir_pin, 1 if forward else 0)
            lgpio.tx_pwm(self._gpio_chip, self._pwm_pin, self._frequency, duty)
        except Exception as e:
            self.bioreactor.logger.error(f"Failed to update peltier PWM: {e}")
            return False

        self._last_duty = duty
        self._last_forward = forward
        self.bioreactor.logger.info(
            f"Peltier set to {duty:.1f}% duty, direction {'forward' if forward else 'reverse'}"
        )
        return True

    def stop(self) -> None:
        """Stop PWM output."""
        try:
            import lgpio
            lgpio.tx_pwm(self._gpio_chip, self._pwm_pin, self._frequency, 0)
            self._last_duty = 0.0
            self.bioreactor.logger.info("Peltier PWM stopped.")
        except Exception as e:
            self.bioreactor.logger.error(f"Failed to stop peltier PWM: {e}")

    @property
    def is_active(self) -> bool:
        return self._last_duty > 0.0


class StirrerDriver:
    """Single-pin PWM stirrer controller using lgpio."""

    def __init__(self, bioreactor, gpio_chip, pwm_pin: int, frequency: int, default_duty: float = 0.0):
        self.bioreactor = bioreactor
        self._gpio_chip = gpio_chip
        self._pwm_pin = pwm_pin
        self._frequency = frequency
        self._duty = max(0.0, min(100.0, float(default_duty)))

    def set_speed(self, duty_cycle: float) -> bool:
        """Set stirrer PWM duty cycle (0-100)."""
        try:
            import lgpio
        except Exception as e:
            self.bioreactor.logger.error(f"Stirrer driver requires lgpio: {e}")
            return False

        try:
            duty = max(0.0, min(100.0, float(duty_cycle)))
        except (TypeError, ValueError):
            raise ValueError("Duty cycle must be numeric between 0 and 100") from None

        try:
            lgpio.tx_pwm(self._gpio_chip, self._pwm_pin, self._frequency, duty)
            self._duty = duty
            self.bioreactor.logger.info(f"Stirrer duty set to {duty:.1f}%")
            return True
        except Exception as e:
            self.bioreactor.logger.error(f"Failed to update stirrer PWM: {e}")
            return False

    def stop(self) -> None:
        """Stop stirrer (0% duty)."""
        try:
            import lgpio
            lgpio.tx_pwm(self._gpio_chip, self._pwm_pin, self._frequency, 0)
            self._duty = 0.0
            self.bioreactor.logger.info("Stirrer stopped (0% duty).")
        except Exception as e:
            self.bioreactor.logger.error(f"Failed to stop stirrer: {e}")

    @property
    def duty_cycle(self) -> float:
        return self._duty


class LEDDriver:
    """PWM LED controller using lgpio."""

    def __init__(self, bioreactor, gpio_chip, pwm_pin: int, frequency: int):
        self.bioreactor = bioreactor
        self._gpio_chip = gpio_chip
        self._pwm_pin = pwm_pin
        self._frequency = frequency
        self._power = 0.0

    def set_power(self, power: float) -> bool:
        """Set LED PWM power (0-100)."""
        try:
            import lgpio
        except Exception as e:
            self.bioreactor.logger.error(f"LED driver requires lgpio: {e}")
            return False

        try:
            power = max(0.0, min(100.0, float(power)))
        except (TypeError, ValueError):
            raise ValueError("Power must be numeric between 0 and 100") from None

        try:
            lgpio.tx_pwm(self._gpio_chip, self._pwm_pin, self._frequency, power)
            self._power = power
            self.bioreactor.logger.info(f"LED power set to {power:.1f}%")
            return True
        except Exception as e:
            self.bioreactor.logger.error(f"Failed to update LED PWM: {e}")
            return False

    def off(self) -> None:
        """Turn LED off (0% power)."""
        try:
            import lgpio
            lgpio.tx_pwm(self._gpio_chip, self._pwm_pin, self._frequency, 0)
            self._power = 0.0
            self.bioreactor.logger.info("LED turned off (0% power).")
        except Exception as e:
            self.bioreactor.logger.error(f"Failed to turn LED off: {e}")

    @property
    def power(self) -> float:
        return self._power


class RingLightDriver:
    """Neopixel ring light controller using pi5neo library."""

    def __init__(self, bioreactor, spi_device: str, num_leds: int, spi_speed: int):
        self.bioreactor = bioreactor
        self._spi_device = spi_device
        self._num_leds = num_leds
        self._spi_speed = spi_speed
        self._neo = None
        self._current_color = (0, 0, 0)
        self._is_on = False

    def _ensure_initialized(self) -> bool:
        """Ensure the Pi5Neo object is initialized."""
        if self._neo is None:
            try:
                from pi5neo import Pi5Neo
                self._neo = Pi5Neo(self._spi_device, self._num_leds, self._spi_speed)
                self.bioreactor.logger.info(f"Ring light initialized: {self._num_leds} LEDs on {self._spi_device}")
                return True
            except Exception as e:
                self.bioreactor.logger.error(f"Failed to initialize ring light: {e}")
                return False
        return True

    def set_color(self, color: tuple, pixel: Optional[int] = None) -> bool:
        """
        Set ring light color.
        
        Args:
            color: RGB tuple (r, g, b) with values 0-255
            pixel: Optional pixel index (0-based). If None, sets all pixels.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._ensure_initialized():
            return False

        try:
            # Clamp color values to 0-255
            r = max(0, min(255, int(color[0])))
            g = max(0, min(255, int(color[1])))
            b = max(0, min(255, int(color[2])))
            color_tuple = (r, g, b)

            if pixel is None:
                self._neo.fill_strip(r, g, b)
            else:
                if pixel < 0 or pixel >= self._num_leds:
                    self.bioreactor.logger.error(f"Pixel index {pixel} out of range (0-{self._num_leds-1})")
                    return False
                self._neo.set_led_color(pixel, r, g, b)
            
            self._neo.update_strip()
            self._current_color = color_tuple
            self._is_on = any(c > 0 for c in color_tuple)
            
            if pixel is None:
                self.bioreactor.logger.info(f"Ring light set to color {color_tuple}")
            else:
                self.bioreactor.logger.info(f"Ring light pixel {pixel} set to color {color_tuple}")
            return True
        except Exception as e:
            self.bioreactor.logger.error(f"Failed to set ring light color: {e}")
            return False

    def off(self) -> None:
        """Turn ring light off (all pixels to black)."""
        if not self._ensure_initialized():
            return
        
        try:
            self._neo.fill_strip(0, 0, 0)
            self._neo.update_strip()
            self._current_color = (0, 0, 0)
            self._is_on = False
            self.bioreactor.logger.info("Ring light turned off")
        except Exception as e:
            self.bioreactor.logger.error(f"Failed to turn ring light off: {e}")

    @property
    def is_on(self) -> bool:
        """Check if ring light is on (any pixel has non-zero color)."""
        return self._is_on

    @property
    def current_color(self) -> tuple:
        """Get current color setting."""
        return self._current_color


def read_voltage(bioreactor, channel_name: str) -> Optional[float]:
    """
    Read voltage from an optical density ADC channel.
    
    Args:
        bioreactor: Bioreactor instance
        channel_name: Name of the channel (e.g., 'Trx', 'Ref', 'Sct')
        
    Returns:
        float: Voltage reading in volts, or None if error
    """
    if not bioreactor.is_component_initialized('optical_density'):
        bioreactor.logger.warning("Optical density sensor not initialized")
        return None
    
    if not hasattr(bioreactor, 'od_channels'):
        bioreactor.logger.warning("OD channels not available")
        return None
    
    if channel_name not in bioreactor.od_channels:
        bioreactor.logger.warning(f"OD channel '{channel_name}' not found. Available: {list(bioreactor.od_channels.keys())}")
        return None
    
    try:
        channel = bioreactor.od_channels[channel_name]
        voltage = channel.voltage
        return voltage
    except Exception as e:
        bioreactor.logger.error(f"Error reading voltage from channel {channel_name}: {e}")
        return None


def set_led(bioreactor, power: float) -> bool:
    """
    Set LED power level.
    
    Args:
        bioreactor: Bioreactor instance
        power: Power level (0-100)
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not bioreactor.is_component_initialized('led'):
        bioreactor.logger.warning("LED driver not initialized")
        return False
    
    if not hasattr(bioreactor, 'led_driver'):
        bioreactor.logger.warning("LED driver not available")
        return False
    
    try:
        return bioreactor.led_driver.set_power(power)
    except Exception as e:
        bioreactor.logger.error(f"Error setting LED power: {e}")
        return False


def set_ring_light(bioreactor, color: tuple, pixel: Optional[int] = None) -> bool:
    """
    Set ring light color.
    
    Args:
        bioreactor: Bioreactor instance
        color: RGB tuple (r, g, b) with values 0-255
        pixel: Optional pixel index (0-based). If None, sets all pixels.
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not bioreactor.is_component_initialized('ring_light'):
        bioreactor.logger.warning("Ring light driver not initialized")
        return False
    
    if not hasattr(bioreactor, 'ring_light_driver'):
        bioreactor.logger.warning("Ring light driver not available")
        return False
    
    try:
        return bioreactor.ring_light_driver.set_color(color, pixel)
    except Exception as e:
        bioreactor.logger.error(f"Error setting ring light color: {e}")
        return False


def turn_off_ring_light(bioreactor) -> None:
    """
    Turn ring light off.
    
    Args:
        bioreactor: Bioreactor instance
    """
    if not bioreactor.is_component_initialized('ring_light'):
        return
    
    if not hasattr(bioreactor, 'ring_light_driver'):
        return
    
    try:
        bioreactor.ring_light_driver.off()
    except Exception as e:
        bioreactor.logger.error(f"Error turning off ring light: {e}")


def measure_od(bioreactor, led_power: float, averaging_duration: float, channel_name: str = 'Trx') -> Optional[Union[float, Dict[str, float]]]:
    """
    Measure optical density by turning LED on, taking readings, and averaging.
    Also reads eyespy ADC boards if initialized.
    
    This function:
    1. Stores current ring light state and turns it OFF (dodging) to avoid interference with OD measurement
    2. Turns LED on at the given power for 1 second
    3. Starts taking readings for the specified duration (OD channels and eyespy boards)
    4. Averages the readings
    5. Turns LED off
    6. Restores ring light to its previous state (after IR LED is off)
    7. Returns the averaged voltage value(s)
    
    Args:
        bioreactor: Bioreactor instance
        led_power: LED power level (0-100)
        averaging_duration: Duration in seconds to average readings
        channel_name: Name of the ADC channel to read, or 'all' to measure all channels (default: 'Trx')
        
    Returns:
        float: Averaged voltage reading for single channel, or None if error
        dict: Dictionary mapping channel names to averaged voltages when channel_name='all', or None if error
              Includes both OD channels and eyespy boards (e.g., 'Trx', 'eyespy1', 'eyespy2')
    """
    import time
    
    if not bioreactor.is_component_initialized('led'):
        bioreactor.logger.error("LED driver not initialized")
        return None
    
    od_initialized = bioreactor.is_component_initialized('optical_density')
    eyespy_initialized = bioreactor.is_component_initialized('eyespy_adc')
    
    if not od_initialized and not eyespy_initialized:
        bioreactor.logger.error("Neither optical density sensor nor eyespy ADC initialized")
        return None
    
    # Warn if both are initialized
    if od_initialized and eyespy_initialized:
        bioreactor.logger.warning(
            "Both optical_density and eyespy_adc are initialized. "
            "Reading both while LED is on - ensure they don't interfere with each other."
        )
    
    # Determine which OD channels to measure
    channels_to_measure = []
    if od_initialized:
        if channel_name.lower() == 'all':
            if not hasattr(bioreactor, 'od_channels') or not bioreactor.od_channels:
                bioreactor.logger.error("No OD channels available")
                return None
            channels_to_measure = list(bioreactor.od_channels.keys())
        else:
            channels_to_measure = [channel_name]
    
    # Get eyespy board names if initialized
    eyespy_boards = []
    if eyespy_initialized and hasattr(bioreactor, 'eyespy_boards'):
        eyespy_boards = list(bioreactor.eyespy_boards.keys())
    
    # Store ring light state and turn it off (dodging) before OD measurement
    ring_light_was_on = False
    ring_light_previous_color = (0, 0, 0)
    if bioreactor.is_component_initialized('ring_light') and hasattr(bioreactor, 'ring_light_driver'):
        ring_light_was_on = bioreactor.ring_light_driver.is_on
        ring_light_previous_color = bioreactor.ring_light_driver.current_color
        bioreactor.ring_light_driver.off()
        bioreactor.logger.info("Ring light turned off for OD measurement (dodging)")
    
    try:
        # Turn LED on at specified power
        if not set_led(bioreactor, led_power):
            bioreactor.logger.error("Failed to set LED power")
            return None
        
        # Wait 1 second for LED to stabilize
        time.sleep(1.0)
        
        # Collect readings for the specified duration
        # Use dictionary to store readings for each channel/board
        all_readings = {ch: [] for ch in channels_to_measure}
        eyespy_readings = {board: [] for board in eyespy_boards}
        start_time = time.time()
        sample_interval = 0.01  # Sample every 10ms for smooth averaging
        
        while (time.time() - start_time) < averaging_duration:
            # Read OD channels
            for ch in channels_to_measure:
                voltage = read_voltage(bioreactor, ch)
                if voltage is not None:
                    all_readings[ch].append(voltage)
            
            # Read eyespy boards
            for board_name in eyespy_boards:
                voltage = read_eyespy_voltage(bioreactor, board_name)
                if voltage is not None:
                    eyespy_readings[board_name].append(voltage)
            
            time.sleep(sample_interval)
        
        # Turn LED off
        bioreactor.led_driver.off()
        
        # Restore ring light to previous state (after IR LED is off)
        if bioreactor.is_component_initialized('ring_light') and hasattr(bioreactor, 'ring_light_driver'):
            if ring_light_was_on:
                bioreactor.ring_light_driver.set_color(ring_light_previous_color)
                bioreactor.logger.info(f"Ring light restored to previous state: color={ring_light_previous_color}")
            else:
                # Was already off, ensure it's off
                bioreactor.ring_light_driver.off()
        
        # Calculate averages for OD channels
        results = {}
        for ch in channels_to_measure:
            if not all_readings[ch]:
                bioreactor.logger.warning(f"No valid readings collected for OD channel {ch}")
                results[ch] = None
            else:
                avg_voltage = sum(all_readings[ch]) / len(all_readings[ch])
                results[ch] = avg_voltage
                bioreactor.logger.info(
                    f"OD measurement complete: {len(all_readings[ch])} readings averaged, "
                    f"LED power {led_power}%, duration {averaging_duration}s, "
                    f"channel {ch}, avg voltage: {avg_voltage:.4f}V"
                )
        
        # Calculate averages for eyespy boards
        for board_name in eyespy_boards:
            if not eyespy_readings[board_name]:
                bioreactor.logger.warning(f"No valid readings collected for eyespy board {board_name}")
                results[board_name] = None
            else:
                avg_voltage = sum(eyespy_readings[board_name]) / len(eyespy_readings[board_name])
                results[board_name] = avg_voltage
                bioreactor.logger.info(
                    f"Eyespy measurement complete: {len(eyespy_readings[board_name])} readings averaged, "
                    f"LED power {led_power}%, duration {averaging_duration}s, "
                    f"board {board_name}, avg voltage: {avg_voltage:.4f}V"
                )
        
        # Return single value if single channel, dict if all channels
        if channel_name.lower() == 'all' or eyespy_boards:
            # Return dict with all results (OD + eyespy)
            valid_results = {k: v for k, v in results.items() if v is not None}
            if not valid_results:
                bioreactor.logger.warning("No valid readings collected for any channel or board")
                return None
            return valid_results
        else:
            # Single channel mode - return float or None
            if channel_name in results:
                if results[channel_name] is None:
                    return None
                return results[channel_name]
            return None
        
    except Exception as e:
        bioreactor.logger.error(f"Error during OD measurement: {e}")
        # Ensure LED is turned off on error
        try:
            bioreactor.led_driver.off()
        except:
            pass
        # Restore ring light to previous state even on error
        if bioreactor.is_component_initialized('ring_light') and hasattr(bioreactor, 'ring_light_driver'):
            try:
                if ring_light_was_on:
                    bioreactor.ring_light_driver.set_color(ring_light_previous_color)
                    bioreactor.logger.info(f"Ring light restored to previous state after error: color={ring_light_previous_color}")
                else:
                    bioreactor.ring_light_driver.off()
            except Exception as restore_error:
                bioreactor.logger.error(f"Failed to restore ring light state: {restore_error}")
        return None


def get_temperature(bioreactor, sensor_index=0):
    """Get temperature from DS18B20 sensor(s).
    
    Args:
        sensor_index (int): Index of sensor to read (default: 0 for first sensor)
        
    Returns:
        float: Temperature in Celsius, or NaN if sensor not available
    """
    if not bioreactor.is_component_initialized('temp_sensor'):
        return float('nan')
    
    try:
        if hasattr(bioreactor, 'temp_sensors') and len(bioreactor.temp_sensors) > sensor_index:
            bioreactor.logger.info(f"Reading temperature from sensor {sensor_index}")
            temperature = bioreactor.temp_sensors[sensor_index].get_temperature()
            
            # Check if temperature is within valid bounds (0-100°C)
            if not math.isnan(temperature):
                if temperature < 0.0 or temperature > 100.0:
                    bioreactor.logger.warning(
                        f"Temperature reading {temperature:.2f}°C is outside valid bounds (0-100°C), returning NaN"
                    )
                    return float('nan')
            
            bioreactor.logger.info(f"Temperature: {temperature}")
            return temperature
        else:
            bioreactor.logger.warning(f"Temperature sensor index {sensor_index} not available")
            return float('nan')
    except Exception as e:
        bioreactor.logger.error(f"Error reading temperature sensor {sensor_index}: {e}")
        return float('nan')


def set_peltier_power(bioreactor, duty_cycle: Union[int, float], forward: Union[bool, str] = True) -> bool:
    """
    Set the PWM duty cycle and direction for the peltier driver.
    
    Args:
        bioreactor: Bioreactor instance
        duty_cycle: Target duty percentage (0-100)
        forward: Direction flag or descriptive string ('heat', 'cool', etc.)
        
    Returns:
        bool: True if successful, False otherwise
    """
    driver = getattr(bioreactor, 'peltier_driver', None)
    if not bioreactor.is_component_initialized('peltier_driver') or driver is None:
        bioreactor.logger.warning("Peltier driver not initialized; skipping command.")
        return False

    if isinstance(forward, str):
        fwd = forward.lower()
        if fwd in ('forward', 'cool', 'cold'):
            forward_bool = True
        elif fwd in ('reverse', 'heat', 'warm', 'hot'):
            forward_bool = False
        else:
            # Fallback: interpret truthy string as True
            forward_bool = fwd in ('true', '1', 'on')
    else:
        forward_bool = bool(forward)

    return driver.set(duty_cycle, forward=forward_bool)


def stop_peltier(bioreactor) -> None:
    """
    Stop PWM output on the peltier driver, if available.
    
    Args:
        bioreactor: Bioreactor instance
    """
    driver = getattr(bioreactor, 'peltier_driver', None)
    if not driver:
        return
    try:
        driver.stop()
    except Exception as e:
        bioreactor.logger.error(f"Failed to stop peltier driver: {e}")


def set_stirrer_speed(bioreactor, duty_cycle: Union[int, float]) -> bool:
    """
    Set stirrer PWM duty cycle.
    
    Args:
        bioreactor: Bioreactor instance
        duty_cycle: Target duty (0-100)
        
    Returns:
        bool: True if successful, False otherwise
    """
    driver = getattr(bioreactor, 'stirrer_driver', None)
    if not bioreactor.is_component_initialized('stirrer') or driver is None:
        bioreactor.logger.warning("Stirrer driver not initialized; skipping command.")
        return False
    return driver.set_speed(duty_cycle)


def stop_stirrer(bioreactor) -> None:
    """Stop stirrer PWM output.
    
    Args:
        bioreactor: Bioreactor instance
    """
    driver = getattr(bioreactor, 'stirrer_driver', None)
    if not driver:
        return
    try:
        driver.stop()
    except Exception as e:
        bioreactor.logger.error(f"Failed to stop stirrer: {e}")


def read_eyespy_adc(bioreactor, board_name: str = None) -> Optional[int]:
    """
    Read raw ADC value from an eyespy board (ADS1114).
    
    Args:
        bioreactor: Bioreactor instance
        board_name: Name of the eyespy board (e.g., 'eyespy1'). 
                   If None and only one board is configured, uses that board.
        
    Returns:
        int: Raw 16-bit signed integer ADC reading (-32768 to 32767), or None if error
    """
    if not bioreactor.is_component_initialized('eyespy_adc'):
        bioreactor.logger.warning("Eyespy ADC not initialized")
        return None
    
    if not hasattr(bioreactor, 'eyespy_boards') or not bioreactor.eyespy_boards:
        bioreactor.logger.warning("Eyespy ADC boards not available")
        return None
    
    # Determine which board to use
    if board_name is None:
        if len(bioreactor.eyespy_boards) == 1:
            board_name = list(bioreactor.eyespy_boards.keys())[0]
        else:
            bioreactor.logger.warning(
                f"Multiple eyespy boards configured. Specify board_name. Available: {list(bioreactor.eyespy_boards.keys())}"
            )
            return None
    
    if board_name not in bioreactor.eyespy_boards:
        bioreactor.logger.warning(
            f"Eyespy board '{board_name}' not found. Available: {list(bioreactor.eyespy_boards.keys())}"
        )
        return None
    
    board_cfg = bioreactor.eyespy_boards[board_name]
    read_func = getattr(bioreactor, '_eyespy_read_func', None)
    
    if not read_func:
        bioreactor.logger.error("Eyespy read function not available")
        return None
    
    try:
        reading = read_func(
            i2c_address=board_cfg['i2c_address'],
            i2c_bus=board_cfg['i2c_bus'],
            gain=board_cfg['gain']
        )
        return reading
    except Exception as e:
        bioreactor.logger.error(f"Error reading eyespy ADC board {board_name}: {e}")
        return None


def read_eyespy_voltage(bioreactor, board_name: str = None) -> Optional[float]:
    """
    Read voltage from an eyespy board (ADS1114), converting raw ADC value to voltage.
    
    The voltage conversion depends on the gain setting:
    - gain 2/3: ±6.144 V -> 1 LSB = 0.1875 mV
    - gain 1.0: ±4.096 V -> 1 LSB = 0.125 mV
    - gain 2.0: ±2.048 V -> 1 LSB = 0.0625 mV
    - gain 4.0: ±1.024 V -> 1 LSB = 0.03125 mV
    - gain 8.0: ±0.512 V -> 1 LSB = 0.015625 mV
    - gain 16.0: ±0.256 V -> 1 LSB = 0.0078125 mV
    
    Args:
        bioreactor: Bioreactor instance
        board_name: Name of the eyespy board (e.g., 'eyespy1'). 
                   If None and only one board is configured, uses that board.
        
    Returns:
        float: Voltage reading in volts, or None if error
    """
    raw_value = read_eyespy_adc(bioreactor, board_name)
    if raw_value is None:
        return None
    
    if not hasattr(bioreactor, 'eyespy_boards'):
        return None
    
    # Determine board config for gain
    if board_name is None:
        if len(bioreactor.eyespy_boards) == 1:
            board_name = list(bioreactor.eyespy_boards.keys())[0]
        else:
            return None
    
    if board_name not in bioreactor.eyespy_boards:
        return None
    
    board_cfg = bioreactor.eyespy_boards[board_name]
    gain = board_cfg['gain']
    
    # Full-scale ranges for different gains (from ADS1114 datasheet)
    fsr_map = {
        2/3: 6.144,
        1.0: 4.096,
        2.0: 2.048,
        4.0: 1.024,
        8.0: 0.512,
        16.0: 0.256,
    }
    
    fsr = fsr_map.get(gain, 4.096)  # Default to 4.096 if gain not found
    
    # Convert raw value to voltage: voltage = (raw_value / 32767) * FSR
    voltage = (raw_value / 32767.0) * fsr
    
    return voltage


def read_all_eyespy_boards(bioreactor) -> Optional[Dict[str, int]]:
    """
    Read raw ADC values from all configured eyespy boards.
    
    Args:
        bioreactor: Bioreactor instance
        
    Returns:
        dict: Dictionary mapping board names to raw ADC readings, or None if error
    """
    if not bioreactor.is_component_initialized('eyespy_adc'):
        bioreactor.logger.warning("Eyespy ADC not initialized")
        return None
    
    if not hasattr(bioreactor, 'eyespy_boards') or not bioreactor.eyespy_boards:
        bioreactor.logger.warning("Eyespy ADC boards not available")
        return None
    
    readings = {}
    for board_name in bioreactor.eyespy_boards.keys():
        reading = read_eyespy_adc(bioreactor, board_name)
        if reading is not None:
            readings[board_name] = reading
        else:
            readings[board_name] = None
    
    return readings


def read_co2(bioreactor) -> Optional[int]:
    """
    Read CO2 concentration from Senseair K33 sensor via I2C.
    
    Args:
        bioreactor: Bioreactor instance
        
    Returns:
        int: CO2 concentration in ppm, or None if error
    """
    if not bioreactor.is_component_initialized('co2_sensor'):
        bioreactor.logger.warning("CO2 sensor not initialized")
        return None
    
    if not hasattr(bioreactor, 'co2_sensor_config'):
        bioreactor.logger.warning("CO2 sensor configuration not available")
        return None
    
    try:
        from smbus2 import SMBus, i2c_msg
        import time
    except ImportError as e:
        bioreactor.logger.error(f"CO2 sensor requires smbus2: {e}")
        return None
    
    # Senseair K33 protocol constants
    READRAM_CMD = 0x22
    CO2_RAM_ADDR_HI = 0x00
    CO2_RAM_ADDR_LO = 0x08
    
    def calc_checksum(data_bytes):
        """Calculate checksum for Senseair K33 protocol."""
        return sum(data_bytes) & 0xFF
    
    try:
        config = bioreactor.co2_sensor_config
        i2c_addr = config['i2c_address']
        bus_num = config['i2c_bus']
        
        with SMBus(bus_num) as bus:
            # Prepare ReadRAM command frame: [command, addr_hi, addr_lo, checksum]
            command = READRAM_CMD
            addr_hi = CO2_RAM_ADDR_HI
            addr_lo = CO2_RAM_ADDR_LO
            
            # Calculate checksum for write command
            write_checksum = calc_checksum([command, addr_hi, addr_lo])
            write_packet = [command, addr_hi, addr_lo, write_checksum]
            
            # Use raw i2c_msg for reliable communication
            write_msg = i2c_msg.write(i2c_addr, write_packet)
            bus.i2c_rdwr(write_msg)
            time.sleep(0.05)  # Wait for sensor to prepare data (50ms recommended)
            
            # Read response: 4 bytes [status, co2_high, co2_low, checksum]
            read_msg = i2c_msg.read(i2c_addr, 4)
            bus.i2c_rdwr(read_msg)
            response = list(read_msg)
            
            if len(response) < 4:
                bioreactor.logger.error(f"Incomplete CO2 response: got {len(response)} bytes, expected 4")
                return None
            
            status = response[0]
            co2_high = response[1]
            co2_low = response[2]
            read_checksum = response[3]
            
            # Validate status byte (bit 0 should be 1 for success)
            if (status & 0x01) == 0:
                bioreactor.logger.error(f"Senseair K33 error: status byte indicates failure (0x{status:02X})")
                return None
            
            # Validate checksum
            expected_checksum = calc_checksum([status, co2_high, co2_low])
            if read_checksum != expected_checksum:
                bioreactor.logger.error(
                    f"CO2 checksum mismatch: got 0x{read_checksum:02X}, expected 0x{expected_checksum:02X}"
                )
                return None
            
            # Combine high and low bytes to get CO2 value, then multiply by 10 to get PPM
            co2_raw = (co2_high << 8) | co2_low
            co2_ppm = co2_raw * 10
            
            return co2_ppm
        
    except OSError as e:
        if e.errno == 121:
            config = bioreactor.co2_sensor_config
            i2c_addr = config['i2c_address']
            bus_num = config['i2c_bus']
            bioreactor.logger.error(
                f"Remote I/O error (121): CO2 sensor not responding at address 0x{i2c_addr:02X} on bus {bus_num}"
            )
        else:
            bioreactor.logger.error(f"I2C communication error reading CO2: {e}")
        return None
    except Exception as e:
        bioreactor.logger.error(f"Failed to read from CO2 sensor: {e}")
        return None
