"""
Input/Output functions for bioreactor.
These are not intended to be used directly by the user, but rather to be used by the bioreactor class.
"""

import logging
from typing import Optional, Dict, Union

logger = logging.getLogger("Bioreactor.IO")


class RelayController:
    """Controller class for managing all relay operations."""
    
    def __init__(self, bioreactor, relays_dict: Dict, gpio_chip):
        """
        Initialize RelayController.
        
        Args:
            bioreactor: Bioreactor instance
            relays_dict: Dictionary mapping relay names to {'pin': int, 'chip': gpio_chip}
            gpio_chip: GPIO chip handle
        """
        self.bioreactor = bioreactor
        self._relays = relays_dict
        self._gpio_chip = gpio_chip
    
    def on(self, relay_name: str) -> bool:
        """Turn a relay ON.
        
        Args:
            relay_name: Name of the relay
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.set(relay_name, True)
    
    def off(self, relay_name: str) -> bool:
        """Turn a relay OFF.
        
        Args:
            relay_name: Name of the relay
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.set(relay_name, False)
    
    def set(self, relay_name: str, state: bool) -> bool:
        """Set relay state (ON or OFF).
        
        Args:
            relay_name: Name of the relay
            state: True for ON, False for OFF
            
        Returns:
            bool: True if successful, False otherwise
        """
        if relay_name not in self._relays:
            self.bioreactor.logger.warning(f"Relay '{relay_name}' not found")
            return False
        
        try:
            import lgpio
            pin = self._relays[relay_name]['pin']
            # Inverted logic: 0 = ON, 1 = OFF
            lgpio.gpio_write(self._gpio_chip, pin, 0 if state else 1)
            self.bioreactor.logger.info(f"Relay {relay_name} turned {'ON' if state else 'OFF'}")
            return True
        except Exception as e:
            self.bioreactor.logger.error(f"Error setting relay {relay_name}: {e}")
            return False
    
    def get_state(self, relay_name: str) -> Optional[bool]:
        """Get the current state of a relay.
        
        Args:
            relay_name: Name of the relay
            
        Returns:
            bool: True if ON, False if OFF, None if error
        """
        if relay_name not in self._relays:
            self.bioreactor.logger.warning(f"Relay '{relay_name}' not found")
            return None
        
        try:
            import lgpio
            pin = self._relays[relay_name]['pin']
            state = lgpio.gpio_read(self._gpio_chip, pin)
            # Inverted logic: 0 = ON, 1 = OFF
            return not bool(state)
        except Exception as e:
            self.bioreactor.logger.error(f"Error reading relay {relay_name} state: {e}")
            return None
    
    def get_all_states(self) -> Dict[str, bool]:
        """Get the state of all relays.
        
        Returns:
            dict: Dictionary mapping relay names to their states (True=ON, False=OFF)
        """
        states = {}
        for relay_name in self._relays.keys():
            state = self.get_state(relay_name)
            if state is not None:
                states[relay_name] = state
        return states
    
    def all_on(self) -> bool:
        """Turn all relays ON.
        
        Returns:
            bool: True if successful, False otherwise
        """
        success = True
        for relay_name in self._relays.keys():
            if not self.on(relay_name):
                success = False
        return success
    
    def all_off(self) -> bool:
        """Turn all relays OFF.
        
        Returns:
            bool: True if successful, False otherwise
        """
        success = True
        for relay_name in self._relays.keys():
            if not self.off(relay_name):
                success = False
        return success
    
    def list_relays(self) -> list:
        """Get list of available relay names.
        
        Returns:
            list: List of relay names
        """
        return list(self._relays.keys())


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


def measure_od(bioreactor, led_power: float, averaging_duration: float, channel_name: str = 'Trx') -> Optional[Union[float, Dict[str, float]]]:
    """
    Measure optical density by turning LED on, taking readings, and averaging.
    
    This function:
    1. Turns LED on at the given power for 1 second
    2. Starts taking readings for the specified duration
    3. Averages the readings
    4. Returns the averaged voltage value(s)
    
    Args:
        bioreactor: Bioreactor instance
        led_power: LED power level (0-100)
        averaging_duration: Duration in seconds to average readings
        channel_name: Name of the ADC channel to read, or 'all' to measure all channels (default: 'Trx')
        
    Returns:
        float: Averaged voltage reading for single channel, or None if error
        dict: Dictionary mapping channel names to averaged voltages when channel_name='all', or None if error
    """
    import time
    
    if not bioreactor.is_component_initialized('led'):
        bioreactor.logger.error("LED driver not initialized")
        return None
    
    if not bioreactor.is_component_initialized('optical_density'):
        bioreactor.logger.error("Optical density sensor not initialized")
        return None
    
    # Determine which channels to measure
    if channel_name.lower() == 'all':
        if not hasattr(bioreactor, 'od_channels') or not bioreactor.od_channels:
            bioreactor.logger.error("No OD channels available")
            return None
        channels_to_measure = list(bioreactor.od_channels.keys())
    else:
        channels_to_measure = [channel_name]
    
    try:
        # Turn LED on at specified power
        if not set_led(bioreactor, led_power):
            bioreactor.logger.error("Failed to set LED power")
            return None
        
        # Wait 1 second for LED to stabilize
        time.sleep(1.0)
        
        # Collect readings for the specified duration
        # Use dictionary to store readings for each channel
        all_readings = {ch: [] for ch in channels_to_measure}
        start_time = time.time()
        sample_interval = 0.01  # Sample every 10ms for smooth averaging
        
        while (time.time() - start_time) < averaging_duration:
            for ch in channels_to_measure:
                voltage = read_voltage(bioreactor, ch)
                if voltage is not None:
                    all_readings[ch].append(voltage)
            time.sleep(sample_interval)
        
        # Turn LED off
        bioreactor.led_driver.off()
        
        # Calculate averages for each channel
        results = {}
        for ch in channels_to_measure:
            if not all_readings[ch]:
                bioreactor.logger.warning(f"No valid readings collected for channel {ch}")
                results[ch] = None
            else:
                avg_voltage = sum(all_readings[ch]) / len(all_readings[ch])
                results[ch] = avg_voltage
                bioreactor.logger.info(
                    f"OD measurement complete: {len(all_readings[ch])} readings averaged, "
                    f"LED power {led_power}%, duration {averaging_duration}s, "
                    f"channel {ch}, avg voltage: {avg_voltage:.4f}V"
                )
        
        # Return single value if single channel, dict if all channels
        if channel_name.lower() == 'all':
            # Filter out None values if any channel failed
            valid_results = {k: v for k, v in results.items() if v is not None}
            if not valid_results:
                bioreactor.logger.warning("No valid readings collected for any channel")
                return None
            return valid_results
        else:
            # Single channel mode - return float or None
            if results[channel_name] is None:
                return None
            return results[channel_name]
        
    except Exception as e:
        bioreactor.logger.error(f"Error during OD measurement: {e}")
        # Ensure LED is turned off on error
        try:
            bioreactor.led_driver.off()
        except:
            pass
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
                bioreactor.logger.info(f"Temperature: {temperature}")
                return temperature
            else:
                bioreactor.logger.warning(f"Temperature sensor index {sensor_index} not available")
                return float('nan')
        except Exception as e:
            bioreactor.logger.error(f"Error reading temperature sensor {sensor_index}: {e}")
            return float('nan')
