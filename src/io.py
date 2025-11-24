"""
Input/Output functions for bioreactor.
These are not intended to be used directly by the user, but rather to be used by the bioreactor class.
"""

import logging
logger = logging.getLogger("Bioreactor.IO")

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

def get_relay_state(bioreactor, relay_name: str) -> bool:
        """Change the state of a specific relay.
        
        Args:
            relay_name (str): Name of the relay (e.g., 'relay_1', 'relay_2', etc.)
            state (bool): True to turn ON, False to turn OFF
        """
        if not bioreactor.is_component_initialized('relays'):
            bioreactor.logger.warning("Relays not initialized")
            return
        
        if relay_name not in bioreactor.relays:
            bioreactor.logger.warning(f"Relay {relay_name} not found")
            return
        
        relay_info = bioreactor.relays[relay_name]
        gpio_chip = relay_info['chip']
        pin = relay_info['pin']
        try:
            import lgpio
            state = lgpio.gpio_read(gpio_chip, pin)
            bioreactor.logger.info(f"Relay {relay_name} state: {'ON' if not state else 'OFF'}")
            return bool(state)
        except Exception as e:
            bioreactor.logger.error(f"Error reading relay {relay_name} state: {e}")
            return None