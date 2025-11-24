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
                return bioreactor.temp_sensors[sensor_index].get_temperature()
            else:
                bioreactor.logger.warning(f"Temperature sensor index {sensor_index} not available")
                return float('nan')
        except Exception as e:
            bioreactor.logger.error(f"Error reading temperature sensor {sensor_index}: {e}")
            return float('nan')