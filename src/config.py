"""
Configuration class for bioreactor components.
Modify INIT_COMPONENTS to enable/disable specific components.
"""

from typing import Union


class Config:
    """Bioreactor configuration"""
    
    # Logging Configuration
    LOG_LEVEL: str = 'INFO'
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE: str = 'bioreactor.log'
    LOG_TO_TERMINAL: bool = True  # Print logs to terminal/console
    DATA_OUT_FILE: str = 'bioreactor_data.csv'
    
    # Component Initialization Control
    # Set to True to initialize, False to skip
    INIT_COMPONENTS: dict[str, bool] = {
        'relays': True,
        'co2_sensor': True,
        'o2_sensor': True,
        'i2c': False,  # Only needed if other I2C components are used
    }
    
    # Relay Configuration
    RELAY_PINS: list[int] = [6, 13, 19, 26]  # GPIO pins for relays
    RELAY_NAMES: list[str] = ['relay_1', 'relay_2', 'relay_3', 'relay_4']  # Names for each relay
    
    # Sensor Configuration
    CO2_SENSOR_ADDRESS: int = 105
    O2_SENSOR_ADDRESS: int = 108
    
    # Sensor Labels for CSV output
    SENSOR_LABELS: dict = {
        'co2': 'CO2_ppm',
        'o2': 'O2_percent',
    }

