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
        'co2_sensor_2': True,  # Second CO2 sensor
        'o2_sensor': True,
        'i2c': False,  # Only needed if other I2C components are used
        'temp_sensor': True,  # DS18B20 temperature sensor
    }
    
    # Relay Configuration
    RELAY_PINS: list[int] = [6, 13, 19, 26]  # GPIO pins for relays
    RELAY_NAMES: list[str] = ['relay_1', 'relay_2', 'relay_3', 'relay_4']  # Names for each relay
    
    # Sensor Configuration
    # CO2 sensor uses serial interface
    CO2_SERIAL_PORT: str = '/dev/ttyUSB0'
    CO2_SERIAL_BAUDRATE: int = 9600
    # Second CO2 sensor uses serial interface
    CO2_SERIAL_PORT_2: str = '/dev/ttyUSB1'
    CO2_SERIAL_BAUDRATE_2: int = 9600
    # O2 sensor uses I2C (Atlas Scientific)
    O2_SENSOR_ADDRESS: int = 108
    
    # Temperature Sensor Configuration
    # DS18B20 sensor order (optional, uses all sensors in order if None)
    TEMP_SENSOR_ORDER: list[int] = None  # e.g., [0, 1, 2, 3] to reorder sensors
    
    # Sensor Labels for CSV output
    SENSOR_LABELS: dict = {
        'co2': 'CO2_ppm',
        'co2_2': 'CO2_2_ppm',
        'o2': 'O2_percent',
        'temperature': 'temperature_C',
    }
    
    # Auto-flush on initialization
    AUTO_FLUSH_ON_INIT: bool = True  # Automatically flush tank on startup
    AUTO_FLUSH_DURATION: int = 30  # Duration in seconds for auto-flush

