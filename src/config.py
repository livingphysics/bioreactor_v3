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
    USE_TIMESTAMPED_FILENAME: bool = True  # If True, adds timestamp prefix (e.g., "20250113_153000_bioreactor_data.csv"). If False, uses base filename only.
    
    # Component Initialization Control
    # Set to True to initialize, False to skip
    INIT_COMPONENTS: dict[str, bool] = {
        'relays': True,
        'co2_sensor': True,
        'co2_sensor_2': True,  # Second CO2 sensor
        'o2_sensor': True,
        'i2c': False,  # Only needed if other I2C components are used
        'temp_sensor': True,
        'peltier_driver': True,  # Enable PWM peltier driver (uses lgpio)
        'stirrer': True,  # PWM stirrer driver
        'led': False,  # LED PWM control
        'optical_density': False,  # Optical density sensor (ADS1115)
    }
    
    # Relay Configuration
    RELAY_PINS: dict[str, int] = {
        'relay_1': 6,
        'relay_2': 13,
        'relay_3': 19,
        'relay_4': 26,
    }  # Dictionary mapping relay names to GPIO pins
    
    # Sensor Configuration
    # CO2 sensor uses serial interface
    CO2_SERIAL_PORT: str = '/dev/ttyUSB0'
    CO2_SERIAL_BAUDRATE: int = 9600
    # Second CO2 sensor uses serial interface
    CO2_SERIAL_PORT_2: str = '/dev/ttyUSB1'
    CO2_SERIAL_BAUDRATE_2: int = 9600
    # O2 sensor uses I2C (Atlas Scientific)
    O2_SENSOR_ADDRESS: int = 108
    
    # Sensor Labels for CSV output
    SENSOR_LABELS: dict = {
        'co2': 'CO2_ppm',
        'co2_2': 'CO2_2_ppm',
        'o2': 'O2_percent',
        'temperature': 'temperature_C',
    }

    # Peltier Driver Configuration (Raspberry Pi 5 GPIO via lgpio)
    PELTIER_PWM_PIN: int = 12  # BCM pin for PWM output
    PELTIER_DIR_PIN: int = 16  # BCM pin for direction control
    PELTIER_PWM_FREQ: int = 1000  # PWM frequency in Hz

    # Stirrer Configuration (PWM only)
    STIRRER_PWM_PIN: int = 21  # BCM pin for stirrer PWM output
    STIRRER_PWM_FREQ: int = 1000  # PWM frequency in Hz
    STIRRER_DEFAULT_DUTY: float = 20.0  # Default duty cycle (0-100)

    # LED Configuration (PWM control)
    LED_PWM_PIN: int = 25  # BCM pin for LED PWM output
    LED_PWM_FREQ: int = 500  # PWM frequency in Hz

    # Optical Density (OD) Configuration (ADS1115 ADC)
    OD_ADC_CHANNELS: dict[str, str] = {
        'Trx': 'A0',
        'Ref': 'A1',
        'Sct': 'A2',
    }  # Dictionary mapping channel names to ADS1115 pins (A0-A3)
