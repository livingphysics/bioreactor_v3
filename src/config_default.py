"""
Configuration class for bioreactor components.
Modify INIT_COMPONENTS to enable/disable specific components.
"""

from typing import Union, Optional


class Config:
    """Bioreactor configuration"""
    
    # Logging Configuration
    LOG_LEVEL: str = 'INFO'
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE: str = 'bioreactor.log'
    LOG_TO_TERMINAL: bool = True  # Print logs to terminal/console
    CLEAR_LOG_ON_START: bool = True  # If True, clears/truncates the log file on startup
    DATA_OUT_FILE: str = 'bioreactor_data.csv'
    USE_TIMESTAMPED_FILENAME: bool = True  # If True, adds timestamp prefix (e.g., "20250113_153000_bioreactor_data.csv"). If False, uses base filename only.
    
    # Component Initialization Control
    # Set to True to initialize, False to skip
    INIT_COMPONENTS: dict[str, bool] = {
        'i2c': False,  # Only needed if other I2C components are used
        'temp_sensor': True,
        'peltier_driver': True,  # Enable PWM peltier driver (uses lgpio)
        'stirrer': True,  # PWM stirrer driver
        'led': False,  # LED PWM control
        'ring_light': False,  # Neopixel ring light (uses pi5neo)
        'optical_density': False,  # Optical density sensor (ADS1115)
        'eyespy_adc': False,  # Eyespy ADC component (ADS1114, based on pioreactor)
        'co2_sensor': False,  # Senseair K33 CO2 sensor (I2C)
        'pumps': False,  # Pump control via ticUSB
    }
    
    # Sensor Labels for CSV output
    SENSOR_LABELS: dict = {
        'temperature': 'temperature_C',
        # OD channel labels are auto-generated from OD_ADC_CHANNELS in bioreactor.py
        # Remove old entries if channel names have changed
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

    # Ring Light Configuration (Neopixel, using pi5neo)
    RING_LIGHT_SPI_DEVICE: str = '/dev/spidev0.0'  # SPI device path
    RING_LIGHT_COUNT: int = 32  # Number of LEDs in the ring
    RING_LIGHT_SPI_SPEED: int = 800  # SPI speed in kHz

    # Optical Density (OD) Configuration (ADS1115 ADC)
    OD_ADC_CHANNELS: dict[str, str] = {
        '135': 'A0',
        'Ref': 'A1',
        '90': 'A2',
    }  # Dictionary mapping channel names to ADS1115 pins (A0-A3)
    
    # Eyespy ADC Configuration (ADS1114, based on pioreactor pattern)
    # Supports multiple eyespy boards, each at a different I2C address
    # Each eyespy board is a single-channel ADS1114 ADC
    EYESPY_ADC: dict = {
        'eyespy1': {
            'i2c_address': 0x49,  # I2C address (default for eyespy/pd2)
            'i2c_bus': 1,  # I2C bus number (typically 1 for /dev/i2c-1)
            'gain': 1.0,  # PGA gain: 2/3, 1.0, 2.0, 4.0, 8.0, 16.0 (default: 1.0 = ±4.096 V)
        },
        # Add more eyespy boards as needed:
        'eyespy2': {
            'i2c_address': 0x4a,  # Different I2C address
            'i2c_bus': 1,
            'gain': 1.0,
        },
    }
    
    # CO2 Sensor Configuration
    # CO2_SENSOR_TYPE options:
    #   - 'sensair' or'sensair_k33' (default): Senseair K33 sensor over I2C (default address: 0x68)
    #   - 'atlas' or 'atlas_i2c': Atlas Scientific CO2 sensor over I2C using atlas_i2c library (default address: 0x69)
    CO2_SENSOR_TYPE: str = 'sensair_k33'
    CO2_SENSOR_ENABLED: bool = False  # Set to True to enable CO2 sensor
    CO2_SENSOR_I2C_ADDRESS: Optional[int] = None  # I2C address for CO2 sensor (None = use type-specific default: 0x68 for sensair_k33, 0x69 for atlas)
    CO2_SENSOR_I2C_BUS: int = 1  # I2C bus number (typically 1 for /dev/i2c-1)
    
    # Pump Configuration (ticUSB protocol)
    # Default configuration: 2 pumps (inflow and outflow)
    # Add more pumps by extending the PUMPS dictionary
    # Each pump requires a serial number (from TicUSB device)
    PUMPS: dict[str, dict[str, Union[str, int]]] = {
        'inflow': {
            'serial': '00473498',  # Replace with your pump's serial number
            'step_mode': 3,  # Step mode (0-3, typically 3 for microstepping)
            'current_limit': 32,  # Current limit in units (check TicUSB docs)
        },
        'outflow': {
            'serial': '00473497',  # Replace with your pump's serial number
            'step_mode': 3,
            'current_limit': 32,
        },
        # Add more pumps as needed:
        # 'pump_3': {
        #     'serial': '00473504',
        #     'step_mode': 3,
        #     'current_limit': 32,
        # },
    }