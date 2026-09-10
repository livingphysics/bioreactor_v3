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
    DATA_LOGGING: bool = True  # If False, the Bioreactor writes NO data file (CSV) or results package. For hardware utilities (e.g. pump cleaning) that only drive actuators.
    
    # Results package: put each run in a dated directory with output + copy of script
    RESULTS_PACKAGE: bool = True  # If True, create a dated dir and write output + script copy there
    RESULTS_BASE_DIR: str = 'bioreactor_data'  # Base dir for results packages (relative to src/); dated subdirs created here
    RUN_SCRIPT_PATH: Optional[str] = None  # Path to script to copy into results package (None = use sys.argv[0] if valid)
    
    # Component Initialization Control
    # Set to True to initialize, False to skip
    INIT_COMPONENTS: dict[str, bool] = {
        'i2c': True,  # Only needed if other I2C components are used
        'temp_sensor': True,
        'peltier_driver': True,  # Enable PWM peltier driver (uses lgpio)
        'stirrer': True,  # PWM stirrer driver
        'led': True,  # LED PWM control
        'ring_light': True,  # Neopixel ring light (uses pi5neo)
        'optical_density': True,  # Optical density sensor (ADS1115)
        'eyespy_adc': False,  # Eyespy ADC component (ADS1114, based on pioreactor)
        'co2_sensor': False,  # Senseair K33 CO2 sensor (I2C)
        'o2_sensor': False,  # Atlas Scientific O2 sensor (I2C)
        'ambient_temp': False,  # PCT2075 ambient temperature sensor (I2C)
        'peltier_current': False,  # INA228 current monitor on the peltier supply (I2C)
        'pumps': False,  # Pump control via ticUSB
        'relays': False,  # GPIO relay control
    }
    
    # Sensor Labels for CSV output
    # Labels are auto-populated in bioreactor.py based on INIT_COMPONENTS.
    # Only add custom labels here if you want to override the defaults.
    # Possible keys: 'temperature', 'co2', 'o2', 'ambient_temp', 'peltier_current';
    # 'od_45', 'od_ref', 'od_90', 'od_135' (the OD measurements below);
    # 'voltage_<source>' (a voltage source no OD measurement consumes);
    # legacy configs: 'od_<channel>', 'eyespy_<board>_raw', 'eyespy_<board>_voltage';
    # 'peltier_duty', 'peltier_forward'; 'ring_light_R', 'ring_light_G', 'ring_light_B'.
    SENSOR_LABELS: dict = {}

    # Peltier Driver Configuration (Raspberry Pi 5 GPIO via lgpio)
    PELTIER_PWM_PIN: int = 21  # BCM pin for PWM output
    PELTIER_DIR_PIN: int = 20  # BCM pin for direction control
    PELTIER_PWM_FREQ: int = 1000  # PWM frequency in Hz
    PELTIER_MAX_DUTY_HEAT: float = 70.0  # Max duty cycle for heating (0-100, hardware safety limit)
    PELTIER_MAX_DUTY_COOL: float = 100.0  # Max duty cycle for cooling (0-100, hardware safety limit)
    PELTIER_MIN_DUTY_COOL: float = 50.0   # Min cooling duty when active; cooling is ineffective below this on this rig
    PELTIER_DIR_INVERTED: bool = False    # Invert the DIR pin level; set True if peltier wiring produces opposite physical direction from the driver convention

    # Stirrer Configuration (PWM only)
    STIRRER_PWM_PIN: int = 12  # BCM pin for stirrer PWM output
    STIRRER_PWM_FREQ: int = 1000  # PWM frequency in Hz
    STIRRER_DEFAULT_DUTY: float = 30.0  # Default duty cycle (0-100)

    # LED Configuration (PWM control)
    LED_PWM_PIN: int = 25  # BCM pin for LED PWM output
    LED_PWM_FREQ: int = 500  # PWM frequency in Hz

    # Ring Light Configuration (Neopixel, using pi5neo)
    RING_LIGHT_SPI_DEVICE: str = '/dev/spidev0.0'  # SPI device path
    RING_LIGHT_COUNT: int = 32  # Number of LEDs in the ring
    RING_LIGHT_SPI_SPEED: int = 800  # SPI speed in kHz

    # ------------------------------------------------------------------------
    # Optical inputs: VOLTAGE SOURCES + OD MEASUREMENTS  (see src/optics.py, docs/optics.md)
    # ------------------------------------------------------------------------
    # Every photodiode/ADC input the rig has, under a name of YOUR choosing. A source is
    # either one ADS1115 channel (kind 'adc', A0-A3; hardware component 'optical_density')
    # or one ADS1114 eyespy board (kind 'eyespy'; hardware component 'eyespy_adc').
    # Each source can be read by name (io.read_voltage), gets its own API endpoint
    # (GET /api/voltage/<name>) and, when no OD measurement consumes it, its own CSV
    # column '<name>_V'. Shorthand strings work too: 'adc:A0', 'eyespy:0x49'.
    VOLTAGE_SOURCES: dict = {
        'pd_135': {'kind': 'adc', 'channel': 'A0'},
        'pd_ref': {'kind': 'adc', 'channel': 'A1'},
        'pd_90':  {'kind': 'adc', 'channel': 'A2'},
        # eyespy boards (enable INIT_COMPONENTS['eyespy_adc'] to use them):
        'eyespy_ref': {'kind': 'eyespy', 'i2c_address': 0x49, 'i2c_bus': 1, 'gain': 1.0},
        'eyespy_sct': {'kind': 'eyespy', 'i2c_address': 0x4a, 'i2c_bus': 1, 'gain': 1.0},
    }

    # The four canonical OD measurements. Each is enabled or not and, when enabled, fed by
    # ONE voltage source of either kind. Enabled measurements are IR-gated, logged as
    # OD_<x>_V (OD_45_V, OD_ref_V, OD_90_V, OD_135_V), served at GET /api/od/state and are
    # what the dashboard plots. Shorthands: 'pd_135' (enabled, that source), False, True
    # (enabled, source 'pd_<x>').
    OD_MEASUREMENTS: dict = {
        'OD_45':  {'enabled': False},
        'OD_ref': {'enabled': True, 'source': 'pd_ref'},
        'OD_90':  {'enabled': True, 'source': 'pd_90'},
        'OD_135': {'enabled': True, 'source': 'pd_135'},
    }

    # LEGACY form (still honoured when VOLTAGE_SOURCES / OD_MEASUREMENTS are absent):
    #   OD_ADC_CHANNELS = {'135': 'A0', 'Ref': 'A1', '90': 'A2'}  -> columns OD_135_V, OD_Ref_V, OD_90_V
    #   EYESPY_ADC = {'ref': {'i2c_address': 0x49, 'i2c_bus': 1, 'gain': 1.0}, ...}
    #                                                            -> columns Eyespy_ref_raw, Eyespy_ref_V
    # Existing per-rig config.py files keep working with their column names unchanged.

    # EKF OD channel: an OD measurement ('OD_135'), its suffix ('135', 'ref') or a voltage
    # source name. Used by the standalone EKF in measure_and_record_sensors and by
    # turbidostat_ekf_mode (which resolves it to the CSV column automatically).
    EKF_OD_CHANNEL: str = 'OD_135'

    # CO2 Sensor Configuration
    # CO2_SENSOR_TYPE options:
    #   - 'sensair' or'sensair_k33' (default): Senseair K33 sensor over I2C (default address: 0x68)
    #   - 'atlas' or 'atlas_i2c': Atlas Scientific CO2 sensor over I2C using atlas_i2c library (default address: 0x69)
    # Enable/disable via INIT_COMPONENTS['co2_sensor']
    CO2_SENSOR_TYPE: str = 'atlas_i2c'
    CO2_SENSOR_I2C_ADDRESS: Optional[int] = None  # I2C address for CO2 sensor (None = use type-specific default: 0x68 for sensair_k33, 0x69 for atlas)
    CO2_SENSOR_I2C_BUS: int = 1  # I2C bus number (typically 1 for /dev/i2c-1)
    
    # O2 Sensor Configuration (Atlas Scientific)
    # Enable/disable via INIT_COMPONENTS['o2_sensor']
    O2_SENSOR_I2C_ADDRESS: Optional[int] = None  # I2C address for O2 sensor (None = use default: 0x6C)
    O2_SENSOR_I2C_BUS: int = 1  # I2C bus number (typically 1 for /dev/i2c-1)

    # Ambient Temperature Sensor Configuration (NXP PCT2075, I2C)
    # Enable/disable via INIT_COMPONENTS['ambient_temp']. Reads in °C.
    # The PCT2075's standard address range is 0x48-0x4F; this rig reports it at 0x37.
    AMBIENT_TEMP_I2C_ADDRESS: int = 0x37  # I2C address of the PCT2075
    AMBIENT_TEMP_I2C_BUS: int = 1  # I2C bus number (typically 1 for /dev/i2c-1)

    # Peltier Current Sensor Configuration (TI INA228 current/power monitor, I2C)
    # Enable/disable via INIT_COMPONENTS['peltier_current']. Reads current in Amps.
    # Current is derived from the shunt voltage: I = V_shunt / INA228_SHUNT_OHMS.
    PELTIER_CURRENT_I2C_ADDRESS: int = 0x40  # I2C address of the INA228 (default 0x40)
    PELTIER_CURRENT_I2C_BUS: int = 1  # I2C bus number (typically 1 for /dev/i2c-1)
    INA228_SHUNT_OHMS: float = 0.015  # Shunt resistor value in ohms. CALIBRATE to your board's shunt
                                      # (0.015 Ω is the Adafruit INA228 breakout default) — the absolute
                                      # current reading scales directly with this value.

    # Relay Configuration (GPIO, active-low by default)
    # Maps relay names to BCM GPIO pin numbers
    RELAYS: dict[str, int] = {
        'relay_1': 5,
        'relay_2': 6,
        'relay_3': 13,
        'relay_4': 19,
    }
    RELAY_ACTIVE_LOW: bool = True  # True = pin LOW turns relay ON (most relay modules)

    # Pump Configuration (ticUSB protocol)
    # Default configuration: 2 pumps (inflow and outflow)
    # Add more pumps by extending the PUMPS dictionary
    # Each pump requires a serial number (from TicUSB device)
    # Direction: 'forward' or 'reverse' - determines velocity sign in change_pump
    # steps_per_ml: Conversion factor for this specific pump (calibrate per pump)
    PUMPS: dict[str, dict[str, Union[str, int, float]]] = {
        'inflow': {
            'serial': '00473498',  # Replace with your pump's serial number
            'step_mode': 2,  # Step mode (0-3, typically 3 for microstepping)
            'current_limit': 32,  # Current limit in units (check TicUSB docs)
            'direction': 'forward',  # Direction: 'forward' or 'reverse'
            'steps_per_ml': 10000000.0,  # Steps per ml conversion factor (calibrate for this pump)
        },
        'outflow': {
            'serial': '00473497',  # Replace with your pump's serial number
            'step_mode': 2,
            'current_limit': 32,
            'direction': 'forward',  # Direction: 'forward' or 'reverse'
            'steps_per_ml': 10000000.0,  # Steps per ml conversion factor (calibrate for this pump)
        },
        # Add more pumps as needed:
        # 'pump_3': {
        #     'serial': '00473504',
        #     'step_mode': 3,
        #     'current_limit': 32,
        #     'direction': 'forward',
        #     'steps_per_ml': 10000000.0,
        # },
    }
