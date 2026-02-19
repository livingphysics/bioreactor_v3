# Config used for this run (serialized from in-memory config at startup)

CLEAR_LOG_ON_START = True

CO2_SENSOR_I2C_ADDRESS = None

CO2_SENSOR_I2C_BUS = 1

CO2_SENSOR_TYPE = 'atlas_i2c'

DATA_OUT_FILE = 'bioreactor_data.csv'

EYESPY_ADC = {'eyespy1': {'gain': 1.0, 'i2c_address': 73, 'i2c_bus': 1},
 'eyespy2': {'gain': 1.0, 'i2c_address': 74, 'i2c_bus': 1}}

INIT_COMPONENTS = {'co2_sensor': True,
 'eyespy_adc': False,
 'i2c': True,
 'led': True,
 'o2_sensor': False,
 'optical_density': True,
 'peltier_driver': True,
 'pumps': False,
 'ring_light': True,
 'stirrer': True,
 'temp_sensor': True}

LED_PWM_FREQ = 500

LED_PWM_PIN = 25

LOG_FILE = 'bioreactor.log'

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

LOG_LEVEL = 'INFO'

LOG_TO_TERMINAL = True

O2_SENSOR_I2C_ADDRESS = None

O2_SENSOR_I2C_BUS = 1

OD_ADC_CHANNELS = {'135': 'A0', '90': 'A2', 'Ref': 'A1'}

PELTIER_DIR_PIN = 20

PELTIER_PWM_FREQ = 1000

PELTIER_PWM_PIN = 21

PUMPS = {'inflow': {'current_limit': 32,
            'direction': 'forward',
            'serial': '00473498',
            'step_mode': 3,
            'steps_per_ml': 10000000.0},
 'outflow': {'current_limit': 32,
             'direction': 'forward',
             'serial': '00473497',
             'step_mode': 3,
             'steps_per_ml': 10000000.0}}

RESULTS_BASE_DIR = 'bioreactor_data'

RESULTS_PACKAGE = True

RING_LIGHT_COUNT = 32

RING_LIGHT_SPI_DEVICE = '/dev/spidev0.0'

RING_LIGHT_SPI_SPEED = 800

RUN_SCRIPT_PATH = None

SENSOR_LABELS = {'co2': 'CO2_ppm',
 'od_135': 'OD_135_V',
 'od_90': 'OD_90_V',
 'od_ref': 'OD_Ref_V',
 'peltier_duty': 'peltier_duty',
 'peltier_forward': 'peltier_forward',
 'ring_light_B': 'ring_light_B',
 'ring_light_G': 'ring_light_G',
 'ring_light_R': 'ring_light_R',
 'temperature': 'temperature_C'}

STIRRER_DEFAULT_DUTY = 30.0

STIRRER_PWM_FREQ = 1000

STIRRER_PWM_PIN = 12

USE_TIMESTAMPED_FILENAME = False
