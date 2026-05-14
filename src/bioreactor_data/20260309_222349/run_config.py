# Config used for this run (serialized from in-memory config at startup)

CLEAR_LOG_ON_START = True

CO2_SENSOR_I2C_ADDRESS = 104

CO2_SENSOR_I2C_BUS = 1

DATA_OUT_FILE = 'bioreactor_data.csv'

EYESPY_ADC = {'ref': {'gain': 1.0, 'i2c_address': 73, 'i2c_bus': 1},
 'sct': {'gain': 1.0, 'i2c_address': 74, 'i2c_bus': 1}}

INIT_COMPONENTS = {'co2_sensor': False,
 'eyespy_adc': True,
 'i2c': True,
 'led': True,
 'o2_sensor': True,
 'optical_density': False,
 'peltier_driver': True,
 'pumps': False,
 'relays': True,
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

PELTIER_MAX_DUTY_COOL = 10.0

PELTIER_MAX_DUTY_HEAT = 70.0

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

RELAYS = {'relay_1': 5, 'relay_2': 6, 'relay_3': 13, 'relay_4': 19}

RELAY_ACTIVE_LOW = True

RESULTS_BASE_DIR = 'bioreactor_data'

RESULTS_PACKAGE = True

RUN_SCRIPT_PATH = None

SENSOR_LABELS = {'eyespy_ref_raw': 'Eyespy_ref_raw',
 'eyespy_ref_voltage': 'Eyespy_ref_V',
 'eyespy_sct_raw': 'Eyespy_sct_raw',
 'eyespy_sct_voltage': 'Eyespy_sct_V',
 'o2': 'O2_percent',
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
