"""
Component initialization functions for bioreactor hardware.
Each function initializes a specific component and returns a dict with the component objects.
"""

import logging

logger = logging.getLogger("Bioreactor.Components")


def init_relays(bioreactor, config):
    """
    Initialize relay components.
    
    Args:
        bioreactor: Bioreactor instance
        config: Configuration object with RELAY_PINS and RELAY_NAMES
        
    Returns:
        dict: {'relays': dict mapping relay names to pins, 'initialized': bool}
    """
    try:
        import lgpio
        
        # Try to open GPIO chip
        try:
            gpio_chip = lgpio.gpiochip_open(4)  # Raspberry Pi 5
        except:
            gpio_chip = lgpio.gpiochip_open(0)  # Fallback for older Pi
        
        relays = {}
        relay_pins = getattr(config, 'RELAY_PINS', [])
        relay_names = getattr(config, 'RELAY_NAMES', [f'relay_{i+1}' for i in range(len(relay_pins))])
        
        for pin, name in zip(relay_pins, relay_names):
            lgpio.gpio_claim_output(gpio_chip, pin, 1)  # Initialize to OFF (1 = OFF with inverted logic)
            relays[name] = {'pin': pin, 'chip': gpio_chip}
            logger.info(f"Relay {name} initialized on pin {pin}")
        
        bioreactor.gpio_chip = gpio_chip
        bioreactor.relays = relays
        
        # Create RelayController for clean API (import here to avoid circular dependency)
        try:
            from .io import RelayController
            bioreactor.relay_controller = RelayController(bioreactor, relays, gpio_chip)
        except ImportError:
            # Fallback if RelayController not available
            bioreactor.relay_controller = None
            logger.warning("RelayController not available")
        
        return {'relays': relays, 'gpio_chip': gpio_chip, 'initialized': True}
    except Exception as e:
        logger.error(f"Relay initialization failed: {e}")
        return {'initialized': False, 'error': str(e)}


def init_co2_sensor(bioreactor, config):
    """
    Initialize CO2 sensor via serial (TTYUSB0).
    
    Args:
        bioreactor: Bioreactor instance
        config: Configuration object
        
    Returns:
        dict: {'sensor': serial object, 'initialized': bool}
    """
    try:
        import serial
        import time
        
        serial_port = getattr(config, 'CO2_SERIAL_PORT', '/dev/ttyUSB0')
        baudrate = getattr(config, 'CO2_SERIAL_BAUDRATE', 9600)
        
        sensor = serial.Serial(serial_port, baudrate=baudrate, timeout=1)
        sensor.flushInput()
        time.sleep(1)  # Allow sensor to initialize
        
        bioreactor.co2_sensor = sensor
        logger.info(f"CO2 sensor initialized on {serial_port} at {baudrate} baud")
        
        return {'sensor': sensor, 'initialized': True}
    except Exception as e:
        logger.error(f"CO2 sensor initialization failed: {e}")
        return {'initialized': False, 'error': str(e)}


def init_co2_sensor_2(bioreactor, config):
    """
    Initialize second CO2 sensor via serial (TTYUSB1).
    
    Args:
        bioreactor: Bioreactor instance
        config: Configuration object
        
    Returns:
        dict: {'sensor': serial object, 'initialized': bool}
    """
    try:
        import serial
        import time
        
        serial_port = getattr(config, 'CO2_SERIAL_PORT_2', '/dev/ttyUSB1')
        baudrate = getattr(config, 'CO2_SERIAL_BAUDRATE_2', 9600)
        
        sensor = serial.Serial(serial_port, baudrate=baudrate, timeout=1)
        sensor.flushInput()
        time.sleep(1)  # Allow sensor to initialize
        
        bioreactor.co2_sensor_2 = sensor
        logger.info(f"CO2 sensor 2 initialized on {serial_port} at {baudrate} baud")
        
        return {'sensor': sensor, 'initialized': True}
    except Exception as e:
        logger.error(f"CO2 sensor 2 initialization failed: {e}")
        return {'initialized': False, 'error': str(e)}


def init_o2_sensor(bioreactor, config):
    """
    Initialize O2 sensor.
    
    Args:
        bioreactor: Bioreactor instance
        config: Configuration object
        
    Returns:
        dict: {'sensor': sensor object, 'initialized': bool}
    """
    try:
        from atlas_i2c import sensors
        
        address = getattr(config, 'O2_SENSOR_ADDRESS', 108)
        sensor = sensors.Sensor("O2", address)
        sensor.connect()
        
        bioreactor.o2_sensor = sensor
        logger.info(f"O2 sensor initialized at address {address}")
        
        return {'sensor': sensor, 'initialized': True}
    except Exception as e:
        logger.error(f"O2 sensor initialization failed: {e}")
        return {'initialized': False, 'error': str(e)}


def init_i2c(bioreactor, config):
    """
    Initialize I2C bus.
    
    Args:
        bioreactor: Bioreactor instance
        config: Configuration object
        
    Returns:
        dict: {'i2c': i2c object, 'initialized': bool}
    """
    try:
        import board
        import busio
        
        i2c = busio.I2C(board.SCL, board.SDA)
        bioreactor.i2c = i2c
        
        logger.info("I2C bus initialized")
        return {'i2c': i2c, 'initialized': True}
    except Exception as e:
        logger.error(f"I2C initialization failed: {e}")
        return {'initialized': False, 'error': str(e)}

def init_temp_sensor(bioreactor, config):
    """
    Initialize DS18B20 temperature sensor(s).
    
    Args:
        bioreactor: Bioreactor instance
        config: Configuration object
        
    Returns:
        dict: {'sensors': list of sensor objects, 'initialized': bool}
    """
    try:
        from ds18b20 import DS18B20
        import numpy as np
        
        # Get sensor order from config, or use all sensors in order
        sensor_order = getattr(config, 'TEMP_SENSOR_ORDER', None)
        
        all_sensors = DS18B20.get_all_sensors()
        if sensor_order is not None:
            sensors = np.array(all_sensors)[sensor_order]
        else:
            sensors = np.array(all_sensors)
        
        bioreactor.temp_sensors = sensors
        logger.info(f"DS18B20 temperature sensors initialized ({len(sensors)} sensors)")
        
        return {'sensors': sensors, 'initialized': True}
    except Exception as e:
        logger.error(f"DS18B20 temperature sensor initialization failed: {e}")
        return {'initialized': False, 'error': str(e)}


def init_peltier_driver(bioreactor, config):
    """
    Initialize PWM/DIR control for the peltier module using lgpio (Pi 5 compatible).
    
    Args:
        bioreactor: Bioreactor instance
        config: Configuration object with PELTIER pin assignments
        
    Returns:
        dict: {'initialized': bool}
    """
    try:
        import lgpio
        from .io import PeltierDriver
    except Exception as import_error:
        logger.error(f"Peltier driver dependencies missing: {import_error}")
        return {'initialized': False, 'error': str(import_error)}
    
    pwm_pin = getattr(config, 'PELTIER_PWM_PIN', None)
    dir_pin = getattr(config, 'PELTIER_DIR_PIN', None)
    frequency = getattr(config, 'PELTIER_PWM_FREQ', 1000)
    
    if pwm_pin is None or dir_pin is None:
        error_msg = "PELTIER_PWM_PIN and PELTIER_DIR_PIN must be set in Config"
        logger.error(error_msg)
        return {'initialized': False, 'error': error_msg}
    
    gpio_chip = getattr(bioreactor, 'gpio_chip', None)
    if gpio_chip is None:
        try:
            gpio_chip = lgpio.gpiochip_open(4)  # Raspberry Pi 5 default
        except Exception:
            gpio_chip = lgpio.gpiochip_open(0)  # Fallback
        bioreactor.gpio_chip = gpio_chip
    
    try:
        lgpio.gpio_claim_output(gpio_chip, dir_pin, 0)
        lgpio.gpio_claim_output(gpio_chip, pwm_pin, 0)
        lgpio.tx_pwm(gpio_chip, pwm_pin, frequency, 0)
    except Exception as e:
        logger.error(f"Peltier driver GPIO setup failed: {e}")
        return {'initialized': False, 'error': str(e)}
    
    driver = PeltierDriver(bioreactor, gpio_chip, pwm_pin, dir_pin, frequency)
    bioreactor.peltier_driver = driver
    logger.info(f"Peltier driver initialized (PWM pin {pwm_pin}, DIR pin {dir_pin}, {frequency} Hz)")
    return {'initialized': True, 'driver': driver}


def init_stirrer(bioreactor, config):
    """
    Initialize PWM stirrer driver using lgpio (Pi 5 compatible).
    """
    try:
        import lgpio
        from .io import StirrerDriver
    except Exception as import_error:
        logger.error(f"Stirrer driver dependencies missing: {import_error}")
        return {'initialized': False, 'error': str(import_error)}

    pwm_pin = getattr(config, 'STIRRER_PWM_PIN', None)
    frequency = getattr(config, 'STIRRER_PWM_FREQ', 1000)
    default_duty = getattr(config, 'STIRRER_DEFAULT_DUTY', 0.0)

    if pwm_pin is None:
        error_msg = "STIRRER_PWM_PIN must be set in Config"
        logger.error(error_msg)
        return {'initialized': False, 'error': error_msg}

    gpio_chip = getattr(bioreactor, 'gpio_chip', None)
    if gpio_chip is None:
        try:
            gpio_chip = lgpio.gpiochip_open(4)
        except Exception:
            gpio_chip = lgpio.gpiochip_open(0)
        bioreactor.gpio_chip = gpio_chip

    try:
        lgpio.gpio_claim_output(gpio_chip, pwm_pin, 0)
        lgpio.tx_pwm(gpio_chip, pwm_pin, frequency, 0)
    except Exception as e:
        logger.error(f"Stirrer GPIO setup failed: {e}")
        return {'initialized': False, 'error': str(e)}

    driver = StirrerDriver(bioreactor, gpio_chip, pwm_pin, frequency, default_duty)
    bioreactor.stirrer_driver = driver
    logger.info(f"Stirrer driver initialized (PWM pin {pwm_pin}, {frequency} Hz)")
    if default_duty:
        driver.set_speed(default_duty)

    return {'initialized': True, 'driver': driver}

# Component registry - maps component names to initialization functions
COMPONENT_REGISTRY = {
    'relays': init_relays,
    'co2_sensor': init_co2_sensor,
    'co2_sensor_2': init_co2_sensor_2,
    'o2_sensor': init_o2_sensor,
    'i2c': init_i2c,
    'temp_sensor': init_temp_sensor,
    'peltier_driver': init_peltier_driver,
    'stirrer': init_stirrer,
}

