"""
Component initialization functions for bioreactor hardware.
Each function initializes a specific component and returns a dict with the component objects.
"""

import logging

logger = logging.getLogger("Bioreactor.Components")


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


def init_led(bioreactor, config):
    """
    Initialize LED PWM control using lgpio (Pi 5 compatible).
    
    Args:
        bioreactor: Bioreactor instance
        config: Configuration object with LED pin assignments
        
    Returns:
        dict: {'initialized': bool}
    """
    try:
        import lgpio
        from .io import LEDDriver
    except Exception as import_error:
        logger.error(f"LED driver dependencies missing: {import_error}")
        return {'initialized': False, 'error': str(import_error)}
    
    pwm_pin = getattr(config, 'LED_PWM_PIN', None)
    frequency = getattr(config, 'LED_PWM_FREQ', 500)
    
    if pwm_pin is None:
        error_msg = "LED_PWM_PIN must be set in Config"
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
        lgpio.gpio_claim_output(gpio_chip, pwm_pin, 0)
        lgpio.tx_pwm(gpio_chip, pwm_pin, frequency, 0)
    except Exception as e:
        logger.error(f"LED GPIO setup failed: {e}")
        return {'initialized': False, 'error': str(e)}
    
    driver = LEDDriver(bioreactor, gpio_chip, pwm_pin, frequency)
    bioreactor.led_driver = driver
    logger.info(f"LED driver initialized (PWM pin {pwm_pin}, {frequency} Hz)")
    return {'initialized': True, 'driver': driver}


def init_optical_density(bioreactor, config):
    """
    Initialize optical density sensor using ADS1115 ADC.
    
    Args:
        bioreactor: Bioreactor instance
        config: Configuration object with OD_ADC_CHANNELS mapping
        
    Returns:
        dict: {'initialized': bool}
    """
    try:
        from adafruit_ads1x15 import ADS1115, AnalogIn, ads1x15
        import board
        import busio
    except Exception as import_error:
        logger.error(f"Optical density sensor dependencies missing: {import_error}")
        return {'initialized': False, 'error': str(import_error)}
    
    # Ensure I2C is initialized
    if not hasattr(bioreactor, 'i2c') or bioreactor.i2c is None:
        i2c_result = init_i2c(bioreactor, config)
        if not i2c_result.get('initialized', False):
            logger.error("I2C initialization required for optical density sensor")
            return {'initialized': False, 'error': 'I2C initialization failed'}
    
    try:
        # Initialize ADS1115 ADC
        ads = ADS1115(bioreactor.i2c)
        
        # Get channel mapping from config
        channel_map = getattr(config, 'OD_ADC_CHANNELS', {
            'Trx': 'A0',
            'Ref': 'A1',
            'Sct': 'A2',
        })
        
        # Map pin names to ads1x15.Pin objects
        pin_map = {
            'A0': ads1x15.Pin.A0,
            'A1': ads1x15.Pin.A1,
            'A2': ads1x15.Pin.A2,
            'A3': ads1x15.Pin.A3,
        }
        
        # Create ADC channels
        adc_channels = {}
        for channel_name, pin_name in channel_map.items():
            if pin_name not in pin_map:
                logger.warning(f"Invalid pin name {pin_name} for channel {channel_name}, skipping")
                continue
            adc_channels[channel_name] = AnalogIn(ads, pin_map[pin_name])
            logger.info(f"OD channel {channel_name} initialized on {pin_name}")
        
        if not adc_channels:
            error_msg = "No valid OD channels configured"
            logger.error(error_msg)
            return {'initialized': False, 'error': error_msg}
        
        bioreactor.od_adc = ads
        bioreactor.od_channels = adc_channels
        logger.info(f"Optical density sensor initialized with {len(adc_channels)} channels")
        
        return {'initialized': True, 'adc': ads, 'channels': adc_channels}
    except Exception as e:
        logger.error(f"Optical density sensor initialization failed: {e}")
        return {'initialized': False, 'error': str(e)}


# Component registry - maps component names to initialization functions
COMPONENT_REGISTRY = {
    'i2c': init_i2c,
    'temp_sensor': init_temp_sensor,
    'peltier_driver': init_peltier_driver,
    'stirrer': init_stirrer,
    'led': init_led,
    'optical_density': init_optical_density,
}

