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


# Component registry - maps component names to initialization functions
COMPONENT_REGISTRY = {
    'relays': init_relays,
    'co2_sensor': init_co2_sensor,
    'co2_sensor_2': init_co2_sensor_2,
    'o2_sensor': init_o2_sensor,
    'i2c': init_i2c,
}

