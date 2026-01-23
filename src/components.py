"""
Component initialization functions for bioreactor hardware.
Each function initializes a specific component and returns a dict with the component objects.
"""

import logging
import struct
import time
from typing import Final

logger = logging.getLogger("Bioreactor.Components")

# ============================================================================
# Eyespy ADC (ADS1114) support functions
# ============================================================================

# ADS1114 register addresses
_EYESPY_CONVERSION: Final[int] = 0x00
_EYESPY_CONFIG: Final[int] = 0x01

# Data rate (samples/s); conversion time ~= 1/DR
_EYESPY_DATA_RATE: Final[int] = 128

# Gain -> PGA bitfield (Config[11:9]) per datasheet
_EYESPY_PGA_BITS: Final[dict[float, int]] = {
    2 / 3: 0b000,  # ±6.144 V
    1.0: 0b001,  # ±4.096 V
    2.0: 0b010,  # ±2.048 V
    4.0: 0b011,  # ±1.024 V
    8.0: 0b100,  # ±0.512 V
    16.0: 0b101,  # ±0.256 V (also 110/111 map to the same FSR)
}

# DR code (Config[7:5]) per datasheet
_EYESPY_DR_CODE: Final[dict[int, int]] = {
    8: 0b000,
    16: 0b001,
    32: 0b010,
    64: 0b011,
    128: 0b100,
    250: 0b101,
    475: 0b110,
    860: 0b111,
}

# Comparator disabled (COMP_QUE = 11), active-low, non-latching, traditional
_EYESPY_COMP_BITS: Final[int] = 0x0003  # bits [1:0] = 11; bits [4:2] left at reset (0)


def _eyespy_build_config(gain: float, start: bool) -> int:
    """Build the configuration register value for ADS1114."""
    # Bit 15: OS (write 1 to start in single-shot, reads back 0 while converting)
    os_bit = 1 if start else 0

    # Bits 14:12 are RESERVED on ADS1114 -> write 000b
    reserved_14_12 = 0

    # PGA bits [11:9]
    pga_bits = _EYESPY_PGA_BITS[gain] & 0b111

    # MODE bit [8]: 1 = single-shot, 0 = continuous
    mode_bit = 1

    # DR bits [7:5]
    dr_bits = _EYESPY_DR_CODE[_EYESPY_DATA_RATE] & 0b111

    # Comparator/control bits [4:0] (we disable comparator)
    comp_bits = _EYESPY_COMP_BITS & 0x1F

    cfg = (
        (os_bit << 15)
        | (reserved_14_12 << 12)
        | (pga_bits << 9)
        | (mode_bit << 8)
        | (dr_bits << 5)
        | comp_bits
    )
    return cfg & 0xFFFF


def _eyespy_write_register(bus, i2c_addr: int, reg: int, value: int) -> None:
    """Write a 16-bit value to an ADS1114 register."""
    data = [(value >> 8) & 0xFF, value & 0xFF]
    bus.write_i2c_block_data(i2c_addr, reg, data)


def _eyespy_read_config_ready(bus, i2c_addr: int) -> bool:
    """Check if conversion is ready by reading the OS bit (Config[15])."""
    msb, lsb = bus.read_i2c_block_data(i2c_addr, _EYESPY_CONFIG, 2)
    cfg = (msb << 8) | lsb
    return bool(cfg & (1 << 15))  # OS bit


def _read_eyespy_adc(
    i2c_address: int,
    i2c_bus: int,
    gain: float = 1.0,
) -> int:
    """
    Read a single conversion from the eyespy board (ADS1114 ADC).
    
    Args:
        i2c_address: I2C address of the ADS1114
        i2c_bus: I2C bus number (typically 1 for /dev/i2c-1)
        gain: PGA gain setting (default: 1.0 for ±4.096 V range)
              Valid values: 2/3, 1.0, 2.0, 4.0, 8.0, 16.0
    
    Returns:
        Raw 16-bit signed integer ADC reading (-32768 to 32767)
    
    Raises:
        ValueError: If gain is not supported
        OSError: If I2C communication fails
    """
    try:
        from smbus2 import SMBus
    except ImportError:
        raise ImportError("smbus2 is required for eyespy ADC. Install with: pip install smbus2")
    
    if gain not in _EYESPY_PGA_BITS:
        raise ValueError(f"Unsupported ADS1114 gain: {gain}. Valid values: {list(_EYESPY_PGA_BITS.keys())}")
    
    # Open I2C bus
    bus = SMBus(i2c_bus)
    
    try:
        # Set initial configuration (without starting conversion)
        cfg = _eyespy_build_config(gain, start=False)
        _eyespy_write_register(bus, i2c_address, _EYESPY_CONFIG, cfg)
        
        # Start a single-shot conversion
        cfg = _eyespy_build_config(gain, start=True)
        _eyespy_write_register(bus, i2c_address, _EYESPY_CONFIG, cfg)
        
        # Poll OS bit (Config[15]) until conversion completes
        # At 128 SPS, max ~7.8 ms; include a tiny sleep to avoid busy loop
        for _ in range(50):
            if _eyespy_read_config_ready(bus, i2c_address):
                break
            time.sleep(0.001)
        else:
            # If we somehow never saw OS=1, fall through and still read conversion
            pass
        
        # Read conversion register (MSB first), convert to signed
        msb, lsb = bus.read_i2c_block_data(i2c_address, _EYESPY_CONVERSION, 2)
        value = struct.unpack(">h", bytes((msb, lsb)))[0]
        return int(value)
    
    finally:
        bus.close()


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


def init_ring_light(bioreactor, config):
    """
    Initialize ring light (neopixel) using pi5neo library.
    
    Args:
        bioreactor: Bioreactor instance
        config: Configuration object with ring light settings
        
    Returns:
        dict: {'initialized': bool, 'driver': RingLightDriver}
    """
    try:
        from .io import RingLightDriver
    except Exception as import_error:
        logger.error(f"Ring light driver dependencies missing: {import_error}")
        return {'initialized': False, 'error': str(import_error)}
    
    spi_device = getattr(config, 'RING_LIGHT_SPI_DEVICE', '/dev/spidev0.0')
    num_leds = getattr(config, 'RING_LIGHT_COUNT', 32)
    spi_speed = getattr(config, 'RING_LIGHT_SPI_SPEED', 800)  # kHz
    
    try:
        driver = RingLightDriver(bioreactor, spi_device, num_leds, spi_speed)
        # Initialize by turning off (ensures hardware is ready)
        driver.off()
        bioreactor.ring_light_driver = driver
        logger.info(f"Ring light initialized: {num_leds} LEDs on {spi_device} at {spi_speed}kHz")
        return {'initialized': True, 'driver': driver}
    except Exception as e:
        logger.error(f"Ring light initialization failed: {e}")
        return {'initialized': False, 'error': str(e)}


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


def init_eyespy_adc(bioreactor, config):
    """
    Initialize eyespy ADC component (ADS1114 based, from pioreactor pattern).
    
    Supports multiple eyespy boards at different I2C addresses.
    Each eyespy board is a single-channel ADS1114 ADC.
    
    Args:
        bioreactor: Bioreactor instance
        config: Configuration object with EYESPY_ADC configuration
        
    Returns:
        dict: {'initialized': bool, 'eyespy_boards': dict of board configs}
    """
    try:
        # Test that smbus2 is available
        from smbus2 import SMBus
    except ImportError as import_error:
        logger.error(f"Eyespy ADC dependencies missing: {import_error}. Install with: pip install smbus2")
        return {'initialized': False, 'error': str(import_error)}
    
    try:
        # Get eyespy ADC configuration from config
        eyespy_config = getattr(config, 'EYESPY_ADC', {})
        
        if not eyespy_config:
            logger.warning("No EYESPY_ADC configuration found, using defaults")
            # Default: single eyespy board at address 0x49
            eyespy_config = {
                'eyespy1': {
                    'i2c_address': 0x49,
                    'i2c_bus': 1,
                    'gain': 1.0,  # ±4.096 V range
                }
            }
        
        eyespy_boards = {}
        
        # Store configuration for each eyespy board
        for board_name, board_cfg in eyespy_config.items():
            i2c_address = board_cfg.get('i2c_address', 0x49)
            i2c_bus = board_cfg.get('i2c_bus', 1)
            gain = board_cfg.get('gain', 1.0)
            
            eyespy_boards[board_name] = {
                'i2c_address': i2c_address,
                'i2c_bus': i2c_bus,
                'gain': gain,
            }
            logger.info(f"Eyespy ADC board {board_name} configured: address={hex(i2c_address)}, bus={i2c_bus}, gain={gain}")
        
        if not eyespy_boards:
            error_msg = "No valid eyespy ADC boards configured"
            logger.error(error_msg)
            return {'initialized': False, 'error': error_msg}
        
        # Store on bioreactor instance
        bioreactor.eyespy_boards = eyespy_boards
        bioreactor._eyespy_read_func = _read_eyespy_adc  # Store the read function
        
        logger.info(f"Eyespy ADC initialized with {len(eyespy_boards)} board(s)")
        
        return {'initialized': True, 'eyespy_boards': eyespy_boards}
    except Exception as e:
        logger.error(f"Eyespy ADC initialization failed: {e}")
        return {'initialized': False, 'error': str(e)}


def init_co2_sensor(bioreactor, config):
    """
    Initialize CO2 sensor via I2C.
    
    Args:
        bioreactor: Bioreactor instance
        config: Configuration object with CO2_SENSOR configuration
        
    Returns:
        dict: {'initialized': bool}
    """
    try:
        # Get CO2 sensor configuration from config
        co2_enabled = getattr(config, 'CO2_SENSOR_ENABLED', False)
        co2_i2c_bus = getattr(config, 'CO2_SENSOR_I2C_BUS', 1)
        co2_type = getattr(config, 'CO2_SENSOR_TYPE', 'sensair_k33').lower()

        if not co2_enabled:
            logger.info("CO2 sensor disabled in configuration")
            return {'initialized': False, 'error': 'CO2 sensor disabled'}
        
        # Set default I2C address based on sensor type if not specified
        co2_i2c_address = getattr(config, 'CO2_SENSOR_I2C_ADDRESS', None)
        if co2_i2c_address is None:
            # Use type-specific defaults
            if co2_type.startswith('sensair'):
                co2_i2c_address = 0x68  # Default for Senseair K33
            elif co2_type.startswith('atlas'):
                co2_i2c_address = 0x69  # Default for Atlas Scientific
            else:
                # Fallback to 0x68 if unknown type
                co2_i2c_address = 0x68
                logger.warning(f"Unknown CO2 sensor type '{co2_type}', using default address 0x68")

        # Check dependencies based on sensor type
        if co2_type.startswith('sensair'):
            try:
                from smbus2 import SMBus  # noqa: F401
            except ImportError as import_error:
                logger.error(f"CO2 sensor (Senseair K33) dependencies missing: {import_error}. Install with: pip install smbus2")
                return {'initialized': False, 'error': str(import_error)}
        elif co2_type.startswith('atlas'):
            try:
                # atlas_i2c is provided by Atlas Scientific library (external)
                from atlas_i2c import atlas_i2c
                # Initialize Atlas I2C device once during startup
                atlas_device = atlas_i2c.AtlasI2C()
                atlas_device.set_i2c_address(co2_i2c_address)
                logger.info(f"Atlas CO2 sensor device initialized at address 0x{co2_i2c_address:02X}")
            except ImportError as import_error:
                logger.error(f"CO2 sensor (Atlas) dependencies missing: {import_error}. Install the atlas_i2c library.")
                return {'initialized': False, 'error': str(import_error)}
            except Exception as e:
                logger.error(f"Failed to initialize Atlas CO2 sensor device: {e}")
                return {'initialized': False, 'error': str(e)}
        else:
            error_msg = f"Unsupported CO2 sensor type: {co2_type}"
            logger.error(error_msg)
            return {'initialized': False, 'error': error_msg}

        # Store configuration on bioreactor instance
        bioreactor.co2_sensor_config = {
            'i2c_address': co2_i2c_address,
            'i2c_bus': co2_i2c_bus,
            'type': co2_type,
        }
        
        # Store Atlas device if initialized
        if co2_type.startswith('atlas'):
            bioreactor.co2_sensor_config['atlas_device'] = atlas_device

        logger.info(
            f"CO2 sensor initialized: type={co2_type}, address={hex(co2_i2c_address)}, bus={co2_i2c_bus}"
        )

        return {'initialized': True}
    except Exception as e:
        logger.error(f"CO2 sensor initialization failed: {e}")
        return {'initialized': False, 'error': str(e)}


def init_pumps(bioreactor, config):
    """
    Initialize pump controllers using ticUSB protocol.
    
    Args:
        bioreactor: Bioreactor instance
        config: Configuration object with PUMPS configuration
        
    Returns:
        dict: {'initialized': bool, 'pumps': dict of pump objects}
    """
    try:
        from ticlib import TicUSB
    except ImportError as import_error:
        logger.error(f"Pump dependencies missing: {import_error}. Install with: pip install ticlib")
        return {'initialized': False, 'error': str(import_error)}
    
    try:
        # Get pump configuration from config
        pumps_config = getattr(config, 'PUMPS', {})
        
        if not pumps_config:
            logger.warning("No pumps configured in PUMPS dictionary")
            return {'initialized': False, 'error': 'No pumps configured'}
        
        pumps = {}
        pump_configs = {}
        
        # Initialize each pump
        for pump_name, pump_cfg in pumps_config.items():
            serial = pump_cfg.get('serial')
            step_mode = pump_cfg.get('step_mode', 3)
            current_limit = pump_cfg.get('current_limit', 32)
            
            if not serial:
                logger.error(f"Pump {pump_name} missing serial number in configuration")
                continue
            
            try:
                # Initialize TicUSB device
                pump = TicUSB(serial_number=serial)
                
                # Configure pump
                pump.energize()
                pump.exit_safe_start()
                pump.set_step_mode(step_mode)
                pump.set_current_limit(current_limit)
                
                # Store pump object and config
                pumps[pump_name] = pump
                pump_configs[pump_name] = {
                    'serial': serial,
                    'step_mode': step_mode,
                    'current_limit': current_limit,
                }
                
                logger.info(f"Pump {pump_name} initialized (serial: {serial})")
                
            except Exception as e:
                logger.error(f"Failed to initialize pump {pump_name} (serial: {serial}): {e}")
                continue
        
        if not pumps:
            error_msg = "No pumps successfully initialized"
            logger.error(error_msg)
            return {'initialized': False, 'error': error_msg}
        
        # Store on bioreactor instance
        bioreactor.pumps = pumps
        bioreactor.pump_configs = pump_configs
        
        logger.info(f"Pumps initialized: {len(pumps)} pump(s) - {list(pumps.keys())}")
        
        return {'initialized': True, 'pumps': pumps}
    except Exception as e:
        logger.error(f"Pump initialization failed: {e}")
        return {'initialized': False, 'error': str(e)}


# Component registry - maps component names to initialization functions
COMPONENT_REGISTRY = {
    'i2c': init_i2c,
    'temp_sensor': init_temp_sensor,
    'peltier_driver': init_peltier_driver,
    'stirrer': init_stirrer,
    'led': init_led,
    'ring_light': init_ring_light,
    'optical_density': init_optical_density,
    'eyespy_adc': init_eyespy_adc,
    'co2_sensor': init_co2_sensor,
    'pumps': init_pumps,
}

