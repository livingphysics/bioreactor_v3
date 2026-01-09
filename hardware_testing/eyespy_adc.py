# -*- coding: utf-8 -*-
"""
Standalone function to read from eyespy board (ADS1114 ADC) at I2C address 0x49.

The eyespy board uses an ADS1114 16-bit ADC chip for reading photodiode values.
"""
from __future__ import annotations

import struct
import time
from typing import Final


# ADS1114 register addresses
_CONVERSION: Final[int] = 0x00
_CONFIG: Final[int] = 0x01

# Data rate (samples/s); conversion time ~= 1/DR
_DATA_RATE: Final[int] = 128

# Gain -> PGA bitfield (Config[11:9]) per datasheet
_PGA_BITS: Final[dict[float, int]] = {
    2 / 3: 0b000,  # ±6.144 V
    1.0: 0b001,  # ±4.096 V
    2.0: 0b010,  # ±2.048 V
    4.0: 0b011,  # ±1.024 V
    8.0: 0b100,  # ±0.512 V
    16.0: 0b101,  # ±0.256 V (also 110/111 map to the same FSR)
}

# DR code (Config[7:5]) per datasheet
_DR_CODE: Final[dict[int, int]] = {
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
_COMP_BITS: Final[int] = 0x0003  # bits [1:0] = 11; bits [4:2] left at reset (0)

# Default I2C address for eyespy board (pd2)
_EYESPY_I2C_ADDRESS: Final[int] = 0x49

# Default I2C bus (typically /dev/i2c-1 on Raspberry Pi)
_I2C_BUS: Final[int] = 1


def _build_config(gain: float, start: bool) -> int:
    """Build the configuration register value for ADS1114."""
    # Bit 15: OS (write 1 to start in single-shot, reads back 0 while converting)
    os_bit = 1 if start else 0

    # Bits 14:12 are RESERVED on ADS1114 -> write 000b
    reserved_14_12 = 0

    # PGA bits [11:9]
    pga_bits = _PGA_BITS[gain] & 0b111

    # MODE bit [8]: 1 = single-shot, 0 = continuous
    mode_bit = 1

    # DR bits [7:5]
    dr_bits = _DR_CODE[_DATA_RATE] & 0b111

    # Comparator/control bits [4:0] (we disable comparator)
    comp_bits = _COMP_BITS & 0x1F

    cfg = (
        (os_bit << 15)
        | (reserved_14_12 << 12)
        | (pga_bits << 9)
        | (mode_bit << 8)
        | (dr_bits << 5)
        | comp_bits
    )
    return cfg & 0xFFFF


def _write_register(bus, i2c_addr: int, reg: int, value: int) -> None:
    """Write a 16-bit value to an ADS1114 register."""
    data = [(value >> 8) & 0xFF, value & 0xFF]
    bus.write_i2c_block_data(i2c_addr, reg, data)


def _read_config_ready(bus, i2c_addr: int) -> bool:
    """Check if conversion is ready by reading the OS bit (Config[15])."""
    msb, lsb = bus.read_i2c_block_data(i2c_addr, _CONFIG, 2)
    cfg = (msb << 8) | lsb
    return bool(cfg & (1 << 15))  # OS bit


def read_eyespy_adc(
    i2c_address: int = _EYESPY_I2C_ADDRESS,
    i2c_bus: int = _I2C_BUS,
    gain: float = 1.0,
) -> int:
    """
    Read a single conversion from the eyespy board (ADS1114 ADC) at I2C address 0x49.

    Args:
        i2c_address: I2C address of the ADS1114 (default: 0x49 for eyespy/pd2)
        i2c_bus: I2C bus number (default: 1 for /dev/i2c-1)
        gain: PGA gain setting (default: 1.0 for ±4.096 V range)
              Valid values: 2/3, 1.0, 2.0, 4.0, 8.0, 16.0

    Returns:
        Raw 16-bit signed integer ADC reading (-32768 to 32767)

    Raises:
        ValueError: If gain is not supported
        OSError: If I2C communication fails

    Example:
        >>> reading = read_eyespy_adc()
        >>> print(f"Raw ADC value: {reading}")
    """
    from smbus2 import SMBus

    if gain not in _PGA_BITS:
        raise ValueError(f"Unsupported ADS1114 gain: {gain}. Valid values: {list(_PGA_BITS.keys())}")

    # Open I2C bus
    bus = SMBus(i2c_bus)

    try:
        # Set initial configuration (without starting conversion)
        cfg = _build_config(gain, start=False)
        _write_register(bus, i2c_address, _CONFIG, cfg)

        # Start a single-shot conversion
        cfg = _build_config(gain, start=True)
        _write_register(bus, i2c_address, _CONFIG, cfg)

        # Poll OS bit (Config[15]) until conversion completes
        # At 128 SPS, max ~7.8 ms; include a tiny sleep to avoid busy loop
        for _ in range(50):
            if _read_config_ready(bus, i2c_address):
                break
            time.sleep(0.001)
        else:
            # If we somehow never saw OS=1, fall through and still read conversion
            pass

        # Read conversion register (MSB first), convert to signed
        msb, lsb = bus.read_i2c_block_data(i2c_address, _CONVERSION, 2)
        value = struct.unpack(">h", bytes((msb, lsb)))[0]
        return int(value)

    finally:
        bus.close()

