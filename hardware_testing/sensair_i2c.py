"""
Senseair K33 CO2 Sensor I2C Interface
Uses smbus2 to read CO2 data from Senseair K33 sensor via I2C
"""

from smbus2 import SMBus
import time


# Senseair K33 default I2C address (7-bit)
K33_I2C_ADDR = 0x68

# ReadRAM command
READRAM_CMD = 0x22

# CO2 data RAM addresses (high byte, low byte)
CO2_RAM_ADDR_HI = 0x00
CO2_RAM_ADDR_LO = 0x08
CO2_BYTES = 2


def calc_checksum(data_bytes):
    """
    Calculate checksum for Senseair K33 protocol.
    Simple sum-based checksum mod 256.
    
    Args:
        data_bytes: List of bytes to calculate checksum for
        
    Returns:
        Checksum byte (0-255)
    """
    return sum(data_bytes) & 0xFF


def read_co2(bus_num=1, i2c_addr=K33_I2C_ADDR):
    """
    Read CO2 concentration from Senseair K33 sensor via I2C.
    
    Args:
        bus_num: I2C bus number (default: 1, use 68 if that's your bus)
        i2c_addr: I2C address of the sensor (default: 0x68)
        
    Returns:
        CO2 concentration in ppm (integer)
        
    Raises:
        IOError: If sensor communication fails or checksum is invalid
    """
    try:
        # Open I2C bus
        bus = SMBus(bus_num)
        
        # Prepare ReadRAM command frame
        # Format: [command, nbytes, addr_hi, addr_lo, checksum]
        command = READRAM_CMD
        nbytes = CO2_BYTES
        addr_hi = CO2_RAM_ADDR_HI
        addr_lo = CO2_RAM_ADDR_LO
        
        # Calculate checksum for write command
        write_checksum = calc_checksum([command, nbytes, addr_hi, addr_lo])
        
        # Write command to sensor
        # Using write_i2c_block_data: first byte is register/command, rest is data
        write_data = [nbytes, addr_hi, addr_lo, write_checksum]
        bus.write_i2c_block_data(i2c_addr, command, write_data)
        
        # Wait for sensor to prepare data (typically 10-50ms)
        time.sleep(0.02)
        
        # Read response: [status, co2_high, co2_low, checksum]
        read_length = 4
        response = bus.read_i2c_block_data(i2c_addr, 0x00, read_length)
        
        status = response[0]
        co2_high = response[1]
        co2_low = response[2]
        read_checksum = response[3]
        
        # Validate status byte (bit 0 should be 1 for success)
        if (status & 0x01) == 0:
            raise IOError(f"Senseair K33 error: status byte indicates failure (0x{status:02X})")
        
        # Validate checksum
        expected_checksum = calc_checksum([status, co2_high, co2_low])
        if read_checksum != expected_checksum:
            raise IOError(
                f"Checksum mismatch: got 0x{read_checksum:02X}, "
                f"expected 0x{expected_checksum:02X}"
            )
        
        # Combine high and low bytes to get CO2 value in ppm
        co2_ppm = (co2_high << 8) | co2_low
        
        return co2_ppm
        
    except Exception as e:
        raise IOError(f"Failed to read from Senseair K33: {e}")
    finally:
        # Close bus if it was opened
        if 'bus' in locals():
            bus.close()


def read_co2_continuous(bus_num=1, i2c_addr=K33_I2C_ADDR, interval=1.0):
    """
    Continuously read CO2 values from the sensor.
    
    Args:
        bus_num: I2C bus number
        i2c_addr: I2C address of the sensor
        interval: Time between readings in seconds (default: 1.0)
    """
    print(f"Reading CO2 from Senseair K33 (I2C addr: 0x{i2c_addr:02X}, bus: {bus_num})")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            try:
                co2 = read_co2(bus_num, i2c_addr)
                print(f"CO2: {co2} ppm")
                time.sleep(interval)
            except IOError as e:
                print(f"Error: {e}")
                time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped reading.")


if __name__ == "__main__":
    # Example usage
    # If you need bus 68, change bus_num to 68
    # If using standard Raspberry Pi I2C, bus_num is typically 1
    
    # Single reading
    try:
        co2_value = read_co2(bus_num=68)  # Change to your actual bus number
        print(f"CO2 concentration: {co2_value} ppm")
    except Exception as e:
        print(f"Error reading CO2: {e}")
    
    # Uncomment to run continuous readings:
    # read_co2_continuous(bus_num=68, interval=1.0)
