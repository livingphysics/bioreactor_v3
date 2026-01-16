"""
Senseair K33 CO2 Sensor I2C Interface
Uses smbus2 to read CO2 data from Senseair K33 sensor via I2C
"""

from smbus2 import SMBus, i2c_msg
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


def scan_i2c_bus(bus_num=1):
    """
    Scan I2C bus for devices using method similar to i2cdetect.
    Uses i2c_rdwr with empty write message to probe devices (most reliable method).
    
    Args:
        bus_num: I2C bus number to scan
        
    Returns:
        List of found I2C addresses (as hex strings)
    """
    found_devices = []
    try:
        with SMBus(bus_num) as bus:
            print(f"Scanning I2C bus {bus_num}...")
            for address in range(0x08, 0x78):
                # Skip reserved addresses (0x00-0x07 and 0x78-0x7F)
                device_found = False
                
                # Method 1: Try i2c_rdwr with empty write (most reliable, like i2cdetect)
                try:
                    write_msg = i2c_msg.write(address, [])
                    bus.i2c_rdwr(write_msg)
                    device_found = True
                except OSError:
                    # Method 2: Try reading a byte (works for some devices)
                    try:
                        bus.read_byte(address)
                        device_found = True
                    except OSError:
                        pass
                
                if device_found:
                    addr_hex = hex(address)
                    found_devices.append(addr_hex)
                    print(f"  Device found at address {addr_hex}")
    except Exception as e:
        print(f"Error scanning bus: {e}")
    
    if not found_devices:
        print("  No devices found")
    return found_devices


def read_co2(bus_num=1, i2c_addr=K33_I2C_ADDR, use_raw_i2c=False):
    """
    Read CO2 concentration from Senseair K33 sensor via I2C.
    
    Args:
        bus_num: I2C bus number (default: 1, use 68 if that's your bus)
        i2c_addr: I2C address of the sensor (default: 0x68)
        use_raw_i2c: If True, use raw i2c_msg for more control
        
    Returns:
        CO2 concentration in ppm (integer)
        
    Raises:
        IOError: If sensor communication fails or checksum is invalid
    """
    try:
        with SMBus(bus_num) as bus:
            # Prepare ReadRAM command frame
            # Format: [command, nbytes, addr_hi, addr_lo, checksum]
            command = READRAM_CMD
            nbytes = CO2_BYTES
            addr_hi = CO2_RAM_ADDR_HI
            addr_lo = CO2_RAM_ADDR_LO
            
            # Calculate checksum for write command
            write_checksum = calc_checksum([command, nbytes, addr_hi, addr_lo])
            
            if use_raw_i2c:
                # Method 1: Use raw i2c_msg for more control
                write_data = [command, nbytes, addr_hi, addr_lo, write_checksum]
                write_msg = i2c_msg.write(i2c_addr, write_data)
                read_msg = i2c_msg.read(i2c_addr, 4)
                
                bus.i2c_rdwr(write_msg)
                time.sleep(0.05)  # Wait for sensor to prepare data
                bus.i2c_rdwr(read_msg)
                
                response = list(read_msg)
            else:
                # Method 2: Use block data write/read
                # Try writing command byte first, then data
                try:
                    # Alternative: write byte data
                    bus.write_byte(i2c_addr, command)
                    time.sleep(0.01)
                    bus.write_byte(i2c_addr, nbytes)
                    bus.write_byte(i2c_addr, addr_hi)
                    bus.write_byte(i2c_addr, addr_lo)
                    bus.write_byte(i2c_addr, write_checksum)
                except:
                    # Fallback: write block data
                    write_data = [nbytes, addr_hi, addr_lo, write_checksum]
                    bus.write_i2c_block_data(i2c_addr, command, write_data)
                
                # Wait for sensor to prepare data (typically 10-50ms)
                time.sleep(0.05)
                
                # Read response: [status, co2_high, co2_low, checksum]
                try:
                    # Try reading block data
                    response = bus.read_i2c_block_data(i2c_addr, 0x00, 4)
                except:
                    # Fallback: read bytes individually
                    response = []
                    for i in range(4):
                        response.append(bus.read_byte(i2c_addr))
                        time.sleep(0.001)
            
            if len(response) < 4:
                raise IOError(f"Incomplete response: got {len(response)} bytes, expected 4")
            
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
        
    except OSError as e:
        if e.errno == 121:
            raise IOError(
                f"Remote I/O error (121): Device not responding at address 0x{i2c_addr:02X} on bus {bus_num}. "
                f"Check wiring, power, and I2C address. Run scan_i2c_bus({bus_num}) to detect devices."
            )
        else:
            raise IOError(f"I2C communication error: {e}")
    except Exception as e:
        raise IOError(f"Failed to read from Senseair K33: {e}")


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
    
    bus_num = 68  # Change to your actual bus number
    
    # First, scan the bus to see if device is detected
    print("Scanning I2C bus for devices...")
    devices = scan_i2c_bus(bus_num)
    
    if hex(K33_I2C_ADDR) not in devices:
        print(f"\nWarning: Device at address {hex(K33_I2C_ADDR)} not found!")
        print("Check wiring, power, and I2C address.")
    else:
        print(f"\nDevice found at {hex(K33_I2C_ADDR)}")
    
    # Try reading CO2
    print("\nAttempting to read CO2...")
    try:
        # Try standard method first
        co2_value = read_co2(bus_num=bus_num)
        print(f"CO2 concentration: {co2_value} ppm")
    except Exception as e:
        print(f"Error with standard method: {e}")
        print("\nTrying alternative method (raw I2C)...")
        try:
            co2_value = read_co2(bus_num=bus_num, use_raw_i2c=True)
            print(f"CO2 concentration: {co2_value} ppm")
        except Exception as e2:
            print(f"Error with alternative method: {e2}")
    
    # Uncomment to run continuous readings:
    # read_co2_continuous(bus_num=bus_num, interval=1.0)
