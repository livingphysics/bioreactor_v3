"""
Senseair K33 CO2 Sensor I2C Library

A Python library for reading CO2 concentration from Senseair K33 sensors via I2C.

Example usage:
    from sensair_k33 import SenseairK33
    
    # Create sensor instance
    sensor = SenseairK33(bus_num=1, i2c_addr=0x68)
    
    # Read CO2 value
    co2_ppm = sensor.read_co2()
    print(f"CO2: {co2_ppm} ppm")
    
    # Or use functional interface
    from sensair_k33 import read_co2
    co2_ppm = read_co2(bus_num=1)
"""

from smbus2 import SMBus, i2c_msg
import time
from typing import Optional, List


# Senseair K33 protocol constants
DEFAULT_I2C_ADDRESS = 0x68
READRAM_CMD = 0x22
CO2_RAM_ADDR_HI = 0x00
CO2_RAM_ADDR_LO = 0x08


class SenseairK33Error(Exception):
    """Base exception for Senseair K33 sensor errors."""
    pass


class SenseairK33IOError(SenseairK33Error):
    """I/O error when communicating with the sensor."""
    pass


class SenseairK33ChecksumError(SenseairK33Error):
    """Checksum validation error."""
    pass


class SenseairK33StatusError(SenseairK33Error):
    """Sensor status byte indicates an error."""
    pass


def calc_checksum(data_bytes: List[int]) -> int:
    """
    Calculate checksum for Senseair K33 protocol.
    Simple sum-based checksum mod 256.
    
    Args:
        data_bytes: List of bytes to calculate checksum for
        
    Returns:
        Checksum byte (0-255)
    """
    return sum(data_bytes) & 0xFF


class SenseairK33:
    """
    Senseair K33 CO2 Sensor I2C Interface
    
    This class provides an interface to read CO2 concentration from a Senseair K33
    sensor connected via I2C.
    
    Attributes:
        bus_num (int): I2C bus number (typically 1 for /dev/i2c-1)
        i2c_addr (int): I2C address of the sensor (default: 0x68)
    
    Example:
        >>> sensor = SenseairK33(bus_num=1, i2c_addr=0x68)
        >>> co2_ppm = sensor.read_co2()
        >>> print(f"CO2: {co2_ppm} ppm")
    """
    
    def __init__(self, bus_num: int = 1, i2c_addr: int = DEFAULT_I2C_ADDRESS):
        """
        Initialize Senseair K33 sensor interface.
        
        Args:
            bus_num: I2C bus number (default: 1)
            i2c_addr: I2C address of the sensor (default: 0x68)
        """
        self.bus_num = bus_num
        self.i2c_addr = i2c_addr
    
    def read_co2(self, debug: bool = False) -> int:
        """
        Read CO2 concentration from the sensor.
        
        Args:
            debug: If True, print debug information about the communication
            
        Returns:
            CO2 concentration in ppm (integer, multiplied by 10)
            
        Raises:
            SenseairK33IOError: If I2C communication fails
            SenseairK33ChecksumError: If checksum validation fails
            SenseairK33StatusError: If sensor status indicates an error
        """
        try:
            with SMBus(self.bus_num) as bus:
                # Prepare ReadRAM command frame
                # According to Senseair K33 protocol:
                # Write frame: [command, addr_hi, addr_lo, checksum]
                # Checksum = sum(command + addr_hi + addr_lo) & 0xFF
                command = READRAM_CMD
                addr_hi = CO2_RAM_ADDR_HI
                addr_lo = CO2_RAM_ADDR_LO
                
                # Calculate checksum for write command
                write_checksum = calc_checksum([command, addr_hi, addr_lo])
                write_packet = [command, addr_hi, addr_lo, write_checksum]
                
                if debug:
                    print(f"Write packet: {[f'0x{b:02X}' for b in write_packet]}")
                    print(f"  Command: 0x{command:02X}, Addr: 0x{addr_hi:02X}{addr_lo:02X}, Checksum: 0x{write_checksum:02X}")
                
                # Use raw i2c_msg for reliable communication
                write_msg = i2c_msg.write(self.i2c_addr, write_packet)
                bus.i2c_rdwr(write_msg)
                time.sleep(0.05)  # Wait for sensor to prepare data (50ms recommended)
                
                # Read response: 4 bytes [status, co2_high, co2_low, checksum]
                read_msg = i2c_msg.read(self.i2c_addr, 4)
                bus.i2c_rdwr(read_msg)
                response = list(read_msg)
                
                if debug:
                    print(f"Read response: {[f'0x{b:02X}' for b in response]}")
                
                if len(response) < 4:
                    raise SenseairK33IOError(
                        f"Incomplete response: got {len(response)} bytes, expected 4"
                    )
                
                status = response[0]
                co2_high = response[1]
                co2_low = response[2]
                read_checksum = response[3]
                
                # Validate status byte (bit 0 should be 1 for success)
                if (status & 0x01) == 0:
                    raise SenseairK33StatusError(
                        f"Senseair K33 error: status byte indicates failure (0x{status:02X})"
                    )
                
                # Validate checksum
                expected_checksum = calc_checksum([status, co2_high, co2_low])
                if read_checksum != expected_checksum:
                    raise SenseairK33ChecksumError(
                        f"Checksum mismatch: got 0x{read_checksum:02X}, "
                        f"expected 0x{expected_checksum:02X}"
                    )
                
                # Combine high and low bytes to get CO2 value, multiply by 10 to get PPM
                co2_raw = (co2_high << 8) | co2_low
                co2_ppm = co2_raw * 10
                
                return co2_ppm
        
        except OSError as e:
            if e.errno == 121:
                raise SenseairK33IOError(
                    f"Remote I/O error (121): Device not responding at address 0x{self.i2c_addr:02X} "
                    f"on bus {self.bus_num}. Check wiring, power, and I2C address."
                ) from e
            else:
                raise SenseairK33IOError(f"I2C communication error: {e}") from e
        except (SenseairK33IOError, SenseairK33ChecksumError, SenseairK33StatusError):
            raise
        except Exception as e:
            raise SenseairK33Error(f"Failed to read from Senseair K33: {e}") from e
    
    def read_continuous(self, interval: float = 1.0, callback=None):
        """
        Continuously read CO2 values from the sensor.
        
        Args:
            interval: Time between readings in seconds (default: 1.0)
            callback: Optional callback function that receives (co2_ppm, timestamp) tuples.
                     If None, prints to stdout.
        
        Example:
            >>> def my_callback(co2, timestamp):
            ...     print(f"CO2: {co2} ppm at {timestamp}")
            >>> sensor.read_continuous(interval=1.0, callback=my_callback)
        """
        if callback is None:
            def default_callback(co2, timestamp):
                print(f"CO2: {co2} ppm")
            callback = default_callback
        
        print(f"Reading CO2 from Senseair K33 (I2C addr: 0x{self.i2c_addr:02X}, bus: {self.bus_num})")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                try:
                    co2 = self.read_co2()
                    callback(co2, time.time())
                    time.sleep(interval)
                except SenseairK33Error as e:
                    print(f"Error: {e}")
                    time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped reading.")


# Functional interface for backward compatibility and convenience
def read_co2(bus_num: int = 1, i2c_addr: int = DEFAULT_I2C_ADDRESS, debug: bool = False) -> int:
    """
    Read CO2 concentration from Senseair K33 sensor via I2C (functional interface).
    
    This is a convenience function that creates a temporary SenseairK33 instance
    and reads the CO2 value.
    
    Args:
        bus_num: I2C bus number (default: 1)
        i2c_addr: I2C address of the sensor (default: 0x68)
        debug: If True, print debug information about the communication
        
    Returns:
        CO2 concentration in ppm (integer, multiplied by 10)
        
    Raises:
        SenseairK33Error: If sensor communication fails
        
    Example:
        >>> co2_ppm = read_co2(bus_num=1)
        >>> print(f"CO2: {co2_ppm} ppm")
    """
    sensor = SenseairK33(bus_num=bus_num, i2c_addr=i2c_addr)
    return sensor.read_co2(debug=debug)


def scan_i2c_bus(bus_num: int = 1, verbose: bool = True) -> List[str]:
    """
    Scan I2C bus for devices using method similar to i2cdetect.
    
    Args:
        bus_num: I2C bus number to scan
        verbose: If True, print found devices to stdout
        
    Returns:
        List of found I2C addresses (as hex strings)
        
    Example:
        >>> devices = scan_i2c_bus(bus_num=1)
        >>> print(f"Found {len(devices)} devices")
    """
    found_devices = []
    try:
        with SMBus(bus_num) as bus:
            if verbose:
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
                    if verbose:
                        print(f"  Device found at address {addr_hex}")
    except Exception as e:
        if verbose:
            print(f"Error scanning bus: {e}")
        raise SenseairK33Error(f"Failed to scan I2C bus: {e}") from e
    
    if not found_devices and verbose:
        print("  No devices found")
    
    return found_devices


def read_co2_continuous(bus_num: int = 1, i2c_addr: int = DEFAULT_I2C_ADDRESS, 
                        interval: float = 1.0):
    """
    Continuously read CO2 values from the sensor (functional interface).
    
    Args:
        bus_num: I2C bus number
        i2c_addr: I2C address of the sensor
        interval: Time between readings in seconds (default: 1.0)
    
    Example:
        >>> read_co2_continuous(bus_num=1, interval=1.0)
    """
    sensor = SenseairK33(bus_num=bus_num, i2c_addr=i2c_addr)
    sensor.read_continuous(interval=interval)


# Module-level constants for convenience
K33_I2C_ADDR = DEFAULT_I2C_ADDRESS


if __name__ == "__main__":
    # Example usage when run as script
    import sys
    
    bus_num = 1
    if len(sys.argv) > 1:
        try:
            bus_num = int(sys.argv[1])
        except ValueError:
            print(f"Invalid bus number: {sys.argv[1]}, using default: 1")
    
    # Scan the bus
    print("Scanning I2C bus for devices...")
    devices = scan_i2c_bus(bus_num)
    
    if hex(DEFAULT_I2C_ADDRESS) not in devices:
        print(f"\nWarning: Device at address {hex(DEFAULT_I2C_ADDRESS)} not found!")
        print("Check wiring, power, and I2C address.")
    else:
        print(f"\nDevice found at {hex(DEFAULT_I2C_ADDRESS)}")
    
    # Try reading CO2
    print("\nAttempting to read CO2...")
    try:
        sensor = SenseairK33(bus_num=bus_num)
        co2_value = sensor.read_co2()
        print(f"CO2 concentration: {co2_value} ppm")
    except SenseairK33Error as e:
        print(f"Error reading CO2: {e}")
        print("\nTrying with debug output...")
        try:
            sensor = SenseairK33(bus_num=bus_num)
            co2_value = sensor.read_co2(debug=True)
            print(f"CO2 concentration: {co2_value} ppm")
        except SenseairK33Error as e2:
            print(f"Error: {e2}")
