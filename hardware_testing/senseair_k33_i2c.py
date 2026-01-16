import time

try:
    from smbus2 import SMBus
except ImportError:
    from smbus import SMBus


# Senseair K33/K30 I2C read CO2 command (common example)
_REG_READ_CO2 = 0x22
_CMD_READ_CO2 = [0x00, 0x08, 0x2A]
_REPLY_LEN = 4


def read_co2_ppm(bus_num=1, address=0x68, delay_s=0.05, retries=3):
    """
    Read CO2 (ppm) from Senseair K33 over I2C.
    bus_num: I2C bus number (e.g., 1 on RPi)
    address: I2C address of sensor (0x68 shows as "68" in i2cdetect)
    """
    last_error = None
    for _ in range(retries):
        try:
            with SMBus(bus_num) as bus:
                bus.write_i2c_block_data(address, _REG_READ_CO2, _CMD_READ_CO2)
                time.sleep(delay_s)
                data = bus.read_i2c_block_data(address, _REG_READ_CO2, _REPLY_LEN)
            if len(data) < _REPLY_LEN:
                raise RuntimeError(f"Short read from K33: {data}")
            return (data[2] << 8) | data[3]
        except OSError as exc:
            last_error = exc
            time.sleep(delay_s)

    raise RuntimeError(f"I2C read failed after {retries} retries: {last_error}")


def read_co2_ppm_with_bus(bus, address=0x68, delay_s=0.05, retries=3):
    """
    Same as read_co2_ppm but reuses an existing SMBus instance.
    """
    last_error = None
    for _ in range(retries):
        try:
            bus.write_i2c_block_data(address, _REG_READ_CO2, _CMD_READ_CO2)
            time.sleep(delay_s)
            data = bus.read_i2c_block_data(address, _REG_READ_CO2, _REPLY_LEN)
            if len(data) < _REPLY_LEN:
                raise RuntimeError(f"Short read from K33: {data}")
            return (data[2] << 8) | data[3]
        except OSError as exc:
            last_error = exc
            time.sleep(delay_s)

    raise RuntimeError(f"I2C read failed after {retries} retries: {last_error}")


if __name__ == "__main__":
    while True:
        try:
            co2 = read_co2_ppm()
            print(f"CO2: {co2} ppm")
        except Exception as exc:
            print(f"Read error: {exc}")
        time.sleep(1)
