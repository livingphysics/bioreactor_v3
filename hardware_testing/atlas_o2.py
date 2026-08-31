"""
Atlas Scientific O2 Sensor I2C Library

A small wrapper around the atlas_i2c library for reading oxygen concentration
from an Atlas Scientific O2 (EZO) sensor. Mirrors the interface style of
sensair_k33.py so both sensors can be driven the same way.

The atlas_i2c import is deferred until the first read, so importing this module
works even on machines where the library is not installed.

Example usage:
    from atlas_o2 import AtlasO2

    sensor = AtlasO2(i2c_addr=0x6C)
    o2_percent = sensor.read_o2()
    print(f"O2: {o2_percent} %")

Usage as a script:
    python hardware_testing/atlas_o2.py [i2c_addr_hex]
"""

import time
from typing import Optional


# Atlas Scientific O2 sensor defaults
DEFAULT_I2C_ADDRESS = 0x6C

# Milliseconds the sensor needs to process an "R" (read) command
# (datasheet: 900 ms; a little headroom does no harm)
DEFAULT_PROCESSING_DELAY = 1500

# Extra attempts per command, and the pause before each one. The Pi 5 I2C
# controller times out in bursts when a device stretches the clock
# ("i2c_designware ...: controller timed out" in dmesg), so a transfer that
# fails now often succeeds a moment later on a freshly opened device file.
DEFAULT_RETRIES = 2
RETRY_BACKOFF_S = 0.4

# Response code 254 means "still processing" — re-reading (without re-sending
# the command) after a short wait usually returns the pending answer.
NOT_READY_WAIT_S = 0.3

# I2C response codes from the EZO-O2 datasheet
STATUS_SUCCESS = 1
STATUS_NOT_READY = 254
STATUS_MESSAGES = {
    1: "successful request",
    2: "syntax error",
    254: "still processing, not ready (processing delay too short)",
    255: "no data to send",
}


class AtlasO2Error(Exception):
    """Base exception for Atlas O2 sensor errors."""
    pass


class AtlasO2ImportError(AtlasO2Error):
    """The atlas_i2c library is not installed."""
    pass


class AtlasO2IOError(AtlasO2Error):
    """I/O error when communicating with the sensor."""
    pass


class AtlasO2:
    """
    Atlas Scientific O2 Sensor I2C interface.

    Attributes:
        i2c_addr (int): I2C address of the sensor (default: 0x6C)
        processing_delay (int): Delay in ms allowed for the sensor to answer "R"

    Example:
        >>> sensor = AtlasO2(i2c_addr=0x6C)
        >>> o2 = sensor.read_o2()
        >>> print(f"O2: {o2} %")
    """

    def __init__(self, i2c_addr: int = DEFAULT_I2C_ADDRESS,
                 processing_delay: int = DEFAULT_PROCESSING_DELAY,
                 retries: int = DEFAULT_RETRIES):
        """
        Initialize the Atlas O2 sensor interface.

        Args:
            i2c_addr: I2C address of the sensor (default: 0x6C)
            processing_delay: Time in ms to wait for the sensor's response
            retries: Extra attempts per command after an I2C error
        """
        self.i2c_addr = i2c_addr
        self.processing_delay = processing_delay
        self.retries = retries
        self._device = None

    def connect(self):
        """
        Create the underlying AtlasI2C device (idempotent).

        Raises:
            AtlasO2ImportError: If the atlas_i2c library is unavailable
            AtlasO2IOError: If the device cannot be initialized
        """
        if self._device is not None:
            return self._device

        try:
            from atlas_i2c import atlas_i2c
        except ImportError as e:
            raise AtlasO2ImportError(
                f"O2 sensor requires the atlas_i2c library: {e}. "
                "Install with: pip install atlas-i2c"
            ) from e

        try:
            device = atlas_i2c.AtlasI2C()
            device.set_i2c_address(self.i2c_addr)
        except Exception as e:
            raise AtlasO2IOError(
                f"Failed to initialize Atlas O2 device at 0x{self.i2c_addr:02X}: {e}"
            ) from e

        self._device = device
        return self._device

    def reset(self):
        """Close the device file so the next command opens a fresh one."""
        device, self._device = self._device, None
        if device is not None:
            try:
                device.close()
            except Exception:
                pass  # Already unusable — reopening is what matters

    def command(self, cmd: str, processing_delay: Optional[int] = None,
                debug: bool = False, retries: Optional[int] = None) -> str:
        """
        Send a raw EZO command and return the response text.

        The command is written, the processing delay is waited out, and the
        response is read back as separate transfers (rather than one query call)
        so a "still processing" answer can be re-read without re-sending the
        command. I2C errors are retried on a freshly opened device file.

        Args:
            cmd: EZO command string, e.g. "R", "Cal,20.95", "Cal,?"
            processing_delay: Time in ms to wait for the response
                (defaults to this instance's processing_delay)
            debug: If True, print the status code and raw response
            retries: Extra attempts after an I2C error (defaults to self.retries)

        Returns:
            The response text with the status byte stripped (may be empty for
            commands that only acknowledge, e.g. "Cal,20.95")

        Raises:
            AtlasO2ImportError: If the atlas_i2c library is unavailable
            AtlasO2IOError: If every attempt fails or the sensor reports an error
        """
        delay = self.processing_delay if processing_delay is None else processing_delay
        attempts = (self.retries if retries is None else retries) + 1
        last_error = None

        for attempt in range(attempts):
            if attempt:
                # The bus usually clears within a second or so; reopen and retry
                time.sleep(RETRY_BACKOFF_S)
                self.reset()

            device = self.connect()

            try:
                device.write(cmd)
                time.sleep(delay / 1000.0)
                result = device.read(cmd)

                status = getattr(result, "status_code", None)
                if status == STATUS_NOT_READY:
                    # Give the sensor a moment and read again, same command
                    time.sleep(NOT_READY_WAIT_S)
                    result = device.read(cmd)
                    status = getattr(result, "status_code", None)
            except OSError as e:
                # Bus-level failure (timeout, remote I/O) — worth retrying
                last_error = AtlasO2IOError(
                    f"Atlas O2 command {cmd!r} failed at 0x{self.i2c_addr:02X}: {e}")
                if debug:
                    print(f"Atlas O2 {cmd!r} attempt {attempt + 1}/{attempts}: {e}")
                continue
            except Exception as e:
                raise AtlasO2IOError(
                    f"Atlas O2 command {cmd!r} failed at 0x{self.i2c_addr:02X}: {e}"
                ) from e

            data = getattr(result, "data", b"")
            try:
                text = data.decode(errors="ignore").strip()
            except AttributeError:
                # Fallback if data is already str
                text = str(data).strip()

            if debug:
                print(f"Atlas O2 {cmd!r} -> status {status}, data {text!r} "
                      f"(attempt {attempt + 1}/{attempts})")

            if status is None:
                last_error = AtlasO2IOError(
                    f"Atlas O2 sensor returned no data for {cmd!r}")
                continue
            if status != STATUS_SUCCESS:
                raise AtlasO2IOError(
                    f"Atlas O2 command {cmd!r}: {STATUS_MESSAGES.get(status, 'unknown')} "
                    f"(status {status})"
                )

            return text

        raise last_error

    def read_o2(self, debug: bool = False) -> float:
        """
        Read O2 concentration from the sensor.

        Args:
            debug: If True, print the raw response text

        Returns:
            O2 concentration in percent

        Raises:
            AtlasO2ImportError: If the atlas_i2c library is unavailable
            AtlasO2IOError: If communication fails or the response is unusable
        """
        text = self.command("R", debug=debug)

        if not text:
            raise AtlasO2IOError("Atlas O2 sensor returned empty data")

        # Strip units if present (e.g. '20.9 %') and convert to float
        first_token = text.split()[0]
        try:
            return float(first_token)
        except ValueError as e:
            raise AtlasO2IOError(
                f"Could not parse O2 value from response {text!r}"
            ) from e


def read_o2(i2c_addr: int = DEFAULT_I2C_ADDRESS, debug: bool = False) -> float:
    """
    Read O2 concentration from an Atlas Scientific sensor (functional interface).

    Args:
        i2c_addr: I2C address of the sensor (default: 0x6C)
        debug: If True, print the raw response text

    Returns:
        O2 concentration in percent

    Raises:
        AtlasO2Error: If sensor communication fails
    """
    return AtlasO2(i2c_addr=i2c_addr).read_o2(debug=debug)


if __name__ == "__main__":
    import sys

    i2c_addr = DEFAULT_I2C_ADDRESS
    if len(sys.argv) > 1:
        try:
            i2c_addr = int(sys.argv[1], 16) if sys.argv[1].lower().startswith("0x") \
                else int(sys.argv[1])
        except ValueError:
            print(f"Invalid address: {sys.argv[1]}, using default: 0x{i2c_addr:02X}")

    print(f"Reading O2 from Atlas sensor at 0x{i2c_addr:02X}...")
    try:
        print(f"O2 concentration: {read_o2(i2c_addr=i2c_addr, debug=True)} %")
    except AtlasO2Error as e:
        print(f"Error: {e}")
