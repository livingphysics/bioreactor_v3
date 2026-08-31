from eyespy_adc import read_eyespy_adc
import time
# Read with default settings (address 0x49, gain 1.0)
while True:
	reading = read_eyespy_adc()
	print(reading)
	time.sleep(1)

# Or customize the gain
reading = read_eyespy_adc(gain=2.0)  # ±2.048 V range

