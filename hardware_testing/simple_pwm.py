import lgpio
import time
import board
import busio
from adafruit_ads1x15 import ADS1115, AnalogIn, ads1x15

i2c = busio.I2C(board.SCL, board.SDA)

# Initialize ADS1115 ADC
ads = ADS1115(i2c)
# Create single-ended inputs on channels A0, A1, and A2
adc_channels = {}
adc_pin_map = {
	'Trx': ads1x15.Pin.A0,
	'Ref': ads1x15.Pin.A1,
	'Sct': ads1x15.Pin.A2,
}
for channel_name, pin in adc_pin_map.items():
	adc_channels[channel_name] = AnalogIn(ads, pin)

pwm_pin = 25
frequency = 500
default_duty = 0.0

gpio_chip = lgpio.gpiochip_open(4)
lgpio.gpio_claim_output(gpio_chip, pwm_pin, 0)
lgpio.tx_pwm(gpio_chip, pwm_pin, frequency, 0)
for duty in [10, 20, 30, 40, 50]:
	lgpio.tx_pwm(gpio_chip, pwm_pin, frequency, duty)
	time.sleep(5)
	# Read ADC values from all channels
	print(f"PWM Duty: {duty}%")
	for channel_name, channel in adc_channels.items():
		adc_value = channel.value
		adc_voltage = channel.voltage
		print(f"  {channel_name} - Raw: {adc_value}, Voltage: {adc_voltage:.3f}V")
lgpio.tx_pwm(gpio_chip, pwm_pin, frequency, 0)

