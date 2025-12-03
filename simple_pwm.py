import lgpio
import time
import board
import busio
from adafruit_ads1x15 import ADS1115, AnalogIn, ads1x15

i2c = busio.I2C(board.SCL, board.SDA)

# Initialize ADS1115 ADC
ads = ADS1115(i2c)
# Create single-ended input on channel 0
adc_channel = AnalogIn(ads, ads1x15.Pin.A0)

pwm_pin = 16
frequency = 500
default_duty = 0.0

gpio_chip = lgpio.gpiochip_open(4)
lgpio.gpio_claim_output(gpio_chip, pwm_pin, 0)
lgpio.tx_pwm(gpio_chip, pwm_pin, frequency, 0)
for duty in [5, 10, 15, 20, 25, 30]:
	lgpio.tx_pwm(gpio_chip, pwm_pin, frequency, duty)
	time.sleep(5)
	# Read ADC value
	adc_value = adc_channel.value
	adc_voltage = adc_channel.voltage
	print(f"PWM Duty: {duty}%, ADC Raw: {adc_value}, ADC Voltage: {adc_voltage:.3f}V")
lgpio.tx_pwm(gpio_chip, pwm_pin, frequency, 0)

