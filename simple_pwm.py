import lgpio
import time
import board
import busio

i2c = busio.I2C(board.SCL, board.SDA)

pwm_pin = 16
frequency = 50
default_duty = 0.0

gpio_chip = lgpio.gpiochip_open(4)
lgpio.gpio_claim_output(gpio_chip, pwm_pin, 0)
lgpio.tx_pwm(gpio_chip, pwm_pin, frequency, 0)
for duty in [10, 20, 30, 40, 50]:
	lgpio.tx_pwm(gpio_chip, pwm_pin, frequency, duty)
	time.sleep(5)
lgpio.tx_pwm(gpio_chip, pwm_pin, frequency, 0)

