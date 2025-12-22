import lgpio
import time

pin_ids = [26, 19, 13, 6]
# Open GPIO chip and set pin 29 (GPIO 5) as output
chip = lgpio.gpiochip_open(4)

for pin_id in pin_ids:
    lgpio.gpio_claim_output(chip, pin_id, 0)
    def relay(state):
        """Turn relay on (True/1) or off (False/0)"""
        lgpio.gpio_write(chip, pin_id, 1 if state else 0)

    # Example usage:
    relay(False)
    time.sleep(1)
    relay(True)

