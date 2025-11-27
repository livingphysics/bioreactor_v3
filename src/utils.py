"""
Utility functions for bioreactor operations. These are not intended to be used directly by the user, but rather to be used by the bioreactor class.
These functions are designed to be used with bioreactor.run() for scheduled tasks.
"""

import time
import logging
from typing import Union
from collections import deque
import matplotlib.pyplot as plt

logger = logging.getLogger("Bioreactor.Utils")


def set_peltier_power(bioreactor, duty_cycle: Union[int, float], forward: Union[bool, str] = True) -> bool:
    """
    Set the PWM duty cycle and direction for the peltier driver.
    
    Args:
        bioreactor: Bioreactor instance
        duty_cycle: Target duty percentage (0-100)
        forward: Direction flag or descriptive string ('heat', 'cool', etc.)
    """
    driver = getattr(bioreactor, 'peltier_driver', None)
    if not bioreactor.is_component_initialized('peltier_driver') or driver is None:
        bioreactor.logger.warning("Peltier driver not initialized; skipping command.")
        return False

    if isinstance(forward, str):
        fwd = forward.lower()
        if fwd in ('forward', 'cool', 'cold'):
            forward_bool = True
        elif fwd in ('reverse', 'heat', 'warm', 'hot'):
            forward_bool = False
        else:
            # Fallback: interpret truthy string as True
            forward_bool = fwd in ('true', '1', 'on')
    else:
        forward_bool = bool(forward)

    return driver.set(duty_cycle, forward=forward_bool)


def stop_peltier(bioreactor) -> None:
    """
    Stop PWM output on the peltier driver, if available.
    
    Args:
        bioreactor: Bioreactor instance
    """
    driver = getattr(bioreactor, 'peltier_driver', None)
    if not driver:
        return
    try:
        driver.stop()
    except Exception as e:
        bioreactor.logger.error(f"Failed to stop peltier driver: {e}")


def set_stirrer_speed(bioreactor, duty_cycle: Union[int, float]) -> bool:
    """
    Set stirrer PWM duty cycle.
    
    Args:
        bioreactor: Bioreactor instance
        duty_cycle: Target duty (0-100)
    """
    driver = getattr(bioreactor, 'stirrer_driver', None)
    if not bioreactor.is_component_initialized('stirrer') or driver is None:
        bioreactor.logger.warning("Stirrer driver not initialized; skipping command.")
        return False
    return driver.set_speed(duty_cycle)


def stop_stirrer(bioreactor) -> None:
    """Stop stirrer PWM output."""
    driver = getattr(bioreactor, 'stirrer_driver', None)
    if not driver:
        return
    try:
        driver.stop()
    except Exception as e:
        bioreactor.logger.error(f"Failed to stop stirrer: {e}")

