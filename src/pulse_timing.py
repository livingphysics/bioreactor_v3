"""Short software pulse waits shared by manual dosing and the CO2 worker.

This reduces thread/scheduler overhead; it is not a hardware timing guarantee.
Always measure completed writes and validate gas delivery on the actual rig.
"""
import time


SHORT_PULSE_S = 0.010


def wait_pulse(seconds, stop, *, clock=time.monotonic):
    """Wait with cancellation; spin only the final part of a short pulse.

    A caller must place OFF in its finally block. All potentially blocking
    preparation must happen before ON. Longer pulses retain Event.wait.
    """
    if seconds > SHORT_PULSE_S:
        return stop.wait(seconds)
    deadline = clock() + seconds
    if seconds > 0.001:
        if stop.wait(seconds - 0.00025):
            return True
    while clock() < deadline:
        if stop.is_set():
            return True
    return stop.is_set()
