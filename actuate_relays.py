import lgpio

# GPIO Setup for Relays
# Physical pin to GPIO (BCM) mapping for Raspberry Pi 5
PHYSICAL_TO_BCM = {
    29: 5,   # Physical pin 29 -> GPIO 5
    31: 6,   # Physical pin 31 -> GPIO 6
    33: 13,  # Physical pin 33 -> GPIO 13
    37: 26   # Physical pin 37 -> GPIO 26
}

RELAY_PINS = {
    'relay1': 29,
    'relay2': 31,
    'relay3': 33,
    'relay4': 37
}

# Initialize GPIO chip
try:
    gpio_chip = lgpio.gpiochip_open(4)  # Raspberry Pi 5 uses gpiochip4
except Exception as e:
    print(f"Error opening GPIO chip: {e}")
    print("Trying gpiochip0 as fallback...")
    try:
        gpio_chip = lgpio.gpiochip_open(0)  # Fallback for older Pi models
    except Exception as e2:
        print(f"Error opening GPIO chip 0: {e2}")
        gpio_chip = None

# Initialize all relay pins as outputs and set to LOW (OFF)
if gpio_chip is not None:
    for physical_pin in RELAY_PINS.values():
        bcm_pin = PHYSICAL_TO_BCM[physical_pin]
        lgpio.gpio_claim_output(gpio_chip, bcm_pin, 0)  # 0 = LOW/OFF

def actuate_relay(relay_name, state):
    """
    Actuate a specific relay on or off.
    
    Parameters:
    -----------
    relay_name : str
        Name of the relay ('relay1', 'relay2', 'relay3', or 'relay4')
    state : bool or str
        True/'on'/1 to turn relay ON, False/'off'/0 to turn relay OFF
    
    Returns:
    --------
    bool : True if successful, False otherwise
    
    Example:
    --------
    actuate_relay('relay1', True)   # Turn relay 1 ON
    actuate_relay('relay2', 'off')  # Turn relay 2 OFF
    """
    if gpio_chip is None:
        print("Error: GPIO chip not initialized")
        return False
    
    if relay_name not in RELAY_PINS:
        print(f"Error: Invalid relay name '{relay_name}'. Valid names: {list(RELAY_PINS.keys())}")
        return False
    
    physical_pin = RELAY_PINS[relay_name]
    bcm_pin = PHYSICAL_TO_BCM[physical_pin]
    
    # Convert state to boolean
    if isinstance(state, str):
        state = state.lower() in ['on', 'true', '1']
    else:
        state = bool(state)
    
    try:
        lgpio.gpio_write(gpio_chip, bcm_pin, 1 if state else 0)
        print(f"{relay_name} (Physical Pin {physical_pin}, GPIO {bcm_pin}): {'ON' if state else 'OFF'}")
        return True
    except Exception as e:
        print(f"Error actuating {relay_name}: {e}")
        return False

def actuate_relay_by_pin(physical_pin, state):
    """
    Actuate a relay by its physical pin number.
    
    Parameters:
    -----------
    physical_pin : int
        Physical pin number (29, 31, 33, or 37)
    state : bool or str
        True/'on'/1 to turn relay ON, False/'off'/0 to turn relay OFF
    
    Returns:
    --------
    bool : True if successful, False otherwise
    
    Example:
    --------
    actuate_relay_by_pin(29, True)   # Turn relay on pin 29 ON
    actuate_relay_by_pin(31, False)  # Turn relay on pin 31 OFF
    """
    if gpio_chip is None:
        print("Error: GPIO chip not initialized")
        return False
    
    if physical_pin not in RELAY_PINS.values():
        print(f"Error: Invalid pin number {physical_pin}. Valid pins: {list(RELAY_PINS.values())}")
        return False
    
    bcm_pin = PHYSICAL_TO_BCM[physical_pin]
    
    # Convert state to boolean
    if isinstance(state, str):
        state = state.lower() in ['on', 'true', '1']
    else:
        state = bool(state)
    
    try:
        lgpio.gpio_write(gpio_chip, bcm_pin, 1 if state else 0)
        relay_name = [name for name, p in RELAY_PINS.items() if p == physical_pin][0]
        print(f"{relay_name} (Physical Pin {physical_pin}, GPIO {bcm_pin}): {'ON' if state else 'OFF'}")
        return True
    except Exception as e:
        print(f"Error actuating relay on pin {physical_pin}: {e}")
        return False

def actuate_all_relays(state):
    """
    Actuate all relays at once.
    
    Parameters:
    -----------
    state : bool or str
        True/'on'/1 to turn all relays ON, False/'off'/0 to turn all relays OFF
    
    Example:
    --------
    actuate_all_relays(True)   # Turn all relays ON
    actuate_all_relays(False)  # Turn all relays OFF
    """
    for relay_name in RELAY_PINS.keys():
        actuate_relay(relay_name, state)

def get_relay_states():
    """
    Get the current state of all relays.
    
    Returns:
    --------
    dict : Dictionary with relay names as keys and their states (True/False) as values
    """
    if gpio_chip is None:
        print("Error: GPIO chip not initialized")
        return {}
    
    states = {}
    for relay_name, physical_pin in RELAY_PINS.items():
        bcm_pin = PHYSICAL_TO_BCM[physical_pin]
        try:
            states[relay_name] = bool(lgpio.gpio_read(gpio_chip, bcm_pin))
        except Exception as e:
            print(f"Error reading {relay_name}: {e}")
            states[relay_name] = None
    return states

def cleanup_gpio():
    """
    Cleanup GPIO settings. Call this before exiting the program.
    """
    global gpio_chip
    if gpio_chip is not None:
        # Turn off all relays before cleanup
        for physical_pin in RELAY_PINS.values():
            bcm_pin = PHYSICAL_TO_BCM[physical_pin]
            try:
                lgpio.gpio_write(gpio_chip, bcm_pin, 0)
            except:
                pass
        
        lgpio.gpiochip_close(gpio_chip)
        gpio_chip = None
        print("GPIO cleanup completed")

if __name__ == "__main__":
    # Example usage
    print("Relay Control Example (Raspberry Pi 5 Compatible)")
    print("=" * 50)
    
    if gpio_chip is None:
        print("Failed to initialize GPIO. Exiting.")
        exit(1)
    
    try:
        # Turn on individual relays
        actuate_relay('relay1', True)
        actuate_relay('relay2', True)
        
        # Check states
        print("\nCurrent relay states:")
        print(get_relay_states())
        
        # Turn off by pin number
        actuate_relay_by_pin(29, False)
        
        # Turn all on
        print("\nTurning all relays ON:")
        actuate_all_relays(True)
        
        # Turn all off
        print("\nTurning all relays OFF:")
        actuate_all_relays(False)
        
    finally:
        # Cleanup
        cleanup_gpio()
