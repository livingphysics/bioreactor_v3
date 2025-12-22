import lgpio

RELAY_PINS = {
    'relay1': 6,
    'relay2': 13,
    'relay3': 19,
    'relay4': 26
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

# Initialize all relay pins as outputs and set to HIGH (OFF)
# Note: 0 = relay ON, 1 = relay OFF (inverted logic)
if gpio_chip is not None:
    for physical_pin in RELAY_PINS.values():
        lgpio.gpio_claim_output(gpio_chip, physical_pin, 1)  # 1 = OFF

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
    
    # Convert state to boolean
    if isinstance(state, str):
        state = state.lower() in ['on', 'true', '1']
    else:
        state = bool(state)
    
    try:
        # Inverted logic: 0 = ON, 1 = OFF
        lgpio.gpio_write(gpio_chip, physical_pin, 0 if state else 1)
        print(f"{relay_name} (Physical Pin {physical_pin}): {'ON' if state else 'OFF'}")
        return True
    except Exception as e:
        print(f"Error actuating {relay_name}: {e}")
        return False

def actuate_relay_by_pin(physical_pin, state):
    """
    Actuate a relay by its GPIO pin number.
    
    Parameters:
    -----------
    physical_pin : int
        GPIO pin number (6, 13, 19, or 26)
    state : bool or str
        True/'on'/1 to turn relay ON, False/'off'/0 to turn relay OFF
    
    Returns:
    --------
    bool : True if successful, False otherwise
    
    Example:
    --------
    actuate_relay_by_pin(6, True)   # Turn relay on pin 6 ON
    actuate_relay_by_pin(13, False)  # Turn relay on pin 13 OFF
    """
    if gpio_chip is None:
        print("Error: GPIO chip not initialized")
        return False
    
    if physical_pin not in RELAY_PINS.values():
        print(f"Error: Invalid pin number {physical_pin}. Valid pins: {list(RELAY_PINS.values())}")
        return False
        
    # Convert state to boolean
    if isinstance(state, str):
        state = state.lower() in ['on', 'true', '1']
    else:
        state = bool(state)
    
    try:
        # Inverted logic: 0 = ON, 1 = OFF
        lgpio.gpio_write(gpio_chip, physical_pin, 0 if state else 1)
        relay_name = [name for name, p in RELAY_PINS.items() if p == physical_pin][0]
        print(f"{relay_name} (Physical Pin {physical_pin}): {'ON' if state else 'OFF'}")
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
        try:
            # Inverted logic: GPIO reads 0 = relay ON (True), 1 = relay OFF (False)
            gpio_value = lgpio.gpio_read(gpio_chip, physical_pin)
            states[relay_name] = not bool(gpio_value)  # Invert: 0 -> True (ON), 1 -> False (OFF)
        except Exception as e:
            print(f"Error reading {relay_name}: {e}")
            states[relay_name] = None
    return states

def is_gpio_initialized():
    """
    Check if GPIO chip is initialized.
    
    Returns:
    --------
    bool : True if GPIO is initialized, False otherwise
    """
    return gpio_chip is not None

def cleanup_gpio():
    """
    Cleanup GPIO settings. Call this before exiting the program.
    """
    global gpio_chip
    if gpio_chip is not None:
        # Turn off all relays before cleanup (write 1 = OFF)
        for physical_pin in RELAY_PINS.values():
            try:
                lgpio.gpio_write(gpio_chip, physical_pin, 1)  # 1 = OFF
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
        actuate_relay_by_pin(6, False)
        
        # Turn all on
        print("\nTurning all relays ON:")
        actuate_all_relays(True)
        
        # Turn all off
        print("\nTurning all relays OFF:")
        actuate_all_relays(False)
        
    finally:
        # Cleanup
        cleanup_gpio()
