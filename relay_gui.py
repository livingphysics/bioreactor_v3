import tkinter as tk
from tkinter import messagebox
import sys
import os
import threading

# Add src directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from actuate_relays import actuate_relay, get_relay_states, cleanup_gpio, is_gpio_initialized
from src import Bioreactor, Config
from src.io import set_stirrer_speed, stop_stirrer

class RelayGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Relay Control")
        self.root.geometry("400x600")
        
        # Verify GPIO is initialized
        if not is_gpio_initialized():
            messagebox.showerror("GPIO Error", 
                "GPIO chip not initialized!\n\nCheck:\n- lgpio is installed\n- Running with proper permissions\n- GPIO pins are available")
            root.destroy()
            return
        
        self.relays = ['relay1', 'relay2', 'relay3', 'relay4']
        self.buttons = {}
        self.state_labels = {}
        
        # Bioreactor for stirrer control
        self.bioreactor = None
        self.stirrer_initialized = False
        self.current_duty = 0.0
        
        self.create_widgets()
        self.update_states()
        self.init_stirrer()
    
    def create_widgets(self):
        # Relay control section
        relay_frame = tk.LabelFrame(self.root, text="Relay Control", font=("Arial", 12, "bold"), padx=10, pady=10)
        relay_frame.pack(pady=10, padx=10, fill='x')
        
        for i, relay in enumerate(self.relays):
            frame = tk.Frame(relay_frame)
            frame.pack(pady=5, fill='x')
            
            # Relay label
            label = tk.Label(frame, text=relay.upper(), width=10, anchor='w')
            label.pack(side='left')
            
            # State label
            state_label = tk.Label(frame, text="OFF", width=8, bg='red')
            state_label.pack(side='left', padx=5)
            self.state_labels[relay] = state_label
            
            # Buttons
            btn_frame = tk.Frame(frame)
            btn_frame.pack(side='left')
            
            tk.Button(btn_frame, text="ON", width=6, 
                     command=lambda r=relay: self.set_relay(r, True)).pack(side='left', padx=2)
            tk.Button(btn_frame, text="OFF", width=6,
                     command=lambda r=relay: self.set_relay(r, False)).pack(side='left', padx=2)
            tk.Button(btn_frame, text="TOGGLE", width=8,
                     command=lambda r=relay: self.toggle_relay(r)).pack(side='left', padx=2)
        
        # Separator
        separator = tk.Frame(self.root, height=2, bg='gray')
        separator.pack(fill='x', padx=10, pady=10)
        
        # Stirrer PWM control section
        stirrer_frame = tk.LabelFrame(self.root, text="Stirrer PWM Control", font=("Arial", 12, "bold"), padx=10, pady=10)
        stirrer_frame.pack(pady=10, padx=10, fill='x')
        
        # Status label
        self.stirrer_status_label = tk.Label(stirrer_frame, text="Initializing...", fg="orange", font=("Arial", 10))
        self.stirrer_status_label.pack(pady=5)
        
        # Duty cycle display
        duty_display_frame = tk.Frame(stirrer_frame)
        duty_display_frame.pack(pady=5)
        
        tk.Label(duty_display_frame, text="Duty Cycle:", font=("Arial", 10)).pack(side='left', padx=5)
        self.duty_label = tk.Label(duty_display_frame, text="0.0%", font=("Arial", 10, "bold"), width=10)
        self.duty_label.pack(side='left', padx=5)
        
        # PWM slider
        slider_frame = tk.Frame(stirrer_frame)
        slider_frame.pack(pady=10, fill='x')
        
        tk.Label(slider_frame, text="0%", font=("Arial", 9)).pack(side='left')
        self.duty_scale = tk.Scale(slider_frame, from_=0, to=100, orient='horizontal', 
                                   length=250, resolution=0.1, command=self.on_scale_change)
        self.duty_scale.set(0)
        self.duty_scale.pack(side='left', padx=5, fill='x', expand=True)
        tk.Label(slider_frame, text="100%", font=("Arial", 9)).pack(side='left')
        
        # Direct input frame
        input_frame = tk.Frame(stirrer_frame)
        input_frame.pack(pady=5)
        
        tk.Label(input_frame, text="Set Duty:", font=("Arial", 10)).pack(side='left', padx=5)
        self.duty_entry = tk.Entry(input_frame, width=8)
        self.duty_entry.pack(side='left', padx=5)
        self.duty_entry.insert(0, "0.0")
        
        tk.Label(input_frame, text="%", font=("Arial", 10)).pack(side='left', padx=2)
        tk.Button(input_frame, text="Set", width=6, command=self.set_duty_from_entry).pack(side='left', padx=5)
        
        # Control buttons
        button_frame = tk.Frame(stirrer_frame)
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="Stop (0%)", width=10, command=self.stop_stirrer).pack(side='left', padx=5)
        tk.Button(button_frame, text="20%", width=8, command=lambda: self.set_duty(20.0)).pack(side='left', padx=5)
        tk.Button(button_frame, text="50%", width=8, command=lambda: self.set_duty(50.0)).pack(side='left', padx=5)
        tk.Button(button_frame, text="100%", width=8, command=lambda: self.set_duty(100.0)).pack(side='left', padx=5)
    
    def set_relay(self, relay, state):
        actuate_relay(relay, state)
        self.update_states()
    
    def toggle_relay(self, relay):
        states = get_relay_states()
        current = states.get(relay, False)
        self.set_relay(relay, not current)
    
    def update_states(self):
        states = get_relay_states()
        for relay, label in self.state_labels.items():
            state = states.get(relay, False)
            label.config(text="ON" if state else "OFF", 
                        bg='green' if state else 'red')
    
    def init_stirrer(self):
        """Initialize stirrer in a separate thread"""
        def init_thread():
            try:
                config = Config()
                config.INIT_COMPONENTS = {
                    'relays': False,  # Already handled by actuate_relays
                    'co2_sensor': False,
                    'co2_sensor_2': False,
                    'o2_sensor': False,
                    'i2c': False,
                    'temp_sensor': False,
                    'peltier_driver': False,
                    'stirrer': True,  # Enable stirrer
                    'led': False,
                    'optical_density': False,
                }
                config.STIRRER_PWM_PIN: int = 21
                STIRRER_PWM_FREQ: int = 25000
                self.bioreactor = Bioreactor(config)
                self.stirrer_initialized = True
                
                # Update UI on main thread
                self.root.after(0, self.update_stirrer_status_ready)
            except Exception as e:
                error_msg = f"Failed to initialize stirrer: {e}"
                print(error_msg)
                self.root.after(0, lambda: self.update_stirrer_status_error(error_msg))
        
        threading.Thread(target=init_thread, daemon=True).start()
    
    def update_stirrer_status_ready(self):
        """Update UI when stirrer initialization is complete"""
        self.stirrer_status_label.config(text="Ready", fg="green")
        self.duty_scale.config(state="normal")
        self.duty_entry.config(state="normal")
    
    def update_stirrer_status_error(self, error_msg):
        """Update UI when stirrer initialization fails"""
        self.stirrer_status_label.config(text=f"Error: {error_msg[:50]}...", fg="red")
        self.duty_scale.config(state="disabled")
        self.duty_entry.config(state="disabled")
        messagebox.showerror("Stirrer Initialization Error", error_msg)
    
    def on_scale_change(self, value):
        """Called when slider is moved"""
        try:
            duty = float(value)
            self.set_duty(duty, update_scale=False)
        except ValueError:
            pass
    
    def set_duty_from_entry(self):
        """Set duty cycle from entry field"""
        try:
            duty = float(self.duty_entry.get())
            self.set_duty(duty)
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number between 0 and 100")
    
    def set_duty(self, duty, update_scale=True):
        """Set stirrer PWM duty cycle"""
        if not self.stirrer_initialized or self.bioreactor is None:
            messagebox.showwarning("Not Initialized", "Stirrer not initialized yet")
            return
        
        # Clamp duty cycle to 0-100
        duty = max(0.0, min(100.0, float(duty)))
        self.current_duty = duty
        
        # Update UI
        self.duty_label.config(text=f"{duty:.1f}%")
        if update_scale:
            self.duty_scale.set(duty)
        self.duty_entry.delete(0, tk.END)
        self.duty_entry.insert(0, f"{duty:.1f}")
        
        # Set stirrer speed
        success = set_stirrer_speed(self.bioreactor, duty)
        if not success:
            messagebox.showerror("Error", "Failed to set stirrer speed")
    
    def stop_stirrer(self):
        """Stop stirrer (set to 0%)"""
        if not self.stirrer_initialized or self.bioreactor is None:
            messagebox.showwarning("Not Initialized", "Stirrer not initialized yet")
            return
        
        self.set_duty(0.0)
        stop_stirrer(self.bioreactor)
    
    def on_closing(self):
        # Stop stirrer before closing
        if self.stirrer_initialized and self.bioreactor is not None:
            try:
                stop_stirrer(self.bioreactor)
            except:
                pass
        cleanup_gpio()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = RelayGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

