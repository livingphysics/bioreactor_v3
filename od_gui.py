"""
Manual OD Reading GUI

A simple GUI to take manual optical density (OD) voltage readings.
When the button is pressed:
1. IR LED turns on for 1 second
2. Readings are averaged over 0.5 seconds
3. Results are displayed on the GUI
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os
import time
import threading

# Add src directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import Bioreactor, Config
from src.io import measure_od, set_led, set_stirrer_speed, stop_stirrer


class ODManualReadingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Manual OD Reading")
        self.root.geometry("500x400")
        
        # Initialize bioreactor
        self.bioreactor = None
        self.initialized = False
        
        # Create widgets
        self.create_widgets()
        
        # Initialize bioreactor components
        self.init_bioreactor()
    
    def create_widgets(self):
        # Title
        title_label = tk.Label(self.root, text="Optical Density Manual Reading", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Status label
        self.status_label = tk.Label(self.root, text="Initializing...", 
                                     fg="orange", font=("Arial", 10))
        self.status_label.pack(pady=5)
        
        # Button frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)
        
        # Take reading button
        self.read_button = tk.Button(button_frame, text="Take OD Reading", 
                                     command=self.take_reading,
                                     font=("Arial", 12, "bold"),
                                     bg="#4CAF50", fg="white",
                                     width=20, height=2,
                                     state="disabled")
        self.read_button.pack(pady=10)
        
        # Results frame
        results_frame = tk.LabelFrame(self.root, text="OD Voltage Readings", 
                                      font=("Arial", 12, "bold"), padx=10, pady=10)
        results_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        # Results labels
        self.result_labels = {}
        channels = ['Trx', 'Sct', 'Ref']
        
        for i, channel in enumerate(channels):
            frame = tk.Frame(results_frame)
            frame.pack(fill="x", pady=5)
            
            label = tk.Label(frame, text=f"{channel}:", width=10, anchor='w', 
                            font=("Arial", 11))
            label.pack(side='left', padx=10)
            
            value_label = tk.Label(frame, text="--- V", width=15, anchor='w',
                                  font=("Arial", 11, "bold"), fg="blue")
            value_label.pack(side='left', padx=10)
            
            self.result_labels[channel] = value_label
        
        # Info label
        info_label = tk.Label(self.root, 
                             text="Click 'Take OD Reading' to measure voltages.\n"
                                  "Stirrer will turn on to 15%, LED will turn on for 1s,\n"
                                  "then readings averaged over 0.5s.",
                             font=("Arial", 9), fg="gray")
        info_label.pack(pady=10)
    
    def init_bioreactor(self):
        """Initialize bioreactor components in a separate thread"""
        def init_thread():
            try:
                config = Config()
                config.INIT_COMPONENTS = {
                    'relays': False,
                    'co2_sensor': False,
                    'co2_sensor_2': False,
                    'o2_sensor': False,
                    'i2c': True,  # Needed for OD sensor
                    'temp_sensor': False,
                    'peltier_driver': False,
                    'stirrer': True,  # Enable stirrer
                    'led': True,  # Enable LED
                    'optical_density': True,  # Enable OD sensor
                }
                
                self.bioreactor = Bioreactor(config)
                self.initialized = True
                
                # Update UI on main thread
                self.root.after(0, self.update_status_ready)
            except Exception as e:
                error_msg = f"Failed to initialize bioreactor: {e}"
                print(error_msg)
                self.root.after(0, lambda: self.update_status_error(error_msg))
        
        threading.Thread(target=init_thread, daemon=True).start()
    
    def update_status_ready(self):
        """Update UI when initialization is complete"""
        self.status_label.config(text="Ready", fg="green")
        self.read_button.config(state="normal")
    
    def update_status_error(self, error_msg):
        """Update UI when initialization fails"""
        self.status_label.config(text=f"Error: {error_msg}", fg="red")
        messagebox.showerror("Initialization Error", error_msg)
    
    def take_reading(self):
        """Take OD reading when button is pressed"""
        if not self.initialized or self.bioreactor is None:
            messagebox.showerror("Error", "Bioreactor not initialized")
            return
        
        # Disable button during reading
        self.read_button.config(state="disabled", text="Reading...")
        self.status_label.config(text="Taking reading...", fg="orange")
        
        # Clear previous results
        for channel in self.result_labels:
            self.result_labels[channel].config(text="--- V", fg="blue")
        
        # Run measurement in separate thread to avoid blocking UI
        def measurement_thread():
            try:
                # Turn on stirrer to 15% before taking reading
                if self.bioreactor.is_component_initialized('stirrer'):
                    set_stirrer_speed(self.bioreactor, 15.0)
                
                # Measure OD: LED on for 1s, then average over 0.5s
                # This matches the measure_od function behavior
                od_results = measure_od(self.bioreactor, 
                                       led_power=15.0,  # LED power level
                                       averaging_duration=0.5,  # Average over 0.5 seconds
                                       channel_name='all')
                
                # Update UI on main thread
                if od_results:
                    self.root.after(0, lambda: self.update_results(od_results))
                else:
                    self.root.after(0, lambda: self.update_status_error("Failed to get OD readings"))
            except Exception as e:
                error_msg = f"Error taking reading: {e}"
                print(error_msg)
                self.root.after(0, lambda: self.update_status_error(error_msg))
            finally:
                # Re-enable button
                self.root.after(0, self.enable_button)
        
        threading.Thread(target=measurement_thread, daemon=True).start()
    
    def update_results(self, od_results):
        """Update the GUI with OD reading results"""
        for channel, value in od_results.items():
            if channel in self.result_labels:
                if value is not None:
                    self.result_labels[channel].config(
                        text=f"{value:.4f} V", 
                        fg="green"
                    )
                else:
                    self.result_labels[channel].config(
                        text="Error", 
                        fg="red"
                    )
        
        self.status_label.config(text="Reading complete", fg="green")
    
    def enable_button(self):
        """Re-enable the read button"""
        self.read_button.config(state="normal", text="Take OD Reading")
    
    def on_closing(self):
        """Cleanup when window is closed"""
        if self.bioreactor:
            try:
                # Turn off LED if it's on
                if self.bioreactor.is_component_initialized('led'):
                    self.bioreactor.led_driver.off()
                # Stop stirrer
                if self.bioreactor.is_component_initialized('stirrer'):
                    stop_stirrer(self.bioreactor)
                self.bioreactor.finish()
            except:
                pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ODManualReadingGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
