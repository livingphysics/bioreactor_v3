"""
Manual OD Reading GUI

A simple GUI to take manual optical density (OD) voltage readings.
When the button is pressed:
1. IR LED turns on for 1 second
2. Readings are averaged over 0.5 seconds
3. Results are displayed on the GUI
"""

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import sys
import os
import time
import threading
import matplotlib.pyplot as plt
import numpy as np

# Add src directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import Bioreactor, Config
from src.io import measure_od, set_led, set_stirrer_speed, stop_stirrer


class ODManualReadingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Manual OD Reading")
        self.root.geometry("500x800")
        
        # Initialize bioreactor
        self.bioreactor = None
        self.initialized = False
        
        # Store last readings
        self.last_readings = {'Trx': None, 'Sct': None, 'Ref': None}
        
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
        
        # LED power control frame
        led_frame = tk.Frame(self.root)
        led_frame.pack(pady=10)
        
        led_label = tk.Label(led_frame, text="LED Power:", font=("Arial", 10))
        led_label.pack(side='left', padx=5)
        
        # Create dropdown for LED power (0-30%)
        self.led_power_var = tk.StringVar(value="15")
        led_power_values = [str(i) for i in range(0, 31)]  # 0 to 30 in 1% increments
        self.led_power_combo = ttk.Combobox(led_frame, textvariable=self.led_power_var,
                                            values=led_power_values, width=8, state="readonly")
        self.led_power_combo.pack(side='left', padx=5)
        
        percent_label = tk.Label(led_frame, text="%", font=("Arial", 10))
        percent_label.pack(side='left', padx=2)
        
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
        self.read_button.pack(pady=5)
        
        # LED sweep button
        self.sweep_button = tk.Button(button_frame, text="LED Power Sweep (0-20%)", 
                                      command=self.run_led_sweep,
                                      font=("Arial", 11, "bold"),
                                      bg="#2196F3", fg="white",
                                      width=20, height=2,
                                      state="disabled")
        self.sweep_button.pack(pady=5)
        
        # Results frame
        results_frame = tk.LabelFrame(self.root, text="OD Voltage Readings", 
                                      font=("Arial", 12, "bold"), padx=10, pady=10)
        results_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        # Last reading info label
        last_reading_frame = tk.Frame(results_frame)
        last_reading_frame.pack(fill="x", pady=5)
        
        self.last_reading_label = tk.Label(last_reading_frame, 
                                          text="Last Reading: ---", 
                                          font=("Arial", 10, "italic"), 
                                          fg="gray")
        self.last_reading_label.pack(anchor='w', padx=10)
        
        # Results labels
        self.result_labels = {}
        self.last_result_labels = {}
        channels = ['Trx', 'Sct', 'Ref']
        
        # Header row
        header_frame = tk.Frame(results_frame)
        header_frame.pack(fill="x", pady=5)
        tk.Label(header_frame, text="Channel", width=10, anchor='w', 
                font=("Arial", 10, "bold")).pack(side='left', padx=10)
        tk.Label(header_frame, text="Current", width=15, anchor='w',
                font=("Arial", 10, "bold")).pack(side='left', padx=10)
        tk.Label(header_frame, text="Last", width=15, anchor='w',
                font=("Arial", 10, "bold")).pack(side='left', padx=10)
        
        for i, channel in enumerate(channels):
            frame = tk.Frame(results_frame)
            frame.pack(fill="x", pady=5)
            
            label = tk.Label(frame, text=f"{channel}:", width=10, anchor='w', 
                            font=("Arial", 11))
            label.pack(side='left', padx=10)
            
            # Current reading
            value_label = tk.Label(frame, text="--- V", width=15, anchor='w',
                                  font=("Arial", 11, "bold"), fg="blue")
            value_label.pack(side='left', padx=10)
            self.result_labels[channel] = value_label
            
            # Last reading
            last_value_label = tk.Label(frame, text="--- V", width=15, anchor='w',
                                       font=("Arial", 11), fg="gray")
            last_value_label.pack(side='left', padx=10)
            self.last_result_labels[channel] = last_value_label
        
        # Info label
        info_label = tk.Label(self.root, 
                             text="Select LED power (0-30%), then click 'Take OD Reading'.\n"
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
        self.sweep_button.config(state="normal")
    
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
                
                # Get LED power from dropdown
                try:
                    led_power = float(self.led_power_var.get())
                except (ValueError, AttributeError):
                    led_power = 15.0  # Default if conversion fails
                
                # Measure OD: LED on for 1s, then average over 0.5s
                # This matches the measure_od function behavior
                od_results = measure_od(self.bioreactor, 
                                       led_power=led_power,  # LED power level from dropdown
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
        # Get LED power used for this reading
        try:
            led_power = float(self.led_power_var.get())
        except (ValueError, AttributeError):
            led_power = None
        
        # Update last reading label
        if led_power is not None:
            self.last_reading_label.config(
                text=f"Last Reading: LED Power = {led_power:.0f}%",
                fg="blue"
            )
        else:
            self.last_reading_label.config(
                text="Last Reading: LED Power = ---",
                fg="gray"
            )
        
        # Update current readings and move previous to last
        for channel, value in od_results.items():
            if channel in self.result_labels:
                # Move current reading to last reading
                if self.last_readings[channel] is not None:
                    self.last_result_labels[channel].config(
                        text=f"{self.last_readings[channel]:.4f} V",
                        fg="gray"
                    )
                else:
                    self.last_result_labels[channel].config(
                        text="--- V",
                        fg="gray"
                    )
                
                # Update current reading
                if value is not None:
                    self.result_labels[channel].config(
                        text=f"{value:.4f} V", 
                        fg="green"
                    )
                    # Store as last reading for next time
                    self.last_readings[channel] = value
                else:
                    self.result_labels[channel].config(
                        text="Error", 
                        fg="red"
                    )
        
        self.status_label.config(text="Reading complete", fg="green")
    
    def enable_button(self):
        """Re-enable the read button"""
        self.read_button.config(state="normal", text="Take OD Reading")
        self.sweep_button.config(state="normal")
    
    def run_led_sweep(self):
        """Run LED power sweep from 0% to 20% in 1% increments"""
        if not self.initialized or self.bioreactor is None:
            messagebox.showerror("Error", "Bioreactor not initialized")
            return
        
        # Disable buttons during sweep
        self.read_button.config(state="disabled")
        self.sweep_button.config(state="disabled", text="Sweeping...")
        self.status_label.config(text="Running LED power sweep...", fg="orange")
        
        # Clear previous results
        for channel in self.result_labels:
            self.result_labels[channel].config(text="--- V", fg="blue")
        
        # Run sweep in separate thread
        def sweep_thread():
            try:
                # Turn on stirrer to 15% before starting sweep
                if self.bioreactor.is_component_initialized('stirrer'):
                    set_stirrer_speed(self.bioreactor, 15.0)
                
                # Storage for results
                led_powers = []
                trx_values = []
                sct_values = []
                ref_values = []
                
                # Sweep from 0% to 20% in 1% increments
                for led_power in range(0, 21):
                    self.root.after(0, lambda p=led_power: self.status_label.config(
                        text=f"Sweeping... LED Power: {p}%", fg="orange"))
                    
                    # Measure OD at this LED power
                    od_results = measure_od(self.bioreactor,
                                          led_power=float(led_power),
                                          averaging_duration=0.5,
                                          channel_name='all')
                    
                    if od_results:
                        led_powers.append(led_power)
                        trx_values.append(od_results.get('Trx', None))
                        sct_values.append(od_results.get('Sct', None))
                        ref_values.append(od_results.get('Ref', None))
                    
                    # Small delay between measurements
                    time.sleep(0.2)
                
                # Update UI with final results
                if led_powers:
                    self.root.after(0, lambda: self.update_results_from_sweep(
                        led_powers, trx_values, sct_values, ref_values))
                    self.root.after(0, lambda: self.plot_sweep_results(
                        led_powers, trx_values, sct_values, ref_values))
                else:
                    self.root.after(0, lambda: self.update_status_error("No valid readings collected"))
                    
            except Exception as e:
                error_msg = f"Error during sweep: {e}"
                print(error_msg)
                self.root.after(0, lambda: self.update_status_error(error_msg))
            finally:
                # Re-enable buttons
                self.root.after(0, self.enable_sweep_button)
        
        threading.Thread(target=sweep_thread, daemon=True).start()
    
    def update_results_from_sweep(self, led_powers, trx_values, sct_values, ref_values):
        """Update GUI with results from the last sweep measurement"""
        if led_powers:
            last_idx = len(led_powers) - 1
            last_led_power = led_powers[last_idx]
            
            # Update last reading label
            self.last_reading_label.config(
                text=f"Last Reading: LED Power = {last_led_power:.0f}% (from sweep)",
                fg="blue"
            )
            
            # Update current readings and move previous to last
            if trx_values[last_idx] is not None:
                # Move current to last
                if self.last_readings['Trx'] is not None:
                    self.last_result_labels['Trx'].config(
                        text=f"{self.last_readings['Trx']:.4f} V",
                        fg="gray"
                    )
                # Update current
                self.result_labels['Trx'].config(text=f"{trx_values[last_idx]:.4f} V", fg="green")
                self.last_readings['Trx'] = trx_values[last_idx]
            
            if sct_values[last_idx] is not None:
                # Move current to last
                if self.last_readings['Sct'] is not None:
                    self.last_result_labels['Sct'].config(
                        text=f"{self.last_readings['Sct']:.4f} V",
                        fg="gray"
                    )
                # Update current
                self.result_labels['Sct'].config(text=f"{sct_values[last_idx]:.4f} V", fg="green")
                self.last_readings['Sct'] = sct_values[last_idx]
            
            if ref_values[last_idx] is not None:
                # Move current to last
                if self.last_readings['Ref'] is not None:
                    self.last_result_labels['Ref'].config(
                        text=f"{self.last_readings['Ref']:.4f} V",
                        fg="gray"
                    )
                # Update current
                self.result_labels['Ref'].config(text=f"{ref_values[last_idx]:.4f} V", fg="green")
                self.last_readings['Ref'] = ref_values[last_idx]
        
        self.status_label.config(text="Sweep complete", fg="green")
    
    def plot_sweep_results(self, led_powers, trx_values, sct_values, ref_values):
        """Plot sweep results in a pop-up window"""
        try:
            # Filter out None values for plotting
            valid_indices = [i for i in range(len(led_powers)) 
                           if trx_values[i] is not None and 
                              sct_values[i] is not None and 
                              ref_values[i] is not None]
            
            if not valid_indices:
                messagebox.showwarning("Plot Warning", "No valid data to plot")
                return
            
            plot_powers = [led_powers[i] for i in valid_indices]
            plot_trx = [trx_values[i] for i in valid_indices]
            plot_sct = [sct_values[i] for i in valid_indices]
            plot_ref = [ref_values[i] for i in valid_indices]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(plot_powers, plot_trx, 'o-', label='Trx', linewidth=2, markersize=6)
            ax.plot(plot_powers, plot_sct, 's-', label='Sct', linewidth=2, markersize=6)
            ax.plot(plot_powers, plot_ref, '^-', label='Ref', linewidth=2, markersize=6)
            
            ax.set_xlabel('LED Power (%)', fontsize=12)
            ax.set_ylabel('Voltage (V)', fontsize=12)
            ax.set_title('OD Voltage vs LED Power', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            error_msg = f"Error plotting results: {e}"
            print(error_msg)
            messagebox.showerror("Plot Error", error_msg)
    
    def enable_sweep_button(self):
        """Re-enable the sweep button"""
        self.read_button.config(state="normal")
        self.sweep_button.config(state="normal", text="LED Power Sweep (0-20%)")
    
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
