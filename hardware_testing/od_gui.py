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

# Add parent directory to path to allow imports from src
# od_gui.py is in hardware_testing/, so we need to go up one level to find src/
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

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
        
        # Store last readings (will be initialized after channels are known)
        self.last_readings = {}
        # Two-phase sweep state
        self.first_sweep_data = None
        self.awaiting_second_sweep = False
        
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
        
        # Two-sweep (swap vials) button
        self.swap_sweep_button = tk.Button(button_frame, text="Two-sweep (swap vials)", 
                                           command=self.start_two_phase_sweep,
                                           font=("Arial", 11, "bold"),
                                           bg="#9C27B0", fg="white",
                                           width=20, height=2,
                                           state="disabled")
        self.swap_sweep_button.pack(pady=5)
        
        # Get channel names from config (OD channels + eyespy boards)
        try:
            config = Config()
            od_channels = getattr(config, 'OD_ADC_CHANNELS', {})
            self.od_channels = list(od_channels.keys()) if od_channels else ['Trx', 'Sct', 'Ref']
            
            # Get eyespy board names
            eyespy_config = getattr(config, 'EYESPY_ADC', {})
            self.eyespy_boards = list(eyespy_config.keys()) if eyespy_config else []
            
            # Combine all channels for display
            self.channels = self.od_channels + self.eyespy_boards
        except:
            # Fallback to default channels
            self.od_channels = ['Trx', 'Sct', 'Ref']
            self.eyespy_boards = []
            self.channels = self.od_channels + self.eyespy_boards
        
        # Results frame
        frame_title = "OD & Eyespy Voltage Readings" if self.eyespy_boards else "OD Voltage Readings"
        results_frame = tk.LabelFrame(self.root, text=frame_title, 
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
        
        # Initialize last readings dictionary
        self.last_readings = {ch: None for ch in self.channels}
        
        # Initialize result labels dictionaries
        self.result_labels = {}
        self.last_result_labels = {}
        
        # Header row
        header_frame = tk.Frame(results_frame)
        header_frame.pack(fill="x", pady=5)
        tk.Label(header_frame, text="Channel", width=10, anchor='w', 
                font=("Arial", 10, "bold")).pack(side='left', padx=10)
        tk.Label(header_frame, text="Current", width=15, anchor='w',
                font=("Arial", 10, "bold")).pack(side='left', padx=10)
        tk.Label(header_frame, text="Last", width=15, anchor='w',
                font=("Arial", 10, "bold")).pack(side='left', padx=10)
        
        # Create channel rows (OD channels first, then eyespy)
        for channel in self.od_channels:
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
        
        # Add separator if both OD and eyespy are present
        if self.od_channels and self.eyespy_boards:
            separator = tk.Frame(results_frame, height=2, bg="gray")
            separator.pack(fill="x", pady=5)
            separator_label = tk.Label(results_frame, text="Eyespy Readings", 
                                       font=("Arial", 10, "bold"), fg="gray")
            separator_label.pack(pady=5)
        
        # Create eyespy board rows
        for board_name in self.eyespy_boards:
            frame = tk.Frame(results_frame)
            frame.pack(fill="x", pady=5)
            
            label = tk.Label(frame, text=f"{board_name}:", width=10, anchor='w', 
                            font=("Arial", 11))
            label.pack(side='left', padx=10)
            
            # Current reading
            value_label = tk.Label(frame, text="--- V", width=15, anchor='w',
                                  font=("Arial", 11, "bold"), fg="blue")
            value_label.pack(side='left', padx=10)
            self.result_labels[board_name] = value_label
            
            # Last reading
            last_value_label = tk.Label(frame, text="--- V", width=15, anchor='w',
                                       font=("Arial", 11), fg="gray")
            last_value_label.pack(side='left', padx=10)
            self.last_result_labels[board_name] = last_value_label
        
        # Info label
        info_text = "Select LED power (0-30%), then click 'Take OD Reading'.\n"
        info_text += "Stirrer will turn on to 15%, LED will turn on for 1s,\n"
        info_text += "then readings averaged over 0.5s."
        if self.eyespy_boards:
            info_text += f"\nEyespy boards ({', '.join(self.eyespy_boards)}) will also be read."
        info_label = tk.Label(self.root, text=info_text,
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
                    'i2c': True,  # Needed for OD sensor and eyespy
                    'temp_sensor': False,
                    'peltier_driver': False,
                    'stirrer': True,  # Enable stirrer
                    'led': True,  # Enable LED
                    'optical_density': True,  # Enable OD sensor
                    'eyespy_adc': True,  # Enable eyespy ADC boards
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
        self.swap_sweep_button.config(state="normal")
    
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
        self.swap_sweep_button.config(state="normal", text="Two-sweep (swap vials)")
    
    def _perform_led_sweep(self, led_range=range(0, 21)):
        """Perform an LED sweep and return collected data."""
        # Turn on stirrer to 15% before starting sweep
        if self.bioreactor.is_component_initialized('stirrer'):
            set_stirrer_speed(self.bioreactor, 15.0)
        
        led_powers = []
        # Initialize with OD channels and eyespy boards
        channel_values = {ch: [] for ch in self.channels}
        
        for led_power in led_range:
            self.root.after(0, lambda p=led_power: self.status_label.config(
                text=f"Sweeping... LED Power: {p}%", fg="orange"))
            
            od_results = measure_od(
                self.bioreactor,
                led_power=float(led_power),
                averaging_duration=0.5,
                channel_name='all'
            )
            
            if od_results:
                led_powers.append(led_power)
                # Collect readings for all channels (OD + eyespy)
                for ch in self.channels:
                    channel_values[ch].append(od_results.get(ch, None))
            
            time.sleep(0.2)
        
        return led_powers, channel_values
    
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
                led_powers, channel_values = self._perform_led_sweep()
                
                # Update UI with final results
                if led_powers:
                    self.root.after(0, lambda: self.update_results_from_sweep(
                        led_powers, channel_values))
                    self.root.after(0, lambda: self.plot_sweep_results(
                        led_powers, channel_values))
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
    
    def update_results_from_sweep(self, led_powers, channel_values):
        """Update GUI with results from the last sweep measurement"""
        if led_powers:
            last_idx = len(led_powers) - 1
            last_led_power = led_powers[last_idx]
            
            # Update last reading label
            self.last_reading_label.config(
                text=f"Last Reading: LED Power = {last_led_power:.0f}% (from sweep)",
                fg="blue"
            )
            
            # Update current readings and move previous to last for each channel
            for channel in self.channels:
                values = channel_values.get(channel, [])
                if len(values) <= last_idx:
                    continue
                value = values[last_idx]
                if value is not None:
                    if self.last_readings.get(channel) is not None:
                        self.last_result_labels[channel].config(
                            text=f"{self.last_readings[channel]:.4f} V",
                            fg="gray"
                        )
                    self.result_labels[channel].config(text=f"{value:.4f} V", fg="green")
                    self.last_readings[channel] = value
        
        self.status_label.config(text="Sweep complete", fg="green")
    
    def plot_sweep_results(self, led_powers, channel_values):
        """Plot sweep results in a pop-up window"""
        try:
            # Prepare per-channel filtered data
            plot_data = {}
            for channel in self.channels:
                vals = channel_values.get(channel, [])
                points = [(p, v) for p, v in zip(led_powers, vals) if v is not None]
                if points:
                    plot_data[channel] = points
            
            if not plot_data:
                messagebox.showwarning("Plot Warning", "No valid data to plot")
                return
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            markers = ['o', 's', '^', 'd', 'v', 'x']
            for idx, (channel, points) in enumerate(plot_data.items()):
                m = markers[idx % len(markers)]
                powers = [p for p, _ in points]
                values = [v for _, v in points]
                ax.plot(powers, values, f'{m}-', label=channel, linewidth=2, markersize=6)
            
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
    
    def start_two_phase_sweep(self):
        """Run two sweeps with a vial swap in between and plot the difference."""
        if not self.initialized or self.bioreactor is None:
            messagebox.showerror("Error", "Bioreactor not initialized")
            return
        
        if self.awaiting_second_sweep:
            self._run_second_sweep()
        else:
            self._run_first_sweep()
    
    def _run_first_sweep(self):
        """Execute the first sweep and prompt for vial swap."""
        # Disable other actions during the two-phase process
        self.read_button.config(state="disabled")
        self.sweep_button.config(state="disabled")
        self.swap_sweep_button.config(state="disabled", text="Running first sweep...")
        self.status_label.config(text="Running first sweep...", fg="orange")
        
        def first_sweep_thread():
            try:
                led_powers, channel_values = self._perform_led_sweep()
                if not led_powers:
                    self.root.after(0, lambda: self.update_status_error("No valid readings collected in first sweep"))
                    self.root.after(0, self.enable_sweep_button)
                    return
                
                # Store first sweep data and prompt user
                self.first_sweep_data = (led_powers, channel_values)
                self.awaiting_second_sweep = True
                self.root.after(0, lambda: self.status_label.config(
                    text="First sweep done. Swap vials, then run second sweep.", fg="blue"))
                self.root.after(0, lambda: self.swap_sweep_button.config(
                    state="normal", text="Run second sweep"))
            except Exception as e:
                error_msg = f"Error during first sweep: {e}"
                print(error_msg)
                self.awaiting_second_sweep = False
                self.root.after(0, lambda: self.update_status_error(error_msg))
                # Re-enable other buttons on failure
                self.root.after(0, self.enable_sweep_button)
            # Do not re-enable other buttons here; wait until second sweep completes
        
        threading.Thread(target=first_sweep_thread, daemon=True).start()
    
    def _run_second_sweep(self):
        """Execute the second sweep, compute diff, and plot."""
        if not self.first_sweep_data:
            messagebox.showwarning("Sweep Warning", "First sweep data missing. Run first sweep again.")
            return
        
        self.swap_sweep_button.config(state="disabled", text="Running second sweep...")
        self.status_label.config(text="Running second sweep...", fg="orange")
        
        def second_sweep_thread():
            try:
                led_powers_2, channel_values_2 = self._perform_led_sweep()
                if not led_powers_2:
                    self.root.after(0, lambda: self.update_status_error("No valid readings collected in second sweep"))
                    return
                
                # Update GUI with second sweep results
                self.root.after(0, lambda: self.update_results_from_sweep(led_powers_2, channel_values_2))
                
                # Compute difference (second - first)
                led_powers_diff, diff_values = self._compute_sweep_difference(
                    self.first_sweep_data, (led_powers_2, channel_values_2)
                )
                
                if led_powers_diff:
                    self.root.after(0, lambda: self.plot_diff_results(led_powers_diff, diff_values))
                    self.root.after(0, lambda: self.status_label.config(
                        text="Swap sweep complete (second - first plotted).", fg="green"))
                else:
                    self.root.after(0, lambda: self.update_status_error(
                        "No overlapping data points to plot difference"))
            except Exception as e:
                error_msg = f"Error during second sweep: {e}"
                print(error_msg)
                self.root.after(0, lambda: self.update_status_error(error_msg))
            finally:
                # Reset state and re-enable buttons
                self.first_sweep_data = None
                self.awaiting_second_sweep = False
                self.root.after(0, self.enable_sweep_button)
                self.root.after(0, lambda: self.swap_sweep_button.config(text="Two-sweep (swap vials)"))
        
        threading.Thread(target=second_sweep_thread, daemon=True).start()
    
    def _compute_sweep_difference(self, first_data, second_data):
        """Compute per-channel differences between two sweeps (second - first)."""
        led_powers_1, channel_values_1 = first_data
        led_powers_2, channel_values_2 = second_data
        
        def build_maps(leds, channel_values):
            maps = {}
            for ch in self.channels:
                vals = channel_values.get(ch, [])
                maps[ch] = {p: vals[i] for i, p in enumerate(leds) if i < len(vals) and vals[i] is not None}
            return maps
        
        maps1 = build_maps(led_powers_1, channel_values_1)
        maps2 = build_maps(led_powers_2, channel_values_2)
        
        # Collect all LED powers that have values in both sweeps for at least one channel
        common_powers = sorted({p for ch in self.channels for p in (set(maps1[ch].keys()) & set(maps2[ch].keys()))})
        
        diff_values = {ch: [] for ch in self.channels}
        aligned_powers = []
        
        for p in common_powers:
            any_value = False
            for ch in self.channels:
                if p in maps1[ch] and p in maps2[ch]:
                    diff = maps2[ch][p] - maps1[ch][p]
                    diff_values[ch].append(diff)
                    any_value = True
                else:
                    diff_values[ch].append(None)
            if any_value:
                aligned_powers.append(p)
            else:
                # Remove trailing None entries if no channel had a value
                for ch in self.channels:
                    if diff_values[ch]:
                        diff_values[ch].pop()
        
        return aligned_powers, diff_values
    
    def plot_diff_results(self, led_powers, diff_values):
        """Plot difference between sweeps (second - first) for each channel."""
        try:
            plot_data = {}
            for channel in self.channels:
                vals = diff_values.get(channel, [])
                points = [(p, v) for p, v in zip(led_powers, vals) if v is not None]
                if points:
                    plot_data[channel] = points
            
            if not plot_data:
                messagebox.showwarning("Plot Warning", "No valid difference data to plot")
                return
            
            fig, ax = plt.subplots(figsize=(10, 6))
            markers = ['o', 's', '^', 'd', 'v', 'x']
            for idx, (channel, points) in enumerate(plot_data.items()):
                m = markers[idx % len(markers)]
                powers = [p for p, _ in points]
                values = [v for _, v in points]
                ax.plot(powers, values, f'{m}-', label=f"{channel} (Δ)", linewidth=2, markersize=6)
            
            ax.set_xlabel('LED Power (%)', fontsize=12)
            ax.set_ylabel('Voltage Difference (V)', fontsize=12)
            ax.set_title('OD Voltage Difference (Second - First)', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            error_msg = f"Error plotting differences: {e}"
            print(error_msg)
            messagebox.showerror("Plot Error", error_msg)
    
    def enable_sweep_button(self):
        """Re-enable the sweep button"""
        self.awaiting_second_sweep = False
        self.first_sweep_data = None
        self.read_button.config(state="normal")
        self.sweep_button.config(state="normal", text="LED Power Sweep (0-20%)")
        self.swap_sweep_button.config(state="normal", text="Two-sweep (swap vials)")
    
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
