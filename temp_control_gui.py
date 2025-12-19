"""
Temperature Control GUI

A GUI for controlling bioreactor temperature using the temperature sensor
and peltier driver. Features real-time temperature plotting and PID control.
"""

import tkinter as tk
from tkinter import messagebox, ttk
import sys
import os
import time
import threading
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Add src directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import Bioreactor, Config
from src.io import get_temperature, set_peltier_power, stop_peltier
from src.utils import temperature_pid_controller


class TemperatureControlGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Temperature Control")
        self.root.geometry("900x700")
        
        # Bioreactor instance
        self.bioreactor = None
        self.initialized = False
        
        # Control state
        self.pid_enabled = False
        self.setpoint = 37.0  # Default setpoint
        self.pid_thread = None
        self.pid_running = False
        
        # Data storage for plotting
        self.max_data_points = 300  # Store last 5 minutes at 1 second intervals
        self.time_data = deque(maxlen=self.max_data_points)
        self.temp_data = deque(maxlen=self.max_data_points)
        self.setpoint_data = deque(maxlen=self.max_data_points)
        self.start_time = time.time()
        
        # PID parameters
        self.kp = 12.0
        self.ki = 0.015
        self.kd = 0.0
        
        # Create widgets
        self.create_widgets()
        
        # Initialize bioreactor components
        self.init_bioreactor()
        
        # Start temperature monitoring
        self.monitoring = True
        self.update_temperature()
    
    def create_widgets(self):
        # Main container
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        left_panel = tk.Frame(main_frame, width=300)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Right panel - Plot
        right_panel = tk.Frame(main_frame)
        right_panel.pack(side='right', fill='both', expand=True)
        
        # === LEFT PANEL: Controls ===
        
        # Status section
        status_frame = tk.LabelFrame(left_panel, text="Status", font=("Arial", 12, "bold"), padx=10, pady=10)
        status_frame.pack(fill='x', pady=(0, 10))
        
        self.status_label = tk.Label(status_frame, text="Initializing...", fg="orange", font=("Arial", 10))
        self.status_label.pack(pady=5)
        
        # Current temperature display
        temp_display_frame = tk.Frame(status_frame)
        temp_display_frame.pack(pady=5)
        
        tk.Label(temp_display_frame, text="Current Temp:", font=("Arial", 11)).pack()
        self.temp_label = tk.Label(temp_display_frame, text="---", font=("Arial", 24, "bold"), fg="blue")
        self.temp_label.pack()
        tk.Label(temp_display_frame, text="°C", font=("Arial", 14)).pack()
        
        # Setpoint section
        setpoint_frame = tk.LabelFrame(left_panel, text="Setpoint Control", font=("Arial", 12, "bold"), padx=10, pady=10)
        setpoint_frame.pack(fill='x', pady=(0, 10))
        
        # Setpoint input
        input_frame = tk.Frame(setpoint_frame)
        input_frame.pack(pady=5)
        
        tk.Label(input_frame, text="Setpoint:", font=("Arial", 10)).pack(side='left', padx=5)
        self.setpoint_entry = tk.Entry(input_frame, width=10)
        self.setpoint_entry.insert(0, str(self.setpoint))
        self.setpoint_entry.pack(side='left', padx=5)
        tk.Label(input_frame, text="°C", font=("Arial", 10)).pack(side='left', padx=2)
        
        tk.Button(input_frame, text="Set", width=6, command=self.update_setpoint).pack(side='left', padx=5)
        
        # Quick setpoint buttons
        quick_frame = tk.Frame(setpoint_frame)
        quick_frame.pack(pady=5)
        
        tk.Button(quick_frame, text="25°C", width=6, command=lambda: self.set_setpoint(25.0)).pack(side='left', padx=2)
        tk.Button(quick_frame, text="30°C", width=6, command=lambda: self.set_setpoint(30.0)).pack(side='left', padx=2)
        tk.Button(quick_frame, text="37°C", width=6, command=lambda: self.set_setpoint(37.0)).pack(side='left', padx=2)
        tk.Button(quick_frame, text="42°C", width=6, command=lambda: self.set_setpoint(42.0)).pack(side='left', padx=2)
        
        # Setpoint display
        setpoint_display_frame = tk.Frame(setpoint_frame)
        setpoint_display_frame.pack(pady=5)
        
        tk.Label(setpoint_display_frame, text="Target:", font=("Arial", 10)).pack(side='left', padx=5)
        self.setpoint_label = tk.Label(setpoint_display_frame, text=f"{self.setpoint:.1f}°C", 
                                       font=("Arial", 12, "bold"), fg="green")
        self.setpoint_label.pack(side='left', padx=5)
        
        # PID Control section
        pid_frame = tk.LabelFrame(left_panel, text="PID Control", font=("Arial", 12, "bold"), padx=10, pady=10)
        pid_frame.pack(fill='x', pady=(0, 10))
        
        # PID enable/disable
        self.pid_toggle = tk.Button(pid_frame, text="Enable PID", width=15, height=2,
                                    command=self.toggle_pid, bg="#4CAF50", fg="white",
                                    font=("Arial", 11, "bold"), state="disabled")
        self.pid_toggle.pack(pady=5)
        
        # PID parameters
        params_frame = tk.Frame(pid_frame)
        params_frame.pack(pady=5)
        
        # Kp
        kp_frame = tk.Frame(params_frame)
        kp_frame.pack(fill='x', pady=2)
        tk.Label(kp_frame, text="Kp:", width=8, anchor='w').pack(side='left')
        self.kp_entry = tk.Entry(kp_frame, width=10)
        self.kp_entry.insert(0, str(self.kp))
        self.kp_entry.pack(side='left')
        
        # Ki
        ki_frame = tk.Frame(params_frame)
        ki_frame.pack(fill='x', pady=2)
        tk.Label(ki_frame, text="Ki:", width=8, anchor='w').pack(side='left')
        self.ki_entry = tk.Entry(ki_frame, width=10)
        self.ki_entry.insert(0, str(self.ki))
        self.ki_entry.pack(side='left')
        
        # Kd
        kd_frame = tk.Frame(params_frame)
        kd_frame.pack(fill='x', pady=2)
        tk.Label(kd_frame, text="Kd:", width=8, anchor='w').pack(side='left')
        self.kd_entry = tk.Entry(kd_frame, width=10)
        self.kd_entry.insert(0, str(self.kd))
        self.kd_entry.pack(side='left')
        
        tk.Button(params_frame, text="Update PID", width=12, command=self.update_pid_params).pack(pady=5)
        
        # Manual Control section
        manual_frame = tk.LabelFrame(left_panel, text="Manual Control", font=("Arial", 12, "bold"), padx=10, pady=10)
        manual_frame.pack(fill='x', pady=(0, 10))
        
        # Peltier power control
        power_frame = tk.Frame(manual_frame)
        power_frame.pack(pady=5)
        
        tk.Label(power_frame, text="Power:", font=("Arial", 10)).pack(side='left', padx=5)
        self.power_entry = tk.Entry(power_frame, width=8)
        self.power_entry.insert(0, "0")
        self.power_entry.pack(side='left', padx=5)
        tk.Label(power_frame, text="%", font=("Arial", 10)).pack(side='left', padx=2)
        
        # Direction
        direction_frame = tk.Frame(manual_frame)
        direction_frame.pack(pady=5)
        
        self.direction_var = tk.StringVar(value="heat")
        tk.Radiobutton(direction_frame, text="Heat", variable=self.direction_var, 
                      value="heat", font=("Arial", 10)).pack(side='left', padx=5)
        tk.Radiobutton(direction_frame, text="Cool", variable=self.direction_var, 
                      value="cool", font=("Arial", 10)).pack(side='left', padx=5)
        
        # Manual control buttons
        manual_btn_frame = tk.Frame(manual_frame)
        manual_btn_frame.pack(pady=5)
        
        tk.Button(manual_btn_frame, text="Apply", width=8, command=self.apply_manual_control).pack(side='left', padx=2)
        tk.Button(manual_btn_frame, text="Stop", width=8, command=self.stop_peltier_manual).pack(side='left', padx=2)
        
        # === RIGHT PANEL: Plot ===
        
        plot_frame = tk.LabelFrame(right_panel, text="Temperature Plot", font=("Arial", 12, "bold"), padx=10, pady=10)
        plot_frame.pack(fill='both', expand=True)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Time (seconds)', fontsize=10)
        self.ax.set_ylabel('Temperature (°C)', fontsize=10)
        self.ax.set_title('Real-time Temperature', fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_ylim(10, 50)  # Fixed y-axis limits
        
        # Initialize plot lines
        self.temp_line, = self.ax.plot([], [], 'b-', linewidth=2, label='Temperature')
        self.setpoint_line, = self.ax.plot([], [], 'r--', linewidth=2, label='Setpoint')
        self.ax.legend(loc='upper right')
        
        # Embed plot in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
    
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
                    'i2c': False,
                    'temp_sensor': True,  # Enable temperature sensor
                    'peltier_driver': True,  # Enable peltier driver
                    'stirrer': False,
                    'led': False,
                    'optical_density': False,
                }
                config.LOG_TO_TERMINAL = True 
                
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
        self.pid_toggle.config(state="normal")
        self.setpoint_entry.config(state="normal")
        self.power_entry.config(state="normal")
    
    def update_status_error(self, error_msg):
        """Update UI when initialization fails"""
        self.status_label.config(text=f"Error: {error_msg[:50]}...", fg="red")
        messagebox.showerror("Initialization Error", error_msg)
    
    def update_setpoint(self):
        """Update setpoint from entry field"""
        try:
            new_setpoint = float(self.setpoint_entry.get())
            self.set_setpoint(new_setpoint)
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number for setpoint")
    
    def set_setpoint(self, value):
        """Set temperature setpoint"""
        self.setpoint = float(value)
        self.setpoint_entry.delete(0, tk.END)
        self.setpoint_entry.insert(0, f"{self.setpoint:.1f}")
        self.setpoint_label.config(text=f"{self.setpoint:.1f}°C")
    
    def update_pid_params(self):
        """Update PID parameters from entry fields"""
        try:
            self.kp = float(self.kp_entry.get())
            self.ki = float(self.ki_entry.get())
            self.kd = float(self.kd_entry.get())
            messagebox.showinfo("PID Updated", f"PID parameters updated:\nKp={self.kp}, Ki={self.ki}, Kd={self.kd}")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for PID parameters")
    
    def toggle_pid(self):
        """Enable/disable PID control"""
        if not self.initialized or self.bioreactor is None:
            messagebox.showwarning("Not Initialized", "Bioreactor not initialized yet")
            return
        
        if self.pid_enabled:
            # Disable PID
            self.pid_running = False
            self.pid_enabled = False
            self.pid_toggle.config(text="Enable PID", bg="#4CAF50")
            if self.pid_thread and self.pid_thread.is_alive():
                # Wait a bit for thread to finish
                pass
            stop_peltier(self.bioreactor)
            self.status_label.config(text="PID Disabled", fg="orange")
        else:
            # Enable PID
            self.pid_enabled = True
            self.pid_running = True
            self.pid_toggle.config(text="Disable PID", bg="#f44336")
            
            # Reset PID state
            if hasattr(self.bioreactor, '_temp_integral'):
                self.bioreactor._temp_integral = 0.0
            if hasattr(self.bioreactor, '_temp_last_error'):
                self.bioreactor._temp_last_error = 0.0
            if hasattr(self.bioreactor, '_temp_last_time'):
                self.bioreactor._temp_last_time = None
            if hasattr(self.bioreactor, '_temp_start_time'):
                self.bioreactor._temp_start_time = time.time()
            
            # Start PID control thread
            self.pid_thread = threading.Thread(target=self.pid_control_loop, daemon=True)
            self.pid_thread.start()
            self.status_label.config(text="PID Enabled", fg="green")
    
    def pid_control_loop(self):
        """PID control loop running in separate thread"""
        while self.pid_running and self.pid_enabled:
            try:
                if self.bioreactor and self.initialized:
                    temperature_pid_controller(
                        self.bioreactor,
                        setpoint=self.setpoint,
                        kp=self.kp,
                        ki=self.ki,
                        kd=self.kd,
                        max_duty=70.0,
                        warmup_time=0.0  # No warmup for GUI control
                    )
                time.sleep(1)  # Run PID every second
            except Exception as e:
                print(f"PID control error: {e}")
                time.sleep(1)
    
    def apply_manual_control(self):
        """Apply manual peltier control"""
        if not self.initialized or self.bioreactor is None:
            messagebox.showwarning("Not Initialized", "Bioreactor not initialized yet")
            return
        
        if self.pid_enabled:
            messagebox.showwarning("PID Active", "Please disable PID control before using manual control")
            return
        
        try:
            power = float(self.power_entry.get())
            power = max(0.0, min(100.0, power))  # Clamp to 0-100
            direction = self.direction_var.get()
            
            success = set_peltier_power(self.bioreactor, power, forward=direction)
            if success:
                self.status_label.config(text=f"Manual: {power:.1f}% {direction}", fg="blue")
            else:
                messagebox.showerror("Error", "Failed to set peltier power")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number for power (0-100)")
    
    def stop_peltier_manual(self):
        """Stop peltier (manual control)"""
        if not self.initialized or self.bioreactor is None:
            return
        
        if self.pid_enabled:
            messagebox.showwarning("PID Active", "Please disable PID control before using manual control")
            return
        
        stop_peltier(self.bioreactor)
        self.status_label.config(text="Peltier Stopped", fg="orange")
    
    def update_temperature(self):
        """Update temperature reading and plot"""
        if not self.monitoring:
            return
        
        if self.initialized and self.bioreactor is not None:
            try:
                current_temp = get_temperature(self.bioreactor, sensor_index=0)
                
                if not np.isnan(current_temp):
                    # Update display
                    self.temp_label.config(text=f"{current_temp:.2f}", fg="blue")
                    
                    # Update data
                    elapsed = time.time() - self.start_time
                    self.time_data.append(elapsed)
                    self.temp_data.append(current_temp)
                    self.setpoint_data.append(self.setpoint)
                    
                    # Update plot
                    if len(self.time_data) > 0:
                        self.temp_line.set_data(list(self.time_data), list(self.temp_data))
                        self.setpoint_line.set_data(list(self.time_data), list(self.setpoint_data))
                        
                        # Auto-scale x-axis, fixed y-axis limits (10-50°C)
                        if len(self.time_data) > 1:
                            self.ax.set_xlim(max(0, elapsed - 300), max(300, elapsed + 10))
                        else:
                            self.ax.set_xlim(0, 300)
                        self.ax.set_ylim(10, 50)  # Fixed y-axis limits
                        
                        self.canvas.draw()
                else:
                    self.temp_label.config(text="---", fg="gray")
            except Exception as e:
                print(f"Error reading temperature: {e}")
                self.temp_label.config(text="Error", fg="red")
        
        # Schedule next update (every second)
        self.root.after(1000, self.update_temperature)
    
    def on_closing(self):
        """Handle window closing"""
        self.monitoring = False
        self.pid_running = False
        
        if self.initialized and self.bioreactor is not None:
            try:
                stop_peltier(self.bioreactor)
            except:
                pass
        
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = TemperatureControlGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
