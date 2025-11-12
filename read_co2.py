import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import deque
import time
from atlas_i2c import atlas_i2c, sensors, commands
import serial

ser = serial.Serial("/dev/ttyUSB0",baudrate=9600,timeout=1)

ser.flushInput()
time.sleep(1)

# Initialize sensors
# sensor_co2 = sensors.Sensor("CO2", 105)
# sensor_co2.connect()
sensor_o2 = sensors.Sensor("O2", 108)
sensor_o2.connect()

# Data storage
max_points = 1000  # Number of data points to display
co2_data = deque(maxlen=max_points)
o2_data = deque(maxlen=max_points)
time_data = deque(maxlen=max_points)

# Setup the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle('Live CO2 and O2 Monitoring')

# CO2 subplot (top)
ax1.set_title('CO2 Concentration')
ax1.set_ylabel('CO2 (ppm)')
ax1.set_ylim(300, 100000)  # Fixed scale as requested
ax1.grid(True, alpha=0.3)
co2_line, = ax1.plot([], [], 'b-', linewidth=2, label='CO2')

# O2 subplot (bottom)
ax2.set_title('O2 Concentration')
ax2.set_ylabel('O2 (%)')
ax2.set_xlabel('Time')
ax2.set_ylim(0, 41)  # Fixed scale as requested
ax2.grid(True, alpha=0.3)
o2_line, = ax2.plot([], [], 'r-', linewidth=2, label='O2')

# Add legends
ax1.legend()
ax2.legend()

# Adjust layout
plt.tight_layout()

def animate(frame):
    try:
        # Read sensor data
        # co2_reading = sensor_co2.query(commands.READ)
        o2_reading = sensor_o2.query(commands.READ)
        
        # Parse the data (remove units and convert to float)
        # co2_value = float(co2_reading.data.decode().replace('ppm', '').strip())
        o2_value = float(o2_reading.data.decode().replace('%', '').strip())
        
        ser.write(b"\xFE\x44\x00\x08\x02\x9F\x25")
        time.sleep(1)
        resp=ser.read(7)
        tmp = resp
        high = tmp[3]
        low = tmp[4]
        co2_value = 10*((high*256)+low)
        
        # Add to data arrays
        current_time = time.time()
        co2_data.append(co2_value)
        o2_data.append(o2_value)
        time_data.append(current_time)
        
        # Update plots
        if len(time_data) > 1:
            # Convert time to relative seconds for better display
            time_relative = [(t - time_data[0]) for t in time_data]
            
            ax1.clear()
            ax1.set_title('CO2 Concentration')
            ax1.set_ylabel('CO2 (ppm)')
            ax1.set_ylim(300, 100000)
            ax1.grid(True, alpha=0.3)
            ax1.plot(time_relative, co2_data, 'b-', linewidth=2, label='CO2')
            ax1.legend()
            
            ax2.clear()
            ax2.set_title('O2 Concentration')
            ax2.set_ylabel('O2 (%)')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylim(0, 41)
            ax2.grid(True, alpha=0.3)
            ax2.plot(time_relative, o2_data, 'r-', linewidth=2, label='O2')
            ax2.legend()
        
        # Print current readings
        print(f"CO2: {co2_value:.1f} ppm, O2: {o2_value:.1f}%")
        time.sleep(1)
    except Exception as e:
        print(f"Error reading sensors: {e}")
    
    return co2_line, o2_line

# Start the animation
print("Starting live monitoring... Press Ctrl+C to stop")
ani = animation.FuncAnimation(fig, animate, interval=1000, blit=False)
plt.show()

# sensor_address=105
# dev = atlas_i2c.AtlasI2C()
# dev.set_i2c_address(sensor_address)
# dev.set_i2c_address(sensor_address)

# while True:
	# result = dev.query("R",processing_delay=1500)
	# print(result.data)

	
