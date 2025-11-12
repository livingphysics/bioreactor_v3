import serial
import time

ser = serial.Serial("/dev/ttyUSB0",baudrate=9600,timeout=1)
ser.write(b"hello\n")
time.sleep(0.1)
print(ser.read(6))

ser.flushInput()
time.sleep(1)

while True:
	ser.flushInput()
	ser.write(b"\xFE\x44\x00\x08\x02\x9F\x25")
	time.sleep(1)
	resp=ser.read(7)
	tmp = resp
	print([f"{b:02X}" for b in resp])
	high = tmp[3]
	low = tmp[4]
	co2 = (high*256)+low
	print(" CO2 = "+str(co2*10))
	time.sleep(0.1)
	
