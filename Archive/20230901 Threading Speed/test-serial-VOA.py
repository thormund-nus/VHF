import serial

my_addr = '/dev/serial/by-id/usb-Centre_for_Quantum_Technologies_VOA_Driver_VOAD-QO01-if00'
ser = serial.Serial(my_addr, timeout=5)
ser.baudrate = 57600

print(f"{ser.get_settings() = }")
res = ser.write(b'*IDN?\n')
print(f"{ser.readline() = }")
res = ser.write(b'*IDN?\n')
print(f"{ser.readline() = }")
ser.close()

