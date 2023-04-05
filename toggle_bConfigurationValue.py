import subprocess
import os

DEVICE_BASE_PATH = '/sys/bus/usb/devices'

def find_device():
    devices = os.listdir(DEVICE_BASE_PATH)
    for dev in devices:
        prod_path = os.path.join(DEVICE_BASE_PATH, dev, 'product')
        prod = os.path.isfile(prod_path)
        if not prod:
            continue

        with open(prod_path, 'r') as f:
            content = f.read()
            if 'VHF' in content:
                return dev

bConfigPath = os.path.join(DEVICE_BASE_PATH, find_device(), 'bConfigurationValue')
mode = ''
with open(bConfigPath, 'r') as f:
    mode = f.read()
    print(f'{mode = }')

if '1' in mode:
    print('Setting VHF board bConfigurationValue to 2')
    subprocess.run(['./set_device_mode', 'set', bConfigPath, '2'])
elif '2' in mode:
    print('Setting VHF board bConfigurationValue to 1')
    subprocess.run(['./set_device_mode', 'set', bConfigPath, '1'])
else:
    print('wtf??')
    os.exit(0)

newmode = ''
with open(bConfigPath, 'r') as f:
    newmode = f.read()

assert newmode != mode
print(f'{newmode = }')