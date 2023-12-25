"""This script aims to prepare the VHF board for data sampling."""
import os
from serial import Serial
from time import sleep
import subprocess

DEVICE_BASE_PATH = '/sys/bus/usb/devices'
SET_DEVICE_MODE = 'VHF/board_init/set_device_mode'

def find_device_by_sys():
    devices = os.listdir(DEVICE_BASE_PATH)
    for device in devices:
        prod_path = os.path.join(DEVICE_BASE_PATH, device, 'product')
        prod = os.path.isfile(prod_path)
        if not prod:
            continue

        with open(prod_path, 'r') as f:
            content = f.read()
            if 'VHF' in content:
                print(f"{device = }")
                return device

def get_vhf_address(x):
    if x > 10:
        return None
    base = '/dev/serial/by-id'
    vhf_address = None
    for item in os.listdir(base):
        if 'VHF' in item.split('_'):
            print(f"Item has found to be {item = }")
            vhf_address = os.path.realpath(os.path.join(base, item))
            print(f"Address is {vhf_address = }")
    if vhf_address is None:
        subprocess.run(['python3', 'toggle_bConfigurationValue.py'])
        sleep(0.5)
        subprocess.run(['python3', 'toggle_bConfigurationValue.py'])
        sleep(0.5)
        subprocess.run(['python3', 'toggle_bConfigurationValue.py'])
        sleep(0.5)
        subprocess.run(['python3', 'toggle_bConfigurationValue.py'])
        sleep(3.5)
        return get_vhf_address(x + 1)
    return vhf_address

def read_out_buffer():
    cmd = [
            str(os.path.realpath(os.path.join(os.path.dirname(__file__), 'teststream.exec'))),
            '-U', f"{str(os.path.realpath(os.path.join(os.path.dirname(__file__), 'vhf_board.softlink')))}",
            '-q', '1200', '-s', "99",
            '-b',  # binary
            '-l',  # 10 MS/s sampling
          ]
    for i in range(10):
        output = subprocess.run(cmd, capture_output=True, check=True, cwd=os.path.dirname(__file__))
        print(f"{len(output.stdout) = }")
        sleep(0.5)
    return

def main():

    # check if bConfigurationValue = 1, sets to 1 otherwise
    bConfigPath = os.path.join(DEVICE_BASE_PATH, find_device_by_sys(), 'bConfigurationValue')
    with open(bConfigPath, 'r') as f:
        mode = f.read().strip()
    
    if mode == '2':
        print(f"VHF board found to be in Hybrid Mode. Setting to ACM mode...")
        subprocess.run([SET_DEVICE_MODE, 'set', bConfigPath, '1'])
        sleep(2)
        with open(bConfigPath, 'r') as f:
            mode = f.read().strip()
        sleep(3)
    
    if mode != '1':
        print(f"{mode = }")
        print(f"{type(mode) = }")
        print('Error setting VHF board to ACM mode.')
        return
    print('VHF board is now in ACM mode.')

    sleep(4)
    vhf = Serial(get_vhf_address(0))
    print(f"Connected to {vhf = }")
    print(f"{vhf.name = }")
    # vhf.open()
    sleep(2)
    vhf.write(b"CONFIG 16\n")  # raises
    print('Raised Clear')
    sleep(2)
    vhf.write(b"CONFIG 0\n")  # lowers
    vhf.write(b"SKIP\n")  # skips packets stuck in FIFOs
    vhf.write(b"CLOCKINIT\nADCINIT\n")  # inits as need be
    print('Lowered Clear\nFIFO should be flushed!')

    sleep(2)
    vhf.close()
    print('VFH has been closed')

    # set back to ACM
    print(f"Setting into Hybrid mode")
    subprocess.run([SET_DEVICE_MODE, 'set', bConfigPath, '2'])
    sleep(1)
    with open(bConfigPath, 'r') as f:
        mode = f.read().strip()
    print(f"Now, we have {mode = }")
    sleep(2)
    # read_out_buffer()
    print(f"{os.listdir('/dev/ioboards') = }")

if __name__ == '__main__':
    main()
