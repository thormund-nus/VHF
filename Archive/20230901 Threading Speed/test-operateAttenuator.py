# from qodevices.homemade.qo_fibre_switch_driver import qoFibreSwitchDriver
from qodevices.baseclass.baseserial import serial_comm
from pathlib import Path

class qoVOAdriver(serial_comm):
    def __init__(self, device_path:str = '', timeout:float = 2) -> None:
        """
        Creates a qoVariableOpticalAttenuator instance.

        Input
        -----
        device_path (str): full path to the serial device as arguments
        timeout (float): Optional. serial device timeout in seconds
        """
        if not device_path:
            raise ValueError('No device path given')
        # Do not catch all errors in init method haphazardly
        super().__init__(device_path, timeout=2)

    CHANNEL_MIN = 0
    CHANNEL_MAX = 3
    RAW_CHANNEL_MIN = 0
    RAW_CHANNEL_MAX = 7

    def raw_channel(self, raw_channel_number):
        if raw_channel_number < self.RAW_CHANNEL_MIN or raw_channel_number > self.RAW_CHANNEL_MAX:
            raise ValueError(f"Out of range! Raw Channel number recieved: {raw_channel_number}")
        return qoVOArawchannel(self, raw_channel_number)

    def channel(self, channel_number):
        if channel_number < self.CHANNEL_MIN or channel_number > self.CHANNEL_MAX:
            raise ValueError(f"Out of range! Channel number recieved: {channel_number}")
        return qoVOAchannel(self, channel_number)

    @property
    def save(self) -> None:
        self.write("SAVE")

    @property
    def help(self) -> None:
        self.ask('HELP')
        print(self.read_all().decode())

    @property
    def idn(self) -> bytes:
        """
        returns device identifier
        """
        return self.ask('*IDN?')

    @property
    def reset(self) -> bytes:
        """
        reset device
        """
        self.write('*RST')

class qoVOAchannel:
    def __init__(self, parentClass: qoVOAdriver, channel_number):
        self._par = parentClass
        self._channel = channel_number
        self.ask = self._par.ask
        self.write = self._par.write

    VOLT_MIN = 0
    VOLT_MAX = 4.095

    @property
    def volt(self) -> float:
        """Query voltage across specified channel."""
        return float(self.ask(f"VOLT? {self._channel}"))

    @volt.setter
    def volt(self, val) -> None:
        if float(val) >= self.VOLT_MIN and float(val) <= self.VOLT_MAX:
            self.write(f"VOLT {self._channel} {val}")
        else:
            print(f"Illegal voltage set with value of {val}")

    @property
    def offset(self) -> float:
        return float(self.ask(f"OFFSET? {self._channel}"))

    @offset.setter
    def offset(self, val) -> None:
        self.write(f"OFFSET {self._channel} {val}")

    @property
    def factor(self) -> float:
        return float(self.ask(f"FACTOR? {self._channel}"))

    @factor.setter
    def factor(self, val):
        self.write(f"FACTOR {self._channel} {val}")

class qoVOArawchannel:
    def __init__(self, parentClass: qoVOAdriver, raw_channel_number):
        self._par = parentClass
        self._channel = raw_channel_number
        self.ask = self._par.ask
        self.write = self._par.write

    VOLT_MIN = 0
    VOLT_MAX = 4.095

    @property
    def dac(self) -> float:
        return float(self.ask(f"DAC? {self._channel}").decode())

    @dac.setter
    def dac(self, val):
        if self.VOLT_MIN <= val <= self.VOLT_MAX:
            self.write(f"DAC {self._channel} {val}")
        else:
            print(f"Illegal value into DAC recieved: {dac}")

def main():
    my_addr = '/dev/serial/by-id/usb-Centre_for_Quantum_Technologies_VOA_Driver_VOAD-QO01-if00'
    print(f"{Path(my_addr).exists() = }")
    with qoVOAdriver(my_addr) as fsd:
        res = fsd.ask("*IDN?")
        print(f"{res = }")
        print()
        print(f"{fsd.channel(0).volt = }")
        res = fsd.raw_channel(1).dac
        print(f"{type(res) = }")
        print(f"{res = }")
        fsd.help

if __name__ == '__main__':
    main()
