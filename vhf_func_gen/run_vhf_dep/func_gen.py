"""Function Generator relevant methods."""

from functools import wraps
from logging import basicConfig, getLogger, DEBUG
from typing import Optional
from pyvisa import ResourceManager  # LecroyDSO uses the pyvisa library
from pyvisa.resources import USBInstrument
from time import sleep


class SyncSingleChannelFuncGen():
    """Tektronix Function Generator Single Channel synchrous operation wrapper."""

    def __init__(self, rsc_str: str = '', channel: str = '1', timeout: float = 2) -> None:
        """Synchrous Tektronix Function Generator instance with single channel.

        Input
        -----
        device_path (str): full path to the serial device as arguments
        timeout (float): Optional. serial device timeout in seconds
        """
        # Manual obtainable from: https://www.tek.com/en/signal-generator/afg3000-manual/afg3000-series-1
        rm = ResourceManager()
        self.my_resource: USBInstrument = rm.open_resource(
            rsc_str, open_timeout=timeout
        )
        self.__init_logger()
        self.logger.info(
            "Resource has been opened. Close connection by using del on instance.")
        self.channel = channel

    # Typecasts all methods and properties of self.my_resource to self
    def __getattr__(self, __name: str):
        if __name in ('_logging_extra', '_resource_name', '_session', 'visalib'):
            return getattr(self.my_resource, __name)
        else:
            raise AttributeError(f"{__name=}")

    def __del__(self) -> None:
        if self.my_resource._session is not None:
            # self.my_resource.flush()
            self.my_resource.close()

    def __init_logger(self):
        self.logger = getLogger("TektronixFuncGen")
        self.logger.setLevel(DEBUG)

    @staticmethod
    def log_query(func):
        @wraps(func)
        def func_wrap(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            self.logger.debug("%s returned: %s", func.__name__, result)
            return result
        return func_wrap

    def ask(self, string):
        """PyVisa query method."""
        return self.my_resource.query(string).strip()

    @log_query
    def write(
        self, message: str, termination: str | None = None, encoding: str | None = None
    ) -> int:
        """Jank write redirector using property attribute."""
        self.logger.debug("write called with: message=%s, termination=%s, encoding=%s",
                          message, termination, encoding)
        return self.my_resource.write(message=message, termination=termination, encoding=encoding)

    def assert_sinusoidal_mode(self, channel: Optional[str | int] = None):
        if channel is None:
            channel = self.channel
        self.beeper_q(channel)
        try:
            assert int(self.am_q(channel)) == 0
        except AssertionError:
            self.am(0, channel)
        assert int(self.fm_q(channel)) == 0
        assert int(self.pm_q(channel)) == 0
        assert int(self.pwm_q(channel)) == 0

    def prepare_sinusoidal(self, channel: Optional[str | int] = None):
        """Ensure channel output is sinusoidal"""
        if channel is None:
            channel = self.channel
        self.write("SOURce:ROSCillator:SOURce EXTernal")  # reference clock
        self.write(f"OUTPut{channel}:IMPedance 50 ohms")  # Sets impendance
        self.write(f"OUTPUT{channel}:STATE ON")  # Turns output on
        self.write(f"SOURce{channel}:FREQuency:MODE CW")
        self.write(f"SOURce{channel}:VOLTage:UNIT Vpp")
        self.write(f"SOURce{channel}:VOLTage:LEVel:IMMediate:OFFSet 0mV")
        self.logger.info("prepare_sinusoidal completed!")

    @log_query
    def beeper_q(self, channel: Optional[str | int] = None):
        if channel is None:
            channel = self.channel
        return self.ask("SYSTem:BEEPer:STATe?")

    @log_query
    def am_q(self, channel: Optional[str | int] = None):
        if channel is None:
            channel = self.channel
        return self.ask(f"SOURce{channel}:AM:STATe?")

    def am(self, val: str | int = '0', channel: Optional[str | int] = None):
        if channel is None:
            channel = self.channel
        return self.ask(f"SOURce{channel}:AM:STATe {val}")

    @log_query
    def pm_q(self, channel: Optional[str | int] = None):
        if channel is None:
            channel = self.channel
        return self.ask(f"SOURce{channel}:PM:STATe?")

    @log_query
    def fm_q(self, channel: Optional[str | int] = None):
        if channel is None:
            channel = self.channel
        return self.ask(f"SOURce{channel}:FM:STATe?")

    @log_query
    def pwm_q(self, channel: Optional[str | int] = None):
        """Query Pulse setting on Func Gen. Corresponds to Pulse Width Modulated."""
        if channel is None:
            channel = self.channel
        return self.ask(f"SOURce{channel}:PWM:STATe?")

    def get_min_freq(self, channel: Optional[str | int] = None):
        """Get the minimum settable frequency."""
        if channel is None:
            channel = self.channel
        # Have to account for the side effect that MIN can only be obtained via setting
        tmp_freq = self.ask(f"SOURce{channel}:FREQuency?")
        self.write(f"SOURce{channel}:FREQuency MINimum")
        result = self.ask(f"SOURce{channel}:FREQuency?")
        self.write(f"SOURce{channel}:FREQuency {tmp_freq}")
        assert float(tmp_freq) == float(
            self.ask(f"SOURce{channel}:FREQuency?"))
        return result

    def get_max_freq(self, channel: Optional[str | int] = None):
        """Get the maximum settable frequency."""
        if channel is None:
            channel = self.channel
        # Have to account for the side effect that MAX can only be obtained via setting
        tmp_freq = self.ask(f"SOURce{channel}:FREQuency?")
        self.write(f"SOURce{channel}:FREQuency MAXimum")
        result = self.ask(f"SOURce{channel}:FREQuency?")
        self.write(f"SOURce{channel}:FREQuency {tmp_freq}")
        assert tmp_freq == self.ask(f"SOURce{channel}:FREQuency?")
        return result

    def set_freq(self, freq: str = "60.000MHz", channel: Optional[str | int] = None):
        """Set the channel to the specified frequency. No checking is done."""
        if channel is None:
            channel = self.channel
        self.write(f"SOURce{channel}:FREQuency {freq}")

    @log_query
    def volt_lim_low_q(self, channel: Optional[str | int] = None):
        if channel is None:
            channel = self.channel
        return self.ask(f"SOURce{channel}:VOLTage:LIMit:LOW?")

    @log_query
    def volt_lim_high_q(self, channel: Optional[str | int] = None):
        if channel is None:
            channel = self.channel
        return self.ask(f"SOURce{channel}:VOLTage:LIMit:HIGH?")

    def set_amplitude(self, ampl_vpp: float = 1.0, channel: Optional[str | int] = None):
        """Set the channel to the specified peak-to-peak voltage. No checking is done.

        This setting has a resolution of 0.1 mVpp.

        Inputs
        ------
        ampl_vpp: float
        channel: str
        """
        if channel is None:
            channel = self.channel
        self.logger.debug(
            "Setting channel %s to %sVpp. Note that if too large or small, it will not be set.",
            channel,
            ampl_vpp
        )
        self.write(f"SOURce{channel}:VOLTage:LEVel:IMMediate:AMPLitude {ampl_vpp}Vpp")  # noqa
        self.logger.debug("Setted.")
        # assert float(ampl_vpp) == (
        #     res := float(self.get_amplitude(channel))), f"result={res}"

    @log_query
    def get_amplitude(self, channel: Optional[str | int] = None):
        if channel is None:
            channel = self.channel
        return self.ask(f"SOURce{channel}:VOLTage:LEVel:IMMediate:AMPLitude?")
