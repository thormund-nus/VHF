from subprocess import run, CompletedProcess, CalledProcessError, TimeoutExpired
from time import sleep
import logging
import os
import sys
import pathlib
import functools
from shlex import quote
from typing import Union

PathLike = Union[str, bytes, os.PathLike, pathlib.Path]

class DDS:
    """
    Controls single channel of Direct Digital Synthesizer.

    Inputs
    ------
    channel: int
        Specifies between channel 0 or 1 for DDS output.
    dds_prog_path: PathLike
        Programs path for DDS. This is the apps folder that comes from
        the SVN repository for USBDDS.
    dds_dev_path: PathLike
        Resolved path from /dev/ioboards/* for DDS board. This is due to
        write permissions that are different between the different files

    Methods
    -------
    reset_dds
        Calls the SVN repository's reset_dds executable
    """
    def __init__(self, channel: int,
            dds_prog_path: PathLike=pathlib.Path("/home/qitlab/programs/ \
                usbdds/apps/"),
            dds_dev_path: PathLike=pathlib.Path("/dev/dds0")):
        self.__createLogger()
            
        assert channel == 0 or channel == 1, "Channel must be 0 or 1"
        self._channel: str = str(channel)
        self._freq: float = 60 # MHz

        # add only resolved paths to self.dds_prog_path, self.dds_dev_path
        self.__add_DDS_paths(programs_path=dds_prog_path,
                             device_path=dds_dev_path)

        # clears AOM to be ready
        self.reset_dds()
        sleep(2)
        self.init_dds() 
        print("DDS is ready.")

    def __createLogger(self):
        self.logger = logging.getLogger("DDS")

    def __add_DDS_paths(self, programs_path: PathLike, device_path: PathLike):
        """
        This method ensures that the provided paths are suitable. If 
        they are, they will be stored in class attributes 
        self.dds_prog_path and self.dds_dev_path.

        Raises error if unsuitable.
        """
        # filedirectory parameters
        # resolving paths first
        if not isinstance(device_path, pathlib.Path):
            device_path = pathlib.Path(device_path)
        device_path = device_path.resolve()
        if not device_path.exists():
            self.logger.warning("Could not find provided device path: '%s'", 
                device_path)
            device_path = self.__greedy_device_path()
        if not isinstance(programs_path, pathlib.Path):
            programs_path = pathlib.Path(programs_path)
        
        # check if apps/... that are needed for this class to function exists
        expected_progs = ['reset_dds', 'dds_encode']
        if any([not programs_path.joinpath(x).exists() for x in expected_progs]):
            self.logger.error("Provided dds_prog_path does not have the \
                expected apps.")
            self.logger.error("dds_prog_path = %s", programs_path)
            self.logger.error("Expected apps include %s", expected_progs)
            print(f"Malformed Input {programs_path = }")
            print(f"Provided dds_prog_path does not contain apps: {expected_progs}")
            print("Exiting!")
            sys.exit(0)

        # finally, include only the resolved paths
        self.dds_prog_path: PathLike = programs_path  # location of apps folder from dds svn repository
        self.dds_dev_path: PathLike = device_path  # device path, not in IO boards for user perms
    
    def __greedy_device_path(self):
        """Returns first obtained resolved DDS device path."""
        self.logger.warning("Greedy device path has been invoked!")
        ioboards_path = pathlib.Path('/dev/ioboards')
        obtained_rscs = set(x for x in ioboards_path.iterdir() if x.name.startswith('dds_Q'))
        if len(obtained_rscs) == 0:
            self.logger.critical('No suitable DDS could be found in %s', 
                ioboards_path)
            print(f"\x1B[31mNo suitable DDS found in {ioboards_path}.\x1B[39m")
            if ioboards_path.exists():
                print(f"Only found: {list(ioboards_path.iterdir())}")
            sys.exit(0)
        dev_p = obtained_rscs.pop()
        dev_p_resolved = dev_p.resolve()
        self.logger.warning("__greedy_device_path has provided executable \
            device path to be:")
        self.logger.warning("    %s", dev_p_resolved)
        print(f"Greedy search has obtained '{dev_p}' with executable path \
            '{dev_p_resolved}'.")
        return dev_p_resolved
    
    def __handle_subprocess_run(fnc):
        # Provides error checking for subprocess.run
        @functools.wraps(fnc)  # Preserves dunder methods of func
        def wrap(self, *args, **kwargs) -> CompletedProcess:
            self.logger.info('%s is being invoked with args:', fnc)
            self.logger.info('%s', args)
            self.logger.info('and kwargs:')
            self.logger.info('%s', kwargs)
            self.logger.debug("Message has been sent to device:")
            self.logger.debug("%s", self.dds_dev_path)

            try:
                res = fnc(self, *args, **kwargs)
            except TimeoutExpired as exc:
                self.logger.critical('Timeout has occured! Details:')
                self.logger.critical('%s', exc)
                self.logger.critical('')
                self.logger.critical('exc.stderr = ')
                self.logger.critical('%s', exc.stderr)
                self.logger.critical('')
                print(f'Process Timed out!')
                sys.exit(0)
            except CalledProcessError as exc:
                self.logger.critical('Process Error has occured! Details:')
                self.logger.critical('%s', exc)
                self.logger.critical('')
                self.logger.critical('exc.stderr = ')
                self.logger.critical('%s', exc.stderr)
                self.logger.critical('')
                print(f"Process returned with error code {255-exc.returncode}.")
                print(f"{exc.stderr = }")
                sys.exit(0)
                
            self.logger.info('Process has executed as expected. Details of result:')
            self.logger.info('%s', res)
            return res
            
        return wrap

    @__handle_subprocess_run
    def reset_dds(self) -> CompletedProcess:
        """Cleans state of DDS."""
        print("\x1B[34mDDS is being resetted.\x1B[39m")
        return run([quote(str(self.dds_prog_path.joinpath("reset_dds"))), 
            '-d', quote(str(self.dds_dev_path))], timeout=6, check=True)

    @__handle_subprocess_run
    def _call_dds(self, msg: str) -> CompletedProcess:
        """Calls dds_encode as located in `dds_prog` with args given by 
        `msg`.
        """
        return run([quote(str(self.dds_prog_path.joinpath("dds_encode"))), "-d", quote(str(self.dds_dev_path))], 
            input=msg.encode(), timeout=4, check=True)
    
    def init_dds(self, freq: float=60, pow: float=0, deg: float=0):
        """
        Resets and initiates DDS.

        Input
        -----
        freq: float
            Frequency to be emitted from specified channel.
            Between 0 and 200 MHz.
        pow: float
            Power in dBm to be emitted. Not to exceed 10 dBm.
        """
        # Update to allow for various units?
        if freq > 210:
            raise ValueError(f'Frequency given too high! Recieved: {freq} (MHz).')
        if freq < 0:
            raise ValueError(f'Frequency cannot be negative! Recieved: {freq} (MHz).')
        if pow > 10:
            raise ValueError(f'Power to be output is too high! Recieved: {pow} dBm.')
        if not (0 <= deg < 360):
            raise ValueError(f'Phase should be between 0 and 360 degrees. Recieved {deg} (degrees).')
        
        freq_str = str(freq)
        power_str = str(pow)
        deg_str = str(deg)
        msg = ("LEVELS 2 ; MODE singletone\n"
            + "FREQUENCY " + self._channel + " " + freq_str + " MHz\n"
            + "AMPLITUDE "  + self._channel + " " + power_str + " " + " dBm\n"
            + "PHASE " + self._channel + " " + deg_str + " deg\n"
            + ".")
        self._call_dds(msg)
        self._freq = freq
        self._power = pow
        self._phase = deg
        return

    def _set_dds_freq(self, freq: float):
        # Update to allow for various units?
        # Do not wrapper to form float in event of splitting between value/unit
        if freq > 210:
            raise ValueError(f'Frequency given too high! Recieved: {freq} (MHz).')
        if freq < 0:
            raise ValueError(f'Frequency cannot be negative! Recieved: {freq} (MHz).')
        freq_str = str(freq)
        msg = f"FREQUENCY {self._channel} {freq_str} MHz\n."
        self._call_dds(msg)
        return

    def _set_dds_pow(self, pow: float):
        # Update to allow for various units?
        if pow > 10:
            raise ValueError(f'Power to be output is too high! Recieved: {pow} dBm.')
        power_str = str(pow)
        msg = f"AMPLITUDE {self._channel} {power_str} MHz\n."
        self._call_dds(msg)
        return
    
    def _set_dds_deg(self, deg: float):
        # Update to allow for various units?
        if not (0 <= deg < 360):
            raise ValueError(f'Phase should be between 0 and 360 degrees. Recieved {deg} (degrees).')
        deg_str = str(deg)
        msg = f"PHASE {self._channel} {deg_str} MHz\n."
        self._call_dds(msg)
        return
    
    @property
    def channel(self) -> str:
        return self._channel
    
    @property
    def freq(self) -> float:
        return self._freq

    @freq.setter
    def freq(self, val: float):
        """Sets frequency parameter of DDS in MHz."""
        self._set_dds_freq(val)
        self._freq = val
        return
    
    @property
    def power(self) -> float:
        return self._power
    
    @power.setter
    def power(self, val: float):
        """Sets amplitude parameter of DDS in dBm."""
        self._set_dds_pow(val)
        self._power = val
        return
    
    @property
    def phase(self) -> float:
        return self._phase

    @phase.setter
    def phase(self, val: float):
        """Sets phase parameter of DDS in degrees."""
        self._set_dds_deg(val)
        self._phase = val
        return

if __name__ == '__main__':
    print("[Info] AOMclass.py has been called a script.")

    from numpy import arange
    from tqdm import tqdm
    # todo: read from config file for device ID
    dds = DDS(1)
    for f in tqdm(arange(60, 65, 2)):
        dds.freq = f
        sleep(3)
