"""
Provides VHFRunner.

VHFRunner is a config parser that aims to simplify subprocess arguments.
"""
import configparser
import datetime
import logging
from enum import Enum
from os import PathLike
from os.path import realpath
from pathlib import Path
from shlex import quote
from typing import Any
from typing import IO
from typing import Iterable
from typing import Mapping
from typing import Union

__all__ = ["VHFRunner"]

# TTY
RED = '\x1B[31m'
REDBOLD = '\x1B[31;1m'
BLUE = '\x1B[34m'
RESET = '\x1B[0m'

# Typing
_PATH = Union[str, bytes, PathLike, Path]
_FILE = Union[None, int, IO[Any]]
_TXT = Union[bytes, str]


def flatten(items):
    """Yield items from any nested iterable."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x


class EnumWithAttrs(Enum):
    """Factory class."""

    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj


class SamplingSpeed(EnumWithAttrs):
    """Sampling frequency used by VHF board."""

    low = 'l', 10e6
    high = 'h', 20e6

    def __init__(self, flag, _freq):
        self.flag = flag
        self.freq = _freq

    @staticmethod
    def from_str(label: str):
        """Provisions a method to convert a string from config.ini to enum."""
        label = label.strip().lower()
        if label.startswith('l'):
            return SamplingSpeed.low
        elif label.startswith('h'):
            return SamplingSpeed.high
        else:
            raise NotImplementedError(
                "SamplingSpeed.fromstr has recieved unrecognised input: "
                f"{REDBOLD}{label}{RESET}. Accept only 'low' or 'high'.")


class Encode(EnumWithAttrs):
    """Encoding format piped out by VHF board."""

    BIN = 'b', 'bin'
    HEX = 'x', 'hex'
    ASC = 'A', 'txt'

    def __init__(self, flag, ext):
        self.flag = flag
        self.ext = ext

    @staticmethod
    def from_str(label: str):
        """Provisions a method to convert a string from config.ini to enum."""
        label = label.strip().lower()
        match label[:3].lower():
            case "bin":
                return Encode.BIN
            case "hex":
                return Encode.HEX
            case "asc" | "txt" | "tex":
                return Encode.ASC
            case _:
                raise NotImplementedError(
                    "Encode.fromstr has recieved unrecognised input: "
                    f"{REDBOLD}{label}{RESET}. Accepts only 'binary', "
                    "'hexadecimal', 'ascii', or 'text'")


class VHFRunner():
    """Configparses config file into appropriate strings for subprocess."""

    def __init__(self, conf_file: _PATH, force_to_buffer: bool = False,
                 overwrite_properties: Mapping = dict()):
        """VHFRunner oversees the necessary conditions to pass into
        subprocess for sampling data with VHF board from config file.

        Inputs
        ------
        conf_file: PathLike
            Specifies location of VHF_board parameters file to read from.
            Specifications of .ini are in accordance with Python's configparser
            library.
        force_to_buffer: bool
            If true, disregards conf_file's save_to_file option.
        """
        self.__createLogger()

        # Resolve paths automagically with ExtendedInterpoliation
        conf = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
        conf.read(conf_file)

        # Board parameters
        self.num_samples = int(eval(conf.get('Board', 'num_samples')))
        self.skip_num = int(eval(conf.get('Board', 'skip_num')))
        self.speed = SamplingSpeed.from_str(conf.get('Board', 'speed'))
        self.encode = Encode.from_str(conf.get('Board', 'encode'))
        # Map board parameters to corresponding VHF exec flag
        self._board_mappable = {'num_samples': 'q', 'skip_num': 's', }
        # Use enum attrs to get corresponding flag
        self._board_unmappable = ['speed', 'encode',]
        self.board_kwargs = {}
        for k, v in conf['Board'].items():
            if k not in (list(self._board_mappable) + self._board_unmappable):
                if k.endswith('_enable'):  # Do not save _enable keys
                    continue

                # x_enable keys determine if to save x key
                if conf.has_option('Board', k+'_enable'):
                    if conf['Board'][k+'_enable'] == 'False' \
                            or conf.getboolean('Board', k+'_enable') is False:
                        continue

                self.board_kwargs[k] = v

        # Comments for file nomenclature
        self.phasemeter_kwargs = dict(conf['Phasemeter Details'].items())

        # Save locations
        self.path = {
            'base_dir': realpath(conf['Paths']['base_dir']),
            'save_dir': realpath(conf['Paths']['save_dir']),
            'vhf_dev': realpath(conf['Paths']['board']),
            'stream': realpath(conf['Paths']['stream_exec']),
        }

        # Pass via stdout vs to a file
        self.to_file = conf.getboolean('Paths', 'save_to_file')
        if force_to_buffer:
            self.to_file = False

        # Overwrite properties read from file via script
        if len(overwrite_properties) > 0:
            self._overwrite_attr(overwrite_properties)

    def __createLogger(self):
        self.logger = logging.getLogger('VHFRunner')

    def _overwrite_attr(self, p: Mapping) -> None:
        """
        Overwrite existing known attrs in config file with those in script.

        Defaults to self.board_kwargs for attrs passed via here that are not
        previously instantised via the config file.
        """
        # Set of known keys
        rel_k = [list(self._board_mappable), self._board_unmappable,
                 list(self.board_kwargs), list(self.phasemeter_kwargs),
                 list(self.path)]

        for key, v in p.items():
            if key in flatten(rel_k):
                # To find in which kwarg to then overwrite in
                for section in rel_k:
                    if key in section:
                        if section in rel_k[:2]:
                            setattr(self, key, v)
                        else:
                            getattr(self, section)[key] = v
                        break

            else:
                self.logger.warning(
                    'Unrecognised key in overwrite_properties: %s', key)
                print(f"'overwrite_properties' contains unknown key {key} "
                      "and will be added into self.board_kwargs.")
                self.board_kwargs[key] = v

    def sample_time(self) -> float:
        """Return the amount of time in seconds to perform sampling."""
        return self.num_samples*(1+self.skip_num)/self.speed.freq

    def inform_params(self) -> None:
        """Print enabled configurations."""
        sf = self.speed.freq/(1+self.skip_num)  # sampling freq
        if sf < 1e3:
            print(f"Sampling at {sf:.4f} Hz.")
        elif sf < 1e6:
            print(f"Sampling at {sf/1e3:.4f} kHz.")
        elif sf < 1e9:
            print(f"Sampling at {sf/1e6:.4f} MHz.")
        else:
            print(f"Sampling at {sf} Hz.")

        if 'vga_num' in self.board_kwargs:
            print("Onboard gain has been set to: "
                  f"{BLUE}{self.board_kwargs['vga_num']}{RESET}.")
        if 'filter_const' in self.board_kwargs:
            print("Filter constant has been set to: "
                  f"{BLUE}{self.board_kwargs['filter_const']}{RESET}.")

        print("Phasemeter details used: "
              f"{BLUE}{self.phasemeter_kwargs}{RESET}.")
        print(f"Directory to be saved to {RED}{self.path['save_dir']}{RESET}")

        if self.to_file:
            print(f"Output will be written to {BLUE}files{RESET}.")
        else:
            print(f"Output will be captured from {BLUE}STDIN{RESET}.")

        st = self.sample_time()
        if st < 60:
            print(f"Sampling is expected to take {REDBOLD}{st:.2f} s{RESET}.")
        elif st < 3600:
            print("Sampling is expected to take "
                  f"{REDBOLD}{st/60:.3f} min{RESET}.")
        elif st < 86400:
            print("Sampling is expected to take "
                  f"{REDBOLD}{st/3600:.3f} hours{RESET}.")
        else:
            print("Sampling is expected to take "
                  f"{REDBOLD}{int(st//86400)} days "
                  f"{int((st%86400)//3600)} hours "
                  f"{((st%86400)%3600)/60:.3f} mins{RESET}.")

    def subprocess_cmd(self) -> list:
        """First argument of subprocess.run(...)."""
        def s(x): return str(x)
        def qs(x): return quote(str(x))

        def result_mappable_extend(res: list, param: str):
            res.extend(['-'+self._board_mappable[param],
                        s(getattr(self, param))])

        def result_unmappable_extend(res: list, param: str):
            res.extend(['-'+getattr(self, param).flag])

        result = [qs(self.path['stream'])]  # stream executable
        result.extend(['-U', qs(self.path['vhf_dev'])])  # board location
        for m in self._board_mappable:
            result_mappable_extend(result, m)
        for m in self._board_unmappable:
            result_unmappable_extend(result, m)
        # hardcoded x_enabled flags
        if 'vga_num' in self.board_kwargs:
            result.extend(['-G', self.board_kwargs['vga_num']])
        if 'filter_const' in self.board_kwargs:
            result.extend(['-F', self.board_kwargs['filter_const']])
        # other board params
        for k, v in self.board_kwargs.items():
            result.extend(['-'+k, s(v)])

        # Filename generated
        if self.to_file:
            if ' ' in self.path["save_dir"]:
                raise ValueError(
                    "Save directory has spaces included in it. Please use a different directory.")
            fn = datetime.datetime.now().isoformat() + \
                "".join(result[3:]).replace('-', '_')
            fn += '_' + '_'.join(flatten(self.phasemeter_kwargs.items()))
            fn += "." + self.encode.ext
            fn = qs(Path(self.path['save_dir']).joinpath(fn).resolve())
            result.extend(['-o', fn])
        return result

    def subprocess_run(self, stdout: _FILE = None) -> dict:
        """All arguments for subprocess.run(...).

        If writing to stdout instead, user is to provide their own pipe to pass
        into subprocess.run, through `stdout` arg.
        """
        result = {
            'args': self.subprocess_cmd(),
            'check': True,
            'cwd': self.path['base_dir'],
            'timeout': 7 + self.sample_time()
        }
        self.logger.debug("subprocess_run called. Returning %s", result)

        if self.to_file:
            result['capture_output'] = True
        else:
            if stdout is None:
                raise ValueError("Stdout expected argument!")
            result['stdout'] = stdout

        return result
