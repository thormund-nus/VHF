"""This script aims to prepare the VHF board for data sampling."""
from enum import Enum
from functools import cached_property
from logging import Logger
import os
from os import PathLike
from pathlib import Path
from serial import Serial
from shlex import quote
from stat import S_IRGRP, S_IWGRP
import subprocess
from tempfile import TemporaryFile
from time import sleep
from typing import List, Union
from VHF.process import board_in_use
from VHFRunner.VHFRunner import VHFRunner

_PATH = Union[str, PathLike, Path]

logger = Logger("clear_FIFO")

DEVICE_BASE_PATH = '/sys/bus/usb/devices'
SET_DEVICE_MODE = 'VHF/board_init/set_device_mode'
EXPECTED_PRODUCT = "VHF Processor"

REDINV = '\x1b[41m'
RESET = '\x1B[0m'

def find_device_by_sys() -> List[str]:
    """List all usb-devices whose product description are VHF.

    bash equivalent: grep -r VHF /sys/bus/usb/devices/*/product
    """
    devices = os.listdir(DEVICE_BASE_PATH)
    result = []
    for device in devices:
        prod_path = os.path.join(DEVICE_BASE_PATH, device, 'product')
        prod = os.path.isfile(prod_path)
        if not prod:
            continue

        with open(prod_path, 'r') as f:
            content = f.read()
            if 'VHF' in content:
                print(f"{device=}")
                result.append(device)
    return result

class EnumWithAttrs(Enum):
    """Factory class."""

    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj


class USBMode(EnumWithAttrs):
    """VHF Boards can run in 2 USB modes."""

    ACM = 1, "ACM"
    Hybrid = 2, "Hybrid"

    def __init__(self, bConfVal: int, _string):
        self.bConfVal = bConfVal
        self.string = _string
        self.val = str(self.bConfVal)

    def __str__(self):
        return self.string


class Board:
    """Manages paths for clearing FIFO queue of VHF Board."""

    def __init__(self, sys_bus_dev_id: str, aggressive: bool = False,
                 vhf_config_path: _PATH = Path(
                     __file__).parent.joinpath("VHF_board_params.ini"),
                 verbose: bool = False, very_verbose: bool = False):
        # Class configuration
        self._aggressive = aggressive
        self.verbose = verbose
        self.very_verbose = very_verbose
        self.vhf_config_path = vhf_config_path

        # Board details
        self.usb_device_id = sys_bus_dev_id
        self.product = self.get_product()
        assert EXPECTED_PRODUCT == self.product, \
            f"Product as read from {self.usb_device_id} was {
                self.product}, was expecting {EXPECTED_PRODUCT}"
        self.board_id = open(
            Path(DEVICE_BASE_PATH)
            .joinpath(self.usb_device_id)
            .joinpath("serial")
        ).read().strip()  # VHFP-QO01

    def __repr__(self):
        string = "\n".join(
            [
                f"Usb Device ID: {self.usb_device_id}",
                f"Board ID: {self.board_id}",
                f"USB Mode: {self.usb_mode()}\n"
                f"Access Address: {self.interface_path()}",
                f"In use: {self.in_use()}"
            ]
        )
        if self.verbose:
            string += f"\nUdev Address: {self.hotplug_path()}"
        return string

    def get_product(self) -> str:
        return open(
            Path(DEVICE_BASE_PATH)
            .joinpath(self.usb_device_id)
            .joinpath("product")
        ).read().strip()

    @cached_property
    def cwd(self) -> Path:
        """Sets CWD subprocess.run(...) for setting device made."""
        return Path(__file__).parent.resolve()

    @cached_property
    def bConfigPath(self) -> Path:
        return Path(DEVICE_BASE_PATH).joinpath(self.usb_device_id).joinpath("bConfigurationValue")

    def get_bConfig(self) -> str:
        return open(self.bConfigPath).read().strip()

    def usb_mode(self) -> USBMode:
        result = USBMode(int(self.get_bConfig()))
        logger.info("Found board to be in mode %s", result)
        if self.very_verbose:
            print(f"{self.board_id} found to be in {result}.")
        return result

    def hotplug_path(self) -> Path:
        """Path as created from udev rule."""
        bConfigValue = self.usb_mode()
        if bConfigValue == USBMode.ACM:
            # check /dev/serial/by-id, since this is a generic serial device
            iterator = Path("/dev/serial/by-id").iterdir()
        elif bConfigValue == USBMode.Hybrid:
            # check /dev/ioboards
            iterator = Path("/dev/ioboards").iterdir()
        else:
            raise ValueError("Unknown USB Mode")

        found = list(filter(lambda x: self.board_id in str(x), iterator))
        if len(found) > 1:
            logger.warn(
                "More than 1 hotplug_path found. Searched with %s", self.board_id)
            logger.warn("Found: %s", found)
            if self.verbose:
                print(f"Search: {self.board_id}\nMore than 1 found: {found}")
        return found[0]

    def interface_path(self) -> Path:
        """Path to read and write from for this VHF board."""
        return self.hotplug_path().resolve()

    def valid_interface_perms(self) -> bool:
        """Check if interface_path() has correct read/write perms."""
        # We just need the group of the interface_path to be matching
        # with read/write access
        stat_info = os.stat(self.interface_path()).st_mode
        return self.interface_path().stat().st_gid == os.getgid() and \
            bool(stat_info & S_IRGRP) and \
            bool(stat_info & S_IWGRP)

    def set_acm(self) -> None:
        """Set VHF Board to USB ACM mode."""
        logger.info("Attempting to set into ACM mode.")
        if self.verbose:
            print("Attempting to set into ACM mode.")
        if self.usb_mode() != USBMode.ACM:
            try:
                subprocess.run(
                    [SET_DEVICE_MODE, 'set', quote(str(self.bConfigPath)), USBMode.ACM.val],
                    check=True,
                    stderr=subprocess.PIPE,
                    cwd=self.cwd,
                    timeout=3
                )
            except subprocess.CalledProcessError as exc:
                logger.critical("set_device_mode error! exc = %s", exc)
                logger.critical("exc.stderr = %s", exc.stderr)
                raise exc
            sleep(1)
            assert self.usb_mode() == USBMode.ACM

    def set_hybrid(self) -> None:
        """Set VHF Board to USB Hybrid mode."""
        logger.info("Attempting to set into Hybrid mode.")
        if self.verbose:
            print("Attempting to set into Hybrid mode.")
        if self.usb_mode() != USBMode.Hybrid:
            try:
                subprocess.run(
                    [SET_DEVICE_MODE, 'set', quote(str(self.bConfigPath)), USBMode.Hybrid.val],
                    check=True,
                    stderr=subprocess.PIPE,
                    cwd=self.cwd,
                    timeout=3
                )
            except subprocess.CalledProcessError as exc:
                logger.critical("set_device_mode error! exc = %s", exc)
                logger.critical("exc.stderr = %s", exc.stderr)
                raise exc
            sleep(1)
            assert self.usb_mode() == USBMode.Hybrid

    def toggle_USB_mode(self) -> None:
        """Toggles between ACM and Hybrid mode on VHF board."""
        m = self.usb_mode()
        if m == USBMode.ACM:
            self.set_hybrid()
        elif m == USBMode.Hybrid:
            self.set_acm()

    def acm_clear(self) -> None:
        """ACM method of clearing FIFO queue."""
        self.set_acm()
        with Serial(quote(str(self.interface_path()))) as vhf:
            vhf.write(b"CONFIG 16\n")  # raises
            print("Raised Clear")
            sleep(1)
            vhf.write(b"CONFIG 0\n")  # lowers
            vhf.write(b"SKIP\n")  # skips packets stuck in FIFOs
            vhf.write(b"CLOCKINIT\nADCINIT\n")  # inits as need be
            print("Lowered Clear\nFIFO should be flushed!")

    def in_use(self) -> bool:
        """Tell if board is currently in use."""
        return board_in_use(self.interface_path())

    def hybrid_clear(self) -> None:
        """If ACM raising and lowering was not enough to clear FIFO, reading
        data out from Hybrid mode might just do the trick!"""
        if self._aggressive != True:
            logger.warning("Not in aggressive mode.")
            return
        if self.usb_mode() != USBMode.Hybrid:
            logger.warning("Board not found to be in Hybrid mode!")
            return
        if self.in_use():
            logger.warning("VHF Board found to be in use!.")
            return

        vhf_runner = VHFRunner(
            self.vhf_config_path,
            force_to_buffer=True,
            overwrite_properties={
                'skip_num': 99,
                'num_samples': 1200,
                'v': 0,
                'vhf_dev': quote(str(self.interface_path())),
            },
        )
        for _ in range(10):
            try:
                with TemporaryFile(dir="/dev/shm") as f:
                    output = subprocess.run(
                        **(vhf_runner.subprocess_run(f)),
                        stderr=subprocess.PIPE,
                    )
                    print(f"[Board.hybrid_clear] readout has len = {f.tell()}")
            except subprocess.CalledProcessError as exc:
                logger.critical("CalledProcess error with exception! Details:")
                logger.critical("%s", exc)
                logger.critical("exc.stderr = ")
                logger.critical("%s", exc.stderr)
            except subprocess.TimeoutExpired as exc:
                logger.critical("TimeoutExpired with exception:")
                logger.critical("%s", exc)
                logger.critical("exc.stderr = ")
                logger.critical("%s", exc.stderr)
            sleep(0.1)


def clear_fifo_per_board(dev_id: str, aggressive: bool = False,
                         verbose: bool = True):
    board = Board(dev_id, aggressive=aggressive, verbose=verbose)
    board.acm_clear()
    board.set_hybrid()
    if aggressive:
        board.hybrid_clear()
    if not board.valid_interface_perms():
        logger.warn(
            "Possibly VHF drivers installed under sudo. Please reinstall.")
    print(f"\n{board}")

def show_all_dev_symlinks():
    """Prints to user what devices are being pointed to, and if they exist."""
    def relative_to_dev(x: Path):
        try:
            return x.resolve().relative_to("/dev", walk_up=False)
        except ValueError:
            return False
    ps = filter(lambda p: p.is_symlink(), Path(__file__).parent.rglob("[!.]*"))
    ps = filter(relative_to_dev, ps)
    for p in ps:
        print(f"./{p.relative_to(Path(__file__).parent)}: "
            f"{REDINV if not p.resolve().exists() else RESET}"
            f"{str(p.resolve())}{RESET}")
    return ps

def main():
    """Uplift VHF board into USB Hybrid mode."""
    # argparse asking for aggressive, i.e.: to read out from buffer. verbose
    # flag is passed too

    # 1. Get all /sys/... addresses for each
    # 2. If more than 1, Inform of board ID; notify and ask to select which
    #    board to elevate.
    devices = find_device_by_sys()
    if len(devices) > 1:
        print("TODO: clear the more than 1 board found. Clearing only the first")
        print("Found: ")
        for x in devices:
            print(f"{Board(x)}\n")
    devices = devices[0]

    clear_fifo_per_board(devices, True, True)


if __name__ == '__main__':
    main()
