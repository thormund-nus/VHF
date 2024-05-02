"""Provide information on state of VHF board."""
import logging
from os import PathLike
from pathlib import Path
import subprocess
from typing import Union


__all__ = [
    "exec_exists",
    "board_in_use",
]

# Typing
_PATH = Union[str, PathLike, Path]

logger = logging.getLogger("processes")


def exec_exists(exec_name: str) -> bool:
    """Determine if exec_name exists on the computer with which command."""
    try:
        if subprocess.check_call(['which', exec_name]) == 0:
            return True
        return False
    except subprocess.CalledProcessError:
        # This occurs when the check_call has a non-zero return code.
        return False
    except Exception as e:
        logger.error("exec_exists obtained exception: %s", e)
        raise e


def board_in_use(device: _PATH) -> bool:
    """Identify if a VHF board is under use."""
    # fuser is a util that identifies if a process is using a file or socket.
    # For example, both fuser /dev/usbhybrid1 and /dev/ioboards/VHFP-QO01
    # would return
    #   /dev/usbhybrid1:     16276m
    # as fuser resolves /dev/ioboards/VHFP-QO01 to /dev/usbhybrid1.
    # As such, we only need to check the returned output to see if the board
    # is under usage.

    # dev: cast from str to Path if necessary
    dev: Path = Path(device).resolve()
    # assumes that device is in indeed in Hybrid mode instead of ACM mode
    # as such, if the device cannot be found, it must be in ACM mode, and is
    # not being used for sampling.
    if not dev.exists():
        return False
    try:
        result = subprocess.check_output(
            ["fuser", str(dev)],
            shell=False,
            # stderr=subprocess.STDOUT
            stderr=subprocess.DEVNULL,
        )
        return len(result) > 0
    except subprocess.CalledProcessError as e:
        # Somehow has e,returncode = 1 in Python but not in the shell if device
        # is not being used.
        logger.info("Tried `fuser %s`, received: %s", dev, e)
        logger.info("This is expected if device is not being used.")
        return False
