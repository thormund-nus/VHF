"""Collection of user-facing I/O."""

from .file_select.cli import get_files_cli
from .file_select.gui import get_files_portal, get_files_qfile, get_files_util_fallback
import logging
import os
from pathlib import Path
import sys
from typing import List

__all__ = [
    "get_file",
    "get_files",
    "user_input_bool",
    "user_input_bool_force",
]

logger = logging.getLogger("VHF.user_io")


def _cast_to_path(l: List[str]) -> map:
    """Convert List[str] to map[Path]."""
    return map(lambda x: Path(x), l)


def get_file(init_dir: bytes | str, title: str = "Select Files") -> Path:
    """User-facing file selection.

    Determines if a GUI application can be spawned before falling back
    onto CLI. Returns Path (objects).
    """
    try:
        return next(get_files(init_dir, False, title))
    except StopIteration:
        return None


def get_files(init_dir: bytes | str, multiple: bool = True,
              title: str = "Select Files") -> map:
    """User-facing file selection.

    Determines if a GUI application can be spawned before falling back
    onto CLI. Returns a map (object) of Path (objects).
    """
    if os.environ.get("DISPLAY"):
        # GUI-capable environment found, as there is a screen
        # 1. Try XDG-Desktop-Portals method first ?Non-wayland DE?
        # 2. Try KDialog/Zenity next
        # 3. Try QFileDialog last

        if not os.environ.get("SSH_CONNECTION"):
            try:
                results = get_files_portal(
                    init_dir, multiple=multiple, title=title)
                # convert to Path
                return _cast_to_path(results)
            except ModuleNotFoundError:
                pass
            except KeyboardInterrupt:
                logger.info("Recieved Keyboard Interrupt.")
                sys.exit(0)
            except Exception as e:
                logger.error("get_files_portal - Exception encountered: %s", e)
        else:
            logger.info("Skipping get_files_portal. Found to be in SSH.")

        try:
            results = get_files_util_fallback(
                init_dir, multiple=multiple, title=title)
            # convert to Path
            return _cast_to_path(results)
        except KeyboardInterrupt:
            logger.info("Recieved Keyboard Interrupt.")
            sys.exit(0)
        except Exception as e:
            logger.error(
                "get_files_util_fallback - Exception encountered: %s", e)

        try:
            results = get_files_qfile(init_dir, multiple=multiple, title=title)
            # convert to Path
            return _cast_to_path(results)
        except ModuleNotFoundError:
            pass
        except KeyboardInterrupt:
            logger.info("Recieved Keyboard Interrupt.")
            sys.exit(0)
        except Exception as e:
            logger.error(
                "get_files_util_fallback - Exception encountered: %s", e)

        logger.warning("No GUI method was successful. Falling back to CLI.")

    # Not GUI-capable, likely because ssh without -X
    if not multiple:
        results = [
            get_files_cli(init_dir, multiple=multiple,
                          dir_sort=True, name_sort=True),
        ]
    else:
        results = get_files_cli(
            init_dir, multiple=multiple, dir_sort=True, name_sort=True
        )
    return _cast_to_path(results)


def user_input_bool(prompt: str) -> bool:
    """CLI yes/no prompt, defaults to default value after 1 try.

    Inputs
    ------
    prompt: str
        Prompt to be printed before (y/n): string

    Output
    ------
    True / False : bool
    """
    while True:
        try:
            my_input = input(f"{prompt} \x1B[31m(y/n)\x1B[39m: ")
        except KeyboardInterrupt:
            print("\nKeyboard Interrupt recieved. Exiting.")
            sys.exit(0)

        try:
            my_input = my_input[0].lower()
            if my_input == 'y':
                return True
            elif my_input == 'n':
                return False
            else:
                print("Invalid value!")
        except ValueError:
            print(f"Invalid value given! Input = {my_input}")


def user_input_bool_force(prompt: str, default: bool = False) -> bool:
    """CLI yes/no prompt.

    Inputs
    ------
    prompt: str
        Prompt to be printed before (y/n): string
    default: bool
        Returns default if user input fails

    Output
    ------
    True / False : bool
    """
    try:
        my_input = input(f"{prompt} \x1B[41m(y/n)\x1B[49m: ")
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt recieved. Exiting.")
        sys.exit(0)

    try:
        my_input = my_input[0].lower()
        if my_input == 'y':
            return True
        elif my_input == 'n':
            return False
        else:
            print("Invalid value!")
    except ValueError:
        print(f"Invalid value given! Input = {my_input}")
    print(f"Defaulting to {default}.")
    return default
