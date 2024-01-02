"""This script aims to prepare the VHF board for data sampling."""
from argparse import ArgumentParser
import logging
from logging import getLogger
from pathlib import Path
import sys
from tabulate import tabulate
from VHF.user_io import user_input_bool_force
from VHF.board_init.board import Board, find_device_by_sys

REDINV = '\x1b[41m'
RESET = '\x1B[0m'

# root logger (nameless) has to be created before importing other libraries
logger = getLogger()


def clear_fifo_per_board(dev_id: str, aggressive: bool = False,
                         verbose: bool = True):
    """ACM clears VHF board, and read out buffer too if specified."""
    board = Board(
        dev_id, aggressive=aggressive, verbose=verbose,
        # vhf_config_path=Path(__file__).parent.joinpath("VHF_board_params.ini")
    )
    if board.in_use() and \
            not user_input_bool_force(f"Board {board.board_id} found to be in use! Continue?", False):
        print("Not resetting!")
        return
    board.acm_clear()
    board.set_hybrid()
    if aggressive:
        board.hybrid_clear()
    if not board.valid_interface_perms():
        logger.warn(
            "Possibly VHF drivers installed under sudo. Please reinstall.")
    print(f"\n{board}")


def show_all_dev_symlinks():
    """Print to user what devices are being pointed to, and if they exist."""
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
    args = ArgumentParser(
        description="Clear FIFO queue to VHF board, and set into Hybrid mode.",
    )
    args.add_argument(
        "-a", "--aggressive",
        help="Reads out data from VHF board while in Hybrid mode.",
        action="store_true",
    )
    args.add_argument(
        "-s", "--status",
        help="",
        action="store_true",
    )
    args.add_argument(
        "-v", "--verbose",
        help="Prints out more information.",
        action="store_true",
    )
    args.add_argument(
        "--debug",
        help="Prints out logger.",
        action="store_true",
    )
    selection = args.parse_args()
    aggressiveness = selection.aggressive
    show_only = selection.status
    verbosity = selection.verbose
    debug = selection.debug

    if debug:
        logger.setLevel(logging.DEBUG)
        fmtter = logging.Formatter(
            '[%(asctime)s%(msecs)d] (%(levelname)s) %(name)s:%(funcName)s - \t %(message)s', datefmt='%H:%M:%S:')
        streamhandler = logging.StreamHandler(sys.stdout)
        streamhandler.setLevel(logging.DEBUG)
        streamhandler.setFormatter(fmtter)
        logger.addHandler(streamhandler)
        logger.debug("clear_FIFO started")

    # 1. Get all /sys/... addresses for each
    # 2. If more than 1, Inform of board ID; notify and ask to select which
    #    board to elevate.
    devices = find_device_by_sys()

    # inform user of all connected boards
    j, table_head = 0, ["idx", "Serial NO", "In Use", "USB Mode", "Interface"]
    if verbosity:
        table_head.append("Udev Address")

    def table_method(sys_bus: str):
        """List of board attributes for displaying to user."""
        d = Board(sys_bus)
        nonlocal j
        ans = [j, d.board_id, d.in_use(), d.usb_mode(), d.interface_path()]
        j += 1
        if verbosity:
            ans.append(d.hotplug_path())
        return ans

    if len(devices) > 1:
        devices.sort(key=lambda d: Board(d).board_id)

    print(tabulate(map(table_method, devices), headers=table_head))
    print()

    # Skip to end
    if show_only:
        print("Summary:")
        show_all_dev_symlinks()
        return

    # request for selection of boards to perform on.
    if len(devices) > 1:
        idxs = input(
            "Please select which boards you wish to reset (col: idx). Please delimit with commas:")
        idxs = list(map(int, idxs.split(",")))
        print(f"Selection: {idxs}")
        devices_to_reset = [devices[i] for i in idxs]
    else:
        devices_to_reset = devices
    # print(f"[Debug] {devices_to_reset=}")

    for d in devices_to_reset:
        clear_fifo_per_board(d, aggressiveness, verbosity)
        print()

    print("Summary:")
    show_all_dev_symlinks()


if __name__ == '__main__':
    main()
