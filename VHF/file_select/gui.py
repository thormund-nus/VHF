"""Collection of GUI file-selection functions for user_io.py."""

import logging
import os
import platform
from ..process import exec_exists
import subprocess
import sys
from typing import List

__all__ = [
    "get_files_portal",
    "get_files_qfile",
    "get_files_util_fallback",
]

logger = logging.getLogger("VHF.file_select.gui")


def get_files_portal(init_dir: bytes | str, multiple: bool = True,
                     title: str = "Select Files") -> List[str]:
    """Use xdg-desktop-portals File Dialog for file selection."""
    try:
        import gi
    except ModuleNotFoundError:
        logger.warning("PyGObject likely to not be installed under pip.")
        raise ModuleNotFoundError
    gi.require_version('Gtk', '3.0')
    from gi.repository import Gtk

    if os.environ.get("GTK_USE_PORTAL") != '1':
        logger.debug(
            "Native File Chooser cannot be selected as environment variable "
            "was not found be 1."
        )

    # https://docs.gtk.org/gtk3/class.FileChooserNative.html
    dialog = Gtk.FileChooserNative.new(
        title,  # Title
        None,  # Parent Window
        Gtk.FileChooserAction.OPEN,  # Action
        None,  # "_Open"
        None  # "_Cancel"
    )
    dialog.set_select_multiple(multiple)

    # doesn't seem to work when passing from GTK to KDE
    dialog.set_current_folder(init_dir)

    response = dialog.run()
    filename = dialog.get_filenames()  # get_filename for single file
    if response == Gtk.ResponseType.ACCEPT:
        logger.debug("Portal Dialog Accepted")
    elif response == Gtk.ResponseType.CANCEL:
        logger.debug("Portal Dialog Cancelled")
    else:
        logger.warning("dialog response = %s", response)
    dialog.destroy()

    return filename


def get_files_util_fallback(init_dir: bytes | str, multiple: bool = True,
                            title: str = "Select Files") -> List[str]:
    """Use KDialog/Zenity's file dialog for file selection."""
    if platform.system() != "Linux":
        logger.warning(
            "kdialog and zenity are KDE/GNOME specific CLI tools on Linux.")
        raise Exception("Running on non-Linux system.")

    if exec_exists("kdialog"):
        logger.debug("kdialog was found.")
        cmd = ["kdialog", "--getopenfilename", "--separate-output", init_dir]
        if multiple:
            cmd.append("--multiple")
    elif exec_exists("zenity"):
        logger.debug("zenity was found.")
        # init_dir doesn't seem to work for zenity in GNOME
        cmd = ["zenity", "--file-selection",
               "--filename=" + init_dir, "--separator=\n"]
        # filter can be added with "--file-filter=NAME | PATTERN1 PATTERN2 ..."
        if multiple:
            cmd.append("--multiple")
    else:
        logger.warning("kdialog and zenity were not found.")
        raise Exception("KDialog and Zenity were not found.")

    result = subprocess.check_output(cmd)
    result = result.decode()
    result = result.split("\n")
    result.pop()
    return result


def get_files_qfile(init_dir: bytes | str, multiple: bool = True,
                    title: str = "Select Files", filt: str = '') -> List[str]:
    """Use PyQt(6) file dialog for file selection."""
    # https://doc.qt.io/qtforpython-5/PySide2/QtWidgets/QFileDialog.html#PySide2.QtWidgets.PySide2.QtWidgets.QFileDialog.getOpenFileNames
    try:
        from PyQt6.QtWidgets import QApplication, QFileDialog
    except ModuleNotFoundError:
        logger.warning("PyQt6 likely to not be installed under pip.")
        raise ModuleNotFoundError

    with QApplication(sys.argv) as _:
        if multiple:
            calling = QFileDialog.getOpenFileNames
        else:
            calling = QFileDialog.getOpenFileName
        result = calling(
            None,
            title,
            init_dir,
            filt
        )

    if multiple:
        return result[0]
    else:
        return [result[0],]
