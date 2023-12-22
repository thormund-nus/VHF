# This python file aims to get the 0th to 4th moment of the radius as a function
# laser power for the heterodyned phasemeter; and to measure the noise floor of
# the root mean square error of the phase, both with increasing and decreasing
# laser current. Simultaneously, the laser current and voltage can be measured.

import configparser
import logging
from moment_multiprocess.set_current import set_current_thread
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
import numpy as np
from pathlib import Path
from time import sleep
from usbtmc import list_resources


def main():
    # Directory for saving all .npz numpy arrays.
    NPZ_SAVE_DIR = Path(__file__).parent.joinpath('Moment Data')

