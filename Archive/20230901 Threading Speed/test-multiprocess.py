"""
To test if multicore is viable.
"""
from collections import deque
import configparser
import datetime
import logging
import multiprocessing as mp
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
import os
from os import PathLike
from pathlib import Path
import sys
import subprocess
from time import sleep
from typing import Union

module_path = str(Path(__file__).parents[1].joinpath("20230902 Moment Record"))
if module_path not in sys.path:
    sys.path.append(module_path)

from test_vhf_thread import vhf_manager  # noqa
from moment_multiprocess.vhf_output_analysis import vhf_output_analysis as analysis  # noqa


def main():
    """
    This is a test driver loop.

    This test aims to create a few multicore VHF controllers to see if
    staggered sampling is possible.
    """
    # logging.basicConfig(filename= f"{Path(__file__).parents[2].joinpath('Log').joinpath('test-multi_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.log')}",
    #     filemode='a', format='[%(asctime)s%(msecs)d] (%(levelname)s) %(name)s - \t %(message)s',
    #     datefmt='%H:%M:%S:',
    #     level=logging.DEBUG
    # )
    logger = logging.getLogger('test-multiprocess')
    # base logger has to be lower level than handlers
    logger.setLevel(logging.DEBUG)

    filehandler = logging.FileHandler(
        filename=f"{Path(__file__).parents[2].joinpath('Log').joinpath('test-multi_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.log')}",
        mode='a'
    )
    filehandler.setLevel(logging.DEBUG)
    streamhandler = logging.StreamHandler(sys.stdout)
    streamhandler.setLevel(logging.INFO)
    fmtter = logging.Formatter(
        '[%(asctime)s%(msecs)d] (%(levelname)s) %(name)s - \t %(message)s', datefmt='%H:%M:%S:')
    filehandler.setFormatter(fmtter)
    streamhandler.setFormatter(fmtter)

    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logger.info("Starting test-multiprocess!")

    # Create save files
    npz_dir = Path(__file__).parent.joinpath('test_Results')
    # init_npz_file(npz_dir)  # TODO

    # Create all child processes
    num_vhf_managers = 3
    vhf_pipes = [Pipe() for _ in range(num_vhf_managers)]
    pipes_str_to_num = {}
    for i, p in enumerate(vhf_pipes):
        logger.debug("Parent and child pipe for vhf manager %s: %s", i, p)
        pipes_str_to_num[str(p)] = i
    vhfs = [
        Process(
            target=vhf_manager,
            args=(vhf_pipes[i][1], i, Path(
                __file__).parent.joinpath('vhf-params.ini'))
        ) for i in range(num_vhf_managers)
    ]

    # Start child processes
    _ = [j.start() for j in vhfs]
    # Ensure initialisation complete
    for pipe, _ in vhf_pipes:
        assert pipe.recv_bytes() == b'0', f"Error on VHF process: {pipe = }"

    # Queue of available pipes
    available_pipes = deque([p for p, _ in vhf_pipes], maxlen=len(vhf_pipes))
    # print(f"[Debug] {id(available_pipes[0]) = }")
    # print(f"[Debug] {id(vhf_pipes[0][0]) = }")
    # print(f"[Debug] {available_pipes = }")
    # print(f"[Debug] {vhf_pipes = }")

    def requeue_all_free():
        """Used for collecting all free VHF processes back into use."""
        for pipe, _ in vhf_pipes:
            # print(f"[Driver] requeue looping: {pipe = }")
            try:
                if pipe.poll():
                    logger.info("[Driver] Polled pipe.")
                    msg = pipe.recv_bytes()
                    if msg != b'0':
                        logger.warning("[Driver] Recieved issue on pipe.")

                    available_pipes.append(pipe)
                    logger.info("[Driver] Requeued a pipe.")
            except BlockingIOError:
                logger.info(
                    ("[Driver] Polling raised blocking error... skipping pipe."))

        return

    def populate_queue(s: int):
        """Ensures at least s pipes available."""
        requeue_all_free()
        while len(available_pipes) < s:
            logging.debug(f"[Driver] {len(available_pipes) = } < {s = }")
            requeue_all_free()
            sleep(1)

    # Driving loop
    runs_to_test = 9
    for i in range(runs_to_test):
        populate_queue(1)
        pipe = available_pipes.popleft()
        str_pipe = str(pipe)
        logger.info("[Driver] Current pipe = %s", str_pipe)
        pipe.send((b"0", b"Ascending", str(i).encode(), b"0.3500"))
        logger.info("[Driver] Signaled to a VHF process to start.")

        sleep(17)  # hard coded
        assert pipe.recv_bytes() == b'0', \
            f"Error obtained on VHF {i%num_vhf_managers}"

    logger.info("[Driver] Loop is completed!")

    # Close processes and pipes
    populate_queue(len(vhf_pipes))
    _ = [p.send((b'2', b'0', b'0', b'0')) for p, _ in vhf_pipes]
    _ = [j.join() for j in vhfs]
    _ = [(p.close(), q.close()) for p, q in vhf_pipes]

    logging.info("Main is exiting")
    return


if __name__ == "__main__":
    main()
