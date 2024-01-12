"""
To test if multicore is viable.
"""
from collections import deque
import configparser
import datetime
import logging
from logging.handlers import QueueHandler
import multiprocessing as mp
from multiprocessing import Process, Pipe, Queue, get_logger
from multiprocessing.connection import Connection
import os
from os import PathLike
from pathlib import Path
from queue import Empty
import queue
import sys
import subprocess
from threading import Thread
from time import sleep
from typing import Deque, Union

module_path = str(Path(__file__).parents[1].joinpath("20230902 Moment Record"))
if module_path not in sys.path:
    sys.path.append(module_path)
module_path = str(Path(__file__).parents[3].joinpath("20230902 Moment Record"))
if module_path not in sys.path:
    sys.path.append(module_path)

from test_vhf_thread import vhf_manager  # noqa
from moment_multiprocess.vhf_output_analysis import vhf_output_analysis as analysis  # noqa
from VHF.multiprocess.signals import HUP  # noqa


def log_listener(q):
    while True:
        record = q.get()
        if record == HUP:
            break
        logger = logging.getLogger(record.name)
        logger.handle(record)
        # try:
        #     record = q.get_nowait()
        #     if record == HUP:
        #         break
        #     logger = logging.getLogger(record.name)
        #     logger.handle(record)
        # except Empty:
        #     pass
        # except Exception as exc:
        #     logger = logging.getLogger()
        #     logger.critical("exc= %s", exc)
        #     logger.critical("%s", exc_info=True)
    return


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
    logger = logging.getLogger()
    # base logger has to be lower level than handlers
    logger.setLevel(logging.DEBUG)

    filehandler = logging.FileHandler(
        filename=f"{Path(__file__).parents[2].joinpath('Log').joinpath(
            'test-multi_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.log')}",
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
    # thread safe, we ignore logging via Queue
    logg_q = Queue()
    # lp = Thread(
    #     target=log_listener,
    #     args=(logg_q,)
    # )
    # lp.start()

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
        assert pipe.recv_bytes() == b'0', f"Error on VHF process: {pipe =}"

    # Queue of available pipes
    available_pipes = deque([p for p, _ in vhf_pipes], maxlen=len(vhf_pipes))
    # print(f"[Debug] {id(available_pipes[0]) = }")
    # print(f"[Debug] {id(vhf_pipes[0][0]) = }")
    # print(f"[Debug] {available_pipes = }")
    # print(f"[Debug] {vhf_pipes = }")

    class Requeue(Thread):
        def __init__(self, q: Queue, requeue_comm: queue.Queue,
                     all: list[tuple[Connection, Connection]], avail) -> None:
            super().__init__()
            self.logger = logging.getLogger("Requeue")
            # qh = QueueHandler(q)
            # if not self.logger.handlers:
            #     self.logger.addHandler(qh)
            self.comm = requeue_comm
            # self.logger.info("requeue_comm added")
            self.all: list[tuple[Connection, Connection]] = all
            self.avail: Deque = avail
            # self.logger.info("Requeue init completed; Entering requeue_loop")
            # self.start()
            # self.requeue_loop()
            return

        def run(self):
            self.requeue_loop()

        def requeue_one(self, p):
            try:
                # self.logger.info("requeue_one(p) called")
                if p.poll():
                    self.logger.debug("Trying to requeue %s", p)
                    msg = p.recv_bytes()
                    if msg != b'0':
                        self.logger.warning(
                            "[Driver] Recieved issue on pipe.")

                    self.avail.append(p)
                    self.logger.info("[Driver] Requeued a pipe.")
                    self.logger.info("[Driver] avail = %s", self.avail)
                    self.logger.info("[Driver] all = %s", self.all)
            except BlockingIOError:
                self.logger.warning(
                    "BlockingIO trying to requeue %s. Skipping.", p)

        def requeue_loop(self):
            self.logger.info("Entered requeue_loop")
            while True:
                # Driving loop
                try:
                    record = self.comm.get(timeout=0.01)
                    if record == HUP:
                        for p, _ in self.all:
                            self.requeue_one(p)
                        break
                except Empty:
                    pass
                except Exception as exc:
                    self.logger.critical("exc= %s", exc)
                    # self.logger.critical("exc.stderr= %s", exc.stderr)
                    self.logger.critical("%s", exc_info=True)
                for p, _ in self.all:
                    # self.logger.info("Rotating through self.all")
                    if p not in self.avail:
                        self.requeue_one(p)

                sleep(0.3)
    requ_q = queue.Queue()
    logger.info("requ_q created")
    # r_thread = Thread(
    #     target=Requeue,
    #     args=(
    #         logg_q,
    #         requ_q,
    #         vhf_pipes,
    #         available_pipes,
    #     ),
    # )
    # r_thread.start()
    r_thread = Requeue(
        logg_q,
        requ_q,
        vhf_pipes,
        available_pipes,
    )
    r_thread.start()

    runs_to_test = 9
    for i in range(runs_to_test):
        pipe = r_thread.avail.popleft()
        str_pipe = str(pipe)
        logger.info("[Driver] Current pipe = %s", str_pipe)
        pipe.send((b"0", b"Ascending", str(i).encode(), b"0.3500"))
        logger.info("[Driver] Signaled to a VHF process to start.")

        sleep(17)  # hard coded
        assert pipe.recv_bytes() == b'0', \
            f"Error obtained on VHF {i % num_vhf_managers}"

    logger.info("[Driver] Loop is completed!")

    # Close processes and pipes
    requ_q.put(HUP)
    r_thread.join()
    _ = [p.send((b'2', b'0', b'0', b'0')) for p, _ in vhf_pipes]
    _ = [j.join() for j in vhfs]
    _ = [(p.close(), q.close()) for p, q in vhf_pipes]

    logg_q.put(HUP)
    # lp.join()
    logger.info("Main is exiting")
    return


if __name__ == "__main__":
    main()
