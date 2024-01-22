from abc import abstractmethod
from datetime import timedelta
from functools import cache
from inspect import getmembers, isroutine
import logging
from logging import getLogger, Logger
from logging.handlers import QueueHandler
from threading import Thread
from time import sleep
import multiprocessing
from multiprocessing import Queue
from multiprocessing.connection import Connection
from pathlib import Path
import pytest
import sys
from typing import Callable, Mapping
module_path = str(Path(__file__).parents[1])
if module_path not in sys.path:
    sys.path.append(module_path)
from VHF.multiprocess.root import IdentifiedProcess  # noqa
from VHF.multiprocess.signals import HUP, Signals, ChildSignals  # noqa
from VHF.multiprocess.vhf import genericVHF  # noqa
from VHF.multiprocess.vhf_pool import VHFPool  # noqa

# multiprocessing.set_start_method('spawn')


def log_listener(q: Queue):
    """Record logs across multiple processes."""
    while True:
        record = q.get()
        if record == HUP:
            logger = getLogger()
            logger.info("[Listener] Recieved HUP")
            break
        logger = logging.getLogger(record.name)
        logger.handle(record)
    return


class GenericChildMultiProcess():
    """Generic Child Process that is not threaded for this unit test file."""

    def __init__(self, comm: Connection, q: Queue):
        """Initializer for all children in this unit test file."""
        self.init_log(q)
        self.comm: Connection = comm
        self.pid: int | None = multiprocessing.current_process().pid
        self.logger.debug("Obtained PID %s.", self.pid)
        if self.pid is None:
            # This really should not be occuring!
            self.logger.error("Failed to obtain a PID for child process.")
            self.comm.send_bytes(b'1')
            self.exit_code = 1
            self.close()
        self.comm.send((0, self.pid))
        self.main_func()

    def init_log(self, q: Queue):
        """Create logger."""
        # https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes
        rootLogger = getLogger()
        if not rootLogger.handlers:
            rootLogger.addHandler(QueueHandler(q))
        self.logger = getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    def close(self):
        """Close up class that pulls data from VHF."""
        self.logger.info("Initiating Closing...")
        # Does q from parameters of __init__ require to be closed too?
        self.comm.close()
        self.logger.info("Closed child VHF process end of Pipe().")
        self.logger.info("Closed with exit_code %s", -self.exit_code)
        sys.exit(self.exit_code)

    @abstractmethod
    def main_func(self):
        raise NotImplementedError


class Regular(GenericChildMultiProcess):
    def main_func(self):
        self.exit_code = 0
        c_sig = ChildSignals()
        sig = Signals()

        while data := self.comm.recv():
            action = data[0]
            match action:
                case sig.action_cont:
                    msg = data[1]
                    self.logger.info(
                        "Child is doing busy work with msg = %s", msg)
                    sleep(1.4)
                    # signal to let next child proceeed
                    self.comm.send_bytes(c_sig.action_cont)
                    # signal to request requeue
                    sleep(0.6)
                    self.comm.send_bytes(c_sig.action_request_requeue)
                case sig.action_hup:
                    break
        self.close()


class BadA(GenericChildMultiProcess):
    """Fails to release to next child."""

    def main_func(self):
        c_sig = ChildSignals()
        sig = Signals()

        while data := self.comm.recv():
            action = data[0]
            match action:
                case sig.action_cont:
                    msg = data[1]
                    self.logger.info(
                        "Child is STUCK on work with msg = %s", msg)
                    while True:
                        sleep(10)
                case sig.action_hup:
                    break


class BadB(GenericChildMultiProcess):
    """Fails to requeue."""

    def main_func(self):
        c_sig = ChildSignals()
        sig = Signals()

        while data := self.comm.recv():
            action = data[0]
            match action:
                case sig.action_cont:
                    msg = data[1]
                    self.logger.info(
                        "Child is doing busy work with msg = %s", msg)
                    sleep(0.2)
                    # signal to let next child proceeed
                    self.comm.send_bytes(c_sig.action_cont)
                    # signal to request requeue
                    sleep(0.15)
                    self.comm.send_bytes(c_sig.action_request_requeue)
                case sig.action_hup:
                    break
        self.close()


# @pytest.mark.filterwarnings("ignore::PytestCollectionWarning")
class VHFPool_init_check_failforward(VHFPool):
    """Subset of VHFPool for testing if to specification."""
    # dunder had to be set to signal to pytest not to check since class name used to start with Test
    # __test__ = False

    def __init__(
        self,
        fail_forward: Mapping[ChildSignals.type, bool] = None,
        count: int = 3,
        target: Callable = None,
        *args,
        **kwargs
    ) -> None:
        self.logger: Logger = getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.fail_forward = fail_forward

        self.signals = Signals()
        self.c_sig = ChildSignals()


# @cache
def fail_forward_default():
    attributes = list(filter(
        lambda x: not (x[0].startswith('__') and x[0].endswith('__')),
        getmembers(ChildSignals, lambda x: not (isroutine(x)))
    ))
    defn_not_req = ['type', 'action_cont',
                    'action_request_requeue', 'action_hup']
    testing_keys = set(map(lambda x: x[0], attributes)) - set(defn_not_req)
    attributes = dict(filter(lambda x: x[0] in testing_keys, attributes))
    test_map = dict([(ChildSignals().__getattribute__(x), False)
                     for x in testing_keys])
    logging.debug("attributes = %s", attributes)
    logging.debug("test_map = %s", test_map)
    return attributes, test_map


def test_VHFPool_failforward_map_eq():
    """Checks if _init_checks_fail_forward is to expectation."""
    _, test_map = fail_forward_default()

    testVHF = VHFPool_init_check_failforward(
        fail_forward=test_map
    )
    result = testVHF._init_checks_fail_forward()
    logging.debug("result = %s", result)
    expected = dict()
    assert result == expected


# @pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_VHFPool_failforward_map_subset():
    """Checks if _init_checks_fail_forward is to expectation."""
    attr, test_map = fail_forward_default()
    missing = list(test_map)[0]
    del test_map[missing]
    logging.debug("removed key = %s", missing)

    testVHF = VHFPool_init_check_failforward(
        fail_forward=test_map
    )
    result = testVHF._init_checks_fail_forward()
    logging.debug("result = %s", result)
    expected = {'ChildSignals': [(k, v)
                                 for k, v in attr.items() if v == missing]}
    assert result == expected


def test_VHFPool_failforward_map_supset():
    """Checks if _init_checks_fail_forward is to expectation."""
    _, test_map = fail_forward_default()
    excess = b'NONSENSE'
    test_map[excess] = False
    logging.debug("added key = %s", excess)

    testVHF = VHFPool_init_check_failforward(
        fail_forward=test_map
    )
    result = testVHF._init_checks_fail_forward()
    logging.debug("result = %s", result)
    expected = {'fail_forward': [excess]}
    assert result == expected


def test_VHFPool_regular():
    """When child processes are behaving to expectation, ensure that Pool can
    behave to requirement."""
    # Specify IdentifiedProcess classvar
    IdentifiedProcess.set_process_name("Regular")
    IdentifiedProcess.set_close_delay(timedelta(1.4))
    # Setup logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    streamhandler = logging.StreamHandler(sys.stdout)
    streamhandler.setLevel(logging.DEBUG)
    fmtter = logging.Formatter(
        # the current datefmt str discards date information
        '[%(asctime)s%(msecs)d] (%(levelname)s)\t[%(processName)s] %(name)s: \t %(message)s', datefmt='%H:%M:%S:'
    )
    fmtter.default_msec_format = "%s.03d"
    streamhandler.setFormatter(fmtter)
    # streamhandler.addFilter(no_matplot)
    logger.addHandler(streamhandler)
    logger_q: Queue = Queue()
    lp = Thread(  # listening process
        target=log_listener,
        args=(logger_q,),
    )
    lp.start()

    # Define properties necessary for init of VHFPool instance
    c_sig = ChildSignals()
    fail_forward = {
        c_sig.action_generic_error: True,
        c_sig.too_many_attempts: False,
        # b'unknown case': True,
    }
    pool = VHFPool(
        fail_forward,
        3,
        Regular,
        logger_q,
    )
    for _ in range(9):
        logger.info("About to start next iteration")
        pool.continue_child("WORK")
        logger.info("Iteration completed.")

    sleep(2)
    pool._close_all()
    logger.info("Root sees that pool has closed all children threads.")
    pool.close()
    logger.info("Root sees that pool is closed.")
    sleep(0.1)
    logger.info("HUP to listener thread")
    logger_q.put(HUP)
    lp.join()
