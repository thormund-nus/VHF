from datetime import timedelta
from functools import cache
from inspect import getmembers, isroutine
import logging
from logging import getLogger, Logger
from time import sleep
from multiprocessing import Queue
from pathlib import Path
import pytest
import sys
from typing import Callable, Mapping
module_path = str(Path(__file__).parents[1])
if module_path not in sys.path:
    sys.path.append(module_path)
from test_IdentifiedProcess import GenericChild  # noqa
from VHF.multiprocess.root import IdentifiedProcess  # noqa
from VHF.multiprocess.signals import Signals, ChildSignals  # noqa
from VHF.multiprocess.vhf import genericVHF  # noqa
from VHF.multiprocess.vhf_pool import VHFPool  # noqa


class Regular(GenericChild):
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
                    sleep(0.4)
                    # signal to let next child proceeed
                    self.comm.send_bytes(c_sig.action_cont)
                    # signal to request requeue
                    sleep(0.2)
                    self.comm.send_bytes(c_sig.action_request_requeue)
                case sig.action_hup:
                    break
        self.close()


class BadA(GenericChild):
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


class BadB(GenericChild):
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
    IdentifiedProcess.set_close_delay(timedelta(0.4))
    # Define properties necessary for init of VHFPool instance
    c_sig = ChildSignals()
    fail_forward = {
        c_sig.action_generic_error: True,
        c_sig.too_many_attempts: False,
    }
    logger_q = Queue
    pool = VHFPool(
        fail_forward,
        3,
        Regular,
        logger_q,
    )
    for _ in range(9):
        pool.continue_child("WORK")
        sleep(0.5)

    pool._close_all()
    pool.close()
