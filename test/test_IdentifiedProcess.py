from abc import abstractmethod
from datetime import datetime, timedelta
import logging
from logging import getLogger
from logging.handlers import QueueHandler
import multiprocessing
from multiprocessing import Pipe, Process, Queue
from multiprocessing.connection import Connection
import os
from pathlib import Path
import pytest
import sys
from threading import Thread
from time import sleep
module_path = str(Path(__file__).parents[1])
if module_path not in sys.path:
    sys.path.append(module_path)
from VHF.multiprocess.root import IdentifiedProcess  # noqa
from VHF.multiprocess.signals import cont, HUP  # noqa


class GenericChild:
    # define a generic child
    def __init__(self, comm: Connection, q: Queue):
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
        try:
            self.comm.send((0, self.pid))
            self.logger.debug("Initialisation sent!")
        except BrokenPipeError as e:
            self.logger.warning("Failed to sent initialisation!")
            self.logger.critical("exc = %s", e, exc_info=True)
            self.exit_code = 1
            self.close()
        self.main_func()

    def init_log(self, q):
        r = getLogger()  # assume that root logger settled by root thread as being
        qh = QueueHandler(q)
        if not r.handlers:
            r.addHandler(qh)
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

    def close(self):
        """Close up class that pulls data from VHF."""
        self.logger.info("Initiating Closing...")
        # Does q from parameters of __init__ require to be closed too?
        self.comm.close()
        self.logger.info("Closed child VHF process end of Pipe().")
        self.logger.info("Closed with exit_code %s", -self.exit_code)
        sys.exit(self.exit_code)


class Regular(GenericChild):
    def main_func(self):
        class Signals:
            action_cont = cont()[0]
            action_hup = HUP[0]

        signal = Signals()
        while (data := self.comm.recv()):
            action = data[0]
            match action:
                case signal.action_cont:
                    msg = data[1]
                    self.logger.info(
                        "Child is doing busy work with msg = %s", msg)
                    sleep(0.3)
                case signal.action_hup:
                    self.close()

class DelayedGeneric(GenericChild):
    def __init__(self, comm: Connection, q: Queue):
        self.init_log(q)
        self.comm: Connection = comm
        self.pid: int | None = multiprocessing.current_process().pid
        self.logger.debug("Obtained PID %s.", self.pid)
        self.exit_code = 0
        if self.pid is None:
            # This really should not be occuring!
            self.logger.error("Failed to obtain a PID for child process.")
            self.comm.send_bytes(b'1')
            self.exit_code = 1
            self.close()
        sleep(1.2)
        try:
            self.comm.send((0, self.pid))
            self.logger.warning("BAD: Initialisation sent!")
        except BrokenPipeError as e:
            self.logger.warning("GOOD: Failed to sent initialisation!")
            self.logger.critical("exc = %s", e, exc_info=True)
            self.exit_code = 1
        self.close()
        # self.main_func()

class DelayedRegular(DelayedGeneric, Regular):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)


def test_IdentifiedProcess_classvariable():
    """Ensure that set_close_delay on IdentifiedProcess works."""
    expected = timedelta(seconds=1)
    IdentifiedProcess.set_close_delay(expected)

    q = Queue()
    sink, src = Pipe()
    instance = IdentifiedProcess(
        process=Process(target=GenericChild, args=(src, q)),
        connection=sink,
    )
    assert type(instance)._to_close_delay == IdentifiedProcess._to_close_delay \
        and type(instance)._to_close_delay


def test_unresponsive_child_process():
    """In the event child process is stuck ad infinitum, ensure that identified
    process is capable of closing child process."""
    # define a bad class first
    class Bad(GenericChild):
        def main_func(self):
            while True:
                self.logger.info("Bad is sleeping!")
                sleep(2)

    # start of test
    IdentifiedProcess.set_close_delay(timedelta(seconds=0.5))
    IdentifiedProcess.set_process_name("BadChild")
    q = Queue()
    sink, src = Pipe()

    badproc = IdentifiedProcess(
        Process(
            target=Bad,
            args=(src, q),
        ),
        sink,
    )
    logging.info("badproc.pid = %d", badproc.pid)
    for _ in range(6):
        retval = badproc.close_proc()
        if retval:
            assert not badproc.is_alive()
        sleep(0.6)

    assert not badproc.is_alive()
    badproc.close()


def test_regular_child_process():
    """In the event child process is stuck ad infinitum, ensure that identified
    process is capable of closing child process."""
    # start of test
    IdentifiedProcess.set_close_delay(timedelta(seconds=0.5))
    IdentifiedProcess.set_process_name("RegularChild")
    q = Queue()
    sink, src = Pipe()

    child = IdentifiedProcess(
        Process(
            target=Regular,
            args=(src, q),
        ),
        sink,
    )
    logging.info("child.pid = %d", child.pid)
    child.connection.send(cont('1'))
    while not child.close_proc():
        sleep(0.6)
    assert not child.is_alive()
    child.close()


def test_logger_freeup():
    """Ensure that creation of multiple processes do not leave a mess."""
    logger = getLogger()
    IdentifiedProcess.set_close_delay(timedelta(seconds=0.5))
    IdentifiedProcess.set_process_name("RegularChild")
    childs = []
    q = Queue()
    for _ in range(10):
        sink, src = Pipe()
        childs.append(
            IdentifiedProcess(
                Process(
                    target=Regular,
                    args=(src, q),
                ),
                sink,
            )
        )
    logger.info("Created children")
    # for c in childs:
    #     c.connection.send(cont('1'))
    # sleep(0.6)

    for c in childs[:]:
        c.close_proc()
        childs.remove(c)
        c.close()

    import gc
    gc.collect()

    # https://stackoverflow.com/a/53250066
    loggers = [logging.getLogger(name)
               for name in logging.root.manager.loggerDict]
    logger.info("loggers = %s", loggers)
    # logger.info('filter(lambda x: "05" in str(logging.getLogger(x), logging.root.manager.loggerDict) = %s', list(
    #     filter(lambda x: "05" in str(logging.getLogger(
    #         x)), logging.root.manager.loggerDict))
    # )
    assert not any(filter(
        lambda x: "09" in str(logging.getLogger(x)),
        logging.root.manager.loggerDict
    ))


def test_broken_parent_mock():
    """Ensure that the mock class used for the next test has appropriate MRO."""
    logging.info("DelayedRegular.mro() = %s", DelayedRegular.mro())
    assert DelayedRegular.main_func == Regular.main_func

def test_broken_parent_process():
    """Parent's pipe might be broken before child gets to init. Child should
    just terminate as such. Faulty test not properly replicating Errno 32
    Broken Pipe."""
    # start of test
    IdentifiedProcess.set_close_delay(timedelta(seconds=1.0))
    IdentifiedProcess.set_process_name("BrokenParent")
    q = Queue()
    sink, src = Pipe()
    def pipe_killer_fn(p: Connection):
        sleep(0.15)
        logging.info("Parent process crashed pipe!")
        p.close()
    def pipe_checker_fn(p: Connection):
        sleep(0.25)
        logging.info("pipe closed? = %s", p._close)

    pipe_killer = Thread(
        target=pipe_killer_fn,
        args=(sink,)
    )
    pipe_checker = Thread(
        target=pipe_checker_fn,
        args=(sink,)
    )

    pipe_killer.start()
    pipe_checker.start()

    with pytest.raises((BrokenPipeError, OSError)) as e:
        child = IdentifiedProcess(
            Process(
                target=DelayedRegular,
                args=(src, q),
            ),
            sink,
        )  # this is a blocking step, closing of sink needs to be done between
        # the start and end of this...
        # pytest scope ends here as error was obtained
    sleep(1.6)
    logging.info("sink.readable = %s", sink.readable)
    # assert not child.is_alive()
    # child.close_proc()
    # child.close()
    logging.info("pipe_killer.is_alive() = %s", pipe_killer.is_alive())
    # pipe_killer.join()
