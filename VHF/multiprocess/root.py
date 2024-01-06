"""Abstract base class for root-managed VHF collection.

During an experiment, some independent variable is changed before the VHF board
is to sample data, write it into permanent storage (likely NAS), and also
analyse the data (ideally from a tmpfs, such as /dev/shm, to avoid additional
disk read/write), and send the analysed data into some congregated dataset.

Common things such as disk I/O can be provided in this generic class, but the
rest need to be implemented on a per-experiment basis.
"""

from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timedelta
import logging
from logging import Logger, getLogger
from multiprocessing import Lock, Process, Pipe
from multiprocessing.connection import Connection
import os
from time import sleep
from typing import Callable, ClassVar, Deque, List

__all__ = [
    "IdentifiedProcess",
]


class IdentifiedProcess:
    """Child Process wrapper.

    Inputs
    ------
    process: Process
        The process being managed by this process wrapper.
    connection: Connection
        The parent end of the connection used to talk to (child) process.

    Class Methods
    -------------
    set_close_delay(delay: timedelta): None
        Number of attempts at close_proc before killing process forcefully is
        spaced out with the delay prescribed here.

    Methods
    -------
    close_proc: bool
        Attempts to join the process, and cleanup relevant objects.
    is_alive: bool
        Determines if process is alive.

    """

    _to_close_delay: ClassVar[timedelta]

    @classmethod
    def set_close_delay(cls, delay: timedelta):
        """Set delay across all instances between Process.join attempts."""
        cls._to_close_delay = delay

    def __init__(self, process: Process, connection: Connection,
                 child_connection: Connection):
        self._createLogger()
        self._process: Process = process
        self._connection: Connection = connection
        self._childconnection: Connection = child_connection
        self._pid: int = None
        self._to_close: bool = False
        self._to_close_last_time: datetime
        self._to_close_attempt: int = 0
        self._closed: bool = False
        self._process.start()
        if self.connection.poll(2):
            retval = self.connection.recv()
        else:
            raise ValueError("Failed to recieve initialisation status.")
        self.logger.debug("retval = %s", retval)
        if type(retval) != tuple or len(retval) != 2:
            raise ValueError(
                f"Unknown initialisation state recieved from {self._process}")
        if retval[0] != 0:
            raise ValueError(f"{self._process} initialised with errors.")
        self.pid = retval[1]

    def _createLogger(self):
        self.logger = getLogger(__name__)

    @property
    def process(self) -> Process:
        """Process object that is being managed by IdentifiedProcess."""
        return self._process

    @property
    def connection(self):
        """Parent end of Pipe used to send message to child process."""
        return self._connection

    @property
    def pid(self):
        """PID associated with running process."""
        return self._pid

    # Setter has to come after
    @pid.setter
    def pid(self, value: int):
        if self._pid is None:
            self._pid = value
        else:
            raise ValueError

    def is_alive(self) -> bool:
        """Determine if the identified process is alive."""
        if self.process._closed:  # This is with reference to Process._check_closed() method
            result = False
            self.logger.debug("self._process was already closed")
        else:
            result = self._process.is_alive()
            self.logger.debug("self.is_alive yields: %s", result)
        return result

    def _join_proc(self):
        try:
            self._process.join(0.1)  # try to close
            if not self.is_alive():
                self._process.close()  # GC release
                self._closed = True
        except TimeoutError:
            self.logger.warning("Failed to close connection.")

    def close_proc(self) -> bool:
        """If process is not joined, attempt to close and join."""
        # Treat _closed as read only as much as possible. _join_proc mutates
        # _closed as necessary.
        if self._closed:
            self.logger.debug("Process was already closed")
            return True
        self.logger.info("Trying to join process")
        if not self._to_close:
            # First time trying to close child process.
            # close pipe on child end without gurantee
            self._connection.send((b'2',))  # This is be specification
            # close pipe from root(parent) end
            self._connection.close()
            self._to_close = True
            self._to_close_last_time = datetime.now()
            self._join_proc()
            return self._closed
        else:
            # Not first time trying to close.
            if self._to_close_last_time + type(self)._to_close_delay\
                    < datetime.now():
                # Try to join again only after delayed interval
                self._to_close_attempt += 1
                if self._to_close_attempt <= 3:
                    self.logger.debug("Calling join proc")
                    self._join_proc()
                    self._to_close_last_time = datetime.now()
                    return self._closed
                else:
                    # Too many attempts, will forcefully kill
                    self.logger.warning("Attempting to forcefully kill.")
                    # os.system(f"kill -9 {self._pid}")
                    self._process.kill()
                    self._childconnection.close()  # Is this enough?
                    sleep(0.2)
                    self._join_proc()
                    return self._closed
            else:
                # No attempt to join again, only check status
                if not self.is_alive():
                    self._closed = True
            return self._closed
