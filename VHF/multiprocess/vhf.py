"""Abstract base class for root-managed VHF collection.

During an experiment, some independent variable is changed before the VHF board
is to sample data, write it into permanent storage (likely NAS), and also
analyse the data (ideally from a tmpfs, such as /dev/shm, to avoid additional
disk read/write), and send the analysed data into some congregated dataset.

Common things such as disk I/O can be provided in this generic class, but the
rest need to be implemented on a per-experiment basis.
"""

from abc import ABCMeta
from datetime import datetime
import logging
from logging import getLogger
from logging.handlers import QueueHandler
from multiprocessing import Queue
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Lock as LockType
import os
from os import PathLike
from pathlib import Path
from shlex import quote
import subprocess
from subprocess import PIPE
import sys
from tempfile import NamedTemporaryFile
from tempfile import _TemporaryFileWrapper
from typing import Optional, Union
from ..metatype import abstract_attribute
from ..runner import VHFRunner
from ..parse import VHFparser

__all__ = [
    "genericVHF",
]

_PATH = Union[str, Path, PathLike]


class genericVHF(metaclass=ABCMeta):
    """genericVHF ABC for managing concrete instance of VHF.

    This is the generic class with common methods that can be inherited from
    across multiple experiments. The concrete implementation should then be
    able to fetch data out the VHF board and process said data in a manner that
    does not become CPU blocking during data analysis, while having the next
    multiprocess (concrete) instance perform the data sampling. w
    """

    def __init__(self, comm: Connection, q: Queue, vhf_conf_path: _PATH):
        """Provisions generic methods associated to VHF collection of traces.

        Inputs
        ------
        comm: multiprocessing.connection.Connection
           Pipe that allows child process here to send state of this process
           back to parent
        q: Queue
            For sharing a queue handler to ensure multiprcess-safe logging.
            Passed into root logger before creating self.logger =
            logging.getLogger(__name__);
        """
        # 1. logging!
        self.init_log(q)
        # class definitions
        self.exit_code: int = 0  # Exit code for sys.exit()
        self.comm: Connection = comm
        self.vhf_conf_path: _PATH = vhf_conf_path

    def close(self):
        """Close up class that pulls data from VHF."""
        self.logger.info("Initiating Closing...")
        # Does q from parameters of __init__ require to be closed too?
        self.comm.close()
        self.logger.info("Closed child VHF process end of Pipe().")
        self.logger.info("Closed with exit_code %s", -self.exit_code)
        sys.exit(self.exit_code)

    def __exit__(self):
        self.close()

    def init_log(self, q: Queue) -> None:
        """Set up logger in a multiprocess-safe manner."""
        r = getLogger()  # assume that root logger settled by root thread as being
        qh = QueueHandler(q)
        if not r.handlers:
            r.addHandler(qh)
        self.logger = getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    def sample_once(self, perm_target: _PATH) -> bool:
        """
        Take one copy of VHF trace data, and saves onto both NAS and tmpfs.

        Goals
        -----
        Signals to core_loop if sampling with tee succeeds with True, or if
        error was obtained during the sampling process, by returning False.
        Core loop should follow up by performing analysis of self._tmp copy.
        We leave for the core_loop to then take perform the data processing and
        cleanup.

        Inputs
        ------
        perm_targt: _PATH
            Permanent location for sampled data to be saved into. This should
            be file location on a permanent storage location.

        Outputs
        -------
        bool:
            Signals if data was collected with success and piped into NAS
        """
        # VHF subprocess is to write to STDOUT. Tee into perm_target, STDOUT of
        # tee goes into NamedTemporaryFile
        self.logger.info(
            "sample_once has been called. perm_target = %s",
            perm_target)
        self._tmp_open()
        self.logger.debug("perm_target = %s", perm_target)
        self.logger.debug("self._tmp.closed = %s", self._tmp.closed)
        self.logger.debug("self._tmp = %s", self._tmp)

        try:
            start_time_sample = datetime.now()
            with open(self._tmpName, "wb") as f:
                with subprocess.Popen(
                    **(sb_run := self.vhf_runner.subprocess_Popen()),
                    stdout=PIPE, stderr=PIPE
                ) as vhf:

                    # vhf.communicate(timeout=self.vhf_runner.sample_time()+3)
                    with subprocess.Popen(
                        ["tee", "-a", quote(str(perm_target))],
                        stdin=vhf.stdout, stdout=self._tmp
                    ) as retcode:
                        self.logger.debug("Retcode: %s", retcode)

            end_time = datetime.now()
            self.comm.send_bytes(b'0')
            self.logger.debug("subprocess.run had arguments: %s", str(sb_run))
            self.logger.debug("subprocess.run took time: %s",
                              end_time - start_time_sample)
            self.logger.info("Sampling success.")
            return True
        # except KeyboardInterrupt:
        #     We now handle this in the concrete implementation of genericVHF.
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            self.logger.info("Subprocess ran with %s", str(sb_run))
            self.logger.critical("exc: %s", exc)
            self.logger.critical("exc.stderr = ", exc.stderr)
            self.logger.critical("", exc_info=True)
            if isinstance(exc, subprocess.CalledProcessError):
                self.logger.critical(
                    f"Process returned with error code {exc.returncode}")
            return False
        # else:
        # No error occured during sampling
        # Everything here and lower needs  to move to core_loop
        #     self.parse_data()
        #     self.write_to_collated()
        # finally:
        #     self._tmp_close()

    def _tmp_open(self) -> None:
        """For use by self.sample_once(). Creates a temporary file."""
        self._tmp: _TemporaryFileWrapper = NamedTemporaryFile(
            dir="/dev/shm", delete=False)
        self._tmpName: str = self._tmp.name
        self.logger.debug("Opened NamedTemporaryFile: %s", self._tmpName)

    def tmp_close(self) -> None:
        """Cleanup temporary file."""
        self._tmp.close()
        self._tmp = None  # Free variable up if something goes wrong
        self.logger.debug("Closed NamedTemporaryFile.")

        try:
            self.logger.debug("Deleting temporary file...")
            os.unlink(self._tmpName)
            self.logger.debug("Deleted temporary file...")
        except FileNotFoundError:
            self.logger.warning("Deleted File was not in tmp dir.")
        except Exception as e:
            self.logger.error("Exception encounterd: %s", e)
            self.logger.error("", exc_info=True)

        self._tmpName = None

    def get_parsed(self, perm_target: str) -> Optional[VHFparser]:
        """Binary file obtained from stdout of VHF is parsed here.

        Tries to read data out from self._tmp, analyze, and then save into the
        common npz file. Reads from perm_target if self._tmp is missing. Raises
        b'1' error as stated in main_func() if neither could be found.

        Inputs
        ------
        perm_target: str
            sample_once would have taken duplicated the STDOUT from the
            VHFboard into (self.tmp, perm_target). This is the same
            perm_target.
        """
        def get_data(perm_target: str) -> Optional[VHFparser]:
            for attempt in (self._tmpName, perm_target):
                try:
                    parsed = VHFparser(attempt)
                    return parsed
                except FileNotFoundError:
                    self.logger.warning(
                        "VHFParser could not find %s. ", attempt)
                    continue
        return get_data(perm_target)

    # This requires that we use ABCMeta instead of ABC for class inheritance
    @abstract_attribute
    def vhf_runner(self) -> VHFRunner:
        """VHFRunner object."""
        raise NotImplementedError
