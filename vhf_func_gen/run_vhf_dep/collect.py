"""Provides VHF class that is to be managed by root process."""

from configparser import ConfigParser, ExtendedInterpolation
import matplotlib
from matplotlib import pyplot as plt
from multiprocessing import Queue
import multiprocessing
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Lock as LockType
import numpy as np
from os import PathLike
from pathlib import Path
from shlex import quote
import sys
from time import sleep
from typing import Union
module_path = str(Path(__file__).parents[2])
if module_path not in sys.path:
    sys.path.append(module_path)
from VHF.runner import VHFRunner  # noqa
from VHF.multiprocess.vhf import genericVHF  # noqa
from VHF.parse import VHFparser  # noqa

_PATH = Union[str, PathLike, Path]


class FuncGenExpt(genericVHF):
    """Process for performing 1x sample of VHF and relevant followup."""

    def __init__(self, comm: Connection, q: Queue, conf_path: _PATH, npz_lock: LockType):
        """Perform data collection and analysis as instructed by root.

        Communicated with via VHF Pool; in charge of collecting 1 round of
        data, keeping the data into external storage, communicate to root that
        the collection is done, then process the data and write into the common
        npz file, before awaiting subsequent call.

        Successful initialisation is signaled to root with (0, PID).

        Inputs
        ------
        comm: Connection
            For receiving instructions and sending instructions to root. Refer
            to main_func for message codes.
        q: Queue
            For sharing a queue handler to ensure multiprcess-safe logging.
            Passed into root logger before creating self.logger =
            logging.getLogger(__name__);
        npz_lock: Lock object
            This is a lock create from root that is shared across all VHF
            processes, to ensure that there is no race condition from writing
            to the shared npz file.
        """
        # This function is immediately runned when being started from
        # multiprocess. We need to send back to the main loop that
        # instantization has occured without issue, and pop into the waiting
        # loop.

        # 1. init class specific attributes
        super().__init__(comm, q, conf_path)
        self.npz_lock: LockType = npz_lock  # Acquired prior to writing to common npz
        self.fail_count: int = 0
        # https://github.com/matplotlib/matplotlib/issues/20300#issuecomment-848201196
        matplotlib.use('agg')  # No GUI is being displayed. Saves memory.

        # 2. parse conf file
        self.conf_parse(conf_path)

        self.vhf_runner = VHFRunner(conf_path)

        # 3. No issue found, drop into core loop.
        self.logger.info("Initialisation complete.")
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

    def conf_parse(self, conf_path: _PATH):
        """Initialise FuncGenExpt with properties from config file."""
        self.logger.debug(
            "Reading from configuration file, path: %s", conf_path)
        self.conf = ConfigParser(interpolation=ExtendedInterpolation())
        self.conf.read(conf_path)
        self.save_dir = Path(self.conf.get("Paths", "save_dir"))
        self.npz_path = Path(self.conf.get("Paths", "collated_npz_save"))
        self.FAIL_MAX = self.conf.getint("Multiprocess", "max_attempts")
        self.ignore_first = self.conf.getint(
            "Multiprocess", "parse_ignore_left")

    def main_func(self):
        """Awaits and acts on instructions from root process.

        We create the protocol where child processes recieves all messages
        within tuples. data = (action, _).
        1. action = b'0'; (b'0', p: mapping, npz_loc: tuple)
            Signals to continue sampling once.
            - p here is treated as mapping p to be passed to
              VHFRunner._overwrite_attr to obtain relevant file name for
              saving.
            - npz_loc here is passed into analyse_parse, to guide where in the
              common npz_file to be written to. Used directly as a slice, i.e.:
              npz_file["arr_name"][npz_loc] = new value
        2. action = b'2'; (b'2',)
            Signals to gracefully terminate process.

        Messages sent back:
        1. b'0':
            Signals success of data collection. Do not send again for
            successful data analysis.
        2. b'1':
            Generic error. Possible situations:
            a. Tempfile created could not be opened, and file from NAS could
            also not be found.
        3. b'2':
            Recieved SIGINT. Propagate up.
        4. b'3':
            Repeated failed attempts to sample data.
        5. (0, PID):
            Initialisation success, along with PID of child process.
            Relevant: multiprocessing.active_children()


        - During SIGINT, we aim to close all child processes gracefully, and
          only have the root process exit with exit_code 2.
        - In the even main_func fails to be able to collect the data after ?
          times, we propagate the error up and aim to close everything if
          forceful, otherwise only the existing VHF child process, and have the
          root process create a new child. This should be solely managed in
          root, and not for the child VHF process to handle.
        """
        self.logger.info("Now in main_func awaiting.")
        try:
            # This step is blocking until self.comm receives
            while (data := self.comm.recv()):
                self.logger.debug("Recieved %s", data)
                action = data[0]
                match action:
                    case b'0':
                        # Update how vhf_runner property will be like for
                        # sample_once to use.
                        msg = data[1]
                        self.vhf_runner._overwrite_attr(msg)

                        # Start sample.
                        self.logger.info("Sampling now!")
                        while not self.sample_once(
                            perm_target=(pname := quote(str(
                                self.save_dir.joinpath(self.vhf_runner.get_filename(
                                    self.vhf_runner.get_params()))
                            )))
                        ):
                            self.logger.warn("Failed to run VHF.")
                            self.fail_count += 1
                            if self.fail_count >= self.FAIL_MAX:
                                self.logger.error(
                                    "Exceeded allowable number of failed "
                                    "attempts trying to sample out of VHF "
                                    "board. Terminating...")
                                self.comm.send_bytes(b'3')
                                self.close()
                            sleep(0.2)
                        # Now hand off for data analysis, and writing to npz
                        # file.
                        parsed = self.get_parsed(pname)
                        if parsed is None:
                            self.logger.error(
                                "File could not be obtained for analysis.")
                            # failed to obtain files!
                            self.comm.send_bytes(b'1')  # Generic error
                            self.tmp_close()  # Cleanup
                            continue

                        # Release tmp early.
                        self.tmp_close()

                        # Analyse and wrap up.
                        self.analyse_parse(parsed, data[2], pname)
                        self.fail_count = 0
                        self.logger.info("Analysis and plot complete!")

                    case b'2':
                        self.logger.info("Recieved closing signal.")
                        self.close()

                    case _:
                        self.logger.warning("Unknown message recieved!")
                        self.exit_code = 1
                        self.close()

        except KeyboardInterrupt:
            self.logger.warn(
                "Recieved KeyboardInterrupt Signal! Attempting to propagate shutdown gracefully.")
            self.comm.send(b'2')
            self.close()

    def analyse_parse(self, parsed: VHFparser, npz_loc: tuple, pname: str):
        """Take parsed object and saves relevant bits into common npz_file."""
        self.logger.debug("npz_loc = %s", npz_loc)

        # Parsed data -> Relevant information
        radius = parsed.radii[self.ignore_first:]
        a = np.mean(radius)
        b = np.std(radius)
        c = np.min(radius)

        # Save data
        self.logger.debug("Trying to acquire lock.")
        self.npz_lock.acquire()
        self.logger.info("Acquired npz file and lock.")
        f = np.load(self.npz_path)
        f = dict(f)  # this step is impt, as NpzFile is immutable
        f["mean"][npz_loc] = a
        f["std_dev"][npz_loc] = b
        f["min"][npz_loc] = c

        np.savez(
            self.npz_path,
            mean=f["mean"],
            std_dev=f["std_dev"],
            min=f["min"],
        )
        self.npz_lock.release()
        self.logger.info("Written to npz file, and released lock.")

        # Save plot
        image_path = self.save_dir.joinpath(
            Path(pname).stem + ".png").resolve()
        # proper trim left has not been implemented, so import of plotting
        # is not quite suitable
        self.save_plot(quote(str(image_path)), parsed)

    def save_plot(self, fname: _PATH, parsed: VHFparser) -> None:
        """Save plot while still in memory."""
        def scatter_hist(y, ax_histy):
            binwidth = (np.max(y) - np.min(y)) / 100
            bins = np.arange(np.min(y) - binwidth,
                             np.max(y) + binwidth, binwidth)
            ax_histy.hist(y, bins=bins, orientation="horizontal")

        def plotphi(ax, o: VHFparser):
            t = np.arange(len(o.data)) / o.header["sampling freq"]
            ax.plot(t[self.ignore_first:], o.reduced_phase[self.ignore_first:],
                    color="mediumblue", linewidth=0.2, label="Phase")
            # ax.plot(-parsed.m_arr, color='red', linewidth=0.2, label='Manifold')
            ax.set_ylabel(r"$\phi_d$/2$\pi$rad", usetex=True)
            ax.set_xlabel(r"$t$/s", usetex=True)

        def plotr(r_ax, o: VHFparser):
            t = np.arange(len(o.data)) / o.header["sampling freq"]
            r_ax.plot(
                t[self.ignore_first:],
                rs := o.radii[self.ignore_first:],
                color="mediumblue",
                linewidth=0.2,
                label="Radius",
            )
            r_ax.set_ylabel(r"$r$/ADC units", usetex=True)
            r_ax.set_xlabel(r"$t$/s", usetex=True)
            r_ax_histy = r_ax.inset_axes([1.01, 0, 0.08, 1], sharey=r_ax)
            r_ax_histy.tick_params(axis="y", labelleft=False, length=0)
            scatter_hist(rs, r_ax_histy)
        # fig, [phi_ax, r_ax] = plt.subplots(nrows=2, ncols=1, sharex=True)
        # https://stackoverflow.com/a/65910539
        fig = plt.figure(num=1, clear=True)
        phi_ax = fig.add_subplot(2, 1, 1)
        r_ax = fig.add_subplot(2, 1, 2, sharex=phi_ax)
        plotphi(phi_ax, parsed)
        plotr(r_ax, parsed)
        view_const = 2.3
        fig.legend()
        fig.set_size_inches(view_const * 0.85 *
                            (8.25 - 0.875 * 2), view_const * 2.5)
        fig.tight_layout()
        fig.savefig(fname, dpi=300, format="png", transparent=True)
        # plt.close(fig)
        self.logger.debug("Plot completed and saved.")
