"""Pool manager of VHF for root manager.

For the root process, the root process only needs to know there exists a child
VHF process that is able to perform the sampling. We let the management of the
active child be handled by the VHFPool class.
"""

from collections import deque
from inspect import getmembers, isroutine
import logging
from logging import Logger, getLogger
from multiprocessing import Process, Pipe, Queue
from queue import Empty
from threading import Thread
from time import sleep
from typing import Callable, Deque, Mapping
from .root import IdentifiedProcess
from .signals import ChildSignals, Signals, HUP, cont


class VHFPool:
    """Provides pipes to manage VHF processes.

    From the perspective of the root process, root only needs to know that it
    will be sending a message to *some* child process, and that when all is
    said and done, to close all processes.
    To ensure timely handling of SIGINT or errors being recieved in polling
    from children processes, we have the requeue procedure be done in a
    secondary thread. The requeue process can therefore act on signals from
    child processes, such as SIGINT to be propagated to all other processes.
    VHFPool shall handle recreation and destruction of children processes,
    without intervention from root process.
    """

    def __init__(
        self,
        fail_forward: Mapping[ChildSignals.type, bool] = {
            ChildSignals().too_many_attempts: True,
        },
        count: int = 3,
        target: Callable = None,
        logging_queue: Queue = None,
        *args, **kwargs
    ) -> None:
        """Prepare VHFPool instance.

        Inputs
        ------
        fail_forward: Mapping[ChildSignals.*, bool]
            When encountering a ChildSignal, such as too_many_attempts,
            fail_forward[ChildSignals.too_many_attempts] = True|False lets the
            'continue_child' function in charge of awaiting the process that
            sampling the VHF board to determine if to stop abruptly or to "fail
            forward".
        count: int
            Maximum number of children that exists simultaneously.
        target: Callable
            Concrete implementation of VHF.multiprocessing.genericVHF is
            expected here; one available instance for sampling from VHF board
            will thus be automatically provided by the Pool.
        logging_queue: multiprocessing.Queue
            For target to log to in a multiprocess-safe manner.
        *args, **kwargs:
            Passed onto target. Note that the first 2 arguments of Target
            should consume multiprocessing.connection.Connection and
            multiprocessing.Queue in the specified order, for recieving
            messages from this VHFPool object, and for the IdentifiedProcess's
            Target to log to.
            Refer to VHF.multiprocess.vhf.genericVHF for clarification.
        """
        self._init_logger(logging_queue)
        self._closed = False  # Used in instance cleanup method: close()
        self.fail_forward = fail_forward

        if target is None:
            self.logger.warning("VHFPool not made with good target!")
            return
        # Process to be spawned and its arguments
        self.target: Callable = target
        self.target_args = (logging_queue, *args)
        self.target_kwargs = kwargs

        # Signals being returned from spawned process being checked
        self.signals = Signals()
        self.c_sig = ChildSignals()

        self._init_checks_fail_forward()

        # Pool: Number of children
        self.count = count  # Target nuber of Jobs to create
        self._count = 0  # Number of Jobs created to far
        # Pool: List of children that should be alive/killed
        self._children: list[IdentifiedProcess] = list()
        self._dead_children: list[IdentifiedProcess] = list()
        self._populate_children()

        # Pop from left to get an available VHF child process.
        # Needs to be initalised
        self.queue: Deque = deque([x for x in self._children])
        # Child that is to be used for next sampling instance.
        self._current_child: IdentifiedProcess = self.queue.popleft()
        self._previous_current_child: IdentifiedProcess = None

        # Requeue worker
        self.rq_thread_queue = Queue()
        self.start_requeue_worker()

    def _init_logger(self, logging_queue):
        """Create Logger for VHFPool."""
        # There is no need for same thread to have queue handled.
        # root_logger = getLogger()
        # qh = QueueHandler(logging_queue)
        # if not root_logger.handlers:
        #     root_logger.addHandler(qh)
        self.logger: Logger = getLogger("VHFPool")
        self.logger.setLevel(logging.DEBUG)

    def _init_checks_fail_forward(self):
        """Check that what is needed satisfies to criteria.

        Checks if fail_forward is as close to ChildSignals as is expected, that
        is to say, all ChildSignals attributes should be keys of fail_forward,
        but no key of fail_forward should not be a ChildSignals attribute.
        Returns dictionary, where key, value pair has value = list of attrbutes
        found in key but not found in other comparision.
        """
        self.logger.debug("init_check start.")
        result = {}

        # Signals from child_signal that we require be the key in
        # self.fail_forward
        attributes = getmembers(ChildSignals, lambda a: not (isroutine(a)))
        ms = [a for a in attributes if not (
            a[0].startswith('__') and a[0].endswith('__'))]
        self.logger.debug("ms = %s", ms)
        defn_not_req = ['type', 'action_cont',
                        'action_request_requeue', 'action_hup']
        mems_not_req = filter(lambda x: x[0] in defn_not_req, ms)
        for mem_not_req in list(mems_not_req):
            ms.remove(mem_not_req)
        self.logger.debug("ms = %s", ms)

        for m in ms:
            if m[1] not in self.fail_forward:
                self.logger.warning(
                    "Testing fail_forward \\supseteq ChildSignals, "
                    "ChildSignals.%s not found in fail_forward.",
                    m
                )
                if "ChildSignals" not in result:
                    result["ChildSignals"] = []
                result["ChildSignals"].append(m)
            else:
                if not isinstance(self.fail_forward[m[1]], bool):
                    self.logger.warning(
                        "fail_forward[%s] is not bool", m[0]
                    )
                    if "fail_forward_bool" not in result:
                        result["fail_forward_bool"] = []
                    result["fail_forward_bool"].append(m)

        # Conversely, we require that all fail_forward keys are within
        # ChildSignals map will get consumed after yielding
        vs = list(map(lambda x: x[1], ms))
        for k in self.fail_forward:
            if k not in vs:
                self.logger.warning(
                    "Testing fail_forward \\subseteq ChildSignals, "
                    "fail_forward had key = %s not found in ChildSignals.",
                    k
                )
                if "fail_forward" not in result:
                    result["fail_forward"] = []
                result["fail_forward"].append(k)

        return result

    # Section: Child management
    def _create_child(self, target: Callable, *args, **kwargs) -> IdentifiedProcess:
        """Update master list of newly created child.

        self._children neeeds to be updated with a fully live child in this
        stage, with all details filled.
        """
        # IdentifiedProcess wrapper starts child process.
        ip: IdentifiedProcess = None
        while ip is None:
            try:
                sink, src = Pipe()
                job: Process = Process(
                    target=target,
                    args=(src, sink, *args),
                    name="VHF "+str(self._count).zfill(2),
                    kwargs=kwargs,
                )
                ip = IdentifiedProcess(job, sink)
            except ValueError:
                self.logger.warning(
                    "Failed to create an IdentifiedProcess. Trying again.")
                ip = None
                pass
            except Exception as exc:
                self.logger.critical("exc = %s", exc, exc_info=True)
                self.logger("Closing VHF Pool.")
                self.close()

        # https://stackoverflow.com/a/74742887
        # https://stackoverflow.com/a/8595331
        # https://stackoverflow.com/q/71532034
        src.close()  # Closes child end on parent process, due to fork.
        self._children.append(ip)
        # Placing live child into queue of available children is for either
        # init or requeue worker to do
        self.logger.debug("Appended %s to self._children.", ip)
        self.logger.debug("%s associated to %s", ip, job)
        return ip

    def _close_child(self, child: IdentifiedProcess):
        """Attempt to join child with cleanup.

        IdentifiedProcess seperates the actual process being closed, from the
        wrapper of the process and all its things to be closed.
        """
        # Increment number of attempts sent to child to close. #? How do we handle if connection is closed from parent end?
        result = child.close_proc()  # By definition of IdentifiedProcess
        was_live = child in self._children
        self.logger.debug(
            "_close_child attempting to close child='%s' with result=%s. "
            "child in self._children = %s",
            child, result, was_live
        )

        if was_live:
            self._children.remove(child)
            if not result:
                self._dead_children.append(child)
            else:
                # Cleanups wrapper of Process, i.e.: IdentifiedProcess
                child.close()
                return

        was_zombie = child in self._dead_children
        if was_zombie and result:
            self._dead_children.remove(child)
            child.close()
            return
        elif was_zombie and not result:
            return

        self.logger.warning(
            "_close_child invoked on %s found in neither self._children not "
            "self._dead_children",
            child
        )

    def _close_live_child_with_replacement(self, child: IdentifiedProcess):
        """Attempt to close live child, and spawn a replacement."""
        if child not in self._children:
            self.logger.warning(
                "Invoked _close_live_child_with_replacement on possibly "
                "zombie child. Replacement child STILL created."
            )
        self._close_child(child)
        ip = self._create_child(self.target, *self.target_args,
                                **self.target_kwargs)
        # The first time the child process is created, it dooes not ask to be requeued
        self.queue.append(ip)

    def _close_dead_child_with_replacement(self, child: IdentifiedProcess):
        """Attempt to close dead child, and spawn a replacement."""
        self.logger.warning(
            "Closing possibly dead child %s with replacement", child)
        self._close_child(child)
        ip = self._create_child(self.target, *self.target_args,
                                **self.target_kwargs)
        # The first time the child process is created, it dooes not ask to be requeued
        self.queue.append(ip)

    # Section: Thread management
    def _populate_children(self):
        """Fill _children with IdentifiedProcess[target, ...]."""
        for _ in range(self.count):
            self._create_child(
                self.target, *self.target_args, **self.target_kwargs
            )

    def _candidates_of_children_to_requeue(self) -> list[IdentifiedProcess]:
        """List of children who might have signalled ready to requeue."""
        result = list(set(self._children) -
                      set([self._current_child, self._previous_current_child]) -
                      set(self._dead_children))
        # if len(result) > 0:
        #     fmt = list(map(str, result))
        #     self.logger.debug("Candidates to requeue: %s", fmt)
        return result

    def requeue_child(self, ip: IdentifiedProcess):
        """Adds child process back into available queue."""
        self.queue.append(ip)

    def start_requeue_worker(self):
        """Worker thread involved in requeuing children who return ready.

        This function acts as the entry point to spawning this requeue worker
        if existing worker was killed, and does not spawn extra worker if an
        existing worker is found to exist.
        """
        self.logger.info("Starting requeue worker.")
        try:
            if self.rq_thread.is_alive():
                self.logger.warning(
                    "start_requeue_worker called despite alive requeue worker. not spawning!")
                return
            # else:
            #     self.rq_thread.join()  # no harm joining again?
        except AttributeError:
            self.logger.debug("Testing for live rq_thread found no thread.")
        self.rq_thread = Thread(
            target=self.requeue_work
        )
        self.rq_thread.start()

    def requeue_work(self):
        """Requeue children that have signalled being able to do more work.

        This method is to be spawned by start_requeue_worker once to requeue
        children processes that are free to join back self.queue, and also
        tries to close out all zombie child processes.
        """
        self.logger.info("Requeue worker has started.")

        # Counter for not invoking joining of dead children too frequently.
        self.rq_thread_dc: int = 2

        while True:
            # Terminate driving loop if HUP'd
            try:
                record = self.rq_thread_queue.get(timeout=0.01)
                if record == HUP:
                    self.logger.info("[Requeuer] HUP recieved!")
                    # We leave the original thread to cleanup all children.
                    self.rq_thread_active = False
                    break
            except Empty:
                pass
            except BrokenPipeError:
                return
            except Exception as exc:
                self.logger.critical("exc = %s", exc, exc_info=True)

            # Trying to requeue
            # joinable: list[Connection] = connection.wait(
            #     object_list=map(lambda x: x.connection,
            #                     self._children_to_requeue()),
            #     timeout=0.01
            # )
            # for x in joinable:
            #     # recover the corresponding child process from its
            #     # .connection attribute

            # Trying to requeue live child
            for x in self._candidates_of_children_to_requeue()[:]:
                try:
                    if x.connection.poll(timeout=0.01):
                        data = x.connection.recv_bytes()
                        if data == self.c_sig.action_request_requeue:
                            self.requeue_child(x)
                            self.logger.debug("Requeued %s", x)
                        else:
                            # Msg from child has already been consumed!
                            self.logger.warn(
                                "[Requeuer] Recieved from child=%s, data=%s",
                                x, data
                            )
                            self._close_live_child_with_replacement(x)
                except Empty:
                    pass
                except BlockingIOError:
                    self.logger.debug(
                        "BlockingIOError occured while trying to poll from %s.", x)
                except EOFError:
                    self.logger.warning(
                        "EOF Error reached on possibly dead child %s.",
                        x
                    )
                    # Chances are child has died after raising error.
                    self._close_dead_child_with_replacement(x)
                except Exception as exc:
                    self.logger.critical("exc = %s", exc, exc_info=True)

            # Artificial spacer modulo 10
            if len(self._dead_children) > 0:
                self.rq_thread_dc += 1
                if self.rq_thread_dc == 10:
                    self.rq_thread_dc = 1
            if self.rq_thread_dc == 1:
                for x in self._dead_children:
                    # _close_child removes x from list if successful
                    self._close_child(x)

            # Add some delay before next batch of requeue attempts
            sleep(0.2)
            # self.logger.debug(
            #     "[Requeuer] self._children = %s, self.queue = %s, self._dead_children = %s, self._candidates_of_children_to_requeue = %s",
            #     self._children,
            #     self.queue,
            #     self._dead_children,
            #     self._candidates_of_children_to_requeue(),
            # )

    def _close_all(self):
        """Terminates all children."""
        while len(self._children) > 0:
            for x in self._children[:]:
                self._close_child(x)
            sleep(0.2)
        self.logger.info("No live children left.")
        while len(self._dead_children) > 0:
            for x in self._dead_children[:]:
                self._close_child(x)
            sleep(1)
        self.logger.info("No zombie children left.")
        self.logger.info("All children process are closed.")

    # Section: Handling of active child
    def _send_to_child(self, msg) -> tuple[IdentifiedProcess, bool]:
        """Send msg to child process.

        Returns
        -------
        child: IdentifiedProcess
            Poll from IdentifiedProcess.connection to assert the recieved
            result is success prior to sending next message.
        """
        # Propagate message to available child.
        self.logger.info(
            "Sending msg = `%s` to child with PID `%s`", msg, self._current_child.pid)
        try:
            self._current_child.connection.send(msg)
            success = True
        except BrokenPipeError:
            self.logger.warning(
                "Child %s Pipe found to be closed!", self._current_child.pid)
            success = False
        self._previous_current_child = self._current_child
        # Get a new child
        self._current_child = None
        while len(self.queue) < 1:
            sleep(0.2)
            # self.logger.debug(
            #     "Waiting to pop for current_child. self.queue = %s",
            #     self.queue
            # )
            if not self.rq_thread.is_alive():
                self.rq_thread.join()
                self.start_requeue_worker()
                sleep(0.2)
        self._current_child = self.queue.popleft()
        self.logger.debug("Assigned a new current child: %s",
                          self._current_child)
        return self._previous_current_child, success

    # @disable_warn_if_auto
    # def continue_child_nonblock(self, *args) -> IdentifiedProcess:
    #     """Send *args to child process, gets back child that is in use.
    #
    #     This is intended more for manual management by the root process.
    #     VHFPool instance does cycle into a new child for subsequent use of VHF
    #     board.
    #     """
    #     if self._nonblock_warn:
    #         self.logger.warning(
    #             "continue_child_nonblock invoked despite expecting automatic "
    #             "child management! Consider continue_child."
    #         )
    #     return self._send_to_child(self.signals.action_cont(args))

    def continue_child(self, *args) -> bool:
        """Send msg to continue active child, Blocks until sampling complete.

        Check if True to determine if child has performed trace
        successfully.
        """
        # it is possible that sending to child find itself as being a dead child
        child, success = self._send_to_child(cont(*args))
        if not success:
            # Broken Pipe Error was handled within _send_to_child
            self.logger.warning("Child to be replaced.")
            self._close_dead_child_with_replacement(child)
            return False

        try:
            data = child.connection.recv_bytes()
        except EOFError:
            self._close_dead_child_with_replacement(child)
            return False
        if data == self.c_sig.action_cont:
            return True
        elif data == self.c_sig.action_hup:
            self._close_all()
            return False
        elif data in self.fail_forward.keys():
            if not self.fail_forward[data]:
                # fail_forward found to be false, closing all.
                self._close_all()
            else:
                self._close_live_child_with_replacement(child)
            return False
        else:
            self.logger.warning(
                "Unknown signal from active child = %s.",
                child
            )
            self.logger.warning("data = %s", data)
            if True:  # TODO
                self._close_dead_child_with_replacement(child)
            return False

    # Section: Personal cleanup
    def close(self):
        """Cleanup all relevant object in instance."""
        if self._closed:
            self.logger.warning(
                "VHFPool instance has .close invoked more than once! Skipping..."
            )
            return
        self._closed = True

        # cleanup spawned thread
        if self.rq_thread.is_alive():
            self.logger.info("Putting HUP to requeuer thread in VHFPool.")
            self.rq_thread_queue.put(HUP)
            sleep(0.05)
        self.rq_thread.join()  # Doc: A thread can joined many times
        self.logger.debug("self.rq_thread joined.")

        # cleanup children
        self._close_all()
