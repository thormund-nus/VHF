# To test if a single class can spawn and interact with a secondary thread
# within the same class.


from collections import deque
import logging
from logging import getLogger
from queue import Empty, Queue
from threading import Thread
from time import sleep
import sys

HUP = b'2'


class ThreadTester:
    def __init__(self):
        self._initLogger()
        self._closed = False
        self.all = [1, 2, 3, 4]
        self.available = deque(self.all)
        self.logger.info("Parent thread init.")
        self.requeuer = Thread(
            target=self.requeue_worker,
            args=(),
        )
        self.q = Queue()
        self.requeuer.start()

    def _initLogger(self):
        self.logger = getLogger("ThreadTester")

    def requeue_one(self, val):
        self.available.append(val)

    def requeue_worker(self):
        self.logger.info("Requeue worker has started.")
        while True:
            # Terminate driving loop if HUP'd
            try:
                record = self.q.get(timeout=0.01)
                if record == HUP:
                    self.logger.info("[w] HUP recieved!")
                    for x in self.all:
                        # Simulate closing everything in actual case
                        self.requeue_one(x)
                    self.logger.info("[w] self.available = %s", self.available)
                    self.logger.info("[w] requeue_worker quitting")
                    break
            except Empty:
                pass
            except Exception as exc:
                self.logger.critical("exc = %s", exc, exc_info=True)
            for x in self.all:
                if x not in self.available.copy():
                    self.logger.info(
                        "[w] self.all = %s, self.available = %s", self.all, self.available)
                    self.logger.info(
                        "[w] %s not found in self.avail, requeueing", x)
                    self.requeue_one(x)
            sleep(0.2)

    def close(self):
        if self._closed == True:
            self.logger.warning(
                "Closed has been invoked more than once; Not closing again!")
            return
        self.q.put(HUP)
        self.requeuer.join()
        self.logger.debug(
            "ThreadTester should not be invoked anymore! Current state:")
        self.logger.debug("self.all = %s", self.all)
        self.logger.debug("self.available = %s", self.available)
        self._closed = True

    def add_to_all(self, val):
        self.all.append(val)
        self.logger.info("val = %s appended, now self.all = %s", val, self.all)

    def activate_new(self):
        self.active = self.available.popleft()
        self.logger.info("popped %s, now self.avail = %s",
                         self.active, self.available)


def main():
    logger = getLogger()
    logger.setLevel(logging.DEBUG)
    streamhandler = logging.StreamHandler(sys.stdout)
    streamhandler.setLevel(logging.DEBUG)
    fmtter = logging.Formatter(
        # the current datefmt str discards date information
        '[%(asctime)s%(msecs)d] (%(levelname)s)\t[%(processName)s] %(name)s: \t %(message)s', datefmt='%H:%M:%S:'
    )
    fmtter.default_msec_format = "%s.03d"
    streamhandler.setFormatter(fmtter)
    logger.addHandler(streamhandler)
    logger.info("Logger init.")

    t = ThreadTester()
    t.activate_new()
    sleep(0.5)
    t.add_to_all(6)
    for _ in range(12):
        t.activate_new()
        sleep(0.5)
    t.close()

    logger.info("Ending!")


if __name__ == "__main__":
    main()
