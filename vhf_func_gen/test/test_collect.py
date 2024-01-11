from configparser import ConfigParser, ExtendedInterpolation
from datetime import datetime
import logging
from multiprocessing import Lock, Process, Queue
from multiprocessing.connection import Pipe
import numpy as np
import os
from pathlib import Path
import sys
from threading import Thread
module_path = str(Path(__file__).parents[2].joinpath(
    "vhf_func_gen").joinpath("run_vhf_dep"))
if module_path not in sys.path:
    sys.path.append(module_path)
from collect import FuncGenExpt  # noqa
from VHF.multiprocess.signals import cont, HUP  # noqa


def log_listener(q):
    while True:
        record = q.get()
        if record == HUP:
            break
        logger = logging.getLogger(record.name)
        logger.handle(record)
    return


def no_matplot(record: logging.LogRecord):
    return not record.name.startswith("matplotlib") and not record.name.startswith("PIL")


def main():
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
    streamhandler.addFilter(no_matplot)
    logger.addHandler(streamhandler)

    conf_path = Path(__file__).parents[1].joinpath("VHF_FuncGen_params.ini")
    logger.info("conf_path = %s", conf_path)
    logger.info("conf_path.exists() = %s", conf_path.exists())
    conf = ConfigParser(interpolation=ExtendedInterpolation())
    conf.read(conf_path)

    # Spawn up empty npz file
    npz_path = Path(conf.get("Paths", "collated_npz_save"))
    size = (4, 5)
    np.savez(
        npz_path,
        mean=np.zeros(size),
        std_dev=np.zeros(size),
        min=np.zeros(size),
    )

    # Spawn up child processes
    q = Queue()
    lp = Thread(  # listening process
        target=log_listener,
        args=(q,),
    )
    lp.start()
    sink, src = Pipe()
    npz_lock = Lock()
    wp = Process(
        target=FuncGenExpt,
        args=(src, q, conf_path, npz_lock),
        kwargs={"request_requeue": False},
        name="VHF01"
    )
    wp.start()
    logger = logging.getLogger("Main")
    data = sink.recv()
    assert data[0] == 0
    child_pid = data[1]
    logger.info("Child pid: %s", child_pid)
    sink.send(
        cont(
            {"power": "0.3dBm", "i": str(1)},
            (1, 2)
        )
    )
    logger.info("Sent start command at %s", datetime.now())
    data = sink.recv_bytes()
    logger.info("%s: Recieved %s", datetime.now(), data)
    # cleanup and close
    sink.send(HUP)
    wp.join()
    sink.close()
    q.put(HUP)
    lp.join()

    logger.info("Main is exiting.")
    # check that npz is sensible
    with np.load(npz_path) as data:
        print(f"{data=}")
        print(f"{data.items()=}")
        for x in data:
            print(f"{x=}")
            print(f"{data[x]=}")


if __name__ == "__main__":
    assert Path(os.getcwd()) == Path(__file__).parents[2]
    main()
