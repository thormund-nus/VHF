"""Here, we test writing to a NAS directly.
33.55s sampling takes 44.22s to write before timeout
"""

import datetime
import logging
from pathlib import Path
import subprocess

import os, sys
from pathlib import Path
module_path = str(Path(__file__).parents[2])
if module_path not in sys.path:
    sys.path.append(module_path)
from VHFRunner.VHFRunner import VHFRunner


def main():
    """Runs VHF board for some amount of time, and logs output."""

    vhf_config_path = Path(__file__).parent.joinpath('vhf-params.ini')
    vhf_runner = VHFRunner(vhf_config_path)
    vhf_runner.inform_params()
    # confirmation_input = input("Are these parameter alright? [Y/N]: ")
    # if confirmation_input.strip().upper() != "Y":
    #     print("Stopping.")
    #     return

    start_time = datetime.datetime.now()

    try:
        retcode = subprocess.run(
            **(sb_run:=vhf_runner.subprocess_run())
        )
        logging.info("Subprocess ran with %s", str(sb_run))
        logging.info("Retcode %s", retcode)
    except KeyboardInterrupt:
        print("Keyboard Interrupt recieved!")
    except subprocess.CalledProcessError as exc:
        print(f"Process returned with error code {255-exc.returncode}")
        print(f"{exc.stderr = }")
    except subprocess.TimeoutExpired as exc:
        # logging.info("Subprocess ran with %s", str(sb_run))
        # logging.critical("TimeoutExpired with exception:")
        # logging.critical("%s", exc)
        # logging.critical("")
        # logging.critical("exc.stderr = ")
        # logging.critical("%s", exc.stderr)
        # logging.critical("")
        print(f"Process Timed out!")

    end_time = datetime.datetime.now()
    print(f"Sampling was ran for {(end_time - start_time)}.")
    return


if __name__ == "__main__":
    main()
