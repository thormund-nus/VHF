"""Here, we test writing to a NAS through the use of a tempdir with filename.
33.55s sampling takes 14.24s to write after the tempfile has finished collecting the buffer.
"""

import configparser
import datetime
import logging
from pathlib import Path
import subprocess
import tempfile

import os, sys
from pathlib import Path
module_path = str(Path(__file__).parents[2])
if module_path not in sys.path:
    sys.path.append(module_path)
from VHF.runner import VHFRunner



def main():
    """Runs VHF board for some amount of time, and logs output."""


    vhf_config_path = Path(__file__).parent.joinpath('vhf-params.ini')
    vhf_runner = VHFRunner(vhf_config_path)
    vhf_runner.inform_params()
    conf = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
    conf.read(vhf_config_path)
    # confirmation_input = input("Are these parameter alright? [Y/N]: ")
    # if confirmation_input.strip().upper() != "Y":
    #     print("Stopping.")
    #     return


    with tempfile.TemporaryDirectory() as td:
        # fn = os.path.join(td, 'tmp')
        with tempfile.NamedTemporaryFile(dir=td) as f:
            try:
                retcode = subprocess.run(
                    **(sb_run:=vhf_runner.subprocess_run(f))
                )
                logging.info("Subprocess ran with %s", str(sb_run))
                logging.info("Retcode %s", retcode)
                print("Sampling collected.")
                print(f"NamedTempFile = {f}")
                print(f"\t{f.name = }")
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

            start_time = datetime.datetime.now()
            with open((write_target:=Path(conf.get('Paths', 'save_dir'))\
                    .joinpath(f"{str(start_time)}_tempfile.bin")), 'wb') as target:
                filesize = f.tell()
                f.seek(0)
                target.write(f.read(filesize))
            end_time = datetime.datetime.now()
            print(f"Written to {write_target}")
        print(f"Writing took for {(end_time - start_time)}.")
    return


if __name__ == "__main__":
    main()
