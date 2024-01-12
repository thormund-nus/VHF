import configparser
import datetime
import logging
from pathlib import Path
import subprocess
from tqdm import tqdm
from VHF.runner import VHFRunner


def main():
    """Runs VHF board for some amount of time, and logs output."""

    vhf_config_path = Path(__file__).parent.joinpath('VHF_board_params.ini')
    vhf_runner = VHFRunner(vhf_config_path)
    vhf_runner.inform_params()

    vhf_conf = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation())
    vhf_conf.read(vhf_config_path)
    number_runs = vhf_conf.getint('Extended Sampling', 'num_runs')
    if number_runs <= 1:
        print("\x1B[41mUnknown configuration in Extended Sampling, please use run_VHF.py\x1B[49m")
        return
    print(f"{__file__} is being run with {number_runs = }.")

    confirmation_input = input("Are these parameter alright? [y/N]: ")
    if confirmation_input.strip().upper() != "Y":
        print("Stopping.")
        return

    logging.basicConfig(
        filename=datetime.datetime.now().strftime(
            "Log/runprolongedVHFlog_%Y%m%d_%H%M%s.log"
        ),
        filemode="w",
        format="[%(asctime)s] %(name)s -\t%(levelname)s -\t%(message)s",
        level=logging.DEBUG,
    )

    start_time = datetime.datetime.now()

    try:
        for _ in tqdm(range(number_runs)):
            retcode = subprocess.run(
                **(sb_run:=vhf_runner.subprocess_run())
            )
            logging.info("Subprocess ran with %s", str(sb_run))
            logging.info("Retcode %s", retcode)
    except KeyboardInterrupt:
        logging.info("Subprocess ran with %s", str(sb_run))
        logging.info("Keyboard Interrupt")
        print("Keyboard Interrupt recieved!")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logging.info("Subprocess ran with %s", str(sb_run))
        logging.critical("exc: %s", exc)
        logging.critical("exc.stderr = %s", exc.stderr)
        logging.critical("", exc_info=True)
        if isinstance(exc, subprocess.CalledProcessError):
            print(f"Process returned with error code {exc.returncode}")
        print(f"{exc.stderr = }")

    end_time = datetime.datetime.now()
    print(f"Sampling was ran for {(end_time - start_time)}.")
    logging.info("Sampling ended at %s", end_time)
    logging.info("Script ran for %s", str(end_time - start_time))
    return


if __name__ == "__main__":
    main()
