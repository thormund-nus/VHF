import datetime
import logging
from pathlib import Path
import subprocess
from VHF.runner import VHFRunner


def main():
    """Runs VHF board for some amount of time, and logs output."""

    vhf_config_path = Path(__file__).parent.joinpath('VHF_board_params.ini')
    vhf_runner = VHFRunner(vhf_config_path)
    vhf_runner.inform_params()
    confirmation_input = input("Are these parameter alright? [Y/N]: ")
    if confirmation_input.strip().upper() != "Y":
        print("Stopping.")
        return

    logging.basicConfig(
        filename=datetime.datetime.now().strftime(
            "Log/runVHFlog_%Y_%m_%d_%H_%M_%s.log"
        ),
        filemode="w",
        format="[%(asctime)s] %(name)s -\t%(levelname)s -\t%(message)s",
        level=logging.DEBUG,
    )

    start_time = datetime.datetime.now()

    try:
        retcode = subprocess.run(
            **(sb_run:=vhf_runner.subprocess_run())
        )
        logging.info("Subprocess ran with %s", str(sb_run))
        logging.info("Retcode %s", retcode)
    except KeyboardInterrupt:
        logging.info("Subprocess ran with %s", str(sb_run))
        logging.info("Keyboard Interrupt")
        print("Keyboard Interrupt recieved!")
    except subprocess.CalledProcessError as exc:
        logging.info("Subprocess ran with %s", str(sb_run))
        logging.critical("CalledProcess error with exception! Details:")
        logging.critical("%s", exc)
        logging.critical("")
        logging.critical("exc.stderr = ")
        logging.critical("%s", exc.stderr)
        logging.critical("")
        print(f"Process returned with error code {255-exc.returncode}")
        print(f"{exc.stderr = }")
    except subprocess.TimeoutExpired as exc:
        logging.info("Subprocess ran with %s", str(sb_run))
        logging.critical("TimeoutExpired with exception:")
        logging.critical("%s", exc)
        logging.critical("")
        logging.critical("exc.stderr = ")
        logging.critical("%s", exc.stderr)
        logging.critical("")
        print(f"Process Timed out!")

    end_time = datetime.datetime.now()
    print(f"Sampling was ran for {(end_time - start_time)}.")
    logging.info("Sampling ended at %s", end_time)
    logging.info("Script ran for %s", str(end_time - start_time))
    return


if __name__ == "__main__":
    main()
