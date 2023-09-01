import datetime
import logging
import subprocess
from matplotlib import pyplot as plt
from parseVHF import VHFparser  # relative import
from pathlib import Path
from plot_VHF_output import plot_rad_spec, get_phase
from tempfile import TemporaryFile
from VHFRunner.VHFRunner import VHFRunner


def main():
    logging.basicConfig(
        filename=datetime.datetime.now().strftime(
            "Log/runVHFlog_%Y_%m_%d_%H_%M_%s.log"
        ),
        filemode="w",
        format="[%(asctime)s] %(name)s -\t%(levelname)s -\t%(message)s",
        level=logging.DEBUG,
    )
    
    vhf_config_path = Path(__file__).parent.joinpath('VHF_board_params.ini')
    vhf_runner = VHFRunner(vhf_config_path, force_to_buffer=True,
        overwrite_properties={'num_samples': int(2**18), 'skip_num': 5 - 1})
    vhf_runner.inform_params()

    start_time = datetime.datetime.now()
    try:
        with TemporaryFile() as f:
            retcode = subprocess.run(
                **(sb_run:=vhf_runner.subprocess_run(stdout=f))
            )
            logging.info("Subprocess ran with %s", str(sb_run))
            logging.info("Retcode %s", retcode)
            parsed = VHFparser(f)
            phase = get_phase(parsed)
            fig = plot_rad_spec(True, True)(parsed, phase)

    except KeyboardInterrupt:
        logging.info("Keyboard Interrupt")
        print("Keyboard Interrupt recieved!")
    except subprocess.CalledProcessError as exc:
        logging.critical("CalledProcess error with exception! Details:")
        logging.critical("%s", exc)
        logging.critical("")
        logging.critical("exc.stderr = ")
        logging.critical("%s", exc.stderr)
        logging.critical("")
        print(f"Process returned with error code {255-exc.returncode}")
        print(f"{exc.stderr = }")
    except subprocess.TimeoutExpired as exc:
        logging.critical("TimeoutExpired with exception:")
        logging.critical("%s", exc)
        logging.critical("")
        logging.critical("exc.stderr = ")
        logging.critical("%s", exc.stderr)
        logging.critical("")
        print(f"Process Timed out!")


if __name__ == "__main__":
    main()
