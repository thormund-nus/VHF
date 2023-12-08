import datetime
import logging
from parseVHF import VHFparser
from pathlib import Path
from plot_VHF_output import get_phase, plot_rad_spec
from matplotlib import pyplot as plt
import subprocess
from subprocess import PIPE
from tempfile import NamedTemporaryFile
from VHFRunner.VHFRunner import VHFRunner


def main():
    """Runs VHF board for some amount of time, and displays out temporarily."""

    print("\x1b[41mRuns VHF board and shows 15s of data. Does not save data!\x1B[0m")
    vhf_config_path = Path(__file__).parent.joinpath('VHF_board_params.ini')
    vhf_runner = VHFRunner(vhf_config_path,
        force_to_buffer=True,
        overwrite_properties={
            'skip_num': 4,
            'num_samples': 2**25,
        }
    )
    vhf_runner.inform_params()
    # confirmation_input = input("Are these parameter alright? [y/N]: ")
    # if confirmation_input.strip().upper() != "Y":
    #     print("Stopping.")
    #     return

    logging.basicConfig(
        filename=datetime.datetime.now().strftime(
            "Log/runVHFlog_%Y_%m_%d_%H_%M_%s.log"
        ),
        filemode="w",
        format="[%(asctime)s] %(name)s -\t%(levelname)s -\t%(message)s",
        level=logging.DEBUG,
    )

    with NamedTemporaryFile(dir='/dev/shm') as tmp_store:
        start_time = datetime.datetime.now()

        try:
            retcode = subprocess.run(
                **(sb_run:=vhf_runner.subprocess_run(stdout=tmp_store)),
                stderr=PIPE
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

        parsed = VHFparser(tmp_store.name)
        phase = get_phase(parsed)
        print(f"Phase mean: {phase[12000:].mean()}\nPhase Std Dev: {phase[12000:].std()}")
        fig = plot_rad_spec(True, True)(parsed, phase)
        view_const = 2.3
        fig.legend()
        fig.set_size_inches(view_const * 0.85 *
                            (8.25 - 0.875 * 2), view_const * 2.5)
        fig.tight_layout()
        fig.canvas.manager.set_window_title(tmp_store.name)
        plt.show(block=True)

    return


if __name__ == "__main__":
    main()
