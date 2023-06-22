from shlex import quote
import subprocess
import os
from plot_VHF_output import plot_rad_spec, get_phase
# from subprocess import PIPE
from tempfile import TemporaryFile
from matplotlib import pyplot as plt
from parseVHF import VHFparser  # relative import
import logging
import datetime


def main():
    num_samples = int(2**16)  # q
    skip_num = 5 - 1
    vga_num = 5-1  # g
    low_speed = True  # False => High Speed, 20MHz. True => Low Speed, 10 MHz
    encoding = "binary"
    filter_const = 0
    laser_num = 2
    laser_curr = 3825
    # manipulate params
    # 1. match encoding to filetype
    # 2. pass appropriate flag to C executable argc/argv
    # Consider: Enum types?
    match encoding[:3].lower():
        case "bin":
            encode_flag = "-b"
            file_ext = "bin"
        case "hex":
            encode_flag = ""  # ?
            file_ext = "hex"
        case "asc" | "txt":
            encode_flag = "-A"
            file_ext = "txt"
    if file_ext is None:
        raise ValueError("specified encoding is unknown")
    time_seconds = num_samples / ((2 - low_speed) * 1e7 / (1 + skip_num))
    start_time = datetime.datetime.now()
    logging.info("Sampling started at %s", start_time)
    filename = f"{start_time.isoformat()}_s{skip_num}_q{num_samples}_F{filter_const}_laser{laser_num}_{laser_curr}mA.{file_ext}"
    cmd = [
            quote(
                str(
                    os.path.realpath(
                        os.path.join(os.path.dirname(__file__), "teststream.exec")
                    )
                )
            ),
            "-U",
            quote(
                str(
                    os.path.realpath(
                        os.path.join(os.path.dirname(__file__), "vhf_board.softlink")
                    )
                )
            ),
            "-q",
            str(num_samples),
            "-s",
            str(skip_num),
            encode_flag,
            "-F",
            str(filter_const),
            "-l" if low_speed else "-h",  # 10 MS/s sampling
            "-v",
            "3",  # verbosity
            # "-o",
            # "/tmp/show_radius_stddev_tmp.bin"
    ]
    logging.info("Command has been passed into executable. = ")
    logging.info("%s", cmd)
    try:
        with TemporaryFile() as f:
            retcode = subprocess.run(
                cmd,
                # capture_output=True,
                check=True,
                cwd=os.path.dirname(__file__),
                timeout=time_seconds + 7,
                stdout=f
            )
            logging.info("Retcode %s", retcode)
            # parsed = VHFparser("/tmp/show_radius_stddev_tmp.bin")
            parsed = VHFparser(f)
            phase = get_phase(parsed)
            fig = plot_rad_spec(True, True)(parsed, phase)
            # fig.tight_layout()
            # plt.show(block=True)

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
