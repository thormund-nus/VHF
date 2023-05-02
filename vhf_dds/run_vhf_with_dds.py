import pathlib, sys, os
from time import sleep
from tqdm import tqdm
import numpy as np
from shlex import quote
import subprocess
import datetime
import logging
from dds_controller import DDS # relative import

def per_f(d: DDS, freq:  float):

    # set DDS frequency
    d.freq = freq
    print(f'DDS freq has been set to \x1B[34m{d.freq}\x1B[39m.')
    sleep(0.5)

    # collect on VHF
    filename  = f"{freq:.12f}MHz.bin"
    num_samples = 10_000_000  # q
    skip_num = 5-1  # s
    low_speed = True # False => High Speed, 20MHz. True => Low Speed, 10 MHz

    print(f"Sampling at {(10*(2-low_speed))/(skip_num+1)} MHz.\nSampling for {(2-low_speed)*(num_samples*(1+skip_num))/(60*10**7):.3f} mins.")
    time_seconds = num_samples / ((2-low_speed)*1e7/(1+skip_num))
    print(f"Filename = {filename}")

    start_time = datetime.datetime.now()
    logging.info("Sampling started at %s", start_time)

    cmd = [
            str(os.path.realpath(os.path.join(os.path.dirname(__file__), '../teststream.exec'))),
            '-U', f"{str(os.path.realpath(os.path.join(os.path.dirname(__file__), '../vhf_board.softlink')))}",
            '-q', str(num_samples), '-s', str(skip_num),
            # '-A',  # Ascii
            '-b',  # binary
            '-l',  # 10 MS/s sampling
            '-v', '3',  # verbosity
            # '-o', f"Data/{filename}" 
            '-o', quote(os.path.join(os.path.dirname(__file__), 'Data', filename))
            # '-o', os.fsencode(os.path.join(os.path.dirname(__file__), 'Data', filename))
            # '-o', r'{}'.format(os.path.join(os.path.dirname(__file__), 'Data', filename).replace(' ', r'\ '))
            # '-o', os.path.join(os.path.dirname(__file__), 'Data', filename).replace(' ', '\ ')
          ]
    
    print(f"cmd = \n\t{cmd}")
    logging.info("cmd fed = %s", cmd)
    try:
        retcode = subprocess.run(cmd,
                                 capture_output=True, check=True,
                                 cwd=os.path.dirname(__file__),
                                 timeout=time_seconds+15
                                #  shell=True
                                 )
        logging.info("Retcode %s", retcode)
    except subprocess.CalledProcessError as exc:
        logging.critical("CalledProcess error with exception! Details:")
        logging.critical('%s', exc)
        logging.critical('')
        logging.critical('exc.stderr = ')
        logging.critical('%s', exc.stderr)
        logging.critical('')
        print(f"Process returned with error code {255-exc.returncode}")
        print(f"{exc.stderr = }")
    except subprocess.TimeoutExpired as exc:
        logging.critical("TimeoutExpired with exception:")
        logging.critical('%s', exc)
        logging.critical('')
        logging.critical('exc.stderr = ')
        logging.critical('%s', exc.stderr)
        logging.critical('')
        print(f"Process Timed out!")

    end_time = datetime.datetime.now()
    print(f"Sampling was ran for {(end_time - start_time)}.")
    logging.info("Sampling ended at %s", end_time)
    logging.info("Script ran for %s", str(end_time - start_time))

    print()
    return

def main():
    dds = DDS(0, dds_dev_path='/dev/dds1', dds_prog_path='/home/qitlab/programs/usbdds/apps/')
    diff = np.array([1, 3, 5])
    diff = np.concatenate([diff * (10**i) for i in range(0, 4)])
    diff = np.concatenate([diff, -diff])
    freqs = 140 + diff/1e6
    logging.basicConfig(filename='Log/'+datetime.datetime.now().strftime('_%Y%m%d_%H%M:%s.log'),
                        filemode='w', format='[%(asctime)s] %(name)s -%(levelname)s -\t%(message)s',
                        level=logging.DEBUG)
    for f in tqdm(freqs):
        per_f(dds, f)

if __name__ == '__main__':
    main()
