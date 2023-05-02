from shlex import quote
import subprocess
import os
# from pyvisa import ResourceManager
import logging
import datetime


# def resource_addr(serial_no: str) -> str:
    # """Obtain USBTMC resource string of VISA resource."""
    # cached = ResourceManager().list_resources()
    # print(f"Obtained resources: {cached}")
    # for x in cached:
        # if serial_no in x.split("::"):
            # return x
    # return None


# def main_old():
#     """
#     Feed Tektronix function generator sine wave into ADC in, and run board."""
#     # Manipulating function generator was a struggle
#     tektronix_sn = 'C020198'
#     tektronix_rsc_addr = resource_addr(tektronix_sn).replace('::0', '')
#     print(f"{tektronix_rsc_addr = }")
#     # fg = funcGenDriver(tektronix_rsc_addr)
#     # print(f"{fg.idn = }")


def main():
    """Runs VHF board for some amount of time, and logs output."""

    # parameters to run board at
    filename = "60.000_020_000MHz.bin"
    num_samples = 20_000_000  # q
    skip_num = 5-1  # s
    low_speed = True # False => High Speed, 20MHz. True => Low Speed, 10 MHz

    # inform of parameters
    print(f"Sampling at {(10*(2-low_speed))/(skip_num+1)} MHz.\nSampling for {(2-low_speed)*(num_samples*(1+skip_num))/(60*10**7):.3f} mins.")
    time_seconds = (2-low_speed)*1e7/(1+skip_num)
    print(f"Filename = {filename}")
    if input('Are these parameter alright? [Y/N]: ').strip().upper() != 'Y':
        print('Stopping.')
        return

    logging.basicConfig(filename='Log/'+filename+datetime.datetime.now().strftime('_%Y%m%d_%H%M:%s.log'),
                        filemode='w', format='[%(asctime)s] %(name)s -%(levelname)s -\t%(message)s',
                        level=logging.DEBUG)

    start_time = datetime.datetime.now()
    logging.info("Sampling started at %s", start_time)
    
    cmd = [
            quote(str(os.path.realpath(os.path.join(os.path.dirname(__file__), '../teststream.exec')))),
            '-U', quote(str(os.path.realpath(os.path.join(os.path.dirname(__file__), '../vhf_board.softlink')))),
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

    end_time = datetime.datetime.now()
    print(f"Sampling was ran for {(end_time - start_time)}.")
    logging.info("Sampling ended at %s", end_time)
    logging.info("Script ran for %s", str(end_time - start_time))
    return


if __name__ == '__main__':
    main()