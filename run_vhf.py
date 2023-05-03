from shlex import quote
import subprocess
import os
import logging
import datetime


def main():
    """Runs VHF board for some amount of time, and logs output."""

    # parameters to run board at
    # TODO: read params from config file
    num_samples = int(2**21)  # q
    # num_samples = int(2.8e9/8)  # q
    skip_num = 5-1  # s
    low_speed = True # False => High Speed, 20MHz. True => Low Speed, 10 MHz
    encoding = 'binary'

    # manipulate params
    # 1. match encoding to filetype
    # 2. pass appropriate flag to C executable argc/argv
    # Consider: Enum types?
    match encoding[:3].lower():
        case 'bin':
            encode_flag = '-b'
            file_ext = 'bin'
        case 'hex':
            encode_flag = '' # ?
            file_ext = 'hex'
        case 'asc' | 'txt':
            encode_flag = '-A'
            file_ext = 'txt'

    # inform of parameters in CLI before proceeding
    time_seconds = num_samples / ((2-low_speed)*1e7/(1+skip_num))
    print(f"Sampling at {(10*(2-low_speed))/(skip_num+1)} MHz.\nSampling for {time_seconds/60:.3f} mins.")
    
    if input('Are these parameter alright? [Y/N]: ').strip().upper() != 'Y':
        print('Stopping.')
        return

    logging.basicConfig(filename=datetime.datetime.now().strftime('Log/runVHFlog_%Y_%m_%d_%H_%M_%s.log'),
                        filemode='w', format='[%(asctime)s] %(name)s -\t%(levelname)s -\t%(message)s',
                        level=logging.DEBUG)

    start_time = datetime.datetime.now()
    logging.info("Sampling started at %s", start_time)
    
    filename = f"{start_time.isoformat()}_s{skip_num}_q{num_samples}.{file_ext}"
    cmd = [
            quote(str(os.path.realpath(os.path.join(os.path.dirname(__file__), 'teststream.exec')))),
            '-U', quote(str(os.path.realpath(os.path.join(os.path.dirname(__file__), 'vhf_board.softlink')))),
            '-q', str(num_samples), '-s', str(skip_num),
            encode_flag,
            '-l' if low_speed else '-h',  # 10 MS/s sampling
            '-v', '3',  # verbosity
            # '-o', f"Data/{filename}" 
            '-o', quote(os.path.join(os.path.dirname(__file__), 'Data', filename))
            # '-o', os.fsencode(os.path.join(os.path.dirname(__file__), 'Data', filename))
            # '-o', r'{}'.format(os.path.join(os.path.dirname(__file__), 'Data', filename).replace(' ', r'\ '))
            # '-o', os.path.join(os.path.dirname(__file__), 'Data', filename).replace(' ', '\ ')
          ]
    print(f"cmd = \n\t{' '.join(cmd)}")
    logging.info("Command has been passed into executable. = ")
    logging.info("%s", cmd)
    try:
        retcode = subprocess.run(cmd,
                                 capture_output=True, check=True,
                                 cwd=os.path.dirname(__file__),
                                 timeout=time_seconds+7
                                #  shell=True
                                 )
        logging.info("Retcode %s", retcode)
    except KeyboardInterrupt:
        logging.info('Keyboard Interrupt')
        print("Keyboard Interrupt recieved!")
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
    return


if __name__ == '__main__':
    main()
