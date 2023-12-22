# When recieving signal from run_moment.py, set the laser current to the
# signaled value. Sleep for appropriate amount of time for the entire fibre to
# be flushed with new light, before signalling back to run_moment.py of
# completion.

import logging
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from qodevices.thorlabs.thorlabs_laser_driver \
    import thorlabsLaserDriver as ld_controller
from time import sleep
from usbtmc import list_resources

__all__ = ["set_current_thread"]


def set_current_thread(comm: Connection, itc_rsc: str, IVhystersis = None) -> None:
    """
    Thread for interfacing with Thorlabs ITC4001 Laser Driver.
    Collects Laser Diode Current and Voltage if requested.

    Signals to run_moment.py when VHF board is free to continue.

    Args:
        comm: The pipe for communication between the main and this process.
        itc_rsc: The resource name of the Thorlabs ITC4001 Laser Driver.
        IVhystersis: If not None, dictate this thread to measure the voltage and
            current across the laser diode. (num, interval, floc) := IVhystersis
            floc, i.e.: file location denotes the file to save the data to
    """
    logging.info('Laser diode thread has been called.')
    print('[Laser] Set-laser-current thread has been called.')

    with ld_controller(itc_rsc) as itc:
        print('[Laser] With block init success.')
        # set current control to ITC
        if itc.sour_func_mode != "CURR":
            logging.critical("ITC %s is not set to current mode!", itc_rsc)
            print(f"ITC {idn} is not set to current mode! {itc.sour_func_mode = }")
            comm.send_bytes(b'1')
            return
        print('[Laser] Recieved signal from ITC.')

        # signal the readiness to the main loop
        comm.send_bytes(b'0') # Ready
        while data := comm.recv():
            logging.debug("[Laser] recieved %s", data)
            action, msg = data

            match action:
                case b'0':
                    msg = float(msg)
                    # print(f"Child recieved to perform action with {msg = }\n\t{float(msg) = }")
                    # recieved a msg to set current value
                    itc.sour_curr = msg
                    print(f"[Laser] Successfully set the current to {msg:.5f}A.")
                    # wait for optical circuit to flush
                    sleep(2)

                    # Signal completion
                    comm.send_bytes(b'0')


                    # measure if told to do so
                    if not(IVhystersis is None):
                        print('TODO')

                case b'2':
                    # close thread
                    # TODO
                    logging.info('Set-current thread has been called to close.')
                    print('[Laser] Terminating process.')
                    return

                case _:
                    raise ValueError(f"Unrecognised {action = }")

            # Reset while loop
            data = None

    print("[Laser] Process did not terminate at intended exit.")
    comm.send_bytes(b'1')
    return

def test_set_current_thread():
    itc_rscs = filter(lambda x: 'M0' in x, list_resources())
    itc_rsc = 'USB::4883::32842::M00435743::INSTR'
    if itc_rsc not in itc_rscs:
        raise ValueError("Thorlabs Laser Driver could not be found!")

    ld_p_parent, ld_p_child = Pipe()

    ld_proc = Process(
        target=set_current_thread,
        args=(ld_p_child, itc_rsc)
    )
    ld_proc.start()
    print("[Driver] set-laser process started.")
    sleep(1)
    assert ld_p_parent.recv_bytes() == b'0', "LD process did not succeed."
    print("[Driver] set-laser responded ok.")

    sleep(6)

    for laser_curr in [b"0.02", b"0.3500"]:
        ld_p_parent.send((b'0', laser_curr))
        print(f"[Driver] Signal laser current to be {laser_curr}A.")

        parent_recieved = ld_p_parent.recv_bytes()
        assert parent_recieved==b'0' , f"Error obtained on ld_p_parent: {parent_recieved = }"
        print("[Driver] Child process has set laser current")

        sleep(2)

    ld_p_parent.send((b'2', b'0')) # Filler message
    ld_proc.join()
    ld_p_parent.close()
    ld_p_child.close()

    print("Exiting...")

if __name__ == "__main__":
    print("Running set_current.py as main!")
    test_set_current_thread()
