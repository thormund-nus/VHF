## characterise_vhf

*Date: 11 Dec 2023*

By this point in time, several laser diodes have been rotated around, with
several photodiodes (unshielded fast Thorlabs photodiode, shielded GAP100,
linear Thorlabs Avalanche photodiode) have been tested, but erratic dips in
$r(t)$ continue to occur. We have to strictly rule out the board malfunctioning
as being the cause of the strong drops in $r(t)$.

### Development cycle

1. ./test/test_collect.py
exists to test the concrete implementation `run_vhf_dep/collect.py` of
`VHF/multiprocess/vhf.py`.


---

We utilise the function generator to ensure this is the case.

Data is saved to ...



## run_vhf_with_func_gen

*Date: 2 May 2023*

This subfolder is utilised for characterising the flashed firmware (r14 of svn)  
(flased on 4 Apr 2023) behaviour with respect to a well-defined input.

Files will be saved in binary format for concurrent development of binary 
parser.

The output files generated here will be used to determine if the IQ unwrapping
is done correctly.

## obtainPhaseWithIncreasingVoltage.py

November 2023 - 2024

This script is a proof of concept for multiprocessed data collection and
processing simultaneously. This was created partially during a suggestion to
understand the radius of the VHF board as a function of the RF@140MHz power
into the VHF's ADC.

Further TODOs:
1. Seperation of concerns between this file as the main driver that spawns all
   others processes.
2. Have child processes that are processing its collected data instead spawn a
   child process that does the plotting to avoid matplotlib memory leaks.
3. Long term goal: Have a singular sampling process instead, and hand off the
   temp_file variable to other processes that perform the analysis/plotting
   instead.
