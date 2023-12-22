# Multithreading

This archive folder aims to understand what it takes to have a child
thread perform a NAS write while the parent thread performs computation.

The key idea being utilised is to not pipe the entire STDOUT data across
thread boundaries, but to pass just a "file pointer". This is done using a
tmpfs.


## References
1. https://docs.python.org/3/library/threading.html  
Threading
2. https://docs.python.org/3/library/tempfile.html  
In particular, we are interested with `NamedTemporaryFile`.
3. https://www.kernel.org/doc/html/latest/filesystems/tmpfs.html  
Setup of temporary filesystem in RAM.
4. https://wiki.archlinux.org/title/tmpfs  
Mount options

## Development order
1. test-basewrite.py   
This test aims to determine if the original sampling method (2MHz) being
directly written to the NAS would work. The syscall does stall after the buffer
overflows.

2. test-tempfile.py  
The use of `tempfile` library is investigated in this thread. The ability and
limitations of `tempfile` is being determined.  

3. test-filename.py
Persistence of tempfiles for NAS writing prior to memory wipe is an issue. The
use of temporary directories from `tempfile` lib is investigated, but
proved to be an issue. By this point, the concept of a `tmpfs` filesystem
being invoked from /etc/fstab is being discovered.

# Multicore

With the ability for a single proc to thread out the NAS writing, it is now
time to test if it is possible to multicore through various child processes to
coordinate the reading from VHF and NAS writing in a speedy manner.

## Order of Development
1. test-multiprocess.py and test_vhf_thread.py  
These 2 files are continuously being iterated on in the development cycle.
The goal is to eventually step the independent variable (laser
current/optical attenuator), sample the VHF board and write to NAS, start
off the next VHF thread, whilst the previous thread performs and analysis
and writes into the collated numpyz file.  
  a. test-rushed_multiprocess.py  
  To investigate the consequences of test-multiprocess.py's main driving loop
not giving ample time before kicking off the next VHF thread to start sampling.

At some point during the writibng of test-rushed_multiprocess.py, it became
clear that since NamedTemporaryFile could fail sometimes, it was best to have
what was previously the child process of the synchronizing main() thread to
instead relinquish control back to main() only after a successful iteration,
i.e.: that is to be done by attempting single tries repeatedly until able to
have obtained results in both the NAS and the tmpfs for Data Analysis (to be
written in the collated numpyz file).

2. Use of `tee`  
This might solve alot of issues to NamedTemporaryFiles dropping.

## Tangential files
1. manual_tmpFile.py  
  An attempt at trying to create our own names for NamedTemporaryFile. This is
  probably not necessary.
2. test-serial-VOA.py and test-operateAttenuator.py  
  Trying to figure out how to interface with VOA, as this has not yet been
  included in qoDevices.
