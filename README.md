# Python wrapper for VHF (svn) commands

*Author: Thormund*  
*Date: 22 March 2023*

This folder is a collection of scripts intended for invoking and parsing the 
output of teststream.c executable, as provided from svn/usbhybrid (r14).

Please be very aware of the use of symlinks in this folder.

## Usage

`set_device_mode.cpp` is compiled by the GCC to become the executable
`set_device_mode`. For it to function properly, it is necessary that this
executable has the necessary permissions. Ideally, `ls -la` would show
  
  -rwsr-xr-x 1 root   users <size> <date> <time> set_device_mode

This can be done by `chmod u+x` on the file.

Currently, several files utilise symbolic links to the board and streaming
executable, namely,

  /dev/usbhybrid0
  ~/programs/usbhybrid/apps/teststream

We note that the relevant locations to these may change in the future, and is 
therefore the use of a symbolic link helps keep things self-contained.

The symbolic links currently are

  vhf_board.softlink -> /dev/usbhybrid0
  teststream.exec -> ~/programs/usbhybrid/apps/teststream

### Red light is active

The VHF board has a red light used to indicate the FIFO being overflowed.
`clear_FIFO.py` has dependencies `set_device_mode`.

For the most part, this script is run to reset the VHF board into a state ready
for collecting 'stream' data.

### Collecting Data


### Plotting Data

  