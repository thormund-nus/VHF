# Written with svn r14 in mind, where files are now explicitly set to save in 
# binary output

from matplotlib import pyplot as plt
import numpy as np
from os import path, listdir
import sys
from typing import List, Optional
from pathlib import Path
from parseVHF import VHFparser # relative import

def get_files(start_path: Path) -> List[Path] | Path:
    start_path = Path(start_path)
    print(f"\nCurrent directory: \x1B[34m{start_path}\x1B[39m")
    start_path_contents = list(start_path.iterdir())

    # Notify if empty
    if len(start_path_contents) < 1:
        print('This folder is empty.')

    # Print with or without [Dir]
    if any(f.is_dir() for f in start_path_contents):
        dir_marker = '[Dir] '
        marker_len = len(dir_marker)
        for i, f in enumerate(start_path_contents):
            # Print only the filename
            print(f"{i:>3}: \x1B[31m{dir_marker if f.is_dir() else ' '*marker_len}\x1B[39m{f.parts[-1]}")
    else:
        for i, f in enumerate(start_path_contents):
            print(f"{i:>3}: {f.parts[-1]}")

    while True:
        # Get user input
        try:
            my_input = input('Select file to parse, -1 for all files: ')
        except KeyboardInterrupt:
            print('\nKeyboard Interrupt recieved. Exiting.')
            sys.exit(0)
        
        # Convert user input into Path for return
        try:
            my_input = int(my_input)
            if -1 <= my_input < len(start_path_contents):
                break 
            if my_input >= len(start_path_contents):
                print(f"Value given is too large!")
            else:
                print(f"Value given is too small!")
        except ValueError:
            print('Invalid value given!')
    
    if my_input == -1:
        result = list(filter(lambda x: x.is_file(), start_path_contents))
    else:
        file_chosen = start_path_contents[my_input]
        if file_chosen.is_file():
            result = file_chosen
        elif file_chosen.is_dir():
            result = get_files(start_path.joinpath(file_chosen))
        else:
            print("Selection is neither file nor directory.")
            sys.exit(0)
    
    return result

def main():
    print('Please select files intended for plotting.')
    files = get_files(start_path=path.join(path.dirname(__file__)))
    
    print(f"{files = }")
    if files is None:
        print("No files!")
        return

    if type(files) == list:
        file: Path = files[0]
        print('This script has not implemented plotting and saving all files.')
    else:
        file: Path = files
    
    parsed = VHFparser(file)
    phase = np.arctan2(parsed.q_arr, parsed.i_arr)
    phase /= 2*np.pi
    phase += 0.5  # account for I' and Q'
    phase -= parsed.m_arr

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(phase, color = 'mediumblue', linewidth= 0.2, label='Phase')
    ax.plot(-parsed.m_arr, color = 'red', linewidth= 0.2, label='Manifold')
    ax.set_ylabel(r'$\phi_d$/2$\pi$rad', usetex= True)
    ax.set_xlabel(r'$t$/s', usetex= True)
    # ax.set_xlim([-1, 1600])

    view_const = 2.3
    fig.legend()
    fig.set_size_inches(view_const*0.85*(8.25-0.875*2), view_const*2.5) 
    fig.tight_layout()
    plt.show(block=True)
    
if __name__ == '__main__':
    main()
