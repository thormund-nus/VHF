# Written with svn r14 in mind, where files are now explicitly set to save in
# binary output

import sys
from matplotlib import pyplot as plt
import numpy as np
from os import path, listdir
from typing import List, Tuple, Callable
from pathlib import Path
from parseVHF import VHFparser  # relative import
import scipy as sp


def get_files(start_path: Path) -> List[Path] | Path:
    start_path = Path(start_path)
    print(f"\nCurrent directory: \x1B[34m{start_path}\x1B[39m")
    start_path_contents = list(start_path.iterdir())

    # Notify if empty
    if len(start_path_contents) < 1:
        print("This folder is empty.")

    # Print with or without [Dir]
    if any(f.is_dir() for f in start_path_contents):
        dir_marker = "[Dir] "
        marker_len = len(dir_marker)
        for i, f in enumerate(start_path_contents):
            # Print only the filename
            print(
                f"{i:>3}: \x1B[31m{dir_marker if f.is_dir() else ' '*marker_len}\x1B[39m{f.name}"
            )
    else:
        for i, f in enumerate(start_path_contents):
            print(f"{i:>3}: {f.name}")

    while True:
        # Get user input
        try:
            my_input = input("Select file to parse, -1 for all files: ")
        except KeyboardInterrupt:
            print("\nKeyboard Interrupt recieved. Exiting.")
            sys.exit(0)

        # Convert user input into Path for return
        try:
            my_input = int(my_input)
            if -1 <= my_input < len(start_path_contents):
                break
            if my_input >= len(start_path_contents):
                print("Value given is too large!")
            else:
                print("Value given is too small!")
        except ValueError:
            print("Invalid value given!")

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


def user_input_bool(prompt: str) -> bool:
    while True:
        # Get user input
        try:
            my_input = input(f"{prompt} \x1B[31m(y/n)\x1B[39m: ")
        except KeyboardInterrupt:
            print("\nKeyboard Interrupt recieved. Exiting.")
            sys.exit(0)

        # Convert user input into Path for return
        try:
            my_input = my_input[0].lower()
            if my_input == 'y':
                return True
            elif my_input == 'n':
                return False
            else:
                print(f"Invalid value!")
        except ValueError:
            print("Invalid value given!")


def get_phase(o: VHFparser) -> np.ndarray:
    """Return phi/2pi; phi:= atan(Q/I).

    Input
    -----
    o: VHFparser class, initialised with file being parsed.

    Returns
    -----
    phase: Divided by 2pi
    """
    if o.reduced_phase is None:
        phase = -np.arctan2(o.i_arr, o.q_arr)
        phase /= 2 * np.pi
        phase -= o.m_arr
        o.reduced_phase = phase
    return o.reduced_phase


def get_radius(o: VHFparser) -> np.ndarray:
    """Return norm of (I, Q)."""
    result = np.hypot(o.i_arr, o.q_arr)
    lower_lim = int(1e5)
    print(
        f"Radius Mean: {np.mean(result[lower_lim:])}\nRadius Std Dev: {np.std(result[lower_lim:])}"
    )
    return result


def get_spec(o: VHFparser) -> Tuple[np.ndarray, np.ndarray]:
    """(f, Pxx_den) from x(t) signal, by periodogram method."""
    p = o.reduced_phase
    # f, spec = sp.signal.welch(2*np.pi*p, fs=o.header['sampling freq'])
    f, spec = sp.signal.periodogram(
        2 * np.pi * p, fs=o.header["sampling freq"])
    return f, spec


def plot_rad_spec(radius: bool, spectrum: bool) -> Callable:
    """
    Yield desired function for plotting phase, radius and spectrum.

    Input
    -----
    radius: bool
        Plots sqrt(I^2+Q^2)
    spectrum: bool
        Plots periodogram
    """

    # consistent plot methods across functions returned
    def scatter_hist(y, ax_histy):
        binwidth = (np.max(y) - np.min(y)) / 100
        bins = np.arange(np.min(y) - binwidth, np.max(y) + binwidth, binwidth)
        ax_histy.hist(y, bins=bins, orientation="horizontal")

    def plotphi(ax, o: VHFparser, phase: np.ndarray):
        t = np.arange(len(phase)) / o.header["sampling freq"]
        ax.plot(t, phase, color="mediumblue", linewidth=0.2, label="Phase")
        # ax.plot(-parsed.m_arr, color='red', linewidth=0.2, label='Manifold')
        ax.set_ylabel(r"$\phi_d$/2$\pi$rad", usetex=True)
        ax.set_xlabel(r"$t$/s", usetex=True)

    def plotr(r_ax, o: VHFparser, phase: np.ndarray):
        t = np.arange(len(phase)) / o.header["sampling freq"]
        lower_lim = np.max(np.where(t < 0.001))
        r_ax.plot(
            t[lower_lim:],
            rs:= get_radius(o)[lower_lim:],
            color="mediumblue",
            linewidth=0.2,
            label="Radius",
        )
        r_ax.set_ylabel(r"$r$/ADC units", usetex=True)
        r_ax.set_xlabel(r"$t$/s", usetex=True)
        r_ax_histy = r_ax.inset_axes([1.01, 0, 0.08, 1], sharey=r_ax)
        r_ax_histy.tick_params(axis="y", labelleft=False, length=0)
        scatter_hist(rs, r_ax_histy)

    def plotsp(sp_ax, o: VHFparser):
        sp_ax.scatter(
            *get_spec(o),
            color="mediumblue",
            linewidth=0.2,
            label="Spectrum Density",
            s=0.4,
        )
        sp_ax.set_ylabel(
            "Phase 'Spec' Density /rad$^2$Hz$^-$$^1$", usetex=True)
        sp_ax.set_xlabel("$f$ [Hz]", usetex=True)
        sp_ax.set_yscale("log")
        sp_ax.set_xscale("log")

    # functions to return
    def plot_phi(o: VHFparser, phase: np.ndarray):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plotphi(ax, o, phase)
        return fig

    def plot_phi_and_rad(o: VHFparser, phase: np.ndarray):
        fig, [phi_ax, r_ax] = plt.subplots(nrows=2, ncols=1, sharex=True)
        plotphi(phi_ax, o, phase)
        plotr(r_ax, o, phase)
        return fig

    def plot_phi_and_spec(o: VHFparser, phase: np.ndarray):
        fig, [phi_ax, sp_ax] = plt.subplots(nrows=2, ncols=1, sharex=False)
        plotphi(phi_ax, o, phase)
        plotsp(sp_ax, o)
        return fig

    def plot_all(o: VHFparser, phase: np.ndarray):
        fig, [phi_ax, r_ax, sp_ax] = plt.subplots(
            nrows=3, ncols=1, sharex=False)
        phi_ax.sharex(r_ax)
        plotphi(phi_ax, o, phase)
        plotr(r_ax, o, phase)
        plotsp(sp_ax, o)
        return fig

    if not radius and not spectrum:
        return plot_phi
    elif radius and not spectrum:
        return plot_phi_and_rad
    elif not radius and spectrum:
        return plot_phi_and_spec
    elif radius and spectrum:
        return plot_all


def main():
    print("Please select files intended for plotting.")
    files = get_files(start_path=path.join(path.dirname(__file__)))

    print(f"{files = }")
    if files is None:
        print("No files!")
        return

    if isinstance(files, list):
        file: Path = files[0]
        print("This script has not implemented plotting and saving all files.")
    else:
        file: Path = files

    parsed = VHFparser(file)
    print(f"Debug {parsed.header = }")
    plot_radius = user_input_bool("Do you want to plot the radius?")
    plot_spec = user_input_bool("Do you want to plot the spectrum?")

    phase = get_phase(parsed)
    fig = plot_rad_spec(plot_radius, plot_spec)(parsed, phase)

    view_const = 2.3
    fig.legend()
    fig.set_size_inches(view_const * 0.85 *
                        (8.25 - 0.875 * 2), view_const * 2.5)
    fig.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    main()
