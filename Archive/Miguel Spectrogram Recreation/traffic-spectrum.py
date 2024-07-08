# With the generation of Figure 37 in Miguel's report much better understood,
# it is now possible to generate a array-cropped output to pass to
# GNUPlot/matplotlib for SVG format.

from datetime import timedelta
from matplotlib import colors
from matplotlib import gridspec
from matplotlib import mlab
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib import transforms
import math
import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray
from pathlib import Path
# from pynverse import inversefunc
import sys
from time import perf_counter_ns
from typing import Callable, Optional
module_path = str(Path(__file__).parents[2])
if module_path not in sys.path:
    sys.path.append(module_path)
from VHF.parse import VHFparser

cond_type = Optional[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.bool_]]]


def block_avg(my_arr: np.ndarray, N: int):
    """Returns a block average of 1D my_arr in blocks of N."""
    if N == 1:
        return my_arr
    return np.mean(my_arr.reshape(np.shape(my_arr)[0]//N, N), axis=1)


def block_avg_tail(my_arr: np.ndarray, N: int):
    """Block averages the main body, up till last block that might contain less
    than N items. Assumes 1D."""
    # Check for 1D array
    assert my_arr.ndim == 1

    if N == 1:
        return my_arr
    if np.size(my_arr) % N == 0:
        return block_avg(my_arr, N)
    else:
        print("Chosen indices does not give nice block!")
        result = np.zeros(np.size(my_arr)//N + 1)
        result[:-1] = block_avg(my_arr[:np.size(my_arr)//N * N ], N)
        result[-1] = np.mean(my_arr[np.size(my_arr)//N * N :])
        return result


def crop_indices(t_start: int, t_end: int, sampl_freq: float,
                 squash_factor: int, diff_delta: int) -> tuple[int, int]:
    """Obtain index that contains start and end time that gives nice blocks for
    block averaging.

    Args
    -----
    t_start: int
        time start in seconds. Ideally in seconds.
    t_end: int
    sampl_freq: float
        sampling frequency
    squash_factor: int
        Denotes the total compression factor that will be utilised in
        generating spectrogram. For example, if block averaging is done on N =
        5 then N = 10, squash_factor = 50.
    diff_delta: int
        since np.diff gives 1 less element, specify diff_delta to offset
        end_index
    """
    i_s = t_start * sampl_freq
    i_start = int(i_s)
    if abs(i_start - i_s) > 0.5:
        i_start -= 1
    delta_seconds = t_end - t_start
    i_d = delta_seconds * sampl_freq
    i_delta = int(i_d)
    if abs(i_d - i_delta) > 0.5:
        i_delta += 1
    # + 1 to account for [,) slicing
    i_end = i_start + i_delta + squash_factor + diff_delta
    return (i_start, i_end)


def spectrogram_data_array(
    data: NDArray[np.float64],
    samp_rate,
    per_lap=0.9,
    wlen=None,
    dbscale=False,
    mult=None,  # Obspy defaults to 8, we put None
) -> tuple[
    NDArray[np.float64],
    NDArray,
    NDArray,
    tuple[np.float64, np.float64, np.float64, np.float64]
]:
    """Returns f, t, Sxx, ax_extent while using obspy internals (wlen and
    overlap).

    This version is a modification of Obspy-1.4.0's spectrogram function, with
    modifications from Obspy-1.2.2 and some personal requirements.

    Args
    -----
    data: Input data
    samp_rate: [float] Sample rate in Hz
    per_lap: [float] Perfectage of overlap of sliding windows, ranging from 0
        to 1. High overlaps take a long time to compute.
    wlen: [int|float] Window length for fft in seconds. If this parameter is too
        small, the calculation will take forever. If None, it defaults to
        (samp_rate/100.0). [Obspy-1.2.2 default]
    dbscale: [bool] Uses 10 * log10 colors if True, uses Sqrt color otherwise.
    mult: [float] Pad zeros to mult * wlen. This will make the spectrogram
        smoother.


    Returns
    -----
    specgram:
    freq:
    time:
    ax_extent: Bounding box to be used directly in pyplot's imshow
    """
    def _nearest_pow_2(x):
        """
        Find power of two nearest to x

        >>> _nearest_pow_2(3)
        2.0
        >>> _nearest_pow_2(15)
        16.0

        :type x: float
        :param x: Number
        :rtype: int
        :return: Nearest power of 2 to x
        """
        a = math.pow(2, math.ceil(np.log2(x)))
        b = math.pow(2, math.floor(np.log2(x)))
        if abs(a - x) < abs(b - x):
            return a
        else:
            return b

    # Utilise Obspy 1.4 internal code, up til matplotlib.specgram, except
    # guard clauses as given within the internals are to be given explicitly
    # import matplotlib.pyplot as plt
    # enforce float for samp_rate
    samp_rate = float(samp_rate)
    # modified guard clause
    if not wlen:
        # wlen = 128 / samp_rate  # obspy 1.4.0
        wlen = samp_rate / 100  # obspy 1.2.2

    npts = len(data)

    # nfft needs to be an integer, otherwise a deprecation will be raised
    # XXX add condition for too many windows => calculation takes for ever
    nfft = int(_nearest_pow_2(wlen * samp_rate))

    if npts < nfft:
        msg = (f'Input signal too short ({npts} samples, window length '
               f'{wlen} seconds, nfft {nfft} samples, sampling rate '
               f'{samp_rate} Hz)')
        raise ValueError(msg)

    if mult is not None:
        mult = int(_nearest_pow_2(mult))
        mult = mult * nfft
    nlap = int(nfft * float(per_lap))
    data = data - data.mean()
    # end = npts / samp_rate

    specgram: NDArray  # helping out debuggers
    freq: NDArray[np.float64]
    time: NDArray[np.float64]
    specgram, freq, time = mlab.specgram(data, Fs=samp_rate, NFFT=nfft,
                                         pad_to=mult, noverlap=nlap)
    if dbscale:
        specgram = 10 * np.log10(specgram[1:, :])
    else:
        specgram = np.sqrt(specgram[1:, :])
    freq = freq[1:]

    # vmin, vmax = clip
    # if vmin < 0 or vmax > 1 or vmin >= vmax:
    #     msg = "Invalid parameters for clip option."
    #     raise ValueError(msg)
    # _range = float(specgram.max() - specgram.min())
    # vmin = specgram.min() + vmin * _range
    # vmax = specgram.min() + vmax * _range
    # norm = Normalize(vmin, vmax, clip=True)

    # if not axes:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    # else:
    #     ax = axes

    # calculate half bin width
    halfbin_time = (time[1] - time[0]) / 2.0
    halfbin_freq = (freq[1] - freq[0]) / 2.0

    # kwargs = {'zorder': zorder}
    # if log:
    #     # pcolor expects one bin more at the right end
    #     freq = np.concatenate((freq, [freq[-1] + 2 * halfbin_freq]))
    #     time = np.concatenate((time, [time[-1] + 2 * halfbin_time]))
    #     # center bin
    #     time -= halfbin_time
    #     freq -= halfbin_freq
    #     # Log scaling for frequency values (y-axis)
    #     # ax.set_yscale('log')
    #     # Plot times
    #     # ax.pcolormesh(time, freq, specgram, norm=norm, **kwargs)
    #     return specgram, (time, freq)
    # else:
    # this method is much much faster!
    specgram = np.flipud(specgram)
    # center bin
    extent = (time[0] - halfbin_time, time[-1] + halfbin_time,
              freq[0] - halfbin_freq, freq[-1] + halfbin_freq)
    # ax.imshow(specgram, interpolation="nearest", extent=extent, **kwargs)
    return specgram, freq, time, extent

    # set correct way of axis, ... more trimmed out from Obspy-1.4


def spectrogram_crop(
    Sxx: NDArray, f: NDArray, t: NDArray, extent: tuple, f_cond: cond_type, t_cond: cond_type
) -> tuple[NDArray, NDArray, NDArray, tuple]:
    """Function to crop Sxx to within f and t bounds (in natural units)."""

    # Allow for accepting no filtering
    if f_cond is None:
        f_cond = lambda x: np.full(np.shape(x), True)  # noqa: E731
    if t_cond is None:
        t_cond = lambda x: np.full(np.shape(x), True)  # noqa: E731
    if f_cond is None and t_cond is None:
        return Sxx, f, t, extent
    t_r = t_cond(t)
    f_r = f_cond(f)
    f = f[f_r]
    t = t[t_r]
    halfbin_time = (t[1] - t[0]) / 2.0
    halfbin_freq = (f[1] - f[0]) / 2.0
    extent = (t[0] - halfbin_time, t[-1] + halfbin_time,
              f[0] - halfbin_freq, f[-1] + halfbin_freq)
    return Sxx[f_r, ...][..., t_r], f, t, extent


def trunc_cmap(cmap_name: str, minval: float = 0.0, maxval: float = 1.0, n: int = 255):
    """Cmap converts numbers between 0 and 1 to a color. Here,
    we take only a slice of matplotlib's cmap instead."""
    cm = plt.get_cmap(cmap_name)
    if minval == 0.0 and maxval == 1.0:
        return cm
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cm.name, a=minval, b=maxval),
        cm(np.linspace(minval, maxval, n)))
    return new_cmap


def main():
    base_dir = Path(__file__).parent
    data_dir = base_dir.joinpath('Data')

    sparse_num_phase = 5
    sparse_num_velocity = 10

    file1 = data_dir.joinpath('2023-07-06T14:04:22.913767_s499_q268435456_F0_laser2_3674mA_20km.bin')
    parsed1 = VHFparser(
        file1,
        plot_start_time=timedelta(seconds=150),
        plot_duration=timedelta(seconds=400),
    )
    phase1 = parsed1.reduced_phase

    phase1_avg = block_avg_tail(phase1, sparse_num_phase)
    velocity1 = np.diff(phase1_avg) * (parsed1.header["sampling freq"]/sparse_num_phase) # * 1550e-9
    velocity1_avg = block_avg_tail(velocity1, sparse_num_velocity)

    # account for butchering of parsed headers
    parsed1.header["sampling freq"] /= (sparse_num_phase * sparse_num_velocity) # after the dust has settled

    Sxx, f, t, ax_extent = spectrogram_data_array(
        velocity1_avg,
        parsed1.header['sampling freq'],
        wlen=parsed1.header['sampling freq']/100,
        per_lap=0.90  # Following miguel's default
    )

    # Array crop to our needs  SEE COMMENTS 22 lines down
    # Sxx, f, t = spectrogram_crop(Sxx, f, t, lambda f: f<=30, lambda t: t<=350)
    Sxx, f, t, spec_ax_extent = spectrogram_crop(Sxx, f, t, ax_extent, None, None)
    print(f"[Cmap mapping] {np.max(Sxx) = }, {np.min(Sxx) = }")

    matplotlib = True
    if matplotlib:

        # my_norm = colors.FuncNorm((c_forward, c_inv), vmin=180, vmax=440)
        my_norm = None

        # Plot
        # 1. Take a relevant slice of the color bar so that the truncated
        # spectrogram has similar light streaks to what Miguel has done
        # 2. Specify that all values above 440 for vmax are to be mapped to the
        # upper end of cmap.
        # 3. aspect = auto to allow for set_xlim, set_ylim to not squash the ax
        # IMPORTANT: Some reason??: Cropping Sxx prior to the desired (xlim,
        # ylim) destroys the signature plot?
        fig = plt.figure()
        gs = gridspec.GridSpec(
            nrows=2, ncols=2,
            figure=fig,
            wspace=0.1,
            height_ratios=[2, 3], width_ratios=[1, .03]
        )
        ph_ax = plt.subplot(gs[0, 0])
        spec_ax = plt.subplot(gs[1, 0], sharex=ph_ax)
        colb_ax = plt.subplot(gs[1, 1])

        # plot spectrogram
        cmappable = spec_ax.imshow(
            Sxx, cmap=trunc_cmap('terrain', 0, 0.25),
            norm=my_norm, interpolation="nearest",
            vmax=440,
            extent=spec_ax_extent,
            aspect='auto'
        )
        spec_ax.grid(False)

        # add colorbar
        plt.colorbar(cmappable, cax=colb_ax, shrink=1.0, extend='max')

        # plot phase(time)
        ph_ax.plot(np.arange(len(phase1))/2e4, phase1/1e3,
            color='#0000ff', linewidth=0.2)

        # add secondary axis: displacement
        dis_ax = ph_ax.secondary_yaxis('right', functions=(
            lambda x: x * 1.55/1.5,
            lambda x: x / (1.55/1.5)
        ))

        spec_ax.tick_params(axis='both', which='major', labelsize=6)
        colb_ax.tick_params(axis='both', which='major', labelsize=6)
        ph_ax.tick_params(axis='both', which='major', labelsize=6)
        dis_ax.tick_params(axis='both', which='major', labelsize=6)

        spec_ax.set_xlabel('Time (s)', fontsize=6)
        spec_ax.set_ylabel('Frequency (Hz)', fontsize=6)
        colb_ax.set_ylabel('Intensity (a.u.)', fontsize=6)
        ph_ax.set_ylabel(r'$\phi_d$ ($2\pi \cdot 10^3$ rad)',
                         fontsize=6, usetex=True)
        dis_ax.set_ylabel(r'$\Delta L$ (mm)', fontsize=6, usetex=True)
        spec_ax.set_xlim(ax_extent[0], 350.0)
        spec_ax.set_ylim(ax_extent[2], 30)

        fig.set_size_inches(fig_width:=0.495*(8.3-2*0.6), 1.0*fig_width) # A4 paper is 8.3 inches by 11.7 inches
        fig.subplots_adjust(left=0.125, bottom=0.095, right=0.880, top=0.955)

        # add labels
        tr = transforms.ScaledTranslation(-27/72, 2/72, fig.dpi_scale_trans)
        ph_ax.text(0.0, 1.0, '(a)', transform=ph_ax.transAxes + tr,
            fontsize='small', va='bottom', fontfamily='sans')
        spec_ax.text(0.0, 1.0, '(b)', transform=spec_ax.transAxes + tr,
            fontsize='small', va='bottom', fontfamily='sans')

        # add circling
        for cent in [(6.8, 10.75), (72, 11.8), (187.8, 10.3), (223, 10.3), (266, 11.74), (300, 12.26), (323, 12.07)]:
            c = patches.Ellipse(cent, width=(ell_w:=18),
                    height=(np.diff(spec_ax.get_ylim())/np.diff(spec_ax.get_xlim()))*ell_w*1.2,
                    linewidth=1, edgecolor='#ff0000', facecolor='none')
            spec_ax.add_patch(c)

        # plt.show(block=True)
        time_start = perf_counter_ns()
        fig.savefig(base_dir.joinpath('spectrum_miguel.png'), dpi=300, format='png')
        time_end = perf_counter_ns()
        plt.close()
        print(f"Save fig took: {(time_end-time_start)/1e6:.3f}ms.")
    
    gnuplot = False
    if gnuplot:
        Sxx = TODO


if __name__ == '__main__':
    main()
