# With the generation of Figure 37 in Miguel's report much better understood, 
# it is now possible to generate a array-cropped output to pass to 
# GNUPlot/matplotlib for SVG format.

import os, sys
from pathlib import Path
module_path = str(Path(__file__).parents[2])
if module_path not in sys.path:
    sys.path.append(module_path)

from datetime import timedelta
from obspy.core import Stream, Trace, Stats
from obspy.imaging.spectrogram import spectrogram
from scipy.signal import spectrogram as sp_spectrogram
from matplotlib import mlab
from matplotlib import pyplot as plt
import math
import numpy as np
from parseVHF import VHFparser  # relative import
from plot_VHF_output import get_radius, get_spec, get_phase, plot_rad_spec
from pathlib import Path

def block_avg(my_arr: np.ndarray, N: int):
    """Returns a block average of 1D my_arr in blocks of N."""
    if N == 1:
        return my_arr
    return np.mean(my_arr.reshape(np.shape(my_arr)[0]//N, N), axis=1)

def block_avg_tail(my_arr: np.ndarray, N: int):
    """Block averages the main body, up till last block that might 
    contain less than N items. Assumes 1D."""
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

def crop_indices(t_start, t_end, sampl_freq, squash_factor, diff_delta):
    """Obtain index that contains start and end time that gives nice 
    blocks for block averaging.
    
    Args
    -----
    t_start: int
        time start in seconds. Ideally in seconds.
    t_end: int
    sampl_freq: float
        sampling frequency
    squash_factor: int
        Denotes the total compression factor that will be utilised in 
        generating spectrogram. For example, if block averaging is done 
        on N = 5 then N = 10, squash_factor = 50.
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

def spectrogram_data_array(data, samp_rate, wlen=None, per_lap=0.9, dbscale=False, log=False):
    """Returns f, t, Sxx, ax_extent while using obspy internals (wlen and overlap).
    
    Args
    -----
    data:
    samp_rate:
    wlen:
    per_lap:
    dbscale:
    log:

    Returns
    -----
    specgram:
    freq:
    time:
    ax_extent: Bounding box to be used directly in pyplot's imshow
    """

    # Utilise Obspy 1.4 internal code, up til matplotlib.specgram, except
    # guard clauses as given within the internals are to be given explicitly

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
            
    # enforce float for samp_rate
    samp_rate = float(samp_rate)
    # modified guard clause
    if not wlen:
        raise ValueError("wlen is to be explicit!")
    npts = len(data)
    nfft = int(_nearest_pow_2(wlen * samp_rate))

    if npts < nfft:
        msg = (f'Input signal too short ({npts} samples, window length '
               f'{wlen} seconds, nfft {nfft} samples, sampling rate '
               f'{samp_rate} Hz)')
        raise ValueError(msg)
    # modified guard clause
    mult = None
    nlap = int(nfft * float(per_lap))
    data = data - data.mean()
    specgram, freq, time = mlab.specgram(data, Fs=samp_rate, NFFT=nfft,
                                         pad_to=mult, noverlap=nlap)
    if dbscale:
        specgram = 10 * np.log10(specgram[1:, :])
    else:
        specgram = np.sqrt(specgram[1:, :])
    freq = freq[1:]

    # calculate half bin width
    halfbin_time = (time[1] - time[0]) / 2.0
    halfbin_freq = (freq[1] - freq[0]) / 2.0
    ax_extent = (time[0] - halfbin_time, time[-1] + halfbin_time,
                 freq[0] - halfbin_freq, freq[-1] + halfbin_freq)
    return specgram, freq, time, ax_extent

def main():
    base_dir = Path(__file__).parent
    data_dir = base_dir.joinpath('Data')
    view_const = 5
    sparse_num_phase = 5
    sparse_num_velocity = 10

    file1 = data_dir.joinpath('2023-07-06T14:04:22.913767_s499_q268435456_F0_laser2_3674mA_20km.bin')
    parsed1 = VHFparser(file1)

    start_time = 200 # seconds
    end_time = 700
    start_index, end_index = crop_indices(
        start_time, 
        end_time, 
        int(parsed1.header["sampling freq"]), 
        sparse_num_phase * sparse_num_velocity,
        sparse_num_phase
        )

    print(f"[Debug] {start_index = }, {end_index = }")

    # forcefully cut down on memory
    parsed1.i_arr = parsed1.i_arr[:end_index]
    parsed1.q_arr = parsed1.q_arr[:end_index]
    parsed1.m_arr = parsed1.m_arr[:end_index]
    phase1 = get_phase(parsed1)[start_index:end_index]

    phase1_avg = block_avg_tail(phase1, sparse_num_phase)
    velocity1 = np.diff(phase1_avg) * (parsed1.header["sampling freq"]/sparse_num_phase) # * 1550e-9
    velocity1_avg = block_avg_tail(velocity1, sparse_num_velocity)

    # account for butchering of parsed headers
    parsed1.header["sampling freq"] /= (sparse_num_phase * sparse_num_velocity) # after the dust has settled
    parsed1.header["Time start"] += timedelta(seconds=start_time)
    t = np.arange(len(velocity1_avg)) / parsed1.header["sampling freq"]

    shoehorn_header = Stats()
    shoehorn_header.sampling_rate = parsed1.header['sampling freq']
    shoehorn_header.starttime = parsed1.header['Time start']
    shoehorn_header.network = 'QITLAB'
    shoehorn_header.location = 'NUS'
    shoehorn_header.npts = len(velocity1_avg)
    # tr = Trace(data=velocity1_avg, header=shoehorn_header)
    # tr.spectrogram(cmap = 'terrain', wlen=shoehorn_header.sampling_rate / 100.0, per_lap=0.90)
    # plt.show()
    Sxx, f, t, ax_extent = spectrogram_data_array(velocity1_avg, shoehorn_header.sampling_rate,
                                                shoehorn_header.sampling_rate/100, 0.90)

    # why does obspy imaging do flipud? line 183
    Sxx = np.flipud(Sxx)

    # Array crop to our needs

    # Plot
    fig, ax = plt.subplots()
    ax.imshow(Sxx, interpolation="nearest", extent=ax_extent, cmap='terrain')

    ax.axis('tight')
    ax.grid(False)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')

    fig.savefig('_tmp.png')

if __name__ == '__main__':
    main()