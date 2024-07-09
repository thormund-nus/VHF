"""Modified version of obspy-1.4.0's imaging/spectrogram.py, with some 1.2.2
and personal defaults."""
import math
from matplotlib import mlab
import numpy as np
from numpy.typing import NDArray

type ExtentRet = tuple[float, float, float, float]


def spectrogram(
    data: NDArray[np.float64],
    samp_rate,
    per_lap=0.9,
    wlen=None,
    dbscale=False,
    mult=None,  # Obspy defaults to 8, we put None
) -> tuple[NDArray[np.float64], NDArray, NDArray, ExtentRet]:
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
    wlen: [int|float] Window length for fft in seconds. If this parameter is
        too small, the calculation will take forever. If None, it defaults to
        (samp_rate/100.0). [Obspy-1.2.2 default]
    dbscale: [bool] Uses 10 * log10 colors if True, uses Sqrt color otherwise.
        dbscale being False is the same as getting the amplitude (V/√Hz),
        whilst being True is the same as getting the Power (V²/Hz).
        Further reading: https://docs.scipy.org/doc/scipy/tutorial/signal.html#tutorial-stft-legacy-stft  # noqa:E501
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
    # set wlen from samp_rate if not specified otherwise
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

    # Here we call not plt.specgram as this already produces a plot
    # matplotlib.mlab.specgram should be faster as it computes only the
    # arrays
    # XXX mlab.specgram uses fft, would be better and faster use rfft
    specgram: NDArray  # helping out debuggers
    freq: NDArray[np.float64]
    time: NDArray[np.float64]
    specgram, freq, time = mlab.specgram(data, Fs=samp_rate, NFFT=nfft,
                                         pad_to=mult, noverlap=nlap)
    # reading: https://docs.scipy.org/doc/scipy/tutorial/signal.html#continuous-time-sine-signal  # noqa: E501
    # ref: https://github.com/matplotlib/matplotlib/blob/v3.9.1/lib/matplotlib/mlab.py#L638  # noqa: E501
    # 1. mode here defaults to 'psd'
    #   see: https://github.com/matplotlib/matplotlib/blob/v3.9.1/lib/matplotlib/mlab.py#L246  # noqa: E501
    #   This alone would give the Energy Spectral density of abs(X(f))², with units V²s/Hz.  # noqa: E501
    # 2. scale_by_freq here used is True
    #   see: https://github.com/matplotlib/matplotlib/blob/v3.9.1/lib/matplotlib/mlab.py#L284  # noqa: E501
    #   This now gives the Power Spectral density of abs(X(f))²/τ, with units V²/Hz.  # noqa: E501


    if len(time) < 2:
        msg = (f'Input signal too short ({npts} samples, window length '
               f'{wlen} seconds, nfft {nfft} samples, {nlap} samples window '
               f'overlap, sampling rate {samp_rate} Hz)')
        raise ValueError(msg)

    # db scale and remove zero/offset for amplitude
    if dbscale:
        specgram = 10 * np.log10(specgram[1:, :])
    else:
        specgram = np.sqrt(specgram[1:, :])
        # This now gives the Amplitude Spectral density with units V/√Hz.
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
