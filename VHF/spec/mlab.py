"""Spectrogram written with reference to matplotlib's mlab.py."""

from . import logger
import functools
import math
from numbers import Number
from matplotlib.mlab import detrend, detrend_linear, detrend_mean, detrend_none
from matplotlib.mlab import window_hanning
import numpy as np
from numpy.typing import NDArray
from scipy.signal import ZoomFFT

__all__ = [
    "cz_spectrogram",
    "cz_spectrogram_base",
    "detrend",
    "detrend_linear",
    "detrend_mean",
    "detrend_none",
]


def _nearest_pow_2(x) -> int:
    a = math.pow(2, math.ceil(np.log2(x)))
    b = math.pow(2, math.floor(np.log2(x)))
    if abs(a-x) < abs(b-x):
        return int(a)
    else:
        return int(b)


@functools.lru_cache(8)
def _cachedZoomFFT(*args, **kwargs):
    """
    Create a callable zoom FFT transform function.

    This is a specialization of the chirp z-transform (`CZT`) for a set of
    equally-spaced frequencies around the unit circle, used to calculate a
    section of the FFT more efficiently than calculating the entire FFT and
    truncating.

    Input
    -----
    n : int
        The size of the signal.
    fn : array_like
        A length-2 sequence [f1, f2] giving the frequency range, or a scalar,
        for which the range [0, fn] is assumed.
    m : int, optional
        The number of points to evaluate.  Default is `n`.
    fs : float, optional
        The sampling frequency.  If `fs=10` represented 10 kHz, for example,
        then `f1` and `f2` would also be given in kHz. The default sampling
        frequency is 2, so `f1` and `f2` should be in the range [0, 1] to keep
        the transform below the Nyquist frequency.
    endpoint : bool, optional
        If True, `f2` is the last sample. Otherwise, it is not included.
        Default is False.

    Returns
    -----
    f : ZoomFFT
        Callable object `f(x, axis=-1)` for computing the zoom FFT on `x`.
    """
    return ZoomFFT(*args, **kwargs)


def cz_spectrogram_base(
    signal: NDArray,
    fn: tuple[float, float] | list[float] | float,
    fs: int,
    NFFT: int, NOVERLAP: int,
    pad_to=None,
    sides: str = "onesided",
    mode: str = "psd",
    detrend_func=None,
    window=None,
):
    """Assumes trying to obtain Power Spectrum Density. (V²/Hz)

    Input
    -----
    signal: Data to perform chirp Z spectrogram on.
    fn: Lower and upper end of Chirp Z transform.
    fs: Sampling Frequency of data.
    NFFT: Number of points in a sliding window to perform CZT on.
    NOVERLAP: Number of points of overlap between sliding windows.
    window: windowing functions to apply for filtering

    Assumptions
    -----
    scale_by_freq: PSD, scales by 2 in accordance with this mlab reference,
        when it should be doing sqrt2 if following onesided2X of STFT
        (see: https://github.com/matplotlib/matplotlib/blob/4bcc2caa42af7aba0c2d42c462fe92f0e314b7b3/lib/matplotlib/mlab.py#L299);  # noqa: E501
        np.STFT scales by sqrt2 for psd, 2 for magnitude
        (see: https://github.com/scipy/scipy/blob/87c46641a8b3b5b47b81de44c07b840468f7ebe7/scipy/signal/_short_time_fft.py#L1591)  # noqa: E501
    """
    # Guard clauses
    if not isinstance(fn, (tuple, list)) and isinstance(fn, Number):
        fn = (0, fn)
    elif isinstance(fn, (tuple, list)) and len(fn) == 2:
        for f in fn:
            if type(f) not in (int, float):
                raise ValueError("fn not well given")
        pass
    else:
        raise ValueError("fn not well given")
    if isinstance(fn, list) and len(fn) == 2:
        fn: tuple[float, float] = tuple(fn)  # pyright: ignore reportAssignmentType; Pyright cannot infer the length? # noqa: E501

    if type(fs) not in (int, float):
        raise ValueError("Sampling Frequency given is bad!")

    if not isinstance(NFFT, int):
        raise ValueError("NOVERLAP needs to be integer!")

    if not isinstance(NOVERLAP, int):
        raise ValueError("NOVERLAP needs to be integer!")

    if sides != "onesided":
        raise NotImplementedError
    if mode != "psd":
        raise NotImplementedError
    else:
        scale_by_freq = True
        # Enforced by https://github.com/matplotlib/matplotlib/blob/4bcc2caa42af7aba0c2d42c462fe92f0e314b7b3/lib/matplotlib/mlab.py#L284  # noqa: E501

    if mode is None:
        mode = "psd"
    if mode != "psd":
        raise NotImplementedError

    if NOVERLAP is None:
        NOVERLAP = 0
    if detrend_func is None:
        detrend_func = detrend_none
    if window is None:
        window = window_hanning

    if NOVERLAP >= NFFT:
        raise ValueError('noverlap must be less than NFFT')

    if pad_to is None:
        pad_to = NFFT
        # we aren't using in out CZT Transform...

    # For real x, ... (Line 286) ... (294:) if sides == "onesided"
    # numFreqs = (pad_to + 1)//2 if pad_to % 2 else pad_to/2+1  # not used in out sliding CZT
    scaling_factor = 2.0  # See docstring comments as to why not sqrt 2

    if not np.iterable(window):
        window = window(np.ones(NFFT, signal.dtype))
    if len(window) != NFFT:
        emsg = "The window length must match in the data's first dimension"
        raise ValueError(emsg)

    # Core logic (ours!)
    # 1. Get an ideally cached CZT transformer! Since this is a notebook, we're
    # going to recalculate the necessary transformer coeffecients each time
    # this function is called instead.
    transform = _cachedZoomFFT(n=NFFT, fn=fn, fs=fs)

    # 2. Now we performing sliding CZT!
    sliding_windows = np.lib.stride_tricks.sliding_window_view(
        signal,
        NFFT,  # this specifies the width of the window
        axis=0
    )[::NFFT-NOVERLAP].T
    sliding_windows = detrend(sliding_windows, detrend_func, axis=0)
    sliding_windows = sliding_windows * window.reshape((-1, 1))
    results = transform(sliding_windows, axis=0)

    if mode == "psd":
        results = np.conj(results) * results

        slc = slice(1, -1, None) if not NFFT % 2 else slice(1, None, None)
        results[slc] *= scaling_factor

        if scale_by_freq:
            results /= fs
            results /= (window**2).sum()

    # this is now spectrum as returned by mlab.specgram
    spec = results.real
    freq = np.linspace(*fn, NFFT)
    time = np.arange(NFFT/2, len(signal) - NFFT/2 + 1, NFFT-NOVERLAP)/fs
    halfbin_time = (time[1] - time[0]) / 2.0
    halfbin_freq = (freq[1] - freq[0]) / 2.0
    extent = (
        time[0]-halfbin_time, time[-1]+halfbin_time,
        freq[0]-halfbin_freq, freq[-1]+halfbin_freq,
    )

    return spec, freq, time, extent


def cz_spectrogram(
    signal: NDArray[np.number],
    fn: tuple[float, float],
    samp_rate: int,
    win_s: float,
    p_overlap: float,
    detrend_func=None,
    filter_window_func=None,
):
    """Convenience wrapper of cz_spectrogram_base for PowerSD.

    Input
    -----
    signal: Data to perform chirp Z spectrogram on.
    fn: Lower and upper end of Chirp Z transform.
    samp_rate: Sampling frequency of data.
    win_s: Length of window time slice in seconds. Number of points will be
        rounded off to nearest power of 2.
    p_overlap: 0<=p<=1. Percentage overlap between windows.
    detrend_func: Function applied to detrend data with.
    filter_window_func: Function, such as Hanning Windows, to improve SNR in
        spectrogram.

    Output
    ------
    spec: Spectrogram (PSD) passed straight into imshow, has units of V²/Hz.
    freq: Corresponding frequency axis of spectrogram.
    time: Corresponding time axis of spectrogram.
    imshow_kwargs: {aspect, extent, origin}
        These can be passed straight to matplotlib's imshow's kwargs.
    """
    nfft: int = _nearest_pow_2(win_s * samp_rate)
    noverlap: int = int(p_overlap * nfft)
    logger.debug("nfft = %s", nfft)
    logger.debug("noverlap = %s", noverlap)
    spec, freq, time, extent = cz_spectrogram_base(
        signal=signal,
        fn=fn,
        fs=samp_rate,
        NFFT=nfft,
        NOVERLAP=noverlap,
        detrend_func=detrend_func,
        window=filter_window_func,
    )
    kwargs: dict = {
        "aspect": "auto",
        "extent": extent,
        "origin": "lower",
    }

    return spec, freq, time, kwargs


def cz_spectrogram_amplitude(
    signal: NDArray[np.number],
    fn: tuple[float, float],
    samp_rate: int,
    win_s: float,
    p_overlap: float,
    detrend_func=None,
    filter_window_func=None,
):
    """Convenience wrapper of cz_spectrogram_base for AmplitudeSD.

    Input
    -----
    signal: Data to perform chirp Z spectrogram on.
    fn: Lower and upper end of Chirp Z transform.
    samp_rate: Sampling frequency of data.
    win_s: Length of window time slice in seconds. Number of points will be
        rounded off to nearest power of 2.
    p_overlap: 0<=p<=1. Percentage overlap between windows.
    detrend_func: Function applied to detrend data with.
    filter_window_func: Function, such as Hanning Windows, to improve SNR in
        spectrogram.

    Output
    ------
    spec: Spectrogram (ASD) passed straight into imshow, has units of V/√Hz.
    freq: Corresponding frequency axis of spectrogram.
    time: Corresponding time axis of spectrogram.
    imshow_kwargs: {aspect, extent, origin}
        These can be passed straight to matplotlib's imshow's kwargs.
    """
    nfft: int = _nearest_pow_2(win_s * samp_rate)
    noverlap: int = int(p_overlap * nfft)
    logger.debug("nfft = %s", nfft)
    logger.debug("noverlap = %s", noverlap)
    spec, freq, time, extent = cz_spectrogram_base(
        signal=signal,
        fn=fn,
        fs=samp_rate,
        NFFT=nfft,
        NOVERLAP=noverlap,
        detrend_func=detrend_func,
        window=filter_window_func,
    )
    spec = np.sqrt(spec)
    kwargs: dict = {
        "aspect": "auto",
        "extent": extent,
        "origin": "lower",
    }

    return spec, freq, time, kwargs
