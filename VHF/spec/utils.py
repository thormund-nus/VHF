"""Methods for handling spectrogram related objects."""

from matplotlib import colors
from matplotlib import pyplot as plt
from numbers import Number
import numpy as np
from numpy import number
from numpy.typing import NDArray
from typing import Callable, Optional, overload

__all__ = [
    "spectrogram_crop",
    "trunc_cmap",
]

type N = number | Number
type cond_type = Optional[Callable[[NDArray[Number]], NDArray[np.bool_]]]
type Extent = tuple[N, N, N, N]
type ExtentRet = tuple[float, float, float, float]


@overload
def spectrogram_crop(
    Sxx: NDArray, f: NDArray, t: NDArray, extent: Optional[Extent | ExtentRet],
    f_cond: cond_type, t_cond: cond_type
) -> tuple[NDArray, NDArray, NDArray, ExtentRet]:
    # Extent is generated so long as f, t, f_cond and t_cond are there
    ...


@overload
def spectrogram_crop(
    Sxx: Optional[NDArray], f: Optional[NDArray], t: Optional[NDArray],
    extent: Optional[Extent],
    f_cond: None, t_cond: None
) -> tuple[Optional[NDArray], Optional[NDArray],
           Optional[NDArray], Optional[ExtentRet]]:
    # if f_cond and t_cond are None, Sxx, f, t, extent are passed through.
    ...


def spectrogram_crop(Sxx, f, t, extent, f_cond, t_cond):
    """Function to crop Sxx to within f and t bounds (in natural units).

    Input
    -----
    Sxx: spectrogram, 2-dim NDArray
        Assumed to be of form Sxx[f, t], where f is frequency and t is time.
    f: Associated frequency axis to Sxx.
    t: Associated time axis to Sxx.
    extent: Associated imshow's extent to Sxx.
    f_cond: Manner by which to truncate Sxx in frequency by. Example:
        > lambda f: np.logical_and(10 < f, f < 20)
    t_cond: Manner by which to truncate Sxx in time by.
    """

    # Allow for accepting no filtering
    if f_cond is None and t_cond is None:
        return Sxx, f, t, extent
    if f_cond is None:
        f_cond = lambda x: np.full(np.shape(x), True)  # noqa: E731
    if t_cond is None:
        t_cond = lambda x: np.full(np.shape(x), True)  # noqa: E731
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
