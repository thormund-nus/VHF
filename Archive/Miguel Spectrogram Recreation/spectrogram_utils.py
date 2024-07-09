from matplotlib import colors
from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import Callable, Optional
cond_type = Optional[Callable[[NDArray[np.float64]], NDArray[np.bool_]]]


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
