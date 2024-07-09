# Downsamplers equivalents used by Miguel
from logging import getLogger
import numpy as np
from numpy.typing import NDArray

logger = getLogger("downsampler")


def block_avg(my_arr: NDArray, N: int):
    """Returns a block average of 1D my_arr in blocks of N."""
    if N == 1:
        return my_arr
    return np.mean(my_arr.reshape(np.shape(my_arr)[0]//N, N), axis=1)


def block_avg_tail(my_arr: NDArray, N: int):
    """Block averages the main body, up till last block that might contain less
    than N items. Assumes 1D."""
    # Check for 1D array
    assert my_arr.ndim == 1

    if N == 1:
        return my_arr
    if np.size(my_arr) % N == 0:
        return block_avg(my_arr, N)
    else:
        logger.debug("Chosen indices does not give nice block!")
        result = np.zeros(np.size(my_arr)//N + 1)
        result[:-1] = block_avg(my_arr[:np.size(my_arr)//N * N], N)
        result[-1] = np.mean(my_arr[np.size(my_arr)//N * N:])
        return result
