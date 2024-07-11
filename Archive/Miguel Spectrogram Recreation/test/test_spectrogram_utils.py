import pytest
from logging import getLogger
import numpy as np
from pathlib import Path
import sys
module_path = str(Path(__file__).parents[1])
if module_path not in sys.path:
    sys.path.append(module_path)
from spectrogram_utils import spectrogram_crop

logger = getLogger("test_spectrogram_utils")


def get_data():
    # There is assumption in (matplotlib.)ax.imshow(..., origin="lower"), we
    # are plotting ...

    # freq
    #  M  M0 .........  MN
    # M-1              M(N-1)
    #  .                 .
    #  .      Sxx        .
    #  .                 .
    #  1  10 .........  1N
    #  0  00 .........  0N
    #     0   1 ......   N   time
    freq = np.linspace(0, 7, ny := 8, dtype=int)
    time = np.linspace(0, 5, nx := 6, dtype=int)
    grid = np.meshgrid(freq, time, indexing="ij")
    spec = np.zeros(shape=(ny, nx), dtype='U42')
    for i, g in enumerate(grid):
        grid[i] = g.astype("str", casting="unsafe", copy=False)
    spec = np.char.add(*grid)
    assert freq[1] == 1
    assert np.shape(spec) == (ny, nx)
    assert spec[1, 2] == "12"
    logger.info("get_data occurred without issue")
    return spec, freq, time


def test_crop():
    spec, freq, time = get_data()

    filter_freq = lambda f: np.logical_and(1 <= f, f <= 3)
    filter_time = lambda t: np.logical_and(3 <= t, t <= 4)
    spec, freq, time, extent = spectrogram_crop(
        spec, freq, time, None,
        filter_freq,
        filter_time
    )
    assert np.shape(spec) == (3, 2)
    assert spec[0, 0] == "13"
    assert spec[1, 0] == "23"
    assert spec[2, 0] == "33"
    assert spec[0, 1] == "14"
    assert spec[1, 1] == "24"
    assert spec[2, 1] == "34"
    assert np.shape(freq) == (3,)
    assert np.shape(time) == (2,)
    assert extent == (2.5, 4.5, 0.5, 3.5)
