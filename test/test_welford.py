import numpy as np
from numpy import testing
from pathlib import Path
import pytest
import sys
module_path = str(Path(__file__).parents[1])
if module_path not in sys.path:
    sys.path.append(module_path)
from VHF.stat.roll import Welford


def test_welford_effective_axis():
    w = Welford()
    assert w._effective_axis(-1, (5, 7, 8)) == 2
    assert w._effective_axis(-2, (5, 7, 8)) == 1
    assert w._effective_axis(-3, (5, 7, 8)) == 0
    assert w._effective_axis(0, (5, 7, 8)) == 0
    assert w._effective_axis(1, (5, 7, 8)) == 1
    assert w._effective_axis(-1, (5, 7, 8, 3)) == 3
    with pytest.raises(Exception) as e:
        _ = w._effective_axis(-5, (1, 2))


def test_welford_1Da():
    s = 400
    data = np.random.rand(s) * 1e2
    w = Welford(initial_data=data[:s//2])
    w.update(data[s//2:])
    testing.assert_almost_equal(w.mean, data.mean())
    testing.assert_almost_equal(w.variance, data.var())


def test_welford_1Db():
    s = 400
    data = np.random.rand(s) * 1e2
    w = Welford(initial_data=data[:s//2])
    w.update(data[s//2:2*s//3])
    w.update(data[2*s//3:])
    testing.assert_almost_equal(w.mean, data.mean())
    testing.assert_almost_equal(w.variance, data.var())


def test_welford_2D0a():
    s = 450
    t = 130
    ax = 0
    data = np.random.rand(s, t) * 1e2
    w = Welford(initial_data=data[:s//2, :], axis=ax)
    w.update(data[s//2:, :], axis=ax)
    testing.assert_almost_equal(w.mean, data.mean(axis=ax))
    testing.assert_almost_equal(w.variance, data.var(axis=ax))


def test_welford_2D0b():
    s = 450
    t = 1
    ax = 0
    data = np.random.rand(s, t) * 1e2
    w = Welford(initial_data=data[:s//2, :], axis=ax)
    w.update(data[s//2:, :], axis=ax)
    testing.assert_almost_equal(w.mean, data.mean(axis=ax))
    testing.assert_almost_equal(w.variance, data.var(axis=ax))


def test_welford_2D0c():
    s = 1
    t = 100
    ax = 0
    data = np.random.rand(s, t) * 1e2
    w = Welford(initial_data=data[:, :], axis=ax)
    testing.assert_almost_equal(w.mean, data.mean(axis=ax))
    testing.assert_almost_equal(w.variance, data.var(axis=ax))


def test_welford_2D0d():
    s = 50000
    t = 100
    ax = 0
    data = np.random.rand(s, t) * 1e2
    w = Welford()
    blk_size = 1000
    for i in range(s//blk_size):
        w.update(data[i*blk_size:(i+1)*blk_size, :], axis=ax)
    testing.assert_almost_equal(w.mean, data.mean(axis=ax))
    testing.assert_almost_equal(w.variance, data.var(axis=ax))


def test_welford_2D1a():
    s = 450
    t = 130
    ax = 1
    data = np.random.rand(s, t) * 1e2
    w = Welford(initial_data=data[:, :t//2], axis=ax)
    w.update(data[:, t//2:], axis=ax)
    testing.assert_almost_equal(w.mean, data.mean(axis=ax))
    testing.assert_almost_equal(w.variance, data.var(axis=ax))


def test_welford_2Dm1a():
    s = 450
    t = 130
    ax = -1
    data = np.random.rand(s, t) * 1e2
    w = Welford(initial_data=data[:, :t//2], axis=ax)
    w.update(data[:, t//2:], axis=ax)
    testing.assert_almost_equal(w.mean, data.mean(axis=ax))
    testing.assert_almost_equal(w.variance, data.var(axis=ax))


def test_welford_2DFAIL():
    s = 450
    t = 130
    data = np.random.rand(s, t) * 1e2
    w = Welford(initial_data=data[:s//2, :], axis=0)
    with pytest.raises(Exception) as e:
        # This should fail as it now defaults to axis=-1
        w.update(data[s//2:, :])


def test_welford_3D0a():
    s = 50000
    t = 10
    u = 5
    ax = 0
    data = np.random.rand(s, t, u) * 1e2
    w = Welford()
    blk_size = 1000
    for i in range(s//blk_size):
        w.update(data[i*blk_size:(i+1)*blk_size, :, :], axis=ax)
    testing.assert_almost_equal(w.mean, data.mean(axis=ax))
    testing.assert_almost_equal(w.variance, data.var(axis=ax))


def test_welford_3D1a():
    s = 10
    t = 50000
    u = 5
    ax = 1
    data = np.random.rand(s, t, u) * 1e2
    w = Welford()
    blk_size = 1000
    for i in range(t//blk_size):
        w.update(data[:, i*blk_size:(i+1)*blk_size, :], axis=ax)
    testing.assert_almost_equal(w.mean, data.mean(axis=ax))
    testing.assert_almost_equal(w.variance, data.var(axis=ax))


def test_welford_3D2a():
    s = 10
    t = 5
    u = 50000
    ax = 2
    data = np.random.rand(s, t, u) * 1e2
    w = Welford()
    blk_size = 1000
    for i in range(u//blk_size):
        w.update(data[:, :, i*blk_size:(i+1)*blk_size], axis=ax)
    testing.assert_almost_equal(w.mean, data.mean(axis=ax))
    testing.assert_almost_equal(w.variance, data.var(axis=ax))


def test_welford_3Dm2a():
    s = 10
    t = 50000
    u = 5
    ax = -2
    data = np.random.rand(s, t, u) * 1e2
    w = Welford()
    blk_size = 1000
    for i in range(t//blk_size):
        w.update(data[:, i*blk_size:(i+1)*blk_size, :], axis=ax)
    testing.assert_almost_equal(w.mean, data.mean(axis=ax))
    testing.assert_almost_equal(w.variance, data.var(axis=ax))


def test_welford_3Dm1a():
    s = 10
    t = 5
    u = 50000
    ax = -1
    data = np.random.rand(s, t, u) * 1e2
    w = Welford()
    blk_size = 1000
    for i in range(u//blk_size):
        w.update(data[:, :, i*blk_size:(i+1)*blk_size], axis=ax)
    testing.assert_almost_equal(w.mean, data.mean(axis=ax))
    testing.assert_almost_equal(w.variance, data.var(axis=ax))


def test_welford_3DFAIL():
    s = 10
    t = 5
    u = 50000
    ax = -1
    data = np.random.rand(s, t, u) * 1e2
    w = Welford()
    blk_size = 1000
    with pytest.raises(Exception) as e:
        for i in range(u//blk_size):
            w.update(data[:, i*blk_size:(i+1)*blk_size, :], axis=ax)
