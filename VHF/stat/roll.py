"""Online algorithms for handling stat"""
from functools import reduce
import numpy as np
from numpy import inexact, number
from numpy.typing import NDArray
from typing import Generic, Optional, TypeVar

M = TypeVar('M', bound=inexact, covariant=True)
N = TypeVar('N', bound=number, covariant=True)


class Welford(Generic[M, N]):
    _count: int
    _mean: Optional[NDArray[M] | M]
    _M2c: Optional[NDArray[M] | M]  # be careful of how arr.var() is defined

    # We keep the dimensionality for easier type handling, please refer to:
    # https://numpy.org/doc/stable/reference/typing.html#d-arrays
    # Oh well

    def __init__(
        self,
        initial_data: Optional[NDArray[N] | N] = None,
        axis: int = -1
    ) -> None:
        # self._axis = None  # pyright: ignore reportAttributeAccessIssue
        self._count = 0
        self._mean = None
        self._M2c = None
        if initial_data is not None:
            self._first_update(initial_data, axis)

    @property
    def mean(self) -> NDArray[M] | M:
        """Mean obtained thus far."""
        if self._count == 0 or self._mean is None:
            raise ValueError("Welford has not been populated with any data!")
        return self._mean

    @property
    def variance(self) -> NDArray[M] | M:
        """Variance obtained thus far."""
        if self._count == 0 or self._M2c is None:
            raise ValueError("Welford has not been populated with any data!")
        return self._M2c

    def _first_update(self, data: NDArray[N] | N, axis: int = -1) -> None:
        """Sets count, mean, M2C for the first time.

        Inputs
        -----
        data: T
            N-dimensional initial data, with Welford algorithm acting on axis.
        axis: int
            Axis being averaged/variance/etc out. Between 0 and N-1.
        """
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        ax = self._effective_axis(axis, data.shape)
        self._axis = ax
        self._count += data.shape[ax]
        self._mean = data.mean(axis=ax)
        self._M2c = data.var(axis=ax)

    def _effective_axis(
        self, axis: int, init_shape: Optional[tuple] = None
    ) -> int:
        """Get the axis by which mean/variance will be acting on."""
        if init_shape is not None:
            pass
        else:
            assert self._mean is not None
            init_shape: list[int] = list(self._mean.shape)
            init_shape.append(1)

        if axis >= 0:
            result = axis
        else:
            result = len(init_shape) + axis

        if result > len(init_shape):
            emsg = "Axis to perform mean on is larger than available data!"
            raise ValueError(emsg)
        elif result < 0:
            emsg = "Axis value obtained was too negative!"
            raise ValueError(emsg)
        return result

    def update(self, data: NDArray[N] | N, axis: int = -1) -> None:
        """Update Welford statistics with new data block.

        Inputs
        -----
        data: T
            data being consumed to update mean, variance etc
        axis: int
            The axis of data being averaged away.
        """
        if self._count == 0:
            # This is for the special case where data was not passed in the
            # initialisation stage.
            self._first_update(data, axis=axis)
            return

        # We now handle the regular update case.
        emsg = "Axis given was not consistent with prior given axis value!"
        assert self._axis == self._effective_axis(axis), ValueError(emsg)

        if self._mean is None or self._M2c is None:
            raise RuntimeError
        nA, meanA, m2ca = self._count, self._mean, self._M2c

        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        ax: int = self._axis
        nB: int = data.shape[ax]
        meanB: NDArray[M] | M = data.mean(axis=ax)
        m2cb: NDArray[M] | M = data.var(axis=ax)

        nAB = nB + nA
        delta: NDArray[M] | M = np.subtract(meanB, meanA)
        # if nA >> nB, use nAB \bar{x}_AB = nA*\bar{x}A + nB*\bar{x}B
        meanAB: NDArray[M] | M = np.add(meanA,  np.multiply(delta, nB / nAB))
        # m2cab: NDArray[M] | M = (m2ca * nA / nAB) \
        #     + m2cb / nAB + ((delta/nAB)**2. * nA * nB)
        m2cab: NDArray[M] | M = reduce(
            np.add,
            [
                np.multiply(m2ca, nA/nAB),
                np.multiply(m2cb, nB/nAB),
                np.multiply(np.power(np.divide(delta, nAB), 2.), nA*nB)
            ]
        )
        self._count, self._mean, self._M2c = nAB, meanAB, m2cab
