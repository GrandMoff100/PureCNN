from typing import Callable, Generator

import numpy as np
from numpy.typing import ArrayLike


def convolution_targets(
    arr: ArrayLike,
    kernel: ArrayLike,
    spline: int = 1,
    pad: int | None = None,
) -> Generator[ArrayLike, None, None]:
    """Generate the vectorized target segments of a kernel."""
    width, height = arr.shape
    for i in range(0, kernel.shape[0]):
        for j in range(0, kernel.shape[1]):
            target = (
                kernel[i, j]
                * arr[
                    i : i + width - kernel.shape[0] : spline,
                    j : j + height - kernel.shape[1] : spline,
                ]
            )
            if pad is not None:
                target = np.pad(
                    target,
                    [[i, kernel.shape[0] - i], [j, kernel.shape[1] - j]],
                    constant_values=pad,
                )
            yield target


def convolution(
    arr: ArrayLike,
    kernel: ArrayLike,
    pad: int | None = None,
    spline: int = 1,
) -> ArrayLike:
    """Perform a convolution with a kernel."""
    return convolution_targets(arr, kernel, spline=spline, pad=pad) / kernel.size


def pool(
    arr: ArrayLike,
    size: tuple[int, int] = (2, 2),
    callback: Callable[[ArrayLike], int] = np.max,
) -> ArrayLike:
    """Implement all types of pooling (ie max-pooling, min-pooling, etc)"""
    return np.array(
        [
            callback(target)
            for target in convolution_targets(
                arr,
                np.ones(np.array(arr.shape) // np.array(size)),
                spline=2,
            )
        ]
    ).reshape(np.array(arr.shape) // np.array(size))


def relu(x: ArrayLike | int) -> ArrayLike | int:
    """Declare a ReLU activation function to use."""
    return np.maximum(x, 0)


def relu_prime(x: ArrayLike | int) -> ArrayLike | int:
    """Declare a ReLU activation function derivative to use."""
    return np.where(x > 0, 1, 0)
