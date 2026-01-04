"""
Kernel functions for nonparametric regression.

All kernels are normalized and support vectorized operations.
"""

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def gaussian_kernel(u: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Gaussian (normal) kernel.

    K(u) = (1/sqrt(2*pi)) * exp(-0.5 * u^2)

    Parameters
    ----------
    u : ndarray
        Scaled distances (x - x_i) / h

    Returns
    -------
    ndarray
        Kernel weights
    """
    return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)


def epanechnikov_kernel(u: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Epanechnikov kernel (optimal in MSE sense).

    K(u) = 0.75 * (1 - u^2) for |u| <= 1, else 0

    Parameters
    ----------
    u : ndarray
        Scaled distances (x - x_i) / h

    Returns
    -------
    ndarray
        Kernel weights
    """
    weights = 0.75 * (1 - u**2)
    weights = np.where(np.abs(u) <= 1, weights, 0.0)
    return weights


def uniform_kernel(u: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Uniform (rectangular) kernel.

    K(u) = 0.5 for |u| <= 1, else 0

    Parameters
    ----------
    u : ndarray
        Scaled distances (x - x_i) / h

    Returns
    -------
    ndarray
        Kernel weights
    """
    return np.where(np.abs(u) <= 1, 0.5, 0.0)


def tricube_kernel(u: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Tricube kernel (used in LOESS).

    K(u) = (70/81) * (1 - |u|^3)^3 for |u| <= 1, else 0

    Parameters
    ----------
    u : ndarray
        Scaled distances (x - x_i) / h

    Returns
    -------
    ndarray
        Kernel weights
    """
    abs_u = np.abs(u)
    weights = (70 / 81) * (1 - abs_u**3) ** 3
    weights = np.where(abs_u <= 1, weights, 0.0)
    return weights


def biweight_kernel(u: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Biweight (quartic) kernel.

    K(u) = (15/16) * (1 - u^2)^2 for |u| <= 1, else 0

    Parameters
    ----------
    u : ndarray
        Scaled distances (x - x_i) / h

    Returns
    -------
    ndarray
        Kernel weights
    """
    weights = (15 / 16) * (1 - u**2) ** 2
    weights = np.where(np.abs(u) <= 1, weights, 0.0)
    return weights


def triweight_kernel(u: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Triweight kernel.

    K(u) = (35/32) * (1 - u^2)^3 for |u| <= 1, else 0

    Parameters
    ----------
    u : ndarray
        Scaled distances (x - x_i) / h

    Returns
    -------
    ndarray
        Kernel weights
    """
    weights = (35 / 32) * (1 - u**2) ** 3
    weights = np.where(np.abs(u) <= 1, weights, 0.0)
    return weights


def cosine_kernel(u: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Cosine kernel.

    K(u) = (pi/4) * cos(pi * u / 2) for |u| <= 1, else 0

    Parameters
    ----------
    u : ndarray
        Scaled distances (x - x_i) / h

    Returns
    -------
    ndarray
        Kernel weights
    """
    weights = (np.pi / 4) * np.cos(np.pi * u / 2)
    weights = np.where(np.abs(u) <= 1, weights, 0.0)
    return weights


KERNEL_FUNCTIONS: dict[str, Callable[[NDArray[np.floating]], NDArray[np.floating]]] = {
    "gaussian": gaussian_kernel,
    "epanechnikov": epanechnikov_kernel,
    "uniform": uniform_kernel,
    "tricube": tricube_kernel,
    "biweight": biweight_kernel,
    "triweight": triweight_kernel,
    "cosine": cosine_kernel,
}


def get_kernel(
    kernel: str | Callable[[NDArray[np.floating]], NDArray[np.floating]],
) -> Callable[[NDArray[np.floating]], NDArray[np.floating]]:
    """
    Get kernel function by name or return callable directly.

    Parameters
    ----------
    kernel : str or callable
        Kernel name or custom kernel function

    Returns
    -------
    callable
        Kernel function
    """
    if callable(kernel):
        return kernel
    if kernel not in KERNEL_FUNCTIONS:
        valid = ", ".join(KERNEL_FUNCTIONS.keys())
        raise ValueError(f"Unknown kernel '{kernel}'. Valid options: {valid}")
    return KERNEL_FUNCTIONS[kernel]


def multivariate_kernel_weights(
    x: NDArray[np.floating],
    x_i: NDArray[np.floating],
    bandwidth: NDArray[np.floating] | float,
    kernel_func: Callable[[NDArray[np.floating]], NDArray[np.floating]],
) -> NDArray[np.floating]:
    """
    Compute multivariate kernel weights using product kernel.

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_features) or (n_features,)
        Evaluation points
    x_i : ndarray of shape (n_train, n_features)
        Training points
    bandwidth : float or ndarray of shape (n_features,)
        Bandwidth(s) for each dimension
    kernel_func : callable
        Univariate kernel function

    Returns
    -------
    ndarray of shape (n_samples, n_train) or (n_train,)
        Kernel weights for each training point
    """
    x = np.atleast_2d(x)
    x_i = np.atleast_2d(x_i)
    bandwidth = np.atleast_1d(bandwidth)

    if bandwidth.size == 1:
        bandwidth = np.full(x_i.shape[1], bandwidth[0])

    # Compute scaled distances for each dimension
    # x: (n_samples, n_features), x_i: (n_train, n_features)
    # Result: (n_samples, n_train, n_features)
    diff = x[:, np.newaxis, :] - x_i[np.newaxis, :, :]
    scaled = diff / bandwidth[np.newaxis, np.newaxis, :]

    # Apply kernel to each dimension and take product
    kernel_vals = kernel_func(scaled)  # (n_samples, n_train, n_features)
    weights = np.prod(kernel_vals, axis=2)  # (n_samples, n_train)

    # Normalize by bandwidth product
    weights = weights / np.prod(bandwidth)

    return weights
