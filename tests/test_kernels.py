"""Tests for kernel functions."""

import numpy as np
import pytest

from kernel_regression.kernels import (
    gaussian_kernel,
    epanechnikov_kernel,
    uniform_kernel,
    tricube_kernel,
    biweight_kernel,
    triweight_kernel,
    cosine_kernel,
    get_kernel,
    multivariate_kernel_weights,
    KERNEL_FUNCTIONS,
)


class TestKernelFunctions:
    """Tests for individual kernel functions."""

    def test_gaussian_kernel_at_zero(self):
        """Gaussian kernel peaks at u=0."""
        u = np.array([0.0])
        result = gaussian_kernel(u)
        expected = 1 / np.sqrt(2 * np.pi)
        np.testing.assert_almost_equal(result[0], expected)

    def test_gaussian_kernel_symmetry(self):
        """Gaussian kernel is symmetric."""
        u = np.array([-1.0, 1.0])
        result = gaussian_kernel(u)
        np.testing.assert_almost_equal(result[0], result[1])

    def test_gaussian_kernel_decay(self):
        """Gaussian kernel decays away from zero."""
        u = np.array([0.0, 1.0, 2.0, 3.0])
        result = gaussian_kernel(u)
        assert all(result[i] > result[i + 1] for i in range(len(result) - 1))

    def test_epanechnikov_kernel_compact_support(self):
        """Epanechnikov kernel is zero outside [-1, 1]."""
        u = np.array([-1.5, 1.5, 2.0, -2.0])
        result = epanechnikov_kernel(u)
        np.testing.assert_array_equal(result, 0.0)

    def test_epanechnikov_kernel_at_zero(self):
        """Epanechnikov kernel value at zero."""
        u = np.array([0.0])
        result = epanechnikov_kernel(u)
        np.testing.assert_almost_equal(result[0], 0.75)

    def test_uniform_kernel_constant_in_support(self):
        """Uniform kernel is constant within support."""
        u = np.array([-0.5, 0.0, 0.5, 0.9])
        result = uniform_kernel(u)
        np.testing.assert_array_equal(result, 0.5)

    def test_uniform_kernel_compact_support(self):
        """Uniform kernel is zero outside [-1, 1]."""
        u = np.array([-1.01, 1.01])
        result = uniform_kernel(u)
        np.testing.assert_array_equal(result, 0.0)

    def test_tricube_kernel_compact_support(self):
        """Tricube kernel is zero outside [-1, 1]."""
        u = np.array([-2.0, 2.0])
        result = tricube_kernel(u)
        np.testing.assert_array_equal(result, 0.0)

    def test_tricube_kernel_at_zero(self):
        """Tricube kernel value at zero."""
        u = np.array([0.0])
        result = tricube_kernel(u)
        np.testing.assert_almost_equal(result[0], 70 / 81)

    def test_biweight_kernel_compact_support(self):
        """Biweight kernel is zero outside [-1, 1]."""
        u = np.array([-1.1, 1.1])
        result = biweight_kernel(u)
        np.testing.assert_array_equal(result, 0.0)

    def test_triweight_kernel_compact_support(self):
        """Triweight kernel is zero outside [-1, 1]."""
        u = np.array([-1.5, 1.5])
        result = triweight_kernel(u)
        np.testing.assert_array_equal(result, 0.0)

    def test_cosine_kernel_compact_support(self):
        """Cosine kernel is zero outside [-1, 1]."""
        u = np.array([-1.5, 1.5])
        result = cosine_kernel(u)
        np.testing.assert_array_equal(result, 0.0)

    @pytest.mark.parametrize("kernel_name", list(KERNEL_FUNCTIONS.keys()))
    def test_all_kernels_nonnegative(self, kernel_name):
        """All kernels should return non-negative values."""
        kernel_func = KERNEL_FUNCTIONS[kernel_name]
        u = np.linspace(-3, 3, 100)
        result = kernel_func(u)
        assert np.all(result >= 0)

    @pytest.mark.parametrize("kernel_name", list(KERNEL_FUNCTIONS.keys()))
    def test_all_kernels_symmetric(self, kernel_name):
        """All kernels should be symmetric."""
        kernel_func = KERNEL_FUNCTIONS[kernel_name]
        u = np.linspace(0, 3, 50)
        result_pos = kernel_func(u)
        result_neg = kernel_func(-u)
        np.testing.assert_array_almost_equal(result_pos, result_neg)


class TestGetKernel:
    """Tests for get_kernel function."""

    def test_get_kernel_by_name(self):
        """Can retrieve kernel by name."""
        for name in KERNEL_FUNCTIONS:
            kernel = get_kernel(name)
            assert callable(kernel)

    def test_get_kernel_with_callable(self):
        """Returns callable directly."""
        custom_kernel = lambda u: np.exp(-u**2)
        result = get_kernel(custom_kernel)
        assert result is custom_kernel

    def test_get_kernel_invalid_name(self):
        """Raises error for unknown kernel name."""
        with pytest.raises(ValueError, match="Unknown kernel"):
            get_kernel("invalid_kernel")


class TestMultivariateKernelWeights:
    """Tests for multivariate kernel weight computation."""

    def test_1d_weights_shape(self):
        """Weights have correct shape for 1D data."""
        x = np.array([[1.0]])  # 1 prediction point
        x_i = np.array([[0.0], [1.0], [2.0]])  # 3 training points
        bandwidth = np.array([1.0])

        weights = multivariate_kernel_weights(x, x_i, bandwidth, gaussian_kernel)

        assert weights.shape == (1, 3)

    def test_2d_weights_shape(self):
        """Weights have correct shape for 2D data."""
        x = np.array([[1.0, 1.0], [2.0, 2.0]])  # 2 prediction points
        x_i = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])  # 3 training points
        bandwidth = np.array([1.0, 1.0])

        weights = multivariate_kernel_weights(x, x_i, bandwidth, gaussian_kernel)

        assert weights.shape == (2, 3)

    def test_weights_nonnegative(self):
        """All weights are non-negative."""
        x = np.random.randn(5, 2)
        x_i = np.random.randn(10, 2)
        bandwidth = np.array([1.0, 1.0])

        weights = multivariate_kernel_weights(x, x_i, bandwidth, gaussian_kernel)

        assert np.all(weights >= 0)

    def test_closer_points_higher_weight(self):
        """Points closer to prediction point have higher weight."""
        x = np.array([[0.0]])
        x_i = np.array([[0.0], [0.5], [1.0], [2.0]])
        bandwidth = np.array([1.0])

        weights = multivariate_kernel_weights(x, x_i, bandwidth, gaussian_kernel)

        # Weights should decrease as distance increases
        weights_flat = weights.flatten()
        for i in range(len(weights_flat) - 1):
            assert weights_flat[i] >= weights_flat[i + 1]

    def test_scalar_bandwidth(self):
        """Scalar bandwidth is broadcast to all dimensions."""
        x = np.array([[1.0, 1.0]])
        x_i = np.array([[0.0, 0.0], [1.0, 1.0]])
        bandwidth = 0.5  # scalar

        weights = multivariate_kernel_weights(x, x_i, bandwidth, gaussian_kernel)

        assert weights.shape == (1, 2)
