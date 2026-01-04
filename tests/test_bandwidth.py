"""Tests for bandwidth selection methods."""

import numpy as np
import pytest

from kernel_regression.bandwidth import (
    silverman_bandwidth,
    scott_bandwidth,
    RuleOfThumbBandwidth,
    CrossValidatedBandwidth,
)


class TestSilvermanBandwidth:
    """Tests for Silverman's rule of thumb."""

    def test_returns_correct_shape(self):
        """Returns bandwidth for each feature."""
        X = np.random.randn(100, 3)
        h = silverman_bandwidth(X)
        assert h.shape == (3,)

    def test_positive_bandwidths(self):
        """All bandwidths are positive."""
        X = np.random.randn(100, 2)
        h = silverman_bandwidth(X)
        assert np.all(h > 0)

    def test_larger_variance_larger_bandwidth(self):
        """Features with larger variance get larger bandwidth."""
        X = np.column_stack([
            np.random.randn(100) * 1,  # low variance
            np.random.randn(100) * 10,  # high variance
        ])
        h = silverman_bandwidth(X)
        assert h[1] > h[0]

    def test_factor_scaling(self):
        """Factor scales bandwidth linearly."""
        X = np.random.randn(100, 2)
        h1 = silverman_bandwidth(X, factor=1.0)
        h2 = silverman_bandwidth(X, factor=2.0)
        np.testing.assert_array_almost_equal(h2, 2 * h1)

    def test_sample_size_effect(self):
        """Larger sample size leads to smaller bandwidth."""
        np.random.seed(42)
        X_small = np.random.randn(50, 1)
        X_large = np.random.randn(500, 1)

        h_small = silverman_bandwidth(X_small)
        h_large = silverman_bandwidth(X_large)

        # With same variance, larger n should give smaller h
        # Need to account for variance differences in random samples
        # Use proportional comparison
        assert h_large[0] < h_small[0]

    def test_handles_constant_feature(self):
        """Handles features with zero variance."""
        X = np.column_stack([
            np.random.randn(100),
            np.ones(100),  # constant
        ])
        h = silverman_bandwidth(X)
        assert np.all(h > 0)
        assert np.isfinite(h[1])


class TestScottBandwidth:
    """Tests for Scott's rule."""

    def test_returns_correct_shape(self):
        """Returns bandwidth for each feature."""
        X = np.random.randn(100, 4)
        h = scott_bandwidth(X)
        assert h.shape == (4,)

    def test_positive_bandwidths(self):
        """All bandwidths are positive."""
        X = np.random.randn(100, 2)
        h = scott_bandwidth(X)
        assert np.all(h > 0)


class TestRuleOfThumbBandwidth:
    """Tests for RuleOfThumbBandwidth class."""

    def test_silverman_method(self):
        """Silverman method works."""
        selector = RuleOfThumbBandwidth(method="silverman")
        X = np.random.randn(100, 2)
        h = selector(X)
        assert h.shape == (2,)
        assert np.all(h > 0)

    def test_scott_method(self):
        """Scott method works."""
        selector = RuleOfThumbBandwidth(method="scott")
        X = np.random.randn(100, 2)
        h = selector(X)
        assert h.shape == (2,)
        assert np.all(h > 0)

    def test_invalid_method(self):
        """Invalid method raises error."""
        with pytest.raises(ValueError):
            RuleOfThumbBandwidth(method="invalid")

    def test_factor_applied(self):
        """Factor is applied to bandwidth."""
        X = np.random.randn(100, 2)
        selector1 = RuleOfThumbBandwidth(method="silverman", factor=1.0)
        selector2 = RuleOfThumbBandwidth(method="silverman", factor=0.5)
        h1 = selector1(X)
        h2 = selector2(X)
        np.testing.assert_array_almost_equal(h2, 0.5 * h1)


class TestCrossValidatedBandwidth:
    """Tests for cross-validated bandwidth selection."""

    def test_returns_correct_shape(self, simple_1d_data):
        """Returns bandwidth for each feature."""
        X, y = simple_1d_data
        selector = CrossValidatedBandwidth(n_bandwidths=5)
        h = selector(X, y)
        assert h.shape == (1,)

    def test_positive_bandwidth(self, simple_1d_data):
        """Selected bandwidth is positive."""
        X, y = simple_1d_data
        selector = CrossValidatedBandwidth(n_bandwidths=5)
        h = selector(X, y)
        assert h[0] > 0

    def test_cv_results_stored(self, simple_1d_data):
        """CV results are stored after fitting."""
        X, y = simple_1d_data
        selector = CrossValidatedBandwidth(n_bandwidths=5)
        selector(X, y)
        assert selector.cv_results_ is not None
        assert "best_bandwidth" in selector.cv_results_

    def test_kfold_cv(self, simple_1d_data):
        """K-fold CV works."""
        X, y = simple_1d_data
        selector = CrossValidatedBandwidth(cv=5, n_bandwidths=5)
        h = selector(X, y)
        assert h.shape == (1,)
        assert h[0] > 0

    def test_optimization_method(self, simple_1d_data):
        """Optimization-based selection works."""
        X, y = simple_1d_data
        selector = CrossValidatedBandwidth(use_grid=False)
        h = selector(X, y)
        assert h.shape == (1,)
        assert h[0] > 0

    def test_custom_bandwidth_range(self, simple_1d_data):
        """Custom bandwidth range is respected."""
        X, y = simple_1d_data
        selector = CrossValidatedBandwidth(
            bandwidth_range=(0.1, 0.5),
            n_bandwidths=5,
        )
        h = selector(X, y)
        assert 0.1 <= h[0] <= 0.5

    def test_multivariate_data(self, simple_2d_data):
        """Works with multivariate data."""
        X, y = simple_2d_data
        selector = CrossValidatedBandwidth(n_bandwidths=5)
        h = selector(X, y)
        assert h.shape == (2,)
        assert np.all(h > 0)

    def test_polynomial_order(self, simple_1d_data):
        """Can select bandwidth for local polynomial."""
        X, y = simple_1d_data
        selector = CrossValidatedBandwidth(
            polynomial_order=1,
            n_bandwidths=5,
        )
        h = selector(X, y)
        assert h.shape == (1,)
        assert h[0] > 0
