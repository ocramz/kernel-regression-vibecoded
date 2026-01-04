"""Tests for kernel regression estimators."""

import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from kernel_regression.estimators import (
    NadarayaWatson,
    LocalPolynomialRegression,
)


class TestNadarayaWatson:
    """Tests for Nadaraya-Watson estimator."""

    def test_fit_returns_self(self, simple_1d_data):
        """Fit returns self for method chaining."""
        X, y = simple_1d_data
        model = NadarayaWatson(bandwidth=0.5)
        result = model.fit(X, y)
        assert result is model

    def test_fit_stores_data(self, simple_1d_data):
        """Fit stores training data."""
        X, y = simple_1d_data
        model = NadarayaWatson(bandwidth=0.5).fit(X, y)
        np.testing.assert_array_equal(model.X_, X)
        np.testing.assert_array_equal(model.y_, y)

    def test_predict_shape(self, simple_1d_data):
        """Predict returns correct shape."""
        X, y = simple_1d_data
        model = NadarayaWatson(bandwidth=0.5).fit(X, y)
        y_pred = model.predict(X[:10])
        assert y_pred.shape == (10,)

    def test_predict_interpolates(self, simple_1d_data):
        """Predictions at training points are close to targets."""
        X, y = simple_1d_data
        model = NadarayaWatson(bandwidth=0.3).fit(X, y)
        y_pred = model.predict(X)
        # Should be reasonably close (not exact due to smoothing)
        correlation = np.corrcoef(y, y_pred)[0, 1]
        assert correlation > 0.8

    def test_bandwidth_silverman(self, simple_1d_data):
        """Silverman bandwidth selection works."""
        X, y = simple_1d_data
        model = NadarayaWatson(bandwidth="silverman").fit(X, y)
        assert hasattr(model, "bandwidth_")
        assert model.bandwidth_[0] > 0

    def test_bandwidth_cv(self, simple_1d_data):
        """Cross-validated bandwidth selection works."""
        X, y = simple_1d_data
        model = NadarayaWatson(bandwidth="cv", cv=5).fit(X, y)
        assert hasattr(model, "bandwidth_")
        assert model.bandwidth_[0] > 0

    def test_different_kernels(self, simple_1d_data):
        """Different kernel functions work."""
        X, y = simple_1d_data
        kernels = ["gaussian", "epanechnikov", "uniform", "tricube"]
        for kernel in kernels:
            model = NadarayaWatson(kernel=kernel, bandwidth=0.5).fit(X, y)
            y_pred = model.predict(X[:5])
            assert y_pred.shape == (5,)
            assert np.all(np.isfinite(y_pred))

    def test_custom_kernel(self, simple_1d_data):
        """Custom kernel function works."""
        X, y = simple_1d_data
        custom_kernel = lambda u: np.exp(-np.abs(u))  # Laplacian kernel
        model = NadarayaWatson(kernel=custom_kernel, bandwidth=0.5).fit(X, y)
        y_pred = model.predict(X[:5])
        assert y_pred.shape == (5,)

    def test_multivariate_data(self, simple_2d_data):
        """Works with multivariate data."""
        X, y = simple_2d_data
        model = NadarayaWatson(bandwidth=0.5).fit(X, y)
        y_pred = model.predict(X)
        assert y_pred.shape == (len(y),)
        correlation = np.corrcoef(y, y_pred)[0, 1]
        assert correlation > 0.7

    def test_per_dimension_bandwidth(self, simple_2d_data):
        """Per-dimension bandwidth works."""
        X, y = simple_2d_data
        model = NadarayaWatson(bandwidth=[0.5, 1.0]).fit(X, y)
        np.testing.assert_array_equal(model.bandwidth_, [0.5, 1.0])

    def test_get_weights(self, simple_1d_data):
        """Get weights returns normalized weights."""
        X, y = simple_1d_data
        model = NadarayaWatson(bandwidth=0.5).fit(X, y)
        weights = model.get_weights(X[:5])
        assert weights.shape == (5, len(X))
        # Each row should sum to ~1 (normalized)
        row_sums = np.sum(weights, axis=1)
        np.testing.assert_array_almost_equal(row_sums, 1.0)

    def test_unfitted_error(self, simple_1d_data):
        """Raises error when predicting on unfitted model."""
        X, y = simple_1d_data
        model = NadarayaWatson(bandwidth=0.5)
        with pytest.raises(Exception):  # NotFittedError
            model.predict(X)

    def test_wrong_n_features(self, simple_1d_data):
        """Raises error for wrong number of features."""
        X, y = simple_1d_data
        model = NadarayaWatson(bandwidth=0.5).fit(X, y)
        X_wrong = np.random.randn(10, 2)  # 2 features instead of 1
        with pytest.raises(ValueError):
            model.predict(X_wrong)


class TestLocalPolynomialRegression:
    """Tests for Local Polynomial Regression estimator."""

    def test_fit_returns_self(self, simple_1d_data):
        """Fit returns self for method chaining."""
        X, y = simple_1d_data
        model = LocalPolynomialRegression(order=1, bandwidth=0.5)
        result = model.fit(X, y)
        assert result is model

    def test_order_0_like_nw(self, simple_1d_data):
        """Order 0 is equivalent to Nadaraya-Watson."""
        X, y = simple_1d_data
        nw = NadarayaWatson(bandwidth=0.5).fit(X, y)
        lp = LocalPolynomialRegression(order=0, bandwidth=0.5).fit(X, y)

        y_nw = nw.predict(X[:10])
        y_lp = lp.predict(X[:10])

        np.testing.assert_array_almost_equal(y_nw, y_lp, decimal=5)

    def test_order_1_local_linear(self, simple_1d_data):
        """Order 1 (local linear) works."""
        X, y = simple_1d_data
        model = LocalPolynomialRegression(order=1, bandwidth=0.5).fit(X, y)
        y_pred = model.predict(X)
        assert y_pred.shape == (len(y),)
        correlation = np.corrcoef(y, y_pred)[0, 1]
        assert correlation > 0.8

    def test_order_2_local_quadratic(self, simple_2d_data):
        """Order 2 (local quadratic) works."""
        X, y = simple_2d_data
        model = LocalPolynomialRegression(order=2, bandwidth=0.5).fit(X, y)
        y_pred = model.predict(X)
        assert y_pred.shape == (len(y),)

    def test_cv_order_selection(self, simple_1d_data):
        """Cross-validated order selection works."""
        X, y = simple_1d_data
        model = LocalPolynomialRegression(
            order="cv",
            max_order=2,
            bandwidth=0.5,
        ).fit(X, y)
        assert hasattr(model, "order_")
        assert model.order_ in [0, 1, 2]

    def test_cv_bandwidth_selection(self, simple_1d_data):
        """Cross-validated bandwidth selection works."""
        X, y = simple_1d_data
        model = LocalPolynomialRegression(
            order=1,
            bandwidth="cv",
            cv=5,
        ).fit(X, y)
        assert hasattr(model, "bandwidth_")
        assert model.bandwidth_[0] > 0

    def test_predict_with_derivatives(self, simple_1d_data):
        """Predict with derivatives returns gradients."""
        X, y = simple_1d_data
        model = LocalPolynomialRegression(order=1, bandwidth=0.5).fit(X, y)
        y_pred, gradients = model.predict_with_derivatives(X[:10])

        assert y_pred.shape == (10,)
        assert gradients is not None
        assert gradients.shape == (10, 1)

    def test_predict_with_derivatives_order_0(self, simple_1d_data):
        """Predict with derivatives returns None for order 0."""
        X, y = simple_1d_data
        model = LocalPolynomialRegression(order=0, bandwidth=0.5).fit(X, y)
        y_pred, gradients = model.predict_with_derivatives(X[:10])

        assert y_pred.shape == (10,)
        assert gradients is None

    def test_multivariate_gradients(self, simple_2d_data):
        """Gradients have correct shape for multivariate data."""
        X, y = simple_2d_data
        model = LocalPolynomialRegression(order=1, bandwidth=0.5).fit(X, y)
        y_pred, gradients = model.predict_with_derivatives(X[:10])

        assert gradients.shape == (10, 2)

    def test_regularization(self, simple_1d_data):
        """Regularization parameter is used."""
        X, y = simple_1d_data
        model1 = LocalPolynomialRegression(
            order=2,
            bandwidth=0.5,
            regularization=1e-10,
        ).fit(X, y)
        model2 = LocalPolynomialRegression(
            order=2,
            bandwidth=0.5,
            regularization=1.0,
        ).fit(X, y)

        y1 = model1.predict(X[:10])
        y2 = model2.predict(X[:10])

        # Different regularization should give different results
        assert not np.allclose(y1, y2)

    def test_different_kernels(self, simple_1d_data):
        """Different kernels work with local polynomial."""
        X, y = simple_1d_data
        for kernel in ["gaussian", "epanechnikov", "tricube"]:
            model = LocalPolynomialRegression(
                order=1,
                kernel=kernel,
                bandwidth=0.5,
            ).fit(X, y)
            y_pred = model.predict(X[:5])
            assert np.all(np.isfinite(y_pred))


class TestSklearnCompatibility:
    """Tests for sklearn compatibility."""

    def test_nw_clone(self, simple_1d_data):
        """NadarayaWatson can be cloned."""
        from sklearn.base import clone
        X, y = simple_1d_data
        model = NadarayaWatson(bandwidth=0.5)
        cloned = clone(model)
        assert cloned.bandwidth == model.bandwidth

    def test_lp_clone(self, simple_1d_data):
        """LocalPolynomialRegression can be cloned."""
        from sklearn.base import clone
        model = LocalPolynomialRegression(order=1, bandwidth=0.5)
        cloned = clone(model)
        assert cloned.order == model.order

    def test_nw_score(self, simple_1d_data):
        """NadarayaWatson has score method (R^2)."""
        X, y = simple_1d_data
        model = NadarayaWatson(bandwidth=0.5).fit(X, y)
        score = model.score(X, y)
        assert 0 <= score <= 1

    def test_lp_score(self, simple_1d_data):
        """LocalPolynomialRegression has score method."""
        X, y = simple_1d_data
        model = LocalPolynomialRegression(order=1, bandwidth=0.5).fit(X, y)
        score = model.score(X, y)
        assert 0 <= score <= 1

    def test_nw_get_params(self):
        """NadarayaWatson implements get_params."""
        model = NadarayaWatson(kernel="epanechnikov", bandwidth=0.3)
        params = model.get_params()
        assert params["kernel"] == "epanechnikov"
        assert params["bandwidth"] == 0.3

    def test_nw_set_params(self):
        """NadarayaWatson implements set_params."""
        model = NadarayaWatson()
        model.set_params(kernel="tricube", bandwidth=0.7)
        assert model.kernel == "tricube"
        assert model.bandwidth == 0.7

    def test_pipeline_integration(self, simple_1d_data):
        """Can be used in sklearn Pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X, y = simple_1d_data
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", NadarayaWatson(bandwidth=0.5)),
        ])
        pipe.fit(X, y)
        y_pred = pipe.predict(X[:5])
        assert y_pred.shape == (5,)

    def test_cross_val_score(self, simple_1d_data):
        """Can be used with cross_val_score."""
        from sklearn.model_selection import cross_val_score

        X, y = simple_1d_data
        model = NadarayaWatson(bandwidth=0.5)
        scores = cross_val_score(model, X, y, cv=3)
        assert len(scores) == 3
        assert all(np.isfinite(scores))
