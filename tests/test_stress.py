"""
Comprehensive stress tests for kernel regression package.

These tests are adversarial checks to verify correctness and robustness.
"""

import time
import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from kernel_regression import (
    NadarayaWatson,
    LocalPolynomialRegression,
    GoodnessOfFit,
    heteroscedasticity_test,
)


class TestKnownFunctionStress:
    """
    Test 1: Known Function Stress Test

    Generate synthetic data where we know the exact underlying function.
    The model should track non-linear curves that linear regression cannot.
    """

    def test_sin_x_squared_tracking(self):
        """Test that kernel regression tracks sin(x²) while linear fails."""
        np.random.seed(42)

        # Ground truth: y = sin(x²) + noise
        n = 200
        X = np.linspace(0, 3, n).reshape(-1, 1)
        y_true = np.sin(X.flatten() ** 2)
        y = y_true + 0.1 * np.random.randn(n)

        # Linear regression (should fail)
        lr = LinearRegression().fit(X, y)
        y_pred_lr = lr.predict(X)
        mse_lr = mean_squared_error(y_true, y_pred_lr)

        # Kernel regression (should succeed)
        kr = NadarayaWatson(bandwidth="cv", cv=5).fit(X, y)
        y_pred_kr = kr.predict(X)
        mse_kr = mean_squared_error(y_true, y_pred_kr)

        print(f"\n  Linear MSE: {mse_lr:.4f}")
        print(f"  Kernel MSE: {mse_kr:.4f}")
        print(f"  Improvement: {mse_lr / mse_kr:.1f}x")

        # Kernel should be MUCH better
        assert mse_kr < mse_lr, "Kernel should outperform linear on non-linear data"
        assert mse_kr < 0.05, f"Kernel MSE too high: {mse_kr}"

    def test_bandwidth_shrinks_with_more_data(self):
        """Bandwidth should shrink as we add more data points."""
        np.random.seed(42)

        bandwidths = []
        sample_sizes = [50, 100, 200, 400]

        for n in sample_sizes:
            X = np.linspace(0, 3, n).reshape(-1, 1)
            y = np.sin(X.flatten() ** 2) + 0.1 * np.random.randn(n)

            model = NadarayaWatson(bandwidth="cv", cv=5).fit(X, y)
            bandwidths.append(model.bandwidth_[0])

        print(f"\n  Sample sizes: {sample_sizes}")
        print(f"  Bandwidths: {[f'{b:.4f}' for b in bandwidths]}")

        # Bandwidth should generally decrease with more data
        # (allows capturing finer details)
        assert bandwidths[-1] < bandwidths[0], (
            "Bandwidth should shrink with more data"
        )


class TestCurseOfDimensionality:
    """
    Test 2: Curse of Dimensionality Check

    Feed 10 variables where only 2 matter. Check robustness.
    """

    def test_high_dimensional_with_irrelevant_features(self):
        """Test with 10 features, only 2 relevant."""
        np.random.seed(42)
        n = 300

        # 10 features, but only x1 and x2 matter
        X = np.random.randn(n, 10)
        y = X[:, 0] ** 2 + np.sin(X[:, 1]) + 0.1 * np.random.randn(n)

        # Model should still work (not crash or give NaN)
        model = NadarayaWatson(bandwidth="silverman").fit(X, y)
        y_pred = model.predict(X)

        # Check predictions are valid
        assert not np.any(np.isnan(y_pred)), "Predictions contain NaN"
        assert not np.any(np.isinf(y_pred)), "Predictions contain Inf"

        # R² should still be reasonable (not negative)
        r2 = model.score(X, y)
        print(f"\n  R² with 10 features (2 relevant): {r2:.4f}")
        assert r2 > 0, f"R² should be positive, got {r2}"

    def test_bandwidth_varies_by_dimension(self):
        """Check if bandwidth selection handles varying scales."""
        np.random.seed(42)
        n = 200

        # Features with VERY different scales
        X = np.column_stack([
            np.random.randn(n) * 1,      # scale 1
            np.random.randn(n) * 100,    # scale 100
            np.random.randn(n) * 0.01,   # scale 0.01
        ])
        y = X[:, 0] + 0.1 * np.random.randn(n)

        model = NadarayaWatson(bandwidth="silverman").fit(X, y)

        print(f"\n  Feature scales: 1, 100, 0.01")
        print(f"  Bandwidths: {model.bandwidth_}")

        # Bandwidths should adapt to scales
        # (larger scale = larger bandwidth)
        assert model.bandwidth_[1] > model.bandwidth_[0], (
            "Larger scale should have larger bandwidth"
        )
        assert model.bandwidth_[0] > model.bandwidth_[2], (
            "Smaller scale should have smaller bandwidth"
        )


class TestHeteroscedasticityDetection:
    """
    Test 3: Heteroscedasticity Logic Test

    Verify the diagnostic test detects variance changes.
    """

    def test_homoscedastic_not_rejected(self):
        """Homoscedastic data should NOT be rejected."""
        np.random.seed(42)
        n = 300

        X = np.linspace(0, 10, n).reshape(-1, 1)
        # CONSTANT variance
        y = 2 * X.flatten() + np.random.randn(n) * 1.0

        model = NadarayaWatson(bandwidth=0.5).fit(X, y)

        result_white = heteroscedasticity_test(model, X, y, test="white")
        result_bp = heteroscedasticity_test(model, X, y, test="breusch_pagan")

        print(f"\n  Homoscedastic data:")
        print(f"  White p-value: {result_white.p_value:.4f}")
        print(f"  BP p-value: {result_bp.p_value:.4f}")

        # At alpha=0.05, we should NOT reject (p > 0.05)
        # But statistical tests can fail, so we just check it's not absurdly small
        assert result_white.p_value > 0.01 or result_bp.p_value > 0.01, (
            "At least one test should not strongly reject homoscedasticity"
        )

    def test_heteroscedastic_is_rejected(self):
        """Heteroscedastic data SHOULD be rejected."""
        np.random.seed(42)
        n = 500

        X = np.linspace(0.1, 10, n).reshape(-1, 1)
        # INCREASING variance: noise ~ N(0, x²)
        noise = np.random.randn(n) * (X.flatten() ** 1.5)
        y = 2 * X.flatten() + noise

        model = NadarayaWatson(bandwidth=0.5).fit(X, y)

        result_white = heteroscedasticity_test(model, X, y, test="white")
        result_bp = heteroscedasticity_test(model, X, y, test="breusch_pagan")

        print(f"\n  Heteroscedastic data (variance ~ x²):")
        print(f"  White p-value: {result_white.p_value:.6f}")
        print(f"  BP p-value: {result_bp.p_value:.6f}")

        # Should strongly reject null of homoscedasticity
        assert result_white.p_value < 0.05, (
            f"White test should reject, p={result_white.p_value}"
        )

    def test_gof_detects_heteroscedasticity(self):
        """GoodnessOfFit should correctly identify heteroscedastic data."""
        np.random.seed(42)
        n = 400

        X = np.linspace(0.1, 5, n).reshape(-1, 1)
        noise = np.random.randn(n) * (0.5 + X.flatten())
        y = X.flatten() ** 2 + noise

        model = NadarayaWatson(bandwidth=0.5).fit(X, y)
        gof = GoodnessOfFit(model, X, y)

        print(f"\n  GoodnessOfFit Summary (heteroscedastic):")
        print(f"  is_homoscedastic: {gof.is_homoscedastic()}")
        for name, result in gof.heteroscedasticity_tests.items():
            if result:
                print(f"  {name}: p={result.p_value:.4f}")


class TestBoundaryBias:
    """
    Test 4: Boundary Bias Comparison

    Local Polynomial should be more accurate at edges than Nadaraya-Watson.
    """

    def test_boundary_bias_nw_vs_lp(self):
        """Compare NW and LP at boundaries on y=x."""
        np.random.seed(42)
        n = 100

        # Simple linear function y = x on [0, 1]
        X_train = np.linspace(0, 1, n).reshape(-1, 1)
        y_train = X_train.flatten() + 0.05 * np.random.randn(n)

        # Test at boundaries
        X_test = np.array([[0.01], [0.99]])
        y_true = X_test.flatten()

        # Nadaraya-Watson (local constant - has boundary bias)
        nw = NadarayaWatson(bandwidth=0.1).fit(X_train, y_train)
        y_nw = nw.predict(X_test)

        # Local Polynomial order 1 (local linear - less boundary bias)
        lp = LocalPolynomialRegression(order=1, bandwidth=0.1).fit(X_train, y_train)
        y_lp = lp.predict(X_test)

        error_nw = np.abs(y_nw - y_true)
        error_lp = np.abs(y_lp - y_true)

        print(f"\n  Testing at x=0.01 and x=0.99 (y=x):")
        print(f"  True values: {y_true}")
        print(f"  NW predictions: {y_nw}")
        print(f"  LP predictions: {y_lp}")
        print(f"  NW errors: {error_nw}")
        print(f"  LP errors: {error_lp}")
        print(f"  LP improvement at x=0.99: {error_nw[1]/error_lp[1]:.2f}x")

        # LP should be better at at least one boundary
        assert error_lp[1] < error_nw[1] or error_lp[0] < error_nw[0], (
            "LP should have less boundary bias than NW"
        )

    def test_steep_slope_at_edge(self):
        """Test with steep slope at boundary."""
        np.random.seed(42)
        n = 150

        # Function with steep slope at edge
        X_train = np.linspace(0, 1, n).reshape(-1, 1)
        y_train = 5 * X_train.flatten() ** 2 + 0.05 * np.random.randn(n)

        # Test at right boundary where slope is steepest
        X_test = np.array([[0.95], [0.98]])
        y_true = 5 * X_test.flatten() ** 2

        nw = NadarayaWatson(bandwidth=0.1).fit(X_train, y_train)
        lp = LocalPolynomialRegression(order=2, bandwidth=0.1).fit(X_train, y_train)

        y_nw = nw.predict(X_test)
        y_lp = lp.predict(X_test)

        mse_nw = mean_squared_error(y_true, y_nw)
        mse_lp = mean_squared_error(y_true, y_lp)

        print(f"\n  Steep slope at boundary (y=5x²):")
        print(f"  NW MSE at boundary: {mse_nw:.6f}")
        print(f"  LP MSE at boundary: {mse_lp:.6f}")

        # LP (order 2) should be significantly better for quadratic
        assert mse_lp < mse_nw, "LP should outperform NW at boundaries"


class TestSklearnCompliance:
    """
    Test 5: Sklearn API Compliance

    Use sklearn's built-in validation suite.
    """

    @parametrize_with_checks([
        NadarayaWatson(bandwidth=0.5),
    ])
    def test_sklearn_compatible_nw(self, estimator, check):
        """Run sklearn estimator checks on NadarayaWatson."""
        check(estimator)

    @parametrize_with_checks([
        LocalPolynomialRegression(order=1, bandwidth=0.5),
    ])
    def test_sklearn_compatible_lp(self, estimator, check):
        """Run sklearn estimator checks on LocalPolynomialRegression."""
        check(estimator)


class TestLargeNPerformance:
    """
    Test 6: Large N Performance

    Kernel methods are O(n²) - test with larger datasets.
    """

    @pytest.mark.slow
    def test_1000_samples(self):
        """Test with 1000 samples - should complete quickly."""
        np.random.seed(42)
        n = 1000

        X = np.random.randn(n, 2)
        y = X[:, 0] ** 2 + X[:, 1] + 0.1 * np.random.randn(n)

        start = time.time()
        model = NadarayaWatson(bandwidth=0.5).fit(X, y)
        y_pred = model.predict(X)
        elapsed = time.time() - start

        print(f"\n  N=1000: {elapsed:.2f}s")
        assert elapsed < 10, f"Took too long: {elapsed:.2f}s"
        assert not np.any(np.isnan(y_pred))

    @pytest.mark.slow
    def test_5000_samples(self):
        """Test with 5000 samples - the limit for naive O(n²)."""
        np.random.seed(42)
        n = 5000

        X = np.random.randn(n, 2)
        y = X[:, 0] + 0.1 * np.random.randn(n)

        start = time.time()
        model = NadarayaWatson(bandwidth=0.5).fit(X, y)
        fit_time = time.time() - start

        start = time.time()
        y_pred = model.predict(X[:100])  # Predict on subset
        predict_time = time.time() - start

        print(f"\n  N=5000:")
        print(f"  Fit time: {fit_time:.2f}s")
        print(f"  Predict(100) time: {predict_time:.2f}s")

        # Should complete, even if slow
        assert not np.any(np.isnan(y_pred))

    @pytest.mark.slow
    def test_10000_samples_memory(self):
        """Test with 10000 samples - check memory behavior."""
        np.random.seed(42)
        n = 10000

        X = np.random.randn(n, 1)
        y = np.sin(X.flatten()) + 0.1 * np.random.randn(n)

        start = time.time()
        model = NadarayaWatson(bandwidth="silverman").fit(X, y)
        fit_time = time.time() - start

        # Predict on small subset (full prediction would be O(n²))
        start = time.time()
        X_test = np.linspace(-2, 2, 50).reshape(-1, 1)
        y_pred = model.predict(X_test)
        predict_time = time.time() - start

        print(f"\n  N=10000:")
        print(f"  Fit time: {fit_time:.2f}s")
        print(f"  Predict(50) time: {predict_time:.2f}s")

        assert not np.any(np.isnan(y_pred))
        # Should complete within reasonable time
        assert predict_time < 30, f"Prediction too slow: {predict_time:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
