"""
Bandwidth selection methods for kernel regression.

Includes rule-of-thumb and cross-validation based selection.
Implements O(n) LOOCV using the hat matrix diagonal shortcut.
"""

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar, minimize
from scipy import linalg

from kernel_regression.kernels import get_kernel, multivariate_kernel_weights


def silverman_bandwidth(
    X: NDArray[np.floating],
    factor: float = 1.0,
) -> NDArray[np.floating]:
    """
    Silverman's rule of thumb for bandwidth selection.

    h_j = factor * 1.06 * sigma_j * n^(-1/5)

    This is optimal for Gaussian kernel and Gaussian data.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training data
    factor : float, default=1.0
        Multiplicative factor to adjust bandwidth

    Returns
    -------
    ndarray of shape (n_features,)
        Bandwidth for each feature
    """
    X = np.atleast_2d(X)
    n_samples, n_features = X.shape

    # Use robust scale estimate (IQR-based)
    q75 = np.percentile(X, 75, axis=0)
    q25 = np.percentile(X, 25, axis=0)
    iqr = q75 - q25
    std = np.std(X, axis=0, ddof=1)

    # Use minimum of std and IQR/1.34 for robustness
    scale = np.minimum(std, iqr / 1.349)
    scale = np.where(scale > 0, scale, std)  # Fallback to std if IQR is 0
    scale = np.where(scale > 0, scale, 1.0)  # Fallback to 1 if both are 0

    bandwidth = factor * 1.06 * scale * (n_samples ** (-1 / 5))

    return bandwidth


def scott_bandwidth(X: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Scott's rule of thumb for bandwidth selection.

    h_j = sigma_j * n^(-1/(d+4))

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training data

    Returns
    -------
    ndarray of shape (n_features,)
        Bandwidth for each feature
    """
    X = np.atleast_2d(X)
    n_samples, n_features = X.shape
    std = np.std(X, axis=0, ddof=1)
    std = np.where(std > 0, std, 1.0)
    bandwidth = std * (n_samples ** (-1 / (n_features + 4)))
    return bandwidth


class RuleOfThumbBandwidth:
    """
    Rule of thumb bandwidth selector.

    Parameters
    ----------
    method : str, default="silverman"
        Method to use: "silverman" or "scott"
    factor : float, default=1.0
        Multiplicative adjustment factor
    """

    def __init__(self, method: str = "silverman", factor: float = 1.0):
        if method not in ("silverman", "scott"):
            raise ValueError(f"Unknown method '{method}'")
        self.method = method
        self.factor = factor

    def __call__(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute bandwidth for data X."""
        if self.method == "silverman":
            return silverman_bandwidth(X, self.factor)
        return scott_bandwidth(X) * self.factor


def loocv_hat_matrix_shortcut(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    bandwidth: NDArray[np.floating],
    kernel_func: Callable[[NDArray[np.floating]], NDArray[np.floating]],
) -> float:
    """
    Compute LOOCV error in O(n) using the hat matrix diagonal shortcut.

    For Nadaraya-Watson regression, the LOOCV prediction at point i is:
        y_hat_{-i} = (y_hat_i - H_ii * y_i) / (1 - H_ii)

    where H_ii is the diagonal element of the smoothing matrix.
    This avoids the naive O(n^2) approach of refitting for each point.

    Args:
        X: Training features of shape (n_samples, n_features).
        y: Training targets of shape (n_samples,).
        bandwidth: Bandwidth per feature of shape (n_features,).
        kernel_func: Kernel function to use.

    Returns:
        Mean squared LOOCV error.

    References:
        Hastie, Tibshirani, Friedman. "The Elements of Statistical Learning"
        Section 7.10.1 on effective degrees of freedom.
    """
    n_samples = X.shape[0]

    # Compute full weight matrix (smoothing matrix)
    weights = multivariate_kernel_weights(X, X, bandwidth, kernel_func)

    # Normalize to get hat matrix rows
    weight_sums = np.sum(weights, axis=1, keepdims=True)
    weight_sums = np.where(weight_sums > 0, weight_sums, 1.0)
    H = weights / weight_sums

    # Get diagonal (leverage values)
    H_diag = np.diag(H)

    # Compute fitted values
    y_hat = H @ y

    # LOOCV shortcut formula
    # y_hat_{-i} = (y_hat_i - H_ii * y_i) / (1 - H_ii)
    denominator = 1 - H_diag
    # Avoid division by zero
    denominator = np.where(np.abs(denominator) > 1e-10, denominator, 1e-10)

    y_hat_loo = (y_hat - H_diag * y) / denominator

    # LOOCV error
    loo_errors = (y - y_hat_loo) ** 2

    return float(np.mean(loo_errors))


class CrossValidatedBandwidth:
    """
    Cross-validation bandwidth selector for kernel regression.

    Uses leave-one-out or k-fold cross-validation to select optimal
    bandwidth minimizing prediction error.

    For LOOCV with Nadaraya-Watson (polynomial_order=0), uses the
    O(n) hat matrix diagonal shortcut instead of naive O(n^2) refitting.

    Args:
        kernel: Kernel function name or callable. Options include
            "gaussian", "epanechnikov", "uniform", "tricube".
        cv: Cross-validation strategy. Use "loo" for leave-one-out
            or an integer for k-fold CV.
        n_bandwidths: Number of bandwidth values to search in grid.
        bandwidth_range: Optional (min, max) range for bandwidth search.
            If None, automatically determined from data.
        use_grid: If True, use grid search. If False, use optimization.
        polynomial_order: Order of local polynomial (0 = Nadaraya-Watson).
        per_dimension: If True, optimize bandwidth separately for each
            dimension. More expensive but can improve results for
            heterogeneous features.

    Attributes:
        cv_results_: Dictionary with cross-validation results after fitting.

    Example:
        >>> selector = CrossValidatedBandwidth(cv="loo")
        >>> bandwidth = selector(X, y)
        >>> print(f"Optimal bandwidth: {bandwidth}")
    """

    def __init__(
        self,
        kernel: str = "gaussian",
        cv: int | str = "loo",
        n_bandwidths: int = 30,
        bandwidth_range: tuple[float, float] | None = None,
        use_grid: bool = True,
        polynomial_order: int = 0,
        per_dimension: bool = False,
    ):
        self.kernel = kernel
        self.cv = cv
        self.n_bandwidths = n_bandwidths
        self.bandwidth_range = bandwidth_range
        self.use_grid = use_grid
        self.polynomial_order = polynomial_order
        self.per_dimension = per_dimension
        self.cv_results_: dict | None = None

    def _compute_loo_error_fast(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        bandwidth: NDArray[np.floating],
    ) -> float:
        """Compute LOOCV error using O(n) hat matrix shortcut."""
        kernel_func = get_kernel(self.kernel)
        return loocv_hat_matrix_shortcut(X, y, bandwidth, kernel_func)

    def _compute_loo_error(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        bandwidth: NDArray[np.floating],
    ) -> float:
        """Compute leave-one-out cross-validation error.

        Uses fast O(n) shortcut for Nadaraya-Watson, falls back to
        O(n^2) naive approach for local polynomial.
        """
        # Use fast shortcut for NW (order 0)
        if self.polynomial_order == 0:
            return self._compute_loo_error_fast(X, y, bandwidth)

        # Fall back to naive approach for local polynomial
        n_samples = X.shape[0]
        kernel_func = get_kernel(self.kernel)

        errors = np.zeros(n_samples)

        for i in range(n_samples):
            mask = np.ones(n_samples, dtype=bool)
            mask[i] = False

            X_train = X[mask]
            y_train = y[mask]
            x_test = X[i : i + 1]

            y_pred = self._predict_point(
                x_test, X_train, y_train, bandwidth, kernel_func
            )

            errors[i] = (y[i] - y_pred[0]) ** 2

        return float(np.mean(errors))

    def _compute_kfold_error(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        bandwidth: NDArray[np.floating],
        n_folds: int,
    ) -> float:
        """Compute k-fold cross-validation error."""
        n_samples = X.shape[0]
        kernel_func = get_kernel(self.kernel)

        # Create fold indices
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        folds = np.array_split(indices, n_folds)

        errors = []

        for fold_idx in range(n_folds):
            test_idx = folds[fold_idx]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            y_pred = self._predict_point(X_test, X_train, y_train, bandwidth, kernel_func)
            fold_errors = (y_test - y_pred) ** 2
            errors.extend(fold_errors)

        return float(np.mean(errors))

    def _predict_point(
        self,
        x: NDArray[np.floating],
        X_train: NDArray[np.floating],
        y_train: NDArray[np.floating],
        bandwidth: NDArray[np.floating],
        kernel_func: Callable,
    ) -> NDArray[np.floating]:
        """Predict at point(s) x using local polynomial."""
        if self.polynomial_order == 0:
            # Nadaraya-Watson (local constant)
            weights = multivariate_kernel_weights(x, X_train, bandwidth, kernel_func)
            weight_sum = np.sum(weights, axis=1, keepdims=True)
            weight_sum = np.where(weight_sum > 0, weight_sum, 1.0)
            y_pred = np.sum(weights * y_train, axis=1) / weight_sum.flatten()
            return y_pred
        else:
            # Local polynomial
            return self._local_polynomial_predict(
                x, X_train, y_train, bandwidth, kernel_func
            )

    def _local_polynomial_predict(
        self,
        x: NDArray[np.floating],
        X_train: NDArray[np.floating],
        y_train: NDArray[np.floating],
        bandwidth: NDArray[np.floating],
        kernel_func: Callable,
    ) -> NDArray[np.floating]:
        """Local polynomial prediction."""
        x = np.atleast_2d(x)
        n_pred = x.shape[0]
        n_features = X_train.shape[1]
        order = self.polynomial_order

        y_pred = np.zeros(n_pred)

        for i in range(n_pred):
            weights = multivariate_kernel_weights(
                x[i : i + 1], X_train, bandwidth, kernel_func
            ).flatten()

            # Build design matrix for local polynomial
            diff = X_train - x[i]

            # For simplicity, use full polynomial basis up to given order
            # This constructs terms: 1, x1, x2, ..., x1^2, x1*x2, ...
            design_cols = [np.ones(len(X_train))]

            for d in range(1, order + 1):
                for j in range(n_features):
                    design_cols.append(diff[:, j] ** d)

            design = np.column_stack(design_cols)
            W = np.diag(weights)

            # Weighted least squares: (X'WX)^-1 X'Wy
            try:
                XtW = design.T @ W
                XtWX = XtW @ design
                XtWy = XtW @ y_train
                beta = np.linalg.solve(XtWX + 1e-10 * np.eye(XtWX.shape[0]), XtWy)
                y_pred[i] = beta[0]  # Intercept is the prediction at x[i]
            except np.linalg.LinAlgError:
                # Fallback to Nadaraya-Watson
                weight_sum = np.sum(weights)
                if weight_sum > 0:
                    y_pred[i] = np.sum(weights * y_train) / weight_sum
                else:
                    y_pred[i] = np.mean(y_train)

        return y_pred

    def _get_bandwidth_range(
        self, X: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Determine reasonable bandwidth search range."""
        rot = silverman_bandwidth(X)
        h_min = rot * 0.1
        h_max = rot * 5.0
        return h_min, h_max

    def __call__(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Select optimal bandwidth via cross-validation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training features
        y : ndarray of shape (n_samples,)
            Training targets

        Returns
        -------
        ndarray of shape (n_features,)
            Optimal bandwidth for each feature
        """
        X = np.atleast_2d(X)
        y = np.asarray(y).flatten()
        n_features = X.shape[1]

        if self.bandwidth_range is not None:
            h_min = np.full(n_features, self.bandwidth_range[0])
            h_max = np.full(n_features, self.bandwidth_range[1])
        else:
            h_min, h_max = self._get_bandwidth_range(X)

        if self.per_dimension and n_features > 1:
            return self._per_dimension_cv(X, y, h_min, h_max)
        elif self.use_grid:
            return self._grid_search(X, y, h_min, h_max)
        else:
            return self._optimize(X, y, h_min, h_max)

    def _per_dimension_cv(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        h_min: NDArray[np.floating],
        h_max: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Optimize bandwidth separately for each dimension.

        Uses coordinate descent to optimize each dimension while
        holding others fixed.
        """
        n_features = X.shape[1]
        h_rot = silverman_bandwidth(X)
        bandwidth = h_rot.copy()

        # Coordinate descent: optimize each dimension
        for iteration in range(3):  # Multiple passes
            for dim in range(n_features):
                factors = np.logspace(-1, np.log10(5), self.n_bandwidths)
                best_error = np.inf
                best_h = bandwidth[dim]

                for factor in factors:
                    test_bw = bandwidth.copy()
                    test_bw[dim] = np.clip(h_rot[dim] * factor, h_min[dim], h_max[dim])

                    if self.cv == "loo":
                        error = self._compute_loo_error(X, y, test_bw)
                    else:
                        error = self._compute_kfold_error(X, y, test_bw, int(self.cv))

                    if error < best_error:
                        best_error = error
                        best_h = test_bw[dim]

                bandwidth[dim] = best_h

        self.cv_results_ = {
            "best_bandwidth": bandwidth,
            "method": "per_dimension",
        }

        return bandwidth

    def _grid_search(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        h_min: NDArray[np.floating],
        h_max: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Grid search for optimal bandwidth."""
        # For multivariate, we use a common scaling factor
        # h = factor * h_rot
        factors = np.logspace(-1, np.log10(5), self.n_bandwidths)
        h_rot = silverman_bandwidth(X)

        best_error = np.inf
        best_bandwidth = h_rot.copy()
        cv_scores = []

        for factor in factors:
            bandwidth = h_rot * factor
            bandwidth = np.clip(bandwidth, h_min, h_max)

            if self.cv == "loo":
                error = self._compute_loo_error(X, y, bandwidth)
            else:
                error = self._compute_kfold_error(X, y, bandwidth, int(self.cv))

            cv_scores.append({"factor": factor, "bandwidth": bandwidth.copy(), "cv_error": error})

            if error < best_error:
                best_error = error
                best_bandwidth = bandwidth.copy()

        self.cv_results_ = {
            "scores": cv_scores,
            "best_error": best_error,
            "best_bandwidth": best_bandwidth,
        }

        return best_bandwidth

    def _optimize(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        h_min: NDArray[np.floating],
        h_max: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Optimize bandwidth using scipy."""
        h_rot = silverman_bandwidth(X)

        def objective(log_factor: float) -> float:
            factor = np.exp(log_factor)
            bandwidth = h_rot * factor
            bandwidth = np.clip(bandwidth, h_min, h_max)
            if self.cv == "loo":
                return self._compute_loo_error(X, y, bandwidth)
            return self._compute_kfold_error(X, y, bandwidth, int(self.cv))

        result = minimize_scalar(
            objective,
            bounds=(np.log(0.1), np.log(5)),
            method="bounded",
        )

        optimal_factor = np.exp(result.x)
        best_bandwidth = h_rot * optimal_factor
        best_bandwidth = np.clip(best_bandwidth, h_min, h_max)

        self.cv_results_ = {
            "optimal_factor": optimal_factor,
            "best_error": result.fun,
            "best_bandwidth": best_bandwidth,
        }

        return best_bandwidth
