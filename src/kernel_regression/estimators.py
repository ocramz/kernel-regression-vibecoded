"""
Sklearn-compatible kernel regression estimators.

Includes Nadaraya-Watson and Local Polynomial regression.
"""

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from kernel_regression.bandwidth import (
    CrossValidatedBandwidth,
    silverman_bandwidth,
)
from kernel_regression.kernels import get_kernel, multivariate_kernel_weights


class KernelRegression(RegressorMixin, BaseEstimator):
    """
    Base class for kernel regression estimators.

    This class implements common functionality for kernel-based
    nonparametric regression methods.

    Parameters
    ----------
    kernel : str or callable, default="gaussian"
        Kernel function. Options: "gaussian", "epanechnikov", "uniform",
        "tricube", "biweight", "triweight", "cosine", or a callable.

    bandwidth : float, array-like, or str, default="cv"
        Bandwidth parameter(s).
        - float: Single bandwidth for all dimensions
        - array-like: Per-dimension bandwidths
        - "cv": Cross-validated bandwidth selection
        - "silverman": Silverman's rule of thumb

    cv : int or str, default="loo"
        Cross-validation strategy when bandwidth="cv":
        - "loo": Leave-one-out cross-validation
        - int: Number of folds for k-fold CV

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        Training data

    y_ : ndarray of shape (n_samples,)
        Training targets

    bandwidth_ : ndarray of shape (n_features,)
        Fitted bandwidth values

    n_features_in_ : int
        Number of features seen during fit
    """

    def __init__(
        self,
        kernel: str | Callable = "gaussian",
        bandwidth: float | NDArray[np.floating] | str = "cv",
        cv: int | str = "loo",
    ):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.cv = cv

    def _validate_data_fit(
        self, X: NDArray, y: NDArray
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Validate and convert input data for fitting."""
        X, y = validate_data(self, X, y, y_numeric=True, dtype=np.float64)
        return X, y.astype(np.float64)

    def _validate_data_predict(self, X: NDArray) -> NDArray[np.floating]:
        """Validate and convert input data for prediction."""
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        return X

    def _compute_bandwidth(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        polynomial_order: int = 0,
    ) -> NDArray[np.floating]:
        """Compute or validate bandwidth."""
        if isinstance(self.bandwidth, str):
            if self.bandwidth == "cv":
                selector = CrossValidatedBandwidth(
                    kernel=self.kernel if isinstance(self.kernel, str) else "gaussian",
                    cv=self.cv,
                    polynomial_order=polynomial_order,
                )
                return selector(X, y)
            elif self.bandwidth == "silverman":
                return silverman_bandwidth(X)
            else:
                raise ValueError(f"Unknown bandwidth method: {self.bandwidth}")
        else:
            bandwidth = np.atleast_1d(self.bandwidth).astype(np.float64)
            if bandwidth.size == 1:
                bandwidth = np.full(X.shape[1], bandwidth[0])
            if bandwidth.size != X.shape[1]:
                raise ValueError(
                    f"bandwidth has {bandwidth.size} values, expected {X.shape[1]}"
                )
            return bandwidth

    def fit(self, X: NDArray, y: NDArray) -> "KernelRegression":
        """
        Fit the kernel regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self
            Fitted estimator
        """
        X, y = self._validate_data_fit(X, y)

        self.X_ = X
        self.y_ = y
        self.bandwidth_ = self._compute_bandwidth(X, y)
        self.kernel_func_ = get_kernel(self.kernel)

        return self

    def predict(self, X: NDArray) -> NDArray[np.floating]:
        """
        Predict using the kernel regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values
        """
        raise NotImplementedError("Subclasses must implement predict()")


class NadarayaWatson(KernelRegression):
    """
    Nadaraya-Watson kernel regression estimator.

    Implements local constant (order 0) kernel regression using the
    Nadaraya-Watson estimator:

        ŷ(x) = Σ K((x - x_i) / h) * y_i / Σ K((x - x_i) / h)

    Parameters
    ----------
    kernel : str or callable, default="gaussian"
        Kernel function. Options: "gaussian", "epanechnikov", "uniform",
        "tricube", "biweight", "triweight", "cosine", or a callable.

    bandwidth : float, array-like, or str, default="cv"
        Bandwidth parameter(s).
        - float: Single bandwidth for all dimensions
        - array-like: Per-dimension bandwidths
        - "cv": Cross-validated bandwidth selection
        - "silverman": Silverman's rule of thumb

    cv : int or str, default="loo"
        Cross-validation strategy when bandwidth="cv"

    Examples
    --------
    >>> import numpy as np
    >>> from kernel_regression import NadarayaWatson
    >>> X = np.random.randn(100, 2)
    >>> y = np.sin(X[:, 0]) + 0.1 * np.random.randn(100)
    >>> model = NadarayaWatson(kernel="gaussian", bandwidth="cv")
    >>> model.fit(X, y)
    >>> predictions = model.predict(X[:5])
    """

    def predict(self, X: NDArray) -> NDArray[np.floating]:
        """
        Predict using Nadaraya-Watson estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values
        """
        X = self._validate_data_predict(X)

        weights = multivariate_kernel_weights(
            X, self.X_, self.bandwidth_, self.kernel_func_
        )

        # Nadaraya-Watson: weighted average
        weight_sums = np.sum(weights, axis=1)
        # Avoid division by zero
        weight_sums = np.where(weight_sums > 0, weight_sums, 1.0)

        y_pred = np.sum(weights * self.y_, axis=1) / weight_sums

        # Handle points with no neighbors (all weights zero)
        no_neighbors = np.sum(weights, axis=1) == 0
        if np.any(no_neighbors):
            y_pred[no_neighbors] = np.mean(self.y_)

        return y_pred

    def get_weights(self, X: NDArray) -> NDArray[np.floating]:
        """
        Get kernel weights for prediction points.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Points to get weights for

        Returns
        -------
        weights : ndarray of shape (n_samples, n_train)
            Normalized kernel weights
        """
        X = self._validate_data_predict(X)

        weights = multivariate_kernel_weights(
            X, self.X_, self.bandwidth_, self.kernel_func_
        )

        # Normalize
        weight_sums = np.sum(weights, axis=1, keepdims=True)
        weight_sums = np.where(weight_sums > 0, weight_sums, 1.0)
        return weights / weight_sums


class LocalPolynomialRegression(KernelRegression):
    """
    Local polynomial kernel regression estimator.

    Fits a weighted polynomial locally at each prediction point.
    Order 0 is equivalent to Nadaraya-Watson; order 1 is local linear.

    Parameters
    ----------
    kernel : str or callable, default="gaussian"
        Kernel function.

    bandwidth : float, array-like, or str, default="cv"
        Bandwidth parameter(s).

    cv : int or str, default="loo"
        Cross-validation strategy when bandwidth="cv"

    order : int or str, default="cv"
        Polynomial order.
        - int: Fixed polynomial order (0=constant, 1=linear, 2=quadratic, ...)
        - "cv": Cross-validated order selection (searches 0, 1, 2)

    max_order : int, default=2
        Maximum order to consider when order="cv"

    regularization : float, default=1e-10
        Ridge regularization for numerical stability

    Examples
    --------
    >>> import numpy as np
    >>> from kernel_regression import LocalPolynomialRegression
    >>> X = np.random.randn(100, 2)
    >>> y = X[:, 0]**2 + 0.1 * np.random.randn(100)
    >>> model = LocalPolynomialRegression(order=2, bandwidth="cv")
    >>> model.fit(X, y)
    >>> predictions = model.predict(X[:5])
    """

    def __init__(
        self,
        kernel: str | Callable = "gaussian",
        bandwidth: float | NDArray[np.floating] | str = "cv",
        cv: int | str = "loo",
        order: int | str = 1,
        max_order: int = 2,
        regularization: float = 1e-10,
    ):
        super().__init__(kernel=kernel, bandwidth=bandwidth, cv=cv)
        self.order = order
        self.max_order = max_order
        self.regularization = regularization

    def fit(self, X: NDArray, y: NDArray) -> "LocalPolynomialRegression":
        """
        Fit the local polynomial regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self
            Fitted estimator
        """
        X, y = self._validate_data_fit(X, y)

        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]
        self.kernel_func_ = get_kernel(self.kernel)

        # Select polynomial order
        if isinstance(self.order, str) and self.order == "cv":
            self.order_ = self._select_order_cv(X, y)
        else:
            self.order_ = int(self.order)

        # Select bandwidth (after order is determined)
        self.bandwidth_ = self._compute_bandwidth(X, y, self.order_)

        return self

    def _select_order_cv(
        self, X: NDArray[np.floating], y: NDArray[np.floating]
    ) -> int:
        """Select polynomial order via cross-validation."""
        best_order = 0
        best_error = np.inf

        for order in range(self.max_order + 1):
            # Get bandwidth for this order
            selector = CrossValidatedBandwidth(
                kernel=self.kernel if isinstance(self.kernel, str) else "gaussian",
                cv=self.cv,
                polynomial_order=order,
            )
            bandwidth = selector(X, y)

            # Compute CV error
            error = self._compute_cv_error(X, y, bandwidth, order)

            if error < best_error:
                best_error = error
                best_order = order

        return best_order

    def _compute_cv_error(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        bandwidth: NDArray[np.floating],
        order: int,
    ) -> float:
        """Compute leave-one-out CV error for given order and bandwidth."""
        n_samples = X.shape[0]
        errors = np.zeros(n_samples)

        for i in range(n_samples):
            mask = np.ones(n_samples, dtype=bool)
            mask[i] = False

            y_pred = self._predict_point(
                X[i : i + 1], X[mask], y[mask], bandwidth, order
            )
            errors[i] = (y[i] - y_pred[0]) ** 2

        return float(np.mean(errors))

    def _build_design_matrix(
        self,
        diff: NDArray[np.floating],
        order: int,
    ) -> NDArray[np.floating]:
        """
        Build polynomial design matrix.

        For multivariate case, includes all polynomial terms up to given order.
        """
        n_samples, n_features = diff.shape

        # Start with intercept
        columns = [np.ones(n_samples)]

        if order >= 1:
            # Linear terms
            for j in range(n_features):
                columns.append(diff[:, j])

        if order >= 2:
            # Quadratic terms: x_j^2 and cross terms x_j * x_k
            for j in range(n_features):
                columns.append(diff[:, j] ** 2)
            for j in range(n_features):
                for k in range(j + 1, n_features):
                    columns.append(diff[:, j] * diff[:, k])

        if order >= 3:
            # Cubic terms
            for j in range(n_features):
                columns.append(diff[:, j] ** 3)

        return np.column_stack(columns)

    def _predict_point(
        self,
        x: NDArray[np.floating],
        X_train: NDArray[np.floating],
        y_train: NDArray[np.floating],
        bandwidth: NDArray[np.floating],
        order: int,
    ) -> NDArray[np.floating]:
        """Predict at point(s) using local polynomial."""
        x = np.atleast_2d(x)
        n_pred = x.shape[0]

        y_pred = np.zeros(n_pred)

        for i in range(n_pred):
            weights = multivariate_kernel_weights(
                x[i : i + 1], X_train, bandwidth, self.kernel_func_
            ).flatten()

            # Build design matrix
            diff = X_train - x[i]
            design = self._build_design_matrix(diff, order)

            W = np.diag(weights)

            # Weighted least squares with regularization
            XtW = design.T @ W
            XtWX = XtW @ design
            XtWy = XtW @ y_train

            # Add ridge regularization
            reg = self.regularization * np.eye(XtWX.shape[0])

            try:
                beta = np.linalg.solve(XtWX + reg, XtWy)
                y_pred[i] = beta[0]  # Intercept = prediction at x[i]
            except np.linalg.LinAlgError:
                # Fallback: weighted average
                weight_sum = np.sum(weights)
                if weight_sum > 0:
                    y_pred[i] = np.sum(weights * y_train) / weight_sum
                else:
                    y_pred[i] = np.mean(y_train)

        return y_pred

    def predict(self, X: NDArray) -> NDArray[np.floating]:
        """
        Predict using local polynomial regression.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values
        """
        X = self._validate_data_predict(X)
        return self._predict_point(X, self.X_, self.y_, self.bandwidth_, self.order_)

    def predict_with_derivatives(
        self, X: NDArray
    ) -> tuple[NDArray[np.floating], NDArray[np.floating] | None]:
        """
        Predict with gradient estimates.

        Only available when order >= 1.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values
        gradients : ndarray of shape (n_samples, n_features) or None
            Estimated gradients (None if order=0)
        """
        X = self._validate_data_predict(X)

        if self.order_ == 0:
            return self.predict(X), None

        n_pred = X.shape[0]
        n_features = self.n_features_in_

        y_pred = np.zeros(n_pred)
        gradients = np.zeros((n_pred, n_features))

        for i in range(n_pred):
            weights = multivariate_kernel_weights(
                X[i : i + 1], self.X_, self.bandwidth_, self.kernel_func_
            ).flatten()

            diff = self.X_ - X[i]
            design = self._build_design_matrix(diff, self.order_)

            W = np.diag(weights)
            XtW = design.T @ W
            XtWX = XtW @ design
            XtWy = XtW @ self.y_

            reg = self.regularization * np.eye(XtWX.shape[0])

            try:
                beta = np.linalg.solve(XtWX + reg, XtWy)
                y_pred[i] = beta[0]
                # Gradient is the linear coefficients (indices 1 to n_features)
                gradients[i] = beta[1 : n_features + 1]
            except np.linalg.LinAlgError:
                weight_sum = np.sum(weights)
                if weight_sum > 0:
                    y_pred[i] = np.sum(weights * self.y_) / weight_sum
                else:
                    y_pred[i] = np.mean(self.y_)
                gradients[i] = 0.0

        return y_pred, gradients
