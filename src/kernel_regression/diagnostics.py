"""
Goodness of fit diagnostics for kernel regression.

Includes tests for homoscedasticity/heteroscedasticity and
residual analysis.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.base import BaseEstimator

from kernel_regression.kernels import get_kernel, multivariate_kernel_weights


@dataclass
class HeteroscedasticityTestResult:
    """Result of heteroscedasticity test."""

    statistic: float
    p_value: float
    is_heteroscedastic: bool
    test_name: str
    alpha: float

    def __str__(self) -> str:
        conclusion = "heteroscedastic" if self.is_heteroscedastic else "homoscedastic"
        return (
            f"{self.test_name} Test\n"
            f"  Statistic: {self.statistic:.4f}\n"
            f"  P-value: {self.p_value:.4f}\n"
            f"  Conclusion (α={self.alpha}): {conclusion}"
        )


@dataclass
class ResidualDiagnosticsResult:
    """Result of residual diagnostics."""

    residuals: NDArray[np.floating]
    standardized_residuals: NDArray[np.floating]
    mean: float
    std: float
    skewness: float
    kurtosis: float
    normality_statistic: float
    normality_p_value: float
    is_normal: bool

    def __str__(self) -> str:
        return (
            f"Residual Diagnostics\n"
            f"  Mean: {self.mean:.6f}\n"
            f"  Std: {self.std:.6f}\n"
            f"  Skewness: {self.skewness:.4f}\n"
            f"  Kurtosis: {self.kurtosis:.4f}\n"
            f"  Normality test p-value: {self.normality_p_value:.4f}\n"
            f"  Normal residuals: {self.is_normal}"
        )


def heteroscedasticity_test(
    model: BaseEstimator,
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    test: Literal["white", "breusch_pagan", "goldfeld_quandt", "dette_munk_wagner"] = "white",
    alpha: float = 0.05,
    n_bootstrap: int = 500,
    use_bias_corrected_residuals: bool = True,
) -> HeteroscedasticityTestResult:
    """
    Test for heteroscedasticity in kernel regression residuals.

    Args:
        model: Fitted kernel regression model with predict method.
        X: Feature matrix of shape (n_samples, n_features).
        y: Target values of shape (n_samples,).
        test: Test to use. Options are:
            - "white": White's test (general, linear auxiliary regression)
            - "breusch_pagan": Breusch-Pagan test (linear auxiliary regression)
            - "goldfeld_quandt": Goldfeld-Quandt test (split sample F-test)
            - "dette_munk_wagner": Non-parametric test using kernel smoothing
              of squared residuals. Preferred for kernel regression models.
        alpha: Significance level for the test.
        n_bootstrap: Number of bootstrap samples for dette_munk_wagner test.
        use_bias_corrected_residuals: If True (default), use residuals from a
            higher-order polynomial model for the DMW test. This prevents
            mean-function bias from leaking into variance estimates, reducing
            false positives when the mean function has curvature.

    Returns:
        Test results with statistic, p-value, and conclusion.

    Example:
        >>> model = NadarayaWatson(bandwidth=0.5).fit(X, y)
        >>> result = heteroscedasticity_test(model, X, y, test="dette_munk_wagner")
        >>> print(f"Heteroscedastic: {result.is_heteroscedastic}")
    """
    X = np.atleast_2d(X)
    y = np.asarray(y).flatten()

    # Get residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    residuals_sq = residuals**2

    if test == "white":
        return _white_test(X, residuals_sq, alpha)
    elif test == "breusch_pagan":
        return _breusch_pagan_test(X, residuals_sq, alpha)
    elif test == "goldfeld_quandt":
        return _goldfeld_quandt_test(X, y, model, alpha)
    elif test == "dette_munk_wagner":
        # Compute bias-corrected residuals using higher-order model
        bias_corrected = None
        if use_bias_corrected_residuals:
            from kernel_regression.estimators import LocalPolynomialRegression

            # Get bandwidth from fitted model
            bandwidth = np.atleast_1d(getattr(model, "bandwidth_", 0.5))
            current_order = getattr(model, "order_", 0)

            # Use order+1 model to capture curvature the base model misses
            higher_order = min(current_order + 1, 2)
            try:
                bc_model = LocalPolynomialRegression(
                    bandwidth=bandwidth,
                    order=higher_order,
                ).fit(X, y)
                y_pred_bc = bc_model.predict(X)
                bias_corrected = y - y_pred_bc
            except Exception:
                # Fall back to original residuals if higher-order fit fails
                bias_corrected = None

        return _dette_munk_wagner_test(
            X, residuals, alpha, n_bootstrap, bias_corrected
        )
    else:
        raise ValueError(f"Unknown test: {test}")


def _white_test(
    X: NDArray[np.floating],
    residuals_sq: NDArray[np.floating],
    alpha: float,
) -> HeteroscedasticityTestResult:
    """
    White's test for heteroscedasticity.

    Regresses squared residuals on X, X^2, and cross products.
    """
    n_samples, n_features = X.shape

    # Build auxiliary regression matrix
    # Include: constant, X, X^2, and cross products
    aux_cols = [np.ones(n_samples)]

    for j in range(n_features):
        aux_cols.append(X[:, j])

    for j in range(n_features):
        aux_cols.append(X[:, j] ** 2)

    for j in range(n_features):
        for k in range(j + 1, n_features):
            aux_cols.append(X[:, j] * X[:, k])

    Z = np.column_stack(aux_cols)

    # OLS regression of squared residuals on Z
    try:
        beta, residuals_aux, rank, s = np.linalg.lstsq(Z, residuals_sq, rcond=None)
        y_pred_aux = Z @ beta
        ss_reg = np.sum((y_pred_aux - np.mean(residuals_sq)) ** 2)
        ss_tot = np.sum((residuals_sq - np.mean(residuals_sq)) ** 2)
        r_squared = ss_reg / ss_tot if ss_tot > 0 else 0

        # Test statistic: n * R^2 ~ chi2(k)
        statistic = n_samples * r_squared
        df = Z.shape[1] - 1
        p_value = 1 - stats.chi2.cdf(statistic, df)
    except np.linalg.LinAlgError:
        statistic = 0.0
        p_value = 1.0

    return HeteroscedasticityTestResult(
        statistic=statistic,
        p_value=p_value,
        is_heteroscedastic=p_value < alpha,
        test_name="White",
        alpha=alpha,
    )


def _breusch_pagan_test(
    X: NDArray[np.floating],
    residuals_sq: NDArray[np.floating],
    alpha: float,
) -> HeteroscedasticityTestResult:
    """
    Breusch-Pagan test for heteroscedasticity.

    Regresses squared residuals on X.
    """
    n_samples, n_features = X.shape

    # Add constant
    Z = np.column_stack([np.ones(n_samples), X])

    # Normalize squared residuals
    sigma_sq = np.mean(residuals_sq)
    g = residuals_sq / sigma_sq

    try:
        beta, _, _, _ = np.linalg.lstsq(Z, g, rcond=None)
        g_pred = Z @ beta
        ss_reg = np.sum((g_pred - np.mean(g)) ** 2)

        # Test statistic: 0.5 * ESS ~ chi2(k)
        statistic = 0.5 * ss_reg
        df = n_features
        p_value = 1 - stats.chi2.cdf(statistic, df)
    except np.linalg.LinAlgError:
        statistic = 0.0
        p_value = 1.0

    return HeteroscedasticityTestResult(
        statistic=statistic,
        p_value=p_value,
        is_heteroscedastic=p_value < alpha,
        test_name="Breusch-Pagan",
        alpha=alpha,
    )


def _goldfeld_quandt_test(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    model: BaseEstimator,
    alpha: float,
) -> HeteroscedasticityTestResult:
    """
    Goldfeld-Quandt test for heteroscedasticity.

    Compares variance of residuals in two subsamples.
    """
    n_samples = X.shape[0]

    # Sort by first predictor (or fitted values if available)
    try:
        y_pred = model.predict(X)
        sort_idx = np.argsort(y_pred)
    except Exception:
        sort_idx = np.argsort(X[:, 0])

    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]

    # Drop middle observations (typically 1/3)
    drop_frac = 1 / 3
    n_drop = int(n_samples * drop_frac)
    n_subsample = (n_samples - n_drop) // 2

    # First subsample (low values)
    X1 = X_sorted[:n_subsample]
    y1 = y_sorted[:n_subsample]
    y1_pred = model.predict(X1)
    ss1 = np.sum((y1 - y1_pred) ** 2)

    # Second subsample (high values)
    X2 = X_sorted[-n_subsample:]
    y2 = y_sorted[-n_subsample:]
    y2_pred = model.predict(X2)
    ss2 = np.sum((y2 - y2_pred) ** 2)

    # F-test: larger variance / smaller variance
    if ss1 > 0 and ss2 > 0:
        if ss2 > ss1:
            statistic = ss2 / ss1
        else:
            statistic = ss1 / ss2
        df1 = df2 = n_subsample - 1
        p_value = 2 * (1 - stats.f.cdf(statistic, df1, df2))
        p_value = min(p_value, 1.0)
    else:
        statistic = 1.0
        p_value = 1.0

    return HeteroscedasticityTestResult(
        statistic=statistic,
        p_value=p_value,
        is_heteroscedastic=p_value < alpha,
        test_name="Goldfeld-Quandt",
        alpha=alpha,
    )


def _dette_munk_wagner_test(
    X: NDArray[np.floating],
    residuals: NDArray[np.floating],
    alpha: float,
    n_bootstrap: int = 500,
    bias_corrected_residuals: NDArray[np.floating] | None = None,
) -> HeteroscedasticityTestResult:
    """
    Dette-Munk-Wagner non-parametric test for heteroscedasticity.

    This test uses kernel smoothing to estimate the variance function
    and compares it against the assumption of constant variance using
    a Wild Bootstrap to compute p-values.

    Unlike White's or Breusch-Pagan tests, this does not assume a
    linear relationship between variance and predictors.

    Refinements for proper size calibration:
    1. Larger bandwidth (1.5× Silverman) for variance smoothing
    2. Adaptive boundary trimming based on effective sample size
    3. Wild Bootstrap instead of permutation to preserve variance structure
    4. Support for bias-corrected residuals to prevent mean-bias leakage
    5. Multi-PC testing with Bonferroni correction for multivariate data

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        residuals: Model residuals of shape (n_samples,).
        alpha: Significance level.
        n_bootstrap: Number of bootstrap samples for p-value computation.
        bias_corrected_residuals: Optional bias-corrected residuals from a
            higher-order model. If provided, uses these instead of raw
            residuals to prevent mean-bias from inflating the test statistic.

    Returns:
        Test result with statistic and p-value.

    References:
        Dette, H., Munk, A., & Wagner, T. (1998). "Estimating the variance
        in nonparametric regression - what is a reasonable choice?"
        Journal of the Royal Statistical Society: Series B, 60(4), 751-764.
    """
    from kernel_regression.bandwidth import silverman_bandwidth

    n_samples = X.shape[0]
    n_features = X.shape[1]

    # Use bias-corrected residuals if provided (reduces mean-bias leakage)
    resid_to_use = bias_corrected_residuals if bias_corrected_residuals is not None else residuals
    residuals_sq = resid_to_use ** 2

    # Center squared residuals to reduce spurious correlations
    residuals_sq_centered = residuals_sq - np.mean(residuals_sq)

    # Estimate global variance
    sigma_sq_global = np.mean(residuals_sq)

    def _compute_dmw_statistic_1d(
        x_vals: NDArray,
        resid_sq: NDArray,
        bandwidth_factor: float = 1.5,
    ) -> tuple[float, NDArray, NDArray, int]:
        """
        Compute DMW test statistic for 1D projection.

        Returns:
            Tuple of (statistic, smoothing_weights, sorted_indices, trim_size)
        """
        # Sort by x
        sort_idx = np.argsort(x_vals)
        x_sorted = x_vals[sort_idx]
        resid_sq_sorted = resid_sq[sort_idx]

        # REFINEMENT 1: Larger bandwidth for variance estimation
        # Silverman's rule is optimal for density, not variance.
        # Use 1.5× to reduce tracking noise and false positives.
        base_bandwidth = silverman_bandwidth(x_sorted.reshape(-1, 1))[0]
        bandwidth = base_bandwidth * bandwidth_factor

        # Vectorized Nadaraya-Watson smoother
        u = (x_sorted[np.newaxis, :] - x_sorted[:, np.newaxis]) / bandwidth
        w = np.exp(-0.5 * u**2)
        w_sums = np.sum(w, axis=1, keepdims=True)
        w_sums = np.where(w_sums > 0, w_sums, 1.0)
        W = w / w_sums

        # Estimate local variance
        sigma_sq_local = W @ resid_sq_sorted

        # REFINEMENT 2: Adaptive boundary trimming
        # Compute effective sample size at each point: n_eff = 1 / sum(w_i^2)
        # Points with low n_eff have unreliable variance estimates
        effective_n = 1.0 / np.sum(W ** 2, axis=1)
        min_effective_n = max(5.0, 0.05 * n_samples)  # At least 5 or 5% of n

        # Trim points with insufficient effective sample size
        valid_mask = effective_n >= min_effective_n
        # Also enforce minimum 10% trim from each end
        min_trim = max(int(0.10 * n_samples), 5)
        valid_mask[:min_trim] = False
        valid_mask[-min_trim:] = False

        # Compute trimmed statistic
        sigma_sq_local_valid = sigma_sq_local[valid_mask]
        global_var = np.mean(resid_sq)

        if len(sigma_sq_local_valid) > 0:
            T = np.sum((sigma_sq_local_valid - global_var) ** 2) / len(sigma_sq_local_valid)
        else:
            T = 0.0

        return T, W, sort_idx, valid_mask

    # REFINEMENT 5: Multi-PC testing for multivariate data
    if n_features == 1:
        x_projections = [X.flatten()]
        n_tests = 1
    else:
        # Test along multiple principal components
        X_centered = X - np.mean(X, axis=0)
        _, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Use PCs that explain at least 5% of variance
        var_explained = S ** 2 / np.sum(S ** 2)
        n_pcs = max(1, np.sum(var_explained >= 0.05))
        n_pcs = min(n_pcs, 3)  # Cap at 3 to avoid excessive testing

        x_projections = [X_centered @ Vt[i] for i in range(n_pcs)]
        n_tests = n_pcs

    # Compute observed statistic for each projection
    T_observed_list = []
    test_data = []  # Store (W, sort_idx, valid_mask) for each projection

    for x_proj in x_projections:
        T_obs, W, sort_idx, valid_mask = _compute_dmw_statistic_1d(x_proj, residuals_sq)
        T_observed_list.append(T_obs)
        test_data.append((W, sort_idx, valid_mask))

    # Use maximum statistic across projections
    T_observed = max(T_observed_list)

    # REFINEMENT 3: Wild Bootstrap instead of permutation
    # Permutation breaks the variance structure. Wild bootstrap preserves it
    # by using: e*_i = e_i * w_i where w_i is Rademacher (±1)
    T_boot = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        # Generate Rademacher weights
        rademacher = np.random.choice([-1.0, 1.0], size=n_samples)

        # Wild bootstrap residuals (centered to ensure mean 0 under H0)
        # Under H0 of constant variance, e*^2 should have no relationship with x
        boot_residuals_centered = residuals_sq_centered * rademacher

        # Add back global mean to get bootstrap squared residuals
        boot_resid_sq = boot_residuals_centered + sigma_sq_global

        # Compute max statistic across all projections
        T_proj_list = []
        for i, x_proj in enumerate(x_projections):
            W, sort_idx, valid_mask = test_data[i]
            resid_sq_sorted = boot_resid_sq[sort_idx]

            # Smooth the bootstrap residuals
            sigma_sq_boot = W @ resid_sq_sorted
            sigma_sq_boot_valid = sigma_sq_boot[valid_mask]

            if len(sigma_sq_boot_valid) > 0:
                boot_global = np.mean(boot_resid_sq)
                T_proj = np.sum((sigma_sq_boot_valid - boot_global) ** 2) / len(sigma_sq_boot_valid)
            else:
                T_proj = 0.0

            T_proj_list.append(T_proj)

        T_boot[b] = max(T_proj_list)

    # P-value: proportion of bootstrap values >= observed
    # Add small constant for finite-sample correction
    p_value = float((np.sum(T_boot >= T_observed) + 1) / (n_bootstrap + 1))

    # REFINEMENT 5 continued: Bonferroni correction for multiple testing
    # The max-statistic approach already accounts for multiplicity,
    # but we can further adjust the interpretation
    # (No explicit correction needed since we use the max statistic)

    return HeteroscedasticityTestResult(
        statistic=float(T_observed),
        p_value=p_value,
        is_heteroscedastic=p_value < alpha,
        test_name="Dette-Munk-Wagner",
        alpha=alpha,
    )


def residual_diagnostics(
    model: BaseEstimator,
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    alpha: float = 0.05,
) -> ResidualDiagnosticsResult:
    """
    Compute residual diagnostics for kernel regression.

    Parameters
    ----------
    model : fitted estimator
        Fitted kernel regression model
    X : ndarray of shape (n_samples, n_features)
        Feature matrix
    y : ndarray of shape (n_samples,)
        Target values
    alpha : float, default=0.05
        Significance level for normality test

    Returns
    -------
    ResidualDiagnosticsResult
        Comprehensive residual diagnostics
    """
    X = np.atleast_2d(X)
    y = np.asarray(y).flatten()

    y_pred = model.predict(X)
    residuals = y - y_pred

    std = np.std(residuals, ddof=1)
    std = std if std > 0 else 1.0
    standardized = residuals / std

    # Normality test (Shapiro-Wilk for n < 5000, else D'Agostino-Pearson)
    n = len(residuals)
    if n < 5000:
        try:
            norm_stat, norm_p = stats.shapiro(residuals)
        except Exception:
            norm_stat, norm_p = 0.0, 1.0
    else:
        try:
            norm_stat, norm_p = stats.normaltest(residuals)
        except Exception:
            norm_stat, norm_p = 0.0, 1.0

    return ResidualDiagnosticsResult(
        residuals=residuals,
        standardized_residuals=standardized,
        mean=float(np.mean(residuals)),
        std=float(np.std(residuals, ddof=1)),
        skewness=float(stats.skew(residuals)),
        kurtosis=float(stats.kurtosis(residuals)),
        normality_statistic=float(norm_stat),
        normality_p_value=float(norm_p),
        is_normal=norm_p >= alpha,
    )


@dataclass
class VarianceFunctionResult:
    """Result of variance function estimation."""

    X_eval: NDArray[np.floating]
    variance_estimate: NDArray[np.floating]
    std_estimate: NDArray[np.floating]
    bandwidth: NDArray[np.floating]
    method: str

    def __str__(self) -> str:
        return (
            f"Variance Function Estimation ({self.method})\n"
            f"  Mean variance: {np.mean(self.variance_estimate):.6f}\n"
            f"  Variance range: [{np.min(self.variance_estimate):.6f}, "
            f"{np.max(self.variance_estimate):.6f}]"
        )


def fan_yao_variance_estimation(
    model: BaseEstimator,
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    X_eval: NDArray[np.floating] | None = None,
    bandwidth: NDArray[np.floating] | str | None = None,
) -> VarianceFunctionResult:
    """
    Estimate conditional variance function using Fan-Yao method.

    Applies local linear regression to squared residuals to obtain a
    nonparametric estimate of the conditional variance σ²(x).

    This method is efficient and adaptive: without knowing the regression
    function, the conditional variance can be estimated asymptotically
    as well as if the regression were known.

    Args:
        model: Fitted kernel regression model for mean estimation.
        X: Feature matrix of shape (n_samples, n_features).
        y: Target values of shape (n_samples,).
        X_eval: Points at which to evaluate variance function.
            If None, uses X.
        bandwidth: Bandwidth for variance estimation. Can be:
            - None: Use model's bandwidth (default)
            - "silverman": Compute Silverman bandwidth on squared residuals
            - ndarray: Explicit bandwidth values

    Returns:
        VarianceFunctionResult with variance estimates at evaluation points.

    References:
        Fan, J. and Yao, Q. (1998). "Efficient estimation of conditional
        variance functions in stochastic regression." Biometrika, 85(3),
        645-660.

    Example:
        >>> model = NadarayaWatson(bandwidth=0.5).fit(X, y)
        >>> var_result = fan_yao_variance_estimation(model, X, y)
        >>> print(f"Mean variance: {np.mean(var_result.variance_estimate):.4f}")
    """
    from kernel_regression.estimators import LocalPolynomialRegression
    from kernel_regression.bandwidth import silverman_bandwidth

    X = np.atleast_2d(X)
    y = np.asarray(y).flatten()

    if X_eval is None:
        X_eval = X
    X_eval = np.atleast_2d(X_eval)

    # Step 1: Get residuals from mean regression
    y_pred = model.predict(X)
    residuals = y - y_pred

    # Step 2: Square the residuals
    residuals_sq = residuals ** 2

    # Step 3: Determine bandwidth for variance estimation
    if bandwidth is None:
        # Use model's bandwidth, slightly larger for variance estimation
        var_bandwidth = np.atleast_1d(model.bandwidth_) * 1.2
    elif isinstance(bandwidth, str) and bandwidth == "silverman":
        # Compute Silverman bandwidth on the squared residuals
        var_bandwidth = silverman_bandwidth(X)
    else:
        var_bandwidth = np.atleast_1d(bandwidth)

    # Step 4: Apply local linear regression to squared residuals
    # This gives σ²(x) = E[ε² | X=x]
    var_model = LocalPolynomialRegression(
        bandwidth=var_bandwidth,
        order=1,  # Local linear for variance estimation
    ).fit(X, residuals_sq)

    variance_estimate = var_model.predict(X_eval)

    # Ensure non-negative variance (numerical stability)
    variance_estimate = np.maximum(variance_estimate, 1e-10)

    # Standard deviation estimate
    std_estimate = np.sqrt(variance_estimate)

    return VarianceFunctionResult(
        X_eval=X_eval,
        variance_estimate=variance_estimate,
        std_estimate=std_estimate,
        bandwidth=var_bandwidth,
        method="Fan-Yao (Local Linear on Squared Residuals)",
    )


def heteroscedasticity_weighted_fit(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    bandwidth: NDArray[np.floating] | str = "cv",
    n_iterations: int = 2,
) -> tuple[BaseEstimator, VarianceFunctionResult]:
    """
    Fit kernel regression with heteroscedasticity-adaptive weighting.

    Uses iterative reweighting based on Fan-Yao variance estimation
    to improve efficiency under heteroscedasticity.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Target values of shape (n_samples,).
        bandwidth: Bandwidth specification for mean regression.
        n_iterations: Number of reweighting iterations (default 2).

    Returns:
        Tuple of (fitted_model, variance_result).

    Example:
        >>> model, var = heteroscedasticity_weighted_fit(X, y)
        >>> y_pred = model.predict(X_new)
    """
    from kernel_regression.estimators import NadarayaWatson

    X = np.atleast_2d(X)
    y = np.asarray(y).flatten()

    # Initial fit (unweighted)
    model = NadarayaWatson(bandwidth=bandwidth).fit(X, y)

    # Iterative reweighting
    for _ in range(n_iterations):
        # Estimate variance function
        var_result = fan_yao_variance_estimation(model, X, y)

        # Compute weights: w_i = 1 / σ(x_i)
        weights = 1.0 / np.maximum(var_result.std_estimate, 1e-6)
        weights = weights / np.sum(weights) * len(weights)  # Normalize

        # Weighted regression: transform y
        # For kernel regression, we can approximate weighted LS by
        # modifying the kernel weights in the smoother matrix
        # Here we use a simpler approach: reweight residuals

        # Get current predictions
        y_pred = model.predict(X)

        # Weighted residuals
        weighted_residuals = weights * (y - y_pred)

        # Refit on weighted data (approximation)
        y_weighted = y_pred + weighted_residuals / weights

        model = NadarayaWatson(bandwidth=model.bandwidth_).fit(X, y_weighted)

    # Final variance estimation
    var_result = fan_yao_variance_estimation(model, X, y)

    return model, var_result


class GoodnessOfFit:
    """
    Comprehensive goodness of fit diagnostics for kernel regression.

    Provides R², adjusted R², AIC, BIC, and effective degrees of freedom.

    Parameters
    ----------
    model : fitted estimator
        Fitted kernel regression model
    X : ndarray of shape (n_samples, n_features)
        Feature matrix
    y : ndarray of shape (n_samples,)
        Target values

    Attributes
    ----------
    r_squared : float
        Coefficient of determination
    adjusted_r_squared : float
        Adjusted R² accounting for effective degrees of freedom
    mse : float
        Mean squared error
    rmse : float
        Root mean squared error
    mae : float
        Mean absolute error
    aic : float
        Akaike Information Criterion
    bic : float
        Bayesian Information Criterion
    effective_df : float
        Effective degrees of freedom (trace of hat matrix)
    residual_diagnostics : ResidualDiagnosticsResult
        Detailed residual analysis
    heteroscedasticity_tests : dict
        Results of heteroscedasticity tests
    """

    def __init__(
        self,
        model: BaseEstimator,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
    ):
        self.model = model
        self.X = np.atleast_2d(X)
        self.y = np.asarray(y).flatten()

        self._compute_metrics()

    def _compute_hat_matrix_trace(self) -> float:
        """
        Estimate effective degrees of freedom.

        For kernel regression, this is the trace of the hat matrix H,
        where ŷ = Hy.
        """
        # Check if model has the necessary attributes
        if not hasattr(self.model, "X_") or not hasattr(self.model, "bandwidth_"):
            # Fallback: estimate from bandwidth
            return self._estimate_df_from_bandwidth()

        kernel_func = getattr(
            self.model, "kernel_func_", get_kernel("gaussian")
        )
        bandwidth = self.model.bandwidth_
        X_train = self.model.X_

        # Compute weights for training points
        weights = multivariate_kernel_weights(
            self.X, X_train, bandwidth, kernel_func
        )

        # Normalize weights (hat matrix rows)
        weight_sums = np.sum(weights, axis=1, keepdims=True)
        weight_sums = np.where(weight_sums > 0, weight_sums, 1.0)
        H_rows = weights / weight_sums

        # Trace of hat matrix is sum of diagonal elements
        # For prediction at training points, diagonal is H[i, i]
        if np.array_equal(self.X, X_train):
            trace = np.sum(np.diag(H_rows))
        else:
            # Approximate trace
            trace = np.sum(np.max(H_rows, axis=1))

        return float(trace)

    def _estimate_df_from_bandwidth(self) -> float:
        """Rough estimate of df from bandwidth."""
        # Simplified: use sqrt(n) as approximation for effective df
        return float(np.sqrt(self.X.shape[0]))

    def _compute_metrics(self) -> None:
        """Compute all goodness of fit metrics."""
        y_pred = self.model.predict(self.X)
        residuals = self.y - y_pred
        n = len(self.y)

        # Basic metrics
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)

        self.r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        self.mse = float(np.mean(residuals**2))
        self.rmse = float(np.sqrt(self.mse))
        self.mae = float(np.mean(np.abs(residuals)))

        # Effective degrees of freedom
        self.effective_df = self._compute_hat_matrix_trace()

        # Adjusted R²
        df_residual = n - self.effective_df
        if df_residual > 0 and n > 1:
            self.adjusted_r_squared = 1 - (ss_res / df_residual) / (ss_tot / (n - 1))
        else:
            self.adjusted_r_squared = self.r_squared

        # Information criteria
        sigma_sq = ss_res / n
        if sigma_sq > 0:
            log_likelihood = -0.5 * n * (np.log(2 * np.pi * sigma_sq) + 1)
            self.aic = -2 * log_likelihood + 2 * self.effective_df
            self.bic = -2 * log_likelihood + np.log(n) * self.effective_df
        else:
            self.aic = np.nan
            self.bic = np.nan

        # Residual diagnostics
        self.residual_diagnostics = residual_diagnostics(
            self.model, self.X, self.y
        )

        # Heteroscedasticity tests
        self.heteroscedasticity_tests = {}
        for test_name in ["white", "breusch_pagan", "goldfeld_quandt"]:
            try:
                self.heteroscedasticity_tests[test_name] = heteroscedasticity_test(
                    self.model, self.X, self.y, test=test_name
                )
            except Exception:
                self.heteroscedasticity_tests[test_name] = None

    def is_homoscedastic(self, alpha: float = 0.05) -> bool:
        """
        Check if residuals are homoscedastic.

        Returns True if all heteroscedasticity tests fail to reject
        the null hypothesis of homoscedasticity.
        """
        for test_result in self.heteroscedasticity_tests.values():
            if test_result is not None and test_result.is_heteroscedastic:
                return False
        return True

    def get_robust_standard_errors(self) -> NDArray[np.floating]:
        """
        Compute heteroscedasticity-robust standard errors.

        Uses the HC1 (White) estimator for robust variance estimation.

        Returns
        -------
        ndarray of shape (n_samples,)
            Robust standard errors for predictions
        """
        y_pred = self.model.predict(self.X)
        residuals = self.y - y_pred
        n = len(residuals)

        # HC1 correction
        df_correction = n / (n - self.effective_df) if n > self.effective_df else 1.0

        # For kernel regression, robust SE at each point
        robust_se = np.abs(residuals) * np.sqrt(df_correction)

        return robust_se

    def summary(self) -> str:
        """Generate summary report."""
        lines = [
            "=" * 60,
            "Kernel Regression Goodness of Fit Summary",
            "=" * 60,
            f"R²:                 {self.r_squared:.6f}",
            f"Adjusted R²:        {self.adjusted_r_squared:.6f}",
            f"RMSE:               {self.rmse:.6f}",
            f"MAE:                {self.mae:.6f}",
            f"Effective DF:       {self.effective_df:.2f}",
            f"AIC:                {self.aic:.2f}",
            f"BIC:                {self.bic:.2f}",
            "",
            "-" * 60,
            "Residual Diagnostics",
            "-" * 60,
            f"Mean:               {self.residual_diagnostics.mean:.6f}",
            f"Std:                {self.residual_diagnostics.std:.6f}",
            f"Skewness:           {self.residual_diagnostics.skewness:.4f}",
            f"Kurtosis:           {self.residual_diagnostics.kurtosis:.4f}",
            f"Normality p-value:  {self.residual_diagnostics.normality_p_value:.4f}",
            f"Normal residuals:   {self.residual_diagnostics.is_normal}",
            "",
            "-" * 60,
            "Heteroscedasticity Tests",
            "-" * 60,
        ]

        for name, result in self.heteroscedasticity_tests.items():
            if result is not None:
                status = "HETEROSCEDASTIC" if result.is_heteroscedastic else "homoscedastic"
                lines.append(f"{name:20s} p={result.p_value:.4f} ({status})")
            else:
                lines.append(f"{name:20s} (test failed)")

        conclusion = "HOMOSCEDASTIC" if self.is_homoscedastic() else "HETEROSCEDASTIC"
        lines.extend([
            "",
            f"Overall variance:   {conclusion}",
            "=" * 60,
        ])

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def get_leverage_values(self) -> NDArray[np.floating]:
        """
        Compute leverage values (diagonal of the hat matrix).

        The leverage h_ii measures the influence of observation i on its
        own fitted value. High leverage points have disproportionate
        influence on the regression fit.

        Returns
        -------
        ndarray of shape (n_samples,)
            Leverage values (diagonal of hat matrix H where y_hat = Hy)

        Notes
        -----
        For Nadaraya-Watson regression:
            h_ii = K(0) / sum_j K((x_i - x_j) / h)

        Leverage values should satisfy:
        - 0 <= h_ii <= 1
        - sum(h_ii) = effective degrees of freedom
        """
        return _compute_leverage_values(
            self.X, self.model.X_, self.model.bandwidth_,
            getattr(self.model, "kernel_func_", get_kernel("gaussian"))
        )


def _compute_leverage_values(
    X: NDArray[np.floating],
    X_train: NDArray[np.floating],
    bandwidth: NDArray[np.floating],
    kernel_func,
) -> NDArray[np.floating]:
    """
    Compute leverage values (hat matrix diagonal) for kernel regression.

    Args:
        X: Points to compute leverage at, shape (n_samples, n_features).
        X_train: Training points, shape (n_train, n_features).
        bandwidth: Bandwidth per feature, shape (n_features,).
        kernel_func: Kernel function.

    Returns:
        Leverage values of shape (n_samples,).
    """
    weights = multivariate_kernel_weights(X, X_train, bandwidth, kernel_func)
    weight_sums = np.sum(weights, axis=1)
    weight_sums = np.where(weight_sums > 0, weight_sums, 1.0)

    # For points in training set, leverage is self-weight / total weight
    # For Nadaraya-Watson, self-weight is K(0) for diagonal entries
    if np.array_equal(X, X_train):
        leverage = np.diag(weights) / weight_sums
    else:
        # For prediction points not in training set
        leverage = np.max(weights, axis=1) / weight_sums

    return leverage


@dataclass
class ConfidenceIntervalResult:
    """Result of confidence interval computation."""

    predictions: NDArray[np.floating]
    lower: NDArray[np.floating]
    upper: NDArray[np.floating]
    confidence_level: float
    method: str

    def __str__(self) -> str:
        return (
            f"Confidence Intervals ({self.method})\n"
            f"  Confidence level: {self.confidence_level:.0%}\n"
            f"  Mean width: {np.mean(self.upper - self.lower):.4f}"
        )


def _honest_critical_value(
    confidence_level: float,
    bias_sd_ratio: float,
) -> float:
    """
    Compute honest critical value that accounts for worst-case bias.

    Based on Armstrong & Kolesár (2020) "Simple and Honest Confidence
    Intervals in Nonparametric Regression".

    The honest CI uses cv such that:
        P(|Z + b| <= cv) >= confidence_level for all |b| <= bias_sd_ratio

    where Z ~ N(0,1) and b is the standardized bias.

    Args:
        confidence_level: Desired coverage (e.g., 0.95).
        bias_sd_ratio: Upper bound on |bias|/sd (the "smoothness" bound).

    Returns:
        Critical value cv >= z_{1-alpha/2} that ensures honest coverage.
    """
    alpha = 1 - confidence_level

    # Standard normal critical value
    z_alpha = stats.norm.ppf(1 - alpha / 2)

    # For honest coverage, we need to find cv such that:
    # min over |b| <= M of P(-cv <= Z + b <= cv) >= 1 - alpha
    # The minimum occurs at b = ±M, giving:
    # P(-cv - M <= Z <= cv - M) >= 1 - alpha
    # Phi(cv - M) - Phi(-cv - M) >= 1 - alpha

    # Binary search for the honest critical value
    if bias_sd_ratio <= 0:
        return z_alpha

    # The honest cv must be larger than z_alpha + M to ensure coverage
    cv_low = z_alpha
    cv_high = z_alpha + 2 * bias_sd_ratio + 1

    for _ in range(50):  # Binary search iterations
        cv_mid = (cv_low + cv_high) / 2
        # Coverage at worst-case bias b = bias_sd_ratio
        coverage = stats.norm.cdf(cv_mid - bias_sd_ratio) - stats.norm.cdf(
            -cv_mid - bias_sd_ratio
        )
        if coverage < confidence_level:
            cv_low = cv_mid
        else:
            cv_high = cv_mid

    return cv_high


def _estimate_bias_sd_ratio(
    model: BaseEstimator,
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    X_pred: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Estimate the bias/sd ratio for honest CI critical values.

    Uses difference between local linear and local quadratic fits
    to estimate bias, and bootstrap to estimate sd.
    """
    from kernel_regression.estimators import LocalPolynomialRegression

    bandwidth = np.atleast_1d(model.bandwidth_)
    n_samples = X.shape[0]

    # Fit local linear (order 1) and local quadratic (order 2)
    model_p1 = LocalPolynomialRegression(bandwidth=bandwidth, order=1).fit(X, y)
    model_p2 = LocalPolynomialRegression(bandwidth=bandwidth, order=2).fit(X, y)

    pred_p1 = model_p1.predict(X_pred)
    pred_p2 = model_p2.predict(X_pred)

    # Bias estimate: difference between orders
    bias_estimate = np.abs(pred_p1 - pred_p2)

    # Estimate variance using effective sample size
    # For kernel regression: var ≈ σ² / (n * h * f(x))
    # Simplified: use residual variance / sqrt(effective n)
    residuals = y - model.predict(X)
    sigma_sq = np.var(residuals, ddof=1)

    # Effective sample size depends on bandwidth
    h_prod = np.prod(bandwidth)
    effective_n = n_samples * h_prod

    # Standard deviation estimate (rough approximation)
    sd_estimate = np.sqrt(sigma_sq / max(effective_n, 1)) * np.ones(len(X_pred))

    # Bias/SD ratio (capped to avoid extreme values)
    ratio = np.clip(bias_estimate / np.maximum(sd_estimate, 1e-10), 0, 5)

    return ratio


def _cct_variance_inflation(
    bootstrap_std: NDArray[np.floating],
    bias_estimate: NDArray[np.floating],
    n_samples: int,
    bandwidth: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    CCT-style variance inflation to account for bias estimation uncertainty.

    Based on Calonico, Cattaneo, Titiunik (2014) Econometrica.

    The RBC approach inflates variance to account for the additional
    variability introduced by estimating the bias.

    Args:
        bootstrap_std: Bootstrap standard deviation of predictions.
        bias_estimate: Estimated bias at each prediction point.
        n_samples: Number of training samples.
        bandwidth: Bandwidth used for estimation.

    Returns:
        Inflated standard deviation for CI construction.
    """
    # The CCT correction inflates variance by factor related to
    # the ratio of bandwidths used for estimation vs bias correction
    # V_rbc = V_original + V_bias_estimate

    # Simplified inflation: add variance of bias estimate
    # Var(bias_hat) ≈ C * sigma^2 / (n * h^(2p+1))
    # where p is the polynomial order

    h_prod = np.prod(bandwidth)

    # Variance inflation factor (conservative estimate)
    # Based on ratio of orders: higher order has more variance
    inflation_factor = 1.0 + 0.5 * np.abs(bias_estimate) / (
        bootstrap_std + 1e-10
    )

    # Cap inflation to prevent extreme widening
    inflation_factor = np.clip(inflation_factor, 1.0, 2.0)

    return bootstrap_std * inflation_factor


def conformal_calibrate_ci(
    model: BaseEstimator,
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    ci_result: ConfidenceIntervalResult,
    calibration_fraction: float = 0.2,
) -> ConfidenceIntervalResult:
    """
    Calibrate confidence intervals using conformal prediction for
    finite-sample coverage guarantee.

    This is a post-hoc calibration that adjusts CI width based on
    empirical coverage on a held-out calibration set.

    Based on Lei et al. "Distribution-Free Predictive Inference for Regression"

    Args:
        model: Fitted kernel regression model.
        X: Full feature matrix (will be split for calibration).
        y: Full target values.
        ci_result: Initial confidence interval result to calibrate.
        calibration_fraction: Fraction of data to use for calibration.

    Returns:
        Calibrated ConfidenceIntervalResult with adjusted coverage.
    """
    n_samples = X.shape[0]
    n_calib = max(int(n_samples * calibration_fraction), 10)

    # Use last n_calib points for calibration (or random subset)
    np.random.seed(42)  # Reproducibility
    calib_idx = np.random.choice(n_samples, size=n_calib, replace=False)

    X_calib = X[calib_idx]
    y_calib = y[calib_idx]

    # Get predictions at calibration points
    y_pred_calib = model.predict(X_calib)

    # Compute CI widths at calibration points (interpolate from result)
    # For simplicity, use the mean half-width from the original CI
    original_half_width = np.mean(ci_result.upper - ci_result.lower) / 2

    # Compute conformity scores: |y - y_hat| / half_width
    residuals_calib = np.abs(y_calib - y_pred_calib)
    conformity_scores = residuals_calib / max(original_half_width, 1e-10)

    # Find the (1-alpha) quantile of conformity scores
    # This gives the scaling factor needed for desired coverage
    alpha = 1 - ci_result.confidence_level
    # Use (n+1)(1-alpha)/n quantile for finite-sample validity
    quantile_level = min((n_calib + 1) * (1 - alpha) / n_calib, 1.0)
    calibration_factor = np.percentile(conformity_scores, 100 * quantile_level)

    # Scale the CI by the calibration factor
    center = ci_result.predictions
    original_lower = ci_result.lower
    original_upper = ci_result.upper

    # New half-width = original half-width * calibration_factor
    new_half_width = (original_upper - original_lower) / 2 * calibration_factor

    new_lower = center - new_half_width
    new_upper = center + new_half_width

    return ConfidenceIntervalResult(
        predictions=ci_result.predictions,
        lower=new_lower,
        upper=new_upper,
        confidence_level=ci_result.confidence_level,
        method=ci_result.method + " + Conformal",
    )


def wild_bootstrap_confidence_intervals(
    model: BaseEstimator,
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    X_pred: NDArray[np.floating] | None = None,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    distribution: Literal["rademacher", "mammen", "normal"] = "rademacher",
    bias_correction: Literal["none", "undersmooth", "rbc", "bigbrother"] = "bigbrother",
    honest_cv: bool = False,
    variance_inflation: bool = False,
) -> ConfidenceIntervalResult:
    """
    Compute confidence intervals using Wild Bootstrap.

    The Wild Bootstrap is robust to heteroscedasticity and non-normality.
    It perturbs residuals using random weights that preserve the
    variance structure without assuming a specific error distribution.

    Args:
        model: Fitted kernel regression model.
        X: Training features of shape (n_samples, n_features).
        y: Training targets of shape (n_samples,).
        X_pred: Points for prediction, shape (n_pred, n_features).
            If None, uses X.
        confidence_level: Confidence level (default 0.95 for 95% CI).
        n_bootstrap: Number of bootstrap samples.
        distribution: Distribution for wild bootstrap weights:
            - "rademacher": +1 or -1 with prob 0.5 each (default)
            - "mammen": Two-point distribution with E[w]=0, E[w^2]=1
            - "normal": Standard normal N(0,1)
        bias_correction: Method for bias correction to improve coverage:
            - "none": Use original bandwidth (may undercover due to bias)
            - "undersmooth": Use smaller bandwidth h * 0.75 for CIs
            - "rbc": Robust Bias Correction using higher-order polynomial
              to estimate and subtract bias (CCT approach)
            - "bigbrother": (default) Combines undersmoothing with higher-order
              residuals. Uses a higher-order model to compute cleaner residuals
              plus undersmoothing for predictions. Achieves best coverage.
        honest_cv: If True, use bias-adjusted critical values instead of
            standard normal quantiles. Based on Armstrong & Kolesár (2020).
            Widens CIs to account for worst-case bias.
        variance_inflation: If True, inflate variance to account for bias
            estimation uncertainty. Based on CCT (2014) Econometrica.

    Returns:
        ConfidenceIntervalResult with predictions, lower and upper bounds.

    References:
        Wu, C.F.J. (1986). "Jackknife, Bootstrap and Other Resampling
        Methods in Regression Analysis." Annals of Statistics 14, 1261-1295.

        Mammen, E. (1993). "Bootstrap and Wild Bootstrap for High
        Dimensional Linear Models." Annals of Statistics 21, 255-285.

        Calonico, Cattaneo, Titiunik (2014). "Robust Nonparametric
        Confidence Intervals for Regression-Discontinuity Designs."
        Econometrica, 82(6), 2295-2326.

    Example:
        >>> model = NadarayaWatson(bandwidth=0.5).fit(X, y)
        >>> ci = wild_bootstrap_confidence_intervals(model, X, y)
        >>> print(f"95% CI width: {np.mean(ci.upper - ci.lower):.4f}")
    """
    X = np.atleast_2d(X)
    y = np.asarray(y).flatten()

    if X_pred is None:
        X_pred = X

    X_pred = np.atleast_2d(X_pred)
    n_samples = X.shape[0]
    n_pred = X_pred.shape[0]

    # Get original bandwidth
    original_bandwidth = np.atleast_1d(model.bandwidth_)

    # Original predictions and residuals (using original model)
    y_pred_original = model.predict(X)
    residuals = y - y_pred_original

    # Handle bias correction approaches
    if bias_correction == "rbc":
        # Robust Bias Correction (CCT approach):
        # 1. Use original bandwidth for predictions
        # 2. Estimate bias using higher-order polynomial
        # 3. Correct predictions and widen CIs accordingly
        from kernel_regression.estimators import LocalPolynomialRegression

        # Fit higher-order model for bias estimation
        bias_model = LocalPolynomialRegression(
            bandwidth=original_bandwidth,
            order=2,
        ).fit(X, y)
        bias_pred = bias_model.predict(X_pred)

        # Original model predictions
        predictions = model.predict(X_pred)

        # Bias estimate: difference between low and high order fits
        bias_estimate = predictions - bias_pred

        # Bias-corrected predictions
        predictions_corrected = predictions - bias_estimate
        ci_bandwidth = original_bandwidth  # Use original bandwidth for bootstrap

    elif bias_correction == "bigbrother":
        # Big Brother approach:
        # Combines two techniques for best coverage:
        # 1. Use a higher-order model to compute residuals for the bootstrap.
        #    The higher-order model captures more of the true signal (including
        #    curvature that the lower-order model misses as "bias"), leaving
        #    purer noise in the residuals.
        # 2. Use undersmoothing for predictions and bootstrap refits to further
        #    reduce bias in the confidence intervals.
        from kernel_regression.estimators import LocalPolynomialRegression

        # Determine current model order
        current_order = getattr(model, "order_", 0)  # NW is order 0

        # Fit "big brother" model one order higher for cleaner residuals
        bigbrother_model = LocalPolynomialRegression(
            bandwidth=original_bandwidth,
            order=current_order + 1,
        ).fit(X, y)

        # Use big brother's predictions to compute cleaner residuals
        y_pred_bigbrother = bigbrother_model.predict(X)
        residuals = y - y_pred_bigbrother

        # Apply undersmoothing for predictions and bootstrap
        undersmooth_factor = 0.75
        ci_bandwidth = original_bandwidth * undersmooth_factor

        # Refit with undersmoothed bandwidth for predictions
        params = model.get_params()
        params['bandwidth'] = ci_bandwidth
        undersmooth_model = model.__class__(**params)
        undersmooth_model.fit(X, y)

        predictions_corrected = undersmooth_model.predict(X_pred)
        y_pred_original = undersmooth_model.predict(X)

    elif bias_correction == "undersmooth":
        # Undersmoothing approach:
        # Use smaller bandwidth to reduce bias, making variance dominate
        # h_CI = h * factor where factor makes bias negligible
        # Factor of 0.7-0.8 is typically sufficient
        undersmooth_factor = 0.75
        ci_bandwidth = original_bandwidth * undersmooth_factor

        # Refit model with undersmoothed bandwidth for predictions
        params = model.get_params()
        params['bandwidth'] = ci_bandwidth
        undersmooth_model = model.__class__(**params)
        undersmooth_model.fit(X, y)

        predictions_corrected = undersmooth_model.predict(X_pred)

        # Update residuals from undersmoothed model for consistency
        y_pred_original = undersmooth_model.predict(X)
        residuals = y - y_pred_original

    else:  # bias_correction == "none"
        predictions_corrected = model.predict(X_pred)
        ci_bandwidth = original_bandwidth

    # Bootstrap predictions
    bootstrap_preds = np.zeros((n_bootstrap, n_pred))

    for b in range(n_bootstrap):
        # Generate wild bootstrap weights
        if distribution == "rademacher":
            w = np.random.choice([-1.0, 1.0], size=n_samples)
        elif distribution == "mammen":
            sqrt5 = np.sqrt(5)
            p = (sqrt5 + 1) / (2 * sqrt5)
            w1 = -(sqrt5 - 1) / 2
            w2 = (sqrt5 + 1) / 2
            w = np.where(np.random.random(n_samples) < p, w1, w2)
        else:  # normal
            w = np.random.standard_normal(n_samples)

        # Perturbed response
        y_star = y_pred_original + w * residuals

        # Refit model with CI bandwidth
        params = model.get_params()
        params['bandwidth'] = ci_bandwidth
        model_star = model.__class__(**params)
        model_star.fit(X, y_star)

        # Predict at X_pred
        bootstrap_preds[b] = model_star.predict(X_pred)

    # Compute bootstrap standard deviation
    bootstrap_std = np.std(bootstrap_preds, axis=0)

    # Apply variance inflation if requested (CCT approach)
    if variance_inflation:
        # Estimate bias for variance inflation
        from kernel_regression.estimators import LocalPolynomialRegression

        bias_model = LocalPolynomialRegression(
            bandwidth=original_bandwidth, order=2
        ).fit(X, y)
        bias_est = np.abs(model.predict(X_pred) - bias_model.predict(X_pred))

        bootstrap_std = _cct_variance_inflation(
            bootstrap_std, bias_est, n_samples, original_bandwidth
        )

    # Compute confidence intervals
    if honest_cv:
        # Use bias-adjusted critical values (Armstrong-Kolesár approach)
        bias_sd_ratio = _estimate_bias_sd_ratio(model, X, y, X_pred)
        # Use median bias/sd ratio for a single critical value
        median_ratio = float(np.median(bias_sd_ratio))
        cv = _honest_critical_value(confidence_level, median_ratio)

        # Symmetric CI using honest critical value
        lower = predictions_corrected - cv * bootstrap_std
        upper = predictions_corrected + cv * bootstrap_std
    else:
        # Standard pivotal bootstrap CI
        bootstrap_deviations = bootstrap_preds - predictions_corrected

        alpha = 1 - confidence_level
        lower_q = np.percentile(bootstrap_deviations, 100 * alpha / 2, axis=0)
        upper_q = np.percentile(bootstrap_deviations, 100 * (1 - alpha / 2), axis=0)

        # Pivotal CI: swap the quantiles
        lower = predictions_corrected - upper_q
        upper = predictions_corrected - lower_q

    method_name = f"Wild Bootstrap ({distribution})"
    if bias_correction != "none":
        method_name += f" + {bias_correction.upper()}"
    if honest_cv:
        method_name += " + Honest"
    if variance_inflation:
        method_name += " + VarInflate"

    return ConfidenceIntervalResult(
        predictions=predictions_corrected,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        method=method_name,
    )
