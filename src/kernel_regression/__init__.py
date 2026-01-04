"""
Kernel Regression Package

Multivariate Nadaraya-Watson and Local Polynomial Kernel Regression
with sklearn-compatible interface.
"""

from kernel_regression.bandwidth import (
    CrossValidatedBandwidth,
    RuleOfThumbBandwidth,
    silverman_bandwidth,
)
from kernel_regression.diagnostics import (
    GoodnessOfFit,
    heteroscedasticity_test,
    residual_diagnostics,
)
from kernel_regression.estimators import (
    KernelRegression,
    LocalPolynomialRegression,
    NadarayaWatson,
)
from kernel_regression.kernels import (
    epanechnikov_kernel,
    gaussian_kernel,
    tricube_kernel,
    uniform_kernel,
)

__version__ = "0.1.0"

__all__ = [
    # Estimators
    "KernelRegression",
    "NadarayaWatson",
    "LocalPolynomialRegression",
    # Kernels
    "gaussian_kernel",
    "epanechnikov_kernel",
    "uniform_kernel",
    "tricube_kernel",
    # Bandwidth
    "CrossValidatedBandwidth",
    "RuleOfThumbBandwidth",
    "silverman_bandwidth",
    # Diagnostics
    "GoodnessOfFit",
    "heteroscedasticity_test",
    "residual_diagnostics",
]
