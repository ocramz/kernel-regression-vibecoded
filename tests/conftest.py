"""Pytest fixtures for kernel regression tests."""

import numpy as np
import pytest


@pytest.fixture
def random_state():
    """Fixed random state for reproducibility."""
    return np.random.RandomState(42)


@pytest.fixture
def simple_1d_data(random_state):
    """Simple 1D regression data."""
    n = 100
    X = random_state.uniform(-3, 3, (n, 1))
    y = np.sin(X[:, 0]) + 0.1 * random_state.randn(n)
    return X, y


@pytest.fixture
def simple_2d_data(random_state):
    """Simple 2D regression data."""
    n = 150
    X = random_state.uniform(-2, 2, (n, 2))
    y = X[:, 0] ** 2 + X[:, 1] + 0.1 * random_state.randn(n)
    return X, y


@pytest.fixture
def heteroscedastic_data(random_state):
    """Data with heteroscedastic errors."""
    n = 200
    X = random_state.uniform(0, 5, (n, 1))
    # Variance increases with X
    noise = random_state.randn(n) * (0.1 + 0.3 * X[:, 0])
    y = 2 * X[:, 0] + noise
    return X, y


@pytest.fixture
def homoscedastic_data(random_state):
    """Data with homoscedastic errors."""
    n = 200
    X = random_state.uniform(0, 5, (n, 1))
    noise = 0.3 * random_state.randn(n)
    y = 2 * X[:, 0] + noise
    return X, y
