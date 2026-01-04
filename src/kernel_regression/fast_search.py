"""
Fast neighborhood search using KD-Trees.

This module provides efficient O(n log n) neighborhood lookups
instead of the naive O(n²) approach.
"""

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree

from kernel_regression.kernels import get_kernel


class KDTreeSearch:
    """
    Fast neighborhood search using scipy's KDTree.

    For kernel regression, we need to find all points within the
    kernel bandwidth. KDTree provides O(log n) lookup per query.

    Args:
        leaf_size: Number of points at which to switch to brute force.
            Smaller values give faster queries but slower construction.

    Attributes:
        tree_: The fitted KDTree structure.
        X_: The training data used to build the tree.
        scale_: Per-feature scaling factors for anisotropic bandwidths.

    Example:
        >>> searcher = KDTreeSearch()
        >>> searcher.fit(X_train, bandwidth)
        >>> neighbors, distances = searcher.query_radius(X_test, radius=1.0)
    """

    def __init__(self, leaf_size: int = 40):
        """Initialize the KDTree searcher.

        Args:
            leaf_size: Number of points at which to switch to brute force.
        """
        self.leaf_size = leaf_size
        self.tree_: KDTree | None = None
        self.X_: NDArray[np.floating] | None = None
        self.scale_: NDArray[np.floating] | None = None

    def fit(
        self,
        X: NDArray[np.floating],
        bandwidth: NDArray[np.floating] | None = None,
    ) -> "KDTreeSearch":
        """Build the KDTree from training data.

        Args:
            X: Training data of shape (n_samples, n_features).
            bandwidth: Per-feature bandwidth for scaling. If provided,
                data is scaled so that queries use unit bandwidth.

        Returns:
            self: The fitted searcher.
        """
        X = np.atleast_2d(X)
        self.X_ = X.copy()

        if bandwidth is not None:
            self.scale_ = np.atleast_1d(bandwidth)
            # Scale data so bandwidth becomes 1 in scaled space
            X_scaled = X / self.scale_
        else:
            self.scale_ = np.ones(X.shape[1])
            X_scaled = X

        self.tree_ = KDTree(X_scaled, leafsize=self.leaf_size)
        return self

    def query_radius(
        self,
        X: NDArray[np.floating],
        radius: float = 1.0,
    ) -> tuple[list[NDArray[np.intp]], list[NDArray[np.floating]]]:
        """Find all points within radius of query points.

        Args:
            X: Query points of shape (n_queries, n_features).
            radius: Search radius (in scaled space if bandwidth was provided).

        Returns:
            indices: List of arrays, one per query, containing neighbor indices.
            distances: List of arrays containing distances to neighbors.
        """
        if self.tree_ is None:
            raise RuntimeError("Must call fit() before query_radius()")

        X = np.atleast_2d(X)
        X_scaled = X / self.scale_

        # Query for all points within radius
        indices = self.tree_.query_ball_point(X_scaled, r=radius)

        # Compute actual distances
        distances = []
        for i, idx in enumerate(indices):
            if len(idx) > 0:
                dists = np.linalg.norm(
                    (X_scaled[i] - self.tree_.data[idx]), axis=1
                )
                distances.append(dists)
            else:
                distances.append(np.array([]))

        return indices, distances

    def query_knn(
        self,
        X: NDArray[np.floating],
        k: int,
    ) -> tuple[NDArray[np.floating], NDArray[np.intp]]:
        """Find k nearest neighbors for each query point.

        Args:
            X: Query points of shape (n_queries, n_features).
            k: Number of neighbors to find.

        Returns:
            distances: Array of shape (n_queries, k) with distances.
            indices: Array of shape (n_queries, k) with neighbor indices.
        """
        if self.tree_ is None:
            raise RuntimeError("Must call fit() before query_knn()")

        X = np.atleast_2d(X)
        X_scaled = X / self.scale_

        distances, indices = self.tree_.query(X_scaled, k=k)
        return distances, indices


def compute_kernel_weights_kdtree(
    X_query: NDArray[np.floating],
    X_train: NDArray[np.floating],
    bandwidth: NDArray[np.floating],
    kernel_func: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    tree: KDTreeSearch | None = None,
    compact_support_radius: float = 3.0,
) -> NDArray[np.floating]:
    """
    Compute kernel weights using KDTree for fast lookup.

    For kernels with compact support (epanechnikov, uniform, etc.),
    this is much faster than computing all pairwise distances.
    For Gaussian kernel, uses a cutoff radius.

    Args:
        X_query: Query points of shape (n_queries, n_features).
        X_train: Training points of shape (n_train, n_features).
        bandwidth: Bandwidth per feature of shape (n_features,).
        kernel_func: Kernel function to apply.
        tree: Pre-built KDTree searcher. If None, builds a new one.
        compact_support_radius: Radius multiplier for kernel support.
            Points beyond bandwidth * radius are assumed to have zero weight.

    Returns:
        weights: Array of shape (n_queries, n_train) with kernel weights.
    """
    X_query = np.atleast_2d(X_query)
    X_train = np.atleast_2d(X_train)
    bandwidth = np.atleast_1d(bandwidth)

    n_queries = X_query.shape[0]
    n_train = X_train.shape[0]

    # Build tree if not provided
    if tree is None:
        tree = KDTreeSearch()
        tree.fit(X_train, bandwidth)

    # Initialize sparse weights
    weights = np.zeros((n_queries, n_train))

    # Query neighbors within support radius
    indices, distances = tree.query_radius(X_query, radius=compact_support_radius)

    # Compute kernel weights only for neighbors
    for i, (idx, dist) in enumerate(zip(indices, distances)):
        if len(idx) > 0:
            idx = np.array(idx)
            # Distances are already in scaled space, apply kernel
            kernel_vals = kernel_func(dist)
            # Product kernel: multiply across dimensions
            # Note: dist is Euclidean in scaled space, need per-dim for product kernel
            diff = (X_query[i] - X_train[idx]) / bandwidth
            prod_weights = np.prod(kernel_func(diff), axis=1)
            weights[i, idx] = prod_weights / np.prod(bandwidth)

    return weights


def adaptive_bandwidth_knn(
    X: NDArray[np.floating],
    k: int = 10,
    min_bandwidth: float | None = None,
    max_bandwidth: float | None = None,
) -> NDArray[np.floating]:
    """
    Compute adaptive bandwidth based on k-nearest neighbor distances.

    The bandwidth at each point is set proportional to the distance
    to its k-th nearest neighbor, allowing the bandwidth to adapt
    to local data density.

    Args:
        X: Data points of shape (n_samples, n_features).
        k: Number of neighbors to use for bandwidth estimation.
        min_bandwidth: Minimum allowed bandwidth per feature.
        max_bandwidth: Maximum allowed bandwidth per feature.

    Returns:
        bandwidths: Array of shape (n_samples,) with per-point bandwidths.
    """
    X = np.atleast_2d(X)
    n_samples = X.shape[0]

    # Build KDTree
    tree = KDTree(X)

    # Find k+1 nearest neighbors (includes self)
    k_eff = min(k + 1, n_samples)
    distances, _ = tree.query(X, k=k_eff)

    # Use distance to k-th neighbor as bandwidth
    # (column 0 is distance to self = 0)
    if k_eff > 1:
        bandwidths = distances[:, -1]  # k-th neighbor distance
    else:
        bandwidths = np.ones(n_samples)

    # Apply min/max constraints
    if min_bandwidth is not None:
        bandwidths = np.maximum(bandwidths, min_bandwidth)
    if max_bandwidth is not None:
        bandwidths = np.minimum(bandwidths, max_bandwidth)

    # Avoid zero bandwidth
    bandwidths = np.maximum(bandwidths, 1e-10)

    return bandwidths
