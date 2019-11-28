#!/bin/bash
# Cluster measurement.
# Author: David Young, 2019
"""Clustering measurements."""


import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn import neighbors

from clrbrain import plot_2d


def knn_dist(blobs, n, max_dist=None, show=True):
    """Measure the k-nearest-neighbors distance.
    
    Args:
        blobs (:obj:`np.ndarray`): Sequence given as
            ``[n_samples, n_features]``, where features typically is of the
            form, ``[z, y, x, ...]``.
        n (int): Number of neighbors. The farthest neighbor will be used
            for sorting, filtering, and plotting.
        max_dist (int): Maximum distance to plot; defaults to None to show
            neighbors of all distances.
        show (bool): True to plot the distances; defaults to True.

    Returns:
        :obj:`neighbors.NearestNeighbors`, :obj:`np.ndarray`:
        Tuple of ``NearestNeighbors`` object and all distances from
        ``kneighbors`` sorted by the ``n``th neighbor.

    """
    knn = neighbors.NearestNeighbors(n).fit(blobs)
    print(knn)
    dist, _ = knn.kneighbors(blobs)
    # sort distances based on nth neighbors
    dist = dist[np.argsort(dist[:, n - 1])]
    if max_dist:
        # remove all distances where nth neighbor is beyond threshold
        dist = dist[dist[:, n - 1] < max_dist]
    if show:
        # line plot of nth neighbor distances by ascending order
        df = pd.DataFrame(
            {"point": np.arange(len(dist)), "dist": dist[:, n - 1]})
        plot_2d.plot_lines("", "point", ("dist", ), df=df)
    return knn, dist


def cluster_dbscan(blobs, eps, minpts):
    """Cluster by DBSCAN and report cluster and noise metrics.
    
    Args:
        blobs (:obj:`np.ndarray`): Sequence given as
            ``[n_samples, n_features]``, where features typically is of the
            form, ``[z, y, x, ...]``.
        eps (int, float): Maximum distance between points within cluster.
        minpts (int): Minimum points/samples per cluster.

    Returns:
        :obj:`cluster.DBSCAN`, int, int, int: Tuple of ``DBSCAN`` cluster
        object, number of clusters, number of noise blobs, and number of
        blobs contained within the largest cluster.

    """
    # find clusters
    clusters = cluster.DBSCAN(
        eps=eps, min_samples=minpts, leaf_size=30).fit(blobs)
    print(clusters)
    
    # cluster metrics
    lbl_unique, lbl_counts = np.unique(
        clusters.labels_[clusters.labels_ != -1], return_counts=True)
    num_clusters = len(lbl_unique)
    # number of blobs in largest cluster
    num_largest = np.nan if len(lbl_counts) == 0 else np.amax(lbl_counts)
    # number of blobs not in a cluster
    num_noise = np.sum(clusters.labels_ == -1)
    
    return clusters, num_clusters, num_noise, num_largest
