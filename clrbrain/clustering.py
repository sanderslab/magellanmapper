#!/bin/bash
# Cluster measurement.
# Author: David Young, 2019
"""Clustering measurements."""

import multiprocessing as mp

import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn import neighbors

from clrbrain import cli
from clrbrain import config
from clrbrain import importer
from clrbrain import lib_clrbrain
from clrbrain import ontology
from clrbrain import plot_2d
from clrbrain import profiles
from clrbrain import sitk_io


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
    """Wrapper for clustering by DBSCAN.
    
    Args:
        blobs (:obj:`np.ndarray`): Sequence given as
            ``[n_samples, n_features]``, where features typically is of the
            form, ``[z, y, x, ...]``.
        eps (int, float): Maximum distance between points within cluster.
        minpts (int): Minimum points/samples per cluster.

    Returns:
        :obj:`cluster.DBSCAN`: Tuple of ``DBSCAN`` cluster object.

    """
    # find clusters
    clusters = cluster.DBSCAN(
        eps=eps, min_samples=minpts, leaf_size=30).fit(blobs)
    return clusters


def cluster_dbscan_metrics(labels):
    """Calculate basic metrics for DBSCAN.
    
    Args:
        labels (:obj:`np.ndarray`): Cluster labels.

    Returns:
        int, int, int: Tuple of number of clusters, number of noise blobs,
        and number of blobs contained within the largest cluster.

    """
    lbl_unique, lbl_counts = np.unique(
        labels[labels != -1], return_counts=True)
    num_clusters = len(lbl_unique)
    # number of blobs in largest cluster
    num_largest = np.nan if len(lbl_counts) == 0 else np.amax(lbl_counts)
    # number of blobs not in a cluster
    num_noise = np.sum(labels == -1)
    return num_clusters, num_noise, num_largest


class ClusterByLabel(object):
    blobs = None
    
    @classmethod
    def cluster_by_label(cls, blobs, labels_img_np, blobs_lbl_scaling,
                         blobs_iso_scaling):
        blobs_lbls = ontology.get_label_ids_from_position(
            blobs, labels_img_np, blobs_lbl_scaling)
        blobs = np.multiply(blobs[:, :3], blobs_iso_scaling)
        blobs_clus = np.zeros((len(blobs), 5), dtype=int)
        blobs_clus[:, :3] = blobs
        blobs_clus[:, 3] = blobs_lbls
        cls.blobs = blobs_clus
        print(np.unique(blobs_clus[:, 3]))
        print(cls.blobs)
        
        label_ids = np.unique(labels_img_np)
        cluster_settings = config.register_settings[
            profiles.RegKeys.METRICS_CLUSTER]
        eps = cluster_settings[profiles.RegKeys.DBSCAN_EPS]
        minpts = cluster_settings[profiles.RegKeys.DBSCAN_MINPTS]
        
        pool = mp.Pool()
        pool_results = []
        for label_id in label_ids:
            # add rotation argument if necessary
            pool_results.append(
                pool.apply_async(
                    cls.cluster_within_label, args=(label_id, eps, minpts)))

        for result in pool_results:
            label_id, labels = result.get()
            cls.blobs[cls.blobs[:, 3] == label_id, 4] = labels
        pool.close()
        pool.join()
        cls.blobs[:, :3] = np.divide(blobs[:, :3], blobs_iso_scaling)
        
        return cls.blobs
    
    @classmethod
    def cluster_within_label(cls, label_id, eps, minpts):
        blobs = cls.blobs[cls.blobs[:, 3] == label_id]
        clusters = cluster_dbscan(blobs, eps, minpts)
        num_clusters, num_noise, num_largest = cluster_dbscan_metrics(
            clusters.labels_)
        print("label {}: num clusters: {}, noise blobs: {}, largest cluster: {}"
              .format(label_id, num_clusters, num_noise, num_largest))
        return label_id, clusters.labels_


def cluster_blobs(img_path, suffix=None):
    """Cluster blobs and save to Numpy archive.
    
    Args:
        img_path (str): Base path from which registered labels and blobs files
            will be found and output blobs file save location will be
            constructed.
        suffix (str): Suffix for ``path``; defaults to None.

    Returns:

    """
    mod_path = img_path
    if suffix is not None:
        mod_path = lib_clrbrain.insert_before_ext(img_path, suffix)
    labels_img_np = sitk_io.load_registered_img(
        mod_path, config.RegNames.IMG_LABELS.value)
    try:
        cli.setup_images(
            config.filename, proc_mode=config.ProcessTypes.LOAD.name)
    except FileNotFoundError as e:
        print(e)
    if cli.segments_proc is None:
        lib_clrbrain.warn("unable to load nuclei coordinates")
        return
    # append label IDs to blobs and scale to make isotropic
    blobs = ClusterByLabel.cluster_by_label(
        cli.segments_proc[:, :3], labels_img_np, importer.calc_scaling(
            None, labels_img_np, config.image5d_shapes[0, 1:]),
        config.resolutions[0])
    print(blobs)
    out_path = lib_clrbrain.combine_paths(mod_path, config.SUFFIX_BLOB_CLUSTERS)
    np.save(out_path, blobs)
