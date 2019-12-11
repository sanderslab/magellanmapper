#!/bin/bash
# Cluster measurement.
# Author: David Young, 2019
"""Clustering measurements."""

import os
import multiprocessing as mp

import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn import neighbors

from clrbrain import config
from clrbrain import lib_clrbrain
from clrbrain import np_io
from clrbrain import ontology
from clrbrain import plot_2d
from clrbrain import profiles
from clrbrain import sitk_io
from clrbrain import stats


def knn_dist(blobs, n, max_dist=None, show=True):
    """Measure the k-nearest-neighbors distance.
    
    Args:
        blobs (:obj:`np.ndarray`): Sequence given as
            ``[n_samples, n_features]``, where features typically is of the
            form, ``[z, y, x, ...]``.
        n (int): Number of neighbors. The farthest neighbor will be used
            for sorting, filtering, and plotting.
        max_dist (int, float): Maximum distance given as a percentile to plot;
            defaults to None to show neighbors of all distances.
        show (bool): True to plot the distances; defaults to True.

    Returns:
        :obj:`neighbors.NearestNeighbors`, :obj:`np.ndarray`,
        List[:obj:`pd.DataFrame`]:
        Tuple of ``NearestNeighbors`` object, all distances from
        ``kneighbors`` sorted by the ``n``th neighbor, and a list of
        data frames at different zoom levels (``[df_overview, df_zoomed]``).

    """
    def plot(mod=""):
        # plot sorted distances as line and return data frame
        df = pd.DataFrame(
            {"point": np.arange(len(dist_disp)), "dist": dist_disp})
        plot_2d.plot_lines(
            "knn_dist{}".format(mod), "point", ("dist", ), df=df, show=show)
        return df
    
    knn = neighbors.NearestNeighbors(n, n_jobs=-1).fit(blobs)
    print(knn)
    dist, _ = knn.kneighbors(blobs)
    # sort distances based on nth neighbors
    dist = dist[np.argsort(dist[:, n - 1])]
    if max_dist:
        # remove all distances where nth neighbor is beyond threshold
        dist = dist[dist[:, n - 1] < np.percentile(dist, max_dist)]
    dfs = []
    if show or config.savefig:
        # line plot of nth neighbor distances by ascending order
        distn = dist[:, n - 1]
        step = int(len(distn) / 1000)
        dist_disp = distn[::step]  # downsample for overview plot
        dfs.append(plot())
        # zoom to >= 90th percentile
        dist_disp = distn[distn > np.percentile(distn, 90)]
        dfs.append(plot("_zoomed"))
    return knn, dist, dfs


def plot_knns(img_paths, suffix=None, show=False, names=None):
    """Plot k-nearest-neighbor distances for multiple sets of blobs,
    overlaying on a single plot.

    Args:
        img_paths (List[str]): Base paths from which registered labels and
            blobs files will be found and output blobs file save location
            will be constructed.
        suffix (str): Suffix for ``path``; defaults to None.
        show (bool): True to plot the distances; defaults to False.
        names (List[str]): Sequence of names corresponding to ``img_paths``
            for the plot legend.

    """
    cluster_settings = config.register_settings[
        profiles.RegKeys.METRICS_CLUSTER]
    knn_n = cluster_settings[profiles.RegKeys.KNN_N]
    if not knn_n:
        knn_n = cluster_settings[profiles.RegKeys.DBSCAN_MINPTS] - 1
    print("Calculating k-nearest-neighbor distances and plotting distances "
          "for neighbor {}".format(knn_n))
    
    # set up combined data frames for all samples at each zoom level
    df_keys = ("ov", "zoom")
    dfs_comb = {key: [] for key in df_keys}
    names_disp = names if names else []
    for i, img_path in enumerate(img_paths):
        # load blobs associated with image
        mod_path = img_path
        if suffix is not None:
            mod_path = lib_clrbrain.insert_before_ext(img_path, suffix)
        labels_img_np = sitk_io.load_registered_img(
            mod_path, config.RegNames.IMG_LABELS.value)
        blobs, scaling, res = np_io.load_blobs(img_path, labels_img_np.shape)
        if blobs is None:
            lib_clrbrain.warn("unable to load nuclei coordinates for", img_path)
            continue
        # convert to physical units and display k-nearest-neighbors for nuclei
        blobs = np.multiply(blobs[:, :3], res)
        _, _, dfs = knn_dist(blobs, knn_n, 99.9, False)
        if names is None:
            # default to naming from filename
            names_disp.append(os.path.basename(mod_path))
        for j, df in enumerate(dfs):
            dfs_comb[df_keys[j]].append(df)
    
    for key in dfs_comb:
        # combine data frames at each zoom level and plot together with
        # different colors for each image
        df = stats.join_dfs(dfs_comb[key], "point")
        dist_cols = [col for col in df.columns if col.startswith("dist")]
        rename_cols = {col: name for col, name in zip(dist_cols, names_disp)}
        df = df.rename(rename_cols, axis=1)
        plot_2d.plot_lines(
            "knn_dist_combine_{}".format(key), "point", rename_cols.values(),
            df=df, show=show)


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
                         blobs_iso_scaling, all_labels=False):
        blobs_lbls = ontology.get_label_ids_from_position(
            blobs, labels_img_np, blobs_lbl_scaling)
        blobs = np.multiply(blobs[:, :3], blobs_iso_scaling)
        blobs_clus = np.zeros((len(blobs), 5), dtype=int)
        blobs_clus[:, :3] = blobs
        blobs_clus[:, 3] = blobs_lbls
        cls.blobs = blobs_clus
        print(np.unique(blobs_clus[:, 3]))
        print(cls.blobs)

        # TODO: shift to separate func once load blobs without req labels img

        label_ids = np.unique(labels_img_np)
        cluster_settings = config.register_settings[
            profiles.RegKeys.METRICS_CLUSTER]
        eps = cluster_settings[profiles.RegKeys.DBSCAN_EPS]
        minpts = cluster_settings[profiles.RegKeys.DBSCAN_MINPTS]
        
        if all_labels:
            # cluster all labels together
            # TODO: n_jobs appears to be ignored despite reported fixes
            _, labels = cls.cluster_within_label(None, eps, minpts, -1)
            cls.blobs[:, 4] = labels
        else:
            # cluster by individual label
            pool = mp.Pool()
            pool_results = []
            for label_id in label_ids:
                # add rotation argument if necessary
                pool_results.append(
                    pool.apply_async(
                        cls.cluster_within_label,
                        args=(label_id, eps, minpts, None)))
    
            for result in pool_results:
                label_id, labels = result.get()
                if labels is not None:
                    cls.blobs[cls.blobs[:, 3] == label_id, 4] = labels
            pool.close()
            pool.join()
        cls.blobs[:, :3] = np.divide(blobs[:, :3], blobs_iso_scaling)
        
        return cls.blobs
    
    @classmethod
    def cluster_within_label(cls, label_id, eps, minpts, n_jobs):
        blobs = cls.blobs
        if label_id is not None:
            blobs = blobs[blobs[:, 3] == label_id]
        clus_lbls = None
        if len(blobs) > 0:
            clusters = cluster.DBSCAN(
                eps=eps, min_samples=minpts, leaf_size=30,
                n_jobs=n_jobs).fit(blobs)
            num_clusters, num_noise, num_largest = cluster_dbscan_metrics(
                clusters.labels_)
            print("label {}: num clusters: {}, noise blobs: {}, "
                  "largest cluster: {}"
                  .format(label_id, num_clusters, num_noise, num_largest))
            clus_lbls = clusters.labels_
        return label_id, clus_lbls


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
    blobs, scaling, res = np_io.load_blobs(img_path, labels_img_np.shape)
    if blobs is None:
        lib_clrbrain.warn("unable to load nuclei coordinates")
        return
    
    # append label IDs to blobs and scale to make isotropic
    blobs = ClusterByLabel.cluster_by_label(
        blobs[:, :3], labels_img_np, scaling, res)
    print(blobs)
    out_path = lib_clrbrain.combine_paths(mod_path, config.SUFFIX_BLOB_CLUSTERS)
    np.save(out_path, blobs)
