#!/bin/bash
# Regional volume and density management
# Author: David Young, 2018
"""Measure volumes and densities by regions.
"""

import multiprocessing as mp
from time import time

import numpy as np
import pandas as pd
from skimage import measure

from clrbrain import config
from clrbrain import lib_clrbrain
from clrbrain import plot_3d

class LabelToEdge(object):
    """Convert a label to an edge with class methods as an encapsulated 
    way to use in multiprocessing without requirement for global variables.
    
    Attributes:
        labels_img_np: Integer labels images as a Numpy array.
    """
    labels_img_np = None
    
    @classmethod
    def set_labels_img_np(cls, val):
        """Set the labels image.
        
        Args:
            val: Labels image to set as class attribute.
        """
        cls.labels_img_np = val
    
    @classmethod
    def find_label_edge(cls, label_id):
        """Convert a label into just its border.
        
        Args:
            label_id: Integer of the label to extract from 
                :attr:``labels_img_np``.
        
        Returns:
            Tuple of the given label ID; list of slices defining the 
            location of the ROI where the edges can be found; and the 
            ROI as a volume mask defining where the edges exist.
        """
        print("getting edge for {}".format(label_id))
        slices = None
        borders = None
        
        # get mask of label to get bounding box
        label_mask = cls.labels_img_np == label_id
        props = measure.regionprops(label_mask.astype(np.int))
        if len(props) > 0 and props[0].bbox is not None:
            _, slices = plot_3d.get_bbox_region(props[0].bbox)
            
            # work on a view of the region for efficiency, obtaining borders 
            # as eroded region and writing into new array
            region = cls.labels_img_np[tuple(slices)]
            label_mask_region = region == label_id
            borders = plot_3d.perimeter_nd(label_mask_region)
        return label_id, slices, borders

def make_labels_edge(labels_img_np):
    """Convert labels image into label borders image.
    
    The atlas is assumed to be a sample (eg microscopy) image on which 
    an edge-detection filter will be applied. 
    
    Args:
        labels_img_np: Image as a Numpy array, assumed to be an 
            annotated image whose edges will be found by obtaining 
            the borders of all annotations.
    
    Returns:
        Binary image array the same shape as ``labels_img_np`` with labels 
        reduced to their corresponding borders.
    """
    start_time = time()
    labels_edge = np.zeros_like(labels_img_np)
    label_ids = np.unique(labels_img_np)
    
    # use a class to set and process the label without having to 
    # reference the labels image as a global variable
    label_to_edge = LabelToEdge()
    label_to_edge.set_labels_img_np(labels_img_np)
    
    pool = mp.Pool()
    pool_results = []
    for label_id in label_ids:
        pool_results.append(
            pool.apply_async(
                label_to_edge.find_label_edge, args=(label_id, )))
    for result in pool_results:
        label_id, slices, borders = result.get()
        if slices is not None:
            borders_region = labels_edge[tuple(slices)]
            borders_region[borders] = label_id
    pool.close()
    pool.join()
    
    print("time elapsed to make labels edge:", time() - start_time)
    
    return labels_edge

class LabelMetrics(object):
    """Meausure metrics within image labels in a way that allows 
    multiprocessing without global variables.
    
    All images should be of the same shape.
    
    Attributes:
        atlas_img_np: Sample image as a Numpy array.
        labels_img_np: Integer labels image as a Numpy array.
        atlas_edge: Numpy array of atlas reduced to binary image of its edges.
        labels_edge: Numpy array of labels reduced to their edges.
        dist_to_orig: Distance map of labels to edges, with intensity values 
            in the same placement as in ``labels_edge``.
        heat_map: Numpy array as a density map.
    """
    atlas_img_np = None
    labels_img_np = None
    atlas_edge = None
    labels_edge = None
    dist_to_orig = None
    heat_map = None
    
    @classmethod
    def set_images(cls, atlas_img_np, labels_img_np, atlas_edge, labels_edge, 
                   dist_to_orig, heat_map=None):
        """Set the images."""
        cls.atlas_img_np = atlas_img_np
        cls.labels_img_np = labels_img_np
        cls.atlas_edge = atlas_edge
        cls.labels_edge = labels_edge
        cls.dist_to_orig = dist_to_orig
        cls.heat_map = heat_map
    
    @classmethod
    def label_metrics(cls, label_id):
        """Calculate metrics for a given label or set of labels.
        
        Wrapper to call :func:``measure_variation`` and 
        :func:``measure_edge_dist``.
        
        Args:
            label_id: Integer of the label of sequence of multiple labels 
                in :attr:``labels_img_np`` for which to measure variation.
        
        Returns:
            Tuple of the given label ID, intensity variation, number of 
            pixels in the label, density variation, number of blobs, 
            edge distance, and number of pixels in the label edge.
        """
        #print("getting label metrics for {}".format(label_id))
        _, var_inten, label_size, var_dens, blobs = cls.measure_variation(
            label_id)
        _, edge_dist, edge_size = cls.measure_edge_dist(label_id)
        return (label_id, var_inten, label_size, var_dens, blobs, edge_dist, 
                edge_size)
    
    @classmethod
    def measure_variation(cls, label_id):
        """Measure the variation in underlying atlas intensity.
        
        Variation is measured by standard deviation of atlas intensity and, 
        if :attr:``heat_map`` is available, that of the blob density.
        
        Args:
            label_id: Integer of the label of sequence of multiple labels 
                in :attr:``labels_img_np`` for which to measure variation.
        
        Returns:
            Tuple of the given label ID, intensity variation, number of 
            pixels in the label, density variation, and number of blobs. 
            The metrics are NaN if the label size is 0.
        """
        label_mask = np.isin(cls.labels_img_np, label_id)
        label_size = np.sum(label_mask)
        
        var_inten = np.nan
        var_dens = np.nan
        blobs = np.nan
        if label_size > 0:
            # find variation in intensity of underlying atlas/sample region
            var_inten = np.std(cls.atlas_img_np[label_mask])
            if cls.heat_map is not None:
                # find number of blob and variation in blob density
                blobs_per_px = cls.heat_map[label_mask]
                var_dens = np.std(blobs_per_px)
                blobs = np.sum(blobs_per_px)
        else:
            label_size = np.nan
        disp_id = get_single_label(label_id)
        print("variation within label {} (size {}): intensity: {}, "
              "density: {}, blobs: {}".format(
                  disp_id, label_size, var_inten, var_dens, blobs))
        return label_id, var_inten, label_size, var_dens, blobs

    @classmethod
    def measure_edge_dist(cls, label_id):
        """Measure the distance between edge images.
        
        Args:
            label_id: Integer of the label of sequence of multiple labels 
                in :attr:``labels_img_np`` for which to measure variation.
        
        Returns:
            Tuple of the given label ID, edge distance, and number of 
            pixels in the label edge. The metrics are NaN if the label 
            size is 0.
        """
        label_mask = np.isin(cls.labels_edge, label_id)
        mean_dist = np.nan
        edge_size = np.nan
        if np.sum(label_mask) > 0:
            region_dists = cls.dist_to_orig[label_mask]
            mean_dist = np.mean(region_dists)
            edge_size = region_dists.size
        disp_id = get_single_label(label_id)
        print("mean dist within edge of label {}: {}"
              .format(disp_id, mean_dist))
        return label_id, mean_dist, edge_size

def get_single_label(label_id):
    """Get an ID as a single element.
    
    Args:
        label_id: Single ID or sequence of IDs.
    
    Returns:
        The first elements if ``label_id`` is a sequence, or the 
        ``label_id`` itself if not.
    """
    if lib_clrbrain.is_seq(label_id) and len(label_id) > 0:
        return label_id[0]
    return label_id

def measure_labels_metrics(sample, atlas_img_np, labels_img_np, atlas_edge, 
                           labels_edge, dist_to_orig, heat_map=None, 
                           spacing=None, unit_factor=None, 
                           combine_sides=True, label_ids=None, grouping={}):
    """Compute metrics such as variation and distances within regions 
    based on maps corresponding to labels image.
    
    Args:
        sample: Sample ID number to be stored in data frame.
        atlas_img_np: Sample image as a Numpy array.
        labels_img_np: Integer labels image as a Numpy array.
        atlas_edge: Numpy array of atlas reduced to binary image of its edges.
        labels_edge: Numpy array of labels reduced to their edges.
        dist_to_orig: Distance map of labels to edges, with intensity values 
            in the same placement as in ``labels_edge``.
        heat_map: Numpy array as a density map; defaults to None to ignore 
            density measurements.
        spacing: Sequence of image spacing for each pixel in the images.
        unit_factor: Factor by which volumes will be divided to adjust units; 
            defaults to None.
        combine_sides: True to combine corresponding labels from opposite 
            sides of the sample; defaults to True. Corresponding labels 
            are assumed to have the same absolute numerical number and 
            differ only in signage.
        labels_ids: Sequence of label IDs to include. Defaults to None, 
            in which case the labels will be taken from unique values 
            in ``labels_img_np``.
        grouping: Dictionary of sample grouping metadata, where each 
            entry will be added as a separate column. Defaults to an 
            empty dictionary.
    
    Returns:
        Pandas data frame of the regions and weighted means for the metrics.
    """
    start_time = time()
    physical_mult = None
    if spacing is not None:
        physical_mult = np.prod(spacing)
    
    # use a class to set and process the label without having to 
    # reference the labels image as a global variable
    label_metrics = LabelMetrics()
    label_metrics.set_images(
        atlas_img_np, labels_img_np, atlas_edge, labels_edge, dist_to_orig, 
        heat_map)
    
    metrics = {}
    grouping[config.SIDE_KEY] = None
    cols = ("Sample", *grouping.keys(), "Region", "Volume", "Nuclei", 
            "Density", "VarNuclei", "VarIntensity", "EdgeDist")
    pool = mp.Pool()
    pool_results = []
    if label_ids is None:
        label_ids = np.unique(labels_img_np)
        if combine_sides: label_ids = label_ids[label_ids >= 0]
    
    for label_id in label_ids:
        # include corresponding labels from opposite sides while skipping 
        # background
        if label_id == 0: continue
        if combine_sides: label_id = [label_id, -label_id]
        pool_results.append(
            pool.apply_async(
                label_metrics.label_metrics, args=(label_id, )))
    
    totals = {}
    for result in pool_results:
        # get metrics by label
        (label_id, var_inten, label_size, var_dens, nuc, edge_dist, 
         edge_size) = result.get()
        vol_physical = label_size
        if physical_mult is not None:
            vol_physical *= physical_mult
        if unit_factor is not None:
            vol_physical /= unit_factor
        density = nuc / vol_physical
        
        # set side, assuming that positive labels are left
        if np.all(np.greater(label_id, 0)):
            side = "L"
        elif np.all(np.less(label_id, 0)):
            side = "R"
        else:
            side = "both"
        grouping[config.SIDE_KEY] = side
        disp_id = get_single_label(label_id)
        vals = (sample, *grouping.values(), abs(disp_id), vol_physical, nuc, 
                density, var_dens, var_inten, edge_dist)
        for col, val in zip(cols, vals):
            metrics.setdefault(col, []).append(val)
        
        # weight and accumulate total metrics
        totals.setdefault("dist", []).append(edge_dist * edge_size)
        totals.setdefault("edges", []).append(edge_size)
        totals.setdefault("var_inten", []).append(var_inten * label_size)
        totals.setdefault("vol", []).append(label_size)
        totals.setdefault("vol_physical", []).append(vol_physical)
        totals.setdefault("var_dens", []).append(var_dens * label_size)
        totals.setdefault("nuc", []).append(nuc)
    pool.close()
    pool.join()
    df = pd.DataFrame(metrics)
    print(df.to_csv())
    
    # add row for total metrics from weighted means
    metrics_all = {}
    grouping[config.SIDE_KEY] = "both"
    for key in totals.keys():
        totals[key] = np.nansum(totals[key])
        if totals[key] == 0: totals[key] = np.nan
    vals = (sample, *grouping.values(), "all", totals["vol_physical"], 
            totals["nuc"], totals["nuc"] / totals["vol_physical"], 
            totals["var_dens"] / totals["vol"], 
            totals["var_inten"] / totals["vol"], 
            totals["dist"] / totals["edges"])
    for col, val in zip(cols, vals):
        metrics_all.setdefault(col, []).append(val)
    df_all = pd.DataFrame(metrics_all)
    print(df_all.to_csv())
    
    print("time elapsed to measure variation:", time() - start_time)
    return df, df_all
