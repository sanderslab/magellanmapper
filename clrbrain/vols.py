#!/bin/bash
# Regional volume and density management
# Author: David Young, 2018, 2019
"""Measure volumes and densities by regions.
"""

from enum import Enum
import multiprocessing as mp
from time import time

import numpy as np
import pandas as pd
from skimage import measure

from clrbrain import config
from clrbrain import lib_clrbrain
from clrbrain import plot_3d

# metric keys and column names
LabelMetrics = Enum(
    "LabelMetrics", [
        "Region", "Volume", "Nuclei", "Density", 
        "VolMean", "NucMean", "DensityMean", 
        "VarNuclei", 
        "VarIntensity", "VarIntensInterior", "VarIntensBorder", 
        "MeanIntensity", "MeanIntensInterior", "MeanIntensBorder", 
        "MedIntensity", "MedIntensInterior", "MedIntensBorder", 
        "LowIntensity", "LowIntensInterior", "LowIntensBorder", 
        "HighIntensity", "HighIntensInterior", "HighIntensBorder", 
        "EntropyIntensity", "EntropyIntensInterior", "EntropyIntensBorder", 
        "VarIntensDiff", "MeanIntensDiff", "MedIntensDiff", 
        "LowIntensDiff", "HighIntensDiff", "EntropyIntensDiff", 
        "EdgeSize", "EdgeDistSum", "EdgeDistMean"
    ]
)

class MetricCombos(Enum):
    """Combinations of metrics.
    
    Each combination should be a tuple of combination name and a nested 
    tuple of metric Enums.
    """
    HOMOGENEITY = (
        "Homogeneity", 
        (LabelMetrics.VarIntensity, LabelMetrics.VarIntensDiff, 
         LabelMetrics.EdgeDistSum))

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
    LabelToEdge.set_labels_img_np(labels_img_np)
    
    pool = mp.Pool()
    pool_results = []
    for label_id in label_ids:
        pool_results.append(
            pool.apply_async(
                LabelToEdge.find_label_edge, args=(label_id, )))
    for result in pool_results:
        label_id, slices, borders = result.get()
        if slices is not None:
            borders_region = labels_edge[tuple(slices)]
            borders_region[borders] = label_id
    pool.close()
    pool.join()
    
    print("time elapsed to make labels edge:", time() - start_time)
    
    return labels_edge

class MeasureLabel(object):
    """Measure metrics within image labels in a way that allows 
    multiprocessing without global variables.
    
    All images should be of the same shape.
    
    Attributes:
        atlas_img_np: Sample image as a Numpy array.
        labels_img_np: Integer labels image as a Numpy array.
        labels_edge: Numpy array of labels reduced to their edges.
        dist_to_orig: Distance map of labels to edges, with intensity values 
            in the same placement as in ``labels_edge``.
        heat_map: Numpy array as a density map.
        sub_seg: Integer sub-segmentations labels image as Numpy array.
        df: Pandas data frame with a row for each sub-region.
    """
    # metric keys
    _COUNT_METRICS = (LabelMetrics.Volume, LabelMetrics.Nuclei)
    _VAR_METRICS = (
        LabelMetrics.VolMean, LabelMetrics.NucMean, LabelMetrics.VarNuclei, 
        LabelMetrics.VarIntensity, LabelMetrics.VarIntensInterior, 
        LabelMetrics.VarIntensBorder, 
        LabelMetrics.MeanIntensity, LabelMetrics.MeanIntensInterior, 
        LabelMetrics.MeanIntensBorder, 
        LabelMetrics.MedIntensity, LabelMetrics.MedIntensInterior, 
        LabelMetrics.MedIntensBorder, 
        LabelMetrics.LowIntensity, LabelMetrics.LowIntensInterior, 
        LabelMetrics.LowIntensBorder, 
        LabelMetrics.HighIntensity, LabelMetrics.HighIntensInterior, 
        LabelMetrics.HighIntensBorder, 
        LabelMetrics.EntropyIntensity, LabelMetrics.EntropyIntensInterior, 
        LabelMetrics.EntropyIntensBorder,
        LabelMetrics.VarIntensDiff, LabelMetrics.MeanIntensDiff, 
        LabelMetrics.MedIntensDiff, LabelMetrics.LowIntensDiff, 
        LabelMetrics.HighIntensDiff, LabelMetrics.EntropyIntensDiff)
    _EDGE_METRICS = (
        LabelMetrics.EdgeSize, LabelMetrics.EdgeDistSum, 
        LabelMetrics.EdgeDistMean)
    
    # images and data frame
    atlas_img_np = None
    labels_img_np = None
    labels_edge = None
    dist_to_orig = None
    labels_markers = None
    heat_map = None
    subseg = None
    df = None
    
    @classmethod
    def set_data(cls, atlas_img_np, labels_img_np, labels_edge, 
                 dist_to_orig, labels_interior=None, heat_map=None, 
                 subseg=None, df=None):
        """Set the images and data frame."""
        cls.atlas_img_np = atlas_img_np
        cls.labels_img_np = labels_img_np
        cls.labels_edge = labels_edge
        cls.dist_to_orig = dist_to_orig
        cls.labels_interior = labels_interior
        cls.heat_map = heat_map
        cls.subseg = subseg
        cls.df = df
    
    @classmethod
    def label_metrics(cls, label_id):
        """Calculate metrics for a given label or set of labels.
        
        Wrapper to call :func:``measure_variation`` and 
        :func:``measure_edge_dist``.
        
        Args:
            label_id: Integer of the label or sequence of multiple labels 
                in :attr:``labels_img_np`` for which to measure variation.
        
        Returns:
            Tuple of the given label ID, intensity variation, number of 
            pixels in the label, density variation, number of blobs, 
            sum edge distances, mean of edge distances, and number of 
            pixels in the label edge.
        """
        #print("getting label metrics for {}".format(label_id))
        _, count_metrics = cls.measure_counts(label_id)
        _, var_metrics = cls.measure_variation(label_id)
        _, edge_metrics = cls.measure_edge_dist(label_id)
        metrics = {**count_metrics, **var_metrics, **edge_metrics}
        return label_id, metrics
    
    @classmethod
    def measure_counts(cls, label_ids):
        """Measure the distance between edge images.
        
        Args:
            label_ids: Integer of the label or sequence of multiple labels 
                in :attr:``labels_img_np`` for which to measure variation.
        
        Returns:
            Tuple of the given label ID, sum of edge distances, mean of 
            edge distances, and number ofpixels in the label edge. 
            The metrics are NaN if the label size is 0.
        """
        metrics = dict.fromkeys(cls._COUNT_METRICS, np.nan)
        nuclei = np.nan
        
        if cls.df is None:
            # sum up counts within the collective region
            label_mask = np.isin(cls.labels_img_np, label_ids)
            label_size = np.sum(label_mask)
            if cls.heat_map is not None:
                nuclei = np.sum(cls.heat_map[label_mask])
        else:
            # get all rows associated with region and sum stats within columns
            labels = cls.df.loc[
                cls.df[LabelMetrics.Region.name].isin(label_ids)]
            label_size = np.nansum(labels[LabelMetrics.Volume.name])
            nuclei = np.nansum(labels[LabelMetrics.Nuclei.name])
        if label_size > 0:
            metrics[LabelMetrics.Volume] = label_size
            metrics[LabelMetrics.Nuclei] = nuclei
        disp_id = get_single_label(label_ids)
        print("counts within label {}: {}"
              .format(disp_id, lib_clrbrain.enum_dict_aslist(metrics)))
        return label_ids, metrics
    
    @classmethod
    def region_props(cls, region, metrics, keys):
        """Measure properties for a region and add to a dictionary.
        
        Args:
            region: Region to measure, which can be a flattened array.
            metrics: Dictionary to store metrics.
            keys: Sequence of keys corresponding to standard deviation, 
                median, and Shannon Entropy measurements.
        """
        if region.size < 1:
            for key in keys: metrics[key] = np.nan
        else:
            print(region.size, len(region))
            metrics[keys[0]] = np.std(region)
            metrics[keys[1]] = np.mean(region)
            metrics[keys[2]] = np.median(region)
            metrics[keys[3]], metrics[keys[4]] = np.percentile(region, (5, 95))
            metrics[keys[5]] = measure.shannon_entropy(region)
    
    @classmethod
    def measure_variation(cls, label_ids):
        """Measure the variation in underlying atlas intensity.
        
        Variation is measured by standard deviation of atlas intensity and, 
        if :attr:``heat_map`` is available, that of the blob density.
        
        Args:
            label_ids: Integer of the label or sequence of multiple labels 
                in :attr:``labels_img_np`` for which to measure variation.
        
        Returns:
            Tuple of the given label ID, intensity variation, number of 
            pixels in the label, density variation, and number of blobs. 
            The metrics are NaN if the label size is 0.
        """
        metrics = dict((key, []) for key in cls._VAR_METRICS)
        if not lib_clrbrain.is_seq(label_ids): label_ids = [label_ids]
        seg_ids = []
        
        for label_id in label_ids:
            # collect all sub-regions
            if cls.subseg is not None:
                # get sub-segmentations within region
                label_mask = cls.labels_img_np == label_id
                seg_ids.extend(np.unique(cls.subseg[label_mask]).tolist())
            else:
                seg_ids.append(label_id)
        
        if cls.df is None:
            # calculate stats for each sub-region
            for seg_id in seg_ids:
                if cls.subseg is not None:
                    seg_mask = cls.subseg == seg_id
                else:
                    seg_mask = cls.labels_img_np == seg_id
                size = np.sum(seg_mask)
                if size > 0:
                    # variation in intensity of underlying atlas/sample region
                    vals = dict((key, np.nan) for key in cls._VAR_METRICS)
                    vals[LabelMetrics.VolMean] = size
                    atlas_mask = cls.atlas_img_np[seg_mask]
                    cls.region_props(
                        atlas_mask, vals, 
                        (LabelMetrics.VarIntensity, 
                         LabelMetrics.MeanIntensity, 
                         LabelMetrics.MedIntensity, 
                         LabelMetrics.LowIntensity, 
                         LabelMetrics.HighIntensity, 
                         LabelMetrics.EntropyIntensity))
                    
                    if cls.labels_interior is not None:
                        # inner vs border variability
                        interior_mask = cls.labels_interior == seg_id
                        border_mask = np.logical_xor(seg_mask, interior_mask)
                        atlas_interior = cls.atlas_img_np[interior_mask]
                        atlas_border = cls.atlas_img_np[border_mask]
                        cls.region_props(
                            atlas_interior, vals, 
                            (LabelMetrics.VarIntensInterior, 
                             LabelMetrics.MeanIntensInterior, 
                             LabelMetrics.MedIntensInterior, 
                             LabelMetrics.LowIntensInterior, 
                             LabelMetrics.HighIntensInterior, 
                             LabelMetrics.EntropyIntensInterior))
                        cls.region_props(
                            atlas_border, vals, 
                            (LabelMetrics.VarIntensBorder, 
                             LabelMetrics.MeanIntensBorder, 
                             LabelMetrics.MedIntensBorder, 
                             LabelMetrics.LowIntensBorder, 
                             LabelMetrics.HighIntensBorder, 
                             LabelMetrics.EntropyIntensBorder))
                        
                        # get abs diffs
                        vals[LabelMetrics.VarIntensDiff] = abs(
                            vals[LabelMetrics.VarIntensBorder] 
                                - vals[LabelMetrics.VarIntensInterior])
                        vals[LabelMetrics.MeanIntensDiff] = abs(
                            vals[LabelMetrics.MeanIntensBorder] 
                                - vals[LabelMetrics.MeanIntensInterior])
                        vals[LabelMetrics.MedIntensDiff] = abs(
                            vals[LabelMetrics.MedIntensBorder] 
                                - vals[LabelMetrics.MedIntensInterior])
                        vals[LabelMetrics.LowIntensDiff] = abs(
                            vals[LabelMetrics.LowIntensBorder] 
                                - vals[LabelMetrics.LowIntensInterior])
                        vals[LabelMetrics.HighIntensDiff] = abs(
                            vals[LabelMetrics.HighIntensBorder] 
                                - vals[LabelMetrics.HighIntensInterior])
                        vals[LabelMetrics.EntropyIntensDiff] = abs(
                            vals[LabelMetrics.EntropyIntensBorder] 
                                - vals[LabelMetrics.EntropyIntensInterior])
                    
                    if cls.heat_map is not None:
                        # number of blob and variation in blob density
                        blobs_per_px = cls.heat_map[seg_mask]
                        vals[LabelMetrics.VarNuclei] = np.std(blobs_per_px)
                        vals[LabelMetrics.NucMean] = np.sum(blobs_per_px)
                    
                    for metric in cls._VAR_METRICS:
                        metrics[metric].append(vals[metric])
        else:
            # get sub-region stats stored in data frame
            labels = cls.df.loc[cls.df[LabelMetrics.Region.name].isin(seg_ids)]
            for i, row in labels.iterrows():
                if row[LabelMetrics.VolMean.name] > 0:
                    for metric in cls._VAR_METRICS:
                        metrics[metric].append(row[metric.name])
        
        # weight totals by sub-region size
        disp_id = get_single_label(label_ids)
        vols = np.copy(metrics[LabelMetrics.VolMean])
        tot_size = np.sum(vols) # assume no nans
        for key in metrics.keys():
            #print("{} {}: {}".format(disp_id, key.name, metrics[key]))
            if tot_size > 0:
                metrics[key] = np.nansum(
                    np.multiply(metrics[key], vols)) / tot_size
            if tot_size <= 0 or metrics[key] == 0: metrics[key] = np.nan
        print("variation within label {}: {}"
              .format(disp_id, lib_clrbrain.enum_dict_aslist(metrics)))
        return label_ids, metrics

    @classmethod
    def measure_edge_dist(cls, label_ids):
        """Measure the distance between edge images.
        
        Args:
            label_ids: Integer of the label or sequence of multiple labels 
                in :attr:``labels_img_np`` for which to measure variation.
        
        Returns:
            Tuple of the given label ID, sum of edge distances, mean of 
            edge distances, and number ofpixels in the label edge. 
            The metrics are NaN if the label size is 0.
        """
        metrics = dict.fromkeys(cls._EDGE_METRICS, np.nan)
        
        # get collective region
        if cls.df is None:
            # get region directly from image
            label_mask = np.isin(cls.labels_edge, label_ids)
            label_size = np.sum(label_mask)
        else:
            # get all row associated with region
            labels = cls.df.loc[
                cls.df[LabelMetrics.Region.name].isin(label_ids)]
            label_size = np.nansum(labels[LabelMetrics.Volume.name])
        
        if label_size > 0:
            if cls.df is None:
                # sum and take average directly from image
                region_dists = cls.dist_to_orig[label_mask]
                dist_sum = np.sum(region_dists)
                dist_mean = np.mean(region_dists)
                size = region_dists.size
            else:
                # take sum from rows and weight means by edge sizes
                dist_sum = np.nansum(labels[LabelMetrics.EdgeDistSum.name])
                sizes = labels[LabelMetrics.EdgeSize.name]
                dist_means = labels[LabelMetrics.EdgeDistMean.name]
                size = np.sum(sizes)
                dist_mean = np.sum(np.multiply(sizes, dist_means)) / size
            metrics[LabelMetrics.EdgeDistSum] = dist_sum
            metrics[LabelMetrics.EdgeDistMean] = dist_mean
            metrics[LabelMetrics.EdgeSize] = size
        disp_id = get_single_label(label_ids)
        print("dist within edge of label {}: {}"
              .format(disp_id, lib_clrbrain.enum_dict_aslist(metrics)))
        return label_ids, metrics

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

def measure_labels_metrics(sample, atlas_img_np, labels_img_np, 
                           labels_edge, dist_to_orig, labels_interior=None, 
                           heat_map=None, 
                           subseg=None, spacing=None, unit_factor=None, 
                           combine_sides=True, label_ids=None, grouping={}, 
                           df=None):
    """Compute metrics such as variation and distances within regions 
    based on maps corresponding to labels image.
    
    Args:
        sample: Sample ID number to be stored in data frame.
        atlas_img_np: Atlas or sample image as a Numpy array.
        labels_img_np: Integer labels image as a Numpy array.
        labels_edge: Numpy array of labels reduced to their edges.
        dist_to_orig: Distance map of labels to edges, with intensity values 
            in the same placement as in ``labels_edge``.
        labels_interior: Numpy array of labels eroded to interior region.
        heat_map: Numpy array as a density map; defaults to None to ignore 
            density measurements.
        sub_seg: Integer sub-segmentations labels image as Numpy array; 
            defaults to None to ignore label sub-divisions.
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
        df: Data frame with rows for all drawn labels to pool into 
            parent labels instead of re-measuring stats for all 
            children of each parent; defaults to None.
    
    Returns:
        Pandas data frame of the regions and weighted means for the metrics.
    """
    start_time = time()
    physical_mult = None
    if spacing is not None:
        physical_mult = np.prod(spacing)
    
    # use a class to set and process the label without having to 
    # reference the labels image as a global variable
    MeasureLabel.set_data(
        atlas_img_np, labels_img_np, labels_edge, dist_to_orig, 
        labels_interior, heat_map, subseg, df)
    
    metrics = {}
    grouping[config.SIDE_KEY] = None
    cols_metadata = ("Sample", *grouping.keys())
    pool = mp.Pool()
    pool_results = []
    if label_ids is None:
        label_ids = np.unique(labels_img_np)
        if combine_sides: label_ids = label_ids[label_ids >= 0]
    
    for label_id in label_ids:
        # include corresponding labels from opposite sides while skipping 
        # background
        if label_id == 0: continue
        if combine_sides: label_id = [label_id, -1 * label_id]
        pool_results.append(
            pool.apply_async(
                MeasureLabel.label_metrics, args=(label_id, )))
    
    totals = {}
    for result in pool_results:
        # get metrics by label
        label_id, label_metrics = result.get()
        label_size = label_metrics[LabelMetrics.Volume]
        nuc = label_metrics[LabelMetrics.Nuclei]
        nuc_mean = label_metrics[LabelMetrics.NucMean]
        edge_size = label_metrics[LabelMetrics.EdgeSize]
        
        vol_physical = label_size
        vol_mean_physical = label_metrics[LabelMetrics.VolMean]
        if df is None:
            # convert to physical units at the given value unless 
            # using data frame, where values presumably already converted
            if physical_mult is not None:
                vol_physical *= physical_mult
                vol_mean_physical *= physical_mult
            if unit_factor is not None:
                vol_physical /= unit_factor
                vol_mean_physical /= unit_factor
        
        # calculate densities based on physical volumes
        label_metrics[LabelMetrics.Volume] = vol_physical
        label_metrics[LabelMetrics.Density] = nuc / vol_physical
        label_metrics[LabelMetrics.VolMean] = vol_mean_physical
        label_metrics[LabelMetrics.DensityMean] = nuc_mean / vol_mean_physical
        
        # set side, assuming that positive labels are left
        if np.all(np.greater(label_id, 0)):
            side = "L"
        elif np.all(np.less(label_id, 0)):
            side = "R"
        else:
            side = "both"
        grouping[config.SIDE_KEY] = side
        disp_id = get_single_label(label_id)
        label_metrics[LabelMetrics.Region] = abs(disp_id)
        vals = (sample, *grouping.values())
        for col, val in zip(cols_metadata, vals):
            metrics.setdefault(col, []).append(val)
        for col in LabelMetrics:
            metrics.setdefault(col.name, []).append(label_metrics[col])
        
        # weight and accumulate total metrics
        totals.setdefault(LabelMetrics.EdgeDistSum, []).append(
            label_metrics[LabelMetrics.EdgeDistSum] * edge_size)
        totals.setdefault(LabelMetrics.EdgeDistMean, []).append(
            label_metrics[LabelMetrics.EdgeDistMean] * edge_size)
        totals.setdefault(LabelMetrics.EdgeSize, []).append(edge_size)
        totals.setdefault(LabelMetrics.VarIntensity, []).append(
            label_metrics[LabelMetrics.VarIntensity] * label_size)
        totals.setdefault("vol", []).append(label_size)
        totals.setdefault(LabelMetrics.Volume, []).append(vol_physical)
        totals.setdefault(LabelMetrics.VarNuclei, []).append(
            label_metrics[LabelMetrics.VarNuclei] * label_size)
        totals.setdefault(LabelMetrics.Nuclei, []).append(nuc)
        totals.setdefault(LabelMetrics.VolMean, []).append(
            vol_mean_physical * label_size)
        totals.setdefault(LabelMetrics.NucMean, []).append(
            nuc_mean * label_size)
    pool.close()
    pool.join()
    df = pd.DataFrame(metrics)
    print(df.to_csv())
    
    # build data frame of total metrics from weighted means
    metrics_all = {}
    grouping[config.SIDE_KEY] = "both"
    vals = (sample, *grouping.values())
    for col, val in zip(cols_metadata, vals):
        metrics_all.setdefault(col, []).append(val)
    for key in totals.keys():
        totals[key] = np.nansum(totals[key])
        if totals[key] == 0: totals[key] = np.nan
    
    # divide weighted values by sum of corresponding weights
    totals[LabelMetrics.Region] = "all"
    totals[LabelMetrics.Density] = (
        totals[LabelMetrics.Nuclei] / totals[LabelMetrics.Volume])
    totals[LabelMetrics.VolMean] /= totals["vol"]
    totals[LabelMetrics.NucMean] /= totals["vol"]
    totals[LabelMetrics.DensityMean] = (
        totals[LabelMetrics.NucMean] / totals[LabelMetrics.VolMean])
    totals[LabelMetrics.VarNuclei] /= totals["vol"]
    totals[LabelMetrics.VarIntensity] /= totals["vol"]
    totals[LabelMetrics.EdgeDistMean] /= totals[LabelMetrics.EdgeSize]
    for col in LabelMetrics:
        if col in totals:
            metrics_all.setdefault(col.name, []).append(totals[col])
    df_all = pd.DataFrame(metrics_all)
    print(df_all.to_csv())
    
    print("time elapsed to measure variation:", time() - start_time)
    return df, df_all

def map_meas_to_labels(labels_img, df, meas, fn_avg, skip_nans=False):
    """Generate a map of a given measurement on a labels image.
    
    The intensity values of labels will be replaced by the given metric 
    of the chosen measurement, such as the mean of the densities. If 
    multiple conditions exist, the difference of metrics for the first 
    two conditions will be taken.
    
    Args:
        labels_img: Labels image as a Numpy array in x,y,z.
        df: Pandas data frame with measurements by regions corresponding 
            to that of ``labels_img``.
        meas: Name of column in ``df`` from which to extract measurements.
        fn_avg: Function to apply to the column for each region. If None, 
            ``df`` is assumed to already contain statistics generated from 
            the ``clrstats`` R package, which will be extracted directly.
        skip_nans: True to skip any region with NaNs, leaving 0 instead; 
            defaults to False to allow NaNs in resulting image. Some 
            applications may not be able to read NaNs, so this parameter 
            allows giving a neutral value instead.
    
    Retunrs:
        A map of averages for the given measurement as an image of the 
        same shape as ``labels_img`` of float data type, or None if no 
        values for ``meas`` are found.
    """
    if meas not in df or np.all(np.isnan(df[meas])):
        print("{} not in data frame or all NaNs, no image to generate"
              .format(meas))
        return None
    # ensure that at least 2 conditions exist to compare
    conds = np.unique(df["Condition"]) if "Condition" in df else []
    labels_diff = np.zeros_like(labels_img, dtype=np.float)
    labels_img_abs = np.abs(labels_img)
    regions = np.unique(labels_img_abs)
    for region in regions:
        df_region = df[df[LabelMetrics.Region.name] == region]
        labels_region = labels_img_abs == region
        if fn_avg is None:
            # assume that df was output by R clrstats package; take  
            # effect size and weight by -logp
            labels_diff[labels_region] = (
                df_region["vals.effect"] * df_region["vals.logp"])
        else:
            if len(conds) >= 2:
                # compare the metrics for the first two conditions
                avgs = []
                for cond in conds:
                    # gather separate metrics for each condition
                    df_region_cond = df_region[df_region["Condition"] == cond]
                    #print(df_region_cond.to_csv())
                    print(region, cond, fn_avg(df_region_cond[meas]))
                    avgs.append(fn_avg(df_region_cond[meas]))
                if skip_nans and np.any(np.isnan(avgs[:2])):
                    # will get NaNs if no row for region or if rows contain 
                    # NaNs; skip if get NaN avgs for first 2 conditions
                    print("region {} has NaNs, skipping".format(region))
                    continue
                labels_diff[labels_region] = avgs[0] - avgs[1]
            else:
                # take the metric for the single condition
                labels_diff[labels_region] = fn_avg(df_region[meas])
    return labels_diff
