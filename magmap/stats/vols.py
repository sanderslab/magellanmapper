# Regional volume and density management
# Author: David Young, 2018, 2019
"""Measure volumes and densities by regions.

Intended to be higher-level, relatively atlas-agnostic measurements.
"""

from enum import Enum
from time import time

import numpy as np
import pandas as pd
from skimage import measure

from magmap.cv import chunking
from magmap.stats import clustering
from magmap.settings import config
from magmap.io import libmag
from magmap.atlas import ontology
from magmap.cv import cv_nd
from magmap.io import df_io

# metric keys and column names
LabelMetrics = Enum(
    "LabelMetrics", [
        "Region", "Volume", "Intensity", "Nuclei", 
        # densities; "Density" = nuclei density
        # TODO: change density to nuclei density
        # TODO: consider changing enum for KEY: name format
        "Density", "DensityIntens",
        "RegVolMean", "RegNucMean", "RegDensityMean", # per region
        "VarNuclei", "VarNucIn", "VarNucOut", 
        "VarIntensity", "VarIntensIn", "VarIntensOut", 
        "MeanIntensity", 
        "MedIntensity", 
        "LowIntensity", 
        "HighIntensity", 
        "EntropyIntensity", 
        "VarIntensMatch", 
        "VarIntensDiff", 
        "MeanNuclei", 
        "VarNucMatch", 
        "EdgeSize", "EdgeDistSum", "EdgeDistMean", 
        "CoefVarIntens", "CoefVarNuc", 
        # shape measurements
        "SurfaceArea", "Compactness", 
        # overlap metrics
        "VolDSC", "NucDSC",  # volume/nuclei Dice Similarity Coefficient
        "VolOut", "NucOut",  # volume/nuclei shifted out of orig position
        # point cloud measurements
        "NucCluster",  # number of nuclei clusters
        "NucClusNoise",  # number of nuclei that do not fit into a cluster
        "NucClusLarg",  # number of nuclei in the largest cluster
    ]
)

# variation metrics
VAR_METRICS = (
    LabelMetrics.RegVolMean, LabelMetrics.RegNucMean, 
    LabelMetrics.VarNuclei, LabelMetrics.VarNucIn, LabelMetrics.VarNucOut, 
    LabelMetrics.VarIntensity, LabelMetrics.VarIntensIn, 
    LabelMetrics.VarIntensOut, 
    LabelMetrics.MeanIntensity, 
    LabelMetrics.MedIntensity, 
    LabelMetrics.LowIntensity, 
    LabelMetrics.HighIntensity, 
    LabelMetrics.EntropyIntensity, 
    LabelMetrics.VarIntensMatch, 
    LabelMetrics.VarIntensDiff, 
    LabelMetrics.MeanNuclei, 
    LabelMetrics.VarNucMatch, 
    LabelMetrics.CoefVarIntens, LabelMetrics.CoefVarNuc, 
)

# nuclei metrics
NUC_METRICS = (
    LabelMetrics.Nuclei, 
    LabelMetrics.RegNucMean, 
    LabelMetrics.MeanNuclei, 
    LabelMetrics.VarNuclei, 
    LabelMetrics.VarNucIn, 
    LabelMetrics.VarNucOut, 
    LabelMetrics.VarNucMatch, 
    LabelMetrics.CoefVarNuc, 
)

# metrics computed from weighted averages
WT_METRICS = (
    *VAR_METRICS, 
    LabelMetrics.EdgeDistMean, 
)

def _coef_var(df):
    # calculate coefficient of variation from data frame columns, 
    # where first column is std and second is mean
    return np.divide(df.iloc[:, 0], df.iloc[:, 1])

class MetricCombos(Enum):
    """Combinations of metrics.
    
    Each combination should be a tuple of combination name, a 
    tuple of metric Enums, and a function to use for aggregation applied 
    across colums to give a new metric value for each row.
    """
    # sum of columns measuring regional homogeneity; missing columns 
    # will be ignored
    HOMOGENEITY = (
        "Homogeneity", 
        (LabelMetrics.VarIntensity, #LabelMetrics.VarIntensDiff, 
         LabelMetrics.EdgeDistSum, LabelMetrics.VarNuclei), 
        lambda x: np.nanmean(x, axis=1))
    
    # coefficient of variation of intensity values
    COEFVAR_INTENS = (
        "CoefVarIntensity", 
        (LabelMetrics.VarIntensity, LabelMetrics.MeanIntensity), 
        _coef_var)

    # coefficient of variation of intensity values
    COEFVAR_NUC = (
        "CoefVarNuclei", 
        (LabelMetrics.VarNuclei, LabelMetrics.MeanNuclei), 
        _coef_var)
    
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
            _, slices = cv_nd.get_bbox_region(props[0].bbox)
            
            # work on a view of the region for efficiency, obtaining borders 
            # as eroded region and writing into new array
            region = cls.labels_img_np[tuple(slices)]
            label_mask_region = region == label_id
            borders = cv_nd.perimeter_nd(label_mask_region)
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
    
    pool = chunking.get_mp_pool()
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
    
    All images should be of the same shape. If :attr:``df`` is available, 
    it will be used in place of underlying images. Typically this 
    data frame contains metrics for labels only at the lowest level, 
    such as drawn or non-overlapping labels. These labels can then be 
    used to aggregate values through summation or weighted means to 
    generate metrics for superseding labels that contains these 
    individual labels.
    
    Attributes:
        atlas_img_np: Sample image as a Numpy array.
        labels_img_np: Integer labels image as a Numpy array.
        labels_edge: Numpy array of labels reduced to their edges.
        dist_to_orig: Distance map of labels to edges, with intensity values 
            in the same placement as in ``labels_edge``.
        heat_map: Numpy array as a density map.
        blobs (:obj:`np.ndarray`): 2D array of blobs such as nuclei in the
            format, ``[[z, y, x, label_id, ...], ...]``. Defaults to None.
        subseg: Integer sub-segmentations labels image as Numpy array.
        df: Pandas data frame with a row for each sub-region.
    """
    # metric keys
    _COUNT_METRICS = (
        LabelMetrics.Volume, LabelMetrics.Intensity, LabelMetrics.Nuclei)
    _EDGE_METRICS = (
        LabelMetrics.EdgeSize, LabelMetrics.EdgeDistSum, 
        LabelMetrics.EdgeDistMean)
    _SHAPE_METRICS = (
        LabelMetrics.SurfaceArea, LabelMetrics.Compactness)
    _PCL_METRICS = (
        LabelMetrics.NucCluster, LabelMetrics.NucClusNoise,
        LabelMetrics.NucClusLarg,
    )
    
    # images and data frame
    atlas_img_np = None
    labels_img_np = None
    labels_edge = None
    dist_to_orig = None
    labels_interior = None
    heat_map = None
    blobs = None
    subseg = None
    df = None
    spacing = None
    
    @classmethod
    def set_data(cls, atlas_img_np, labels_img_np, labels_edge=None, 
                 dist_to_orig=None, labels_interior=None, heat_map=None, 
                 blobs=None, subseg=None, df=None, spacing=None):
        """Set the images and data frame."""
        cls.atlas_img_np = atlas_img_np
        cls.labels_img_np = labels_img_np
        cls.labels_edge = labels_edge
        cls.dist_to_orig = dist_to_orig
        cls.labels_interior = labels_interior
        cls.heat_map = heat_map
        cls.blobs = blobs
        cls.subseg = subseg
        cls.df = df
        cls.spacing = spacing
    
    @classmethod
    def label_metrics(cls, label_id, extra_metrics=None):
        """Calculate metrics for a given label or set of labels.
        
        Wrapper to call :func:``measure_variation``, 
        :func:``measure_variation``, and :func:``measure_edge_dist``.
        
        Args:
            label_id: Integer of the label or sequence of multiple labels 
                in :attr:``labels_img_np`` for which to measure variation.
            extra_metrics (List[:obj:`config.MetricGroups`]): Sequence of 
                additional metric groups to measure; defaults to None. 
        
        Returns:
            Tuple of the given label ID, intensity variation, number of 
            pixels in the label, density variation, number of blobs, 
            sum edge distances, mean of edge distances, and number of 
            pixels in the label edge.
        """
        # process basic metrics
        #print("getting label metrics for {}".format(label_id))
        _, count_metrics = cls.measure_counts(label_id)
        _, var_metrics = cls.measure_variation(label_id)
        _, edge_metrics = cls.measure_edge_dist(label_id)
        metrics = {**count_metrics, **var_metrics, **edge_metrics}
        
        if extra_metrics:
            for extra_metric in extra_metrics:
                # process additional metrics by applying corresponding function
                fn = None
                if extra_metric is config.MetricGroups.SHAPES:
                    fn = cls.measure_shapes
                elif extra_metric is config.MetricGroups.POINT_CLOUD:
                    fn = cls.measure_point_cloud
                if fn:
                    _, extra_metrics = fn(label_id)
                    metrics.update(extra_metrics)
        
        return label_id, metrics
    
    @classmethod
    def measure_counts(cls, label_ids):
        """Measure the distance between edge images.
        
        If :attr:``df`` is available, it will be used to sum values 
        from labels in ``label_ids`` found in the data frame 
        rather than re-measuring values from images.
        
        Args:
            label_ids: Integer of the label or sequence of multiple labels 
                in :attr:``labels_img_np`` for which to measure variation.
        
        Returns:
            Tuple of the given label ID and a dictionary of metrics. 
            The metrics are NaN if the label size is 0.
        """
        metrics = dict.fromkeys(cls._COUNT_METRICS, np.nan)
        nuclei = np.nan
        
        if cls.df is None:
            # sum up counts within the collective region
            label_mask = np.isin(cls.labels_img_np, label_ids)
            label_size = np.sum(label_mask)
            intens = np.sum(cls.atlas_img_np[label_mask]) # tot intensity
            if cls.heat_map is not None:
                nuclei = np.sum(cls.heat_map[label_mask])
        else:
            # get all rows associated with region and sum stats within columns
            labels = cls.df.loc[
                cls.df[LabelMetrics.Region.name].isin(label_ids)]
            label_size = np.nansum(labels[LabelMetrics.Volume.name])
            intens = np.nansum(labels[LabelMetrics.Intensity.name])
            if LabelMetrics.Nuclei.name in labels:
                nuclei = np.nansum(labels[LabelMetrics.Nuclei.name])
        if label_size > 0:
            metrics[LabelMetrics.Volume] = label_size
            metrics[LabelMetrics.Intensity] = intens
            metrics[LabelMetrics.Nuclei] = nuclei
        disp_id = get_single_label(label_ids)
        print("counts within label {}: {}"
              .format(disp_id, libmag.enum_dict_aslist(metrics)))
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
            #print(region.size, len(region))
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
        
        If :attr:``df`` is available, it will be used to calculated 
        weighted averages from labels in ``label_ids`` found in the 
        data frame rather than re-measuring values from images.
        
        Args:
            label_ids: Integer of the label or sequence of multiple labels 
                in :attr:``labels_img_np`` for which to measure variation.
        
        Returns:
            Tuple of the given label ID and a dictionary a metrics. 
            The metrics are NaN if the label size is 0.
        """
        metrics = dict((key, []) for key in VAR_METRICS)
        if not libmag.is_seq(label_ids): label_ids = [label_ids]
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
            # calculate stats for each sub-segmentation; regional ("reg") 
            # means are weighted across regions and sub-segs, where the 
            # mean for each region which should equal total of full region 
            # if only one sub-seg
            for seg_id in seg_ids:
                if cls.subseg is not None:
                    seg_mask = cls.subseg == seg_id
                else:
                    seg_mask = cls.labels_img_np == seg_id
                size = np.sum(seg_mask)
                if size > 0:
                    # variation in intensity of underlying atlas/sample region
                    vals = dict((key, np.nan) for key in VAR_METRICS)
                    vals[LabelMetrics.RegVolMean] = size
                    atlas_mask = cls.atlas_img_np[seg_mask]
                    cls.region_props(
                        atlas_mask, vals, 
                        (LabelMetrics.VarIntensity, 
                         LabelMetrics.MeanIntensity, 
                         LabelMetrics.MedIntensity, 
                         LabelMetrics.LowIntensity, 
                         LabelMetrics.HighIntensity, 
                         LabelMetrics.EntropyIntensity))
                    vals[LabelMetrics.CoefVarIntens] = (
                        vals[LabelMetrics.VarIntensity] 
                        / vals[LabelMetrics.MeanIntensity])

                    interior_mask = None
                    border_mask = None
                    if cls.labels_interior is not None:
                        # inner vs border variability
                        interior_mask = cls.labels_interior == seg_id
                        border_mask = np.logical_xor(seg_mask, interior_mask)
                        atlas_interior = cls.atlas_img_np[interior_mask]
                        atlas_border = cls.atlas_img_np[border_mask]
                        vals[LabelMetrics.VarIntensIn] = np.std(atlas_interior)
                        vals[LabelMetrics.VarIntensOut] = np.std(atlas_border)
                        
                        # get variability interior-border match as abs diff
                        vals[LabelMetrics.VarIntensMatch] = abs(
                            vals[LabelMetrics.VarIntensOut] 
                            - vals[LabelMetrics.VarIntensIn])
                    
                        # get variability interior-border simple difference
                        vals[LabelMetrics.VarIntensDiff] = (
                            vals[LabelMetrics.VarIntensOut] 
                            - vals[LabelMetrics.VarIntensIn])
                    
                    if cls.heat_map is not None:
                        # number of blob and variation in blob density
                        blobs_per_px = cls.heat_map[seg_mask]
                        vals[LabelMetrics.VarNuclei] = np.std(blobs_per_px)
                        vals[LabelMetrics.RegNucMean] = np.sum(blobs_per_px)
                        vals[LabelMetrics.MeanNuclei] = np.mean(blobs_per_px)
                        if (interior_mask is not None and 
                                border_mask is not None): 
                            heat_interior = cls.heat_map[interior_mask]
                            heat_border = cls.heat_map[border_mask]
                            vals[LabelMetrics.VarNucIn] = np.std(heat_interior)
                            vals[LabelMetrics.VarNucOut] = np.std(heat_border)
                            vals[LabelMetrics.VarNucMatch] = abs(
                                vals[LabelMetrics.VarNucOut] 
                                - vals[LabelMetrics.VarNucIn])
                        vals[LabelMetrics.CoefVarNuc] = (
                            vals[LabelMetrics.VarNuclei] 
                            / vals[LabelMetrics.MeanNuclei])
                    
                    for metric in VAR_METRICS:
                        metrics[metric].append(vals[metric])
        else:
            # get sub-region stats stored in data frame
            labels = cls.df.loc[cls.df[LabelMetrics.Region.name].isin(seg_ids)]
            for i, row in labels.iterrows():
                if row[LabelMetrics.RegVolMean.name] > 0:
                    for metric in VAR_METRICS:
                        if metric.name in row:
                            metrics[metric].append(row[metric.name])
                        else:
                            metrics[metric] = np.nan
        
        # weighted average, with weights given by frac of region or 
        # sub-region size from total size
        disp_id = get_single_label(label_ids)
        vols = np.copy(metrics[LabelMetrics.RegVolMean])
        tot_size = np.sum(vols) # assume no nans
        nucs = np.copy(metrics[LabelMetrics.RegNucMean])
        tot_nucs = np.nansum(nucs)
        for key in metrics.keys():
            #print("{} {}: {}".format(disp_id, key.name, metrics[key]))
            if tot_size > 0 and metrics[key] != np.nan:
                # take weighted mean
                if key in NUC_METRICS:
                    # use weighting from nuclei for nuclei-oriented metrics
                    metrics[key] = np.nansum(
                        np.multiply(metrics[key], nucs)) / tot_nucs
                else:
                    # default to weighting by volume
                    metrics[key] = np.nansum(
                        np.multiply(metrics[key], vols)) / tot_size
            if tot_size <= 0 or metrics[key] == 0: metrics[key] = np.nan
        print("variation within label {}: {}"
              .format(disp_id, libmag.enum_dict_aslist(metrics)))
        return label_ids, metrics

    @classmethod
    def measure_edge_dist(cls, label_ids):
        """Measure the distance between edge images.
        
        If :attr:``df`` is available, it will be used to calculated 
        a sum from edge distance sum or weighted averages from edge 
        distance mean values from labels in ``label_ids`` found in the 
        data frame rather than re-measuring values from images.
        
        Args:
            label_ids: Integer of the label or sequence of multiple labels 
                in :attr:``labels_img_np`` for which to measure variation.
        
        Returns:
            Tuple of the given label ID and dictionary of metrics. 
            The metrics are NaN if the label size is 0.
        """
        metrics = dict.fromkeys(cls._EDGE_METRICS, np.nan)
        
        # get collective region
        label_mask = None
        labels = None
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
                metrics[LabelMetrics.EdgeDistSum] = np.sum(region_dists)
                metrics[LabelMetrics.EdgeDistMean] = np.mean(region_dists)
                metrics[LabelMetrics.EdgeSize] = region_dists.size
            else:
                # take sum from rows and weight means by edge sizes
                if LabelMetrics.EdgeDistSum.name in labels:
                    metrics[LabelMetrics.EdgeDistSum] = np.nansum(
                        labels[LabelMetrics.EdgeDistSum.name])
                if LabelMetrics.EdgeSize.name in labels:
                    sizes = labels[LabelMetrics.EdgeSize.name]
                    size = np.sum(sizes)
                    metrics[LabelMetrics.EdgeSize] = size
                    if LabelMetrics.EdgeDistMean.name in labels:
                        metrics[LabelMetrics.EdgeDistMean] = (
                            np.sum(np.multiply(
                                sizes, labels[LabelMetrics.EdgeDistMean.name])) 
                            / size)
        disp_id = get_single_label(label_ids)
        print("dist within edge of label {}: {}"
              .format(disp_id, libmag.enum_dict_aslist(metrics)))
        return label_ids, metrics

    @classmethod
    def measure_shapes(cls, label_ids):
        """Measure label shapes.

        Labels will be measured even if :attr:``df`` is available 
        to account for the global shape rather than using weighted-averages.

        Args:
            label_ids: Integer of the label or sequence of multiple labels 
                in :attr:``labels_img_np`` for which to measure shapes.

        Returns:
            Tuple of the given label ID and a dictionary of metrics.
        """
        metrics = dict.fromkeys(cls._SHAPE_METRICS, np.nan)

        # sum up counts within the collective region
        label_mask = np.isin(cls.labels_img_np, label_ids)
        label_size = np.sum(label_mask)
        
        if label_size > 0:
            compactness, area, _ = cv_nd.compactness_3d(
                label_mask, cls.spacing)
            metrics[LabelMetrics.SurfaceArea] = area
            metrics[LabelMetrics.Compactness] = compactness
            # TODO: high memory consumption with these measurements
            # props = measure.regionprops(label_mask.astype(np.uint8))
            # if props:
            #     prop = props[0]
            #     metrics[LabelMetrics.ConvexVolume] = prop.convex_area
            #     metrics[LabelMetrics.Solidity] = prop.solidity
            props = None
            
        disp_id = get_single_label(label_ids)
        print("shape measurements of label {}: {}"
              .format(disp_id, libmag.enum_dict_aslist(metrics)))
        return label_ids, metrics

    @classmethod
    def measure_point_cloud(cls, label_ids):
        """Measure point cloud statistics such as those from nuclei.
        
        Assumes that the class attribute :attr:`blobs` is available.

        Args:
            label_ids: Integer of the label or sequence of multiple labels 
                in :attr:``labels_img_np`` for which to measure variation.

        Returns:
            Tuple of the given label ID and dictionary of metrics. 
            The metrics are NaN if the label size is 0.
        """
        metrics = dict.fromkeys(cls._PCL_METRICS, np.nan)
        if cls.df is None and cls.blobs is None:
            print("data frame and blobs not available, unable to measure"
                  "point cloud stats")
            return label_ids, metrics
    
        # get collective region
        labels = None
        if cls.df is None:
            # get region directly from image
            label_mask = np.isin(cls.labels_img_np, label_ids)
            label_size = np.sum(label_mask)
        else:
            # get all row associated with region
            labels = cls.df.loc[
                cls.df[LabelMetrics.Region.name].isin(label_ids)]
            label_size = np.nansum(labels[LabelMetrics.Volume.name])
    
        if label_size > 0:
            if cls.df is None:
                # sum and take average directly from image
                blobs = cls.blobs[np.isin(cls.blobs[:, 3], label_ids)]
                num_clusters, num_noise, num_largest = (
                    clustering.cluster_dbscan_metrics(blobs[:, 4]))
                metrics[LabelMetrics.NucCluster] = num_clusters
                metrics[LabelMetrics.NucClusNoise] = num_noise
                metrics[LabelMetrics.NucClusLarg] = num_largest
            else:
                for key in metrics.keys():
                    if key.name not in labels: continue
                    metrics[key] = np.nansum(labels[key.name])
        disp_id = get_single_label(label_ids)
        print("nuclei clusters within label {}: {}"
              .format(disp_id, libmag.enum_dict_aslist(metrics)))
        return label_ids, metrics


def get_single_label(label_id):
    """Get an ID as a single element.
    
    Args:
        label_id: Single ID or sequence of IDs.
    
    Returns:
        The first elements if ``label_id`` is a sequence, or the 
        ``label_id`` itself if not.
    """
    if libmag.is_seq(label_id) and len(label_id) > 0:
        return label_id[0]
    return label_id


def _update_df_side(df):
    # invert label IDs of right-sided regions; assumes that using df 
    # will specify sides explicitly in label_ids
    # TODO: consider removing combine_sides and using label_ids only
    df.loc[df[config.AtlasMetrics.SIDE.value] == config.HemSides.RIGHT.value,
           LabelMetrics.Region.name] *= -1


def _parse_vol_metrics(label_metrics, spacing=None, unit_factor=None,
                       extra_keys=None):
    # parse volume metrics into physical units and nuclei density
    physical_mult = None if spacing is None else np.prod(spacing)
    keys = [LabelMetrics.Volume]
    if extra_keys is not None:
        keys.extend(extra_keys)
    
    vols_phys = []
    found_keys = []
    for key in keys:
        if key in label_metrics:
            vols_phys.append(label_metrics[key])
            found_keys.append(key)
    label_size = vols_phys[0]
    if physical_mult is not None:
        # convert to physical units at the given value unless 
        # using data frame, where values presumably already converted
        vols_phys = np.multiply(vols_phys, physical_mult)
    if unit_factor is not None:
        # further conversion to given unit size
        unit_factor_vol = unit_factor ** 3
        vols_phys = np.divide(vols_phys, unit_factor_vol)
    if unit_factor is not None:
        # convert metrics not extracted from data frame
        if LabelMetrics.SurfaceArea in label_metrics:
            # already incorporated physical units but needs to convert 
            # to unit size
            label_metrics[LabelMetrics.SurfaceArea] /= unit_factor ** 2
    
    # calculate densities based on physical volumes
    for key, val in zip(found_keys, vols_phys):
        label_metrics[key] = val
    nuc = np.nan
    if LabelMetrics.Nuclei in label_metrics:
        nuc = label_metrics[LabelMetrics.Nuclei]
        label_metrics[LabelMetrics.Density] = nuc / vols_phys[0]
    return label_size, nuc, vols_phys


def _update_vol_dicts(label_id, label_metrics, grouping, metrics):
    # parse volume metrics metadata into master metrics dictionary
    side = ontology.get_label_side(label_id)
    grouping[config.AtlasMetrics.SIDE.value] = side
    disp_id = get_single_label(label_id)
    label_metrics[LabelMetrics.Region] = abs(disp_id)
    for key, val in grouping.items():
        metrics.setdefault(key, []).append(val)
    for col in LabelMetrics:
        if col in label_metrics:
            metrics.setdefault(col.name, []).append(label_metrics[col])


def measure_labels_metrics(atlas_img_np, labels_img_np, 
                           labels_edge, dist_to_orig, labels_interior=None, 
                           heat_map=None, blobs=None,
                           subseg=None, spacing=None, unit_factor=None, 
                           combine_sides=True, label_ids=None, grouping={}, 
                           df=None, extra_metrics=None):
    """Compute metrics such as variation and distances within regions 
    based on maps corresponding to labels image.
    
    Args:
        atlas_img_np: Atlas or sample image as a Numpy array.
        labels_img_np: Integer labels image as a Numpy array.
        labels_edge: Numpy array of labels reduced to their edges.
        dist_to_orig: Distance map of labels to edges, with intensity values 
            in the same placement as in ``labels_edge``.
        labels_interior: Numpy array of labels eroded to interior region.
        heat_map: Numpy array as a density map; defaults to None to ignore 
            density measurements.
        blobs (:obj:`np.ndarray`): 2D array of blobs; defaults to None.
        subseg: Integer sub-segmentations labels image as Numpy array; 
            defaults to None to ignore label sub-divisions.
        spacing: Sequence of image spacing for each pixel in the images.
        unit_factor: Unit factor conversion; defaults to None. Eg use 
            1000 to convert from um to mm.
        combine_sides: True to combine corresponding labels from opposite 
            sides of the sample; defaults to True. Corresponding labels 
            are assumed to have the same absolute numerical number and 
            differ only in signage. May be False if combining by passing 
            both pos/neg labels in ``label_ids``.
        label_ids: Sequence of label IDs to include. Defaults to None, 
            in which case the labels will be taken from unique values 
            in ``labels_img_np``.
        grouping: Dictionary of sample grouping metadata, where each 
            entry will be added as a separate column. Defaults to an 
            empty dictionary.
        df: Data frame with rows for all drawn labels to pool into 
            parent labels instead of re-measuring stats for all 
            children of each parent; defaults to None.
        extra_metrics (List[:obj:`config.MetricGroups`]): List of enums 
            specifying additional stats; defaults to None.
    
    Returns:
        Pandas data frame of the regions and weighted means for the metrics.
    """
    start_time = time()
    
    if df is None:
        # convert to physical units based on spacing and unit conversion
        vol_args = {"spacing": spacing, "unit_factor": unit_factor}
    else:
        # units already converted, but need to convert sides
        _update_df_side(df)
        vol_args = {}
    
    # use a class to set and process the label without having to 
    # reference the labels image as a global variable
    MeasureLabel.set_data(
        atlas_img_np, labels_img_np, labels_edge, dist_to_orig, 
        labels_interior, heat_map, blobs, subseg, df, spacing)
    
    metrics = {}
    grouping[config.AtlasMetrics.SIDE.value] = None
    pool = chunking.get_mp_pool()
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
                MeasureLabel.label_metrics, args=(label_id, extra_metrics)))
    
    totals = {}
    for result in pool_results:
        # get metrics by label
        label_id, label_metrics = result.get()
        label_size, nuc, (vol_physical, vol_mean_physical) = _parse_vol_metrics(
            label_metrics, extra_keys=(LabelMetrics.RegVolMean,), **vol_args)
        reg_nuc_mean = label_metrics[LabelMetrics.RegNucMean]
        edge_size = label_metrics[LabelMetrics.EdgeSize]
        
        # calculate densities based on physical volumes
        label_metrics[LabelMetrics.RegVolMean] = vol_mean_physical
        label_metrics[LabelMetrics.RegDensityMean] = (
            reg_nuc_mean / vol_mean_physical)
        
        # transfer all found metrics to master dictionary
        _update_vol_dicts(label_id, label_metrics, grouping, metrics)
        
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
        totals.setdefault(LabelMetrics.RegVolMean, []).append(
            vol_mean_physical * label_size)
        var_nuc = label_metrics[LabelMetrics.VarNuclei]
        if var_nuc != np.nan:
            totals.setdefault(LabelMetrics.VarNuclei, []).append(
                label_metrics[LabelMetrics.VarNuclei] * label_size)
            totals.setdefault(LabelMetrics.Nuclei, []).append(nuc)
        if reg_nuc_mean != np.nan:
            totals.setdefault(LabelMetrics.RegNucMean, []).append(
                reg_nuc_mean * label_size)
    pool.close()
    pool.join()
    
    # make data frame of raw metrics, dropping columns of all NaNs
    df = pd.DataFrame(metrics)
    df = df.dropna(axis=1, how="all")
    df_io.print_data_frame(df)
    
    # build data frame of total metrics from weighted means
    metrics_all = {}
    grouping[config.AtlasMetrics.SIDE.value] = "both"
    for key, val in grouping.items():
        metrics_all.setdefault(key, []).append(val)
    for key in totals.keys():
        totals[key] = np.nansum(totals[key])
        if totals[key] == 0: totals[key] = np.nan
    
    # divide weighted values by sum of corresponding weights
    totals[LabelMetrics.Region] = "all"
    totals[LabelMetrics.RegVolMean] /= totals["vol"]
    if LabelMetrics.Nuclei in totals:
        totals[LabelMetrics.Density] = (
            totals[LabelMetrics.Nuclei] / totals[LabelMetrics.Volume])
    if LabelMetrics.RegNucMean in totals:
        totals[LabelMetrics.RegNucMean] /= totals["vol"]
        totals[LabelMetrics.RegDensityMean] = (
            totals[LabelMetrics.RegNucMean] / totals[LabelMetrics.RegVolMean])
    if LabelMetrics.VarNuclei in totals:
        totals[LabelMetrics.VarNuclei] /= totals["vol"]
    totals[LabelMetrics.VarIntensity] /= totals["vol"]
    totals[LabelMetrics.EdgeDistMean] /= totals[LabelMetrics.EdgeSize]
    for col in LabelMetrics:
        if col in totals:
            metrics_all.setdefault(col.name, []).append(totals[col])
    df_all = pd.DataFrame(metrics_all)
    df_io.print_data_frame(df_all)
    
    print("time elapsed to measure variation:", time() - start_time)
    return df, df_all


class MeasureLabelOverlap(object):
    """Measure metrics comparing two versions of image labels in a way
    that allows multiprocessing without global variables.

    All images should be of the same shape. If :attr:``df`` is available, 
    it will be used in place of underlying images. Typically this 
    data frame contains metrics for labels only at the lowest level, 
    such as drawn or non-overlapping labels. These labels can then be 
    used to aggregate values through summation or weighted means to 
    generate metrics for superseding labels that contains these 
    individual labels.

    Attributes:
        labels_imgs: Sequence of integer labels image as Numpy arrays.
        heat_map: Numpy array as a density map; defaults to None to ignore 
            density measurements.
        df: Pandas data frame with a row for each sub-region.
    """
    _OVERLAP_METRICS = (
        LabelMetrics.Volume, LabelMetrics.Nuclei, 
        LabelMetrics.VolDSC, LabelMetrics.NucDSC,
        LabelMetrics.VolOut, LabelMetrics.NucOut,
    )
    
    # images and data frame
    labels_imgs = None
    heat_map = None
    df = None
    
    @classmethod
    def set_data(cls, labels_imgs, heat_map=None, df=None):
        """Set the images and data frame."""
        cls.labels_imgs = labels_imgs
        cls.heat_map = heat_map
        cls.df = df
    
    @classmethod
    def measure_overlap(cls, label_ids):
        """Measure the overlap between image labels.

        If :attr:``df`` is available, it will be used to sum values 
        from labels in ``label_ids`` found in the data frame 
        rather than re-measuring values from images.

        Args:
            label_ids: Integer of the label or sequence of multiple labels 
                in :attr:``labels_img_np`` for which to measure variation.

        Returns:
            Tuple of the given label ID and a dictionary of metrics. 
            The metrics are NaN if the label size is 0.
        """
        metrics = dict.fromkeys(cls._OVERLAP_METRICS, np.nan)
        nuclei = np.nan
        nuc_dsc = np.nan
        nuc_out = np.nan
        
        if cls.df is None:
            # find DSC between original and updated versions of the 
            # collective region
            label_masks = [np.isin(l, label_ids) for l in cls.labels_imgs]
            label_vol = np.sum(label_masks[0])
            vol_dsc = df_io.meas_dice(label_masks[0], label_masks[1])
            
            # sum up volume and nuclei count in the new version outside of
            # the original version; assume that volume no longer occupied by
            # new version will be accounted for by the other labels that
            # reoccupied that volume
            mask_out = np.logical_and(label_masks[1], ~label_masks[0])
            vol_out = np.sum(mask_out)
            if cls.heat_map is not None:
                nuclei = np.sum(cls.heat_map[label_masks[0]])
                nuc_dsc = df_io.meas_dice(
                    label_masks[0], label_masks[1], cls.heat_map)
                nuc_out = np.sum(cls.heat_map[mask_out])
        else:
            # get weighted average of DSCs from all rows in a super-region,
            # assuming all rows are at the lowest hierarchical level
            labels = cls.df.loc[
                cls.df[LabelMetrics.Region.name].isin(label_ids)]
            label_vols = labels[LabelMetrics.Volume.name]
            label_vol = np.nansum(label_vols)
            vol_dscs = labels[LabelMetrics.VolDSC.name]
            
            vol_dsc = df_io.weight_mean(vol_dscs, label_vols)
            # sum up volume and nuclei outside of original regions
            vol_out = np.nansum(labels[LabelMetrics.VolOut.name])
            if LabelMetrics.Nuclei.name in labels:
                nucs = labels[LabelMetrics.Nuclei.name]
                nuclei = np.nansum(nucs)
                nuc_dscs = labels[LabelMetrics.NucDSC.name]
                nuc_dsc = df_io.weight_mean(nuc_dscs, nucs)
                nuc_out = np.nansum(labels[LabelMetrics.NucOut.name])
        
        if label_vol > 0:
            # update dict with metric values
            metrics[LabelMetrics.Volume] = label_vol
            metrics[LabelMetrics.Nuclei] = nuclei
            metrics[LabelMetrics.VolDSC] = vol_dsc
            metrics[LabelMetrics.NucDSC] = nuc_dsc
            metrics[LabelMetrics.VolOut] = vol_out
            metrics[LabelMetrics.NucOut] = nuc_out
        
        disp_id = get_single_label(label_ids)
        print("overlaps within label {}: {}"
              .format(disp_id, libmag.enum_dict_aslist(metrics)))
        return label_ids, metrics


def measure_labels_overlap(labels_imgs, heat_map=None, spacing=None, 
                           unit_factor=None, combine_sides=True, 
                           label_ids=None, grouping={}, df=None):
    """Compute metrics comparing two version of atlas labels.

    Args:
        labels_imgs: Sequence of integer labels image as Numpy arrays.
        heat_map: Numpy array as a density map; defaults to None to ignore 
            density measurements.
        spacing: Sequence of image spacing for each pixel in the images.
        unit_factor: Unit factor conversion; defaults to None. Eg use 
            1000 to convert from um to mm.
        combine_sides: True to combine corresponding labels from opposite 
            sides of the sample; defaults to True. Corresponding labels 
            are assumed to have the same absolute numerical number and 
            differ only in signage. May be False if combining by passing 
            both pos/neg labels in ``label_ids``.
        label_ids: Sequence of label IDs to include. Defaults to None, 
            in which case the labels will be taken from unique values 
            in ``labels_img_np``.
        grouping: Dictionary of sample grouping metadata, where each 
            entry will be added as a separate column. Defaults to an 
            empty dictionary.
        df: Data frame with rows for all drawn labels to pool into 
            parent labels instead of re-measuring stats for all 
            children of each parent; defaults to None.

    Returns:
        :obj:`pd.DataFrame`: Pandas data frame of the regions and weighted
        means for the metrics.
    """
    start_time = time()
    
    if df is None:
        vol_args = {"spacing": spacing, "unit_factor": unit_factor}
    else:
        _update_df_side(df)
        vol_args = {}
    
    # use a class to set and process the label without having to 
    # reference the labels image as a global variable
    MeasureLabelOverlap.set_data(labels_imgs, heat_map, df)
    
    metrics = {}
    grouping[config.AtlasMetrics.SIDE.value] = None
    pool = chunking.get_mp_pool()
    pool_results = []
    for label_id in label_ids:
        # include corresponding labels from opposite sides while skipping 
        # background
        if label_id == 0: continue
        if combine_sides: label_id = [label_id, -1 * label_id]
        pool_results.append(
            pool.apply_async(
                MeasureLabelOverlap.measure_overlap, args=(label_id,)))
    
    for result in pool_results:
        # get metrics by label
        label_id, label_metrics = result.get()
        label_size, nuc, _ = _parse_vol_metrics(
            label_metrics, extra_keys=(LabelMetrics.VolOut,), **vol_args)

        # transfer all found metrics to master dictionary
        _update_vol_dicts(label_id, label_metrics, grouping, metrics)

    pool.close()
    pool.join()
    
    # make data frame of raw metrics, dropping columns of all NaNs
    df = pd.DataFrame(metrics)
    df = df.dropna(axis=1, how="all")
    df_io.print_data_frame(df)
    
    print("time elapsed to measure variation:", time() - start_time)
    return df


def map_meas_to_labels(labels_img, df, meas, fn_avg, skip_nans=False, 
                       reverse=False, col_wt=None):
    """Generate a map of a given measurement on a labels image.
    
    The intensity values of labels will be replaced by the given metric 
    of the chosen measurement, such as the mean of the densities. If 
    multiple conditions exist, the difference of metrics for the first 
    two conditions will be taken under the assumption that the values for
    each condition are in matching order.
    
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
        reverse: Reverse the order of sorted conditions when generating 
            stats by ``fn_avg`` to compare conditions; defaults to False.
        col_wt (str): Name of column to use for weighting, where the 
            magnitude of ``meas`` will be adjusted as fractions of the max 
            value in this weighting column for labels found in ``labels_img``; 
            defaults to None.
    
    Retunrs:
        A map of averages for the given measurement as an image of the 
        same shape as ``labels_img`` of float data type, or None if no 
        values for ``meas`` are found.
    """
    if meas not in df or np.all(np.isnan(df[meas])):
        # ensure that measurement column is present with non-NaNs
        print("{} not in data frame or all NaNs, no image to generate"
              .format(meas))
        return None
    
    # make image array to map differences for each label and filter data 
    # frame to get only these regions
    labels_diff = np.zeros_like(labels_img, dtype=np.float)
    labels_img_abs = np.abs(labels_img)
    regions = np.unique(labels_img_abs)
    df = df.loc[df["Region"].isin(regions)].copy()
    
    df_cond = None
    conds = None
    if "Condition" in df:
        # get and sort conditions
        df_cond = df["Condition"]
        conds = sorted(np.unique(df_cond), reverse=reverse)

    if col_wt is not None:
        # weight given column for the first condition and normalizing it to
        # its maximum value, or use the whole column if no conditions exist
        print("weighting stats by", col_wt)
        wts = df.loc[df_cond == conds[0], col_wt] if conds else df[col_wt]
        wts /= max(wts)
        if conds:
            for cond in conds:
                # use raw values to avoid multiplying by index; assumes
                # matching order of values between conditions
                df.loc[df_cond == cond, meas] = np.multiply(
                    df.loc[df_cond == cond, meas].values, wts.values)
        else:
            df.loc[:, meas] *= wts
    
    for region in regions:
        # get difference for each region, either from a single column 
        # that already has the difference of effect size of by taking 
        # the difference from two columns
        df_region = df[df[LabelMetrics.Region.name] == region]
        labels_region = labels_img_abs == region
        diff = np.nan
        if fn_avg is None:
            # assume that df was output by R clrstats package
            if df_region.shape[0] > 0:
                diff = df_region[meas]
        else:
            if len(conds) >= 2:
                # compare the metrics for the first two conditions
                avgs = []
                for cond in conds:
                    # gather separate metrics for each condition
                    df_region_cond = df_region[df_region["Condition"] == cond]
                    # print(df_region_cond)
                    reg_avg = fn_avg(df_region_cond[meas])
                    # print(region, cond, reg_avg)
                    avgs.append(reg_avg)
                # TODO: consider making order customizable
                diff = avgs[1] - avgs[0]
            else:
                # take the metric for the single condition
                diff = fn_avg(df_region[meas])
        if skip_nans and np.isnan(diff):
            diff = 0
        labels_diff[labels_region] = diff
        print("label {} difference: {}".format(region, diff))
    return labels_diff


def get_metric_weight_col(stat):
    """Get the weighting column for a given metric.
    
    Args:
        stat (str): The metric for which to find the appropriate weighting 
            metric.

    Returns:
        The name of the corresponding weighting metric as a string.

    """
    col_wt = None
    if stat in [metric.name for metric in WT_METRICS]:
        col_wt = LabelMetrics.Volume.name
        if stat in [metric.name for metric in NUC_METRICS]:
            col_wt = LabelMetrics.Nuclei.name
    return col_wt
