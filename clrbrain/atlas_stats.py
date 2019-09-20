#!/bin/bash
# Atlas measurements and statistics
# Author: David Young, 2019
"""Measurement of atlases and statistics generation.
"""
import os

import numpy as np
import pandas as pd

from clrbrain import config
from clrbrain import lib_clrbrain
from clrbrain import ontology
from clrbrain import plot_2d
from clrbrain import stats
from clrbrain import vols


def plot_region_development(metric, size=None, show=True):
    """Plot regions across development for the given metric.
    
    Args:
        metric (str): Column name of metric to track.
        size (List[int]): Sequence of ``width, height`` to size the figure; 
            defaults to None.
        show (bool): True to display the image; defaults to True.

    """
    # set up access to data frame columns
    id_cols = ["Age", "Condition"]
    extra_cols = ["RegionName"]
    cond_col = "Region"

    # assume that vol stats file is given first, then region IDs;
    # merge in region names and levels
    df_regions = pd.read_csv(config.filenames[1])
    df = pd.read_csv(config.filename).merge(
        df_regions[["Region", "RegionName", "Level"]], on="Region",
        how="left")
    
    # convert sample names to ages
    ages = ontology.rel_to_abs_ages(df["Sample"].unique())
    df["Age"] = df["Sample"].map(ages)
    
    # get large super-structures for normalization to brain tissue, where 
    # "non-brain" are spinal cord and ventricles, which are variably labeled
    df_base = df[df["Region"] == 15564]
    ids_nonbr_large = (17651, 126651558)
    dfs_nonbr_large = [df[df["Region"] == n] for n in ids_nonbr_large]
    
    # get data frame with region IDs of all non-brain structures removed
    labels_ref_lookup = ontology.create_aba_reverse_lookup(
        ontology.load_labels_ref(config.load_labels))
    ids_nonbr = []
    for n in ids_nonbr_large:
        ids_nonbr.extend(
            ontology.get_children_from_id(labels_ref_lookup, n))

    label_id = config.atlas_labels[config.AtlasLabels.ID]
    if label_id is not None:
        # show only selected region and its children
        ids = ontology.get_children_from_id(labels_ref_lookup, label_id)
        df = df[np.isin(df["Region"], ids)]
    df_brain = df.loc[~df["Region"].isin(ids_nonbr)]

    levels = np.sort(df["Level"].unique())
    conds = df["Condition"].unique()
        
    # get aggregated whole brain tissue for normalization 
    cols_show = (*id_cols, cond_col, *extra_cols, metric)
    if dfs_nonbr_large:
        # add all large non-brain structures
        df_nonbr = dfs_nonbr_large[0]
        for df_out in dfs_nonbr_large[1:]:
            df_nonbr = stats.normalize_df(
                df_nonbr, id_cols, cond_col, None, [metric], 
                extra_cols, df_out, stats.df_add)
        # subtract them from whole organism to get brain tissue alone, 
        # updating given metric in db_base
        df_base = stats.normalize_df(
            df_base, id_cols, cond_col, None, [metric], extra_cols, 
            df_nonbr, stats.df_subtract)
    df_base.loc[:, "RegionName"] = "Brain tissue"
    print("Brain {}:".format(metric))
    stats.print_data_frame(df_base.loc[:, cols_show], "\t")
    df_base_piv, regions = stats.pivot_with_conditions(
        df_base, id_cols, "RegionName", metric)
    
    # plot lines with separate styles for each condition and colors for 
    # each region name
    linestyles = ("--", "-.", ":", "-")
    num_conds = len(conds)
    linestyles = linestyles * (num_conds // (len(linestyles) + 1) + 1)
    if num_conds < len(linestyles):
        # ensure that 1st and last styles are dashed and solid unless
        linestyles = (*linestyles[:num_conds-1], linestyles[-1])
    lines_params = {
        "labels": (metric, "Post-Conceptional Age"), 
        "linestyles": linestyles, 
        "size": size, 
        "show": show, 
        "ignore_invis": True, 
        "groups": conds, 
        "marker": ".", 
    }
    line_params_norm = lines_params.copy()
    line_params_norm["labels"] = ("Fraction", "Post-Conceptional Age")
    plot_2d.plot_lines(
        config.filename, "Age", regions, 
        title="Whole Brain Development ({})".format(metric), 
        suffix="_dev_{}_brain".format(metric), 
        df=df_base_piv, **lines_params)
    
    for level in levels:
        # plot raw metric at given level
        df_level = df.loc[df["Level"] == level]
        print("Raw {}:".format(metric))
        stats.print_data_frame(df_level.loc[:, cols_show], "\t")
        df_level_piv, regions = stats.pivot_with_conditions(
            df_level, id_cols, "RegionName", metric)
        plot_2d.plot_lines(
            config.filename, "Age", regions, 
            title="Structure Development ({}, Level {})".format(
                metric, level),
            suffix="_dev_{}_level{}".format(metric, level), 
            df=df_level_piv, **lines_params)
        
        # plot metric normalized to whole brain tissue; structures 
        # above removed regions will still contain them 
        df_brain_level = df_brain.loc[df_brain["Level"] == level]
        df_norm = stats.normalize_df(
            df_brain_level, id_cols, cond_col, None, [metric], 
            extra_cols, df_base)
        print("{} normalized to whole brain:".format(metric))
        stats.print_data_frame(df_norm.loc[:, cols_show], "\t")
        df_norm_piv, regions = stats.pivot_with_conditions(
            df_norm, id_cols, "RegionName", metric)
        plot_2d.plot_lines(
            config.filename, "Age", regions, 
            units=(None, 
                   config.plot_labels[config.PlotLabels.X_UNIT]), 
            title=("Structure Development Normalized to Whole "
                   "Brain ({}, Level {})".format(metric, level)),
            suffix="_dev_{}_level{}_norm".format(metric, level), 
            df=df_norm_piv, **line_params_norm)


def plot_unlabeled_hemisphere(path, cols, size=None, show=True):
    """Plot unlabeled hemisphere fractions as bar and line plots.
    
    Args:
        path (str): Path to data frame.
        cols (List[str]): Sequence of columns to plot.
        size (List[int]): Sequence of ``width, height`` to size the figure; 
            defaults to None.
        show (bool): True to display the image; defaults to True.

    """
    # load data frame and convert sample names to ages
    df = pd.read_csv(path)
    ages = ontology.rel_to_abs_ages(df["Sample"].unique())
    df["Age"] = df["Sample"].map(ages)
    
    # generate a separate graph for each metric
    conds = df["Condition"].unique()
    for col in cols:
        title = "{}".format(col).replace("_", " ")
        y_label = "Fraction of hemisphere unlabeled"
        
        # plot as lines
        df_lines, regions = stats.pivot_with_conditions(
            df, ["Age", "Condition"], "Region", col)
        plot_2d.plot_lines(
            config.filename, "Age", regions, linestyles=("--", "-"), 
            labels=(y_label, "Post-Conceptional Age"), title=title,
            size=size, show=show, ignore_invis=True, 
            suffix="_{}".format(col), df=df_lines, groups=conds)
        
        # plot as bars, pivoting value into separate columns by condition
        df_bars = df.pivot(
            index="Sample", columns="Condition", values=col).reset_index()
        plot_2d.plot_bars(
            config.filename, conds, col_groups="Sample", y_label=y_label, 
            title=title, size=None, show=show, df=df_bars, 
            prefix="{}_{}".format(
                os.path.splitext(config.filename)[0], col))


def meas_plot_zscores(path, metric_cols, extra_cols, composites, size=None,
                      show=True):
    """Measure and plot z-scores for given columns in a data frame.
    
    Args:
        path (str): Path to data frame.
        metric_cols (List[str]): Sequence of column names for which to 
            compute z-scores.
        extra_cols (List[str]): Additional columns to included in the 
            output data frame.
        composites (List[Enum]): Sequence of enums specifying the 
            combination, typically from :class:`vols.MetricCombos`.
        size (List[int]): Sequence of ``width, height`` to size the figure; 
            defaults to None.
        show (bool): True to display the image; defaults to True.

    """
    # generate z-scores
    df = pd.read_csv(path)
    df = stats.zscore_df(df, "Region", metric_cols, extra_cols, True)
    
    # generate composite score column
    df_comb = stats.combine_cols(df, composites)
    stats.data_frames_to_csv(
        df_comb, 
        lib_clrbrain.insert_before_ext(config.filename, "_zhomogeneity"))
    
    # shift metrics from each condition to separate columns
    conds = np.unique(df["Condition"])
    df = stats.cond_to_cols_df(
        df, ["Sample", "Region"], "Condition", "original", metric_cols)
    path = lib_clrbrain.insert_before_ext(config.filename, "_zscore")
    stats.data_frames_to_csv(df, path)
    
    # display as probability plot
    lims = (-3, 3)
    plot_2d.plot_probability(
        path, conds, metric_cols, "Volume", 
        xlim=lims, ylim=lims, title="Region Match Z-Scores", 
        fig_size=size, show=show, suffix=None, df=df)


def meas_plot_coefvar(path, id_cols, cond_col, cond_base, metric_cols, 
                      composites, size_col=None, size=None, show=True):
    """Measure and plot coefficient of variation (CV) as a scatter plot.
    
    CV is computed two ways:
    
    - Based on columns and equation specified in ``composites``, applied 
      across all samples regardless of group
    - For each metric in ``metric_cols``, separated by groups
    
    Args:
        path (str): Path to data frame.
        id_cols (List[str]): Sequence of columns to serve as index/indices.
        cond_col (str): Name of the condition column.
        cond_base (str): Name of the condition to which all other conditions 
            will be normalized.
        metric_cols (List[str]): Sequence of column names for which to 
            compute z-scores.
        composites (List[Enum]): Sequence of enums specifying the 
            combination, typically from :class:`vols.MetricCombos`.
        size_col (str): Name of weighting column for coefficient of 
            variation measurement; defaults to None.
        size (List[int]): Sequence of ``width, height`` to size the figure; 
            defaults to None.
        show (bool): True to display the image; defaults to True.

    """
    # measure coefficient of variation per sample-region regardless of group
    df = pd.read_csv(path)
    df = stats.combine_cols(df, composites)
    stats.data_frames_to_csv(
        df, lib_clrbrain.insert_before_ext(config.filename, "_coefvar"))
    
    # measure CV within each condition and shift metrics from each 
    # condition to separate columns
    df = stats.coefvar_df(df, [*id_cols, cond_col], metric_cols, size_col)
    conds = np.unique(df[cond_col])
    df = stats.cond_to_cols_df(df, id_cols, cond_col, cond_base, metric_cols)
    path = lib_clrbrain.insert_before_ext(config.filename, "_coefvartransp")
    stats.data_frames_to_csv(df, path)
    
    # display CV measured by condition as probability plot
    lims = (0, 0.7)
    plot_2d.plot_probability(
        path, conds, metric_cols, "Volume",
        xlim=lims, ylim=lims, title="Coefficient of Variation", 
        fig_size=size, show=show, suffix=None, df=df)


def smoothing_peak(df, thresh_label_loss=None, filter_size=None):
    """Extract the baseline and peak smoothing quality rows from the given data 
    frame matching the given criteria.
    
    Args:
        df: Data frame from which to extract.
        thresh_label_loss: Only check rows below or equal to this 
            fraction of label loss; defaults to None to ignore.
        filter_size: Only rows with the given filter size; defaults 
            to None to ignore.
    
    Returns:
        New data frame with the baseline (filter size of 0) row and the 
        row having the peak smoothing quality meeting criteria.
    """
    if thresh_label_loss is not None:
        df = df.loc[
            df[config.SmoothingMetrics.LABEL_LOSS.value] <= thresh_label_loss]
    if filter_size is not None:
        df = df.loc[np.isin(
            df[config.SmoothingMetrics.FILTER_SIZE.value], (filter_size, 0))]
    sm_qual = df[config.SmoothingMetrics.SM_QUALITY.value]
    df_peak = df.loc[np.logical_or(
        sm_qual == sm_qual.max(), 
        df[config.SmoothingMetrics.FILTER_SIZE.value] == 0)]
    return df_peak


def plot_intensity_nuclei(paths, labels, size=None, show=True):
    """Plot nuclei vs. intensity as a scatter plot.
    
    Args:
        paths (List[str]): Sequence of paths to CSV files.
        labels (List[str]): Sequence of label metrics corresponding to 
            ``paths``.
        size (List[int]): Sequence of ``width, height`` to size the figure; 
            defaults to None.
        show (bool): True to display the image; defaults to True.

    """
    if len(paths) < 2 or len(labels) < 2: return
    dfs = [pd.read_csv(path) for path in paths]
    # merge data frames with all columns ending with .mean, prepending labels
    extra_cols = [
        config.AtlasMetrics.REGION.value,
        config.AtlasMetrics.REGION_ABBR.value,
        vols.LabelMetrics.Volume.name,
    ]
    df = stats.append_cols(
        dfs[:2], labels, lambda x: x.lower().endswith(".mean"), extra_cols)
    stats.data_frames_to_csv(df, "vols_stats_intensVnuc.csv")
    
    cols_xy = []
    for label in labels:
        # get columns for the given label to plot on a given axis; assume
        # same order of labels for each group of columns so they correspond
        cols_xy.append([
            col for col in df.columns if col.startswith(label)])
    
    names_group = None
    if len(cols_xy[0]) >= 2:
        # extract name from first 2 columns
        names_group = [col.split(".")[1] for col in cols_xy[0][:2]]
    plot_2d.plot_scatter(
        config.filename, cols_xy[0], cols_xy[1], names_group=names_group, 
        labels=labels, title="{} Vs. {} By Region".format(*labels), 
        fig_size=size, show=show, suffix=config.suffix, df=df)
