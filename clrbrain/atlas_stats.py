#!/bin/bash
# Atlas measurements and statistics
# Author: David Young, 2019
"""Low-level measurement of atlases and statistics generation.

Typically applied to specific types of atlases and less generalizable
than measurements in :module:`vols`.
"""
import os

import numpy as np
import pandas as pd

from clrbrain import colormaps
from clrbrain import config
from clrbrain import lib_clrbrain
from clrbrain import ontology
from clrbrain import plot_2d
from clrbrain import plot_support
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


def plot_intensity_nuclei(paths, labels, size=None, show=True, unit=None):
    """Plot nuclei vs. intensity as a scatter plot.
    
    Args:
        paths (List[str]): Sequence of paths to CSV files.
        labels (List[str]): Sequence of label metrics corresponding to 
            ``paths``.
        size (List[int]): Sequence of ``width, height`` to size the figure; 
            defaults to None.
        show (bool): True to display the image; defaults to True.
        unit (str): Denominator unit for density plot; defaults to None.
    
    Returns:
        :obj:`pd.DataFrame`: Data frame with columns matching ``labels``
        for the given ``paths`` concatenated.

    """
    def plot(lbls, suffix=None, unit=None):
        cols_xy = []
        for label in lbls:
            # get columns for the given label to plot on a given axis; assume
            # same order of labels for each group of columns so they correspond
            cols_xy.append([
                c for c in df.columns if c.split(".")[0] == label])
    
        names_group = None
        if cols_xy:
            # extract legend names assuming label.cond format
            names_group = np.unique([c.split(".")[1] for c in cols_xy[0]])
        units = (["{}/{}".format(l.split("_")[0], unit) for l in lbls] 
                 if unit else None)
        lbls = [l.replace("_", " ") for l in lbls]
        title = "{} Vs. {} By Region".format(*lbls)
        plot_2d.plot_scatter(
            "vols_stats_intensVnuc", cols_xy[0], cols_xy[1], units=units,
            # col_annot=config.AtlasMetrics.REGION_ABBR.value,
            names_group=names_group, labels=lbls, title=title,
            fig_size=size, show=show, suffix=suffix, df=df)
    
    if len(paths) < 2 or len(labels) < 2: return
    dfs = [pd.read_csv(path) for path in paths]
    # merge data frames with all columns ending with .mean, prepending labels
    extra_cols = [
        config.AtlasMetrics.REGION.value,
        config.AtlasMetrics.REGION_ABBR.value,
        vols.LabelMetrics.Volume.name,
    ]
    tag = ".mean"
    df = stats.append_cols(
        dfs[:2], labels, lambda x: x.lower().endswith(tag), extra_cols)
    dens = "{}_density"
    for col in df.columns:
        if col.startswith(labels):
            col_split = col.split(".")
            col_split[0] = dens.format(col_split[0])
            df.loc[:, ".".join(col_split)] = (
                    df[col] / df[vols.LabelMetrics.Volume.name])
    # strip the tag from column names
    names = {col: col.rsplit(tag)[0] for col in df.columns}
    df = df.rename(columns=names)
    stats.data_frames_to_csv(df, "vols_stats_intensVnuc.csv")
    
    # plot labels and density labels
    plot(labels)
    plot([dens.format(l) for l in labels], "_density", unit)
    
    return df


def meas_improvement(path, col_effect, col_p, thresh_impr=0, thresh_p=0.05, 
                     col_wt=None, suffix=None, df=None):
    """Measure overall improvement and worsening for a column in a data frame.
    
    Args:
        path (str): Path of file to load into data frame.
        col_effect (str): Name of column with metric to measure.
        col_p (str): Name of column with p-values.
        thresh_impr (float): Threshold of effects below which are considered
            improved.
        thresh_p (float): Threshold of p-values below which are considered
            statistically significant.
        col_wt (str): Name of column for weighting.
        suffix (str): Output path suffix; defaults to None.
        df (:obj:`pd.DataFrame`): Data fram to use instead of loading from
            ``path``; defaults to None.

    Returns:
        :obj:`pd.DataFrame`: Data frame with improvement measurements.
        The data frame will be saved to a filename based on ``path``.

    """
    def add_wt(mask_cond, mask_cond_ss, name):
        # add weighted metrics for the given condition, such as improved
        # vs. worsened
        metrics[col_wt] = [np.sum(df[col_wt])]
        wt_cond = df.loc[mask_cond, col_wt]
        wt_cond_ss = df.loc[mask_cond_ss, col_wt]
        # sum of weighting column fitting the condition (all and statistically
        # significant)
        metrics["{}_{}".format(col_wt, name)] = [np.sum(wt_cond)]
        metrics["{}_{}_ss".format(col_wt, name)] = [np.sum(wt_cond_ss)]
        # sum of filtered effect multiplied by weighting
        metrics["{}_{}_by_{}".format(col_effect, name, col_wt)] = [np.sum(
            wt_cond.multiply(df.loc[mask_cond, col_effect]))]
        metrics["{}_{}_by_{}_ss".format(col_effect, name, col_wt)] = [np.sum(
            wt_cond_ss.multiply(df.loc[mask_cond_ss, col_effect]))]
    
    if df is None:
        df = pd.read_csv(path)

    # masks of improved and worsened, all and statistically significant 
    # for each, where improvement is above the given threshold
    effects = df[col_effect]
    mask_impr = effects > thresh_impr
    mask_ss = df[col_p] < thresh_p
    mask_impr_ss = mask_impr & mask_ss
    mask_wors = effects < thresh_impr
    mask_wors_ss = mask_wors & mask_ss
    metrics = {
        "n": [len(effects)],
        "n_impr": [np.sum(mask_impr)],
        "n_impr_ss": [np.sum(mask_impr_ss)],
        "n_wors": [np.sum(mask_wors)],
        "n_wors_ss": [np.sum(mask_wors_ss)],
        col_effect: [np.sum(effects)],
        "{}_impr".format(col_effect): [np.sum(effects[mask_impr])],
        "{}_impr_ss".format(col_effect): [np.sum(effects[mask_impr_ss])],
        "{}_wors".format(col_effect): [np.sum(effects[mask_wors])],
        "{}_wors_ss".format(col_effect): [np.sum(effects[mask_wors_ss])],
    }
    if col_wt:
        # add columns based on weighting column
        add_wt(mask_impr, mask_impr_ss, "impr")
        add_wt(mask_wors, mask_wors_ss, "wors")
    
    out_path = lib_clrbrain.insert_before_ext(path, "_impr")
    if suffix:
        out_path = lib_clrbrain.insert_before_ext(out_path, suffix)
    df_impr = stats.dict_to_data_frame(metrics, out_path)
    # display transposed version for more compact view given large number
    # of columns, but save un-transposed to preserve data types
    stats.print_data_frame(df_impr.T, index=True, header=False)
    return df_impr


def plot_clusters_by_label(path, z, suffix=None):
    """Plot separate sets of clusters for each label.
    
    Args:
        path (str): Base path to blobs file with clusters.
        z (int): z-plane to plot.
        suffix (str): Suffix for ``path``; defaults to None.

    """
    mod_path = path
    if suffix is not None:
        mod_path = lib_clrbrain.insert_before_ext(path, suffix)
    blobs = np.load(lib_clrbrain.combine_paths(
        mod_path, config.SUFFIX_BLOB_CLUSTERS))
    label_ids = np.unique(blobs[:, 3])
    fig, gs = plot_support.setup_fig(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    plot_support.hide_axes(ax)
    colors = colormaps.discrete_colormap(
        len(np.unique(blobs[:, 4])), prioritize_default="cn") / 255.
    for label_id in label_ids:
        # sort blobs within label by cluster size (descending order),
        # including clusters within all z-planes to keep same order across zs
        blobs_lbl = blobs[blobs[:, 3] == label_id]
        clus_lbls, clus_lbls_counts = np.unique(
            blobs_lbl[:, 4], return_counts=True)
        clus_lbls = clus_lbls[np.argsort(clus_lbls_counts)][::-1]
        for clus_lbl, color in zip(clus_lbls, colors):
        blobs_lbl = blobs_lbl[blobs_lbl[:, 0] == z]
            blobs_clus = blobs_lbl[blobs_lbl[:, 4] == clus_lbl]
            if len(blobs_clus) < 1: continue
            size = 0.2
            if clus_lbl == -1:
                color = (0, 0, 0, 1)
                size = 0.1
            print(label_id, clus_lbl, color, len(blobs_clus))
            print(blobs_clus)
            ax.scatter(
                blobs_clus[:, 2], blobs_clus[:, 1], color=color, s=size)
    plot_support.show()
