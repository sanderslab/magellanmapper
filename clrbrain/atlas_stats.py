#!/bin/bash
# Atlas measurements and statistics
# Author: David Young, 2019
"""Measurement of atlases and statistics generation.
"""

import numpy as np
import pandas as pd

from clrbrain import config
from clrbrain import ontology
from clrbrain import plot_2d
from clrbrain import stats


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
        
    # get whole brain tissue for normalization 
    cols_show = (*id_cols, cond_col, *extra_cols, metric)
    df_nonbr = None
    if dfs_nonbr_large:
        # subtract non-brain tissue structures from whole organism 
        # to get brain tissue alone, updating given metric in db_base
        df_nonbr = dfs_nonbr_large[0]
        for df_out in dfs_nonbr_large[1:]:
            df_nonbr = stats.normalize_df(
                df_nonbr, id_cols, cond_col, None, [metric], 
                extra_cols, df_out, stats.df_add)
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
        
        if df_nonbr is not None:
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
