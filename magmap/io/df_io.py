# Stats for MagellanMapper
# Author: David Young, 2018, 2019
"""Stats calculations and text output for MagellanMapper.

Attributes:
"""

from enum import Enum
import os
import warnings

import numpy as np
import pandas as pd

from magmap.settings import config
from magmap.io import libmag


def weight_mean(vals, weights):
    """Calculate the weighted arithmetic mean.
    
    Args:
        vals (List[float]): Sequence of values, which can include NaNs.
        weights (List[float]): Sequence of weights.

    Returns:
        The weighted arithmetic mean of ``vals``.

    """
    # exclude corresponding weights of NaN values from total weight, 
    # while nansum excludes them from weighted total values
    tot_wt = np.sum(weights[~np.isnan(vals)])
    return np.nansum(np.multiply(vals, weights)) / tot_wt


def weight_std(vals, weights):
    """Calculate the weighted standard deviation.
    
    Args:
        vals (List[float]): Sequence of values, which can include NaNs.
        weights (List[float]): Sequence of weights.

    Returns:
        The weighted arithmetic standard deviation of ``vals``.

    """
    wt_mean = weight_mean(vals, weights)
    wt_std = np.sqrt(weight_mean(np.power(vals - wt_mean, 2), weights))
    return wt_std, wt_mean


def meas_dice(mask1, mask2, img=None):
    """Measure Dice Similarity Coefficient (DSC) between two images.
    
    Args:
        mask1 (:obj:`np.ndarray`): Mask of first image.
        mask2 (:obj:`np.ndarray`): Mask of second image with same shape as
            that of ``mask2``.
        img (:obj:`np.ndarray`): Intensity image whose values within each
            mask will be summed, of the same shape as that of the masks;
            defaults to None.

    Returns:
        :float: DSC between the two images, either based directly on the 
        mask volumes or weighted by intensities of ``img`` if given.

    """
    union = np.logical_and(mask1, mask2)
    if img is None:
        # use direct volume of masks
        out = (mask1, mask2)
    else:
        # weight volumes by underlying intensity
        union = img[union]
        out = (img[mask1], img[mask2])
    denom = np.sum([np.sum(o) for o in out])
    dsc = np.nan if denom == 0 else 2.0 * np.sum(union) / denom
    return dsc


def df_div(df0, df1, axis=1):
    """Wrapper function to divide two Pandas data frames in a functional manner.
    
    Args:
        df0 (:obj:`pd.DataFrame`): First data frame.
        df1 (:obj:`pd.DataFrame`): Second data frame.
        axis (int): Axis; defaults to 1.

    Returns:
        The quotient from applying :meth:`pd.DataFrame.div` from ``df0`` to 
        ``df1``.

    """
    return df0.div(df1, axis=axis)


def df_add(df0, df1, axis=1, fill_value=0):
    """Wrapper function to add two Pandas data frames in a functional manner.

    Args:
        df0 (:obj:`pd.DataFrame`): First data frame.
        df1 (:obj:`pd.DataFrame`): Second data frame.
        axis (int): Axis; defaults to 1.
        fill_value (int): Value with which to fill NaNs; defaults to 0.

    Returns:
        The difference from applying :meth:`pd.DataFrame.subtract` from 
        ``df0`` to ``df1``.

    """
    return df0.add(df1, axis=axis, fill_value=fill_value)


def df_subtract(df0, df1, axis=1, fill_value=0):
    """Wrapper function to subtract two Pandas data frames in a functional 
    manner.
    
    Args:
        df0 (:obj:`pd.DataFrame`): First data frame.
        df1 (:obj:`pd.DataFrame`): Second data frame.
        axis (int): Axis; defaults to 1.
        fill_value (int): Value with which to fill NaNs; defaults to 0.

    Returns:
        The difference from applying :meth:`pd.DataFrame.subtract` from 
        ``df0`` to ``df1``.

    """
    return df0.subtract(df1, axis=axis, fill_value=fill_value)


def exps_by_regions(path, filter_zeros=True, sample_delim="-"):
    """Transform volumes by regions data frame to experiments-condition 
    as columns and regions as rows.
    
    Multiple measurements for each experiment-condition combination such 
    measurements from separate sides of each sample will 
    be summed. A separate data frame will be generated for each 
    measurement.
    
    Args:
        path: Path to data frame generated from :func:``regions_to_pandas`` 
            or an aggregate of these data frames.
        filter_zero: True to remove rows that contain only zeros.
        sample_delim: Split samples column by this delimiter, taking only 
            the first split element. Defaults to "-"; if None, will 
            not split the samples.
    
    Returns:
        Dictionary of transformed dataframes with measurements as keys.
    """
    df = pd.read_csv(path)
    measurements = ("Volume", "Nuclei") # raw measurements
    
    # combine sample name with condition
    samples = df["Sample"]
    if sample_delim is not None:
        # use truncated sample names, eg for sample ID
        samples = samples.str.split(sample_delim, n=1).str[0]
    df["SampleCond"] = df["Condition"] + "_" + samples
    
    dfs = {}
    for meas in measurements:
        # combines values from each side by summing
        df_pivoted = df.pivot_table(
            values=meas, index=["Region"], columns=["SampleCond"], 
            aggfunc=np.sum)
        if filter_zeros:
            # remove rows that contain all zeros and replace remaining zeros 
            # with NaNs since 0 is used for volumes that could not be found, 
            # not necessarily those without any nuclei
            df_pivoted = df_pivoted[(df_pivoted != 0).any(axis=1)]
            df_pivoted[df_pivoted == 0] = np.nan
        dfs[meas] = df_pivoted
    
    # calculate densities directly from values since simple averaging of 
    # density columns would not weight appropriately
    df_dens = dfs[measurements[1]] / dfs[measurements[0]]
    dfs["Dens"] = df_dens
    base_path = os.path.splitext(path)[0]
    
    # export data frames to separate files
    for key in dfs.keys():
        df_pivoted = dfs[key]
        print("df_{}:\n{}".format(key, df_pivoted))
        df_path = "{}_{}.csv".format(base_path, key)
        df_pivoted.to_csv(df_path, na_rep="NaN")
    return dfs


def normalize_df(df, id_cols, cond_col, cond_base, metric_cols, extra_cols, 
                 df_base=None, fn=df_div):
    """Normalize columns from various conditions to the corresponding 
    values in another condition.
    
    Infinite values will be converted to NaNs.
    
    Args:
        df: Pandas data frame.
        id_cols: Sequence of columns to serve as index/indices.
        cond_col: Name of the condition column.
        cond_base: Name of the condition to which all other conditions 
            will be normalized. Ignored if ``df_base`` is given.
        metric_cols: Sequence of metric columns to normalize.
        extra_cols: Sequence of additional columns to include in the 
            output data frame.
        df_base: Data frame to which values will be normalized. If given, 
            ``cond_base`` will be ignored; defaults to None.
        fn: Function by which to normalize along axis 0; defaults to 
            :meth:`df_div`.
    
    Returns:
        New data frame with columns from ``id_cols``, ``cond_col``, 
        ``metric_cols``, and ``extra_cols``. Values with condition equal 
        to ``cond_base`` should be definition be 1 or NaN, while all 
        other conditions should be normalized to the original ``cond_base`` 
        values.
    """
    # set up conditions, output columns, and data frame of base condition
    conds = np.unique(df[cond_col])
    cols = (*id_cols, cond_col, *extra_cols, *metric_cols)
    if df_base is None:
        if cond_base not in conds: return
        df_base = df.loc[df[cond_col] == cond_base, cols]
    df_base = df_base.set_index(id_cols)
    dfs = []
    
    for cond in conds:
        # copy given condition and normalize to base condition, using 
        # the given function, assumed to compare by index
        df_cond = df.loc[df[cond_col] == cond, cols].set_index(id_cols)
        df_cond.loc[:, metric_cols] = fn(
            df_cond.loc[:, metric_cols], df_base.loc[:, metric_cols], axis=0)
        df_cond = df_cond.reset_index()
        #print_data_frame(df_cond, " ")
        dfs.append(df_cond)
    
    # combine and convert inf vals to NaNs
    df_norm = pd.concat(dfs)
    df_norm[np.isinf(df_norm.loc[:, metric_cols])] = np.nan
    return df_norm


def zscore_df(df, group_col, metric_cols, extra_cols, replace_metrics=False):
    """Generate z-scores for each metric within each group.
    
    Args:
        df: Pandas data frame.
        group_col: Name of column specifying groups.
        metric_cols: Sequence of metric column names.
        extra_cols: Sequence of additional column names to include in the 
            output data frame.
        replace_metrics: True to replace ``metric_cols`` with z-scores 
            rather than adding new columns; defaults to False.
    
    Returns:
        New data frame with columns from ``extra_cols`` 
        and z-scores in columns corresponding to ``metric_cols``.
    """
    # set up groups, input cols and extra output cols
    groups = np.unique(df[group_col])
    cols = (*extra_cols, group_col, *metric_cols)
    metric_z_cols = [col + "_z" for col in metric_cols]
    dfs = []
    
    for group in groups:
        # get rows within group and add columns 
        df_group = df.loc[df[group_col] == group, cols]
        df_group = df_group.reindex(columns=[*cols, *metric_z_cols])
        for metric_z_col, metric_col in zip(metric_z_cols, metric_cols):
            col_vals = df_group[metric_col]
            mu = np.nanmean(col_vals)
            std = np.nanstd(col_vals)
            df_group.loc[:, metric_z_col] = (col_vals - mu) / std
        dfs.append(df_group)
    
    # combine data frames from each group with z-scores
    df_zscore = pd.concat(dfs)
    if replace_metrics:
        # replace original metric columns with z-scores
        df_zscore = df_zscore.drop(list(metric_cols), axis=1)
        col_dict = {z_col: col 
                    for z_col, col in zip(metric_z_cols, metric_cols)}
        df_zscore.rename(columns=col_dict, inplace=True)
    return df_zscore


def coefvar_df(df, id_cols, metric_cols, size_col=None):
    """Generate coefficient of variation for each metric within each group.
    
    Args:
        df: Pandas data frame.
        id_cols: List of column names by which to group.
        metric_cols: Sequence of metric column names.
        size_col: Name of size column, typically used for weighting; 
            defaults to None.
    
    Returns:
        New data frame with ``metric_cols`` replaced by coefficient of 
        variation and ``size_col`` replaced by mean.
    """
    def coefvar(vals):
        return np.nanstd(vals) / np.nanmean(vals)
    
    # setup aggregation by coefficient of variation for given metrics 
    # and mean of sizes
    fns_agg = {metric_col: coefvar for metric_col in metric_cols}
    if size_col:
        fns_agg[size_col] = np.nanmean
    
    # group, aggregate, and reinstate index columns
    df_coef = df.groupby(id_cols).agg(fns_agg)
    df_coef = df_coef.reset_index()
    
    return df_coef


def cond_to_cols_df(df, id_cols, cond_col, cond_base, metric_cols, sep="_"):
    """Transpose metric columns from rows within each condition group 
    to separate sets of columns.
    
    Args:
        df: Pandas data frame.
        id_cols: Sequence of columns to serve as index/indices.
        cond_col: Name of the condition column.
        cond_base: Name of first condition in output data frame; if None, 
            defaults to first condition found.
        metric_cols: Sequence of metric columns to normalize.
        sep (str): Separator for metric and condition in new column names.
    
    Returns:
        :obj:`pd.DataFrame: New data frame with ``metric_cols`` expanded
        to have separate columns for each condition in ``cond_cols``.
    """
    # set up conditions, output columns, and copy of base condition
    conds = np.unique(df[cond_col])
    if cond_base is None: cond_base = conds[0]
    if cond_base not in conds: return
    cols = (*id_cols, *metric_cols)
    df_base = df.loc[df[cond_col] == cond_base].set_index(id_cols)
    dfs = []
    
    for cond in conds:
        # copy metric cols from each condition to separate cols
        cols_dict = {
            col: "{}{}{}".format(col, sep, cond) for col in metric_cols}
        df_cond = df_base
        if cond != cond_base:
            df_cond = df.loc[df[cond_col] == cond, cols].set_index(id_cols)
        df_cond.rename(columns=cols_dict, inplace=True)
        dfs.append(df_cond)
    
    # combine cols and remove obsolete condition col
    df_out = pd.concat(dfs, axis=1)
    df_out = df_out.reset_index()
    df_out = df_out.drop(cond_col, axis=1)
    return df_out


def combine_cols(df, combos):
    """Combine columns within a single data frame with the aggregation function 
    specified in each combination.
    
    Args:
        df: Pandas data frame.
        combos: Tuple of combination column name and a nested tuple of 
            the columns to combine as Enums.
    
    Returns:
       Data frame with the combinations each as a new column.
    """
    for combo in combos:
        print(combo.value)
        combo_val = combo.value
        # only include metrics that have a corresponding col
        metrics = [val.name for val in combo_val[1] if val.name in df.columns]
        if len(metrics) < len(combo_val[1]):
            msg = ("Could not find all metrics in {}: {}\nWill combine columns "
                   "from: {}".format(combo_val[0], combo_val[1], metrics))
            warnings.warn(msg)
        # aggregate columns by specified combo function
        fn_aggr = combo_val[2]
        df.loc[:, combo_val[0]] = fn_aggr(df.loc[:, metrics])
    return df


def append_cols(dfs, labels, fn_col=None, extra_cols=None, data_cols=None):
    """Append columns from a group of data frames, optionally filtering
    to keep only columns matching criteria.
    
    Assumes that each data frame contains identical samples and ordering.
    All columns will be kept from the first data frame.
    
    Args:
        dfs (List[:obj:`pd.DataFrame`]: Sequence of data frames.
        labels (List[str]): Labels corresponding to data frame in ``dfs``
             to prepend to columns before appending.
        fn_col (func): Function by which to filter columns; defaults to
            None to keep all columns. Take precedence over ``data_cols``.
        extra_cols (List[str]): List of additional columns to keep from the
            first data frame after filtering by ``fn_col``; defaults to None.
        data_cols (List[str]): List of columns to keep from each data frame;
            defaults to None to keep all columns.

    Returns:
        The combined data frame

    """
    for i, (df, label) in enumerate(zip(dfs, labels)):
        # default to keep all columns
        cols = df.columns
        if fn_col is not None or data_cols:
            # keep only given data columns unless fn_col given
            cols = data_cols
            if fn_col is not None:
                # filter columns to keep instead
                cols = [col for col in cols if fn_col(col)]
            keep_cols = cols
            if i == 0 and extra_cols:
                # keep additional columns from first data frame
                keep_cols = extra_cols + keep_cols
            df = df[keep_cols]
        # prepend label to filtered columns
        cols_lbl = ["{}.{}".format(label, col) for col in cols]
        df = df.rename(columns=dict(zip(cols, cols_lbl)))
        dfs[i] = df
    # concatenate columns
    df = pd.concat(dfs, axis=1)
    return df


def add_cols_df(df, cols):
    """Add columns to a data frame.
    
    Args:
        df (:obj:`pd.DataFrame`): Data frame.
        cols (Dict[str, Any]): Dictionary of ``{column: default_value}``
            to add to ``df``.

    Returns:
        :obj:`pd.DataFrame`: Data frame with columns added.

    """
    for key, val in cols.items():
        df[key] = val
    return df


def join_dfs(dfs, id_col):
    """Join data frames by an ID column.
    
    Args:
        dfs (List[:obj:`pd.DataFrame`]): Sequence of data frames to join.
        id_col (str): Index column.

    Returns:
        Data frame after joining after serially joining data frames.

    """
    df_out = None
    for i, df in enumerate(dfs):
        if i == 0:
            df_out = df.set_index(id_col)
        else:
            df_out = df_out.join(df.set_index(id_col), rsuffix="_{}".format(i))
    df_out = df_out.reset_index()
    return df_out
    

def melt_cols(df, id_cols, melt_cols, var_name=None):
    """Melt down a given set of columns to rows.
    
    Args:
        df: Pandas data frame.
        id_cols: List of column names to treat as IDs.
        melt_cols: List of column names to pivot into separate rows.
        var_name: Name of column with the melted column names; defaults 
            to None to use the default name.
    
    Returns:
       Data frame with columns melted into rows.
    """
    df_melted = df.melt(
        id_vars=id_cols, value_vars=melt_cols, var_name=var_name)
    return df_melted


def pivot_with_conditions(df, index, columns, values, aggfunc="first"):
    """Pivot a data frame to columns with sub-columns for different conditions.
    
    For example, a table of metric values for different regions within 
    each sample under different conditions will be reorganized to region 
    columns that are each split into condition sub-columns.
    
    Args:
        df (:obj:`pd.DataFrame`): Data frame to pivot. 
        index (str, List): Column name or sequence of columns specifying 
            samples, generally a sequence to later unstack.
        columns (str, List): Column name or sequence of columns to pivot 
            into separate columns.
        values (str): Column of values to move into new columns.
        aggfunc (func): Aggregation function for duplicates; defaults to 
            "first" to take the first value.

    Returns:
        Tuple of the pivoted data frame and the list of pivoted columns.

    """
    # use multi-level indexing; assumes that no duplicates exist for
    # a given index-pivot-column combo, and if they do, simply take 1st val
    df_lines = df.pivot_table(
        index=index, columns=columns, values=values, aggfunc=aggfunc)
    cols = df_lines.columns  # may be fewer than orig
    if libmag.is_seq(index) and len(index) > 1:
        # move multi-index into separate sub-cols of each region and
        # reset index to access all columns
        df_lines = df_lines.unstack()
    df_lines = df_lines.reset_index()
    return df_lines, cols


def print_data_frame(df, sep=" ", index=False, header=True):
    """Print formatted data frame.
    
    Args:
        df (:obj:`pd.DataFrame`): Data frame to print.
        sep (str): Separator for columns. True or " " to print the data 
            frame with a space-separated table, or can provide an 
            alternate separator. Defaults to " ".
        index (bool): True to show index; defaults to False.
        header (bool): True to show header; defaulst to True.
    """
    if sep is True or sep == " ":
        print(df.to_string(index=index, header=header, na_rep="NaN"))
    else:
        print(df.to_csv(sep=sep, index=index, header=header, na_rep="NaN"))


def dict_to_data_frame(dict_import, path=None, sort_cols=None, show=None):
    """Import dictionary to data frame, with option to export to CSV.
    
    Args:
        dict_import: Dictionary to import. If dictionary keys are enums, 
            their names will be used instead to shorten column names.
        path: Output path to export data frame to CSV file; defaults to 
            None for no export.
        sort_cols: Column as a string of list of columns by which to sort; 
            defaults to None for no sorting.
        show: True or " " to print the data frame with a space-separated 
            table, or can provide an alternate separator. Defaults to None 
            to not print the data frame.
    
    Returns:
        The imported data frame.
    """
    df = pd.DataFrame(dict_import)
    
    keys = dict_import.keys()
    if len(keys) > 0 and isinstance(next(iter(keys)), Enum):
        # convert enum keys to names of enums
        cols = {}
        for key in keys: cols[key] = key.value
        df.rename(columns=cols, inplace=True)
    df = data_frames_to_csv(df, path, sort_cols, show)
    return df


def data_frames_to_csv(data_frames, path=None, sort_cols=None, show=None):
    """Combine and export multiple data frames to CSV file.
    
    Args:
        data_frames: List of data frames to concatenate, or a single 
            ``DataFrame``.
        path: Output path; defaults to None, in which case the data frame 
            will not be saved.
        sort_cols: Column as a string of list of columns by which to sort; 
            defaults to None for no sorting.
        show: True or " " to print the data frame with a space-separated 
            table, or can provide an alternate separator. Defaults to None 
            to not print the data frame.
    
    Returns:
        The combined data frame.
    """
    ext = ".csv"
    if path:
        if not path.endswith(ext): path += ext
        libmag.backup_file(path)
    combined = data_frames
    if not isinstance(data_frames, pd.DataFrame):
        combined = pd.concat(combined)
    if sort_cols is not None:
        combined = combined.sort_values(sort_cols)
    combined.to_csv(path, index=False, na_rep="NaN")
    if show is not None:
        print_data_frame(combined, show)
    if path:
        print("exported volume data per sample to CSV file: \"{}\""
              .format(path))
    return combined


def merge_csvs(in_paths, out_path):
    """Combine and export multiple CSV files to a single CSV file.
    
    Args:
        in_paths: List of paths to CSV files to import as data frames 
            and concatenate.
        out_path: Output path.
    """
    dfs = [pd.read_csv(path) for path in in_paths]
    data_frames_to_csv(dfs, out_path)


def filter_dfs_on_vals(dfs, cols=None, row_matches=None):
    """Filter data frames for rows matching a value for a given column 
    and concatenate the filtered data frames.
    
    Args:
        dfs (List[:obj:`pd.DataFrame`]): Sequence of data frames to filter.
        cols (List[str]): Sequence of columns to keep; defaults to None
            to keep all columns.
        row_matches (List[Tuple]): Sequence of ``(col, val)`` criteria 
            corresponding to ``dfs``, where only the rows with matching 
            values to ``val`` for the given ``col`` will be kept. Defaults 
            to None to keep all rows.

    Returns:
        Tuple[:obj:`pd.DataFrame`, List[:obj:`pd.DataFrame`]]: Tuple of 
        the concatenated filtered data frames and a list of the filtered 
        data frames.

    """
    dfs_filt = []
    for df, match in zip(dfs, row_matches):
        df_filt = df
        if match:
            # filter to keep only rows matching a value in the given column
            df_filt = df_filt.loc[df_filt[match[0]] == match[1]]
        if cols is not None:
            # keep only the given columns
            df_filt = df_filt[cols]
        dfs_filt.append(df_filt)
    df_merged = pd.concat(dfs_filt)
    return df_merged, dfs_filt


def merge_excels(paths, out_path, names=None):
    """Merge Excel files into separate sheets of a single Excel output file.

    Args:
        paths (List[str]): Sequence of paths to Excel files to load.
        out_path (str): Path to output file.
        names (List[str]): Sequence of sheet names corresponding to ``paths``.
            If None, the filenames without extensions in ``paths`` will be
            used.
    """
    libmag.backup_file(out_path)
    with pd.ExcelWriter(out_path) as writer:
        if not names:
            names = [libmag.get_filename_without_ext(p) for p in paths]
        for path, name in zip(paths, names):
            # TODO: styling appears to be lost during the read step
            df = pd.read_excel(path, index_col=0, engine="openpyxl")
            df.to_excel(writer, sheet_name=name, index=False)


def main():
    """Process stats based on command-line mode."""
    print("Starting MagellanMapper stats...")
    from magmap.io import cli
    cli.main(True)
    
    # process stats based on command-line argument
    
    stats_type = libmag.get_enum(config.stats_type, config.StatsTypes)

    if stats_type is config.StatsTypes.MERGE_CSVS:
        # merge multiple CSV files into single CSV file
        merge_csvs(config.filenames, config.prefix)

    elif stats_type is config.StatsTypes.MERGE_CSVS_COLS:
        # join multiple CSV files based on a given index column into single
        # CSV file
        dfs = [pd.read_csv(f) for f in config.filenames]
        df = join_dfs(dfs, config.plot_labels[config.PlotLabels.ID_COL])
        out_path = config.prefix
        if not out_path:
            out_path = libmag.insert_before_ext(
                config.filename, "_joined")
        data_frames_to_csv(df, out_path)

    elif stats_type is config.StatsTypes.APPEND_CSVS_COLS:
        # join multiple CSV files based on a given index column into single
        # CSV file
        dfs = [pd.read_csv(f) for f in config.filenames]
        labels = libmag.to_seq(
            config.plot_labels[config.PlotLabels.X_LABEL])
        extra_cols = libmag.to_seq(
            config.plot_labels[config.PlotLabels.X_COL])
        data_cols = libmag.to_seq(
            config.plot_labels[config.PlotLabels.Y_COL])
        df = append_cols(
            dfs, labels, extra_cols=extra_cols, data_cols=data_cols)
        out_path = config.prefix
        if not out_path:
            out_path = libmag.insert_before_ext(
                config.filename, "_appended")
        data_frames_to_csv(df, out_path)

    elif stats_type is config.StatsTypes.EXPS_BY_REGION:
        # convert volume stats data frame to experiments by region
        exps_by_regions(config.filename)

    elif stats_type is config.StatsTypes.EXTRACT_FROM_CSV:
        # extract rows from CSV file based on matching rows in given col, where 
        # "X_COL" = name of column on which to filter, and 
        # "Y_COL" = values in this column for which rows should be kept
        df = pd.read_csv(config.filename)
        df_filt, _ = filter_dfs_on_vals(
            [df], None, 
            [(config.plot_labels[config.PlotLabels.X_COL],
              config.plot_labels[config.PlotLabels.Y_COL])])
        out_path = config.prefix
        if not out_path:
            out_path = "filtered.csv"
        data_frames_to_csv(df_filt, out_path)

    elif stats_type is config.StatsTypes.ADD_CSV_COLS:
        # add columns with corresponding values for all rows, where 
        # "X_COL" = name of column(s) to add, and 
        # "Y_COL" = value(s) for corresponding cols
        df = pd.read_csv(config.filename)
        cols = {k: v for k, v in zip(
            libmag.to_seq(config.plot_labels[config.PlotLabels.X_COL]),
            libmag.to_seq(config.plot_labels[config.PlotLabels.Y_COL]))}
        df = add_cols_df(df, cols)
        out_path = config.prefix
        if not out_path:
            out_path = libmag.insert_before_ext(
                config.filename, "_appended")
        data_frames_to_csv(df, out_path)

    elif stats_type is config.StatsTypes.NORMALIZE:
        # normalize values in each group to that of a base group, where
        # "ID_COL" = ID column(s),
        # "X_COL" = condition column
        # "Y_COL" = base condition to which values will be normalized,
        # "GROUP_COL" = metric columns to normalize,
        # "WT_COL" = extra columns to keep
        df = pd.read_csv(config.filename)
        df = normalize_df(
            df, config.plot_labels[config.PlotLabels.ID_COL],
            config.plot_labels[config.PlotLabels.X_COL],
            config.plot_labels[config.PlotLabels.Y_COL], 
            config.plot_labels[config.PlotLabels.GROUP_COL],
            config.plot_labels[config.PlotLabels.WT_COL])
        out_path = config.prefix
        if not out_path:
            out_path = libmag.insert_before_ext(config.filename, "_norm")
        data_frames_to_csv(df, out_path)

    if stats_type is config.StatsTypes.MERGE_EXCELS:
        # merge multiple Excel files into single Excel file, with each
        # original Excel file as a separate sheet in the combined file
        merge_excels(
            config.filenames, config.prefix,
            config.plot_labels[config.PlotLabels.LEGEND_NAMES])


if __name__ == "__main__":
    main()
