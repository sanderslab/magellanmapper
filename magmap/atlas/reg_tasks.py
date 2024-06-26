"""Atlas-related ("register") tasks for MagellanMapper"""

import os
import re
from typing import Optional, Sequence

from magmap.io import export_regions
from magmap.settings import config
from magmap.stats import vols

_logger = config.logger.getChild(__name__)


def build_labels_diff_images(paths: Optional[Sequence[str]] = None):
    """Build labels difference images or regional heat maps for given metrics.
    
    Replaces each label in an atlas labels image with the metric value.
    
    :class:`magmap.settings.config.PlotLabels.X_COL` in
    :attr:`magmap.settings.config.plot_labels` can be used to change the
    metric column, which otherwise defaults to "vals.effect".
    
    Args:
        paths: Paths to volume stat files output from the R pipeline or
            output from the ``--register vol_stats`` pipeline
            (:meth:`magmap.atlas.register.volumes_by_id`).

    """
    # set the measurement column
    col_meas = config.plot_labels[config.PlotLabels.X_COL]
    if not col_meas:
        col_meas = "vals.effect"
    
    if paths:
        # set up metrics from filenames after first (image) filename;
        # extract metrics from R stats filename format
        path_dfs = paths
        metrics = [re.search(r"vols_stats_(.*).csv", p) for p in path_dfs]
        metrics = [m.group(1) if m else m for m in metrics]
        if not metrics[0]:
            # fall back to the measurement column, eg for metrics from the
            # vol_stats pipeline
            metrics[0] = col_meas
    else:
        # set up default metrics and assume corresponding CSVs are in
        # current working directory
        metrics = (
            vols.LabelMetrics.EdgeDistSum.name,
            vols.LabelMetrics.CoefVarNuc.name,
            vols.LabelMetrics.CoefVarIntens.name,
            vols.LabelMetrics.NucCluster.name,
            vols.LabelMetrics.NucClusNoise.name,
            #vols.MetricCombos.HOMOGENEITY.value[0], 
        )
        path_dfs = [f"vols_stats_{m}.csv" for m in metrics]
    
    for path_df, metric in zip(path_dfs, metrics):
        if not os.path.exists(path_df):
            # check for existing R stats file
            _logger.warn(f"{path_df} not found, skipping")
            continue
        if not metric:
            # check for extracted metric name
            _logger.warn(f"Metric not found from {path_df}, skipping")
            continue
        
        # generate difference image
        col_wt = vols.get_metric_weight_col(metric)
        export_regions.map_metric_to_labels_img(
            config.filename, path_df, col_meas, None, config.prefix, 
            config.show, config.labels_level, metric, col_wt)
