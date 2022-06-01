# Machine learning for MagellanMapper
# Author: David Young, 2017, 2019
"""Machine learning and output for MagellanMapper.
"""

# import annotations to allow sub-type hints
from __future__ import annotations
from collections import OrderedDict
from enum import Enum
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from magmap.settings import config
from magmap.io import libmag
from magmap.io import df_io


class GridSearchStats(Enum):
    """Grid search statistics categories."""
    PARAM = "Par"  # (hyper)parameter
    PPV = "PPV"  # positive predictive value
    SENS = "Sens"  # sensitivity
    POS = "Pos"  # condition positive
    TP = "TP"  # true positive
    FP = "FP"  # false positive
    TN = "TN"  # true negative
    FN = "FN"  # false negative
    FDR = "FDR"  # false discovery rate


def grid_search(
        hyperparams: OrderedDict[str, Sequence[float]],
        fnc: Callable[[Any], Tuple[Any, Sequence]],
        *fnc_args
) -> OrderedDict[str, Tuple[Sequence, Sequence, str, OrderedDict]]:
    """Perform a grid search for hyperparameter optimization.

    A separate grid search will be performed for each item in ``roc_dict``.
    Note that currently each subsequent grid search will use the last
    settings from the prior search.

    Args:
        hyperparams: Ordered dictionary with sequences of the format:
            ``(:class:`profiles.ROIProfile` parameter, (a, b, c, ...))``.
        fnc: Function to call during the grid search, which must
            return ``stats, summaries``.
        *fnc_args: Arguments to pass to ``fnc``.

    Returns:
        Dictionary of stats suitable for parsing in :meth:`parse_grid_stats`.

    """
    # gets the ROC settings
    settings = config.roi_profile
    file_summaries = []
    iterable_keys = []  # hyperparameters to iterate through
    iterable_dict = OrderedDict()  # group results
    for key2, value2 in hyperparams.items():
        if np.isscalar(value2):
            # set scalar values rather than iterating and processing
            settings[key2] = value2
            print("changed {} to {}".format(key2, value2))
        else:
            print("adding iterable setting {}".format(key2))
            iterable_keys.append(key2)
            
    def grid_iterate(i, grid_dict, name, parent_params):
        key = iterable_keys[i]
        name = key if name is None else name + "-" + key
        print("name: {}".format(name))
        if i < len(iterable_keys) - 1:
            name += "("
            for j in grid_dict[key]:
                settings[key] = j
                # track parents and their values for given run
                parent_params = parent_params.copy()
                parent_params[key] = j
                paren_i = name.rfind("(")
                if paren_i != -1:
                    name = name[:paren_i]
                if libmag.is_number(j):
                    name += "({:.3g})".format(j)
                else:
                    name += " {}".format(j)
                grid_iterate(
                    i + 1, grid_dict, name, parent_params)
        else:
            # process each value in parameter array
            stats = []
            last_param_vals = grid_dict[key]
            for param in last_param_vals:
                print("===============================================\n"
                      "Grid search hyperparameters {} for {}"
                      .format(name, libmag.format_num(param, 3)))
                settings[key] = param
                stat, summaries = fnc(*fnc_args)
                stats.append(stat)
                file_summaries.extend(summaries)
            iterable_dict[name] = (
                stats, last_param_vals, key, parent_params)
    
    grid_iterate(0, hyperparams, None, OrderedDict())
    
    # summary of each file collected together
    for summary in file_summaries:
        print(summary)
    return iterable_dict


def parse_grid_stats(
        stats: OrderedDict[str, Tuple[Sequence, Sequence, str, OrderedDict]]
) -> Tuple[Dict[str, Tuple[Sequence, Sequence, Sequence]], pd.DataFrame]:
    """Parse stats from a grid search.
    
    Args:
        stats: Dictionary where key is a string with the parameters
            up to the last parameter group, and each value is a tuple of 
            the raw stats as (pos, true_pos, false_pos); the array of
            values for the last parameter; the last parameter key; and an 
            ``OrderedDict`` of the parent parameters and their values for 
            the given set of stats.
    
    Returns:
        Tuple of ``group_stats`` and ``df``:
        - ``group_stats`` is a dictionary of stats, where keys
          correspond go ``stats`` keys, and values are tuples of the
          false discovery rate, sensitivity, and last parameter group value,
          each as sequences
        - ``df`` is a data frame summarizing the stats
    
    """
    
    # parse a grid search
    stats_for_df = {}
    headers = None
    group_dict = {}
    param_keys = []
    for key, value in stats.items():
        # parse stats from a set of parameters
        grid_stats = np.array(value[0])  # raw stats
        # last parameter is given separately since it is actively varying
        last_param_vals, last_param_key, parent_params = value[1:]
        if not headers:
            # set up headers for each stat and insert parameter headers
            # at the start
            headers = [
                GridSearchStats.PARAM.value,
                GridSearchStats.PPV,
                GridSearchStats.SENS,
                GridSearchStats.POS,
                GridSearchStats.TP,
                GridSearchStats.FP,
                GridSearchStats.FDR,
            ]
            headers[0] = "_".join((headers[0], last_param_key))
            for i, parent in enumerate(parent_params.keys()):
                headers.insert(
                    i, "_".join((GridSearchStats.PARAM.value, parent)))
                param_keys.append(parent)
            param_keys.append(last_param_key)
        # false discovery rate, inverse of PPV, since don't have true negs
        fdr = np.subtract(
            1, np.divide(grid_stats[:, 1], 
                         np.add(grid_stats[:, 1], grid_stats[:, 2])))
        sens = np.divide(grid_stats[:, 1], grid_stats[:, 0])
        for i, n in enumerate(last_param_vals):
            stat_list = []
            for parent_val in parent_params.values():
                stat_list.append(parent_val)
            stat_list.extend(
                (last_param_vals[i], 1 - fdr[i], sens[i], 
                 *grid_stats[i].astype(int), fdr[i]))
            for header, stat in zip(headers, stat_list):
                stats_for_df.setdefault(header, []).append(stat)
        group_dict[key] = (fdr, sens, last_param_vals)
    print()
    
    # generate a data frame to summarize stats and save to file
    path_df = libmag.make_out_path(
        "gridsearch_{}.csv".format("_".join(param_keys)))
    df = df_io.dict_to_data_frame(stats_for_df, path_df, show=" ")
    return group_dict, df

    
if __name__ == "__main__":
    print("MagellanMapper machine learning")
