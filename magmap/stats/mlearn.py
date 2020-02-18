#!/bin/bash
# Machine learning for MagellanMapper
# Author: David Young, 2017, 2019
"""Machine learning and output for MagellanMapper.
"""

from collections import OrderedDict
from enum import Enum

import numpy as np

from magmap.settings import config
from magmap.io import libmag
from magmap.io import df_io


class GridSearchStats(Enum):
    """Grid search statistics categories."""
    PARAM = "Par"
    PPV = "PPV"
    SENS = "Sens"
    POS = "Pos"
    TP = "TP"
    FP = "FP"
    FDR = "FDR"


def grid_search(roc_dict, keys, fnc, *fnc_args):
    """Perform a grid search for hyperparameter optimization.

    Multiple grid searches can be performed by specifying multiples ``keys``.
    Note that currently each subsequent grid search will use the last
    settings from the prior search.

    Args:
        roc_dict (dict): Nested dictionary, where each sub-dictionary
            contains sequences of the format:
            ``(:class:`profiles.ProcessSettings` parameter, (a, b, c, ...))``.
        keys (List[str]): Sequence of keys in ``roc_dict`` to specify the
            hyperparameters for the grid search(es).
        fnc (func): Function to call during the grid search, which must
            return ``stats, summaries``.
        *fnc_args: Arguments to pass to ``fnc``.

    Returns:
        :dict: Dictionary of stats suitable for parsing in
        :meth:`parse_grid_stats`.

    """
    # gets the ROC settings
    settings = config.process_settings
    stats_dict = OrderedDict()
    file_summaries = []
    for key in keys:
        # perform grid search based on the given dict of settings in roc_dict
        # TODO: consider whether to reset settings between grid searches
        roc = roc_dict[key]
        iterable_keys = []  # hyperparameters to iterate through
        iterable_dict = OrderedDict() # group results
        for key2, value2 in roc.items():
            if np.isscalar(value2):
                # set scalar values rather than iterating and processing
                settings[key2] = value2
                print("changed {} to {}".format(key2, value2))
            else:
                print("adding iterable setting {}".format(key2))
                iterable_keys.append(key2)
                
        def grid_iterate(i, iterable_keys, grid_dict, name, parent_params):
            key = iterable_keys[i]
            name = key if name is None else name + "-" + key
            print("name: {}".format(name))
            stats = []
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
                        i + 1, iterable_keys, grid_dict, name, parent_params)
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
        
        grid_iterate(0, iterable_keys, roc, None, OrderedDict())
        stats_dict[key] = iterable_dict
    # summary of each file collected together
    for summary in file_summaries:
        print(summary)
    return stats_dict


def parse_grid_stats(stats_dict):
    """Parses stats from a grid search.
    
    Args:
        stats_dict: Dictionary where key is a string with the parameters
            up to the last parameter group, and each value is a tuple of 
            the raw stats as (pos, true_pos, false_pos); the array of
            values for the last parameter; the last parameter key; and an 
            ``OrderedDict`` of the parent parameters and their values for 
            the given set of stats.
    """
    parsed_stats = {}
    dfs = []
    param_keys = []
    for group, iterable_dicts in stats_dict.items():
        stats_for_df = {}
        headers = None
        print("{}:".format(group))
        group_dict = {}
        parsed_stats[group] = group_dict
        for key, value in iterable_dicts.items():
            grid_stats = np.array(value[0])
            last_param_vals, last_param_key, parent_params = value[1:]
            if not headers:
                headers = [e.value for e in GridSearchStats]
                headers[0] = "_".join((headers[0], last_param_key))
                for parent in parent_params.keys():
                    headers.insert(
                        0, "_".join((GridSearchStats.PARAM.value, parent)))
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
        path_df = "gridsearch_{}.csv".format("_".join(param_keys))
        dfs.append(df_io.dict_to_data_frame(stats_for_df, path_df, show=" "))
    return parsed_stats, dfs

    
if __name__ == "__main__":
    print("MagellanMapper machine learning")
