#!/bin/bash
# Machine learning for Clrbrain
# Author: David Young, 2017
"""Machine learning and output for Clrbrain.
"""

import numpy as np

from clrbrain import config

def grid_search(fnc, *fnc_args):
    # gets the ROC settings
    settings = config.process_settings
    stats_dict = {}
    file_summaries = []
    for key, value in config.roc_dict.items():
        # group of settings, where key is the name of the group, and 
        # value is another dictionary with the group's settings
        iterable_keys = []
        for key2, value2 in value.items():
            if np.isscalar(value2):
                # set scalar values rather than iterating and processing
                settings[key2] = value2
                print("changed {} to {}".format(key2, value2))
            else:
                iterable_keys.append(key2)
                
        def grid_iterate(i, iterable_keys, grid_dict, name):
            key = iterable_keys[i]
            name = "-".join((name, key))
            stats = []
            if i < len(iterable_keys) - 1:
                for j in grid_dict[key]:
                    settings[key] = j
                    paren_i = name.rfind("(")
                    if paren_i != -1:
                        name = name[:paren_i]
                    name += "(" + str(j) + ")"
                    grid_iterate(i + 1, iterable_keys, grid_dict, name)
            else:
                # process each value in parameter array
                stats = []
                params = grid_dict[key]
                for param in params:
                    print("Processing with settings {} for {}"
                          .format(name, param))
                    settings[key] = param
                    stat, summaries = fnc(*fnc_args)
                    stats.append(stat)
                    file_summaries.extend(summaries)
                stats_dict[name] = (stats, params)
        
        grid_iterate(0, iterable_keys, value, key)
    # summary of each file collected together
    for summary in file_summaries:
        print(summary)
    return stats_dict

def parse_grid_stats(stats_dict):
    """Parses stats from a grid search.
    
    Params:
        stats_dict: Dictionary where key is a string with the parameters
            up to the last parameter group, and each value is a tuple of 
            the raw stats as (pos, true_pos, false_pos) and the array of
            parameter values.
    """
    label = ""
    colori = 0
    align = ">"
    parsed_stats = {}
    for key, value in stats_dict.items():
        stats = np.array(value[0])
        params = value[1]
        # false discovery rate, the inverse of PPV, since don't have a true negs
        fdr = np.subtract(1, np.divide(stats[:, 1], 
                                       np.add(stats[:, 1], stats[:, 2])))
        sens = np.divide(stats[:, 1], stats[:, 0])
        #print(fdr, sens)
        colori += 1
        print("{}:".format(key))
        headers = ("Param", "PPV", "Sens", "Pos", "TP", "FP")
        for header in headers:
            print("{:{align}{fill}}".format(header, fill=8, align=align), 
                  end=" ")
        print()
        for i, n in enumerate(params):
            stat = (params[i], 1 - fdr[i], sens[i], *stats[i].astype(int))
            for val in stat:
                if isinstance(val, (int, np.integer)):
                    print("{:{align}8}".format(val, align=align), end=" ")
                else:
                    print("{:{align}{fill}}".format(
                        val, fill="8.3f", align=align), end=" ")
            print()
        parsed_stats[key] = (fdr, sens, params)
    return parsed_stats

    
if __name__ == "__main__":
    print("Clrbrain machine learning")