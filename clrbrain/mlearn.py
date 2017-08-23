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
    # plot ROC curve
    from clrbrain import plot_2d
    plot_2d.plot_roc(stats_dict, "roc")

    
if __name__ == "__main__":
    print("Clrbrain machine learning")