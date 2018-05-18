# Stats for Clrbrain
# Author: David Young, 2018
"""Stats calculations and text output for Clrbrain.

Attributes:
"""

import copy
import numpy as np
from scipy import stats

from clrbrain import config

def _volumes_mean_sem(group_dict, key_mean, key_sem, vals, mask):
    """Calculate the mean and standard error of the mean (SEM), storing them 
    in the dictionary.
    
    Values are filtered by ``mask``, and empty (ie near-zero or None) volumes 
    are excluded as well.
    
    Args:
        group_dict: Dictionary where the SEM will be stored.
        key_mean: Key at which the mean will be stored.
        key_sem: Key at which the SEM will be stored.
        vals: Values from which to calculate.
        mask: Boolean array corresponding to ``vals`` of values to keep.
    """
    # convert to Numpy array to filter by make
    vals = np.array(vals)
    if mask is not None:
        vals = vals[mask]
    print("group vals raw: {}, mask: {}, n: {}".format(vals, mask, vals.size))
    
    # further prune to remove None or near-zero values (ie no volume found)
    vals = vals[vals != None] # TODO: check if actually encounter None vals
    vals = vals[vals > config.POS_THRESH]
    mean = np.mean(vals)
    sem = stats.sem(vals)
    print("mean: {}, err: {}, n after pruning: {}".format(mean, sem, vals.size))
    group_dict[key_mean].append(mean)
    group_dict[key_sem].append(sem)

def volume_stats(volumes_dict, densities, groups=[""], unit_factor=1.0):
    # "side" and "mirrored" for opposite side (R/L agnostic)
    SIDE = "side"
    MIR = "mirrored"
    SIDE_SEM = SIDE + "_sem"
    MIR_SEM = MIR + "_sem"
    VOL = "volume"
    DENS = "density"
    multiple = groups is not None
    groups_unique = np.unique(groups)
    groups_dict = {}
    for group in groups_unique:
        print("Finding volumes and densities for group {}".format(group))
        # dictionary of mean and SEM arrays for each side, which will be 
        # populated in same order as experiments in volumes_dict
        vol_group = {SIDE: [], MIR: [], SIDE_SEM: [], MIR_SEM: []}
        dens_group = copy.deepcopy(vol_group)
        groups_dict[group] = {VOL: vol_group, DENS: dens_group}
        group_mask = np.array(groups) == group if multiple else None
        for key in volumes_dict.keys():
            # find negative keys based on the given positive key to show them
            # side-by-side
            if key >= 0:
                # get volumes in the given unit, which are scalar for 
                # individual image, list if multiple images
                vol_side = np.divide(
                    volumes_dict[key][config.VOL_KEY], unit_factor)
                vol_mirrored = np.divide(
                    volumes_dict[-1 * key][config.VOL_KEY], unit_factor)
                # store vol and SEMs in vol_group
                if isinstance(vol_side, np.ndarray):
                    # for multiple experiments, store mean and error
                    _volumes_mean_sem(
                        vol_group, SIDE, SIDE_SEM, vol_side, group_mask)
                    _volumes_mean_sem(
                        vol_group, MIR, MIR_SEM, vol_mirrored, group_mask)
                else:
                    # for single experiment, store only vol
                    vol_group[SIDE].append(vol_side)
                    vol_group[MIR].append(vol_mirrored)
                
                if densities:
                    # calculate densities based on blobs counts and volumes
                    blobs_side = volumes_dict[key][config.BLOBS_KEY]
                    blobs_mirrored = volumes_dict[-1 * key][config.BLOBS_KEY]
                    print("id {}: blobs R {}, L {}".format(
                        key, blobs_side, blobs_mirrored))
                    density_side = np.nan_to_num(
                        np.divide(blobs_side, vol_side))
                    density_mirrored = np.nan_to_num(
                        np.divide(blobs_mirrored, vol_mirrored))
                    if isinstance(density_side, np.ndarray):
                        # density means and SEMs, storing the SEMs
                        _volumes_mean_sem(
                            dens_group, SIDE, SIDE_SEM, density_side, 
                            group_mask)
                        _volumes_mean_sem(
                            dens_group, MIR, MIR_SEM, density_mirrored, 
                            group_mask)
                    else:
                        dens_group[SIDE].append(density_side)
                        dens_group[MIR].append(density_mirrored)
    names = [volumes_dict[key][config.ABA_NAME] 
             for key in volumes_dict.keys() if key >= 0]
    return groups_dict, names, (MIR, SIDE), (MIR_SEM, SIDE_SEM), (VOL, DENS)

if __name__ == "__main__":
    print("Starting Clrbrain stats...")
