# Profile settings
# Author: David Young, 2019, 2020
"""Profile settings to setup common configurations.

Each profile has a default set of settings, which can be modified through 
"modifier" sub-profiles with groups of settings that overwrite the 
given default settings. 
"""
from collections import OrderedDict
from enum import Enum, auto
import os

import numpy as np

from magmap.io import yaml_io
from magmap.settings import config


class RegKeys(Enum):
    """Register setting enumerations."""
    ACTIVE = auto()
    MARKER_EROSION = auto()
    MARKER_EROSION_MIN = auto()
    MARKER_EROSION_USE_MIN = auto()
    SAVE_STEPS = auto()
    EDGE_AWARE_REANNOTAION = auto()
    METRICS_CLUSTER = auto()
    DBSCAN_EPS = auto()
    DBSCAN_MINPTS = auto()
    KNN_N = auto()


class PreProcessKeys(Enum):
    """Pre-processing task enumerations."""
    SATURATE = auto()
    DENOISE = auto()
    REMAP = auto()
    ROTATE = auto()


#: dict: Dictionary mapping the names of Enums used in profiles to their Enum
# classes for parsing Enums given as strings.
_PROFILE_ENUMS = {
    "RegKeys": RegKeys,
    "Cmaps": config.Cmaps,
    "SmoothingModes": config.SmoothingModes,
    "MetricGroups": config.MetricGroups,
    "PreProcessKeys": PreProcessKeys,
    "LoadIO": config.LoadIO,
}


class SettingsDict(dict):
    """Profile dictionary, which contains collections of settings and allows
    modification by applying additional groups of settings specified in
    this dictionary.

    Attributes:
        profiles (dict): Dictionary of profiles to modify the default
            values, where each key is the profile name and the value
            is a nested dictionary that will overwrite or update the
            current values.
        timestamps (dict): Dictionary of profile files to last modified time.

    """

    def __init__(self, *args, **kwargs):
        """Initialize a settings dictionary.

        Args:
            *args:
            **kwargs:
        """
        super().__init__(self)
        self["settings_name"] = "default"
        self.profiles = {}
        self.timestamps = {}

    def add_modifier(self, mod_name, profiles, sep="_"):
        """Add a modifer dictionary, overwriting any existing settings 
        with values from this dictionary.
        
        If both the original and new setting are dictionaries, the original
        dictionary will be updated with rather than overwritten by the
        new setting.
        
        Args:
            mod_name (str): Name of the modifier, which will be appended to
                the name of the current settings.
            profiles (dict): Profiles dictionary, where each key is a profile
                name and value is a profile as a nested dictionary. The
                profile whose name matches ``mod_name`` will be applied
                over the current settings. If both the current and new values
                are dictionaries, the current dictionary will be updated
                with the new values. Otherwise, the corresponding current
                value will be replaced by the new value.
            sep (str): Separator between modifier elements. Defaults to "_".
        """
        if os.path.splitext(mod_name)[1].lower() in (".yml", ".yaml"):
            if not os.path.exists(mod_name):
                print(mod_name, "profile file not found, skipped")
                return
            self.timestamps[mod_name] = os.path.getmtime(mod_name)
            yamls = yaml_io.load_yaml(mod_name, _PROFILE_ENUMS)
            mods = {}
            for yaml in yamls:
                mods.update(yaml)
            print("loaded {}:\n{}".format(mod_name, mods))
        else:
            # if name to check is given, must match modifier name to continue
            if mod_name not in profiles:
                print(mod_name, "profile not found, skipped")
                return
            mods = profiles[mod_name]
        self["settings_name"] += sep + mod_name
        for key in mods.keys():
            if isinstance(self[key], dict) and isinstance(mods[key], dict):
                # update if both are dicts
                self[key].update(mods[key])
            else:
                # replace with modified setting
                self[key] = mods[key]

    def update_settings(self, names_str):
        """Update processing profiles, including layering modifications upon
        existing base layers.

        For example, "lightsheet_5x" will give one profile, while
        "lightsheet_5x_contrast" will layer additional settings on top of the
        original lightsheet profile.

        Args:
            names_str (str): The name of the settings profile to apply,
                with individual profiles separated by "_". Profiles will
                be applied in order of appearance.
        """
        profiles = names_str.split("_")

        for profile in profiles:
            # update default profile with any combo of modifiers, where the
            # order of the profile listing determines the precedence of settings
            self.add_modifier(profile, self.profiles)

        if config.verbose:
            print("settings for {}:\n{}".format(self["settings_name"], self))

    def check_file_changed(self):
        """Check whether any profile files have changed since last loaded.

        Returns:
            bool: True if any file has changed.

        """
        for key, val in self.timestamps.items():
            if val < os.path.getmtime(key):
                return True
        return False

    def refresh_profile(self, check_timestamp=False):
        """Refresh the profile.

        Args:
            check_timestamp (bool): True to refresh only if a loaded
                profile file has changed; defaults to False.

        """
        if not check_timestamp or self.check_file_changed():
            # applied profiles are stored in the settings name
            profile_names = self["settings_name"]
            self.__init__()
            self.update_settings(profile_names)


def make_hyperparm_arr(start, stop, num_steps, num_col, coli, base=1):
    """Make a hyperparameter 2D array that varies across the first axis
    for the given index.

    The 2D array is used for grid searches, where each row is given as a
    parameter. Each parameter is a 1D array with the same values except
    at a given index, which varies across these 1D arrays. The varying
    values are constructed by :meth:`np.linspace`.

    Args:
        start (int, float): Starting value for varying parameter.
        stop (int, float): Ending value for varying parameter, inclusive.
        num_steps (int): Number of steps from ``start`` to ``stop``, which
            determines the number of rows in the output array.
        num_col (int): Number of columns in the output array.
        coli (int): Index of column to vary.
        base (int, float): All values are set to this number except for the
            varying values.

    Returns:
        :obj:`np.ndarray`: 2D array in the format ``[[start, base, base, ...],
        [start0, base, base, ...], ...]``.

    """
    steps = np.linspace(start, stop, num_steps)
    arr = np.ones((len(steps), num_col)) * base
    arr[:, coli] = steps
    return arr


#: OrderedDict[List[int]]: Nested dictionary where each sub-dictionary
# contains a sequence of values over which to perform a grid search to
# generate a receiver operating characteristic curve
roc_dict = OrderedDict([
    ("test", OrderedDict([
        # test single value by iterating on value that should not affect
        # detection ability
        ("points_3d_thresh", [0.7]),

        # unfused baseline
        #("clip_vmax", 98.5),
        #("clip_max", 0.5),
        #("clip_vmax", np.arange(98.5, 99, 0.5)),
        #("clip_max", np.arange(0.5, 0.6, 0.1)),

        # test parameters
        #("isotropic", make_hyperparm_arr(0.2, 1, 9, 3, 0),
        #("isotropic", np.array([(0.96, 1, 1)])),
        #("overlap", np.arange(0.1, 1.0, 0.1)),
        #("prune_tol_factor", np.array([(4, 1.3, 1.3)])),
        #("prune_tol_factor", make_hyperparm_arr(0.5, 1, 2, 0.9, 0)),
        #("clip_min", np.arange(0.0, 0.2, 0.1)),
        #("clip_vmax", np.arange(97, 100.5, 0.5)),
        #("clip_max", np.arange(0.3, 0.7, 0.1)),
        #("erosion_threshold", np.arange(0.16, 0.35, 0.02)),
        #"denoise_size", np.arange(5, 25, 2)
        #("unsharp_strength", np.arange(0.0, 1.1, 0.1)),
        #("tot_var_denoise", (False, True)),
        #("num_sigma", np.arange(5, 16, 1)),
        #("detection_threshold", np.arange(0.001, 0.01, 0.001)),
        #("segment_size", np.arange(130, 160, 20)),
    ])),
    ("size5x", OrderedDict([
        ("min_sigma_factor", np.arange(2, 2.71, 0.1)),
        ("max_sigma_factor", np.arange(2.7, 3.21, 0.1)),
    ])),
    ("size4x", OrderedDict([
        ("min_sigma_factor", np.arange(2.5, 3.51, 0.3)),
        ("max_sigma_factor", np.arange(3.5, 4.51, 0.3)),
    ])),
    ("sizeiso", OrderedDict([
        ("min_sigma_factor", np.arange(2, 3.1, 1)),
        ("max_sigma_factor", np.arange(3, 4.1, 1)),
        ("isotropic", make_hyperparm_arr(0.2, 1, 9, 3, 0)),
    ])),
])
