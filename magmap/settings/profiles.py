# Profile settings
# Author: David Young, 2019, 2020
"""Profile settings to setup common configurations.

Each profile has a default set of settings, which can be modified through 
"modifier" sub-profiles with groups of settings that overwrite the 
given default settings. 
"""
from enum import Enum, auto
import os

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

    def add_modifier(self, mod_name, profiles, sep):
        """Add a modifer dictionary, overwriting any existing settings 
        with values from this dictionary.
        
        If both the original and new setting are dictionaries, the original
        dictionary will be updated with rather than overwritten by the
        new setting.

        The modifier may either match an existing profile in ``profiles``
        or specify a path to a YAML configuration file. YAML filenames will
        first be checked in :const:`config.PATH_PROFILES`, followed by
        ``mod_name`` as the full path.
        
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
            sep (str): Separator between modifier elements.
        """
        if os.path.splitext(mod_name)[1].lower() in (".yml", ".yaml"):
            # load YAML files from profiles directory
            mod_path = os.path.join(
                config.PATH_PROFILES, os.path.basename(mod_name))
            if not os.path.exists(mod_path):
                # fall back to loading from given path
                print("{} profile file not found, checking {}"
                      .format(mod_path, mod_path))
                mod_path = mod_name
                if not os.path.exists(mod_path):
                    print(mod_path, "profile file not found, skipped")
                    return
            self.timestamps[mod_path] = os.path.getmtime(mod_path)
            yamls = yaml_io.load_yaml(mod_path, _PROFILE_ENUMS)
            mods = {}
            for yaml in yamls:
                mods.update(yaml)
            print("loaded {}:\n{}".format(mod_path, mods))
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
                with individual profiles separated by ",". Profiles will
                be applied in order of appearance.
        """
        sep = ","
        profiles = names_str.split(sep)

        for profile in profiles:
            # update default profile with any combo of modifiers, where the
            # order of the profile listing determines the precedence of settings
            self.add_modifier(profile, self.profiles, sep)

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
