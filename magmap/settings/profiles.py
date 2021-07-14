# Profile settings
# Author: David Young, 2019, 2020
"""Profile settings to setup common configurations.

Each profile has a default set of settings, which can be modified through 
"modifier" sub-profiles with groups of settings that overwrite the 
given default settings. 
"""
from enum import Enum, auto
import glob
import os
import pprint
from typing import Any, Dict, Optional, Union

from magmap.io import yaml_io
from magmap.settings import config

_logger = config.logger.getChild(__name__)


class RegKeys(Enum):
    """Register setting enumerations."""
    ACTIVE = auto()
    MARKER_EROSION = auto()
    MARKER_EROSION_MIN = auto()
    MARKER_EROSION_USE_MIN = auto()
    SKELETON_EROSION = auto()
    WATERSHED_MASK_FILTER = auto()
    SAVE_STEPS = auto()
    EDGE_AWARE_REANNOTATION = auto()
    METRICS_CLUSTER = auto()
    DBSCAN_EPS = auto()
    DBSCAN_MINPTS = auto()
    KNN_N = auto()


#: dict: Dictionary mapping the names of Enums used in profiles to their Enum
# classes for parsing Enums given as strings.
_PROFILE_ENUMS = {
    "RegKeys": RegKeys,
    "Cmaps": config.Cmaps,
    "SmoothingModes": config.SmoothingModes,
    "MetricGroups": config.MetricGroups,
    "LoadIO": config.LoadIO,
}


class SettingsDict(dict):
    """Profile dictionary, which contains collections of settings and allows
    modification by applying additional groups of settings specified in
    this dictionary.

    Attributes:
        PATH_PROFILES (str): Path to profiles directory.
        NAME_KEY (str): Key for profile name.
        DEFAULT_NAME (str): Default profile modifier name.
        profiles (dict): Dictionary of profiles to modify the default
            values, where each key is the profile name and the value
            is a nested dictionary that will overwrite or update the
            current values.
        timestamps (dict): Dictionary of profile files to last modified time.
        delimiter (str): Profile names delimiter; defaults to ``,``.

    """
    PATH_PROFILES = "profiles"
    _EXT_YAML = (".yml", ".yaml")
    NAME_KEY = "settings_name"
    DEFAULT_NAME = "default"

    def __init__(self, *args, **kwargs):
        """Initialize a settings dictionary.

        Args:
            *args:
            **kwargs:
        """
        super().__init__(self)
        self[self.NAME_KEY] = "default"
        self.profiles: Dict[str, Dict] = {}
        self.timestamps = {}
        self.delimiter = ","

        #: bool: add a modifier directly as a value rather than updating
        # this dict's settings with the corresponding keys
        self._add_mod_directly = False

    @staticmethod
    def get_files(profiles_dir=None, filename_prefix=""):
        """Get profile files.

        Args:
            profiles_dir (str): Directory from which to get files; defaults
                to None to use :const:`PATH_PROFILES`.
            filename_prefix (str): Only get files starting with this string;
                defaults to an empty string.

        Returns:
            List[str]: List of files in ``profiles_dir`` matching the given
            ``filename_prefix`` and ending with an extension in
            :const:`_EXT_YAML`.

        """
        if not profiles_dir:
            profiles_dir = SettingsDict.PATH_PROFILES
        paths = glob.glob(os.path.join(
            profiles_dir, "{}*".format(filename_prefix)))
        return [p for p in paths
                if os.path.splitext(p)[1].lower() in SettingsDict._EXT_YAML]
    
    def modify_settings(self, mods: Dict[Union[str, Enum], Union[Dict, str]]):
        """Modify dictionary items from another dictionary.
        
        If corresponding values are sub-dictionaries, the existing sub-dict
        will be updated rather than replaced with the new sub-dict.
        
        Args:
            mods: Dictionary to update this class' dictionary.

        """
        for key in mods.keys():
            if isinstance(self[key], dict) and isinstance(mods[key], dict):
                # if both current and new setting values are dicts,
                # update rather than replacing the current dict
                self[key].update(mods[key])
            else:
                # replace the value at the setting with the modified val
                self[key] = mods[key]
    
    def get_profile(
            self, profile_name: str
    ) -> Optional[Dict[Union[str, Enum], Union[Dict, str]]]:
        """Get the dictionary for a given profile.
        
        Profiles may either match an existing profile in :attr:`profiles`
        or specify a path to a YAML configuration file. YAML filenames will
        first be checked in :const:`PATH_PROFILES`, followed by
        ``profile_name`` as the full path. If :attr:`_add_mod_directly` is True,
        the value at ``profile_name`` will be added or replaced with the found
        modifier value.
        
        The profile in :attr:`profiles` whose name matches ``profile_name``
        will be applied over the current settings. If both the current and new
        values are dictionaries, the current dictionary will be updated
        with the new values. Otherwise, the corresponding current value will be
        replaced by the new value.
        
        Args:
            profile_name: Profile name.

        Returns:
            The loaded dictionary, or None if not found.

        """
        if os.path.splitext(profile_name)[1].lower() in self._EXT_YAML:
            # load YAML files from profiles directory
            prof_path = os.path.join(self.PATH_PROFILES, profile_name)
            if not os.path.exists(prof_path):
                # fall back to loading directly from given path
                print("{} profile file not found, checking {}"
                      .format(prof_path, profile_name))
                prof_path = profile_name
                if not os.path.exists(prof_path):
                    print(prof_path, "profile file not found, skipped")
                    return None
            self.timestamps[prof_path] = os.path.getmtime(prof_path)
            mods = {}
            try:
                yamls = yaml_io.load_yaml(prof_path, _PROFILE_ENUMS)
                for yaml in yamls:
                    mods.update(yaml)
                _logger.info("Loaded profile from '%s':\n%s", prof_path, mods)
            except FileNotFoundError:
                _logger.warn("Unable to load profile from: %s", prof_path)
        else:
            if profile_name == self.DEFAULT_NAME:
                # update entries from a new instance for default values;
                # use class name to access any profile subclasses
                mods = self.__class__()
            else:
                # must match an available modifier name
                if profile_name not in self.profiles:
                    print(profile_name, "profile not found, skipped")
                    return None
                mods = self.profiles[profile_name]
        return mods
    
    def add_profile(
            self, profile_name: str,
            mods: Optional[Dict[Union[str, Enum], Union[Dict, str]]]):
        """Add a profile dictionary into this dictionary.
        
        The profile can consist of a subset of keys in this dictionary that
        will override the current values of the corresponding keys.
        If both the original and new value are dictionaries for any given key,
        the original dictionary will be updated with rather than overwritten
        by the new value.

        Args:
            profile_name: Name of the modifier, which will be appended to
                the name of the current settings.
            mods: Dictionary with which to update this instance.
        
        """
        self[self.NAME_KEY] += self.delimiter + profile_name
        if self._add_mod_directly:
            # add/replace the value at mod_name with the found value
            self[profile_name] = mods
        else:
            self.modify_settings(mods)

    def add_profiles(self, names_str):
        """Add profiles by names and files.
        
        Layers profiles on top of one another so that any settings in the
        next profile take precedence over those in the prior profiles.
        For example, "lightsheet_5x" will give one profile, while
        "lightsheet_5x_contrast" will layer additional settings on top of the
        original lightsheet profile.

        Args:
            names_str (str): The name of the settings profile to apply,
                with individual profiles separated by ",". Profiles will
                be applied in order of appearance.
        """
        profiles = names_str.split(self.delimiter)

        for profile in profiles:
            # update self with any combo of profiles, where the order of
            # profiles determines the precedence of settings
            mods = self.get_profile(profile)
            if mods:
                self.add_profile(profile, mods)

        if config.verbose:
            _logger.debug("settings for '%s':", self[self.NAME_KEY])
            _logger.debug(pprint.pprint(self))

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
            profile_names = self[self.NAME_KEY]
            self.__init__()
            self.add_profiles(profile_names)

    @staticmethod
    def is_identical_settings(profs, keys):
        """Check whether the given settings are identical across profiles.

        Args:
            profs (Sequence[:class:`ROIProfile`]): Sequence of ROI profiles.
            keys (Sequence[str]): Sequence of setting keys to check.

        Returns:
            bool: True if the settings are identical, otherwise False.

        """
        prof_first = None
        for prof in profs:
            if prof_first is None:
                # will compare to first profile
                prof_first = prof
            else:
                for key in keys:
                    if prof_first[key] != prof[key]:
                        # any non-equal setting means profiles do not have
                        # identical block settings
                        print("Block settings are not identical")
                        return False
        print("Block settings are identical")
        return True
