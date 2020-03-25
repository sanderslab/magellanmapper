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


#: dict: Dictionary mapping the names of Enums used in profiles to their Enum
# classes for parsing Enums given as strings.
_PROFILE_ENUMS = {
    "RegKeys": RegKeys,
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


class ProcessSettings(SettingsDict):

    def __init__(self, *args, **kwargs):
        super().__init__(self)
        self["settings_name"] = "default"

        # visualization and plotting

        self["vis_3d"] = "points"  # "points" or "surface" 3D visualization
        self["points_3d_thresh"] = 0.85  # frac of thresh (changed in v.0.6.6)
        self["channel_colors"] = (
            config.Cmaps.CMAP_GRBK_NAME, config.Cmaps.CMAP_RDBK_NAME)
        self["scale_bar_color"] = "w"
        self["colorbar"] = False
        # num of times to rotate image by 90deg after loading
        self["load_rot90"] = 0
        self["norm"] = None  # (min, max) normalization of image5d

        # image preprocessing before blob detection

        self["clip_vmin"] = 5  # vmin/vmax set contrast stretching, range 0-100
        self["clip_vmax"] = 99.5
        self["clip_min"] = 0.2  # min/max clip after stretching, range 0-1
        self["clip_max"] = 1.0
        # mult by config.near_max for lower threshold of global max
        self["max_thresh_factor"] = 0.5
        self["tot_var_denoise"] = None  # weight (eg skimage default 0.1)
        self["unsharp_strength"] = 0.3  # unsharp filter (sharpens images)
        self["erosion_threshold"] = 0.2  # erode clumped cells

        # 3D blob detection settings

        self["min_sigma_factor"] = 3
        self["max_sigma_factor"] = 5
        self["num_sigma"] = 10
        self["detection_threshold"] = 0.1
        self["overlap"] = 0.5
        self["thresholding"] = None
        self["thresholding_size"] = -1
        # z,y,x px to exclude along border after blob detection
        self["exclude_border"] = None

        # block processing and automated verification

        # multiprocessing start method; if method not available for the given
        # platform, the default method for the platform will be used instead
        self["mp_start"] = "fork"  # fork, spawn, or forkserver
        self["segment_size"] = 500  # detection ROI max size along longest edge
        # max size along longest edge for denoising blocks within
        # segmentation blobs; None turns off preprocessing in stack proc;
        # make much larger than segment_size (eg 2x) to cover the full segment
        # ROI because of additional overlap in segment ROIs
        self["denoise_size"] = 25
        # z,y,x tolerances for pruning duplicates in overlapped regions
        self["prune_tol_factor"] = (1, 1, 1)
        self["verify_tol_factor"] = (1, 1, 1)
        # module level variable will take precedence
        self["sub_stack_max_pixels"] = (1000, 1000, 1000)

        # resizing for anisotropy

        self["isotropic"] = None  # final relative z,y,x scaling after resizing
        self["isotropic_vis"] = None  # z,y,x scaling factor for vis only
        self["resize_blobs"] = None  # z,y,x coord scaling before verification

        self.profiles = {

            # Lightsheet nuclei
            # pre-v01
            # v1 (MagellanMapper v0.6.1)
            # v2 (MagellanMapper v0.6.2): isotropy (no anisotropic detection), dec
            #     clip_max, use default sub_stack_max_pixels
            # v2.1 (MagellanMapper v0.6.4): erosion_threshold
            # v2.2 (MagellanMapper v0.6.6): narrower and taller stack shape
            # v2.3 (MagellanMapper v0.8.7): added prune_tol_factor
            # v2.4 (MagellanMapper v0.8.8): decreased min/max sigma, segment size
            # v2.5 (MagellanMapper v0.8.9): added exclude_border
            # v2.6 (MagellanMapper v0.9.3): slight dec in x/y verify tol for
            #     Hungarian method
            # v2.6.1 (MagellanMapper v0.9.4): scale_factor, segmenting_mean_thresh
            #     had already been turned off and now removed completely
            "lightsheet": {
                "points_3d_thresh": 0.7,
                "clip_vmax": 98.5,
                "clip_min": 0,
                "clip_max": 0.5,
                "unsharp_strength": 0.3,
                "erosion_threshold": 0.3,
                "min_sigma_factor": 2.6,
                "max_sigma_factor": 2.8,
                "num_sigma": 10,
                "overlap": 0.55,
                "segment_size": 150,
                "prune_tol_factor": (1, 0.9, 0.9),
                "verify_tol_factor": (3, 1.2, 1.2),
                "isotropic": (0.96, 1, 1),
                "isotropic_vis": (1.3, 1, 1),
                "sub_stack_max_pixels": (1200, 800, 800),
                "exclude_border": (1, 0, 0),
            },

            # minimal preprocessing
            "minpreproc": {
                "clip_vmin": 0,
                "clip_vmax": 99.99,
                "clip_max": 1,
                "tot_var_denoise": 0.01,
                "unsharp_strength": 0,
                "erosion_threshold": 0,
            },

            # low resolution
            "lowres": {
                "min_sigma_factor": 10,
                "max_sigma_factor": 14,
                "isotropic": None,
                "denoise_size": 2000,  # will use full detection ROI
                "segment_size": 1000,
                "max_thresh_factor": 1.5,
                "exclude_border": (8, 1, 1),
                "verify_tol_factor": (3, 2, 2),
            },

            # 2-photon 20x nuclei
            "2p20x": {
                "vis_3d": "surface",
                "clip_vmax": 97,
                "clip_min": 0,
                "clip_max": 0.7,
                "tot_var_denoise": True,
                "unsharp_strength": 2.5,
                # smaller threshold since total var denoising
                # "points_3d_thresh": 1.1
                "min_sigma_factor": 2.6,
                "max_sigma_factor": 4,
                "num_sigma": 20,
                "overlap": 0.1,
                "thresholding": None,  # "otsu"
                # "thresholding_size": 41,
                "thresholding_size": 64,  # for otsu
                # "thresholding_size": 50.0, # for random_walker
                "denoise_size": 25,
                "segment_size": 100,
                "prune_tol_factor": (1.5, 1.3, 1.3),
            },

            # 2p 20x of zebrafish nuclei
            "zebrafish": {
                "min_sigma_factor": 2.5,
                "max_sigma_factor": 3,
            },

            # higher contrast colormaps
            "contrast": {
                "channel_colors": ("inferno", "inferno"),
                "scale_bar_color": "w",
            },

            # similar colormaps to greyscale but with a cool blue tinge
            "bone": {
                "channel_colors": ("bone", "bone"),
                "scale_bar_color": "w",
            },

            # diverging colormaps for heat maps centered on 0
            "diverging": {
                "channel_colors": ("RdBu", "BrBG"),
                "scale_bar_color": "k",
                "colorbar": True,
            },

            # lightsheet 5x of cytoplasmic markers
            "cytoplasm": {
                "clip_min": 0.3,
                "clip_max": 0.8,
                "points_3d_thresh": 0.7,
                # adjust sigmas based on extent of cyto staining;
                # TODO: consider adding sigma_mult if ratio remains
                # relatively const
                "min_sigma_factor": 4,
                "max_sigma_factor": 10,
                "num_sigma": 10,
                "overlap": 0.2,
            },

            # isotropic image that does not require interpolating visually
            "isotropic": {
                "points_3d_thresh": 0.3,  # used only if not surface
                "isotropic_vis": (1, 1, 1),
            },

            # binary image
            "binary": {
                "denoise_size": None,
                "detection_threshold": 0.001,
            },

            # adjust nuclei size for 4x magnification
            "4xnuc": {
                "min_sigma_factor": 3,
                "max_sigma_factor": 4,
            },

            # fit into ~32GB RAM instance after isotropic interpolation
            "20x": {
                "segment_size": 50,
            },

            # export to deep learning framework with required dimensions
            "exportdl": {
                "isotropic": (0.93, 1, 1),
            },

            # downsample an image previously upsampled for isotropy
            "downiso": {
                "isotropic": None,  # assume already isotropic
                "resize_blobs": (.2, 1, 1),
            },

            # rotate by 180 deg
            # TODO: replace with plot labels config setting?
            "rot180": {
                "load_rot90": 2,  # rotation by 180deg
            },

            # denoise settings when performing registration
            "register": {
                "unsharp_strength": 1.5,
            },

            # color and intensity geared toward histology atlas images
            "atlas": {
                "channel_colors": ("gray",),
                "clip_vmax": 97,
            },

            # colors for each channel based on randomly generated discrete colormaps
            "randomcolors": {
                "channel_colors": [],
            },

            # normalize image5d and associated metadata to intensity values
            # between 0 and 1
            "norm": {
                "norm": (0.0, 1.0),
            },

        }


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


class RegisterSettings(SettingsDict):

    def __init__(self, *args, **kwargs):
        super().__init__(self)
        self["settings_name"] = "default"
        
        # registration main similarity metric
        self["metric_similarity"] = "AdvancedMattesMutualInformation"
        # fallback to alternate similarity metric if below DSC threshold as
        # given by (threshold, alternate_metric)
        self["metric_sim_fallback"] = None
        
        self["translation_iter_max"] = "2048"
        self["affine_iter_max"] = "1024"
        self["bspline_iter_max"] = "256"
        self["bspline_grid_space_voxels"] = "50"
        self["grid_spacing_schedule"] = None
        self["groupwise_iter_max"] = "1024"
        self["resize_factor"] = 0.7
        self["preprocess"] = False
        self["point_based"] = False
        self["smooth"] = None  # smooth labels
        self["crop_to_labels"] = False  # crop labels and atlas to non-0 labels
        self["curate"] = True  # carve image; in-paint if generating atlas
        
        # erase labels outside of ((x_start, x_end), (y_start, ...) ...) 
        # (applied after transposition), where each val is given as fractions
        # of the full range or None to not truncate that at that position; 
        # None for the entire setting turns off truncation
        self["truncate_labels"] = None
        
        # labels curation

        # ACTIVE (bool): True to apply the setting to the final image and
        # metrics; False to use only for metrics and cropping, etc
        # start (float): fractions of the total planes (0-1); use -1 to
        # set automatically, None to turn off the entire setting group

        # mirror labels onto the unlabeled hemisphere
        self["labels_mirror"] = {
            RegKeys.ACTIVE: False,
            "start": None,  # reflect planes starting here
            "neg_labels": True,  # invert values of mirrored labels
        }
        # extend edge labels
        self["labels_edge"] = {
            RegKeys.ACTIVE: False,
            RegKeys.SAVE_STEPS: False,
            "start": None,  # start plane index
            "surr_size": 5,  # dilation filter size for finding histology region
            # smoothing filter size to remove artifacts (None or 0 to ignore)
            "smoothing_size": 3,
            "in_paint": True,  # True to fill pxs missing labels
            # erosion filter size for watershed markers (0 to ignore)
            RegKeys.MARKER_EROSION: 10,
            RegKeys.MARKER_EROSION_MIN: None,  # use default size; 0 for no min
            RegKeys.MARKER_EROSION_USE_MIN: False,  # don't erode if reach min
        }
        self["labels_dup"] = None  # start duplicating planes til last labels
        
        # expand labels within bounds given by 
        # (((x_pixels_start, x_pixels_end), ...), (next_region...)), or None 
        # to avoid expansion
        self["expand_labels"] = None
        
        # atlas and labels rotation by ((angle0, axis0), ...), or None to 
        # avoid rotation, with axis numbers in z,y,x ordering
        self["rotate"] = None
        
        # atlas thresholds for microscopy images
        self["atlas_threshold"] = 10.0  # raise for finer segmentation
        self["atlas_threshold_all"] = 10.0  # keep low to include all signal
        
        self["target_size"] = None  # x,y,z in exp orientation
        
        # carving and max size of small holes for removal, respectively
        self["carve_threshold"] = None
        self["holes_area"] = None
        
        # paste in region from first image during groupwise reg; 
        # x,y,z, same format as truncate_labels except in pixels
        self["extend_borders"] = None
        
        # affine transformation as a dict of ``axis_along`` for the axis along 
        # which to perform transformation (ie the planes that will be 
        # affine transformed); ``axis_shift`` for the axis or 
        # direction in which to shear; ``shift`` for a tuple of indices 
        # of starting to ending shift while traveling from low to high 
        # indices along ``axis_along``; ``bounds`` for a tuple of 
        # ``((z_start z_end), (y_start, ...) ...)`` indices (note the 
        # z,y,x ordering to use directly); and an optional ``axis_attach`` 
        # for the axis along which to perform another affine to attach the 
        # main affine back to the rest of the image
        self["affine"] = None
        
        # Laplacian of Gaussian
        self["log_sigma"] = 5  # Gaussian sigma; use None to skip
        # use atlas_threshold on atlas image to generate mask for finding 
        # background rather than using labels and thresholded LoG image, 
        # useful when ventricular spaces are labeled
        self["log_atlas_thresh"] = False
        
        # edge-aware reannotation: labels erosion for watershed seeds/markers
        # in resegmentation; also used to demarcate the interior of regions;
        # can turn on/off with erode_labels
        self[RegKeys.EDGE_AWARE_REANNOTAION] = {
            RegKeys.MARKER_EROSION: 8,  # filter size for labels to markers
            RegKeys.MARKER_EROSION_MIN: 1,  # None for default, 0 for no min
        }
        self["erosion_frac"] = 0.5  # target size as frac of orig; can be None
        self["erode_labels"] = {"markers": True, "interior": False}
        
        # crop labels back to their original background after smoothing 
        # (ignored during atlas import if no smoothing), given as the filter 
        # size used to open up the background before cropping, 0 to use 
        # the original background as-is, or False not to crop
        self["crop_to_orig"] = 1
        
        # type of label smoothing
        self["smoothing_mode"] = config.SmoothingModes.opening
        
        # combine values from opposite sides when measuring volume stats; 
        # default to use raw values for each label and side to generate 
        # a data frame that can be used for fast aggregation when 
        # grouping into levels
        self["combine_sides"] = False
        
        # make the far hemisphere neg if it is not, for atlases (eg P56) with 
        # bilateral pos labels where one half should be made neg for stats
        self["make_far_hem_neg"] = False
        
        # planar orientation for transposition prior rather than after import
        self["pre_plane"] = None
        
        # labels range given as ``((start0, end0), (start1, end1), ...)``, 
        # where labels >= start and < end will be treated as foreground 
        # when measuring overlap, eg labeled ventricles that would be 
        # background in histology image
        self["overlap_meas_add_lbls"] = None
        
        # sequence of :class:`config.MetricGroups` enums to measure in 
        # addition to basic metrics
        self["extra_metric_groups"] = None
        
        # cluster metrics
        self[RegKeys.METRICS_CLUSTER] = {
            RegKeys.KNN_N: 5,  # num of neighbors for k-nearest-neighbors
            RegKeys.DBSCAN_EPS: 20,  # epsilon for max dist in cluster
            RegKeys.DBSCAN_MINPTS: 6,  # min points/samples per cluster
        }

        self.profiles = {

            # more aggressive parameters for finer tuning
            "finer": {
                "bspline_iter_max": "512",
                "truncate_labels": (None, (0.2, 1.0), (0.45, 1.0)),
                "holes_area": 5000,
            },

            # Normalized Correlation Coefficient similarity metric for registration
            "ncc": {
                "metric_similarity": "AdvancedNormalizedCorrelation",
                # fallback to MMI since it has been rather reliable
                "metric_sim_fallback":
                    (0.85, "AdvancedMattesMutualInformation"),
                "bspline_grid_space_voxels": "60",
            },

            # groupwise registration
            "groupwise": {
                # larger bspline voxels to avoid over deformation of internal
                # structures
                "bspline_grid_space_voxels": "130",

                # need to empirically determine
                "carve_threshold": 0.01,
                "holes_area": 10000,

                # empirically determined to add variable tissue area from
                # first image since this tissue may be necessary to register
                # to other images that contain this variable region
                "extend_borders": ((60, 180), (0, 200), (20, 110)),

                # increased num of resolutions with overall increased spacing
                # schedule since it appears to improve internal alignment
                "grid_spacing_schedule": [
                    "8", "8", "4", "4", "4", "2", "2", "2", "1", "1", "1", "1"],
            },

            # test no iterations
            "testnoiter": {
                "translation_iter_max": "0",
                "affine_iter_max": "0",
                "bspline_iter_max": "0",
                "curate": False,
            },

            # test a target size
            "testsize": {
                "target_size": (50, 50, 50),
            },

            # atlas is big relative to the experimental image, so need to
            # more aggressively downsize the atlas
            "big": {
                "resize_factor": 0.625,
            },

            # new atlas generation: turn on preprocessing
            # TODO: likely remove since not using preprocessing currently
            "new": {
                "preprocess": True,
            },

            # registration to new atlas assumes images are roughly same size and
            # orientation (ie transposed) and already have mirrored labels aligned
            # with the fixed image toward the bottom of the z-dimension
            "generated": {
                "resize_factor": 1.0,
                "truncate_labels": (None, (0.18, 1.0), (0.2, 1.0)),
                "labels_mirror": {RegKeys.ACTIVE: False},
                "labels_edge": None,
            },

            # atlas that uses groupwise image as the atlas itself should
            # determine atlas threshold dynamically
            "grouped": {
                "atlas_threshold": None,
            },

            # ABA E11pt5 specific settings
            "abae11pt5": {
                "target_size": (345, 371, 158),
                "resize_factor": None,  # turn off resizing
                "labels_mirror": {RegKeys.ACTIVE: True, "start": 0.52},
                "labels_edge": {RegKeys.ACTIVE: False, "start": None},
                "log_atlas_thresh": True,
                "atlas_threshold": 75,  # avoid over-extension into ventricles
                "atlas_threshold_all": 5,  # include ventricles since labeled
                # rotate axis 0 to open vertical gap for affines (esp 2nd)
                "rotate": ((-5, 1), (-1, 2), (-30, 0)),
                "affine": ({
                               # shear cord opposite the brain back toward midline
                               "axis_along": 1, "axis_shift": 0, "shift": (25, 0),
                               "bounds": ((None, None), (70, 250), (0, 150))
                           }, {
                               # shear distal cord where the tail wraps back on itself
                               "axis_along": 2, "axis_shift": 0, "shift": (0, 50),
                               "bounds": ((None, None), (0, 200), (50, 150))
                           }, {
                               # counter shearing at far distal end, using attachment for
                               # a more gradual shearing along the y-axis to preserve the
                               # cord along that axis
                               "axis_along": 2, "axis_shift": 0, "shift": (45, 0),
                               "bounds": ((None, None), (160, 200), (90, 150)),
                               "axis_attach": 1
                           }),
                "crop_to_labels": True,  # req because of 2nd affine
                "smooth": 2,
                "overlap_meas_add_lbls": ((126651558, 126652059),),
            },

            # pitch rotation only for ABA E11pt to compare corresponding z-planes
            # with other stages of refinement
            "abae11pt5pitch": {"rotate": ((-30, 0),)},

            # ABA E13pt5 specific settings
            "abae13pt5": {
                "target_size": (552, 673, 340),
                "resize_factor": None,  # turn off resizing
                "labels_mirror": {RegKeys.ACTIVE: True, "start": 0.48},
                # small, default surr size to avoid capturing 3rd labeled area
                # that becomes an artifact
                "labels_edge": {
                    RegKeys.ACTIVE: True,
                    "start": -1,
                },
                "atlas_threshold": 55,  # avoid edge over-extension into skull
                "rotate": ((-4, 1), (-2, 2)),
                "crop_to_labels": True,
                "smooth": 2,
            },

            # ABA E15pt5 specific settings
            "abae15pt5": {
                "target_size": (704, 982, 386),
                "resize_factor": None,  # turn off resizing
                "labels_mirror": {RegKeys.ACTIVE: True, "start": 0.49},
                "labels_edge": {
                    RegKeys.ACTIVE: True,
                    "start": -1,
                    "surr_size": 12,
                    # increase template smoothing to prevent over-extension of
                    # intermediate stratum of Str
                    "smoothing_size": 5,
                    # larger to allow superficial stratum of DPall to take over
                    RegKeys.MARKER_EROSION: 19,
                },
                "atlas_threshold": 45,  # avoid edge over-extension into skull
                "rotate": ((-4, 1),),
                "crop_to_labels": True,
                "smooth": 2,
            },

            # ABA E18pt5 specific settings
            "abae18pt5": {
                "target_size": (278, 581, 370),
                "resize_factor": None,  # turn off resizing
                "labels_mirror": {RegKeys.ACTIVE: True, "start": 0.525},
                # start from smallest BG; remove spurious label pxs around
                # medial pallium by smoothing
                "labels_edge": {
                    RegKeys.ACTIVE: True, "start": 0.137, "surr_size": 12,
                    RegKeys.MARKER_EROSION: 12,
                    RegKeys.MARKER_EROSION_USE_MIN: True,
                },
                "expand_labels": (((None,), (0, 279), (103, 108)),),
                "rotate": ((1.5, 1), (2, 2)),
                "smooth": 3,
                RegKeys.EDGE_AWARE_REANNOTAION: {
                    RegKeys.MARKER_EROSION_MIN: 4,
                }
            },

            # ABA P4 specific settings
            "abap4": {
                "target_size": (724, 403, 398),
                "resize_factor": None,  # turn off resizing
                "labels_mirror": {RegKeys.ACTIVE: True, "start": 0.487},
                "labels_edge": {
                    RegKeys.ACTIVE: True,
                    "start": -1,
                    "surr_size": 12,
                    # balance eroding medial pallium and allowing dorsal
                    # pallium to take over
                    RegKeys.MARKER_EROSION: 8,
                },
                # open caudal labels to allow smallest mirror plane index,
                # though still cross midline as some regions only have
                # labels past midline
                "rotate": ((0.22, 1),),
                "smooth": 4,
            },

            # ABA P14 specific settings
            "abap14": {
                "target_size": (390, 794, 469),
                "resize_factor": None,  # turn off resizing
                # will still cross midline since some regions only have labels
                # past midline
                "labels_mirror": {RegKeys.ACTIVE: True, "start": 0.5},
                "labels_edge": {
                    RegKeys.ACTIVE: True,
                    "start": 0.078,  # avoid alar part size jitter
                    "surr_size": 12,
                    RegKeys.MARKER_EROSION: 40,
                    RegKeys.MARKER_EROSION_MIN: 10,
                },
                # rotate conservatively for symmetry without losing labels
                "rotate": ((-0.4, 1),),
                "smooth": 5,
            },

            # ABA P28 specific settings
            "abap28": {
                "target_size": (863, 480, 418),
                "resize_factor": None,  # turn off resizing
                # will still cross midline since some regions only have labels
                # past midline
                "labels_mirror": {RegKeys.ACTIVE: True, "start": 0.48},
                "labels_edge": {
                    RegKeys.ACTIVE: True,
                    "start": 0.11,  # some lat labels only partially complete
                    "surr_size": 12,
                    "smoothing_size": 0,  # no smoothing to avoid loss of detail
                },
                # "labels_dup": 0.48,
                # rotate for symmetry, which also reduces label loss
                "rotate": ((1, 2),),
                "smooth": 2,
            },

            # ABA P56 (developing mouse) specific settings
            "abap56": {
                "target_size": (528, 320, 456),
                "resize_factor": None,  # turn off resizing
                # stained sections and labels almost but not symmetric
                "labels_mirror": {RegKeys.ACTIVE: True, "start": 0.5},
                "labels_edge": {
                    RegKeys.ACTIVE: True,
                    "start": 0.138,  # some lat labels only partially complete
                    "surr_size": 12,
                    "smoothing_size": 0,  # no smoothing to avoid loss of detail
                    # only mild erosion to minimize layer loss since histology
                    # contrast is low
                    RegKeys.MARKER_EROSION: 5,
                },
                "smooth": 2,
                "make_far_hem_neg": True,
            },

            # ABA P56 (adult) specific settings
            "abap56adult": {
                # same atlas image as ABA P56dev
                "target_size": (528, 320, 456),
                "resize_factor": None,  # turn off resizing
                # same stained sections as for P56dev;
                # labels are already mirrored starting at z=228, but atlas is
                # not here, so mirror starting at the same z-plane to make both
                # sections and labels symmetric and aligned with one another;
                # no need to extend lateral edges
                "labels_mirror": {RegKeys.ACTIVE: True, "start": 0.5},
                "smooth": 2,
                "make_far_hem_neg": True,
            },

            # ABA CCFv3 specific settings
            "abaccfv3": {
                # for "25" image, which has same shape as ABA P56dev, P56adult
                "target_size": (456, 528, 320),
                "resize_factor": None,  # turn off resizing
                # atlas is almost (though not perfectly) symmetric, so turn
                # off mirroring but specify midline (z=228) to make those
                # labels negative; no need to extend lateral edges
                "labels_mirror": {RegKeys.ACTIVE: False, "start": 0.5},
                "make_far_hem_neg": True,
                "smooth": 0,
            },

            # Waxholm rat atlas specific settings
            "whsrat": {
                "target_size": (441, 1017, 383),
                "pre_plane": config.PLANE[2],
                "resize_factor": None,  # turn off resizing
                # mirror, but no need to extend lateral edges
                "labels_mirror": {RegKeys.ACTIVE: True, "start": 0.48},
                "crop_to_labels": True,  # much extraneous, unlabeled tissue
                "smooth": 4,
            },

            # Profile modifiers to turn off settings. These "no..." profiles
            # can be applied on top of atlas-specific profiles to turn off
            # specific settings. Where possible, the ACTIVE flags will be turned
            # off to retain the rest of the settings within the given group
            # so that they can be used for metrics, cropping, etc.

            # turn off most image manipulations to show original atlas and labels
            # while allowing transformations set as command-line arguments
            "raw": {
                "labels_edge": {RegKeys.ACTIVE: False},
                "labels_mirror": {RegKeys.ACTIVE: False},
                "expand_labels": None,
                "rotate": None,
                "affine": None,
                "smooth": None,
                "crop_to_labels": False,
            },

            # turn off atlas rotation
            "norotate": {
                "rotate": None,
            },

            # turn off edge extension along with smoothing
            "noedge": {
                "labels_edge": {RegKeys.ACTIVE: False},
                "labels_mirror": {RegKeys.ACTIVE: True},
                "smooth": None,
            },

            # turn off mirroring along with smoothing
            "nomirror": {
                "labels_edge": {RegKeys.ACTIVE: True},
                "labels_mirror": {RegKeys.ACTIVE: False},
                "smooth": None,
            },

            # turn off both mirroring and edge extension along with smoothing
            # while preserving their settings for measurements and cropping
            "noext": {
                "labels_edge": {RegKeys.ACTIVE: False},
                "labels_mirror": {RegKeys.ACTIVE: False},
                "smooth": None,
            },

            # turn off label smoothing
            "nosmooth": {
                "smooth": None,
            },

            # turn off negative labels
            "noneg": {
                # if mirroring, do not invert mirrored labels
                "labels_mirror": {"neg_labels": False},
                # do not invert far hemisphere labels
                "make_far_hem_neg": False,
            },

            # set smoothing to 4
            "smooth4": {
                "smooth": 4,
            },

            # turn off labels markers generation
            "nomarkers": {
                RegKeys.EDGE_AWARE_REANNOTAION: None,
            },

            # turn off cropping atlas to extent of labels
            "nocropatlas": {
                "crop_to_labels": False,
            },

            # turn off cropping labels to original size
            "nocroplabels": {
                "crop_to_orig": False,
            },

            # test label smoothing over range
            "smoothtest": {
                "smooth": (0, 1, 2, 3, 4, 5, 6),
                # "smooth": (0, ),
            },

            # test label smoothing over longer range
            "smoothtestlong": {
                "smooth": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            },

            # save intermediate steps where supported
            "savesteps": {
                "labels_edge": {RegKeys.SAVE_STEPS: True}
            },

            # crop anterior region of labels during single registration
            "cropanterior": {
                "truncate_labels": (None, (0.2, 0.8), (0.45, 1.0)),
            },

            # turn off image curation to avoid post-processing with carving
            # and in-painting
            "nopostproc": {
                "curate": False,
                "truncate_labels": None
            },

            # smoothing by Gaussian blur
            "smoothgaus": {
                "smoothing_mode": config.SmoothingModes.gaussian,
                "smooth": 0.25
            },

            # smoothing by Gaussian blur
            "smoothgaustest": {
                "smoothing_mode": config.SmoothingModes.gaussian,
                "smooth": (0, 0.25, 0.5, 0.75, 1, 1.25)
            },

            # combine sides for volume stats
            "combinesides": {
                "combine_sides": True,
            },

            # more volume stats
            "morestats": {
                # "extra_metric_groups": (config.MetricGroups.SHAPES,),
                "extra_metric_groups": (config.MetricGroups.POINT_CLOUD,),
            },

            # measure interior-border stats
            "interiorlabels": {
                "erode_labels": {"markers": True, "interior": True},
            },

        }
