#!/bin/bash
# Profile settings
# Author: David Young, 2019, 2020
"""Profile settings to setup common configurations.

Each profile has a default set of settings, which can be modified through 
"modifier" sub-profiles with groups of settings that overwrite the 
given default settings. 
"""
from collections import OrderedDict
from enum import Enum, auto

import numpy as np

from magmap.settings import config


class SettingsDict(dict):
    def __init__(self, *args, **kwargs):
        self["settings_name"] = "default"

    def add_modifier(self, mod_name, mods, name_check=None, sep="_"):
        """Add a modifer dictionary, overwriting any existing settings 
        with values from this dictionary.
        
        If both the original and new setting are dictionaries, the original
        dictionary will be updated with rather than overwritten by the
        new setting.
        
        Args:
            mod_name: Name of the modifier, which will be appended to the 
                name of the current settings.
            mods: Dictionary with keys matching default keys and values to 
                replace the correspondings values (or update if both original
                and new values are dictionaries).
            name_check: Name of a profile modifier to check; defaults to None. 
                If matches ``mod_name`` or is None, ``mods`` will be applied.
            sep: Separator between modifier elements. Defaults to "_".
        """
        # if name to check is given, must match modifier name to continue
        if name_check is not None and name_check != mod_name: return
        self["settings_name"] += sep + mod_name
        for key in mods.keys():
            if isinstance(self[key], dict) and isinstance(mods[key], dict):
                # update if both are dicts
                self[key].update(mods[key])
            else:
                # replace with modified setting
                self[key] = mods[key]


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
        self["clip_vmin"] = 5
        self["clip_vmax"] = 99.5
        self["clip_min"] = 0.2
        self["clip_max"] = 1.0
        # mult by config.near_max for lower threshold of max
        self["max_thresh_factor"] = 0.5
        self["tot_var_denoise"] = False
        self["unsharp_strength"] = 0.3
        self["erosion_threshold"] = 0.2

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
        # None turns off preprocessing in stack proc; make much larger than
        # segment_size (eg 2x) to cover the full segment ROI because of
        # additional overlap in segment ROIs
        self["denoise_size"] = 25
        self["segment_size"] = 500  # detection ROI max size along longest edge
        # z,y,x tolerances for pruning duplicates in overlapped regions
        self["prune_tol_factor"] = (1, 1, 1)
        self["verify_tol_factor"] = (1, 1, 1)
        # module level variable will take precedence
        self["sub_stack_max_pixels"] = (1000, 1000, 1000)

        # resizing for anisotropy
        self["isotropic"] = None  # final relative z,y,x scaling after resizing
        self["isotropic_vis"] = None  # z,y,x scaling factor for vis only
        self["resize_blobs"] = None  # z,y,x coord scaling before verification


def update_process_settings(settings, settings_type):
    """Update processing profiles, including layering modifications upon 
    existing base layers.
    
    For example, "lightsheet_5x" will give one profile, while 
    "lightsheet_5x_contrast" will layer additional settings on top of the 
    original lightsheet profile.
    
    Args:
        settings: A :class:``ProcessSettings`` profile object.
        settings_type: The name of the settings profile to apply. Profiles 
            will be matched by the start of the settings name, with 
            additional modifications made by matching ends of names.
    """
    profiles = settings_type.split("_")
    
    for profile in profiles:
        # update default profile with any combo of modifiers, where the 
        # order of the profile listing determines the precedence of settings
        
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
        settings.add_modifier(
            "lightsheet", 
            {
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
            profile)

        # minimal preprocessing
        settings.add_modifier(
            "minpreproc",
            {
                "clip_vmin": 0,
                "clip_vmax": 100,
                "clip_max": 1,
                "tot_var_denoise": True,
                "unsharp_strength": 0,
                "erosion_threshold": 0,
            },
            profile)

        # low resolution
        settings.add_modifier(
            "lowres",
            {
                "min_sigma_factor": 5,
                "max_sigma_factor": 6,
                "isotropic": None,
                "denoise_size": 2000,  # will use full detection ROI
                "segment_size": 1000,
                "max_thresh_factor": 1.0,
                "exclude_border": (8, 1, 1),
            },
            profile)

        # 2-photon 20x nuclei
        settings.add_modifier(
            "2p20x", 
            {
                "vis_3d": "surface", 
                "clip_vmax": 97, 
                "clip_min": 0, 
                "clip_max": 0.7, 
                "tot_var_denoise": True, 
                "unsharp_strength": 2.5, 
                # smaller threhsold since total var denoising 
                #"points_3d_thresh": 1.1
                "min_sigma_factor": 2.6, 
                "max_sigma_factor": 4, 
                "num_sigma": 20, 
                "overlap": 0.1, 
                "thresholding": None,#"otsu"
                #"thresholding_size": 41, 
                "thresholding_size": 64,  # for otsu
                #"thresholding_size": 50.0, # for random_walker
                "denoise_size": 25, 
                "segment_size": 100, 
                "prune_tol_factor": (1.5, 1.3, 1.3),
            }, 
            profile)

        # 2p 20x of zebrafish nuclei
        settings.add_modifier(
            "zebrafish", 
            {
                "min_sigma_factor": 2.5,
                "max_sigma_factor": 3,
            }, 
            profile)

        # higher contrast colormaps
        settings.add_modifier(
            "contrast", 
            {
                "channel_colors": ("inferno", "bone"),
                "scale_bar_color": "w", 
            }, 
            profile)

        # similar colormaps to greyscale but with a cool blue tinge
        settings.add_modifier(
            "bone",
            {
                "channel_colors": ("bone", "bone"),
                "scale_bar_color": "w", 
            },
            profile)

        # diverging colormaps for heat maps centered on 0
        settings.add_modifier(
            "diverging", 
            {
                "channel_colors": ("RdBu", "BrBG"), 
                "scale_bar_color": "k", 
                "colorbar": True,
            }, 
            profile)

        # lightsheet 5x of cytoplasmic markers
        settings.add_modifier(
            "cytoplasm", 
            {
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
            profile)

        # isotropic image that does not require interpolating visually
        settings.add_modifier(
            "isotropic",
            {
                "points_3d_thresh": 0.3,  # used only if not surface
                "isotropic_vis": (1, 1, 1),
             }, 
            profile)

        # binary image
        settings.add_modifier(
            "binary", 
            {
                "denoise_size": None,
                "detection_threshold": 0.001,
            }, 
            profile)
    
        # adjust nuclei size for 4x magnification
        settings.add_modifier(
            "4xnuc", 
            {
                "min_sigma_factor": 3, 
                "max_sigma_factor": 4,
            }, 
            profile)
    
        # fit into ~32GB RAM instance after isotropic interpolation
        settings.add_modifier(
            "20x", 
            {
                "segment_size": 50,
            }, 
            profile)
    
        # export to deep learning framework with required dimensions
        settings.add_modifier(
            "exportdl", 
            {
                "isotropic": (0.93, 1, 1),
            }, 
            profile)
    
        # downsample an image previously upsampled for isotropy
        settings.add_modifier(
            "downiso",
            {
                "isotropic": None,  # assume already isotropic
                "resize_blobs": (.2, 1, 1),
            },
            profile)

        # rotate by 180 deg
        # TODO: replace with plot labels config setting?
        settings.add_modifier(
            "rot180",
            {
                "load_rot90": 2,  # rotation by 180deg
            },
            profile)

        # denoise settings when performing registration
        settings.add_modifier(
            "register", 
            {
                "unsharp_strength": 1.5,
            }, 
            profile)
        
        # color and intensity geared toward histology atlas images
        settings.add_modifier(
            "atlas", 
            {
                "channel_colors": ("gray", ),
                "clip_vmax": 97,
            }, 
            profile)
        
        # colors for each channel based on randomly generated discrete colormaps
        settings.add_modifier(
            "randomcolors", 
            {
                "channel_colors": [],
            }, 
            profile)
        
        # normalize image5d and associated metadata to intensity values 
        # between 0 and 1
        settings.add_modifier(
            "norm", 
            {
                "norm": (0.0, 1.0),
            }, 
            profile)
    
    if config.verbose:
        print("process settings for {}:\n{}"
              .format(settings["settings_name"], settings))


# isotropic factors
_isotropic_zs = np.arange(0.1, 2, 0.1)
_isotropic_factors = np.ones((len(_isotropic_zs), 3))
_prune_tol_zs = np.arange(0.5, 1.1, 0.5)
_prune_tol_factors = np.ones((len(_prune_tol_zs), 3)) * 0.9

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
        #("isotropic", _isotropic_factors),
        #("isotropic", np.array([(0.96, 1, 1)])),
        #("overlap", np.arange(0.1, 1.0, 0.1)),
        #("prune_tol_factor", np.array([(4, 1.3, 1.3)])),
        #("prune_tol_factor", _prune_tol_factors),
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
    ("size", OrderedDict([
        ("min_sigma_factor", np.arange(2, 2.71, 0.1)),
        ("max_sigma_factor", np.arange(2.7, 3.21, 0.1)),
        #("min_sigma_factor", np.arange(2.5, 3.51, 0.1)),
        #("max_sigma_factor", np.arange(3.5, 4.51, 0.1)),
    ])),
])


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
        
        # labels curation, given as fractions of the total planes; 
        # use None to ignore, -1 to set automatically (for mirror and edge), 
        # or give a fraction between 0 and 1; can turn off with extend_labels 
        # while keeping settings for cropping, etc
        self["labels_mirror"] = None  # reflect planes starting here
        # extend edge labels
        self["labels_edge"] = {
            RegKeys.ACTIVE: False,
            RegKeys.SAVE_STEPS: False,
            "start": -1,  # start plane index (-1 to set automatically)
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
        # TODO: replace with ACTIVE flag?
        self["extend_labels"] = {"edge": True, "mirror": True}
        
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


def update_register_settings(settings, settings_type):
    
    profiles = settings_type.split("_")
    
    for profile in profiles:
        
        # more aggressive parameters for finer tuning
        settings.add_modifier(
            "finer", 
            {
                "bspline_iter_max": "512", 
                "truncate_labels": (None, (0.2, 1.0), (0.45, 1.0)), 
                "holes_area": 5000, 
            }, 
            profile)

        # Normalized Correlation Coefficient similarity metric for registration
        settings.add_modifier(
            "ncc", 
            {
                "metric_similarity": "AdvancedNormalizedCorrelation",
                # fallback to MMI since it has been rather reliable
                "metric_sim_fallback":
                    (0.85, "AdvancedMattesMutualInformation"),
                "bspline_grid_space_voxels": "60",
            }, 
            profile)

        # groupwise registration
        settings.add_modifier(
            "groupwise",
            {
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
            profile)
        
        # test a target size
        settings.add_modifier(
            "test",
            {
                "target_size": (50, 50, 50),
            },
            profile)

        # atlas is big relative to the experimental image, so need to 
        # more aggressively downsize the atlas
        settings.add_modifier(
            "big", 
            {
                "resize_factor": 0.625,
            }, 
            profile)
        
        # new atlas generation: turn on preprocessing
        # TODO: likely remove since not using preprocessing currently
        settings.add_modifier(
            "new", 
            {
                "preprocess": True,
            }, 
            profile)
        
        # registration to new atlas assumes images are roughly same size and
        # orientation (ie transposed) and already have mirrored labels aligned 
        # with the fixed image toward the bottom of the z-dimension
        settings.add_modifier(
            "generated", 
            {
                "resize_factor": 1.0, 
                "truncate_labels": (None, (0.18, 1.0), (0.2, 1.0)),
                "labels_mirror": None, 
                "labels_edge": None, 
            }, 
            profile)
        
        # atlas that uses groupwise image as the atlas itself should 
        # determine atlas threshold dynamically
        settings.add_modifier(
            "grouped", 
            {
                "atlas_threshold": None,
            }, 
            profile)
        
        # ABA E11pt5 specific settings
        settings.add_modifier(
            "abae11pt5", 
            {
                "target_size": (345, 371, 158),
                "resize_factor": None,  # turn off resizing
                "labels_mirror": 0.52, 
                "labels_edge": None, 
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
                "overlap_meas_add_lbls": ((126651558, 126652059), ), 
            }, 
            profile)

        # pitch rotation only for ABA E11pt to compare corresponding z-planes
        # with other stages of refinement
        settings.add_modifier(
            "abae11pt5pitch",
            {"rotate": ((-30, 0),)},
            profile)

        # ABA E13pt5 specific settings
        settings.add_modifier(
            "abae13pt5", 
            {
                "target_size": (552, 673, 340),
                "resize_factor": None,  # turn off resizing
                "labels_mirror": 0.48, 
                # small, default surr size to avoid capturing 3rd labeled area 
                # that becomes an artifact
                "labels_edge": {
                    RegKeys.ACTIVE: True,
                }, 
                "atlas_threshold": 55,  # avoid edge over-extension into skull
                "rotate": ((-4, 1), (-2, 2)),
                "crop_to_labels": True, 
                "smooth": 2, 
            }, 
            profile)
        
        # ABA E15pt5 specific settings
        settings.add_modifier(
            "abae15pt5", 
            {
                "target_size": (704, 982, 386),
                "resize_factor": None,  # turn off resizing
                "labels_mirror": 0.49, 
                "labels_edge": {
                    RegKeys.ACTIVE: True, "surr_size": 12,
                    # increase template smoothing to prevent over-extension of
                    # intermediate stratum of Str
                    "smoothing_size": 5,
                    # larger to allow superficial stratum of DPall to take over
                    RegKeys.MARKER_EROSION: 19,
                }, 
                "atlas_threshold": 45,  # avoid edge over-extension into skull
                "rotate": ((-4, 1), ), 
                "crop_to_labels": True,
                "smooth": 2,
            }, 
            profile)
        
        # ABA E18pt5 specific settings
        settings.add_modifier(
            "abae18pt5", 
            {
                "target_size": (278, 581, 370),
                "resize_factor": None, # turn off resizing
                "labels_mirror": 0.525, 
                # start from smallest BG; remove spurious label pxs around
                # medial pallium by smoothing
                "labels_edge": {
                    RegKeys.ACTIVE: True, "start": 0.137, "surr_size": 12,
                    RegKeys.MARKER_EROSION: 12,
                    RegKeys.MARKER_EROSION_USE_MIN: True,
                }, 
                "expand_labels": (((None, ), (0, 279), (103, 108)),), 
                "rotate": ((1.5, 1), (2, 2)),
                "smooth": 3,
                RegKeys.EDGE_AWARE_REANNOTAION: {
                    RegKeys.MARKER_EROSION_MIN: 4,
                }
            }, 
            profile)
        
        # ABA P4 specific settings
        settings.add_modifier(
            "abap4", 
            {
                "target_size": (724, 403, 398),
                "resize_factor": None,  # turn off resizing
                "labels_mirror": 0.487, 
                "labels_edge": {
                    RegKeys.ACTIVE: True, "surr_size": 12,
                    # balance eroding medial pallium and allowing dorsal
                    # pallium to take over
                    RegKeys.MARKER_EROSION: 8,
                }, 
                # open caudal labels to allow smallest mirror plane index, 
                # though still cross midline as some regions only have 
                # labels past midline
                "rotate": ((0.22, 1), ),
                "smooth": 4,
            }, 
            profile)
        
        # ABA P14 specific settings
        settings.add_modifier(
            "abap14", 
            {
                "target_size": (390, 794, 469),
                "resize_factor": None, # turn off resizing
                # will still cross midline since some regions only have labels 
                # past midline
                "labels_mirror": 0.5, 
                "labels_edge": {
                    RegKeys.ACTIVE: True,
                    "start": 0.078,  # avoid alar part size jitter
                    "surr_size": 12,
                    RegKeys.MARKER_EROSION: 40,
                    RegKeys.MARKER_EROSION_MIN: 10, 
                }, 
                # rotate conservatively for symmetry without losing labels
                "rotate": ((-0.4, 1), ),
                "smooth": 5,
            }, 
            profile)
        
        # ABA P28 specific settings
        settings.add_modifier(
            "abap28", 
            {
                "target_size": (863, 480, 418),
                "resize_factor": None, # turn off resizing
                # will still cross midline since some regions only have labels 
                # past midline
                "labels_mirror": 0.48, 
                "labels_edge": {
                    RegKeys.ACTIVE: True,
                    "start": 0.11,  # some lat labels only partially complete
                    "surr_size": 12,
                    "smoothing_size": 0,  # no smoothing to avoid loss of detail
                }, 
                #"labels_dup": 0.48, 
                # rotate for symmetry, which also reduces label loss
                "rotate": ((1, 2), ),
                "smooth": 2,
            }, 
            profile)
        
        # ABA P56 (developing mouse) specific settings
        settings.add_modifier(
            "abap56", 
            {
                "target_size": (528, 320, 456),
                "resize_factor": None,  # turn off resizing
                # stained sections and labels almost but not symmetric
                "labels_mirror": 0.5,
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
            profile)
        
        # ABA P56 (adult) specific settings
        settings.add_modifier(
            "abap56adult", 
            {
                # same atlas image as ABA P56dev
                "target_size": (528, 320, 456),
                "resize_factor": None, # turn off resizing
                # same stained sections as for P56dev; 
                # labels are already mirrored starting at z=228, but atlas is 
                # not here, so mirror starting at the same z-plane to make both 
                # sections and labels symmetric and aligned with one another
                "labels_mirror": 0.5,
                "labels_edge": None, 
                "smooth": 2, 
                "make_far_hem_neg": True, 
            }, 
            profile)

        # ABA CCFv3 specific settings
        settings.add_modifier(
            "abaccfv3",
            {
                # for "25" image, which has same shape as ABA P56dev, P56adult
                "target_size": (456, 528, 320),
                "resize_factor": None,  # turn off resizing
                # TODO: consider whether to mirror for perfect symmetry
                "extend_labels": {"mirror": False},
                # atlas is almost but not perfectly symmetric; midline at z=228
                "labels_mirror": 0.5,
                "labels_edge": None,
                "smooth": 0,
                "make_far_hem_neg": True,
            },
            profile)

        # Waxholm rat atlas specific settings
        settings.add_modifier(
            "whsrat", 
            {
                "target_size": (441, 1017, 383),
                "pre_plane": config.PLANE[2], 
                "resize_factor": None,  # turn off resizing
                "labels_mirror": 0.48, 
                "labels_edge": None, 
                "crop_to_labels": True,  # much extraneous, unlabeled tissue
                "smooth": 4, 
            }, 
            profile)

        # turn off most image manipulations to show original atlas and labels 
        # while allowing transformations set as command-line arguments
        settings.add_modifier(
            "raw",
            {
                "extend_labels": {"edge": False, "mirror": False},
                "expand_labels": None,
                "rotate": None,
                "affine": None,
                "smooth": None,
                "crop_to_labels": False,
             },
            profile)

        # turn off atlas rotation
        settings.add_modifier(
            "norotate", 
            {
                "rotate": None,
            },
            profile)
        
        # turn off edge extension along with smoothing
        settings.add_modifier(
            "noedge", 
            {
                "extend_labels": {"edge": False, "mirror": True}, 
                "smooth": None,
             },
            profile)

        # turn off mirroring along with smoothing
        settings.add_modifier(
            "nomirror",
            {
                "extend_labels": {"edge": True, "mirror": False},
                "smooth": None,
             },
            profile)

        # turn off both mirroring and edge extension along with smoothing 
        # while preserving their settings for measurements and cropping
        settings.add_modifier(
            "noext", 
            {
                "extend_labels": {"edge": False, "mirror": False}, 
                "smooth": None,
             },
            profile)
        
        # turn off label smoothing
        settings.add_modifier(
            "nosmooth", 
            {
                "smooth": None,
            }, 
            profile)

        # set smoothing to 4
        settings.add_modifier(
            "smooth4",
            {
                "smooth": 4,
            },
            profile)

        # turn off labels markers generation
        settings.add_modifier(
            "nomarkers", 
            {
                RegKeys.EDGE_AWARE_REANNOTAION: None,
            }, 
            profile)
        
        # turn off cropping atlas to extent of labels
        settings.add_modifier(
            "nocropatlas", 
            {
                "crop_to_labels": False,
            }, 
            profile)
        
        # turn off cropping labels to original size
        settings.add_modifier(
            "nocroplabels", 
            {
                "crop_to_orig": False,
            }, 
            profile)
        
        # test label smoothing over range
        settings.add_modifier(
            "smoothtest", 
            {
                "smooth": (0, 1, 2, 3, 4, 5, 6),
                #"smooth": (0, ),
            },
            profile)
        
        # test label smoothing over longer range
        settings.add_modifier(
            "smoothtestlong", 
            {
                "smooth": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            }, 
            profile)
        
        # save intermediate steps where supported
        settings.add_modifier(
            "savesteps", 
            {
                "labels_edge": {RegKeys.SAVE_STEPS: True}
            }, 
            profile)
        
        # groupwise registration batch 02
        settings.add_modifier(
            "grouped02", 
            {
                "bspline_grid_space_voxels": "70", 
                "grid_spacing_schedule": [
                   "8.0", "7.0", "6.0", "5.0", "4.0", "3.0", "2.0", "1.0"], 
                "carve_threshold": 0.009,
            }, 
            profile)
            
        # groupwise registration batch 04
        settings.add_modifier(
            "grouped04", 
            {
                "carve_threshold": 0.015,
            }, 
            profile)
        
        # crop anterior region of labels during single registration
        settings.add_modifier(
            "cropanterior", 
            {
                "truncate_labels": (None, (0.2, 0.8), (0.45, 1.0)),
            }, 
            profile)
        
        # turn off image curation to avoid post-processing with carving 
        # and in-painting
        settings.add_modifier(
            "nopostproc", 
            {
                "curate": False, 
                "truncate_labels": None
            }, 
            profile)
    
        # smoothing by Gaussian blur
        settings.add_modifier(
            "smoothgaus", 
            {
                "smoothing_mode": config.SmoothingModes.gaussian, 
                "smooth": 0.25
            }, 
            profile)
    
        # smoothing by Gaussian blur
        settings.add_modifier(
            "smoothgaustest", 
            {
                "smoothing_mode": config.SmoothingModes.gaussian, 
                "smooth": (0, 0.25, 0.5, 0.75, 1, 1.25)
            }, 
            profile)
        
        # combine sides for volume stats
        settings.add_modifier(
            "combinesides", 
            {
                "combine_sides": True,
            }, 
            profile)

        # more volume stats
        settings.add_modifier(
            "morestats",
            {
                #"extra_metric_groups": (config.MetricGroups.SHAPES,),
                "extra_metric_groups": (config.MetricGroups.POINT_CLOUD,),
            },
            profile)
        
        # measure interior-border stats
        settings.add_modifier(
            "interiorlabels",
            {
                "erode_labels": {"markers": True, "interior": True},
            },
            profile)

    if config.verbose:
        print("process settings for {}:\n{}"
              .format(settings["settings_name"], settings))
