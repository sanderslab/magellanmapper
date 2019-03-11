#!/bin/bash
# Config file for shared settings
# Author: David Young, 2017, 2019
"""Configuration storage module.

This module allows customization of settings for various imaging systems, 
such as grouped settings for particular microscopes. Additional parameters 
such as command-line flag settings and databases can also be stored here 
for program access.

Attributes:
    filename: The filename of the source images. A corresponding file with
        the subset as a 5 digit number (eg 00003) with .npz appended to 
        the end will be checked first based on this filename. Set with
        "img=path/to/file" argument.
    series: The series for multi-stack files, using 0-based indexing. Set
        with "series=n" argument.
    channel: The channel to view. Set with "channel=n" argument.
    load_labels: Path to the labels reference file, which also serves as a 
        flag to references label/annotation images.
    labels_img: Numpy array of a registered labels image, which should 
        correspond to the main ``image5d`` image.
    labels_scaling: Array of ``labels_img`` compared to ``image5d`` 
        for each corresponding dimension.
    labels_ref: Raw reference dictionary imported from the JSON file 
        corresponding to the labels image.
    labels_ref_lookup: Reference dictionary with keys corresponding to the IDs 
        in the labels image.
"""


import numpy as np
from collections import OrderedDict
from enum import Enum

verbose = False
POS_THRESH = 0.001 # threshold for positive values for float comparison

# 2D PLOTTING

# custom colormaps in plot_2d
CMAP_GRBK_NAME = "Green_black"
CMAP_RDBK_NAME = "Red_black"

# processing type directly in module
PLOT_2D_TYPES = ("smoothing_metrics", "barplot")
plot_2d_type = None

# extensions for saving figures
FORMATS_3D = ("obj", "x3d") # save 3D renderings
savefig = None # save files using this extension


# IMAGE VIEWING

no_show = False
max_scroll = 20 # max speed when scrolling through planes


# STACK PROCESSING

rescale = None # rescale image
slice_vals = None # list of slice values to give directly to slice fn
delay = None # delay time between images
# max pixels of sub-stacks for stack processing (z, y, x order), which can 
# be set from command-line and takes precedence over process settings to 
# allow custom configurations depending on instance type
sub_stack_max_pixels = None


# IMAGE EXPORT

# flag to save ROI to file
saveroi = False

# alpha levels for overlaid images, defaulting to main image, labels image
alphas = [1, 0.9, 0.9]

# show scale bars
scale_bar = True


# IMAGE REGISTRATION

# reference atlas labels
load_labels = None
labels_img = None # in Numpy format
labels_scaling = None
labels_ref = None
labels_ref_lookup = None
labels_level = None
labels_mirror = True
borders_img = None
reg_suffixes = None
REG_SUFFIX_ATLAS = "atlas"
REG_SUFFIX_ANNOTATION = "annotation"
REG_SUFFIX_BORDERS = "borders"
VOL_KEY = "volume"
BLOBS_KEY = "blobs"
VARIATION_BLOBS_KEY = "var_blobs" # variation in blob density
VARIATION_EXP_KEY = "var_exp" # variation in experiment intensity
SIDE_KEY = "Side"
GENOTYPE_KEY = "Geno"
SUB_SEG_MULT = 100 # labels multiplier for sub-segmentations
REGION_ALL = "all"

class ABAKeys(Enum):
    """Allen Brain Atlas ontology hierarchy keys.
    
    Values of each enumeration maps to key values in the ABA ontology 
    specification.
    """
    NAME = "name"
    ABA_ID = "id"
    LEVEL = "st_level"
    CHILDREN = "children"
    ACRONYM = "acronym"

# register module modes when called from command-line
RegisterTypes = Enum(
    "RegisterTypes", [
        "single", "group", "overlays", "volumes", "densities", "export_vols", 
        "export_regions", "new_atlas", "import_atlas", "export_common_labels", 
        "make_edge_images", "reg_labels_to_atlas", "merge_atlas_segs", 
        "make_edge_images_exp", "vol_stats", "make_density_images", 
        "merge_atlas_segs_exp", "make_subsegs", "export_metrics_compactness", 
    ]
)
register_type = None

# label smoothing modes
SmoothingModes = Enum(
    "SmoothingModes", [
        "opening", "gaussian", "closing"
    ]
)

# flip/rotate the image; the direction of change can be variable
flip = None

# groups, such as genotypes and sex or combos
GROUPS_NUMERIC = {"WT": 0.0, "het": 0.5, "null":1.0}
groups = None

# smoothing metrics
PATH_SMOOTHING_METRICS = "smoothing.csv"

# whole atlas image import metrics
PATH_ATLAS_IMPORT_METRICS = "stats.csv"

# common labels
PATH_COMMON_LABELS = "labels_common.csv"


# STATS

STATS_TYPES = ("merge_csvs", "exps_by_regions")
stats_type = None
seed = 0 # random number generator seed


# AWS

ec2_start = None
ec2_list = None
ec2_terminate = None


# SLACK NOTIFICATIONS

notify_url = None
notify_msg = None
notify_attach = None


# MICROSCOPE SETTINGS FOR DETECTIONS

class SettingsDict(dict):
    def __init__(self, *args, **kwargs):
        self["settings_name"] = "default"

    def add_modifier(self, mod_name, mods, name_check=None, sep="_"):
        """Add a modifer dictionary, overwriting any existing settings 
        with values from this dictionary.
        
        Args:
            mod_name: Name of the modifier, which will be appended to the 
                name of the current settings.
            mods: Dictionary with keys matching default keys and values to 
                replace the correspondings values.
            profile: Name of a profile modifier to check; defaults to None. 
                If matches ``mod_name`` or is None, ``mods`` will be applied.
            sep: Separator between modifier elements. Defaults to "_".
        """
        # if name to check is given, must match modifier name to continue
        if name_check is not None and name_check != mod_name: return
        self["settings_name"] += sep + mod_name
        for key in mods.keys():
            self[key] = mods[key]

class ProcessSettings(SettingsDict):
    def __init__(self, *args, **kwargs):
        super().__init__(self)
        self["settings_name"] = "default"
        self["vis_3d"] = "points" # "points" or "surface" 3D visualization
        self["points_3d_thresh"] = 0.85 # frac of thresh (changed in v.0.6.6)
        self["clip_vmax"] = 99.5
        self["clip_min"] = 0.2
        self["clip_max"] = 1.0
        self["tot_var_denoise"] = False
        self["unsharp_strength"] = 0.3
        self["erosion_threshold"] = 0.2
        self["min_sigma_factor"] = 3
        self["max_sigma_factor"] = 30
        self["num_sigma"] = 10
        self["detection_threshold"] = 0.1
        self["overlap"] = 0.5
        self["thresholding"] = None
        self["thresholding_size"] = -1
        self["denoise_size"] = 25 # None turns off preprocessing in stack proc
        self["segment_size"] = 500
        self["prune_tol_factor"] = (1, 1, 1)
        self["segmenting_mean_thresh"] = 0.4
        self["scale_factor"] = (1, 1, 1)
        self["channel_colors"] = (CMAP_GRBK_NAME, CMAP_RDBK_NAME)
        self["isotropic"] = None
        self["isotropic_vis"] = None
        self["resize_blobs"] = None
        # module level variable will take precedence
        self["sub_stack_max_pixels"] = (1000, 1000, 1000)
    
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
    # MAIN PROFILES
    
    profiles = settings_type.split("_")
    
    # TODO: consider treating main profiles simply as modifiers
    
    if settings_type.startswith("2p20x"):
        settings["settings_name"] = "2p20x"
        settings["vis_3d"] = "surface"
        settings["clip_vmax"] = 97
        settings["clip_min"] = 0
        settings["clip_max"] = 0.7
        settings["tot_var_denoise"] = True
        settings["unsharp_strength"] = 2.5
        # smaller threhsold since total var denoising
        #settings["points_3d_thresh"] = 1.1
        settings["min_sigma_factor"] = 2.6
        settings["max_sigma_factor"] = 4
        settings["num_sigma"] = 20
        settings["overlap"] = 0.1
        settings["thresholding"] = None#"otsu"
        #settings["thresholding_size"] = 41
        settings["thresholding_size"] = 64 # for otsu
        #settings["thresholding_size"] = 50.0 # for random_walker
        settings["denoise_size"] = 25
        settings["segment_size"] = 100
        settings["prune_tol_factor"] = (1.5, 1.3, 1.3)
        settings["segmenting_mean_thresh"] = -0.25
        
    elif settings_type.startswith("lightsheetv01"):
        # detection settings up through v.0.6.1
        settings["settings_name"] = "lightsheet_v01"
        settings["points_3d_thresh"] = 0.7
        settings["clip_vmax"] = 98.5
        settings["clip_min"] = 0
        settings["clip_max"] = 0.6
        settings["unsharp_strength"] = 0.3
        settings["min_sigma_factor"] = 3
        settings["max_sigma_factor"] = 4
        settings["num_sigma"] = 10
        settings["overlap"] = 0.5
        settings["segment_size"] = 200
        settings["prune_tol_factor"] = (3, 1.3, 1.3)
        settings["segmenting_mean_thresh"] = 0.5
        settings["scale_factor"] = (0.63, 1, 1)
        settings["isotropic_vis"] = (1, 3, 3)
        # mimic absence of z-limit
        settings["sub_stack_max_pixels"] = (100000, 1000, 1000)
        
    elif settings_type.startswith("lightsheetv02"):
        # detection settings from v.0.6.2
        settings["settings_name"] = "lightsheet_v02"
        settings["points_3d_thresh"] = 0.7
        settings["clip_vmax"] = 98.5
        settings["clip_min"] = 0
        settings["clip_max"] = 0.5
        settings["unsharp_strength"] = 0.3
        settings["min_sigma_factor"] = 3
        settings["max_sigma_factor"] = 4
        settings["num_sigma"] = 10
        settings["overlap"] = 0.55
        settings["segment_size"] = 200
        settings["prune_tol_factor"] = (3, 1.3, 1.3)
        settings["segmenting_mean_thresh"] = -10 # unused since scale factor off
        settings["scale_factor"] = None
        settings["isotropic"] = (0.96, 1, 1)
        
        ver_split = settings_type.split(".")
        if len(ver_split) >= 2:
            # minor versioning to allow slight modifications to profile
            minor_ver = int(ver_split[-1])
        
            if minor_ver >= 1:
                # detection settings from v.0.6.4
                settings["settings_name"] += ".1"
                settings["erosion_threshold"] = 0.3
                settings["sub_stack_max_pixels"] = (1000, 1000, 1000)
        
            if minor_ver >= 2:
                # detection settings from v.0.6.6
                settings["settings_name"] += ".2"
                settings["sub_stack_max_pixels"] = (1200, 800, 800)
    
    elif settings_type.startswith("lightsheet"):
        # detection settings optimized for lightsheet
        settings["settings_name"] = "lightsheet"
        #settings["vis_3d"] = "surface"
        settings["points_3d_thresh"] = 0.7
        settings["clip_vmax"] = 98.5
        settings["clip_min"] = 0
        settings["clip_max"] = 0.5
        settings["unsharp_strength"] = 0.3
        settings["erosion_threshold"] = 0.3
        settings["min_sigma_factor"] = 3
        settings["max_sigma_factor"] = 4
        settings["num_sigma"] = 10
        settings["overlap"] = 0.55
        settings["segment_size"] = 200
        settings["prune_tol_factor"] = (3, 1.3, 1.3)
        settings["segmenting_mean_thresh"] = -10 # unused since scale factor off
        settings["scale_factor"] = None
        settings["isotropic"] = (0.96, 1, 1)
        settings["isotropic_vis"] = (1.3, 1, 1)
        settings["sub_stack_max_pixels"] = (1200, 800, 800)
    
    
    for profile in profiles:
        # PROFILE MODIFIERS
        # any/all/none can be combined with any main profile, modifiers lower in 
        # this listing taking precedence over prior ones and the main profile
        
        settings.add_modifier(
            "zebrafish", 
            {"min_sigma_factor": 2.5,
             "max_sigma_factor": 3}, 
            profile)
        
        settings.add_modifier(
            "contrast", 
            {"channel_colors": ("inferno", "bone")}, 
            profile)
        
        settings.add_modifier(
            "cytoplasm", 
            {"clip_min": 0.3,
             "clip_max": 0.8,
             "points_3d_thresh": 0.7, 
             # adjust sigmas based on extent of cyto staining; 
             # TODO: consider adding sigma_mult if ratio remains relatively const
             "min_sigma_factor": 4, 
             "max_sigma_factor": 10,
             "num_sigma": 10,
             "overlap": 0.2}, 
            profile)
        
        settings.add_modifier(
            "small", 
            {"points_3d_thresh": 0.3, # used only if not surface
             "isotropic_vis": (1, 1, 1)}, 
            profile)
        
        settings.add_modifier(
            "binary", 
            {"denoise_size": None,
             "detection_threshold": 0.001}, 
            profile)
    
        # fit into ~32GB RAM instance after isotropic interpolation
        settings.add_modifier(
            "20x", 
            {"segment_size": 50}, 
            profile)
    
        # export to deep learning framework with required dimensions
        settings.add_modifier(
            "exportdl", 
            {"isotropic": (0.93, 1, 1)}, 
            profile)
    
        # import from deep learning predicted image
        settings.add_modifier(
            "importdl", 
            {"isotropic": None, # assume already isotropic
             "resize_blobs": (.2, 1, 1)}, 
            profile)
        
        # denoise settings when performing registration
        settings.add_modifier(
            "register", 
            {"unsharp_strength": 1.5}, 
            profile)
        
        settings.add_modifier(
            "atlas", 
            {"channel_colors": ("gray", ),
             "clip_vmax": 97}, 
            profile)
    
    if verbose:
        print("process settings for {}:\n{}"
              .format(settings["settings_name"], settings))
    
# default settings and list of settings for each channel
process_settings = ProcessSettings()
process_settings_list = [process_settings]

def get_process_settings(i):
    """Get the microscope profile for the given channel.
    
    Args:
        i: Index, typically a channel number.
    
    Returns:
        The profile settings for corresponding to the given channel number, 
        or the default profile if only one is available.
    """
    settings = process_settings
    if len(process_settings_list) > i:
        settings = process_settings_list[i]
    return settings


# REGISTRATION SETTINGS

class RegisterSettings(SettingsDict):
    def __init__(self, *args, **kwargs):
        super().__init__(self)
        self["settings_name"] = "default"
        self["translation_iter_max"] = "2048"
        self["affine_iter_max"] = "1024"
        self["bspline_iter_max"] = "256"
        self["bspline_grid_space_voxels"] = "50"
        self["grid_spacing_schedule"] = None
        self["groupwise_iter_max"] = "1024"
        self["resize_factor"] = 0.7
        self["preprocess"] = False
        self["point_based"] = False
        self["smooth"] = None # smooth labels
        self["crop_to_labels"] = False # crop labels and atlas to nonzero labels
        self["curate"] = True # carve image; in-paint if generating atlas
        
        # erase labels outside of ((x_start, x_end), (y_start, ...) ...) 
        # (applied after transposition), where each val is given as fractions
        # of the full range or None to not truncate that at that position; 
        # None for the entire setting turns off truncation
        self["truncate_labels"] = None
        
        # labels curation, given as fractions of the total planes; 
        # use None to ignore, -1 to set automatically (for mirror and edge), 
        # or give a fraction between 0 and 1
        self["labels_mirror"] = None # reflect planes starting here
        self["labels_edge"] = None # extend edge labels from here
        self["labels_dup"] = None # start duplicating planes til last labels
        
        # expand labels within bounds given by 
        # (((x_pixels_start, x_pixels_end), ...), (next_region...)), or None 
        # to avoid expansion
        self["expand_labels"] = None
        
        # atlas and labels rotation by ((angle0, axis0), ...), or None to 
        # avoid rotation
        self["rotate"] = None
        
        self["atlas_threshold"] = 10.0
        self["target_size"] = None # x,y,z in exp orientation
        
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
        self["log_sigma"] = 5 # Gaussian sigma; use None to skip
        # use atlas_threshold on atlas image to generate mask for finding 
        # background rather than using labels and thresholded LoG image, 
        # useful when ventricular spaces are labeled
        self["log_atlas_thresh"] = False
        
        # erosion size when converting labels to markers; use None to skip
        self["marker_erosion"] = 8
        
        # crop labels back to their original background after smoothing 
        # (ignored during atlas import if no smoothing), given as the filter 
        # size used to open up the backgound before cropping, 0 to use 
        # the original background as-is, or False not to crop
        self["crop_to_orig"] = 1
        
        self["smoothing_mode"] = SmoothingModes.opening

def update_register_settings(settings, settings_type):
    
    profiles = settings_type.split("_")
    
    if settings_type.startswith("finer"):
        # more aggressive parameters for finer tuning
        settings["settings_name"] = "finer"
        settings["bspline_iter_max"] = "512"
        settings["truncate_labels"] = (None, (0.2, 1.0), (0.45, 1.0))
        settings["holes_area"] = 5000
    
    elif settings_type.startswith("groupwise"):
        # groupwise registration
        settings["settings_name"] = "groupwise"
        
        # larger bspline voxels to avoid over deformation of internal structures
        settings["bspline_grid_space_voxels"] = "130"
        
        # need to empirically determine
        settings["carve_threshold"] = 0.01
        settings["holes_area"] = 10000
        
        # empirically determined to add variable tissue area from first image 
        # since this tissue may be necessary to register to other images 
        # that contain this variable region
        settings["extend_borders"] = ((60, 180), (0, 200), (20, 110))
        
        # increased number of resolutions with overall increased spacing 
        # schedule since it appears to improve internal alignment
        settings["grid_spacing_schedule"] = [
            "8", "8", "4", "4", "4", "2", "2", "2", "1", "1", "1", "1"]
    
    elif settings_type.startswith("test"):
        settings["settings_name"] = "test"
        settings["target_size"] = (50, 50, 50)
    
    for profile in profiles:
        # atlas is big relative to the experimental image, so need to 
        # more aggressively downsize the atlas
        settings.add_modifier(
            "big", 
            {"resize_factor": 0.625}, 
            profile)
        
        # new atlas generation: turn on preprocessing
        # TODO: likely remove since not using preprocessing currently
        settings.add_modifier(
            "new", 
            {"preprocess": True}, 
            profile)
        
        # registration to new atlas assumes images are roughly same size and
        # orientation (ie transposed) and already have mirrored labels aligned 
        # with the fixed image toward the bottom of the z-dimension
        settings.add_modifier(
            "generated", 
            {"resize_factor": 1.0, 
             "truncate_labels": (None, (0.18, 1.0), (0.2, 1.0)),
             "labels_mirror": None, 
             "labels_edge": None, 
            }, 
            profile)
        
        # atlas that uses groupwise image as the atlas itself should 
        # determine atlas threshold dynamically
        settings.add_modifier(
            "grouped", 
            {"atlas_threshold": None}, 
            profile)
        
        # ABA E11pt5 specific settings
        settings.add_modifier(
            "abae11pt5", 
            {"target_size": (345, 371, 158),
             "resize_factor": None, # turn off resizing
             "labels_mirror": 0.52, 
             "labels_edge": None, 
             "log_atlas_thresh": True, 
             "atlas_threshold": 75, # to avoid over-extension into ventricles
             # rotate axis 0 to open vertical gap for affines (esp 2nd)
             "rotate": ((-5, 1), (-1, 2), (-30, 0)), 
             "affine": ({
                 # shear tail opposite the brain back toward midline
                 "axis_along": 1, "axis_shift": 0, "shift": (25, 0), 
                 "bounds": ((None, None), (70, 250), (0, 150))
             },{
                 # shear end of tail more distally, where the tail wraps 
                 # back on itself
                 "axis_along": 2, "axis_shift": 0, "shift": (0, 50), 
                 "bounds": ((None, None), (0, 200), (50, 150))
             },{
                 # counter shearing at far distal end, using attachment for 
                 # a more gradual shearing along the y-axis to preserve the 
                 # cord along that axis
                 "axis_along": 2, "axis_shift": 0, "shift": (45, 0), 
                 "bounds": ((None, None), (160, 200), (90, 150)), "axis_attach": 1
             }), 
             "crop_to_labels": True, # req because of 2nd affine
             "smooth": 2
            }, 
            profile)
        
        # ABA E13pt5 specific settings
        settings.add_modifier(
            "abae13pt5", 
            {"target_size": (552, 673, 340),
             "resize_factor": None, # turn off resizing
             "labels_mirror": 0.48, 
             "labels_edge": -1, 
             "atlas_threshold": 80, # to avoid edge over-extension into skull
             "rotate": ((-4, 1), (-2, 2)),
             "crop_to_labels": True, 
             "smooth": 2, 
            }, 
            profile)
        
        # ABA E15pt5 specific settings
        settings.add_modifier(
            "abae15pt5", 
            {"target_size": (704, 982, 386),
             "resize_factor": None, # turn off resizing
             "labels_mirror": 0.49, 
             "labels_edge": -1, 
             "atlas_threshold": 80, # to avoid edge over-extension into skull
             "rotate": ((-4, 1), ), 
             "crop_to_labels": True,
             "smooth": 2
            }, 
            profile)
        
        # ABA E18pt5 specific settings
        settings.add_modifier(
            "abae18pt5", 
            {"target_size": (278, 581, 370),
             "resize_factor": None, # turn off resizing
             "labels_mirror": 0.525, 
             "labels_edge": 0.137, # to use the smallest BG
             "expand_labels": (((None, ), (0, 279), (103, 108)),), 
             "rotate": ((1.5, 1), (2, 2)),
             "smooth": 4
            }, 
            profile)
        
        # ABA P4 specific settings
        settings.add_modifier(
            "abap4", 
            {"target_size": (724, 403, 398),
             "resize_factor": None, # turn off resizing
             "labels_mirror": 0.487, 
             "labels_edge": -1, 
             # open caudal labels to allow smallest mirror plane index, though 
             # still cross midline since some regions only have labels past midline
             "rotate": ((0.22, 1), ),
             "smooth": 4
            }, 
            profile)
        
        # ABA P14 specific settings
        settings.add_modifier(
            "abap14", 
            {"target_size": (390, 794, 469),
             "resize_factor": None, # turn off resizing
             # will still cross midline since some regions only have labels 
             # past midline
             "labels_mirror": 0.5, 
             "labels_edge": -1, 
             # rotate conservatively for symmetry without losing labels
             "rotate": ((-0.4, 1), ),
             "smooth": 4
            }, 
            profile)
        
        # ABA P28 specific settings
        settings.add_modifier(
            "abap28", 
            {"target_size": (863, 480, 418),
             "resize_factor": None, # turn off resizing
             # will still cross midline since some regions only have labels 
             # past midline
             "labels_mirror": 0.48, 
             # set edge since some lateral labels are only partially complete
             "labels_edge": 0.11, 
             #"labels_dup": 0.48, 
             # rotate for symmetry, which also reduces label loss
             "rotate": ((1, 2), ),
             "smooth": 2
            }, 
            profile)
        
        # ABA P56 (developing mouse) specific settings
        settings.add_modifier(
            "abap56", 
            {"target_size": (528, 320, 456),
             "resize_factor": None, # turn off resizing
             # stained sections and labels almost but not symmetric
             "labels_mirror": 0.5,
             # set edge since some lateral labels are only partially complete
             "labels_edge": 0.138, 
             "smooth": 2
            }, 
            profile)
        
        # ABA P56 (adult) specific settings
        settings.add_modifier(
            "abap56adult", 
            {"target_size": (528, 320, 456), # same atlas image as ABA P56dev
             "resize_factor": None, # turn off resizing
             # same stained sections as for P56dev; 
             # labels are already mirrored starting at z=228, but atlas is not
             # here, so mirror starting at the same z-plane to make both sections 
             # and labels symmetric and aligned with one another
             "labels_mirror": 0.5,
             "labels_edge": None, 
             "smooth": 2
            }, 
            profile)
        
        # turn off edge extension along with smoothing
        settings.add_modifier(
            "noedge", 
            {"labels_edge": None, 
             "smooth": None
             },
            profile)
        
        # turn off mirroring along with smoothing and edge extension
        settings.add_modifier(
            "nomirror", 
            {"labels_mirror": None,
             "labels_edge": None, 
             "smooth": None
             },
            profile)
        
        # turn off most image manipulations to show original atlas and labels 
        # while allowing transformations set as command-line arguments
        settings.add_modifier(
            "raw", 
            {"labels_mirror": None,
             "labels_edge": None, 
             "expand_labels": None, 
             "rotate": None, 
             "affine": None,
             "smooth": None, 
             "crop_to_labels": False, 
            }, 
            profile)
        
        # turn off label smoothing
        settings.add_modifier(
            "nosmooth", 
            {"smooth": None}, 
            profile)
        
        # turn off labels markers generation
        settings.add_modifier(
            "nomarkers", 
            {"marker_erosion": None}, 
            profile)
        
        # turn off cropping atlas to extent of labels
        settings.add_modifier(
            "nocropatlas", 
            {"crop_to_labels": False}, 
            profile)
        
        # turn off cropping labels to original size
        settings.add_modifier(
            "nocroplabels", 
            {"crop_to_orig": False}, 
            profile)
        
        # enable label smoothing
        settings.add_modifier(
            "smoothtest", 
            {"smooth": (0, 1, 2, 3, 4, 5)},#, 10)}, 
            #{"smooth": (0, )}, # TESTING
            profile)
        
        # groupwise registration batch 02
        settings.add_modifier(
            "grouped02", 
            {"bspline_grid_space_voxels": "70", 
             "grid_spacing_schedule": [
                "8.0", "7.0", "6.0", "5.0", "4.0", "3.0", "2.0", "1.0"], 
             "carve_threshold": 0.009}, 
            profile)
            
        # groupwise registration batch 04
        settings.add_modifier(
            "grouped04", 
            {"carve_threshold": 0.015}, 
            profile)
        
        # crop anterior region of labels during single registration
        settings.add_modifier(
            "cropanterior", 
            {"truncate_labels": (None, (0.2, 0.8), (0.45, 1.0))}, 
            profile)
        
        # turn off image curation to avoid post-processing with carving 
        # and in-painting
        settings.add_modifier(
            "nopostproc", 
            {"curate": False, 
             "truncate_labels": None}, 
            profile)
    
        # smoothing by Gaussian blur
        settings.add_modifier(
            "smoothgaus", 
            {"smoothing_mode": SmoothingModes.gaussian, 
             "smooth": 0.25}, 
            profile)
    
        # smoothing by Gaussian blur
        settings.add_modifier(
            "smoothgaustest", 
            {"smoothing_mode": SmoothingModes.gaussian, 
             "smooth": (0, 0.25, 0.5, 0.75, 1, 1.25)}, 
            profile)
    
    if verbose:
        print("process settings for {}:\n{}"
              .format(settings["settings_name"], settings))
    

register_settings = RegisterSettings()


# IMAGE FILES

SUFFIX_IMG_PROC = "_image5d_proc.npz"
SUFFIX_INFO_PROC = "_info_proc.npz"

filename = None # current image file path
filenames = None # list of multiple image paths
series = 0 # series for multi-stack files
channel = None # channel of interest, where None specifies all channels

prefix = None # alternate path
suffix = None # modifier to existing base path

PLANE = ("xy", "xz", "yz")
plane = None
vmax_overview = [None]
border = None # clip ROI to border (x,y,z order)
near_max = [-1.0]
near_min = [0.0]

# DATABASE

DB_NAME = "clrbrain.db"
db_name = DB_NAME # path to main DB
db = None # main DB
truth_db_name = None # path to truth DB
truth_db = None # truth blobs DB
verified_db = None # automated verifications DB

# receiver operating characteristic
roc = False
'''
roc_dict = OrderedDict([
    ("threshold_local", OrderedDict([
        ("thresholding", "local"),
        ("thresholding_size", np.arange(9, 75, 2))])
    ),
    ("threshold_otsu", OrderedDict([
        ("thresholding", "otsu"),
        ("thresholding_size", np.array([64, 128, 256, 512, 1024]))])
    ),
    ("random-walker", OrderedDict([
        ("thresholding", "random_walker"),
        ("thresholding_size", np.array([50.0, 130.0, 250.0, 500.0]))])
    )
])
'''

# scale factors
_scale_zs = np.arange(0.57, 0.61, 0.01)
_scale_factors = np.ones((len(_scale_zs), 3))
_scale_factors[:, 0] = _scale_zs
#print(_scale_factors)

# isotropic factors
_isotropic_zs = np.arange(0.9, 1.1, 0.02)
_isotropic_factors = np.ones((len(_isotropic_zs), 3))
_isotropic_factors[:, 0] = _isotropic_zs
#print(_isotropic_factors)

# pruning tolerance factors
_prune_tol_zs = np.arange(2.5, 4.6, 0.5)
_prune_tol_factors = np.ones((len(_prune_tol_zs), 3)) * 1.3
_prune_tol_factors[:, 0] = _prune_tol_zs
#print(_isotropic_factors)

roc_dict = OrderedDict([
    ("hyperparameters", OrderedDict([
        # test single value by iterating on value that should not affect 
        # detection ability
        ("points_3d_thresh", [0.7]),
        
        # unfused baseline
        #("scale_factor", 0.59),
        #("clip_vmax", 98.5),
        #("clip_max", 0.5),
        #("scale_factor", np.array([(0.59, 1, 1)])),
        #("clip_vmax", np.arange(98.5, 99, 0.5)),
        #("clip_max", np.arange(0.5, 0.6, 0.1)),
        
        # test parameters
        #("scale_factor", _scale_factors),
        #("scale_factor", np.array([(0.6, 1, 1)])),
        #("segmenting_mean_thresh", -5),
        #("isotropic", _isotropic_factors),
        #("isotropic", np.array([(0.96, 1, 1)])),
        #("overlap", np.arange(0.1, 1.0, 0.1)),
        #("prune_tol_factor", np.array([(4, 1.3, 1.3)])),
        #("prune_tol_factor", _prune_tol_factors),
        #("clip_min", np.arange(0.0, 0.2, 0.1)),
        #("clip_vmax", np.arange(97, 100.5, 0.5)),
        #("clip_max", np.arange(0.3, 0.7, 0.1)),
        #("erosion_threshold", np.arange(0.16, 0.35, 0.02)),
        #("segmenting_mean_thresh", np.arange(0.2, 0.8, 0.1)),
        #("segmenting_mean_thresh", np.arange(-5, -4.9, 0.1)),
        #("segmenting_mean_thresh", np.arange(5, 5.1, 0.1)),
        #"denoise_size", np.arange(5, 25, 2)
        #("unsharp_strength", np.arange(0.0, 1.1, 0.1)),
        #("tot_var_denoise", (False, True)),
        #("min_sigma_factor", np.arange(2.5, 3.6, 0.1)),
        #("max_sigma_factor", np.arange(3.5, 4.6, 0.1)),
        #("num_sigma", np.arange(5, 16, 1)),
        #("detection_threshold", np.arange(0.001, 0.01, 0.001)),
    ]))
])


# default colors using 7-color palatte for color blindness
# (Wong, B. (2011) Nature Methods 8:441)
colors = np.array(
    [[213, 94, 0], # vermillion
     [0, 114, 178], # blue
     [204, 121, 167], # reddish purple
     [230, 159, 0], # orange
     [86, 180, 233], # sky blue
     [0, 158, 115], # blullish green
     [240, 228, 66], # yellow
     [0, 0, 0]] # black
)



# RC PARAMETERS FOR MATPLOTLIB

# global setting changes
rc_params = {
    "image.interpolation": "bilinear",
    "image.resample": False
}

# Matplotlib2 default image interpoloation
rc_params_mpl2_img_interp = {
    "image.interpolation": "nearest",
    "image.resample": True
}


