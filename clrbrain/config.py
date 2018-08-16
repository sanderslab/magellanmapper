#!/bin/bash
# Config file for shared settings
# Author: David Young, 2017, 2018
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

verbose = False
no_show = False
POS_THRESH = 0.001 # threshold for positive values for float comparison

# custom colormaps in plot_2d
CMAP_GRBK_NAME = "Green_black"
CMAP_RDBK_NAME = "Red_black"

# max pixels of sub-stacks for stack processing (z, y, x order), which can 
# be set from command-line and takes precedence over process settings to 
# allow custom configurations depending on instance type
sub_stack_max_pixels = None

# PROCESSING SETTINGS

class SettingsDict(dict):
    def __init__(self, *args, **kwargs):
        self["settings_name"] = "default"

    def add_modifier(self, mod_name, mods, settings_type=None):
        """Add a modifer dictionary, overwriting any existing settings 
        with values from this dictionary.
        
        Args:
            mod_name: Name of the modifier, which will be appended to the 
                name of the current settings.
            mods: Dictionary with keys matching default keys and values to 
                replace the correspondings values.
            settings_type: The full name of the final settings. If given,  
                the modifier will only be added if ``settings_type`` 
                contains ``mod_name`` within the string. Defaults to None, 
                in which case ``mods`` will be added regardless.
        """
        if settings_type and not mod_name in settings_type: return
        self["settings_name"] += mod_name
        for key in mods.keys():
            self[key] = mods[key]

class ProcessSettings(SettingsDict):
    def __init__(self, *args, **kwargs):
        super().__init__(self)
        self["settings_name"] = "default"
        self["vis_3d"] = "points"
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
    
    if settings_type.startswith("2p_20x"):
        settings["settings_name"] = "2p_20x"
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
        
    elif settings_type.startswith("lightsheet_v01"):
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
        
    elif settings_type.startswith("lightsheet_v02"):
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
    
    
    
    # PROFILE MODIFIERS
    # any/all/none can be combined with any main profile, modifiers lower in 
    # this listing taking precedence over prior ones and the main profile
    
    settings.add_modifier(
        "_zebrafish", 
        {"min_sigma_factor": 2.5,
         "max_sigma_factor": 3}, 
        settings_type)
    
    settings.add_modifier(
        "_contrast", 
        {"channel_colors": ("inferno", "bone")}, 
        settings_type)
    
    settings.add_modifier(
        "_cytoplasm", 
        {"clip_min": 0.3,
         "clip_max": 0.8,
         "points_3d_thresh": 0.7,
         "min_sigma_factor": 8,
         "max_sigma_factor": 20,
         "num_sigma": 10,
         "overlap": 0.2}, 
        settings_type)
    
    settings.add_modifier(
        "_small", 
        {"points_3d_thresh": 0.3, # used only if not surface
         "isotropic_vis": (1, 1, 1)}, 
        settings_type)
    
    settings.add_modifier(
        "_binary", 
        {"denoise_size": None,
         "detection_threshold": 0.001}, 
        settings_type)

    # fit into ~32GB RAM instance after isotropic interpolation
    settings.add_modifier(
        "_20x", 
        {"segment_size": 50}, 
        settings_type)

    # export to deep learning framework with required dimensions
    settings.add_modifier(
        "_exportdl", 
        {"isotropic": (0.93, 1, 1)}, 
        settings_type)

    # import from deep learning predicted image
    settings.add_modifier(
        "_importdl", 
        {"isotropic": None, # assume already isotropic
         "resize_blobs": (.2, 1, 1)}, 
        settings_type)
    
    # denoise settings when performing registration
    settings.add_modifier(
        "_register", 
        {"unsharp_strength": 1.5}, 
        settings_type)
    
    if verbose:
        print("process settings for {}:\n{}"
              .format(settings["settings_name"], settings))
    
# default settings and list of settings for each channel
process_settings = ProcessSettings()
process_settings_list = [process_settings]

def get_process_settings(i):
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
        self["groupwise_iter_max"] = "1024"
        self["resize_factor"] = 0.7
        self["preprocess"] = False
        self["point_based"] = False
        # erase labels outside of x,y,z (applied before transposition), 
        # where each val is (start, end), given as fractions, or None for the 
        # whole range
        self["truncate_labels"] = (None, (0.2, 1.0), (0.45, 1.0))
        self["labels_mirror"] = (None, 0.5)
        self["atlas_threshold"] = 10.0
        self["target_size"] = None # x,y,z in exp orientation threshold for 
        # carving and max size of small holes for removal, respectively
        self["carve_threshold"] = None
        self["holes_area"] = None
        # paste in region from first image during groupwise reg; 
        # x,y,z, same format as truncate_labels except in pixels
        self["extend_borders"] = None
        self["grid_spacing_schedule"] = None

def update_register_settings(settings, settings_type):
    if settings_type.startswith("finer"):
        # more aggressive parameters for finer tuning
        settings["settings_name"] = "finer"
        settings["bspline_iter_max"] = "512"
    
    elif settings_type.startswith("groupwise"):
        settings["settings_name"] = "groupwise"
        # groupwise registration
        settings["bspline_grid_space_voxels"] = "30"
        settings["carve_threshold"] = 0.009
        settings["holes_area"] = 10000
        # manually add variable tissue area from first image to create 
        # bigger atlas to register samples with this variable area
        settings["extend_borders"] = (None, (0, 100), (80, 140))
        settings["grid_spacing_schedule"] = [
            "8.0", "7.0", "6.0", "5.0", "4.0", "3.0", "2.0", "1.0"]
    
    elif settings_type.startswith("test"):
        settings["settings_name"] = "test"
        settings["target_size"] = (50, 50, 50)
    
    # atlas is big relative to the experimental image, so need to 
    # more aggressively downsize the atlas
    settings.add_modifier(
        "_big", 
        {"resize_factor": 0.625}, 
        settings_type)
    
    settings.add_modifier(
        "_new", 
        {"preprocess": True}, 
        settings_type)
    
    # registration to new atlas assumes images are roughly same size and
    # orientation (ie transposed) and already have mirrored labels aligned 
    # with the fixed image toward the bottom of the z-dimension
    settings.add_modifier(
        "_generated", 
        {"resize_factor": 1.0, 
         "truncate_labels": (None, (0.18, 1.0), (0.2, 1.0)),
         "labels_mirror": None}, # turn off mirroring
        settings_type)
    
    # atlas that uses groupwise image as the atlas itself should 
    # determine atlas threshold dynamically
    settings.add_modifier(
        "_grouped", 
        {"atlas_threshold": None}, 
        settings_type)
    
    # ABA E18pt5 specific settings
    settings.add_modifier(
        "_abae18pt5", 
        {"target_size": (278, 581, 370),
         "resize_factor": None}, # turn off resizing
        settings_type)
    
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



# flag to save ROI to file
saveroi = False

# IMAGE REGISTRATION

# reference atlas labels
load_labels = None
labels_img = None
labels_scaling = None
labels_ref = None
labels_ref_lookup = None
labels_level = None
labels_mirror = True
REGISTER_TYPES = (
    "single", "group", "overlays", "volumes", "densities", "export_vols", 
    "export_regions", "new_atlas")
register_type = None
ABA_NAME = "name"
VOL_KEY = "volume"
BLOBS_KEY = "blobs"

# flip/rotate the image; the direction of change can be variable
flip = None

# groups, such as genotypes and sex or combos
GROUPS_NUMERIC = {"WT": 0.0, "het": 0.5, "null":1.0}
groups = None


# STACK PROCESSING

rescale = None # rescale image
interval = None # interval
delay = None # delay time between images

# 2D IMAGES

PLANE = ("xy", "xz", "yz")
plane = None
vmax_overview = 1.0
border = None # clip ROI to border (x,y,z order)


# AWS

ec2_start = None
ec2_list = None
ec2_terminate = None

# SLACK NOTIFICATIONS

notify_url = None
notify_msg = None
notify_attach = None
