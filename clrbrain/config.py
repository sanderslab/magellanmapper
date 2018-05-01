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

# PROCESSING SETTINGS

class ProcessSettings(dict):
    def __init__(self, *args, **kwargs):
        self["microscope_type"] = "default"
        self["vis_3d"] = "points"
        self["points_3d_thresh"] = 0.85
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
        self["denoise_size"] = 25
        self["segment_size"] = 500
        self["prune_tol_factor"] = (1, 1, 1)
        self["segmenting_mean_thresh"] = 0.4
        self["scale_factor"] = (1, 1, 1)
        self["channel_colors"] = (CMAP_GRBK_NAME, CMAP_RDBK_NAME)
        self["isotropic"] = None
        self["isotropic_vis"] = None

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
        settings["microscope_type"] = "2p_20x"
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
        
    elif settings_type.startswith("lightsheet_5x-v01"):
        # detection settings from pre-v.0.6.2
        settings["microscope_type"] = "lightsheet_5x-v01"
        #settings["vis_3d"] = "surface"
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
        
    elif settings_type.startswith("lightsheet"):
        # detection settings from v.0.6.2+, based on lightsheet 5x
        settings["microscope_type"] = "lightsheet"
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
        #settings["isotropic_vis"] = (1, 3, 3)
    
    
    
    # PROFILE MODIFIERS
    # any/all/none can be combined with any main profile, modifiers lower in 
    # this listing taking precedence over prior ones and the main profile
    
    if "_zebrafish" in settings_type:
        settings["microscope_type"] += "_zebrafish"
        settings["min_sigma_factor"] = 2.5
        settings["max_sigma_factor"] = 3
    
    if "_contrast" in settings_type:
        settings["microscope_type"] += "_contrast"
        settings["channel_colors"] = ("inferno", "bone")
  
    if "_cytoplasm" in settings_type:
        settings["microscope_type"] += "_cytoplasm"
        settings["clip_min"] = 0.3
        settings["clip_max"] = 0.8
        settings["points_3d_thresh"] = 0.7
        settings["min_sigma_factor"] = 8
        settings["max_sigma_factor"] = 20
        settings["num_sigma"] = 10
        settings["overlap"] = 0.2
  
    if "_small" in settings_type:
        settings["microscope_type"] += "_small"
        settings["points_3d_thresh"] = 0.3 # used only if not surface
        settings["isotropic_vis"] = (1, 1, 1)

    if "_binary" in settings_type:
        settings["microscope_type"] = "_binary"
        settings["detection_threshold"] = 0.001
    
    if "_20x" in settings_type:
        settings["microscope_type"] += "_20x"
        # fit into ~32GB RAM instance after isotropic interpolation
        settings["segment_size"] = 50
    
    if "_exportdl" in settings_type:
        settings["microscope_type"] += "_exportdl"
        # export to deep learning framework with required dimensions
        settings["isotropic"] = (0.93, 1, 1)


# default settings and list of settings for each channel
process_settings = ProcessSettings()
process_settings_list = [process_settings]

def get_process_settings(i):
    settings = process_settings
    if len(process_settings_list) > i:
        settings = process_settings_list[i]
    return settings


# REGISTRATION SETTINGS

class RegisterSettings(dict):
    def __init__(self, *args, **kwargs):
        self["register_type"] = "default"
        self["translation_iter_max"] = "2048"
        self["affine_iter_max"] = "1024"
        self["bspline_iter_max"] = "256"
        self["bspline_grid_space_voxels"] = "50"
        self["groupwise_iter_max"] = "1024"
        self["resize_factor"] = 0.7

def update_register_settings(settings, settings_type):
    if settings_type.startswith("finer"):
        # more aggressive parameters for finer tuning
        settings["register_type"] = "finer"
        settings["bspline_iter_max"] = "512"
      
        if settings_type.endswith("_big"):
            # atlas is big relative to the experimental image, so need to 
            # more aggressively downsize the atlas
            settings["register_type"] += "_big"
            settings["resize_factor"] = 0.625

        elif settings_type.endswith("_group"):
            # registered to group-registered atlas assumes images are 
            # roughly the same size
            settings["register_type"] += "_group"
            settings["resize_factor"] = 1.0

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
db_name = DB_NAME
db = None # main DB
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



# max pixels of sub-stacks for stack processing (z, y, x order)
sub_stack_max_pixels = (1000, 1000, 1000)

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
REGISTER_TYPES = ("single", "group", "overlays", "volumes", "densities")
register_type = None
ABA_NAME = "name"
VOL_KEY = "volume"
BLOBS_KEY = "blobs"

# flip/rotate the image; the direction of change can be variable
flip = None

# groups, such as genotypes and sex or combos
groups = None


# STACK PROCESSING

rescale = None # rescale image
interval = None # interval
delay = None # delay time between images

# EXPORT IMAGES

border = None # clip ROI to border (x,y,z order)
