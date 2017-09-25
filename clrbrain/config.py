#!/bin/bash
# Config file for shared settings
# Author: David Young, 2017

import numpy as np
from collections import OrderedDict

verbose = False

class ProcessSettings(dict):
    def __init__(self, *args, **kwargs):
        self["microscope_type"] = "lightsheet_5x"
        self["vis_3d"] = "points"
        self["clip_vmax"] = 99.5
        self["clip_min"] = 0.2
        self["clip_max"] = 1.0
        self["tot_var_denoise"] = False
        self["unsharp_strength"] = 0.3
        self["points_3d_thresh"] = 0.7
        self["min_sigma_factor"] = 3
        self["max_sigma_factor"] = 30
        self["num_sigma"] = 10
        self["overlap"] = 0.5
        self["thresholding"] = None
        self["thresholding_size"] = -1
        self["denoise_size"] = 25
        self["segment_size"] = 500
        self["prune_tol_factor"] = (1, 1, 1)
        self["segmenting_mean_thresh"] = 0.4
        self["scale_factor"] = (1, 1, 1)

def update_process_settings(settings, settings_type):
    if settings_type == "2p_20x":
        settings["microscope_type"] = settings_type
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
        settings["scale_factor"] = (1, 1, 1)
        
    elif settings_type == "2p_20x_zebrafish":
        settings["microscope_type"] = settings_type
        settings["vis_3d"] = "surface"
        settings["clip_vmax"] = 97
        settings["clip_min"] = 0
        settings["clip_max"] = 0.7
        settings["tot_var_denoise"] = True
        settings["unsharp_strength"] = 2.5
        # smaller threhsold since total var denoising
        #settings["points_3d_thresh"] = 1.1
        settings["min_sigma_factor"] = 2.5
        settings["max_sigma_factor"] = 3
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
        settings["scale_factor"] = (1, 1, 1)
        
    elif settings_type == "lightsheet_5x_02":
        settings["microscope_type"] = settings_type
        settings["vis_3d"] = "surface"
        settings["clip_vmax"] = 99.5
        settings["clip_min"] = 0
        settings["clip_max"] = 0.6
        settings["tot_var_denoise"] = False
        settings["unsharp_strength"] = 1.5
        settings["points_3d_thresh"] = 0.7
        settings["min_sigma_factor"] = 3
        settings["max_sigma_factor"] = 4
        settings["num_sigma"] = 10
        settings["overlap"] = 0.5
        settings["thresholding"] = None
        settings["thresholding_size"] = 64
        settings["denoise_size"] = 25
        settings["segment_size"] = 200
        settings["prune_tol_factor"] = (3, 1.3, 1.3)
        settings["segmenting_mean_thresh"] = 0.5
        settings["scale_factor"] = (0.63, 1, 1)


# defaults to lightsheet 5x settings
process_settings = ProcessSettings()

# main DB
db = None

# truth blobs DB
truth_db = None

# automated verifications DB
verified_db = None

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

roc_dict = OrderedDict([
    ("threshold_otsu", OrderedDict([
        ("thresholding", "otsu"),
        ("thresholding_size", np.array([64]))])
        #("thresholding_size", np.arange(32, 128, 4))])
        #("thresholding_size", np.array([64, 128, 256, 512, 1024]))])
    )
])
'''
roc_dict = OrderedDict([
    ("overlap_thresh", OrderedDict([
        ("overlap", np.arange(0.5, 0.6, 0.1)),
        #("clip_min", np.arange(0.0, 0.1, 0.1)),
        #("clip_vmax", np.arange(95, 100.5, 0.5)),
        #("clip_max", np.arange(0.6, 0.7, 0.1)),
        #"segmenting_mean_thresh", np.arange(0.2, 0.8, 0.1),
        #"denoise_size", np.arange(5, 25, 2)
        #("unsharp_strength", np.arange(0.0, 1.1, 0.1)),
        #("tot_var_denoise", (False, True)),
        #("min_sigma_factor", np.arange(2.5, 3.6, 0.1)),
        #("max_sigma_factor", np.arange(3.5, 4.6, 0.1)),
        #("num_sigma", np.arange(5, 16, 1)),
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

# max pixels for x- and y- dimensions of sub-stacks for stack processing
sub_stack_max_pixels = 1000
