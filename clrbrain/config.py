#!/bin/bash
# Config file for shared settings
# Author: David Young, 2017

verbose = False

class ProcessSettings(dict):
    def __init__(self, *args, **kwargs):
        self["microscope_type"] = "lightsheet_5x"
        self["clip_vmax"] = 99.5
        self["tot_var_denoise"] = False
        self["unsharp_strength"] = 0.3
        self["points_3d_thresh"] = 1.3
        self["max_sigma_factor"] = 30
        self["num_sigma"] = 10
        self["overlap"] = 0.5
        self["random_walker"] = False
        self["segment_size"] = 500
        self["prune_tol_factor"] = (1, 1, 1)

def update_process_settings(settings, settings_type):
    if settings_type == "2p_20x":
        settings["microscope_type"] = settings_type
        #settings["clip_vmax"] = 70
        #settings["tot_var_denoise"] = True
        #settings["unsharp_strength"] = 1.0
        # smaller threhsold since total var denoising
        settings["points_3d_thresh"] = 1.1
        settings["max_sigma_factor"] = 8
        settings["num_sigma"] = 20
        settings["overlap"] = 0.7
        settings["random_walker"] = True
        settings["segment_size"] = 50
        #settings["prune_tol_factor"] = (1, 1, 1)


# defaults to lightsheet 5x settings
process_settings = ProcessSettings()