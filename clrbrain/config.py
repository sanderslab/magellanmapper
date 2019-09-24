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
    labels_ref_lookup: Reference dictionary with keys corresponding to the IDs 
        in the labels image.
"""

import numpy as np
from collections import OrderedDict
from enum import Enum

verbose = False
POS_THRESH = 0.001 # threshold for positive values for float comparison


# IMAGE FILES

SUFFIX_IMG_PROC = "_image5d_proc.npz"
SUFFIX_INFO_PROC = "_info_proc.npz"

filename = None # current image file path
filenames = None # list of multiple image paths
series = 0 # series for multi-stack files
channel = None # channel of interest, where None specifies all channels
roi_sizes = None # list of regions of interest
offsets = None # list of offsets
image5d_is_roi = False  # flag when image5d was loaded as an ROI

prefix = None # alternate path
suffix = None # modifier to existing base path

#: Tuple[str]: Plane orientations based on the two axes specifying the plane.
PLANE = ("xy", "xz", "yz")
plane = None
vmins = None # cmd-line specified
vmaxs = None
# generated from near_max; overwritten at cmd-line
vmax_overview = [None]
border = None # clip ROI to border (x,y,z order)
near_max = [-1.0] # auto-detected, max of clipped intensities of whole img
near_min = [0.0]
cmaps = None


# MICROSCOPY

# image resolutions as an array of dimensions (n, r),
# where each resolution r is a tuple in (z, y, x) order
resolutions = None
magnification = -1.0  # objective magnification
zoom = -1.0  # objective zoom

#: :class:`Enum`: main processing tasks
# PROC_TYPES: Processing modes. ``importonly`` imports an image stack and 
# exits non-interactively. ``processing`` processes and segments the 
# entire image stack and exits non-interactively. ``load`` loads already 
# processed images and segments. ``extract`` extracts a single plane 
# using the z-value from the offset and exits. ``export_rois`` 
# exports ROIs from the current database to serial 2D plots. 
# ``transpose`` transposes the Numpy image file associated with 
# ``filename`` with the ``--rescale`` option. ``animated`` generates 
# an animated GIF with the ``--interval`` and ``--rescale`` options. 
# ``export_blobs`` exports blob coordinates/radii to compressed CSV file.
ProcessTypes = Enum(
    "ProcessTypes", (
        "IMPORT_ONLY", "PROCESSING", "PROCESSING_MP", "LOAD", "EXTRACT", 
        "EXPORT_ROIS", "TRANSPOSE", "ANIMATED", "EXPORT_BLOBS"
    )
)
proc_type = None

# 2D PLOTTING

# custom colormaps in plot_2d
class Cmaps(Enum):
    CMAP_GRBK_NAME = "Green_black"
    CMAP_RDBK_NAME = "Red_black"

# processing type directly in module
Plot2DTypes = Enum(
    "Plot2DTypes", (
        "BAR_PLOT", "BAR_PLOT_VOLS_STATS", "BAR_PLOT_VOLS_STATS_EFFECTS", 
        "ROC_CURVE", "SCATTER_PLOT",
    )
)
plot_2d_type = None

# plot label keys for command-line parsing
PlotLabels = Enum(
    "PlotLabels", (
        "TITLE", "X_LABEL", "Y_LABEL", "X_UNIT", "Y_UNIT", 
        "X_TICK_LABELS", "Y_TICK_LABELS", 
        "SIZE",  # in x,y 
        "LAYOUT",  # subplot layout in num of columns, rows
        "ALPHAS_CHL",  # alphas for main image's channels
        "X_COL", "Y_COL",  # columns from data frame to plot
        "GROUP_COL",  # data frame group column
    )
)
plot_labels = dict.fromkeys(PlotLabels, None)

# image transformation keys for command-line parsing
Transforms = Enum(
    "Transforms", (
        "ROTATE", "FLIP_VERT", "FLIP_HORIZ"
    )
)
transform = dict.fromkeys(Transforms, None)

# extensions for saving figures
FORMATS_3D = ("obj", "x3d") # save 3D renderings
savefig = None # save files using this extension

# style sheet
matplotlib_style = "seaborn"

# global setting changes
rc_params = {
    "image.interpolation": "bilinear",
    "image.resample": False, 
    "font.family": "sans-serif", 
    # dejavusans is Matplotlib default but not on Mac by default, so 
    # need to change for PDF export; still falls back to DejaVuSans if 
    # none else found for display
    "font.sans-serif": ["Arial", "Helvetica", "Tahoma"], 
    # some styles use strings; change to num for numerical adjustments
    "axes.titlesize": 12,
}

# Matplotlib2 default image interpoloation
rc_params_mpl2_img_interp = {
    "image.interpolation": "nearest",
    "image.resample": True
}


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

# alpha levels for overlaid images, defaulting to main image, labels image; 
# set first value to 0 to prevent display/export of main image, which 
# typically must be loaded
alphas = [1, 0.9, 0.9]

# show scale bars
scale_bar = True


# DATABASE

#: :class:`Enum`: Enum class for truth database modes. ``view`` loads the truth 
# database corresponding to the filename and any offset/size to show 
# alongside the current database. ``verify`` creates a new database 
# to store results from ROC curve building. ``verified`` loads the  
# verified database generated from the prior mode.
TruthDBModes = Enum(
    "TruthDBModes", (
        "VIEW", "VERIFY", "VERIFIED", "EDIT" 
    )
)

#: :obj:`TruthDBModes`: truth database mode enum
truth_db_mode = None

DB_NAME = "clrbrain.db"
db_name = DB_NAME # path to main DB
db = None # main DB
truth_db_name = None # path to truth DB
truth_db = None # truth blobs DB
verified_db = None # automated verifications DB


# IMAGE REGISTRATION

# atlas label keys for command-line parsing
AtlasLabels = Enum(
    "AtlasLabels", (
        "PATH_REF", "LEVEL", "ID", "ORIG_COLORS" 
    )
)
# default to load original labels image if available for ID-color mapping
atlas_labels = dict.fromkeys(AtlasLabels, None)
atlas_labels[AtlasLabels.ORIG_COLORS] = 1


# registered image suffixes
class RegNames(Enum):
    IMG_ATLAS = "atlasVolume.mhd"
    IMG_ATLAS_PRECUR = "atlasVolumePrecur.mhd"
    IMG_LABELS = "annotation.mhd"
    IMG_EXP = "exp.mhd"
    IMG_GROUPED = "grouped.mhd"
    IMG_BORDERS = "borders.mhd"  # TODO: consider removing
    IMG_HEAT_MAP = "heat.mhd"
    IMG_ATLAS_EDGE = "atlasEdge.mhd"
    IMG_ATLAS_LOG = "atlasLoG.mhd"
    IMG_LABELS_TRUNC = "annotationTrunc.mhd"
    IMG_LABELS_EDGE = "annotationEdge.mhd"
    IMG_LABELS_DIST = "annotationDist.mhd"
    IMG_LABELS_MARKERS = "annotationMarkers.mhd"
    IMG_LABELS_INTERIOR = "annotationInterior.mhd"
    IMG_LABELS_SUBSEG = "annotationSubseg.mhd"
    IMG_LABELS_DIFF = "annotationDiff.mhd"
    IMG_LABELS_LEVEL = "annotationLevel{}.mhd"
    IMG_LABELS_EDGE_LEVEL = "annotationEdgeLevel{}.mhd"
    COMBINED = "combined.mhd"  # spliced into other registered names


# reference atlas labels
load_labels = None
labels_img = None  # in Numpy format
labels_img_orig = None  # in Numpy format
labels_scaling = None
labels_ref_lookup = None
labels_level = None
labels_mirror = True
borders_img = None
VOL_KEY = "volume"
BLOBS_KEY = "blobs"
VARIATION_BLOBS_KEY = "var_blobs" # variation in blob density
VARIATION_EXP_KEY = "var_exp" # variation in experiment intensity
SIDE_KEY = "Side"
GENOTYPE_KEY = "Geno"
SUB_SEG_MULT = 100 # labels multiplier for sub-segmentations
REGION_ALL = "all"

# registered image suffix keys for command-line parsing
RegSuffixes = Enum(
    "RegSuffixes", [
        "ATLAS", "ANNOTATION", "BORDERS", 
    ]
)
reg_suffixes = dict.fromkeys(RegSuffixes, None)

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
        "single", "group", "overlays", 
        "export_regions", "new_atlas", "import_atlas", "export_common_labels", 
        "convert_itksnap_labels", 
        "make_edge_images", "make_edge_images_exp", "merge_atlas_segs", 
        "reg_labels_to_atlas", "vol_stats", "make_density_images", 
        "merge_atlas_segs_exp", "make_subsegs", "export_metrics_compactness", 
        "plot_smoothing_metrics", "smoothing_peaks", "smoothing_metrics_aggr", 
        "merge_images", "merge_images_channels", 
        "register_reg", "labels_diff", "labels_diff_stats", 
        "make_labels_level", "combine_cols", "zscores", "coefvar", "melt_cols",
        "plot_region_dev", "plot_lateral_unlabeled", 
        "plot_intens_nuc", 
        "pivot_conds",
    ]
)
register_type = None

# metric groups
MetricGroups = Enum(
    "MetricGroups", [
        "SHAPES", 
    ]
)

# flip/rotate the image; the direction of change can be variable
flip = None

# groups, such as genotypes and sex or combos
GROUPS_NUMERIC = {"WT": 0.0, "het": 0.5, "null":1.0}
groups = None

# smoothing metrics
PATH_SMOOTHING_METRICS = "smoothing.csv"

# raw smoothing metrics (individual labels)
PATH_SMOOTHING_RAW_METRICS = "smoothing_raw.csv"

# whole atlas image import metrics
PATH_ATLAS_IMPORT_METRICS = "stats.csv"

# common labels
PATH_COMMON_LABELS = "regions_common.csv"

class ItkSnapLabels(Enum):
    """Column names to use for ITK-SNAP description labels.
    
    Labels description file is assumed to have this column ordering.
    """
    ID = ABAKeys.ABA_ID.value
    R = "r"
    G = "g"
    B = "b"
    A = "a"
    VIS = "vis"
    MESH = "mesh"
    NAME = ABAKeys.NAME.value


# STATS

#: :class:`Enum`: stats module processing types
StatsTypes = Enum(
    "StatsTypes", [
        "MERGE_CSVS", "EXPS_BY_REGION", "EXTRACT_FROM_CSV"
    ]
)
stats_type = None
seed = 0 # random number generator seed

#: float: measurement unit factor to convert to next larger prefix (eg um to mm)
unit_factor = 1000.0


class AtlasMetrics(Enum):
    """General atlas metric enumerations."""
    SAMPLE = "Sample"
    REGION = "Region"
    REGION_ABBR = "RegionAbbr"
    CONDITION = "Condition"
    DSC_ATLAS_LABELS = "DSC_atlas_labels"
    DSC_ATLAS_LABELS_HEM = "DSC_atlas_labels_hemisphere"
    DSC_ATLAS_SAMPLE = "DSC_atlas_sample"
    DSC_ATLAS_SAMPLE_CUR = "DSC_atlas_sample_curated"
    LAT_UNLBL_VOL = "Lateral_unlabeled_volume"
    LAT_UNLBL_PLANES = "Lateral_unlabeled_planes"
    OFFSET = "Offset"
    SIZE = "Size"
    CHANNEL = "Channel"


# label smoothing modes
SmoothingModes = Enum(
    "SmoothingModes", [
        "opening", "gaussian", "closing"
    ]
)


class SmoothingMetrics(Enum):
    """Smoothing metric enumerations.
    
    Generally with reference to the smoothed stat, so original stats will 
    have an "orig" suffix, while smoothed stats will not have any suffix.
    """
    COMPACTION = "Compaction"
    DISPLACEMENT = "Displacement"
    SM_QUALITY = "Smoothing_quality"
    VOL_ORIG = "Vol_orig"
    VOL = "Vol"
    COMPACTNESS_ORIG = "Compactness_orig"
    COMPACTNESS = "Compactness"
    COMPACTNESS_SD = "Compactness_SD"
    COMPACTNESS_CV = "Compactness_CV"
    SA_VOL_ORIG = "SA_to_vol_orig"
    SA_VOL = "SA_to_vol"
    SA_VOL_FRAC = "SA_to_vol_frac"
    LABEL_LOSS = "Label_loss"
    FILTER_SIZE = "Filter_size"


# AWS

ec2_start = None
ec2_list = None
ec2_terminate = None


# SLACK NOTIFICATIONS

notify_url = None
notify_msg = None
notify_attach = None


# MESSAGES

WARN_IMPORT_SCALEBAR = (
    "Matplotlib ScaleBar could not be found, so scale bars will not be "
    "displayed")

# PROFILE SETTINGS

# microscope profile settings and list of settings for each channel
process_settings = None
process_settings_list = []
register_settings = None


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

# isotropic factors
#_isotropic_zs = np.arange(0.9, 1.1, 0.02)
_isotropic_zs = np.arange(0.1, 2, 0.1)
_isotropic_factors = np.ones((len(_isotropic_zs), 3))
_isotropic_factors[:, 0] = _isotropic_zs
#print(_isotropic_factors)

# pruning tolerance factors
_prune_tol_zs = np.arange(0.5, 1.1, 0.5)
_prune_tol_factors = np.ones((len(_prune_tol_zs), 3)) * 0.9
_prune_tol_factors[:, 0] = _prune_tol_zs
#print(_prune_tol_factors)

roc_dict = OrderedDict([
    ("hyperparameters", OrderedDict([
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
        #("min_sigma_factor", np.arange(2, 2.71, 0.1)),
        #("max_sigma_factor", np.arange(2.7, 3.21, 0.1)),
        #("min_sigma_factor", np.arange(2.5, 3.51, 0.1)),
        #("max_sigma_factor", np.arange(3.5, 4.51, 0.1)),
        #("num_sigma", np.arange(5, 16, 1)),
        #("detection_threshold", np.arange(0.001, 0.01, 0.001)),
        #("segment_size", np.arange(130, 160, 20)),
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
