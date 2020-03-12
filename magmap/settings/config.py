#!/bin/bash
# Config file for shared settings
# Author: David Young, 2017, 2020
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
from enum import Enum

verbose = False
POS_THRESH = 0.001 # threshold for positive values for float comparison


# IMAGE FILES

#: str: Suffix for main image.
SUFFIX_IMAGE5D = "image5d.npy"
#: str: Suffix for metadata of main image.
SUFFIX_META = "meta.npz"
#: str: Suffix for ROI image.
SUFFIX_SUBIMG = "subimg.npy"
#: str: Suffix for blobs archive.
SUFFIX_BLOBS = "blobs.npz"
#: str: Suffix for blob clusters archive.
SUFFIX_BLOB_CLUSTERS = "blobclusters.npy"

filename = None # current image file path
filenames = None # list of multiple image paths
metadata_paths = None  # metadata file paths
metadatas = None  # metadata dicts
#: int: Selected series index for multi-stack files; None for no series.
series = None
channel = None # channel of interest, where None specifies all channels

# ROI settings in x,y,z
# TODO: change to z,y,x ordering
roi_offsets = None  # list of offsets
roi_offset = None  # current offset
roi_sizes = None  # list of regions of interest
roi_size = None  # current region of interest

# sub-image settings in z,y,x
subimg_offsets = None
subimg_sizes = None

image5d = None  # numpy image array
image5d_is_roi = False  # flag when image5d was loaded as an ROI
blobs = None  # blobs

#: :obj:`np.ndarray`: 2D array of shapes per time point in
# ``[n_time_point, n_shape]`` format in case image5d is not available
# TODO: consider simplify to single shape as 1D array
image5d_shapes = None

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

#: List[float]: Image resolutions as an array of dimensions (n, r),
# where each resolution r is a tuple in (z, y, x) order
resolutions = None
magnification = 1.0  #: float: objective magnification
zoom = 1.0  #: float: objective zoom

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
        "EXPORT_ROIS", "TRANSFORM", "ANIMATED", "EXPORT_BLOBS"
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
        "LINE_PLOT",  # generic line plot
    )
)
plot_2d_type = None

# plot label keys for command-line parsing
PlotLabels = Enum(
    "PlotLabels", (
        "TITLE", "X_LABEL", "Y_LABEL", "X_UNIT", "Y_UNIT", 
        "X_LIM", "Y_LIM",  # (min, max) for x-, y-axes
        "X_TICK_LABELS", "Y_TICK_LABELS", 
        "SIZE",  # in x,y 
        "LAYOUT",  # subplot layout in num of columns, rows
        "ALPHAS_CHL",  # alphas for main image's channels
        "X_COL", "Y_COL",  # columns from data frame to plot
        "GROUP_COL",  # data frame group column
        "WT_COL",  # weight column
        "ID_COL",  # ID column
        "ERR_COL",  # error column(s)
        "ZOOM_SHIFT",  # shift plot offset when zooming into ROI
        "HLINE",  # horizontal line, usually fn for each group
        "LEGEND_NAMES",  # names to display in legend
        "PADDING",  # figure tight layout padding
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


#: dict: Dictionary mapping function names as lower-case strings to functions.
STR_FN = {
    "mean": np.nanmean,
    "med": np.nanmedian,
}


#: str: Matplotlib style sheet
matplotlib_style = "seaborn"


class Themes(Enum):
    """GUI themes, where each theme currently contains RC parameters to
    apply to the Matplotlib style."""

    # TODO: consider integrating non-RC parameters such as widget_color
    # TODO: consider importing custom stylesheets as .yml files
    # TODO: consider combining with Matplotlib style sheet handling

    # default theme
    DEFAULT = {
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

    # dark theme
    DARK = {
        "text.color": "w",
        "axes.facecolor": "7a7a7a",
        "axes.edgecolor": "3b3b3b",
        "axes.labelcolor": "w",
        "xtick.color": "w",
        "ytick.color": "w",
        "grid.color": "w",
        "figure.facecolor": "3b3b3b",
        "figure.edgecolor": "3b3b3b",
        "savefig.facecolor": "3b3b3b",
        "savefig.edgecolor": "3b3b3b",
    }


#: List[Enum]: List of theme enums.
rc_params = [Themes.DEFAULT]

#: float: Base "color" value for Matplotlib widget elements such as buttons,
# which actually take intensity values as strings
widget_color = 0.85

# Matplotlib2 default image interpolation
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

#: List[float]: alpha levels for overlaid images (not channels), defaulting
# to main image, labels image; set first value to 0 to prevent display/export
# of main image, which typically must be loaded.
alphas = [1]

# show scale bars
scale_bar = True


# DATABASE

#: :class:`Enum`: Enum class for truth database settings.
TruthDB = Enum(
    "TruthDB", (
        "MODE",  # mode from TruthDBModes
        "PATH",  # path to DB
    )
)
truth_db_params = dict.fromkeys(TruthDB, None)

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

DB_NAME = "magmap.db"
db_name = DB_NAME  # path to main DB
db = None  # main DB
truth_db = None  # truth blobs DB
verified_db = None  # automated verifications DB


# IMAGE REGISTRATION

# atlas label keys for command-line parsing
AtlasLabels = Enum(
    "AtlasLabels", (
        "PATH_REF", "LEVEL", "ID",
        # generate colormap based on original colors, even if some are missing
        "ORIG_COLORS",
        # use symmetric colors, assuming symmetric label values from neg to
        # pos, centered on 0 (eg -5, -3, 0, 3, 5)
        "SYMMETRIC_COLORS",
        # show labels as binary image with transparent background and given
        # color (eg "black" or "white") as foreground
        "BINARY",
    )
)
# default to load original labels image if available for ID-color mapping
atlas_labels = dict.fromkeys(AtlasLabels, None)
atlas_labels[AtlasLabels.ORIG_COLORS] = 1
atlas_labels[AtlasLabels.SYMMETRIC_COLORS] = True


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
        "single",  # register atlas directory to single sample image
        "group",  # groupwise register multiple samples
        "register_rev",  # reverse register sample to atlas
        "overlays",
        "export_regions", "new_atlas", "import_atlas", "export_common_labels", 
        "convert_itksnap_labels", 
        "make_edge_images", "make_edge_images_exp", "merge_atlas_segs", 
        "reg_labels_to_atlas", 
        "vol_stats", "vol_compare", 
        "make_density_images", 
        "merge_atlas_segs_exp", "make_subsegs", "export_metrics_compactness", 
        "plot_smoothing_metrics", "smoothing_peaks", "smoothing_metrics_aggr", 
        "merge_images", "merge_images_channels",
        "labels_diff", "labels_diff_stats",
        "make_labels_level", "combine_cols", "zscores", "coefvar", "melt_cols",
        "plot_region_dev", "plot_lateral_unlabeled", 
        "plot_intens_nuc", 
        "pivot_conds",
        "meas_improvement",
        "cluster_blobs",
        "plot_knns",
        "plot_cluster_blobs",
    ]
)
register_type = None

# metric groups
MetricGroups = Enum(
    "MetricGroups", [
        "SHAPES",  # whole label morphology metrics
        "POINT_CLOUD",  # nuclei as point clouds
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
        "MERGE_CSVS",  # merge rows
        "MERGE_CSVS_COLS",  # merge columns based on ID
        "APPEND_CSVS_COLS",  # concatenate chosen columns
        "EXPS_BY_REGION",  # convert volume stats to experiments by region
        "EXTRACT_FROM_CSV",  # extract rows based on matching rows in given col
        "ADD_CSV_COLS",  # add columns with values to CSV
        "NORMALIZE",  # normalize metrics to a base condition within a CSV
        "MERGE_EXCELS",  # merge Excel files into sheets of single Excel file
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
    LEVEL = "Level"
    SIDE = "Side"
    CONDITION = "Condition"
    DSC_ATLAS_LABELS = "DSC_atlas_labels"
    DSC_ATLAS_LABELS_HEM = "DSC_atlas_labels_hemisphere"
    DSC_ATLAS_SAMPLE = "DSC_atlas_sample"
    DSC_ATLAS_SAMPLE_CUR = "DSC_atlas_sample_curated"
    SIMILARITY_METRIC = "Similarity_metric"
    LAT_UNLBL_VOL = "Lateral_unlabeled_volume"
    LAT_UNLBL_PLANES = "Lateral_unlabeled_planes"
    OFFSET = "Offset"
    SIZE = "Size"
    CHANNEL = "Channel"


class HemSides(Enum):
    """Hemisphere side enumerations."""
    RIGHT = "R"
    LEFT = "L"
    BOTH = "both"


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


#: List[str]: Receiver operating characteristic/grid search profiles.
roc = None

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
