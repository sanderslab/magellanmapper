# Config file for shared settings
# Author: David Young, 2017, 2020
"""Configuration storage module.

This module allows customization of settings for various imaging systems, 
such as grouped settings for particular microscopes. Additional parameters 
such as command-line flag settings and databases can also be stored here 
for program access.

"""

from enum import Enum, auto
import pathlib
from typing import Any, Dict, Optional, Sequence, TYPE_CHECKING

try:
    from appdirs import AppDirs
except ImportError as e:
    raise ImportError(
        "The appdirs package requirement was added in v1.4.0. Please install "
        "it by one of these methods:\n"
        "- Conda env: run `bin/setup_conda` (Mac/Linux) or "
        "`bin\\setup_conda.bat (Windows)\n"
        "- Venv env: run `bin/setup_venv.sh`\n"
        "- Or after activating your env: run `pip install appdirs`") from e
import numpy as np

from magmap.settings import logs

if TYPE_CHECKING:
    from magmap.atlas import labels_meta

#: str: Application name.
APP_NAME = "MagellanMapper"
#: str: Uniform Resource Identifier scheme.
URI_SCHEME = "magmap"
#: str: Reverse Domain Name System identifier.
DNS_REVERSE = f"io.github.sanderslab.{APP_NAME}"
#: float: Threshold for positive values for float comparison.
POS_THRESH = 0.001
#: int: Number of CPUs for multiprocessing tasks; defaults to None to
# use the number determined by the CPU count.
cpus = None
#: PurePath: Application root directory path.
app_dir = pathlib.Path(__file__).resolve().parent.parent.parent
#: str: Accessor to application-related user directories.
user_app_dirs = AppDirs(APP_NAME, False)
#: PurePath: Absolution path to main application icon.
ICON_PATH = app_dir / "images" / "magmap.png"


# LOGGING

class Verbosity(Enum):
    LEVEL = auto()
    LOG_PATH = auto()


#: dict: Command-line arguments for verbosity.
verbosity = dict.fromkeys(Verbosity, None)

#: bool: True for verbose debugging output.
verbose = False

#: :class:`logging.Logger`: Root logger for the application.
logger = logs.setup_logger()

#: Path to log file.
log_path: Optional[pathlib.Path] = None


# IMAGE FILES

#: str: Suffix for main image.
SUFFIX_IMAGE5D = "image5d.npy"
#: str: Suffix for metadata of main image.
SUFFIX_META = "meta.yml"
#: str: Suffix for ROI image.
SUFFIX_SUBIMG = "subimg.npy"
#: str: Suffix for blobs archive.
SUFFIX_BLOBS = "blobs.npz"
#: str: Suffix for blob clusters archive.
SUFFIX_BLOB_CLUSTERS = "blobclusters.npy"

#: Current image file base path; eg for the image path,
# ``/opt/myvolume_image5d.npy``, the base path is ``/opt/myvolume``.
filename: Optional[str] = None
#: List[str]: List of multiple image paths.
filenames = None
#: List[str]: Metadata file paths.
metadata_paths = None
#: List[dict]: Metadata dictionaries.
metadatas = None
#: int: Selected image series index for multi-stack files; None for no series.
series = None
#: list[int]: List of image series/tiles.
series_list = None
#: int: Channel of interest, where None specifies all channels.
channel = None

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
#: :obj:`magmap.io.np_io.Image5d`: Image5d object.
img5d = None

#: obj:`magmap.cv.detector.Blobs`: Blobs object.
blobs = None

#: :obj:`np.ndarray`: 2D array of shapes per time point in
# ``[n_time_point, n_shape]`` format in case image5d is not available
# TODO: consider simplify to single shape as 1D array
image5d_shapes = None


class LoadIO(Enum):
    """Enumerations for I/O load packages."""
    NP = auto()
    SITK = auto()


#: :obj:`LoadIO`: I/O source for image5d array.
image5d_io = None


class LoadData(Enum):
    """Enumerations for specifying data to load."""
    BLOBS = auto()
    BLOB_MATCHES = auto()


#: dict[:obj:`LoadData`, Any]: Data sources to load, where values that evaluate
# to True indicate to load the file.
# TODO: support specifying sources by paths
load_data = dict.fromkeys(LoadData, None)

# Path modifiers

#: Input/output base paths.
prefixes: Optional[Sequence[str]] = None
#: Input/output base path.
prefix: Optional[str] = None
#: Output base paths.
prefixes_out: Optional[Sequence[str]] = None
#: Output base path.
prefix_out: Optional[str] = None
# Modifiers to existing base paths.
suffixes: Optional[Sequence[str]] = None
#: Modifier to existing base path, typically inserted just before the extension.
suffix: Optional[str] = None

#: Tuple[str]: Plane orientations based on the two axes specifying the plane.
PLANE = ("xy", "xz", "yz")
plane = None
vmins = None  # cmd-line specified
vmaxs = None
# generated from near_max; overwritten at cmd-line
vmax_overview = [None]

#: List[float]: Auto-detected near maximum of clipped intensities from the
# whole image.
near_max = [-1.0]  # TODO: consider making None

#: List[float]: Auto-detected near minimum of clipped intensities from the
# whole image.
near_min = [0.0]

#: list[:class:`matplotlib.colors.Colormap`]: List of Matplotlib colormaps
# for the main image, :attr:`img5d`, with a colormap for each channel.
cmaps = None

#: :class:`magmap.plot.colormaps.DiscreteColormap`: Labels image colormap.
cmap_labels = None


# MICROSCOPY

# metadata keys for command-line parsing
MetaKeys = Enum(
    "MetaKeys", (
        "RESOLUTIONS",  # image resolutions in x,y,z
        "MAGNIFICATION",  # objective magnification
        "ZOOM",  # objective zoom
        "SHAPE",  # output image shape
        "DTYPE",  # data type as a string
    )
)
#: Dictionary of metadata for image import.
meta_dict: Dict[MetaKeys, Any] = dict.fromkeys(MetaKeys, None)

#: List[float]: Image resolutions as an array of dimensions (n, r),
# where each resolution r is a tuple in (z, y, x) order
resolutions = None
magnification = 1.0  #: float: objective magnification
zoom = 1.0  #: float: objective zoom


class PreProcessKeys(Enum):
    """Pre-processing task enumerations."""
    SATURATE = auto()
    DENOISE = auto()
    REMAP = auto()
    ROTATE = auto()


class ProcessTypes(Enum):
    """Whole image processing task enumerations."""
    IMPORT_ONLY = auto()  # imports image stack
    DETECT = auto()  # whole image blob detection
    DETECT_COLOC = auto()  # detection with colocalization by intensity
    COLOC_MATCH = auto()  # colocalization by blob matching
    LOAD = auto()  # DEPRECATED: load previously processed images and blobs
    EXTRACT = auto()  # extract single plane using the z-val from offset
    EXPORT_ROIS = auto()  # export ROIs from current database to serial 2D plots
    TRANSFORM = auto()  # transform image (see transformer.transpose_img)
    ANIMATED = auto()  # generate an animated GIF
    EXPORT_BLOBS = auto()  # export blob coordinates/radii to compressed CSV
    EXPORT_PLANES = auto()  # export a 3D+ image to individual planes
    EXPORT_PLANES_CHANNELS = auto()  # also export channels to separate files
    EXPORT_RAW = auto()  # export an array as a raw data file
    EXPORT_TIF = auto()  # export an array as TIF files for each channel
    PREPROCESS = auto()  # pre-process whole image


#: Processing tasks.
proc_type: Dict[ProcessTypes, Any] = dict.fromkeys(ProcessTypes, None)

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
class PlotLabels(Enum):
    TITLE = auto()  # figure title
    X_LABEL = auto()  # axis labels
    Y_LABEL = auto()
    X_UNIT = auto()  # axis units
    Y_UNIT = auto()
    X_LIM = auto()  # (min, max) for axis
    Y_LIM = auto()
    X_TICK_LABELS = auto()  # labels for axis tick marks
    Y_TICK_LABELS = auto()
    X_SCALE = auto()  # scaling, eg "log", "linear" (see Matplotlib)
    Y_SCALE = auto()
    SIZE = auto()  # in x,y
    LAYOUT = auto()  # subplot layout in num of columns, rows
    ALPHAS_CHL = auto()  # alphas for main image's channels
    X_COL = auto()  # column from data frame to plot
    Y_COL = auto()
    GROUP_COL = auto()  # data frame group column
    WT_COL = auto()  # weight column
    ID_COL = auto()  # ID column
    ERR_COL = auto()  # error column(s)
    ANNOT_COL = auto()  # annotation column for each point
    ZOOM_SHIFT = auto()  # shift plot offset when zooming into ROI
    HLINE = auto()  # horizontal line, usually fn for each group
    LEGEND_NAMES = auto()  # names to display in legend
    PADDING = auto()  # image padding, either as scalar or x,y,z
    MARGIN = auto()  # image margin, either as scalar or x,y,z
    SCALE_BAR = auto()  # True to include a scale bar
    MARKER = auto()  # Matplotlib marker style
    DROP_DUPS = auto()  # drop duplicates
    DPI = auto()  # dots per inch
    NAN_COLOR = auto()  # color for NaN values (Matplotlib or RGBA string)
    TEXT_POS = auto()  # text (annotation) position in x,y
    CONDITION = auto()  # condition


#: dict[Any]: Plot labels set from command-line.
plot_labels = dict.fromkeys(PlotLabels, None)
plot_labels[PlotLabels.SCALE_BAR] = True
plot_labels[PlotLabels.DPI] = 150.0

# image transformation keys for command-line parsing
Transforms = Enum(
    "Transforms", (
        "ROTATE",  # num of times to rotate by 90 deg
        "FLIP_VERT",  # 1 to invert top to bottom
        "FLIP_HORIZ",  # 1 to invert left to right
        "RESCALE",  # rescaling factor for an image shape
    )
)
transform = dict.fromkeys(Transforms, None)


# extensions for saving figures.

#: tuple[str, ...]: Extension for 3D renderings.
FORMATS_3D = ("obj", "x3d")
#: str: Default extension for saving figures.
DEFAULT_SAVEFIG = "png"
#: str: # File extension (without period) for saving figures.
savefig = DEFAULT_SAVEFIG


#: dict: Dictionary mapping function names as lower-case strings to functions.
STR_FN = {
    "mean": np.nanmean,
    "med": np.nanmedian,
}


#: str: Matplotlib style sheet.
matplotlib_style = "default"


class Themes(Enum):
    """GUI themes, where each theme currently contains RC parameters to
    apply to the Matplotlib style."""

    # TODO: consider integrating non-RC parameters such as widget_color
    # TODO: consider importing custom stylesheets as .yml files
    # TODO: consider combining with Matplotlib style sheet handling

    # default theme
    DEFAULT = {
        "font.family": "sans-serif",
        # dejavusans is Matplotlib default but not on Mac by default, so
        # need to change for PDF export; still falls back to DejaVuSans if
        # none else found for display
        "font.sans-serif": ["Arial", "Helvetica", "Tahoma"],
        # some styles use strings; change to num for numerical adjustments
        "axes.titlesize": 12,
        # turn off compositing to allow separating layers in vector graphics
        # output
        "image.composite_image": False,
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

#: bool: Show images after a task is performed, if applicable
show = False
max_scroll = 20  # max speed when scrolling through planes


# STACK PROCESSING

slice_vals = None  # list of slice values to give directly to slice fn
delay = None  # delay time between images


# IMAGE EXPORT

# flag to save a sub-image to file
save_subimg = False

#: List[float]: alpha levels for overlaid images (not channels), defaulting
# to main image, labels image; set first value to 0 to prevent display/export
# of main image, which typically must be loaded.
alphas = [1]


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
db_path = DB_NAME  # path to main DB
db = None  # main DB
truth_db = None  # truth blobs DB
verified_db = None  # automated verifications DB


# IMAGE REGISTRATION

# atlas label keys for command-line parsing
AtlasLabels = Enum(
    "AtlasLabels", (
        "PATH_REF",  # path to labels reference file
        "LEVEL",  # level of label
        "ID",  # label ID
        # generate colormap based on original colors, even if some are missing
        "ORIG_COLORS",
        # use symmetric colors, assuming symmetric label values from neg to
        # pos, centered on 0 (eg -5, -3, 0, 3, 5)
        "SYMMETRIC_COLORS",
        # sequence of colors as ``[background, foreground]``, where colors are
        # Matplotlib colors or RGB(A) hex values, to show labels as binary image
        "BINARY",
        # path to CSV file to translate labels
        # (see :meth:`ontology.replace_labels`)
        "TRANSLATE_LABELS",
        # True to translate labels and their children
        "TRANSLATE_CHILDREN",
    )
)
# default to load original labels image if available for ID-color mapping
atlas_labels: Dict[AtlasLabels, Any] = dict.fromkeys(AtlasLabels, None)
atlas_labels[AtlasLabels.ORIG_COLORS] = 1
atlas_labels[AtlasLabels.SYMMETRIC_COLORS] = True


# registered image suffixes
class RegNames(Enum):
    IMG_ATLAS = "atlasVolume.mhd"
    IMG_ATLAS_PRECUR = "atlasVolumePrecur.mhd"
    IMG_LABELS = "annotation.mhd"
    IMG_EXP = "exp.mhd"
    IMG_EXP_MASK = "expMask.mhd"
    IMG_GROUPED = "grouped.mhd"
    IMG_BORDERS = "borders.mhd"  # TODO: consider removing
    IMG_HEAT_MAP = "heat.mhd"
    IMG_HEAT_COLOC = "heatColoc.mhd"
    IMG_ATLAS_EDGE = "atlasEdge.mhd"
    IMG_ATLAS_LOG = "atlasLoG.mhd"
    IMG_ATLAS_MASK = "atlasMask.mhd"
    IMG_LABELS_PRECUR = "annotationPrecur.mhd"
    IMG_LABELS_TRUNC = "annotationTrunc.mhd"
    IMG_LABELS_EDGE = "annotationEdge.mhd"
    IMG_LABELS_DIST = "annotationDist.mhd"
    IMG_LABELS_MARKERS = "annotationMarkers.mhd"
    IMG_LABELS_INTERIOR = "annotationInterior.mhd"
    IMG_LABELS_SUBSEG = "annotationSubseg.mhd"
    IMG_LABELS_DIFF = "annotationDiff.mhd"
    IMG_LABELS_LEVEL = "annotationLevel{}.mhd"
    IMG_LABELS_EDGE_LEVEL = "annotationEdgeLevel{}.mhd"
    IMG_LABELS_TRANS = "annotationTrans.mhd"
    COMBINED = "combined.mhd"  # spliced into other registered names


#: Path to labels image metadata file. 

#: Loaded labels metadata.
labels_metadata: Optional["labels_meta.LabelsMeta"] = None

#: Path to the labels reference file.
load_labels: Optional[str] = None
#: Numpy array of a labels image file, typically corresponding to ``img5d``.
labels_img: Optional = None
#: Labels image as a SimpleITK Image instance.
labels_img_sitk: Optional[np.ndarray] = None
#: Original labels image, before any processing.
labels_img_orig: Optional[np.ndarray] = None
#: Scaling factors from ``labels_img`` to ``img5d``. 
labels_scaling: Optional[Sequence[float]] = None
#: Reference dictionary with keys corresponding to the IDs in the labels image.
labels_ref_lookup: Optional[Dict[str, Any]] = None
labels_level = None
labels_mirror = True
borders_img = None
VOL_KEY = "volume"
BLOBS_KEY = "blobs"
VARIATION_BLOBS_KEY = "var_blobs" # variation in blob density
VARIATION_EXP_KEY = "var_exp" # variation in experiment intensity
GENOTYPE_KEY = "Geno"
SUB_SEG_MULT = 100  # labels multiplier for sub-segmentations
REGION_ALL = "all"

# registered image suffix keys for command-line parsing
RegSuffixes = Enum(
    "RegSuffixes", [
        "ATLAS",  # intensity image
        "ANNOTATION",  # labels image
        "BORDERS",  # label borders image
        "FIXED_MASK",  # registration fixed image mask
        "MOVING_MASK",  # registration moving image mask
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
    PARENT_ID = "parent_structure_id"


# register module modes when called from command-line
RegisterTypes = Enum(
    "RegisterTypes", [
        "SINGLE",  # register atlas directory to single sample image
        "GROUP",  # groupwise register multiple samples
        "REGISTER_REV",  # reverse register sample to atlas
        "OVERLAYS",
        "EXPORT_REGIONS",
        "NEW_ATLAS",
        "IMPORT_ATLAS",
        "EXPORT_COMMON_LABELS",
        "CONVERT_ITKSNAP_LABELS",
        "MAKE_EDGE_IMAGES",
        "MAKE_EDGE_IMAGES_EXP",
        "MERGE_ATLAS_SEGS",
        "REG_LABELS_TO_ATLAS",
        "VOL_STATS",
        "VOL_COMPARE",
        "MAKE_DENSITY_IMAGES",
        "MERGE_ATLAS_SEGS_EXP",
        "MAKE_SUBSEGS",
        "EXPORT_METRICS_COMPACTNESS",
        "PLOT_SMOOTHING_METRICS",
        "SMOOTHING_PEAKS",
        "SMOOTHING_METRICS_AGGR",
        "MERGE_IMAGES",
        "MERGE_IMAGES_CHANNELS",
        "LABELS_DIFF",
        "LABELS_DIFF_STATS",
        "MAKE_LABELS_LEVEL",
        "COMBINE_COLS",
        "ZSCORES",
        "COEFVAR",
        "MELT_COLS",
        "PLOT_REGION_DEV",
        "PLOT_LATERAL_UNLABELED",
        "PLOT_INTENS_NUC",
        "PIVOT_CONDS",
        "MEAS_IMPROVEMENT",
        "CLUSTER_BLOBS",
        "PLOT_KNNS",
        "PLOT_CLUSTER_BLOBS",
        "LABELS_DIST",  # distance between corresponding labels in 2 images
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

#: :class:`Enum`: Data frame module tasks.
DFTasks = Enum(
    "DFTasks", [
        "MERGE_CSVS",  # merge rows
        "MERGE_CSVS_COLS",  # merge columns based on ID
        "APPEND_CSVS_COLS",  # concatenate chosen columns
        "EXPS_BY_REGION",  # convert volume stats to experiments by region
        "EXTRACT_FROM_CSV",  # extract rows based on matching rows in given col
        "ADD_CSV_COLS",  # add columns with values to CSV
        "NORMALIZE",  # normalize metrics to a base condition within a CSV
        "MERGE_EXCELS",  # merge Excel files into sheets of single Excel file
        "SUM_COLS",  # sum columns
        "SUBTRACT_COLS",  # subtract columns
        "MULTIPLY_COLS",  # multiply columns
        "DIVIDE_COLS",  # divide columns
        "REPLACE_VALS",  # replace values
    ]
)

df_task = None
seed = 0  # random number generator seed

#: float: measurement unit factor to convert to next larger prefix (eg um to mm)
unit_factor = 1000.0


class AtlasMetrics(Enum):
    """General atlas metric enumerations."""
    SAMPLE = "Sample"
    REGION = "Region"
    REGION_ABBR = "RegionAbbr"
    REGION_NAME = "RegionName"
    PARENT = "Parent"
    LEVEL = "Level"
    SIDE = "Side"
    CONDITION = "Condition"
    DSC_ATLAS_LABELS = "DSC_atlas_labels"
    DSC_ATLAS_LABELS_HEM = "DSC_atlas_labels_hemisphere"
    DSC_ATLAS_SAMPLE = "DSC_atlas_sample"
    DSC_ATLAS_SAMPLE_CUR = "DSC_atlas_sample_curated"
    DSC_SAMPLE_LABELS = "DSC_sample_labels"
    DSC_LABELS_ORIG_NEW_COMBINED = "DSC_labels_orig_new_combined"
    DSC_LABELS_ORIG_NEW_INDIV = "DSC_labels_orig_new_individual"
    SIMILARITY_METRIC = "Similarity_metric"
    LAT_UNLBL_VOL = "Lateral_unlabeled_volume"
    LAT_UNLBL_PLANES = "Lateral_unlabeled_planes"
    VOL_ATLAS = "Vol_atlas"
    VOL_LABELS = "Vol_labels"
    OFFSET = "Offset"
    SIZE = "Size"
    CHANNEL = "Channel"


class HemSides(Enum):
    """Hemisphere side enumerations."""
    RIGHT = "R"
    LEFT = "L"
    BOTH = "both"


#: :class:`Enum`: Label smoothing modes.
SmoothingModes = Enum(
    "SmoothingModes", [
        # morphological opening, which decreases in size for vols < 5000px
        # and switches to a closing filter if the label would be lost
        "opening",
        # morphological filters that adaptively decrease in size if
        # vol_new:vol_old is < a size ratio
        "adaptive_opening",  # opening filter
        "adaptive_closing",  # closing filter
        "adaptive_erosion",  # erosion filter
        "gaussian",  # gaussian blur
        "closing",  # closing morphological filter
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
roi_profile = None
roi_profiles = []
atlas_profile = None


def get_roi_profile(i):
    """Get the microscope profile for the given channel.
    
    Args:
        i: Index, typically a channel number.
    
    Returns:
        The profile settings for corresponding to the given channel number, 
        or the default profile if only one is available.
    """
    settings = roi_profile
    if len(roi_profiles) > i:
        settings = roi_profiles[i]
    return settings


#: :obj:`settings.grid_search_prof.GridSearchProfile`: Grid search profile.
grid_search_profile = None

# default colors using 7-color palette for color blindness
# (Wong, B. (2011) Nature Methods 8:441)
colors = np.array(
    [[213, 94, 0],  # vermilion
     [0, 114, 178],  # blue
     [204, 121, 167],  # reddish purple
     [230, 159, 0],  # orange
     [86, 180, 233],  # sky blue
     [0, 158, 115],  # blueish green
     [240, 228, 66],  # yellow
     [0, 0, 0]]  # black
)
