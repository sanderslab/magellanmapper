# Colormaps for MagellanMapper
# Author: David Young, 2018, 2020
"""Custom colormaps for MagellanMapper.
"""

from enum import Enum, auto
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from matplotlib import cm
from matplotlib import colors

from magmap.settings import config
from magmap.io import libmag

#: Dict[:class:`config.Cmaps`, :obj:`colors.LinearSegmentedColormap`]:
# Default colormaps.
CMAPS = {}

_logger = config.logger.getChild(__name__)


class DiscreteModes(Enum):
    """Discrete colormap generation modes."""
    RANDOMN = auto()
    GRID = auto()


def make_dark_linear_cmap(name, color):
    """Make a linear colormap starting with black and ranging to 
    ``color``.
    
    Args:
        name: Name to give to colormap.
        color: Colors will range from black to this color.
    
    Returns:
        A `LinearSegmentedColormap` object.
    """
    return colors.LinearSegmentedColormap.from_list(name, ("black", color))


def setup_cmaps():
    """Setup default colormaps, storing them in :const:``CMAPS``."""
    CMAPS[config.Cmaps.CMAP_GRBK_NAME] = make_dark_linear_cmap(
        config.Cmaps.CMAP_GRBK_NAME.value, "green")
    CMAPS[config.Cmaps.CMAP_RDBK_NAME] = make_dark_linear_cmap(
        config.Cmaps.CMAP_RDBK_NAME.value, "red")
    CMAPS[config.Cmaps.CMAP_BUBK_NAME] = make_dark_linear_cmap(
        config.Cmaps.CMAP_BUBK_NAME.value, "blue")
    CMAPS[config.Cmaps.CMAP_CYBK_NAME] = make_dark_linear_cmap(
        config.Cmaps.CMAP_CYBK_NAME.value, "cyan")
    CMAPS[config.Cmaps.CMAP_MGBK_NAME] = make_dark_linear_cmap(
        config.Cmaps.CMAP_MGBK_NAME.value, "magenta")
    CMAPS[config.Cmaps.CMAP_YLBK_NAME] = make_dark_linear_cmap(
        config.Cmaps.CMAP_YLBK_NAME.value, "yellow")


class DiscreteColormap(colors.ListedColormap):
    """Extends :class:``matplotlib.colors.ListedColormap`` to generate a 
    discrete colormap and associated normalization object.
    
    Extend ``ListedColormap`` rather than linear colormap since the 
    number of colors should equal the number of possible vals, without 
    requiring interpolation.
    
    Attributes:
        cmap_labels: Tuple of N lists of RGBA values, where N is equal 
            to the number of colors, with a discrete color for each 
            unique value in ``labels``.
        norm: Normalization object, which is of type 
            :class:``matplotlib.colors.NoNorm`` if indexing directly or 
            :class:``matplotlib.colors.BoundaryNorm`` if otherwise.
        img_labels (List[int]): Sorted sequence of unique labels. May have
            more values than in ``labels`` such as mirrored negative values.
            None if ``index_direct`` is False.
    """
    def __init__(
            self, labels: Optional[Sequence[int]] = None,
            alpha: int = 150, index_direct: bool = True, max_val: int = 255,
            background: Optional[Tuple[int, Tuple[int, int, int, int]]] = None,
            dup_for_neg: bool = False, symmetric_colors: bool = False,
            cmap_labels: Optional[Sequence[str]] = None, **kwargs):
        """Generate discrete colormap for labels.
        
        Args:
            labels: Labels of integers for which a distinct color should be 
                mapped to each unique label. Defults to None, in which case 
                no colormap will be generated.
            alpha: Transparency leve; defaults to 150 for semi-transparent.
            index_direct: True if the colormap will be indexed directly, which
                assumes that the labels will serve as indexes to the colormap
                and should span sequentially from 0, 1, 2, ...; defaults to
                True. If False, a colormap will be generated for the full
                range of integers between the lowest and highest label values,
                inclusive, with a :obj:`colors.BoundaryNorm`, which may
                incur performance cost.
            max_val: Maximum value for random numbers; defaults to 255.
            background: Tuple of ``(background_label, (R, G, B, A))``, where
                ``background_label`` is the label value specifying the
                background, and ``RGBA`` will replace the color for that
                label. Defaults to None.
            dup_for_neg: True to duplicate positive labels as negative
                labels to recreate the same set of labels as for a
                mirrored labels map. Defaults to False.
            symmetric_colors: True to make symmetric colors, assuming
                symmetric labels centered on 0; defaults to False.
            cmap_labels: Sequence of colors as Matplotlib color
                strings or RGB(A) hex (eg "#0fab24ff") strings.
        
        """
        self.norm = None
        self.cmap_labels = None
        self.img_labels = None
        self.symmetric_colors = symmetric_colors

        if labels is None: return
        labels_unique = np.unique(labels)
        if dup_for_neg and np.sum(labels_unique < 0) == 0:
            # for labels that are only >= 0, duplicate the pos portion
            # as neg so that images with or without negs use the same colors
            labels_unique = np.append(
                -1 * labels_unique[labels_unique > 0][::-1], labels_unique)
        num_colors = len(labels_unique)

        labels_offset = 0
        if index_direct:
            # assume label vals increase by 1 from 0 until num_colors; store
            # sorted labels sequence to translate labels based on index
            self.norm = colors.NoNorm()
            self.img_labels = labels_unique
        else:
            # use labels as bounds for each color, including wide bounds
            # for large gaps between successive labels; offset bounds to
            # encompass each label and avoid off-by-one errors that appear
            # when viewing images with additional extreme labels; float32
            # gives unsymmetric colors for large values in mirrored atlases
            # despite remaining within range for unclear reasons, fixed by
            # using float64 instead
            labels_offset = 0.5
            bounds = labels_unique.astype(np.float64)
            bounds -= labels_offset
            # number of boundaries should be one more than number of labels to
            # avoid need for interpolation of boundary bin numbers and
            # potential merging of 2 extreme labels
            bounds = np.append(bounds, [bounds[-1] + 1])
            # TODO: may have occasional colormap inaccuracies from this bug:
            # https://github.com/matplotlib/matplotlib/issues/9937;
            self.norm = colors.BoundaryNorm(bounds, num_colors)
        
        if cmap_labels is None:
            # auto-generate colors for the number of labels
            args = dict(
                alpha=alpha, prioritize_default=False, max_val=max_val,
                symmetric_colors=symmetric_colors)
            args.update(kwargs)
            self.cmap_labels = discrete_colormap(num_colors, **args)
        else:
            # generate RGBA colors from supplied color strings
            self.cmap_labels = colors.to_rgba_array(cmap_labels) * max_val
        
        if background is not None:
            # replace background label color with given color
            bkgdi = np.where(labels_unique == background[0] - labels_offset)
            if len(bkgdi) > 0 and bkgdi[0].size > 0:
                self.cmap_labels[bkgdi[0][0]] = background[1]
        #print(self.cmap_labels)
        self.make_cmap()
    
    def make_cmap(self):
        """Initialize ``ListedColormap`` with stored labels rescaled to 0-1."""
        super(DiscreteColormap, self).__init__(
            self.cmap_labels / 255.0, "discrete_cmap")
    
    def modified_cmap(self, adjust: int) -> "DiscreteColormap":
        """Make a modified discrete colormap from itself.
        
        The resulting colormap is assumed to map to the same range of label
        image values, using the same :attr:`norm` and :attr:`img_labels`.
        
        Args:
            adjust: Value by which to adjust RGB (not A) values.
        
        Returns:
            New ``DiscreteColormap`` instance with :attr:`norm` pointing to
            first instance, :attr:`img_labels`, and :attr:`cmap_labels`
            incremented by ``adjust``.
        
        """
        cmap = DiscreteColormap()
        # TODO: consider whether to copy instead
        cmap.norm = self.norm
        cmap.img_labels = self.img_labels
        
        # cast labels from uint8 (RBG) to int16 to accommodate adjustments
        # outside of 0-255 range but clip back to this range
        cmap.cmap_labels = np.copy(self.cmap_labels).astype(np.int16)
        cmap.cmap_labels[:, :3] += adjust
        cmap.cmap_labels = cmap.cmap_labels.clip(0, 255).astype(np.uint8)
        cmap.make_cmap()
        return cmap

    def convert_img_labels(self, img):
        """Convert an image to the indices in :attr:`img_labels` to give
        a linearly scaled image.

        This image can be displayed using a colormap with
        :class:`matplotlib.colors.NoNorm` to index directly into the colormap.

        Args:
            img (:obj:`np.ndarray`): Image to convert. If
                :attr:`symmetric_colors` is True, the absolute value will
                be taken as a workaround for likely image display resampling
                errors.

        Returns:
            :class:`numpy.ndarray`: Array of same shape as ``img`` with values
            translated to their corresponding indices within :attr:`img_labels`,
            or ``img`` unchanged if :attr:`img_labels` is None.

        """
        conv = img
        if self.img_labels is not None:
            if self.symmetric_colors:
                # WORKAROUND: corresponding pos/neg label vals may display
                # different colors despite mapping to the same colormap value,
                # perhaps because rounding or resampling issues related to:
                # https://github.com/matplotlib/matplotlib/issues/12071
                # https://github.com/matplotlib/matplotlib/issues/16910
                img = np.abs(img)
            conv = np.searchsorted(self.img_labels, img)
            # TESTING: show colormap correspondences with label IDs
            # img_un = np.unique(img)
            # img_conv = np.searchsorted(self.img_labels, img_un)
            # for im, cv in zip(img_un, img_conv):
            #     print(im, cv, self(cv))
        return conv


def discrete_colormap(
        num_colors: int, alpha: int = 255,
        prioritize_default: Union[bool, str, Sequence[Sequence[int]]] = True,
        seed: Optional[int] = None, min_val: Union[int, float] = 0,
        max_val: Union[int, float] = 255, min_any: Union[int, float] = 0,
        symmetric_colors: bool = False, dup_offset: int = 0, jitter: int = 0,
        mode: "DiscreteModes" = DiscreteModes.RANDOMN) -> np.ndarray:
    """Make a discrete colormap using randomly generated RGB values.
    
    Args:
        num_colors: Number of discrete colors to generate.
        alpha: Transparency level, from 0-255; defaults to 255.
        prioritize_default: If True (default), the default colors from
            :attr:``config.colors`` will replace the initial colormap elements.
            Can alternatively be a string for Matplotlib color specs, where
            `cn` = "CN" color spec, `css4` = CSS4/X11 spec, and "tableau" =
            Tableau spec. Can also be a sequence of colors, where each
            colors is in ``r, g, b, a``, ranging from 0-255.
        seed: Random number seed; defaults to None, in which case no seed
            will be set.
        min_val: Minimum value for random numbers; defaults to 0.
        max_val: Maximum value for random numbers; defaults to 255.
            For floating point ranges such as 0.0-1.0, set as a float.
        min_any: Minimum value above which at least one value
            must be in each set of RGB values; defaults to 0. If all
            values in an RGB set are below this value, the lowest
            RGB value will be scaled up by the ratio ``max_val:min_any``.
            Assumes a range of ``min_val < min_any < max_val``; defaults to
            0 to ignore.
        symmetric_colors: True to create a symmetric set of colors,
            assuming the first half of ``num_colors`` mirror those of
            the second half; defaults to False.
        dup_offset: Amount by which to offset duplicate color values
            if ``dup_for_neg`` is enabled; defaults to 0.
        jitter: In :obj:`DiscreteModes.GRID` mode, coordinates are
            randomly shifted by half this value above or below their original
            value; defaults to 0.
        mode: Mode given as an enumeration; defaults to
            :obj:`DiscreteModes.RANDOMN` mode.
    
    Returns:
        2D Numpy array in the format ``[[R, G, B, alpha], ...]`` on a
        scale of 0-255. This colormap will need to be converted into a
        Matplotlib colormap using ``LinearSegmentedColormap.from_list``
        to generate a map that can be used directly in functions such
        as ``imshow``.
    
    """
    if symmetric_colors:
        # make room for offset when duplicating colors
        max_val -= dup_offset

    # generate random combination of RGB values for each number of colors,
    # where each value ranges from min-max
    if mode is DiscreteModes.GRID:
        # discrete colors taken from an evenly spaced grid for min separation
        # between color values
        jitters = None
        if jitter > 0:
            if seed is not None: np.random.seed(seed)
            jitters = np.multiply(
                np.random.random((num_colors, 3)),
                jitter - jitter / 2).astype(int)
            max_val -= np.amax(jitters)
            min_val -= np.amin(jitters)
        # TODO: weight chls or scale non-linearly for better visual distinction
        space = (max_val - min_val) // np.cbrt(num_colors)
        sl = slice(min_val, max_val, space)
        grid = np.mgrid[sl, sl, sl]
        coords = np.c_[grid[0].ravel(), grid[1].ravel(), grid[2].ravel()]
        if min_any > 0:
            # remove all coords where all vals are below threshold
            # TODO: account for lost coords in initial space size determination
            coords = coords[~np.all(np.less(coords, min_any), axis=1)]
        if seed is not None: np.random.seed(seed)
        rand = np.random.choice(len(coords), num_colors, replace=False)
        rand_coords = coords[rand]
        if jitters is not None:
            rand_coords = np.add(rand_coords, jitters)
        rand_coords_shape = list(rand_coords.shape)
        rand_coords_shape[-1] += 1
        cmap = np.zeros(
            rand_coords_shape,
            dtype=libmag.dtype_within_range(min_val, max_val))
        cmap[:, :-1] = rand_coords
    else:
        # randomly generate each color value; 4th values only for simplicity
        # in generating array with shape for alpha channel
        if seed is not None: np.random.seed(seed)
        cmap = (np.random.random((num_colors, 4)) 
                * (max_val - min_val) + min_val).astype(
            libmag.dtype_within_range(min_val, max_val))
        if min_any > 0:
            # if all vals below threshold, scale up lowest value
            below_offset = np.all(np.less(cmap[:, :3], min_any), axis=1)
            axes = np.argmin(cmap[below_offset, :3], axis=1)
            cmap[below_offset, axes] = np.multiply(
                cmap[below_offset, axes], max_val / min_any)
    
    if prioritize_default is not False:
        # prioritize default colors by replacing first colors with default ones
        colors_default = config.colors
        colors_mplot = dict(
            base=colors.BASE_COLORS,
            xkcd=colors.XKCD_COLORS,
            css4=colors.CSS4_COLORS,
            tableau=colors.TABLEAU_COLORS,
        )
        
        if prioritize_default == "cn":
            # "CN" color spec
            colors_default = np.multiply(
                [colors.to_rgb("C{}".format(i)) for i in range(10)], 255)
        
        elif prioritize_default in colors_mplot.keys():
            # Matplotlib colors group
            colors_default = colors_mplot[prioritize_default].values()
            colors_default = np.multiply(
                colors.to_rgba_array(colors_default)[:, :3], 255)
            
            ncolors_default = len(colors_default)
            if num_colors < ncolors_default:
                # increase distinctiveness by picking evenly spaced colors,
                # assuming color list is sorted by similarity
                colors_default = colors_default[np.linspace(
                    0, ncolors_default - 1, num_colors).astype(int)]
        
        elif libmag.is_seq(prioritize_default):
            # use given color dictionary directly
            colors_default = prioritize_default
        
        # keep only required or available number of colors
        end = min((num_colors, len(colors_default)))
        cmap[:end, :3] = colors_default[:end]
    
    if symmetric_colors:
        # invert latter half onto former half, assuming that corresponding
        # labels are mirrored (eg -5, 3, 0, 3, 5), with background centered as 0
        cmap_len = len(cmap)
        mid = cmap_len // 2
        cmap[:mid] = cmap[:cmap_len-mid-1:-1] + dup_offset
    
    cmap[:, -1] = alpha  # set transparency
    
    return cmap


def get_labels_discrete_colormap(
        labels_img: Optional[np.ndarray], alpha_bkgd: int = 255,
        use_orig_labels: bool = False, **kwargs) -> "DiscreteColormap":
    """Get a default discrete colormap for a labels image.
     
    Assumes that background is 0, and the seed is determined by
    :attr:``config.seed``.
    
    Args:
        labels_img: Labels image as a Numpy array. Can be None, in which
            case ``use_orig_labels`` should be True.
        alpha_bkgd: Background alpha level from 0 to 255; defaults to 255
            to turn on background fully.
        use_orig_labels: True to use original labels from
            :attr:`config.labels_img_orig` if available, falling back to
            ``labels_img``. Defaults to False.
    
    Returns:
        A colormap with separate color for each unique value in ``labels_img``.
    
    """
    lbls = labels_img
    if use_orig_labels:
        # use original labels image IDs if available for mapping consistency
        if (config.labels_metadata and
                config.labels_metadata.region_ids_orig is not None):
            # use saved label IDs
            lbls = config.labels_metadata.region_ids_orig
        elif config.labels_img_orig is not None:
            # fallback to use labels from original image if available
            lbls = config.labels_img_orig
    args = dict(
        seed=config.seed, alpha=255, min_any=160, min_val=10,
        background=(0, (0, 0, 0, alpha_bkgd)))
    args.update(kwargs)
    return DiscreteColormap(lbls, **args)


def get_borders_colormap(borders_img, labels_img, cmap_labels):
    """Get a colormap for borders, using corresponding labels with 
    intensity change to distinguish the borders.
    
    If the number of labels differs from that of the original colormap, 
    a new colormap will be generated instead.
    
    Args:
        borders_img: Borders image as a Numpy array, used to determine 
            the number of labels required. If this image has multiple 
            channels, a similar colormap with distinct intensity will 
            be made for each channel.
        labels_img: Labels image as a Numpy array, used to compare 
            the number of labels for each channel in ``borders_img``.
        cmap_labels: The original colormap on which the new colormaps 
            will be based.
    
    Returns:
        List of borders colormaps corresponding to the number of channels, 
        or None if ``borders_img`` is None
    """
    cmap_borders = None
    if borders_img is not None:
        if np.unique(labels_img).size == np.unique(borders_img).size:
            # get matching colors by using labels colormap as template, 
            # with brightest colormap for original (channel 0) borders
            channels = 1
            if borders_img.ndim >= 4:
                channels = borders_img.shape[-1]
            cmap_borders = [
                cmap_labels.modified_cmap(int(40 / (channel + 1)))
                for channel in range(channels)]
        else:
            # get a new colormap if borders image has different number 
            # of labels while still ensuring a transparent background
            cmap_borders = [get_labels_discrete_colormap(borders_img, 0)]
    return cmap_borders


def make_binary_cmap(binary_colors):
    """Make a binary discrete colormap.
    
    Args:
        binary_colors (List[str]): Sequence of colors as
            ``[background, foreground]``.

    Returns:
        :obj:`DiscreteColormap`: Discrete colormap with labels of ``[0, 1]``
        mapped to ``binary_colors``.

    """
    return DiscreteColormap([0, 1], cmap_labels=binary_colors)


def setup_labels_cmap(
        labels_img: np.ndarray, binary_colors: Optional[Sequence[str]] = None,
        **kwargs) -> "DiscreteColormap":
    """Wrapper to set up a colormap for a labels image.
    
    Args:
        labels_img: Labels image.
        binary_colors: Sequence of colors as strings; defaults to None, in
            which case :attr:`config.atlas_labels[config.AtlasLabels.BINARY]`
            is used. 0 is assumed to be background, and 1 is foreground.

    Returns:
        Discrete colormap for the given labels.

    """
    # TODO: consider merging with get_labels_discrete_colormap
    if binary_colors is None:
        binary_colors = config.atlas_labels[config.AtlasLabels.BINARY]
    
    if binary_colors:
        cmap_labels = make_binary_cmap(binary_colors)
    else:
        args = dict(
            dup_for_neg=True, use_orig_labels=True,
            symmetric_colors=config.atlas_labels[
                config.AtlasLabels.SYMMETRIC_COLORS])
        args.update(kwargs)
        cmap_labels = get_labels_discrete_colormap(labels_img, 0, **args)
    return cmap_labels


def get_cmap(cmap, n=None):
    """Get colormap from a list of colormaps, string, or enum.
    
    If ``n`` is given, ``cmap`` is assumed to be a list from which a colormap 
    will be retrieved. Colormaps that are strings will be converted to 
    the associated standard `Colormap` object, while enums in 
    :class:``config.Cmaps`` will be used to retrieve a `Colormap` object 
    from :const:``CMAPS``, which is assumed to have been initialized.
    
    Args:
        cmap: Colormap given as a string of Enum or list of colormaps.
        n: Index of `cmap` to retrieve a colormap, assuming that `cmap` 
            is a sequence. Defaults to None to use `cmap` directly.
    
    Returns:
        The ``Colormap`` object, or None if no corresponding colormap 
        is found.
    """
    if n is not None:
        # assume that cmap is a list
        cmap = config.cmaps[n] if n < len(cmap) else None
    if isinstance(cmap, str):
        # cmap given as a standard Matplotlib colormap name
        cmap = cm.get_cmap(cmap)
    elif isinstance(cmap, config.Cmaps):
        # assume default colormaps have been initialized
        cmap = CMAPS[cmap]
    return cmap


def setup_colormaps(num_channels):
    """Set up colormaps based on the currently loaded main ROI profile.

    Args:
        num_channels (int): Number of channels in the main image; if the
            main ROI profile does not define this many colormaps, new
            colormaps will be randomly generated.

    """
    config.cmaps = list(config.roi_profile["channel_colors"])
    num_cmaps = len(config.cmaps)
    if num_cmaps < num_channels:
        # add colormap for each remaining channel, purposely inducing
        # int wraparound for greater color contrast
        chls_diff = num_channels - num_cmaps
        cmaps = discrete_colormap(
            chls_diff, alpha=255, prioritize_default=False, seed=config.seed,
            min_val=150) / 255.0
        _logger.debug("Generating colormaps from RGBA colors:\n%s", cmaps)
        for cmap in cmaps:
            config.cmaps.append(make_dark_linear_cmap("", cmap))
