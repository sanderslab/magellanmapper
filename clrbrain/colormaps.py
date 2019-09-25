#!/bin/bash
# Colormaps for Clrbrain
# Author: David Young, 2018, 2019
"""Custom colormaps for Clrbrain.
"""

import numpy as np
from matplotlib import cm
from matplotlib import colors

from clrbrain import config

# default colormaps, with keys backed by config.Cmaps enums
CMAPS = {}


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
    """
    def __init__(self, labels=None, seed=None, alpha=150, index_direct=True, 
                 multiplier=255, offset=0, background=None, dup_for_neg=False):
        """Generate discrete colormap for labels using 
        :func:``discrete_colormap``.
        
        Args:
            labels: Labels of integers for which a distinct color should be 
                mapped to each unique label. Deafults to None, in which case 
                no colormap will be generated.
            seed: Seed for randomizer to allow consistent colormap between 
                runs; defaults to None.
            alpha: Transparency leve; defaults to 150 for semi-transparent.
            index_direct: True if the colormap will be indexed directly, which 
                assumes that the labels will serve as indexes to the colormap 
                and should span sequentially from 0, 1, 2, ...; defaults to 
                True. If False, a colormap will be generated for the full 
                range of integers between the lowest and highest label values, 
                inclusive.
            multiplier: Multiplier for random values generated for RGB values; 
                defaults to 255.
            offset: Offset to generated random numbers; defaults to 0.
            background: Tuple of (backround_label, (R, G, B, A)), where 
                background_label is the label value specifying the background, 
                and RGBA value will replace the color corresponding to that 
                label. Defaults to None.
            dup_for_neg: True to duplicate positive labels as negative 
                labels to recreate the same set of labels as for a 
                mirrored labels map. Defaults to False.
        """
        if labels is None: return
        self.norm = None
        labels_unique = np.unique(labels)
        if dup_for_neg and len(labels_unique[labels_unique < 0]) == 0:
            # for labels that are only pos or 0, duplicate the pos portion 
            # and make neg so that images with or without negs use the 
            # same colors
            labels_unique = np.append(
                -1 * labels_unique[labels_unique > 0][::-1], labels_unique)
        num_colors = len(labels_unique)
        # make first boundary slightly below first label to encompass it 
        # to avoid off-by-one errors that appear to occur when viewing an 
        # image with an additional extreme label
        labels_offset = 0.5
        labels_unique = labels_unique.astype(np.float32)
        labels_unique -= labels_offset
        # number of boundaries should be one more than number of labels to 
        # avoid need for interpolation of boundary bin numbers and 
        # potential merging of 2 extreme labels
        labels_unique = np.append(labels_unique, [labels_unique[-1] + 1])
        if index_direct:
            # assume label vals increase by 1 from 0 until num_colors
            self.norm = colors.NoNorm()
        else:
            # labels themselves serve as bounds, allowing for large gaps 
            # between labels while assigning each label to a unique color
            self.norm = colors.BoundaryNorm(labels_unique, num_colors)
        self.cmap_labels = discrete_colormap(
            num_colors, alpha=alpha, prioritize_default=False, seed=seed, 
            multiplier=multiplier, offset=offset)
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
    
    def modified_cmap(self, adjust):
        """Make a modified discrete colormap from itself.
        
        Args:
            adjust: Value by which to adjust RGB (not A) values.
        
        Returns:
            New ``DiscreteColormap`` instance with ``norm`` pointing to first 
            instance and ``cmap_labels`` incremented by the given value.
        """
        cmap = DiscreteColormap()
        # TODO: consider whether to copy instead
        cmap.norm = self.norm
        cmap.cmap_labels = np.copy(self.cmap_labels)
        # labels are uint8 so should already fit within RGB bounds; colors 
        # that exceed these bounds will likely have slightly different tones 
        # since RGB vals will not change uniformly
        cmap.cmap_labels[:, :3] += adjust
        cmap.make_cmap()
        return cmap


def discrete_colormap(num_colors, alpha=255, prioritize_default=True, 
                      seed=None, multiplier=255, offset=0):
    """Make a discrete colormap using :attr:``config.colors`` as the 
    starting colors and filling in the rest with randomly generated RGB values.
    
    Args:
        num_colors (int): Number of discrete colors to generate.
        alpha (int): Transparency level, from 0-255; defaults to 255.
        prioritize_default (bool): If True, the default colors from 
            :attr:``config.colors`` will replace the initial colormap elements; 
            defaults to True. Alternatively, `cn` can be given to use 
            the "CN" color spec instead.
        seed (int): Random number seed; defaults to None, in which case no seed 
            will be set.
        multiplier (int): Multiplier for random values generated for RGB values;
            defaults to 255. Generally range from 1-255, with lower 
            numbers skewing toward darker colors.
        offset (int): Offset to generated random numbers; defaults to 0. The 
            final numbers will be in the range from offset to offset+multiplier.
    
    Returns:
        2D Numpy array in the format [[R, G, B, alpha], ...] on a 
        scale of 0-255. This colormap will need to be converted into a 
        Matplotlib colormap using ``LinearSegmentedColormap.from_list`` 
        to generate a map that can be used directly in functions such 
        as ``imshow``.
    """
    # generate random combination of RGB values for each number of colors, 
    # skewed by offset and limited by multiplier; 
    # TODO: consider giving option to change to 0-1 scale
    if seed is not None:
        np.random.seed(seed)
    cmap = (np.random.random((num_colors, 4)) 
            * multiplier + offset).astype(np.uint8)
    cmap[:, -1] = alpha # make slightly transparent
    if prioritize_default is not False:
        # prioritize default colors by replacing first colors with default ones
        colors_default = config.colors
        if prioritize_default == "cn":
            # "CN" color spec
            colors_default = np.multiply(
                [colors.to_rgb("C{}".format(i)) for i in range(10)], 255)
        end = min((num_colors, len(colors_default)))
        cmap[:end, :3] = colors_default[:end]
    return cmap


def get_labels_discrete_colormap(labels_img, alpha_bkgd=255, dup_for_neg=False, 
                                 use_orig_labels=False):
    """Get a default discrete colormap for a labels image, assuming that 
    background is 0, and the seed is determined by :attr:``config.seed``.
    
    Args:
        labels_img: Labels image as a Numpy array.
        alpha_bkgd: Background alpha level from 0 to 255; defaults to 255 
            to turn on background fully.
        dup_for_neg: True to duplicate positive labels as negative 
            labels; defaults to False.
        use_orig_labels (bool): True to use original labels from 
            :attr:`config.labels_img_orig` if available, falling back to 
            ``labels_img``. Defaults to False.
    
    Returns:
        :class:``DiscreteColormap`` object with a separate color for 
        each unique value in ``labels_img``.
    """
    lbls = labels_img
    if use_orig_labels and config.labels_img_orig is not None:
        # use original labels if available for mapping consistency
        lbls = config.labels_img_orig
    return DiscreteColormap(
        lbls, config.seed, 255, False, 150, 50, 
        (0, (0, 0, 0, alpha_bkgd)), dup_for_neg)


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
    elif cmap in config.Cmaps:
        # assume default colormaps have been initialized
        cmap = CMAPS[cmap]
    return cmap
