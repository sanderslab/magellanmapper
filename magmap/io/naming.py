# File naming conventions for MagellanMapper

import os

from magmap.io import libmag


def make_subimage_name(base, offset, shape, suffix=None):
    """Make name of subimage for a given offset and shape.

    The order of ``offset`` and ``shape`` are assumed to be in z,y,x but
    will be reversed for the output name since the user-oriented ordering
    is x,y,z.
    
    Args:
        base (str): Start of name, which can include full parent path.
        offset (Tuple[int]): Offset, generally given as a tuple.
        shape (Tuple[int]): Shape, generally given as a tuple.
        suffix (str): Suffix to append, replacing any existing extension
            in ``base``; defaults to None.
    
    Returns:
        str: Name (or path) to subimage.
    """
    # sub-image offset/shape stored as z,y,x, but file named as x,y,z
    roi_site = "{}x{}".format(offset[::-1], shape[::-1]).replace(" ", "")
    name = libmag.insert_before_ext(base, roi_site, "_")
    if suffix:
        name = libmag.combine_paths(name, suffix)
    print("subimage name: {}".format(name))
    return name


def get_roi_path(path, offset, roi_size=None):
    """Get a string describing an ROI for an image at a given path.

    Args:
        path (str): Path to include in string, without extension.
        offset (List[int]): Offset of ROI.
        roi_size (List[int]): Shape of ROI; defaults to None to ignore.

    Returns:
        str: String with ``path`` without extension followed immediately by
        ``offset`` and ``roi_size`` as tuples, with all spaces removed.
    """
    size = ""
    if roi_size is not None:
        size = "x{}".format(tuple(roi_size))
    return "{}_offset{}{}".format(
        os.path.splitext(path)[0], tuple(offset), size).replace(" ", "")