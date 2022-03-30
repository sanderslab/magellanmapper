# Cell detection methods
# Author: David Young, 2017, 2020
"""Detects features within a 3D image stack.

Prunes duplicates and verifies detections against truth sets.

Attributes:
    CONFIRMATION: Dictionary of blob confirmation flags.
    OVERLAP_FACTOR: Pixel number multiplier for overlaps between adjacent ROIs.

"""

from enum import Enum
import math
import pprint
from time import time
from typing import Dict, Optional, Sequence

import numpy as np
from skimage.feature import blob_log

from magmap.cv import colocalizer, cv_nd
from magmap.io import libmag, np_io
from magmap.plot import plot_3d
from magmap.settings import config

# blob confirmation flags
CONFIRMATION: Dict[int, str] = {
    -1: "unverified",
    0: "no",
    1: "yes",
    2: "maybe"
}

# pixel number multiplier by scaling for max overlapping pixels per ROI
OVERLAP_FACTOR: int = 5


class Blobs:
    """Blob storage class.
    
    Attributes:
        blobs: 2D Numpy array of blobs in the format
            ``[[z, y, x, radius, ...], ...]``; defaults to None.
        blob_matches: Sequence of blob matches; defaults to None.
        colocalizations: 2D Numpy array of same length
            as ``blobs`` with a column for each channel, where 0 = no
            signal and 1 = signal at the corresponding blob's location
            in the given and channel; defaults to None.
        path: Path from which blobs were loaded; defaults to None.
        ver: Version number; defaults to :const:`BLOBS_NP_VER`.
        roi_offset: Offset in ``z,y,x`` from ROI in which blobs were detected.
        roi_size: Size in ``z,y,x`` from ROI in which blobs were detected.
        resolutions: Physical unit resolution sizes in ``[[z,y,x], ...]``;
            defaults to None.
        basename: Archive name, typically the filename without extension
            of the image file in which blobs were detected; defaults to None.
        scaling: Scaling factor from the blobs' space to the main image's
            space, in ``z,y,x``; defaults to ``[1, 1, 1]``.
    
    """

    #: Current blobs Numpy archive version number.
    #: 0: initial version
    #: 1: added resolutions, basename, offset, roi_size fields
    #: 2: added archive version number
    #: 3: added colocs
    BLOBS_NP_VER: int = 3
    
    class Keys(Enum):
        """Numpy archive metadata keys as enumerations."""
        VER = "ver"
        BLOBS = "segments"
        COLOCS = "colocs"
        RESOLUTIONS = "resolutions"
        BASENAME = "basename"
        ROI_OFFSET = "offset"
        ROI_SIZE = "roi_size"

    def __init__(
            self,
            blobs: Optional[np.ndarray] = None,
            blob_matches: Optional["colocalizer.BlobMatch"] = None,
            colocalizations: Optional[np.ndarray] = None,
            path: str = None):
        """Initialize blobs storage object."""
        self.blobs = blobs
        self.blob_matches = blob_matches
        self.colocalizations = colocalizations
        self.path = path
        
        # additional attributes
        self.ver: int = self.BLOBS_NP_VER
        self.roi_offset: Optional[Sequence[int]] = None
        self.roi_size: Optional[Sequence[int]] = None
        self.resolutions: Optional[Sequence[float]] = None
        self.basename: Optional[str] = None
        self.scaling: np.ndarray = np.ones(3)

    def load_blobs(self, path=None):
        """Load blobs from an archive.

        Also loads associated metadata from the archive.

        Args:
            path (str): Path to set :attr:`path`; defaults to None to use
                the existing path.

        Returns:
            :class:`Blobs`: Blobs object.

        """
        # load blobs and display counts
        if path is not None:
            self.path = path
        print("Loading blobs from", self.path)
        with np.load(self.path) as archive:
            info = np_io.read_np_archive(archive)

            if self.Keys.VER.value in info:
                # load archive version number
                self.basename = info[self.Keys.VER.value]

            if self.Keys.BLOBS.value in info:
                # load blobs as a Numpy array
                self.blobs = info[self.Keys.BLOBS.value]
                print("Loaded {} blobs".format(len(self.blobs)))
                if config.verbose:
                    show_blobs_per_channel(self.blobs)
            
            if self.Keys.COLOCS.value in info:
                # load intensity-based colocalizations
                self.colocalizations = info[self.Keys.COLOCS.value]
                if self.colocalizations is not None:
                    print("Loaded blob co-localizations for {} channels"
                          .format(self.colocalizations.shape[1]))
            
            if self.Keys.RESOLUTIONS.value in info:
                # load resolutions of image from which blobs were detected
                self.resolutions = info[self.Keys.RESOLUTIONS.value]

            if self.Keys.BASENAME.value in info:
                # load basename of image file from which blobs were detected
                self.basename = info[self.Keys.BASENAME.value]

            if self.Keys.ROI_OFFSET.value in info:
                # load offset of ROI from which blobs were detected
                self.roi_offset = info[self.Keys.ROI_OFFSET.value]
            
            if self.Keys.ROI_SIZE.value in info:
                # load size of ROI from which blobs were detected
                self.roi_size = info[self.Keys.ROI_SIZE.value]
            
            if config.verbose:
                pprint.pprint(info)
        return self

    def save_archive(self, to_add=None, update=False):
        """Save the blobs Numpy archive file to :attr:`path`.
        
        Args:
            to_add (dict): Dictionary of items to add; defaults to None
                to use the current attributes.
            update (bool): True to load the Numpy archive at :attr:`path`
                and update it.

        Returns:
            dict: Dictionary saved to :attr:`path`.

        """
        if to_add is None:
            # save current attributes
            blobs_arc = {
                Blobs.Keys.VER.value: self.ver,
                Blobs.Keys.BLOBS.value: self.blobs,
                Blobs.Keys.RESOLUTIONS.value: self.resolutions,
                Blobs.Keys.BASENAME.value: self.basename,
                Blobs.Keys.ROI_OFFSET.value: self.roi_offset,
                Blobs.Keys.ROI_SIZE.value: self.roi_size,
                Blobs.Keys.COLOCS.value: self.colocalizations,
            }
        else:
            blobs_arc = to_add
        
        if update:
            with np.load(self.path) as archive:
                # load archive, convert to dict, and update dict
                blobs_arc = np_io.read_np_archive(archive)
                blobs_arc.update(to_add)
        
        with open(self.path, "wb") as archive:
            # save as uncompressed zip Numpy archive file
            np.savez(archive, **blobs_arc)
            print("Saved blobs archive to:", self.path)
        
        if config.verbose:
            pprint.pprint(blobs_arc)
        return blobs_arc


def calc_scaling_factor():
    """Calculates the tolerance based on the resolutions, using the 
    first resolution.
    
    Return:
        Array of tolerance values in same shape as the first resolution.
    
    Raises:
        ``AttributeError`` if the :attr:``config.resolutions`` is None or has 
        less than one element.
    """
    if config.resolutions is None or len(config.resolutions) < 1:
        raise AttributeError(
            "Must load resolutions from file or set a resolution")
    factor = np.divide(1.0, config.resolutions[0])
    return factor


def calc_overlap(factor: Optional[int] = None):
    """Calculate overlap based on scaling factor and a factor.
    
    Args:
        factor: Overlap factor; defaults to None to use :const:``OVERLAP_FACTOR``

    Returns:
        Overlap as an array in the same shape and order as in 
        :attr:``resolutions``.
    
    """
    if factor is None:
        factor = OVERLAP_FACTOR
    return np.ceil(np.multiply(calc_scaling_factor(), factor)).astype(int)


def _blob_surroundings(blob, roi, padding, plane=False):
    rad = blob[3]
    start = np.subtract(blob[0:3], rad + padding).astype(int)
    start[start < 0] = 0
    end = np.add(blob[0:3], rad + padding).astype(int)
    shape = roi.shape
    for i in range(3):
        if end[i] >= shape[i]:
            end[i] = shape[i] - 1
    if plane:
        z = blob[0]
        if z < 0:
            z = 0
        elif z >= shape[0]:
            z = end[0]
        return roi[z, start[1]:end[1], start[2]:end[2]]
    else:
        return roi[start[0]:end[0], start[1]:end[1], start[2]:end[2]]


def show_blob_surroundings(blobs, roi, padding=1):
    print("showing blob surroundings")
    np.set_printoptions(precision=2, linewidth=200)
    for blob in blobs:
        print("{} surroundings:".format(blob))
        surroundings = _blob_surroundings(blob, roi, padding, True)
        print("{}\n".format(surroundings))
    np.set_printoptions()


def detect_blobs(roi, channel, exclude_border=None):
    """Detects objects using 3D blob detection technique.
    
    Args:
        roi: Region of interest to segment.
        channel (Sequence[int]): Sequence of channels to select, which can
            be None to indicate all channels.
        exclude_border: Sequence of border pixels in x,y,z to exclude;
            defaults to None.
    
    Returns:
        Array of detected blobs, each given as 
            (z, row, column, radius, confirmation).
    """
    time_start = time()
    shape = roi.shape
    multichannel, channels = plot_3d.setup_channels(roi, channel, 3)
    isotropic = config.get_roi_profile(channels[0])["isotropic"]
    if isotropic is not None:
        # interpolate for (near) isotropy during detection, using only the 
        # first process settings since applies to entire ROI
        roi = cv_nd.make_isotropic(roi, isotropic)
    
    blobs_all = []
    for chl in channels:
        roi_detect = roi[..., chl] if multichannel else roi
        settings = config.get_roi_profile(chl)
        # scaling as a factor in pixel/um, where scaling of 1um/pixel  
        # corresponds to factor of 1, and 0.25um/pixel corresponds to
        # 1 / 0.25 = 4 pixels/um; currently simplified to be based on 
        # x scaling alone
        scale = calc_scaling_factor()
        scaling_factor = scale[2]
        
        # find blobs; sigma factors can be sequences by axes for anisotropic 
        # detection in skimage >= 0.15, or images can be interpolated to 
        # isotropy using the "isotropic" MagellanMapper setting
        min_sigma = settings["min_sigma_factor"] * scaling_factor
        max_sigma = settings["max_sigma_factor"] * scaling_factor
        num_sigma = settings["num_sigma"]
        threshold = settings["detection_threshold"]
        overlap = settings["overlap"]
        blobs_log = blob_log(
            roi_detect, min_sigma=min_sigma, max_sigma=max_sigma,
            num_sigma=num_sigma, threshold=threshold, overlap=overlap)
        if config.verbose:
            print("detecting blobs with min size {}, max {}, num std {}, "
                  "threshold {}, overlap {}"
                  .format(min_sigma, max_sigma, num_sigma, threshold, overlap))
            print("time for 3D blob detection: {}".format(time() - time_start))
        if blobs_log.size < 1:
            libmag.printv("no blobs detected")
            continue
        blobs_log[:, 3] = blobs_log[:, 3] * math.sqrt(3)
        blobs = format_blobs(blobs_log, chl)
        #print(blobs)
        blobs_all.append(blobs)
    if not blobs_all:
        return None
    blobs_all = np.vstack(blobs_all)
    if isotropic is not None:
        # if detected on isotropic ROI, need to reposition blob coordinates 
        # for original, non-isotropic ROI
        isotropic_factor = cv_nd.calc_isotropic_factor(isotropic)
        blobs_all = multiply_blob_rel_coords(blobs_all, 1 / isotropic_factor)
        blobs_all = multiply_blob_abs_coords(blobs_all, 1 / isotropic_factor)
    
    if exclude_border is not None:
        # exclude blobs from the border in x,y,z
        blobs_all = get_blobs_interior(blobs_all, shape, *exclude_border)
    
    return blobs_all


def format_blobs(blobs, channel=None):
    """Format blobs with additional fields for confirmation, truth, and 
    channel, abs z, abs y, abs x values.
    
    Blobs in MagellanMapper can be assumed to start with (z, y, x, radius) but should 
    use ``detector`` functions to manipulate other fields of blob arrays to 
    ensure that the correct columns are accessed.
    
    "Confirmed" is given as -1 = unconfirmed, 0 = incorrect, 1 = correct.
    
    "Truth" is given as -1 = not truth, 0 = not matched, 1 = matched, where 
    a "matched" truth blob is one that has a detected blob within a given 
    tolerance.
    
    Args:
        blobs: Numpy 2D array in [[z, y, x, radius, ...], ...] format.
        channel: Channel to set. Defaults to None, in which case the channel 
            will not be updated.
    
    Returns:
        Blobs array formatted as 
        [[z, y, x, radius, confirmation, truth, channel, 
          abs_z, abs_y, abs_x], ...].
    """
    # target num of cols minus current cols
    shape = blobs.shape
    extra_cols = 10 - shape[1]
    #print("extra_cols: {}".format(extra_cols))
    extras = np.ones((shape[0], extra_cols)) * -1
    blobs = np.concatenate((blobs, extras), axis=1)
    # copy relative coords to abs coords
    blobs[:, -3:] = blobs[:, :3]
    channel_dim = 6
    if channel is not None:
        # update channel if given
        blobs[:, channel_dim] = channel
    elif shape[1] <= channel_dim:
        # if original shape of each blob was 6 or less as was the case 
        # prior to v.0.6.0, need to update channel with default value
        blobs[:, channel_dim] = 0
    return blobs


def remove_abs_blob_coords(blobs):
    return blobs[:, :7]


def get_blob_abs_coords(blobs):
    return blobs[:, 7:10]


def set_blob_abs_coords(blobs, coords):
    blobs[:, 7:10] = coords
    return blobs


def shift_blob_rel_coords(blob, offset):
    """Shift blob relative coordinates by offset.
    
    Args:
        blob (List): Either a sequence starting with blob coordinates,
            typically in ``z, y, x, ...``, or a sequence of blobs.
        offset (List[int]): Sequence of coordinates by which to shift
            the corresponding elements from the start of ``blob``.

    Returns:
        List: The shifted blob or sequence of blobs.

    """
    if blob.ndim > 1:
        blob[..., :len(offset)] += offset
    else:
        blob[:len(offset)] += offset
    return blob


def shift_blob_abs_coords(blobs, offset):
    blobs[..., 7:7+len(offset)] += offset
    return blobs


def multiply_blob_rel_coords(blobs, factor):
    if blobs is not None:
        rel_coords = blobs[..., :3] * factor
        blobs[..., :3] = rel_coords.astype(np.int)
    return blobs


def multiply_blob_abs_coords(blobs, factor):
    if blobs is not None:
        abs_slice = slice(7, 7 + len(factor))
        abs_coords = blobs[..., abs_slice] * factor
        blobs[..., abs_slice] = abs_coords.astype(np.int)
    return blobs


def get_blob_confirmed(blob):
    if blob.ndim > 1:
        return blob[..., 4]
    return blob[4]


def set_blob_col(blob, col, val):
    """Set the value for the given column of a blob or blobs.

    Args:
        blob (:class:`numpy.ndarray`): 1D blob array or 2D array of blobs.
        col (int): Column index in ``blob``.
        val (int): Truth value.
    
    Returns:
        :class:`numpy.ndarray`: ``blob`` after modifications.

    """
    if blob.ndim > 1:
        blob[..., col] = val
    else:
        blob[col] = val
    return blob


def set_blob_confirmed(blob, val):
    """Set the confirmed flag of a blob or blobs.

    Args:
        blob (:class:`numpy.ndarray`): 1D blob array or 2D array of blobs.
        val (int): Confirmed flag.
    
    Returns:
        :class:`numpy.ndarray`: ``blob`` after modifications.

    """
    return set_blob_col(blob, 4, val)


def get_blob_truth(blob):
    """Get the truth flag of a blob or blobs.
    
    Args:
        blob (:obj:`np.ndarray`): 1D blob array or 2D array of blobs.

    Returns:
        int or :obj:`np.ndarray`: The truth flag of the blob as an int
        for a single blob or array of truth flags for an array of blobs. 

    """
    if blob.ndim > 1:
        return blob[..., 5]
    return blob[5]


def set_blob_truth(blob, val):
    """Set the truth flag of a blob or blobs.

    Args:
        blob (:class:`numpy.ndarray`): 1D blob array or 2D array of blobs.
        val (int): Truth value.
    
    Returns:
        :class:`numpy.ndarray`: ``blob`` after modifications.

    """
    return set_blob_col(blob, 5, val)


def get_blob_channel(blob):
    return blob[6]


def get_blobs_channel(blobs):
    return blobs[:, 6]


def set_blob_channel(blob, val):
    """Set the channel of a blob or blobs.

    Args:
        blob (:class:`numpy.ndarray`): 1D blob array or 2D array of blobs.
        val (int): Channel value.
    
    Returns:
        :class:`numpy.ndarray`: ``blob`` after modifications.

    """
    return set_blob_col(blob, 6, val)


def replace_rel_with_abs_blob_coords(blobs):
    blobs[:, :3] = blobs[:, 7:10]
    return blobs


def blobs_in_channel(blobs, channel, return_mask=False):
    """Get blobs in the given channels
    
    Args:
        blobs (:obj:`np.ndarray`): Blobs in the format,
            ``[[z, y, x, r, c, ...], ...]``.
        channel (List[int]): Sequence of channels to include.
        return_mask (bool): True to return the mask of blobs in ``channel``.

    Returns:
        :obj:`np.ndarray`: A view of the blobs in the channel, or all
        blobs if ``channel`` is None.

    """
    blobs_chl = blobs
    mask = None
    if channel is not None:
        mask = np.isin(get_blobs_channel(blobs), channel)
        blobs_chl = blobs[mask]
    if return_mask:
        return blobs_chl, mask
    return blobs_chl


def blob_for_db(blob):
    """Convert segment output from the format used within this module 
    to that used in :module:`sqlite`, where coordinates are absolute 
    rather than relative to the offset.
    
    Args:
        seg: Segment in 
            (z, y, x, rad, confirmed, truth, channel, abs_z, abs_y, abs_x) 
            format.
    
    Returns:
        Segment in (abs_z, abs_y, abs_x, rad, confirmed, truth, channel) format.
    """
    return np.array([*blob[-3:], *blob[3:7]])


def remove_duplicate_blobs(blobs, region):
    """Removes duplicate blobs.
    
    Args:
        blobs: The blobs, given as 2D array of [n, [z, row, column, radius]].
        region: Slice within each blob to check, such as slice(0, 2) to check
           for (z, row, column).
    
    Return:
        The blobs array with only unique elements.
    """
    # workaround while awaiting https://github.com/numpy/numpy/pull/7742
    # to become a reality, presumably in Numpy 1.13
    blobs_region = blobs[:, region]
    blobs_contig = np.ascontiguousarray(blobs_region)
    blobs_type = np.dtype(
        (np.void, blobs_region.dtype.itemsize * blobs_region.shape[1]))
    blobs_contig = blobs_contig.view(blobs_type)
    _, unique_indices = np.unique(blobs_contig, return_index=True)
    # TODO: should probably only look within blobs_region
    print("removed {} duplicate blobs"
          .format(blobs.shape[0] - unique_indices.size))
    return blobs[unique_indices]


def sort_blobs(blobs):
    """Sort blobs by their coordinates in priority of z,y,x.
    
    Args:
        blobs: Blobs as a 2D array of [n, [z, row, column, ...]].
    
    Returns:
        Tuple of the sorted blobs as a new array and an array of 
        sorting indices.
    """
    sort = np.lexsort(tuple(blobs[:, i] for i in range(2, -1, -1)))
    blobs = blobs[sort]
    return blobs, sort


def _find_close_blobs(blobs, blobs_master, tol):
    # creates a separate array for each blob in blobs_master to allow
    # comparison for each of its blobs with each blob to add
    blobs_diffs = np.abs(blobs_master[:, None, :3] - blobs[:, :3])
    close_master, close = np.nonzero((blobs_diffs <= tol).all(2))
    #print("close:\n{}\nclose_master:\n{}".format(close, close_master))
    return close_master, close


def remove_close_blobs(blobs, blobs_master, tol, chunk_size=1000):
    """Removes blobs that are close to one another.
    
    Args:
        blobs: The blobs to be checked for closeness and pruning, given as 2D 
            array of [n, [z, row, column, ...]].
        blobs_master: The list by which to check for close blobs, in the same
            format as blobs.
        tol: Tolerance to check for closeness, given in the same format
            as region. Blobs that are equal to or less than the the absolute
            difference for all corresponding parameters will be pruned in
            the returned array.
        chunk_size: Max size along first dimension for each blob array 
            to minimize memory consumption; defaults to 1000.
    
    Return:
        Tuple of the blobs array after pruning and ``blobs_master`` with 
        absolute coordinates updated with the average of any 
        corresponding duplicates.
    """
    num_blobs_check = len(blobs)
    num_blobs_master = len(blobs_master)
    if num_blobs_check < 1 or num_blobs_master < 1:
        # no blobs to remove if either array is empty
        return blobs, blobs_master
    
    # smallest type to hold blob coordinates, signed to use for diffs
    dtype = libmag.dtype_within_range(
        0, np.amax((np.amax(blobs[:, :3]), np.amax(blobs_master[:, :3]))), 
        True, True)
    match_check = None
    match_master = None
    
    # chunk both master and check array for consistent max array size; 
    # compare each master chunk to each check chunk and save matches 
    # to prune at end
    i = 0
    while i * chunk_size < num_blobs_master:
        start_master = i * chunk_size
        end_master = (i + 1) * chunk_size
        blobs_ref = blobs_master[start_master:end_master, :3].astype(dtype)
        j = 0
        while j * chunk_size < num_blobs_check:
            start_check = j * chunk_size
            end_check = (j + 1) * chunk_size
            blobs_check = blobs[start_check:end_check].astype(dtype)
            close_master, close = _find_close_blobs(blobs_check, blobs_ref, tol)
            # shift indices by offsets
            close += start_check
            close_master += start_master
            match_check = (close if match_check is None 
                           else np.concatenate((match_check, close)))
            match_master = (close_master if match_master is None 
                            else np.concatenate((match_master, close_master)))
            j += 1
        i += 1
    pruned = np.delete(blobs, match_check, axis=0)
    #if (len(close) > 0): print("{} removed".format(blobs[close][:, 0:4]))
    
    # shift close blobs to their mean values, storing values in the duplicated
    # coordinates and radius of the blob array after the confirmation value;
    # use the duplicated coordinates to work from any prior shifting; 
    # further duplicate testing will still be based on initial position to
    # allow detection of duplicates that occur in multiple ROI pairs
    abs_between = np.around(
        np.divide(
            np.add(get_blob_abs_coords(blobs_master[match_master]), 
                   get_blob_abs_coords(blobs[match_check])), 2))
    blobs_master[match_master] = set_blob_abs_coords(
        blobs_master[match_master], abs_between)
    #print("blobs_master after shifting:\n{}".format(blobs_master[:, 5:9]))
    return pruned, blobs_master


def remove_close_blobs_within_array(blobs, region, tol):
    """Removes close blobs within a given array.
    
    Uses remove_close_blobs() to detect blobs close to one another inside
    the master array.
    
    Args:
        blobs: The blobs to add, given as 2D array of [n, [z, row, column, 
            radius]].
        region: Slice within each blob to check, such as slice(0, 2) to check
            for (z, row, column).
        tol: Tolerance to check for closeness, given in the same format
            as region. Blobs that are equal to or less than the the absolute
            difference for all corresponding parameters will be pruned in
            the returned array.
    
    Return:
        The blobs array without blobs falling inside the tolerance range.
    """
    if blobs is None:
        return None
    print("checking blobs for close duplicates:\n{}".format(blobs))
    blobs_all = None
    for blob in blobs:
        # check each blob against all blobs accepted thus far to ensure that
        # no blob is close to another blob
        #print("blob: {}".format(blob))
        if blobs_all is None:
            blobs_all = np.array([blob])
        else:
            # check an array of a single blob to add
            blobs_to_add, blobs_all = remove_close_blobs(
                np.array([blob]), blobs_all, region, tol)
            if blobs_to_add is not None:
                blobs_all = np.concatenate((blobs_all, blobs_to_add))
    return blobs_all


def meas_pruning_ratio(num_blobs_orig, num_blobs_after_pruning, num_blobs_next):
    """Measure blob pruning ratio.
    
    Args:
        num_blobs_orig: Number of original blobs, before pruning.
        num_blobs_after_pruning: Number of blobs after pruning.
        num_blobs_next: Number of a blobs in an adjacent segment, presumably 
            of similar size as that of the original blobs.
    
    Returns:
        Pruning ratios as a tuple of the original number of blobs, 
        blobs after pruning to original, and blobs after pruning to 
        the next region.
    """
    ratios = None
    if num_blobs_next > 0 and num_blobs_orig > 0:
        # calculate pruned:original and pruned:adjacent blob ratios
        print("num_blobs_orig: {}, blobs after pruning: {}, num_blobs_next: {}"
              .format(num_blobs_orig, num_blobs_after_pruning, num_blobs_next))
        ratios = (num_blobs_orig, num_blobs_after_pruning / num_blobs_orig, 
                  num_blobs_after_pruning / num_blobs_next)
    return ratios


def remove_close_blobs_within_sorted_array(blobs, tol):
    """Removes close blobs within a given array, first sorting the array by
    z, y, x.
    
    Args:
        blobs: The blobs to add, given as 2D array of [n, [z, row, column, 
            radius]].
        tol: Tolerance to check for closeness, given in the same format
            as region. Blobs that are equal to or less than the the absolute
            difference for all corresponding parameters will be pruned in
            the returned array.
    
    Return:
        Tuple of all blobs, a blobs array without blobs falling inside 
        the tolerance range.
    """
    if blobs is None:
        return None
    sort = np.lexsort((blobs[:, 2], blobs[:, 1], blobs[:, 0]))
    blobs = blobs[sort]
    #print("checking sorted blobs for close duplicates:\n{}".format(blobs))
    blobs_all = None
    for blob in blobs:
        # check each blob against all blobs accepted thus far to ensure that
        # no blob is close to another blob
        #print("blob: {}".format(blob))
        if blobs_all is None:
            # initialize array with first blob
            blobs_all = np.array([blob])
        else:
            # check each blob to add against blobs approved thus far
            i = len(blobs_all) - 1
            while i >= 0:
                blobs_diff = np.abs(np.subtract(
                    blob[:3], blobs_all[i, :3]))
                #print(blobs_diff)
                if (blobs_diff <= tol).all():
                    # remove duplicate blob and shift to mean of coords, 
                    # storing values in saved blob's abs coords; note that 
                    # this shift means that re-pruning the same region 
                    # may lead to further pruning
                    abs_between = np.around(
                        np.divide(
                            np.add(get_blob_abs_coords(blobs_all[i, None]), 
                                   get_blob_abs_coords(blob[None])), 2))
                    set_blob_abs_coords(blobs_all[i, None], abs_between)
                    #print("updated blob:", blobs_all[i])
                    #print("removed blob:", blob)
                    break
                elif i == 0 or not (blobs_diff <= tol).any():
                    # add blob since at start of non-duplicate blobs list
                    # or no further chance for match within sorted list
                    blobs_all = np.concatenate((blobs_all, [blob]))
                    break
                i -= 1
    #print("blobs without close duplicates:\n{}".format(blobs_all))
    return blobs_all


def get_blobs_in_roi(blobs, offset, size, margin=(0, 0, 0), reverse=True):
    """Get blobs within an ROI based on offset and size.
    
    Note that dimensions are in x,y,z for natural ordering but may 
    change for consistency with z,y,x ordering used throughout MagellanMapper.
    
    Args:
        blobs (:obj:`np.ndarray`): The blobs to retrieve, given as 2D array of
            ``[n, [z, row, column, radius, ...]]``.
        offset (List[int]): Offset coordinates in .
        size (List[int]): Size of ROI in x,y,z.
        margin (List[int]): Additional space outside the ROI to include
            in x,y,z.
        reverse (bool): True to reverse the order of ``offset`` and ``size``,
            assuming that they are in x,y,z rather than z,y,x order.
            Defaults to True for backward compatibility with the ROI
            convention used here.
    
    Returns:
        :obj:`np.ndarray`, :obj:`np.ndarray`: Blobs within the ROI and the
        mask used to retrieve these blobs.
    """
    if reverse:
        offset = offset[::-1]
        size = size[::-1]
    mask = np.all([
        blobs[:, 0] >= offset[0] - margin[0],
        blobs[:, 0] < offset[0] + size[0] + margin[0],
        blobs[:, 1] >= offset[1] - margin[1],
        blobs[:, 1] < offset[1] + size[1] + margin[1],
        blobs[:, 2] >= offset[2] - margin[2],
        blobs[:, 2] < offset[2] + size[2] + margin[2]], axis=0)
    segs_all = blobs[mask]
    return segs_all, mask


def get_blobs_interior(blobs, shape, pad_start, pad_end):
    """Get blobs within the interior of a region based on padding.
    
    Args:
        blobs: The blobs to retrieve, given as 2D array of 
            ``[n, [z, row, column, radius, ...]]``.
        shape: Shape of the region in z,y,x.
        pad_start: Offset of interior region in z,y,x to include blobs.
        pad_end: End offset in z,y,x.
    
    Returns:
        Blobs within the given interior.
    """
    return blobs[
        np.all([
            blobs[:, 0] >= pad_start[0], 
            blobs[:, 0] < shape[0] - pad_end[0],
            blobs[:, 1] >= pad_start[1], 
            blobs[:, 1] < shape[1] - pad_end[1],
            blobs[:, 2] >= pad_start[2], 
            blobs[:, 2] < shape[2] - pad_end[2]], axis=0)]


def show_blobs_per_channel(blobs):
    """Show the number of blobs in each channel.
    
    Args:
        blobs: Blobs as 2D array of [n, [z, row, column, radius, ...]].
    """
    channels = np.unique(get_blobs_channel(blobs))
    for channel in channels:
        num_blobs = len(blobs_in_channel(blobs, channel))
        print("- blobs in channel {}: {}".format(int(channel), num_blobs))


def _test_blob_duplicates():
    # tests blob duplication removal
    blobs = np.array([[1, 3, 4, 2.2342], [1, 8, 5, 3.13452], [1, 3, 4, 5.1234],
                      [1, 3, 5, 2.2342], [3, 8, 5, 3.13452]])
    print("sample blobs:\n{}".format(blobs))
    end = 3
    blobs_unique = remove_duplicate_blobs(blobs, slice(0, end))
    print("blobs_unique through first {} elements:\n{}"
          .format(end, blobs_unique))
    
    # tests removal of blobs within a given tolerance level
    tol = (1, 2, 2)
    blobs = remove_close_blobs_within_array(blobs, slice(0, end), tol)
    print("pruned sample blobs within tolerance {}:\n{}".format(tol, blobs))
    blobs_to_add = np.array([[1, 3, 5, 2.2342], [2, 10, 5, 3.13452], 
                             [2, 2, 4, 5.1234], [3, 3, 5, 2.2342]])
    print("blobs to add:\n{}".format(blobs_to_add))
    blobs_to_add = remove_close_blobs(blobs_to_add, blobs, slice(0, end), tol)
    print("pruned blobs to add:\n{}".format(blobs_to_add))


if __name__ == "__main__":
    print("Detector tests...")
