# Cell detection methods
# Author: David Young, 2017, 2020
"""Detects features within a 3D image stack.

Prunes duplicates and verifies detections against truth sets.
"""

from enum import Enum
import math
import pprint
from time import time

import numpy as np
from scipy import optimize
from scipy.spatial import distance
from skimage.feature import blob_log

from magmap.cv import colocalizer, cv_nd
from magmap.io import df_io, libmag, np_io, sqlite
from magmap.plot import plot_3d
from magmap.settings import config
from magmap.stats import atlas_stats, mlearn

# blob confirmation flags
CONFIRMATION = {
    -1: "unverified",
    0: "no",
    1: "yes",
    2: "maybe"
}


class Blobs:
    """Blob storage class.
    
    Attributes:
        BLOBS_NP_VER (int): Current blobs Numpy archive version number.
        blobs (:obj:`np.ndarray`): 2D Numpy array of blobs in the format
            ``[[z, y, x, radius, ...], ...]``; defaults to None.
        blob_matches (:obj:`magmap.io.sqlite.BlobMatch`): Sequence of blob
            matches; defaults to None.
        colocalizations (:obj:`np.ndarray`): 2D Numpy array of same length
            as ``blobs`` with a column for each channel, where 0 = no
            signal and 1 = signal at the corresponding blob's location
            in the given and channel; defaults to None.
        path (str): Path from which blobs were loaded; defaults to None.
        ver (int): Version number; defaults to :const:`BLOBS_NP_VER`.
        roi_offset (Sequence[int]): Offset in ``z,y,x`` from ROI in which
            blobs were detected.
        roi_size (Sequence[int]): Size in ``z,y,x`` from ROI in which
            blobs were detected.
        resolutions (Sequence[int]): Physical unit resolution sizes in
            ``[[z,y,x], ...]``; defaults to None.
        basename (str): Archive name, typically the filename without extension
            of the image file in which blobs were detected; defaults to None.
    
    """

    # blobs Numpy archive versions:
    # 0: initial version
    # 1: added resolutions, basename, offset, roi_size fields
    # 2: added archive version number
    # 3: added colocs
    BLOBS_NP_VER = 3
    
    class Keys(Enum):
        """Numpy archive metadata keys as enumerations."""
        VER = "ver"
        BLOBS = "segments"
        COLOCS = "colocs"
        RESOLUTIONS = "resolutions"
        BASENAME = "basename"
        ROI_OFFSET = "offset"
        ROI_SIZE = "roi_size"

    def __init__(self, blobs=None, blob_matches=None, colocalizations=None,
                 path=None):
        """Initialize blobs storage object."""
        self.blobs = blobs
        self.blob_matches = blob_matches
        self.colocalizations = colocalizations
        self.path = path
        
        self.ver = self.BLOBS_NP_VER
        self.roi_offset = None
        self.roi_size = None
        self.resolutions = None
        self.basename = None

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


def _find_closest_blobs(blobs, blobs_master, tol):
    """Finds the closest matching blobs between two arrays. Each entry will 
    have no more than one match, and the total number of matches will be 
    the size of the shortest list.
    
    Args:
        blobs: The blobs to be checked for closeness, given as 2D 
            array of at least [n, [z, row, column, ...]].
        blobs_master: The list by which to check for close blobs, in the same
            format as blobs.
        tol: Tolerance to check for closeness, given in the same format
            as region. Blobs that are equal to or less than the the absolute
            difference for all corresponding parameters will be considered
            a potential match.
    
    Returns:
        close_master: Indices of blobs_master that are the closest match to
            the corresponding blobs in "blobs", in order of closest to farthest
            match.
        close: Indices of the corresponding blobs in the "blobs" array.
    """
    # NOTE: does not scale well with large numbers of blobs, presumably 
    # because of the need to check blobs individually with a continually 
    # increasing number of accepted blobs
    
    # TESTING: sorting changes blob matches; use find_closest_blobs_cdist 
    # to avoid this issue
    #blobs, sort = sort_blobs(blobs)
    
    close_master = []
    close = []
    # compare each element for differences, weighting based on tolerance; 
    # TODO: incorporate radius
    blobs_diffs_init = np.abs(
        blobs_master[:, :3][:, None] - blobs[:, :3])
    normalize_factor = np.divide(np.max(tol), tol)
    tol = np.multiply(tol, normalize_factor)
    far = np.max(tol) + 1
    #print("weighted tol: {}".format(tol))
    
    # matches limited by length of smallest list
    num_matches = min(len(blobs_master), len(blobs))
    for i in range(num_matches):
        # normalize the diffs
        blobs_diffs = np.multiply(blobs_diffs_init, normalize_factor)
        # sum to find smallest diff
        diffs_sums = np.sum(blobs_diffs, blobs_diffs.ndim - 1)
        #print("blobs_diffs:\n{}".format(blobs_diffs))
        #print("diffs_sums:\n{}".format(diffs_sums))
        for j in range(len(diffs_sums)):
            # iterate up to the number of master blobs
            
            # get indices of minimum differences
            min_master, min_blob = np.where(diffs_sums == diffs_sums.min())
            #print("diffs_sums: {}, min: {}".format(diffs_sums, diffs_sums.min()))
            found = False
            for k in range(len(min_master)):
                blob_master_closest = min_master[k]
                blob_closest = min_blob[k]
                diff = blobs_diffs[blob_master_closest, blob_closest]
                #print("min_master: {}, min_blob: {}".format(min_master, min_blob))
                #print("compare {} to {}".format(diff, tol))
                if (diff <= tol).all():
                    # only keep the match if within tolerance
                    close_master.append(blob_master_closest)
                    close.append(blob_closest)
                    # replace row/column corresponding to each picked blob with  
                    # distant values to ensure beyond tol
                    blobs_diffs_init[blob_master_closest, :, :] = far
                    blobs_diffs_init[:, blob_closest, :] = far
                    found = True
                    break
                elif (diff <= tol).any():
                    # closest value still to far, so set to distant 
                    # value to can check next min diff; 
                    # TODO: add far rather than simply assigning to it?
                    diffs_sums[blob_master_closest] = far
                else:
                    # no match if none of array dims within tolerance of min diff
                    break
            if found:
                break
    if config.verbose:
        # show sorted list of matches to compare between runs
        _match_blobs(
            blobs, blobs_master, close, close_master, np.zeros(len(blobs)))
    return np.array(close_master, dtype=int), np.array(close, dtype=int)


def _match_blobs(blobs, blobs_master, close, close_master, dists):
    """Group matches between blobs.
    
    Args:
        blobs (:obj:`np.ndarray`): Blobs as a 2D array of
            ``[n, [z, row, column, ...]]``.
        blobs_master (:obj:`np.ndarray`): Array in same format as ``blobs``
            from a master list.
        close (List[int]): Sequence of indices of ``blobs``.
        close_master (List[int]): Sequence of indices of ``blobs_master``
            matching the corresponding values in ``close``.
        dists (List[float]): Sequence of distances corresponding to the
            matches in ``close`` and ``close_master``.

    Returns:
        List[List]]: Sequence of matches, which each consist of
        ``blob_master, blob, distance``.

    """
    # show sorted list of matches between blobs and master blobs
    found_master = blobs_master[close_master]
    found_master, sort = sort_blobs(found_master)
    found = blobs[close][sort]
    matches = []
    for f, fm, d in zip(found, found_master, dists[sort]):
        match = (fm, f, d)
        matches.append(match)
    return matches


def find_closest_blobs_cdist(blobs, blobs_master, thresh=None, scaling=None):
    """Find the closest blobs within a given tolerance using the 
    Hungarian algorithm to find blob matches.
    
    Args:
        blobs: Blobs as a 2D array of [n, [z, row, column, ...]].
        blobs_master: Array in same format as ``blobs``.
        thresh: Threshold distance beyond which blob pairings are excluded; 
            defaults to None to include all matches.
        scaling: Sequence of scaling factors by which to multiply the 
            blob coordinates before computing distances, used to 
            scale coordinates from an anisotropic to isotropic 
            ROI before computing distances, which assumes isotropy. 
            Defaults to None.
    
    Returns:
        Tuple of ``rowis`` and ``colis``, arrays of row and corresponding 
        column indices of the closest matches; and ``dists_closest``, an 
        array of corresponding distances for these matches. Only matches 
        within the given tolerance will be included.
    """
    blobs_scaled = blobs
    blobs_master_scaled = blobs_master
    if scaling is not None:
        # scale blobs and tolerance by given factor, eg for isotropy
        len_scaling = len(scaling)
        blobs_scaled = np.multiply(blobs[:, :len_scaling], scaling)
        blobs_master_scaled = np.multiply(blobs_master[:, :len_scaling], scaling)
    
    # find Euclidean distances between each pair of points and determine 
    # the optimal assignments using the Hungarian algorithm
    dists = distance.cdist(blobs_scaled, blobs_master_scaled)
    rowis, colis = optimize.linear_sum_assignment(dists)
    
    dists_closest = dists[rowis, colis]
    if thresh is not None:
        # filter out matches beyond the given threshold distance
        dists_in = dists_closest < thresh
        if config.verbose:
            print("only keeping blob matches within threshold distance of",
                  thresh)
            for blob, blob_sc, blob_base, blob_base_sc, dist, dist_in in zip(
                    blobs[rowis], blobs_scaled[rowis], blobs_master[colis],
                    blobs_master_scaled[colis], dists_closest,
                    dists_in):
                print("blob: {} (scaled {}), base: {} ({}), dist: {}, in? {}"
                      .format(blob[:3], blob_sc[:3], blob_base[:3],
                              blob_base_sc[:3], dist, dist_in))
        rowis = rowis[dists_in]
        colis = colis[dists_in]
        dists_closest = dists_closest[dists_in]
    
    return rowis, colis, dists_closest


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


def setup_match_blobs_roi(blobs, tol):
    """Set up tolerances for matching blobs in an ROI.
    
    Args:
        blobs (:obj:`np.ndarray`): Sequence of blobs to resize if the
            first ROI profile (:attr:`config.roi_profiles`) ``resize_blobs``
            value is given.
        tol (List[int, float]): Sequence of tolerances.

    Returns:
        float, List[float], List[float], List[float], :obj:`np.ndarray`:
        Distance map threshold, scaling normalized by ``tol``, ROI padding
        shape, resize sequence retrieved from ROI profile, and ``blobs``
        after any resizing.

    """
    # convert tolerance seq to scaling and single number distance
    # threshold for point distance map, which assumes isotropy; use
    # custom tol rather than calculating isotropy since may need to give
    # greater latitude along a given axis, such as poorer res in z
    thresh = np.amax(tol)  # similar to longest radius from the tol bounding box
    scaling = thresh / tol
    # casting to int causes improper offset import into db
    inner_padding = np.floor(tol[::-1])
    libmag.printv(
        "verifying blobs with tol {} leading to thresh {}, scaling {}, "
        "inner_padding {}".format(tol, thresh, scaling, inner_padding))
    
    # resize blobs based only on first profile
    resize = config.get_roi_profile(0)["resize_blobs"]
    if resize:
        blobs = multiply_blob_rel_coords(blobs, resize)
        libmag.printv("resized blobs by {}:\n{}".format(resize, blobs))
    
    return thresh, scaling, inner_padding, resize, blobs


def match_blobs_roi(blobs, blobs_base, offset, size, thresh, scaling,
                    inner_padding, resize=None):
    """Match blobs from two sets of blobs in an ROI, prioritizing the inner
    portion of ROIs to avoid missing detections because of edge effects
    while also adding matches between a blob in the inner ROI and another
    blob in the remaining portion of the ROI.
    
    Args:
        blobs (:obj:`np.ndarray`): The blobs to be matched against
            ``blobs_base``, given as 2D array of
            ``[[z, row, column, radius, ...], ...]``.
        blobs_base (:obj:`np.ndarray`): The blobs to which ``blobs`` will
            be matched, in the same format as ``blobs``.
        offset (List[int]): ROI offset from which to select blobs in x,y,z.
        size (List[int]): ROI size in x,y,z.
        thresh (float): Distance map threshold
        scaling (List[float]): Scaling normalized by ``tol``.
        inner_padding (List[float]): ROI padding shape.
        resize (List[float]): Resize sequence retrieved from ROI profile;
            defaults to None.
    
    Returns:
        :class:`numpy.ndarray`, :class:`numpy.ndarray`, list[int], list[int],
        list[list[:class:`numpy.ndarray`, :class:`numpy.ndarray`, float]]:
        Array of blobs from ``blobs``; corresponding array from ``blobs_base``
        matching blobs in ``blobs``; offset of the inner portion of the ROI
        in absolute coordinates of x,y,z; shape of this inner portion of the
        ROI; and list of blob matches, each given as a list of
        ``blob_master, blob, distance``.
    
    """
    # get all blobs in inner and total ROI
    offset_inner = np.add(offset, inner_padding)
    size_inner = np.subtract(size, inner_padding * 2)
    libmag.printv(
        "offset: {}, offset_inner: {}, size: {}, size_inner: {}"
        .format(offset, offset_inner, size, size_inner))
    blobs_roi, _ = get_blobs_in_roi(blobs, offset, size)
    if resize is not None:
        # TODO: doesn't align with exported ROIs
        padding = config.plot_labels[config.PlotLabels.PADDING]
        libmag.printv("shifting blobs in ROI by offset {}, border {}"
                      .format(offset, padding))
        blobs_roi = shift_blob_rel_coords(blobs_roi, offset)
        if padding:
            blobs_roi = shift_blob_rel_coords(blobs_roi, padding)
    blobs_inner, blobs_inner_mask = get_blobs_in_roi(
        blobs_roi, offset_inner, size_inner)
    blobs_base_roi, _ = get_blobs_in_roi(blobs_base, offset, size)
    blobs_base_inner, blobs_base_inner_mask = get_blobs_in_roi(
        blobs_base_roi, offset_inner, size_inner)
    
    # compare blobs from inner region of ROI with all base blobs,
    # prioritizing the closest matches
    found, found_base, dists = find_closest_blobs_cdist(
        blobs_inner, blobs_base_roi, thresh, scaling)
    blobs_inner[:, 4] = 0
    blobs_inner[found, 4] = 1
    blobs_base_roi[blobs_base_inner_mask, 5] = 0
    blobs_base_roi[found_base, 5] = 1
    
    # add any base blobs missed in the inner ROI by comparing with
    # test blobs from outer ROI
    blobs_base_inner_missed = blobs_base_roi[blobs_base_roi[:, 5] == 0]
    blobs_outer = blobs_roi[np.invert(blobs_inner_mask)]
    found_out, found_base_out, dists_out = find_closest_blobs_cdist(
        blobs_outer, blobs_base_inner_missed, thresh, scaling)
    blobs_base_inner_missed[found_base_out, 5] = 1
    
    # combine inner and outer groups
    blobs_truth_inner_plus = np.concatenate(
        (blobs_base_roi[blobs_base_roi[:, 5] == 1],
         blobs_base_inner_missed))
    blobs_outer[found_out, 4] = 1
    blobs_inner_plus = np.concatenate((blobs_inner, blobs_outer[found_out]))

    matches_inner = _match_blobs(
        blobs_inner, blobs_base_roi, found, found_base, dists)
    matches_outer = _match_blobs(
        blobs_outer, blobs_base_inner_missed, found_out,
        found_base_out, dists_out)
    matches = colocalizer.BlobMatch([*matches_inner, *matches_outer])
    if config.verbose:
        '''
        print("blobs_roi:\n{}".format(blobs_roi))
        print("blobs_inner:\n{}".format(blobs_inner))
        print("blobs_base_inner:\n{}".format(blobs_base_inner))
        print("blobs_base_roi:\n{}".format(blobs_base_roi))
        print("found inner:\n{}"
              .format(blobs_inner[found]))
        print("truth found:\n{}"
              .format(blobs_base_roi[found_base]))
        print("blobs_outer:\n{}".format(blobs_outer))
        print("blobs_base_inner_missed:\n{}"
              .format(blobs_base_inner_missed))
        print("truth blobs detected by an outside blob:\n{}"
              .format(blobs_base_inner_missed[found_base_out]))
        print("all those outside detection blobs:\n{}"
              .format(blobs_roi_extra))
        print("blobs_inner_plus:\n{}".format(blobs_inner_plus))
        print("blobs_truth_inner_plus:\n{}".format(blobs_truth_inner_plus))
        '''

        print("Closest matches found (truth, detected, distance):")
        msgs = ("\n- Inner ROI:", "\n- Outer ROI:")
        for msg, matches_sub in zip(msgs, (matches_inner, matches_outer)):
            print(msg)
            for match in matches_sub:
                print("Blob1:", match[0][:3], "chl", get_blob_channel(match[0]),
                      "Blob2:", match[1][:3], "chl", get_blob_channel(match[1]),
                      "dist:", match[2])
        print()
    
    return blobs_inner_plus, blobs_truth_inner_plus, offset_inner, size_inner, \
        matches


def verify_rois(rois, blobs, blobs_truth, tol, output_db, exp_id, exp_name,
                channel):
    """Verify blobs in ROIs by comparing detected blobs with truth sets
    of blobs stored in a database.
    
    Save the verifications to a separate database with a name in the same
    format as saved processed files but with "_verified.db" at the end.
    Prints basic statistics on the verification.
    
    Note that blobs are found from ROI parameters rather than loading from 
    database, so blobs recorded within these ROI bounds but from different 
    ROIs will be included in the verification.
    
    Args:
        rois: Rows of ROIs from sqlite database.
        blobs (:obj:`np.ndarray`): The blobs to be checked for accuracy,
            given as 2D array of ``[[z, row, column, radius, ...], ...]``.
        blobs_truth (:obj:`np.ndarray`): The list by which to check for
            accuracy, in the same format as blobs.
        tol: Tolerance as z,y,x of floats specifying padding for the inner
            ROI and used to generate a single tolerance distance within
            which a detected and ground truth blob will be considered
            potential matches.
        output_db: Database in which to save the verification flags, typical
            the database in :attr:``config.verified_db``.
        exp_id: Experiment ID in ``output_db``.
        exp_name (str): Name of experiment to store as the sample name for
            each row in the output data frame.
        channel (List[int]): Filter ``blobs_truth`` by this channel.
    
    Returns:
        tuple[int, int, int], str, :class:`pandas.DataFrame`: Tuple of
        ``pos, true_pos, false_pos`` stats, feedback message, and accuracy
        metrics in a data frame.
    
    """
    blobs_truth = blobs_in_channel(blobs_truth, channel)
    blobs_truth_rois = None
    blobs_rois = None
    rois_falsehood = []
    thresh, scaling, inner_padding, resize, blobs = setup_match_blobs_roi(
        blobs, tol)
    
    # set up metrics dict for accuracy metrics of each ROI
    metrics = {}
    cols = (
        config.AtlasMetrics.SAMPLE,
        config.AtlasMetrics.CHANNEL,
        config.AtlasMetrics.OFFSET,
        config.AtlasMetrics.SIZE,
        mlearn.GridSearchStats.POS,
        mlearn.GridSearchStats.TP,
        mlearn.GridSearchStats.FP,
        mlearn.GridSearchStats.FN,
    )
    
    for roi in rois:
        # get ROI from database for ground truth blobs
        offset = (roi["offset_x"], roi["offset_y"], roi["offset_z"])
        size = (roi["size_x"], roi["size_y"], roi["size_z"])
        series = roi["series"]
        
        # find matches between truth and detected blobs
        blobs_inner_plus, blobs_truth_inner_plus, offset_inner, size_inner, \
            matches = match_blobs_roi(
                blobs, blobs_truth, offset, size, thresh, scaling,
                inner_padding, resize)
        
        # store blobs in separate verified DB
        roi_id, _ = sqlite.insert_roi(output_db.conn, output_db.cur, exp_id,
                                      series, offset_inner, size_inner)
        sqlite.insert_blobs(output_db.conn, output_db.cur, roi_id,
                            blobs_inner_plus)
        sqlite.insert_blobs(output_db.conn, output_db.cur, roi_id,
                            blobs_truth_inner_plus)
        output_db.insert_blob_matches(roi_id, matches)
        
        # compute accuracy metrics for the ROI
        pos = len(blobs_truth_inner_plus)  # condition pos
        true_pos = np.sum(blobs_inner_plus[:, 4] == 1)
        false_pos = np.sum(blobs_inner_plus[:, 4] == 0)
        false_neg = len(blobs_truth_inner_plus) - true_pos
        if false_neg > 0 or false_pos > 0:
            rois_falsehood.append((offset_inner, false_pos, false_neg))
        vals = (exp_name, channel[0] if channel else 0,
                tuple(offset_inner.astype(int)), tuple(size_inner.astype(int)),
                pos, true_pos, false_pos, pos - true_pos)
        for key, val in zip(cols, vals):
            metrics.setdefault(key, []).append(val)
        
        # combine blobs into total lists for stats across ROIs
        if blobs_truth_rois is None:
            blobs_truth_rois = blobs_truth_inner_plus
        else:
            blobs_truth_rois = np.concatenate(
                (blobs_truth_inner_plus, blobs_truth_rois))
        if blobs_rois is None:
            blobs_rois = blobs_inner_plus
        else:
            blobs_rois = np.concatenate((blobs_inner_plus, blobs_rois))
    
    # generate and show data frame of accuracy metrics for each ROI
    df = df_io.dict_to_data_frame(metrics, show=" ")
    
    # show accuracy metrics of blobs combined across ROIs
    true_pos = df[mlearn.GridSearchStats.TP.value].sum()
    false_pos = df[mlearn.GridSearchStats.FP.value].sum()
    pos = df[mlearn.GridSearchStats.POS.value].sum()
    false_neg = pos - true_pos
    print("Automated verification using tol {}:\n".format(tol))
    fdbk = "Accuracy metrics for channel {}:\n{}".format(
        channel, atlas_stats.calc_sens_ppv(
            pos, true_pos, false_pos, false_neg)[2])
    print(fdbk)
    print("ROIs with falsehood:\n{}".format(rois_falsehood))
    return (pos, true_pos, false_pos), fdbk, df


def meas_detection_accuracy(blobs, verified=False, treat_maybes=0):
    """Measure detection accuracy based on blob confirmation statuses.

    Args:
        blobs (:obj:`np.ndarray`): 2D array of blobs in the format,
            ``[[z, row, column, radius, confirmation, truth, ...], ...]``.
        verified (bool): True to assume that blobs have undergone verification.
        treat_maybes (int): 0 to ignore maybes; 1 to treat maybes as correct,
            and 1 to treat maybes as incorrect.

    Returns:
        float, float, str: Sensivity, positive predictive value (PPV), and
        summary of stats as a string. If ``blobs`` is None or empty, returns
        None for each of these values.

    """
    if blobs is None or len(blobs) < 1:
        return None, None, None
    
    # basic stats based on confirmation status, ignoring maybes; "pos"
    # here means actual positives, whereas "true pos" means correct
    # detection, where radius <= 0 indicates that the blob was manually
    # added rather than detected; "false pos" are incorrect detections
    if verified:
        # basic stats based on confirmation status, ignoring maybes
        blobs_pos = blobs[blobs[:, 5] >= 0]  # all truth blobs
        blobs_detected = blobs[blobs[:, 5] == -1]  # all non-truth blobs
        blobs_true_detected = blobs_detected[blobs_detected[:, 4] == 1]
        blobs_false = blobs[blobs[:, 4] == 0]
    else:
        blobs_pos = blobs[blobs[:, 4] == 1]
        # TODO: consider just checking > 0
        blobs_true_detected = blobs_pos[blobs_pos[:, 3] >= config.POS_THRESH]
        blobs_false = blobs[blobs[:, 4] == 0]

    # calculate sensitivity and PPV; no "true neg" detection so no
    # specificity measurement
    all_pos = blobs_pos.shape[0]
    true_pos = blobs_true_detected.shape[0]
    false_pos = blobs_false.shape[0]
    if verified or treat_maybes == 0:
        # ignore maybes
        maybe_msg = "(ignoring maybes)"
    else:
        blobs_maybe = blobs[blobs[:, 4] == 2]
        blobs_maybe_detected = blobs_maybe[
            blobs_maybe[:, 3] >= config.POS_THRESH]
        num_maybe_detected = len(blobs_maybe_detected)
        if treat_maybes == 1:
            # most generous, where detections that are maybes are treated as
            # true pos, and missed blobs that are maybes are treated as ignored
            all_pos += num_maybe_detected
            true_pos += num_maybe_detected
            maybe_msg = "(treating maybes as correct)"
        else:
            # most conservative, where detections that are maybes are treated
            # as false pos, and missed blobs that are maybes are treated as pos
            all_pos += len(blobs_maybe) - num_maybe_detected
            false_pos += num_maybe_detected
            maybe_msg = "(treating maybes as incorrect)"

    # measure stats
    false_neg = all_pos - true_pos  # not detected but should have been
    sens, ppv, msg = atlas_stats.calc_sens_ppv(
        all_pos, true_pos, false_pos, false_neg)
    msg = "Detection stats {}:\n{}".format(maybe_msg, msg)
    return sens, ppv, msg


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


def _test_blob_verification(a, b, tol):
    # test verifying blobs by checking for closest matches within a tolerance
    print("test (b):\n{}".format(b))
    print("master (a):\n{}".format(a))
    #found_truth, detected = _find_closest_blobs(b, a, tol)
    #dists = np.zeros(len(blobs)
    detected, found_truth, dists = find_closest_blobs_cdist(b, a, tol)
    df_io.dict_to_data_frame(
        {"Testi": detected, "Masteri": found_truth, "Dist": dists}, show=True)


def _test_blob_close_sorted():
    a = np.ones((3, 3))
    a[:, 0] = [0, 1, 2]
    b = np.copy(a)
    b[:, 0] += 1
    print("test (b):\n{}".format(b))
    print("master (a):\n{}".format(a))
    blobs = np.concatenate((a, b))
    sort = np.lexsort((blobs[:, 2], blobs[:, 1], blobs[:, 0]))
    blobs = blobs[sort]
    blobs = remove_close_blobs_within_sorted_array(blobs, (1, 2, 2))
    print("pruned:\n{}".format(blobs))


def _test_blob_autoverifier():
    _test_blob_close_sorted()
    a = np.array([[0, 1, 1], [1, 1, 1], [2, 1, 1]])
    b = np.array([[1, 1, 1], [2, 1, 1], [3, 1, 1], [4, 2, 0]])
    _test_blob_verification(a, b, 1)
    print()
    a = np.array([[24, 52, 346], [20, 55, 252]])
    b = np.array([[24, 54, 351]])
    _test_blob_verification(a, b, 6)


if __name__ == "__main__":
    print("Detector tests...")
