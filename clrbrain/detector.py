# Cell detection methods
# Author: David Young, 2017
"""Detects features within a 3D image stack.

Prunes duplicates and verifies detections against truth sets.

Attributes:
    resolutions: The image resolutions as an array of dimensions (n, r),
        where each resolution r is a tuple in (z, y, x) order.
"""

from time import time
import math
import numpy as np
from scipy import ndimage
from skimage import filters
from skimage import segmentation
from skimage import measure
from skimage import morphology
from skimage.feature import blob_log

from clrbrain import config
from clrbrain import lib_clrbrain
from clrbrain import plot_3d
from clrbrain import sqlite

resolutions = None # (z, y, x) order since given from microscope
magnification = -1.0
zoom = -1.0

# blob confirmation flags
CONFIRMATION = {
    -1: "unverified",
    0: "no",
    1: "yes",
    2: "maybe"
}

def calc_scaling_factor():
    """Calculates the tolerance based on the resolutions, using the 
    first resolution.
    
    Return:
        Array of tolerance values in same shape as the first resolution.
    
    Raises:
        ``AttributeError`` if the :attr:``resolutions`` is None or has 
        less than one element.
    """
    if resolutions is None or len(resolutions) < 1:
        raise AttributeError(
            "Must load resolutions from file or set a resolution")
    factor = np.divide(1.0, resolutions[0])
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
        channel: Channel to select, which can be None to indicate all 
            channels.
        exclude_borders: Sequence of border pixels in x,y,z to exclude; 
            defaults to None.
    
    Returns:
        Array of detected blobs, each given as 
            (z, row, column, radius, confirmation).
    """
    # use 3D blob detection from skimage fork
    time_start = time()
    shape = roi.shape
    isotropic = config.process_settings["isotropic"]
    if isotropic is not None:
        # interpolate for (near) isotropy during detection, using only the 
        # first process settings since applies to entire ROI
        roi = plot_3d.make_isotropic(roi, isotropic)
    multichannel, channels = plot_3d.setup_channels(roi, channel, 3)
    blobs_all = []
    for i in channels:
        roi_detect = roi[..., i] if multichannel else roi
        settings = config.get_process_settings(i)
        # scaling as a factor in pixel/um, where scaling of 1um/pixel  
        # corresponds to factor of 1, and 0.25um/pixel corresponds to
        # 1 / 0.25 = 4 pixels/um; currently simplified to be based on 
        # x scaling alone
        scale = calc_scaling_factor()
        scaling_factor = scale[2]
        
        overlap = settings["overlap"]
        detect_dict = {}
        if settings["scale_factor"] is not None:
            # anisotropic kernel for blob pruning, which requires a custom 
            # Scikit-image repo and should only be used if not interpolating 
            # for isotropy
            res_norm = np.divide(resolutions[0], np.min(resolutions[0]))
            # further tweak, typically scaling down
            res_norm = np.multiply(res_norm, settings["scale_factor"])
            segmenting_mean = np.mean(roi_detect)
            lib_clrbrain.printv("segmenting_mean: {}".format(segmenting_mean))
            if segmenting_mean > settings["segmenting_mean_thresh"]:
                # turn off scaling for higher density region
                res_norm = None
                overlap += 0.05
            else:
                detect_dict["scale"] = res_norm
        #print("detection dict: {}".format(detect_dict))
        
        # find blobs
        try:
            blobs_log = blob_log(
                roi_detect, 
                min_sigma=settings["min_sigma_factor"]*scaling_factor, 
                max_sigma=settings["max_sigma_factor"]*scaling_factor, 
                num_sigma=settings["num_sigma"], 
                threshold=settings["detection_threshold"], 
                overlap=overlap, **detect_dict)
        except TypeError as e:
            print(e)
            raise TypeError(
                 "If you see an error involving the 'scale' keyword, you need "
                 "\na custom Scikit-image repo extended to include an "
                 "anisotropic kernel, "
                 "\navailable at "
                 "https://github.com/the4thchild/scikit-image.git"
                 "\nbut deprecated in Clrbrain 0.6.8")
        lib_clrbrain.printv(
            "time for 3D blob detection: {}".format(time() - time_start))
        if blobs_log.size < 1:
            lib_clrbrain.printv("no blobs detected")
            continue
        blobs_log[:, 3] = blobs_log[:, 3] * math.sqrt(3)
        blobs = format_blobs(blobs_log, i)
        #print(blobs)
        blobs_all.append(blobs)
    if not blobs_all:
        return None
    blobs_all = np.vstack(blobs_all)
    if isotropic is not None:
        # if detected on isotropic ROI, need to reposition blob coordinates 
        # for original, non-isotropic ROI
        isotropic_factor = plot_3d.calc_isotropic_factor(isotropic)
        blobs_all = multiply_blob_rel_coords(blobs_all, 1 / isotropic_factor)
        blobs_all = multiply_blob_abs_coords(blobs_all, 1 / isotropic_factor)
    
    if exclude_border is not None:
        # exclude blobs from the border in x,y,z
        blobs_all = get_blobs_interior(blobs_all, shape, *exclude_border)
    
    return blobs_all

def format_blobs(blobs, channel=None):
    """Format blobs with additional fields for confirmation, truth, and 
    channel, abs z, abs y, abs x values.
    
    Blobs in Clrbrain can be assumed to start with (z, y, x, radius) but should 
    use ``detector`` functions to manipulate other fields of blob arrays to 
    ensure that the correct columns are accessed.
    
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

def shift_blob_rel_coords(blobs, offset):
    blobs[..., :3] += offset
    return blobs

def shift_blob_abs_coords(blobs, offset):
    blobs[..., -1*len(offset):] += offset
    return blobs

def multiply_blob_rel_coords(blobs, factor):
    if blobs is not None:
        rel_coords = blobs[..., :3] * factor
        blobs[..., :3] = rel_coords.astype(np.int)
    return blobs

def multiply_blob_abs_coords(blobs, factor):
    if blobs is not None:
        start = -1*len(factor)
        abs_coords = blobs[..., start:] * factor
        blobs[..., start:] = abs_coords.astype(np.int)
    return blobs

def get_blob_confirmed(blob):
    if blob.ndim > 1:
        return blob[..., 4]
    return blob[4]

def update_blob_confirmed(blob, confirmed):
    blob[..., 4] = confirmed
    return blob

def get_blob_channel(blob):
    return blob[6]

def get_blobs_channel(blobs):
    return blobs[:, 6]

def replace_rel_with_abs_blob_coords(blobs):
    blobs[:, :3] = blobs[:, 7:10]
    return blobs

def blobs_in_channel(blobs, channel):
    if channel is None:
        return blobs
    return blobs[get_blobs_channel(blobs) == channel]

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
    
    close_master = []
    close = []
    far = np.max(tol) + 1
    # compare each element for differences, weighting based on tolerance; 
    # TODO: incorporate radius
    blobs_diffs_init = np.abs(
        blobs_master[:, :3][:, None] - blobs[:, :3])
    normalize_factor = np.divide(np.max(tol), tol)
    tol = np.multiply(tol, normalize_factor)
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
                    # set to distant value so can check next min diff
                    diffs_sums[blob_master_closest] = far
                else:
                    # no match if none of array dims within tolerance of min diff
                    break
            if found:
                break
    #print("closest:\n{}\nclosest_master:\n{}".format(close, close_master))
    return np.array(close_master, dtype=int), np.array(close, dtype=int)

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
    dtype = lib_clrbrain.dtype_within_range(
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

def get_blobs_in_roi(blobs, offset, size, padding=(0, 0, 0)):
    """Get blobs within an ROI based on offset and size.
    
    Note that dimensions are in x,y,z for natural ordering but may 
    change for consistency with z,y,x ordering used throughout Clrbrain.
    
    Args:
        blobs: The blobs to retrieve, given as 2D array of 
            ``[n, [z, row, column, radius, ...]]``.
        offset: Offset coordinates in x,y,z.
        size: Size of ROI in x,y,z.
        padding: Additional padding outside the ROI to include.
    
    Returns:
        Tuple of blobs within the ROI and the mask used to retrieve 
        these blobs.
    """
    mask = np.all([
        blobs[:, 0] >= offset[2] - padding[2], 
        blobs[:, 0] < offset[2] + size[2] + padding[2],
        blobs[:, 1] >= offset[1] - padding[1], 
        blobs[:, 1] < offset[1] + size[1] + padding[1],
        blobs[:, 2] >= offset[0] - padding[0], 
        blobs[:, 2] < offset[0] + size[0] + padding[0]], axis=0)
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

def verify_rois(rois, blobs, blobs_truth, tol, output_db, exp_id, channel):
    """Compares blobs from detections with truth blobs, prioritizing the inner 
    portion of ROIs to avoid missing detections because of edge effects
    while also adding matches between a blob in the inner ROI and another
    blob in the remaining portion of the ROI.
    
    Saves the verifications to a separate database with a name in the same
    format as saved processed files but with "_verified.db" at the end.
    Prints basic statistics on the verification.
    
    Note that blobs are found from ROI parameters rather than loading from 
    database, so blobs recorded within these ROI bounds but from different 
    ROIs will be included in the verification.
    
    Args:
        rois: Rows of ROIs from sqlite database.
        blobs: The blobs to be checked for accuracy, given as 2D 
            array of [n, [z, row, column, radius, ...]].
        blobs_truth: The list by which to check for accuracy, in the same
            format as blobs.
        tol: Tolerance as z,y,x to check for closeness . Blobs that are 
            equal to or less than the the absolute difference for all 
            corresponding parameters will be considered a potential match.
        output_db: Database in which to save the verification flags, typicall 
            the database in :attr:``config.verified_db``.
        exp_id: Experiment ID in ``output_db``.
        channel: Filter ``blobs_truth`` by this channel.
    """
    blobs_truth = blobs_in_channel(blobs_truth, channel)
    blobs_truth_rois = None
    blobs_rois = None
    rois_falsehood = []
    # average overlap and tolerance for padding    
    #tol[0] -= 1
    inner_padding = np.flipud(np.ceil(tol))
    #tol = np.flipud(inner_padding)
    lib_clrbrain.printv(
        "verifying blobs with tol {}, inner_padding {}"
        .format(tol, inner_padding))
    settings = config.get_process_settings(channel)
    resize = settings["resize_blobs"]
    if resize:
        blobs = multiply_blob_rel_coords(blobs, resize)
        #tol = np.multiply(resize, tol).astype(np.int)
        lib_clrbrain.printv("resized blobs by {}:\n{}".format(resize, blobs))
    for roi in rois:
        offset = (roi["offset_x"], roi["offset_y"], roi["offset_z"])
        size = (roi["size_x"], roi["size_y"], roi["size_z"])
        series = roi["series"]
        
        # get all detected and truth blobs for inner and total ROI
        offset_inner = np.add(offset, inner_padding)
        size_inner = np.subtract(size, inner_padding * 2)
        lib_clrbrain.printv(
            "offset: {}, offset_inner: {}, size: {}, size_inner: {}"
            .format(offset, offset_inner, size, size_inner))
        blobs_roi, _ = get_blobs_in_roi(blobs, offset, size)
        if resize is not None:
            # TODO: doesn't align with exported ROIs
            lib_clrbrain.printv("shifting blobs in ROI by offset {}, border {}"
                                .format(offset, config.border))
            blobs_roi = shift_blob_rel_coords(blobs_roi, offset)
            if config.border:
                blobs_roi = shift_blob_rel_coords(blobs_roi, config.border)
        blobs_inner, blobs_inner_mask = get_blobs_in_roi(
            blobs_roi, offset_inner, size_inner)
        blobs_truth_roi, _ = get_blobs_in_roi(blobs_truth, offset, size)
        blobs_truth_inner, blobs_truth_inner_mask = get_blobs_in_roi(
            blobs_truth_roi, offset_inner, size_inner)
        lib_clrbrain.printv("blobs_roi:\n{}".format(blobs_roi))
        lib_clrbrain.printv("blobs_inner:\n{}".format(blobs_inner))
        lib_clrbrain.printv("blobs_truth_inner:\n{}".format(blobs_truth_inner))
        lib_clrbrain.printv("blobs_truth_roi:\n{}".format(blobs_truth_roi))
        
        # compare inner region of detected cells with all truth ROIs, where
        # closest blob detector prioritizes the closest matches
        found_truth, detected = _find_closest_blobs(
            blobs_inner, blobs_truth_roi, tol)
        blobs_inner[: , 4] = 0
        blobs_inner[detected, 4] = 1
        blobs_truth_roi[blobs_truth_inner_mask, 5] = 0
        blobs_truth_roi[found_truth, 5] = 1
        lib_clrbrain.printv("detected inner:\n{}"
              .format(blobs_inner[blobs_inner[:, 4] == 1]))
        lib_clrbrain.printv("truth detected:\n{}"
              .format(blobs_truth_roi[blobs_truth_roi[:, 5] == 1]))
        
        # add any truth blobs missed in the inner ROI by comparing with 
        # outer ROI of detected blobs
        blobs_truth_inner_missed = blobs_truth_roi[blobs_truth_roi[:, 5] == 0]
        blobs_outer = blobs_roi[np.invert(blobs_inner_mask)]
        lib_clrbrain.printv("blobs_outer:\n{}".format(blobs_outer))
        lib_clrbrain.printv(
            "blobs_truth_inner_missed:\n{}".format(blobs_truth_inner_missed))
        found_truth_out, detected = _find_closest_blobs(
            blobs_outer, blobs_truth_inner_missed, tol)
        blobs_truth_inner_missed[found_truth_out, 5] = 1
        blobs_truth_inner_plus = np.concatenate(
            (blobs_truth_roi[blobs_truth_roi[:, 5] == 1], 
             blobs_truth_inner_missed))
        blobs_roi_extra = blobs_outer[detected]
        blobs_roi_extra[:, 4] = 1
        blobs_inner_plus = np.concatenate((blobs_inner, blobs_roi_extra))
        lib_clrbrain.printv(
            "truth blobs detected by an outside blob:\n{}".format(
            blobs_truth_inner_missed[blobs_truth_inner_missed[:, 5] == 1]))
        lib_clrbrain.printv(
            "all those outside detection blobs:\n{}".format(blobs_roi_extra))
        lib_clrbrain.printv(
            "blobs_inner_plus:\n{}".format(blobs_inner_plus))
        lib_clrbrain.printv(
            "blobs_truth_inner_plus:\n{}".format(blobs_truth_inner_plus))
        
        # store blobs in separate verified DB
        roi_id, _ = sqlite.insert_roi(output_db.conn, output_db.cur, exp_id, 
                                      series, offset_inner,size_inner)
        sqlite.insert_blobs(output_db.conn, output_db.cur, roi_id, 
                            blobs_inner_plus)
        sqlite.insert_blobs(output_db.conn, output_db.cur, roi_id, 
                            blobs_truth_inner_plus)
        true_pos = len(blobs_inner_plus[blobs_inner_plus[:, 4] == 1])
        false_pos = len(blobs_inner_plus[blobs_inner_plus[:, 4] == 0])
        false_neg = len(blobs_truth_inner_plus) - true_pos
        if false_neg > 0 or false_pos > 0:
            rois_falsehood.append((offset_inner, false_pos, false_neg))
        
        # combine blobs into total lists for stats
        if blobs_truth_rois is None:
            blobs_truth_rois = blobs_truth_inner_plus
        else:
            blobs_truth_rois = np.concatenate(
                (blobs_truth_inner_plus, blobs_truth_rois))
        if blobs_rois is None:
            blobs_rois = blobs_inner_plus
        else:
            blobs_rois = np.concatenate((blobs_inner_plus, blobs_rois))
    
    true_pos = len(blobs_rois[blobs_rois[:, 4] == 1])
    false_pos = len(blobs_rois[blobs_rois[:, 4] == 0])
    pos = len(blobs_truth_rois)
    false_neg = pos - true_pos
    sens = float(true_pos) / pos
    ppv = float(true_pos) / (true_pos + false_pos)
    print("Automated verification using tol {}:\n".format(tol))
    fdbk = ("cells = {}\ndetected cells = {}\n"
            "false pos cells = {}\nfalse neg cells = {}\nsensitivity = {}\n"
            "PPV = {}\n".format(pos, true_pos, false_pos, 
            false_neg, sens, ppv))
    print(fdbk)
    print("ROIs with falsehood:\n{}".format(rois_falsehood))
    return (pos, true_pos, false_pos), fdbk

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

def _test_blob_verification():
    a = np.ones((3, 3))
    a[:, 0] = [0, 1, 2]
    b = np.copy(a)
    b[:, 0] += 1
    print("test (b):\n{}".format(b))
    print("master (a):\n{}".format(a))
    _find_closest_blobs(b, a, slice(0, 3), (1, 2, 2))
    a = np.array([[24, 52, 346], [20, 55, 252]])
    b = np.array([[24, 54, 351]])
    print("test (b):\n{}".format(b))
    print("master (a):\n{}".format(a))
    _find_closest_blobs(b, a, slice(0, 3), (3, 9, 9))

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

if __name__ == "__main__":
    print("Detector tests...")
    #_test_blob_close_sorted()
    _test_blob_verification()
