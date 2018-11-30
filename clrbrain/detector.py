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
from skimage.feature import peak_local_max
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
    """Calculates the tolerance based on the  
    resolutions, using the first resolution.
    
    Return:
        Array of tolerance values in same shape as resolution.
    """
    if resolutions is None:
        raise AttributeError(
            "Must load resolutions from file or set a resolution")
    factor = np.divide(1.0, resolutions[0])
    lib_clrbrain.printv(
        "microsope scaling factor based on resolutions: {}".format(factor))
    return factor

def markers_from_blobs(roi, blobs):
    # use blobs as seeds by converting blobs into marker image
    markers = np.zeros(roi.shape, dtype=np.uint8)
    coords = np.transpose(blobs[:, :3]).astype(np.int)
    coords = np.split(coords, coords.shape[0])
    markers[tuple(coords)] = 1
    markers = morphology.dilation(markers, morphology.ball(1))
    markers = measure.label(markers)
    return markers

def segment_rw(roi, channel, beta=50.0, vmin=0.6, vmax=0.65, remove_small=None, 
               erosion=None, blobs=None, get_labels=False):
    """Segments an image, drawing contours around segmented regions.
    
    Args:
        roi: Region of interest to segment.
        channel: Channel to pass to :func:``plot_3d.setup_channels``.
        beta: Random-Walker beta term.
        vmin: Values under which to exclude in markers; defaults to 0.6. 
            Ignored if ``blobs`` is given.
        vmax: Values above which to exclude in markers; defaults to 0.65. 
            Ignored if ``blobs`` is given.
        remove_small: Threshold size of small objects to remove; defaults 
            to None to ignore.
        erosion: Structuring element size for erosion; defaults 
            to None to ignore.
        blobs: Blobs to use for markers; defaults to None, in which 
            case markers will be determined based on ``vmin``/``vmax`` 
            thresholds.
        get_labels: True to measure and return labels from the 
            resulting segmentation.
    
    Returns:
        The Random-Walker segmentation, or the measured labels 
        for the segmented regions if ``get_labels`` is True.
    """
    print("Random-Walker based segmentation...")
    labels = []
    walkers = []
    multichannel, channels = plot_3d.setup_channels(roi, channel, 3)
    for i in channels:
        roi_segment = roi[..., i] if multichannel else roi
        #settings = config.get_process_settings(i)
        if blobs is None:
            # mark unknown pixels as 0 by distinguishing known background 
            # and foreground
            markers = np.zeros(roi_segment.shape, dtype=np.uint8)
            markers[roi_segment < vmin] = 2
            markers[roi_segment >= vmax] = 1
        else:
            # derive markers from blobs
            markers = markers_from_blobs(roi_segment, blobs)
        walker = segmentation.random_walker(
            roi_segment, markers, beta=beta, mode="cg_mg")
        if remove_small:
            walker = morphology.remove_small_objects(walker, remove_small)
        
        if erosion:
            # attempt to reduce label connections by eroding
            walker = morphology.erosion(walker, morphology.octahedron(erosion))
        # clean up by using simple threshold to remove all background
        roi_thresh = filters.threshold_mean(roi_segment)
        thresholded = roi_segment > roi_thresh
        walker[~thresholded] = 0
        
        if get_labels:
            # label neighboring pixels to segmented regions
            # TODO: check if necessary; useful only if blobs not given?
            label = measure.label(walker, background=0)
            labels.append(label)
            #print("label:\n", label)
        
        walkers.append(walker)
        #lib_clrbrain.show_full_arrays()
        #print(walker)
    
    if get_labels:
        return labels
    return walkers

def segment_ws(roi, thresholded=None, blobs=None): 
    """Segment an ROI using a 3D-seeded watershed.
    
    Args:
        roi: ROI as a Numpy array in (z, y, x) order.
        thresholded: Thresholded image such as a segmentation into foreground/
            background given by Random-walker (:func:``segment_rw``). 
            Defaults to None, in which case Otsu thresholding will be performed.
        blobs: Blobs as a Numpy array in [[z, y, x, ...], ...] order, which 
            are used as seeds for the watershed. Defaults to None, in which 
            case peaks on a distance transform will be used.
    
    Returns:
        Watershed labels as in the shape of [roi.shape], allowing for 
        multiple sets of labels.
    """
    # TODO: extend to multichannel
    roi = roi[..., 0]
    if thresholded is None:
        # Ostu thresholing and object separate based on local max rather than 
        # seeded watershed approach
        roi_thresh = filters.threshold_otsu(roi, 64)
        thresholded = roi > roi_thresh
    else:
        thresholded = thresholded[0] - 1 # r-w assigned 0 values to > 0 val labels
    
    # distance transform to find boundaries in thresholded image
    distance = ndimage.distance_transform_edt(thresholded)
    
    if blobs is None:
        # default to finding peaks of distance transform if no blobs given, 
        # using an anisotropic footprint
        try:
            local_max = peak_local_max(
                distance, indices=False, footprint=np.ones((1, 3, 3)), 
                labels=thresholded)
        except IndexError as e:
            print(e)
            raise e
        markers = measure.label(local_max)
    else:
        markers = markers_from_blobs(thresholded, blobs)
    
    # watershed with slight increase in compactness to give basins with 
    # more regular, larger shape, and minimize number of small objects
    labels_ws = morphology.watershed(-distance, markers, compactness=0.01)
    labels_ws = morphology.remove_small_objects(labels_ws, min_size=100)
    #print("num ws blobs: {}".format(len(np.unique(labels_ws)) - 1))
    print(labels_ws)
    labels_ws = labels_ws[None]
    return labels_ws

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

def detect_blobs(roi, channel):
    """Detects objects using 3D blob detection technique.
    
    Args:
        roi: Region of interest to segment.
    
    Returns:
        Array of detected blobs, each given as 
            (z, row, column, radius, confirmation).
    """
    # use 3D blob detection from skimage fork
    time_start = time()
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

def shift_blob_rel_coords(blobs, offset):
    blobs[..., :3] += offset
    return blobs

def shift_blob_abs_coords(blobs, offset):
    blobs[..., -1*len(offset):] += offset
    return blobs

def multiply_blob_rel_coords(blobs, factor):
    if blobs is not None:
        rel_coords = blobs[..., :3] * factor
        blobs[..., :3] = np.around(rel_coords).astype(np.int)
    return blobs

def multiply_blob_abs_coords(blobs, factor):
    if blobs is not None:
        start = -1*len(factor)
        abs_coords = blobs[..., start:] * factor
        blobs[..., start:] = np.around(abs_coords).astype(np.int)
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

def replace_rel_with_abs_blob_coords(blobs):
    blobs[:, :3] = blobs[:, 7:]
    return blobs

def blobs_in_channel(blobs, channel):
    if channel is None:
        return blobs
    return blobs[blobs[:, 6] == channel]

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

def _find_close_blobs(blobs, blobs_master, region, tol):
    # creates a separate array for each blob in blobs_master to allow
    # comparison for each of its blobs with each blob to add
    blobs_diffs = np.abs(blobs_master[:, region][:, None] - blobs[:, region])
    close_master, close = np.nonzero((blobs_diffs <= tol).all(2))
    #print("close:\n{}\nclose_master:\n{}".format(close, close_master))
    return close_master, close

def _find_closest_blobs(blobs, blobs_master, region, tol):
    """Finds the closest matching blobs between two arrays. Each entry will 
    have no more than one match, and the total number of matches will be 
    the size of the shortest list.
    
    Args:
        blobs: The blobs to be checked for closeness, given as 2D 
            array of at least [n, [z, row, column, ...]].
        blobs_master: The list by which to check for close blobs, in the same
            format as blobs.
        region: Slice within each blob to check, such as slice(0, 2) to check
            for (z, row, column).
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
    close_master = []
    close = []
    far = np.max(tol) + 1
    # compare each element for differences, weighting based on tolerance; 
    # TODO: incorporate radius
    blobs_diffs_init = np.abs(
        blobs_master[:, region][:, None] - blobs[:, region])
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

def remove_close_blobs(blobs, blobs_master, region, tol):
    """Removes blobs that are close to one another.
    
    Args:
        blobs: The blobs to be checked for closeness and pruning, given as 2D 
            array of [n, [z, row, column, radius]].
        blobs_master: The list by which to check for close blobs, in the same
            format as blobs.
        region: Slice within each blob to check, such as slice(0, 2) to check
            for (z, row, column).
        tol: Tolerance to check for closeness, given in the same format
            as region. Blobs that are equal to or less than the the absolute
            difference for all corresponding parameters will be pruned in
            the returned array.
    
    Return:
        The blobs array without blobs falling inside the tolerance range.
    """
    close_master, close = _find_close_blobs(blobs, blobs_master, region, tol)
    pruned = np.delete(blobs, close, axis=0)
    if (len(close) > 0):
        print("{} removed".format(blobs[close][:, 0:4]))
    
    # shift close blobs to their mean values, storing values in the duplicated
    # coordinates and radius of the blob array after the confirmation value;
    # use the duplicated coordinates to work from any prior shifting; 
    # further duplicate testing will still be based on initial position to
    # allow detection of duplicates that occur in multiple ROI pairs
    blobs_master[close_master, 6:] = np.around(
        np.divide(np.add(blobs_master[close_master, 6:], 
                         blobs[close, 6:]), 2))
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

def remove_close_blobs_within_sorted_array(blobs, region, tol):
    """Removes close blobs within a given array, first sorting the array by
    z, y, x.
    
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
            # check each blob to add against the last approved blob since
            # assumes that blobs are sorted, so only need to check last blob
            i = len(blobs_all) - 1
            while i >= 0:
                blobs_diff = np.abs(np.subtract(
                    blob[region], blobs_all[i, region]))
                #print(blobs_diff)
                if (blobs_diff <= tol).all():
                    # duplicate blob to be removed;
                    # shift close blobs to their mean values, storing values in 
                    # the duplicated coordinates and radius of the blob array
                    blobs_all[i, 6:] = np.around(
                        np.divide(np.add(blobs_all[i, 6:], blob[6:]), 2))
                    #print("{} removed".format(blob))
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
    mask = np.all([
        blobs[:, 0] >= offset[2] - padding[2], 
        blobs[:, 0] < offset[2] + size[2] + padding[2],
        blobs[:, 1] >= offset[1] - padding[1], 
        blobs[:, 1] < offset[1] + size[1] + padding[1],
        blobs[:, 2] >= offset[0] - padding[0], 
        blobs[:, 2] < offset[0] + size[0] + padding[0]], axis=0)
    segs_all = blobs[mask]
    return segs_all, mask

def verify_rois(rois, blobs, blobs_truth, region, tol, output_db, 
                exp_id, channel):
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
        region: Slice within each blob to check, such as slice(0, 3) to check
            for (z, row, column).
        tol: Tolerance to check for closeness, given in the same format
            as region. Blobs that are equal to or less than the the absolute
            difference for all corresponding parameters will be considered
            a potential match.
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
            blobs_inner, blobs_truth_roi, region, tol)
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
            blobs_outer, blobs_truth_inner_missed, region, tol)
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
    blobs = remove_close_blobs_within_sorted_array(blobs, slice(0, 3), (1, 2, 2))
    print("pruned:\n{}".format(blobs))

if __name__ == "__main__":
    print("Detector tests...")
    #_test_blob_close_sorted()
    _test_blob_verification()
