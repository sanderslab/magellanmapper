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
from skimage import exposure
from skimage import segmentation
from skimage import measure
from skimage import morphology
from skimage.feature import blob_log
from skimage.feature import peak_local_max

from clrbrain import config
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
    print("scaling_factor: {}".format(factor))
    return factor

def _histogram_roi(roi):
    #histo = np.histogram(roi, bins=np.arange(0, 2, 0.1))
    histo = exposure.histogram(roi)
    from clrbrain import plot_2d
    plot_2d.plot_histogram_exposure(histo)

def _measure_coords(labels):
    segs = []
    for prop in measure.regionprops(labels):
        seg = list(prop.centroid)
        seg.append(prop.equivalent_diameter / 2)
        print(seg)
        segs.append(seg)
    return segs

def segment_ws(roi):
    """Segments an image, drawing contours around segmented regions.
    
    Args:
        roi: Region of interest to segment.
    
    Returns:
        Labels for the segmented regions, which can be plotted as surfaces.
    """
    #np.set_printoptions(linewidth=200, threshold=1000)
    distance = ndimage.distance_transform_edt(roi)
    try:
        local_max = peak_local_max(distance, indices=False, footprint=morphology.ball(1), labels=roi)
    except IndexError as e:
        print(e)
        raise e
    markers = morphology.label(local_max)
    labels_ws = morphology.watershed(-distance, markers, mask=roi)
    labels_ws = morphology.remove_small_objects(labels_ws, min_size=100)
    #labels_ws = markers
    '''
    print("labels_ws max: {}".format(np.max(labels_ws)))
    print("labels_ws:\n{}".format(labels_ws))
    print("markers:\n{}".format(markers))
    #segs = ndimage.find_objects(labels_ws)
    labels_segs = ndimage.label(labels_ws)
    print("labels_segs:\n{}".format(labels_segs))
    #print(labels_segs[0].shape)
    segs = ndimage.center_of_mass(labels_ws, labels_segs[0], np.arange(1, np.max(labels_segs[0])))
    print("watershed labels:\n{}".format(segs))
    #walker = labels_ws
    #labels = labels_ws
    #labels = segs
    '''
    segs = _measure_coords(labels_ws)
    return labels_ws, segs
    
def segment_rw(roi, beta=50.0):
    print("Random-Walker based segmentation...")
    _histogram_roi(roi)
    # random-walker segmentation
    markers = np.zeros(roi.shape, dtype=np.uint8)
    markers[roi > 1.3] = 1
    markers[roi < 1.0] = 2
    
    #markers[~roi] = -1
    walker = segmentation.random_walker(roi, markers, beta=beta, mode="bf")
    
    # label neighboring pixels to segmented regions
    #walker = morphology.remove_small_objects(walker, 3000)
    labels = measure.label(walker, background=0)
    segs = _measure_coords(labels)
    
    return labels, segs

def _blob_surroundings(blob, roi, padding, plane=False):
    rad = blob[3]
    start = np.subtract(blob[0:3], padding).astype(int)
    start[start < 0] = 0
    end = np.add(blob[0:3], padding).astype(int)
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

def blobs_within(blobs, offset, size):
    return np.all([blobs[:, 0] >= offset[2], blobs[:, 0] < offset[2] + size[2],
                      blobs[:, 1] >= offset[1], blobs[:, 1] < offset[1] + size[1],
                      blobs[:, 2] >= offset[0], blobs[:, 2] < offset[0] + size[0]], 
                     axis=0)

def show_blob_surroundings(blobs, roi, padding=1):
    print("showing blob surroundings")
    np.set_printoptions(precision=2, linewidth=200)
    for blob in blobs:
        print("{} surroundings:".format(blob))
        surroundings = _blob_surroundings(blob, roi, padding, True)
        print("{}\n".format(surroundings))
    np.set_printoptions()

def segment_blob(roi):
    """Detects objects using 3D blob detection technique.
    
    Args:
        roi: Region of interest to segment.
    
    Returns:
        Array of detected blobs, each given as 
            (z, row, column, radius, confirmation).
    """
    # use 3D blob detection from skimage v.0.13pre
    time_start = time()
    settings = config.process_settings
    # scaling as a factor in pixel/um, where scaling of 1um/pixel  
    # corresponds to factor of 1, and 0.25um/pixel corresponds to
    # 1 / 0.25 = 4 pixels/um; currently simplified to be based on 
    # x scaling alone
    scale = calc_scaling_factor()
    scaling_factor = scale[2]
    
    # adjust scaling for blob pruning
    res_norm = np.divide(resolutions[0], np.min(resolutions[0]))
    # further tweak, typically scaling down
    res_norm = np.multiply(res_norm, settings["scale_factor"])
    segmenting_mean = np.mean(roi)
    #print("min: {}, max: {}".format(np.min(roi), np.max(roi)))
    print("segmenting_mean: {}".format(segmenting_mean))
    overlap = settings["overlap"]
    if segmenting_mean > settings["segmenting_mean_thresh"]:
        # turn off scaling for higher density region
        res_norm = None
        overlap += 0.05
    #print("res_norm: {}".format(res_norm))
    
    # find blobs
    blobs_log = blob_log(roi, 
                         min_sigma=settings["min_sigma_factor"]*scaling_factor, 
                         max_sigma=settings["max_sigma_factor"]*scaling_factor, 
                         num_sigma=settings["num_sigma"], 
                         threshold=0.1,
                         overlap=overlap, scale=res_norm)
    print("time for 3D blob detection: %f" %(time() - time_start))
    if blobs_log.size < 1:
        print("no blobs detected")
        return None, None
    blobs_log[:, 3] = blobs_log[:, 3] * math.sqrt(3)
    
    labels = None
    try:
        labels, segs = segment_ws(roi)
        segs = np.rint(segs)
        '''
        if len(segs) > 0:
            blobs_log = np.concatenate((segs, np.multiply(np.ones((segs.shape[0], 1)), 5)), axis=1)
        '''
        mask_big_blobs = blobs_log[:, 3] > 5
        big_blobs = blobs_log[mask_big_blobs]
        blobs_log = blobs_log[np.invert(mask_big_blobs)]
        scaling = calc_scaling_factor()
        for big_blob in big_blobs:
            radius = big_blob[3]
            padding = np.multiply(radius, scaling)
            print("big_blob: {}, padding: {}".format(big_blob, padding))
            #labels, segs = segment_ws(_blob_surroundings(big_blob, roi, padding))
            #segs = np.rint(segs)
            nearby_blobs = segs[blobs_within(segs, np.subtract(big_blob[:3], padding)[::-1], np.multiply(padding, 2)[::-1])]
            ws_blobs = []
            #ws_radius = radius / segs.shape[0]
            for seg in nearby_blobs:
                if not np.any(np.isnan(seg)) and np.any((a == x).all() for x in ws_blobs):
                    ws_blobs.append(seg)
            if len(ws_blobs) > 1:
                ws_blobs = np.array(ws_blobs)
                print("adding from watershed:\n{}".format(ws_blobs))
                blobs_log = np.concatenate((blobs_log, ws_blobs))
            else:
                blobs_log = np.concatenate((blobs_log, [big_blob]))
    except IndexError:
        print("Unable to watershed segment, skipping")
    
    print(blobs_log)
    print("found {} blobs".format(blobs_log.shape[0]))
    # adding fields for confirmation and truth flags
    extras = np.ones((blobs_log.shape[0], 2)) * -1
    blobs = np.concatenate((blobs_log, extras), axis=1)
    return blobs, labels

def remove_duplicate_blobs(blobs, region):
    """Removes duplicate blobs.
    
    Params:
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
    
    Params:
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
    # compare each element for differences, weighting based on tolerance
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
    
    Params:
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
    
    Params:
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
    
    Params:
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
                exp_id):
    """Compares blobs from detections with truth blobs, prioritizing the inner 
    portion of ROIs to avoid missing detections because of edge effects
    while also adding matches between a blob in the inner ROI and another
    blob in the remaining portion of the ROI.
    
    Saves the verifications to a separate database with a name in the same
    format as saved processed files but with "_verified.db" at the end.
    Prints basic statistics on the verification.
    
    Params:
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
    """
    blobs_truth_rois = None
    blobs_rois = None
    rois_falsehood = []
    # average overlap and tolerance for padding    
    #tol[0] -= 1
    inner_padding = np.flipud(np.ceil(tol))
    #tol = np.flipud(inner_padding)
    print("verifying blobs with tol {}, inner_padding {}"
          .format(tol, inner_padding))
    for roi in rois:
        offset = (roi["offset_x"], roi["offset_y"], roi["offset_z"])
        size = (roi["size_x"], roi["size_y"], roi["size_z"])
        series = roi["series"]
        
        # get all detected and truth blobs for inner and total ROI
        offset_inner = np.add(offset, inner_padding)
        size_inner = np.subtract(size, inner_padding * 2)
        print("offset: {}, offset_inner: {}, size: {}, size_inner: {}"
              .format(offset, offset_inner, size, size_inner))
        blobs_roi, _ = get_blobs_in_roi(blobs, offset, size)
        blobs_inner, blobs_inner_mask = get_blobs_in_roi(
            blobs_roi, offset_inner, size_inner)
        blobs_truth_roi, _ = get_blobs_in_roi(blobs_truth, offset, size)
        blobs_truth_inner, blobs_truth_inner_mask = get_blobs_in_roi(
            blobs_truth_roi, offset_inner, size_inner)
        print("blobs_roi:\n{}".format(blobs_roi))
        print("blobs_inner:\n{}".format(blobs_inner))
        print("blobs_truth_inner:\n{}".format(blobs_truth_inner))
        print("blobs_truth_roi:\n{}".format(blobs_truth_roi))
        
        # compare inner region of detected cells with all truth ROIs, where
        # closest blob detector prioritizes the closest matches
        found_truth, detected = _find_closest_blobs(
            blobs_inner, blobs_truth_roi, region, tol)
        blobs_inner[: , 4] = 0
        blobs_inner[detected, 4] = 1
        blobs_truth_roi[blobs_truth_inner_mask, 5] = 0
        blobs_truth_roi[found_truth, 5] = 1
        print("detected inner:\n{}"
              .format(blobs_inner[blobs_inner[:, 4] == 1]))
        print("truth detected:\n{}"
              .format(blobs_truth_roi[blobs_truth_roi[:, 5] == 1]))
        
        # add any truth blobs missed in the inner ROI by comparing with 
        # outer ROI of detected blobs
        blobs_truth_inner_missed = blobs_truth_roi[blobs_truth_roi[:, 5] == 0]
        blobs_outer = blobs_roi[np.invert(blobs_inner_mask)]
        print("blobs_outer:\n{}".format(blobs_outer))
        print("blobs_truth_inner_missed:\n{}".format(blobs_truth_inner_missed))
        found_truth_out, detected = _find_closest_blobs(
            blobs_outer, blobs_truth_inner_missed, region, tol)
        blobs_truth_inner_missed[found_truth_out, 5] = 1
        blobs_truth_inner_plus = np.concatenate(
            (blobs_truth_roi[blobs_truth_roi[:, 5] == 1], 
             blobs_truth_inner_missed))
        blobs_roi_extra = blobs_outer[detected]
        blobs_roi_extra[:, 4] = 1
        blobs_inner_plus = np.concatenate((blobs_inner, blobs_roi_extra))
        print("truth blobs detected by an outside blob:\n{}".format(
              blobs_truth_inner_missed[blobs_truth_inner_missed[:, 5] == 1]))
        print("all those outside detection blobs:\n{}".format(blobs_roi_extra))
        print("blobs_inner_plus:\n{}".format(blobs_inner_plus))
        print("blobs_truth_inner_plus:\n{}".format(blobs_truth_inner_plus))
        
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
