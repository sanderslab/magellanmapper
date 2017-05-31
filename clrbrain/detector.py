# Cell detection methods
# Author: David Young, 2017
"""Detects features within a 3D image stack.

Provides options for segmentation and blob detection techniques.

Attributes:
    resolutions: The image resolutions as an array of dimensions (n, r),
        where each resolution r is a tuple in (z, y, x) order.
"""

from time import time
import math
import numpy as np
from skimage import segmentation
from skimage import measure
from skimage import morphology
from skimage.feature import blob_log

from clrbrain import config
from clrbrain import plot_3d
from clrbrain import sqlite

resolutions = None # (z, y, x) order since given from microscope
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
        raise AttributeError("Must load resolutions from file or set a resolution")
    factor = np.divide(1.0, resolutions[0])
    print("scaling_factor: {}".format(factor))
    return factor

def segment_rw(roi):
    """Segments an image, drawing contours around segmented regions.
    
    Args:
        roi: Region of interest to segment.
    
    Returns:
        Labels for the segmented regions, which can be plotted as surfaces.
    """
    print("Random-Walker based segmentation...")
    # random-walker segmentation
    markers = np.zeros(roi.shape, dtype=np.uint8)
    markers[roi > 0.4] = 1
    markers[roi < 0.33] = 2
    walker = segmentation.random_walker(roi, markers, beta=1000., mode="bf")
    
    # label neighboring pixels to segmented regions
    walker = morphology.remove_small_objects(walker == 1, 200)
    labels = measure.label(walker, background=0)
    
    return labels, walker

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
    # scaling as a factor in pixel/um, where scaling of 1um/pixel  
    # corresponds to factor of 1, and 0.25um/pixel corresponds to
    # 1 / 0.25 = 4 pixels/um; currently simplified to be based on 
    # x scaling alone
    scaling_factor = calc_scaling_factor()[2]
    settings = config.process_settings
    blobs_log = blob_log(roi, min_sigma=3*scaling_factor, 
                         max_sigma=settings["max_sigma_factor"]*scaling_factor, 
                         num_sigma=settings["num_sigma"], 
                         threshold=0.1,
                         overlap=settings["overlap"])
    print("time for 3D blob detection: %f" %(time() - time_start))
    if blobs_log.size < 1:
        print("no blobs detected")
        return None
    blobs_log[:, 3] = blobs_log[:, 3] * math.sqrt(3)
    print(blobs_log)
    print("found {} blobs".format(blobs_log.shape[0]))
    extras = np.ones((blobs_log.shape[0], 2)) * -1
    blobs = np.concatenate((blobs_log, extras), axis=1)
    return blobs

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
    blobs_type = np.dtype((np.void, blobs_region.dtype.itemsize * blobs_region.shape[1]))
    blobs_contig = blobs_contig.view(blobs_type)
    _, unique_indices = np.unique(blobs_contig, return_index=True)
    # TODO: should probably only look within blobs_region
    print("removed {} duplicate blobs".format(blobs.shape[0] - unique_indices.size))
    return blobs[unique_indices]

def _find_close_blobs(blobs, blobs_master, region, tol):
    # creates a separate array for each blob in blobs_master to allow
    # comparison for each of its blobs with each blob to add
    blobs_diffs = np.abs(blobs_master[:, region][:, None] - blobs[:, region])
    close_master, close = np.nonzero((blobs_diffs <= tol).all(2))
    print("close:\n{}\nclose_master:\n{}".format(close, close_master))
    return close_master, close

def _find_closest_blobs(blobs, blobs_master, region, tol):
    close_master = []
    close = []
    far = np.max(tol) + 1
    blobs_diffs = np.abs(blobs_master[:, region][:, None] - blobs[:, region])
    diffs_sums = np.sum(blobs_diffs, blobs_diffs.ndim - 1)
    i = 0
    while i < len(blobs_master) and i < len(blobs):
        print("diffs_sums:\n{}".format(diffs_sums))
        '''
        mins = np.min(diffs_sums, diffs_sums.ndim - 1)
        blob_master_closest = np.argmin(mins)
        blob_closest = np.argmin(diffs_sums[blob_master_closest])
        '''
        min_master, min_blob = np.where(diffs_sums == diffs_sums.min())
        print("min_master: {}, min_blob: {}".format(min_master, min_blob))
        blob_master_closest = min_master[0]
        blob_closest = min_blob[0]
        if (blobs_diffs[blob_master_closest, blob_closest] < tol).all():
            close_master.append(blob_master_closest)
            close.append(blob_closest)
        diffs_sums[blob_master_closest, blob_closest] = far
        i += 1
    print("closest:\n{}\nclosest_master:\n{}".format(close, close_master))
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
    print("removed {} close blobs:\n{}".format(len(close), blobs[close][:, 0:4]))
    
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
    blobs_all = None
    for blob in blobs:
        #print("blob: {}".format(blob))
        if blobs_all is None:
            blobs_all = np.array([blob])
        else:
            blobs_to_add, blobs_all = remove_close_blobs(
                np.array([blob]), blobs_all, region, tol)
            if blobs_to_add is not None:
                blobs_all = np.concatenate((blobs_all, blobs_to_add))
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

def verify_rois(rois, blobs, blobs_truth, region, tol, output_db, exp_id):
    blobs_truth_rois = None
    blobs_rois = None
    np.set_printoptions(linewidth=200, threshold=10000)
    for roi in rois:
        offset = (roi["offset_x"], roi["offset_y"], roi["offset_z"])
        size = (roi["size_x"], roi["size_y"], roi["size_z"])
        series = roi["series"]
        
        # get all detected and truth blobs for in or total ROI
        inner_padding = np.ceil(tol[::-1] * 0.5)
        offset_inner = np.add(offset, inner_padding)
        size_inner = np.subtract(size, inner_padding * 2)
        print("offset: {}, offset_inner: {}, size: {}, size_inner: {}".format(offset, offset_inner, size, size_inner))
        blobs_roi, _ = get_blobs_in_roi(blobs, offset, size)
        blobs_inner, blobs_inner_mask = get_blobs_in_roi(blobs, offset_inner, size_inner)
        blobs_truth_inner, _ = get_blobs_in_roi(blobs_truth, offset_inner, size_inner)
        blobs_truth_roi, _ = get_blobs_in_roi(blobs_truth, offset, size)
        print("blobs_roi:\n{}".format(blobs_roi))
        print("blobs_inner:\n{}".format(blobs_inner))
        print("blobs_truth_inner:\n{}".format(blobs_truth_inner))
        print("blobs_truth_roi:\n{}".format(blobs_truth_roi))
        
        # compare inner regions for simplest overlap
        found_truth, detected = _find_closest_blobs(blobs_inner, blobs_truth_inner, region, tol)
        blobs_inner[: , 4] = 0
        blobs_inner[detected, 4] = 1
        blobs_truth_inner[:, 5] = 0
        blobs_truth_inner[found_truth, 5] = 1
        print("detected inner:\n{}".format(blobs_inner[blobs_inner[:, 4] == 1]))
        
        # for missed blobs, check against a slightly expanded area up to the 
        # full ROI in case a detected blob's corresponding true blob was 
        # actually just outside of it
        blobs_inner_missed = blobs_inner[blobs_inner[: , 4] == 0]
        found_truth, detected_out = _find_close_blobs(blobs_inner_missed, blobs_truth_roi, region, tol)
        blobs_inner_missed[detected_out, 4] = 1
        blobs_inner = np.concatenate((blobs_inner[blobs_inner[:, 4] == 1], blobs_inner_missed))
        unique_detected_out, indices_detected_out = np.unique(detected_out, return_index=True)
        blobs_truth_extra = blobs_truth_roi[found_truth[indices_detected_out]]
        blobs_truth_extra[:, 5] = 1
        print("detected an outer truth blob:\n{}".format(blobs_inner_missed[blobs_inner_missed[:, 4] == 1]))
        print("all those outer truth blobs:\n{}".format(blobs_truth_extra))
        
        # do the same but for truth blobs missed in the inner ROI
        blobs_truth_inner_missed = blobs_truth_inner[blobs_truth_inner[:, 5] == 0]
        found_truth_out, detected = _find_closest_blobs(blobs_roi, blobs_truth_inner_missed, region, tol)
        blobs_truth_inner_missed[found_truth_out, 5] = 1
        blobs_truth_inner = np.concatenate((blobs_truth_inner[blobs_truth_inner[:, 5] == 1], blobs_truth_inner_missed))
        unique_truth_out, indices_truth_out = np.unique(found_truth_out, return_index=True)
        blobs_roi_extra = blobs_roi[detected[indices_truth_out]]
        blobs_roi_extra[:, 4] = 1
        print("truth blobs detected by an outside blob:\n{}".format(blobs_truth_inner_missed[blobs_truth_inner_missed[:, 5] == 1]))
        print("all those outside detection blobs:\n{}".format(blobs_roi_extra))
        
        # combine inner blobs with detections from outside
        blobs_inner_plus = np.concatenate((blobs_inner, blobs_roi_extra))
        blobs_truth_inner_plus = np.concatenate((blobs_truth_inner, blobs_truth_extra))
        print("blobs_inner_plus:\n{}".format(blobs_inner_plus))
        print("blobs_truth_inner_plus:\n{}".format(blobs_truth_inner_plus))
        
        roi_id, _ = sqlite.insert_roi(output_db.conn, output_db.cur, exp_id, series, offset_inner, size_inner)
        sqlite.insert_blobs(output_db.conn, output_db.cur, roi_id, blobs_inner_plus)
        sqlite.insert_blobs(output_db.conn, output_db.cur, roi_id, blobs_truth_inner_plus)
        
        # saves all blobs from inner portion of ROI only to avoid missing
        # detections because of edge effects
        if blobs_truth_rois is None:
            blobs_truth_rois = blobs_truth_inner_plus
        else:
            blobs_truth_rois = np.concatenate((blobs_truth_inner_plus, blobs_truth_rois))
        if blobs_rois is None:
            blobs_rois = blobs_inner_plus
        else:
            blobs_rois = np.concatenate((blobs_inner_plus, blobs_rois))
    #print("blobs 1:\n{}".format(blobs[blobs[:, 4] == 1]))
    
    true_pos = len(blobs_rois[blobs_rois[:, 4] == 1])
    false_pos = len(blobs_rois[blobs_rois[:, 4] == 0])
    pos = len(blobs_truth_rois)
    false_neg = pos - true_pos
    sens = float(true_pos) / pos
    ppv = float(true_pos) / (true_pos + false_pos)
    print("Automated verification:\ncells = {}\ndetected cells = {}\n"
          "false pos cells = {}\nfalse neg cells = {}\nsensitivity = {}\n"
          "PPV = {}\n"
          .format(pos, true_pos, false_pos, false_neg, sens, ppv))

def _test_blob_duplicates():
    # tests blob duplication removal
    blobs = np.array([[1, 3, 4, 2.2342], [1, 8, 5, 3.13452], [1, 3, 4, 5.1234],
                      [1, 3, 5, 2.2342], [3, 8, 5, 3.13452]])
    print("sample blobs:\n{}".format(blobs))
    end = 3
    blobs_unique = remove_duplicate_blobs(blobs, slice(0, end))
    print("blobs_unique through first {} elements:\n{}".format(end, blobs_unique))
    
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
    _find_closest_blobs(b, a, slice(0, 3), (1, 1, 1))

if __name__ == "__main__":
    print("Detector tests...")
    _test_blob_verification()
