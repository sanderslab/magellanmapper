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
from skimage.feature import blob_dog, blob_log, blob_doh

from clrbrain import plot_3d

resolutions = None # (z, y, x) order since given from microscope

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
    
    # ROI is in (z, y, x) order, so need to transpose or swap x,z axes
    roi = np.transpose(roi)
    
    # random-walker segmentation
    markers = np.zeros(roi.shape, dtype=np.uint8)
    markers[roi > 0.4] = 1
    markers[roi < 0.33] = 2
    walker = segmentation.random_walker(roi, markers, beta=1000., mode="bf")
    
    # label neighboring pixels to segmented regions
    walker = morphology.remove_small_objects(walker == 1, 200)
    labels = measure.label(walker, background=0)
    
    return labels

def segment_blob(roi):
    """Detects objects using 3D blob detection technique.
    
    Args:
        roi: Region of interest to segment.
    
    Returns:
        Array of detected blobs, each given as 
            (z, row, column, radius).
    """
    # use 3D blob detection from skimage v.0.13pre
    time_start = time()
    # scaling as a factor in pixel/um, where scaling of 1um/pixel  
    # corresponds to factor of 1, and 0.25um/pixel corresponds to
    # 1 / 0.25 = 4 pixels/um; currently simplified to be based on 
    # x scaling alone
    scaling_factor = calc_scaling_factor()[2]
    blobs_log = blob_log(roi, min_sigma=3 * scaling_factor, 
                         max_sigma=30 * scaling_factor, num_sigma=10, 
                         threshold=0.1)
    '''
    blobs_log = blob_dog(roi, min_sigma=0.5*scaling_factor, 
                         max_sigma=30*scaling_factor, 
                         threshold=0.2)
    '''
    print("time for 3D blob detection: %f" %(time() - time_start))
    if blobs_log.size < 1:
        print("no blobs detected")
        return None
    blobs_log[:, 3] = blobs_log[:, 3] * math.sqrt(3)
    print(blobs_log)
    print("found {} blobs".format(blobs_log.shape[0]))
    confirmed = np.ones((blobs_log.shape[0], 1)) * -1
    blobs_log = np.concatenate((blobs_log, confirmed), axis=1)
    return blobs_log

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
    print("removed {} duplicate blobs".format(blobs.shape[0] - unique_indices.size))
    return blobs[unique_indices]

def remove_close_blobs(blobs, blobs_master, region, tol):
    """Removes blobs that are close to one another.
    
    Params:
        blobs: The blobs to add, given as 2D array of [n, [z, row, column, 
            radius]].
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
    # creates a separate array for each blob in blobs_master to allow
    # comparison for each of its blobs with each blob to add
    blobs_diffs = np.abs(blobs_master[:, region][:, None] - blobs[:, region])
    close_master, close = np.nonzero((blobs_diffs <= tol).all(2))
    pruned = np.delete(blobs, close, axis=0)
    print("removed {} close blobs".format(blobs.shape[0] - pruned.shape[0]))
    return pruned

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
        return
    blobs_all = None
    for blob in blobs:
        #print("blob: {}".format(blob))
        if blobs_all is None:
            blobs_all = np.array([blob])
        else:
            blobs_to_add = remove_close_blobs(np.array([blob]), blobs_all, region, tol)
            blobs_all = np.concatenate((blobs_all, blobs_to_add))
    return blobs_all

if __name__ == "__main__":
    print("Detector tests...")
    
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
