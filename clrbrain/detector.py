# Cell detection methods
# Author: David Young, 2017
"""Detects features within a 3D image stack.

Provides options for segmentation and blob detection techniques.

Attributes:
    scaling_factor: 
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

resolutions = None # (z, y, x) order

def calc_scaling_factor():
    """Calculates the tolerance based on the  
    resolutions, using the first resolution.
    
    Return:
        Array of tolerance values in same shape as resolution.
    """
    if resolutions is None:
        raise AttributeError("Must load resolutions from file or set a resolution")
    factor = np.ceil(np.divide(1.0, resolutions[0])).astype(int)
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
