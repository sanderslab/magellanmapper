# Cell detection methods
# Author: David Young, 2017
"""Detects features within a 3D image stack.

Provides options for segmentation and blob detection techniques.

Attributes:
    scaling_factor: The zoom scaling, where 
        factor = 1 / (um/pixel), so 1um/pixel  
        corresponds to factor of 1; eg 0.25um/pixel would require 
        a factor of 1 / 0.25 = 4
"""

from time import time
import math
import numpy as np
from skimage import segmentation
from skimage import measure
from skimage import morphology
from skimage.feature import blob_dog, blob_log, blob_doh

from clrbrain import plot_3d

scaling_factor = 1

def segment_rw(roi):
    """Segments an image, drawing contours around segmented regions.
    
    Args:
        roi: Region of interest to segment.
        vis: Visualization object on which to draw the contour.
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
        vis: Visualization object on which to draw the contour.
    
    Returns:
        blobs_log: Array of detected blobs, each given as 
            (z, row, column, radius).
        cmap: Randomized colormap, where each blob will get a different
            color.
    """
    print("blob detection...")
    # use 3D blob detection from skimage v.0.13pre
    time_start = time()
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
