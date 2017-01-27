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

def segment_rw(roi, vis):
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
    
    '''
    # Drawing options:
    # 1) draw iso-surface around segmented regions
    scalars = vis.scene.mlab.pipeline.scalar_field(labels)
    surf2 = vis.scene.mlab.pipeline.iso_surface(scalars)
    '''
    # 2) draw a contour or points directly from labels
    vis.scene.mlab.contour3d(labels)
    #surf2 = vis.scene.mlab.points3d(labels)
    return labels

def segment_blob(roi, vis):
    """Detects objects using 3D blob detection technique.
    
    Args:
        roi: Region of interest to segment.
        vis: Visualization object on which to draw the contour.
    """
    print("blob detection...")
    # use 3D blob detection from skimage v.0.13pre
    time_start = time()
    blobs_log = blob_log(roi, min_sigma=4 * scaling_factor, 
                         max_sigma=30 * scaling_factor, num_sigma=10, 
                         threshold=0.1)
    print("time for 3D blob detection: %f" %(time() - time_start))
    blobs_log[:, 3] = blobs_log[:, 3] * math.sqrt(3)
    print(blobs_log)
    scale = 2 * max(blobs_log[:, 3]) * scaling_factor
    print("blob point scaling: {}".format(scale))
    vis.scene.mlab.points3d(blobs_log[:, 2], blobs_log[:, 1], 
                            blobs_log[:, 0], blobs_log[:, 3],
                            scale_mode="none", scale_factor=scale, 
                            opacity=0.5, color=(0, 1, 0))
    print("found {} blobs".format(blobs_log.size))
    return blobs_log

def segment_roi(roi, vis):
    """Segments a region of interest, using the rendering technique,
    specified in the mlab_3d attribute.
    
    Args:
        roi: Region of interest to segment.
        vis: Visualization object on which to draw the contour.
    """
    mlab_3d = plot_3d.get_mlab_3d()
    if mlab_3d == plot_3d.MLAB_3D_TYPES[0]:
        segment_rw(roi, vis)
        return None
    else:
        return segment_blob(roi, vis)

def set_scaling_factor(val):
    """Sets the scaling factor to adjust for length per pixel.
    
    Args:
        val: Scaling factor.
    """
    global scaling_factor
    scaling_factor = val
