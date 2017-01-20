# Cell detection methods
# Author: David Young, 2017

import math
import numpy as np
from mayavi import mlab
from skimage import segmentation
from skimage import measure
from skimage import morphology
from skimage.feature import blob_dog, blob_log, blob_doh

import plot_3d

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
    walker = segmentation.random_walker(roi, markers, beta=1000., mode='cg_mg')
    
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
    surf2 = vis.scene.mlab.contour3d(labels)
    #surf2 = vis.scene.mlab.points3d(labels)

def segment_blob(roi, vis):
    print("blob detection based segmentation...")
    # use 3D blob detection from skimage v.0.13pre
    blobs_log = blob_log(roi, min_sigma=5, max_sigma=30, num_sigma=10, threshold=0.1)
    blobs_log[:, 3] = blobs_log[:, 3] * math.sqrt(3)
    print(blobs_log)
    vis.scene.mlab.points3d(blobs_log[:, 2], blobs_log[:, 1], 
                            blobs_log[:, 0], blobs_log[:, 3],
                            scale_mode="none", scale_factor=20, 
                            opacity=0.5, color=(1, 0, 0))

def segment_roi(roi, vis):
    mlab_3d = plot_3d.get_mlab_3d()
    if mlab_3d == plot_3d.MLAB_3D_TYPES[0]:
        segment_rw(roi, vis)
    else:
        segment_blob(roi, vis)
    