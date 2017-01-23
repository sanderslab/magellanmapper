# 3D plots from stacks of imaging data
# Author: David Young, 2017

from time import time
import numpy as np
from scipy import stats
from skimage import restoration
from mayavi import mlab

MLAB_3D_TYPES = ("surface", "point")
mlab_3d = MLAB_3D_TYPES[1]

def denoise(roi):
    """Denoises an image.
    
    Args:
        roi: Region of interest.
    
    Returns:
        Denoised region of interest.
    """
    # saturating extreme values to maximize contrast
    vmin, vmax = stats.scoreatpercentile(roi, (20.0, 99.5))
    denoised = np.clip(roi, vmin, vmax)
    denoised = (denoised - vmin) / (vmax - vmin)
    
    '''
    # denoise_bilateral apparently only works on 2D images
    t1 = time()
    bilateral = restoration.denoise_bilateral(denoised)
    t2 = time()
    print('time for bilateral filter: %f' %(t2 - t1))
    hi_dat = exposure.histogram(denoised)
    hi_bilateral = exposure.histogram(bilateral)
    plt.plot(hi_dat[1], hi_dat[0], label='data')
    plt.plot(hi_bilateral[1], hi_bilateral[0],
             label='bilateral')
    plt.xlim(0, 0.5)
    plt.legend()
    plt.title('Histogram of voxel values')
    
    sample = bilateral > 0.2
    sample = ndimage.binary_fill_holes(sample)
    open_object = morphology.opening(sample, morphology.ball(3))
    close_object = morphology.closing(open_object, morphology.ball(3))
    bbox = ndimage.find_objects(close_object)
    mask = close_object[bbox[0]]
    '''
    
    '''
    # non-local means denoising, which works but is slower
    # and doesn't seem to add much
    time_start = time()
    denoised = restoration.denoise_nl_means(denoised,
                        patch_size=5, patch_distance=7,
                        h=0.12, multichannel=False)
    print('time for non-local means denoising: %f' %(time() - time_start))
    '''
    
    # total variation denoising
    time_start = time()
    denoised = restoration.denoise_tv_chambolle(denoised, weight=0.2)
    print('time for total variation: %f' %(time() - time_start))
    
    return denoised

def plot_3d_surface(roi, vis):
    # Plot in Mayavi
    #mlab.figure()
    vis.scene.mlab.clf()
    
    # ROI is in (z, y, x) order, so need to transpose or swap x,z axes
    #roi = np.flipud(roi)
    roi = np.transpose(roi)
    #roi = np.swapaxes(roi, 0, 2)
    #roi = np.fliplr(roi)
    
    # prepare the data source
    #np.transpose(roi, (0, 1, 3, 2, 4))
    scalars = vis.scene.mlab.pipeline.scalar_field(roi)
    
    # create the surface
    contour = vis.scene.mlab.pipeline.contour(scalars)
    # TESTING: use when excluding further processing
    #surf = vis.scene.mlab.pipeline.surface(contour)
    
    # removes many more extraneous points
    smooth = vis.scene.mlab.pipeline.user_defined(contour, filter='SmoothPolyDataFilter')
    smooth.filter.number_of_iterations = 400
    smooth.filter.relaxation_factor = 0.015
    # holes within cells?
    curv = vis.scene.mlab.pipeline.user_defined(smooth, filter='Curvatures')
    surf = vis.scene.mlab.pipeline.surface(curv)
    # colorizes
    module_manager = curv.children[0]
    module_manager.scalar_lut_manager.data_range = np.array([-0.6,  0.5])
    module_manager.scalar_lut_manager.lut_mode = 'RdBu'
    
    # based on Surface with contours enabled
    #contour = vis.scene.mlab.pipeline.contour_surface(scalars)
    
    # uses unique IsoSurface module but appears to have 
    # similar output to contour_surface
    #contour = vis.scene.mlab.pipeline.iso_surface(scalars)
    
def plot_3d_points(roi, vis):
    print("plotting as 3D points")
    """
    scalars = vis.scene.mlab.pipeline.scalar_scatter(roi)
    vis.scene.mlab.points3d(scalars)
    """
    vis.scene.mlab.clf()
    shape = roi.shape
    z = np.ones((shape[0], shape[1] * shape[2]))
    for i in range(shape[0]):
        z[i] = z[i] * i
    y = np.ones((shape[0] * shape[1], shape[2]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            y[i * shape[1] + j] = y[i * shape[1] + j] * j
    x = np.ones((shape[0] * shape[1], shape[2]))
    for i in range(shape[0] * shape[1]):
        x[i] = np.arange(shape[2])
        #x[i] = np.multiply(x[i], np.arrange(shape[2]))
    x = np.reshape(x, roi.size)
    y = np.reshape(y, roi.size)
    z = np.reshape(z, roi.size)
    roi_1d = np.reshape(roi, roi.size)
    intensity_threshold = 0.35
    remove = np.where(roi_1d < intensity_threshold)
    x = np.delete(x, remove)
    y = np.delete(y, remove)
    z = np.delete(z, remove)
    roi_1d = np.delete(roi_1d, remove)
    print(roi_1d.size)
    vis.scene.mlab.points3d(x, y, z, roi_1d, 
                            mode="sphere", colormap="inferno", scale_mode="none",
                            line_width=1.0, vmax=1.0, 
                            vmin=(intensity_threshold * 0.5), transparent=True)
    """
    roi_1d[roi_1d < 0.2] = 0
    vis.scene.mlab.points3d(x, y, z, roi_1d, 
                            mode="cube", colormap="Blues", scale_mode="none",
                            transparent=True)
    for i in range(roi_1d.size):
        print("x: {}, y: {}, z: {}, s: {}".format(x[i], y[i], z[i], roi_1d[i]))
    """

def show_roi(image5d, channel, vis, roi_size, offset=(0, 0, 0)):
    """Finds and shows the region of interest.
    
    This region will be denoised and displayed in Mayavi.
    
    Args:
        image5d: Image array.
        vis: Visualization object on which to draw the contour. Any 
            current image will be cleared first.
        cube_len: Length of each side of the region of interest as a 
            cube. Defaults to 100.
        offset: Tuple of offset given as (x, y, z) for the region 
            of interest. Defaults to (0, 0, 0).
    
    Returns:
        The region of interest, including denoising.
    """
    cube_slices = []
    for i in range(len(offset)):
        cube_slices.append(slice(offset[i], offset[i] + roi_size[i]))
    print(cube_slices)
    
    # cube with corner at offset, side of cube_len
    if image5d.ndim >= 5:
        roi = image5d[0, cube_slices[2], cube_slices[1], cube_slices[0], channel]
    else:
        roi = image5d[0, cube_slices[2], cube_slices[1], cube_slices[0]]
    
    roi = denoise(roi)
    if mlab_3d == MLAB_3D_TYPES[0]:
        plot_3d_surface(roi, vis)
    else:
        plot_3d_points(roi, vis)
    
    return roi

def set_mlab_3d(val):
    global mlab_3d
    mlab_3d = val

def get_mlab_3d():
    return mlab_3d