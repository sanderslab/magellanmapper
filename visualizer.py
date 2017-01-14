#!/bin/bash
# Author: David Young, 2017

import sys
import javabridge as jb
import bioformats as bf
import numpy as np
import math
from time import time
from mayavi import mlab
from matplotlib import pyplot as plt, cm
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from scipy import stats
from skimage import restoration
from skimage import exposure
from skimage import segmentation
from skimage import measure
from skimage import morphology
from scipy import ndimage

from traits.api import HasTraits, Range, Instance, \
                    on_trait_change, Button, Int
from traitsui.api import View, Item, HGroup, VGroup, Handler, RangeEditor
from tvtk.pyface.scene_editor import SceneEditor
from mayavi.tools.mlab_scene_model import \
                    MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene


filename = "../../Downloads/P21_L5_CONT_DENDRITE.czi"
filename = "../../Downloads/Rbp4cre_halfbrain_4-28-16_Subset3.czi"
#filename = "../../Downloads/Rbp4cre_4-28-16_Subset3_2.sis"
#filename = "/Volumes/Siavash/CLARITY/P3Ntsr1cre-tdTomato_11-10-16/Ntsr1cre-tdTomato.czi"
subset = 0 # arbitrary series for demonstration
channel = 0 # channel of interest
roi_size = [50, 50, 50]
offset = None

ARG_OFFSET = "offset"
ARG_CHANNEL = "channel"
ARG_SUBSET = "subset"
ARG_SIDES = "sides"

for arg in sys.argv:
    arg_split = arg.split("=")
    if len(arg_split) == 1:
        print("Skipped argument: {}".format(arg_split[0]))
    elif len(arg_split) >= 2:
        if arg_split[0] == ARG_OFFSET:
            offset_split = arg_split[1].split(",")
            if len(offset_split) >= 3:
                offset = tuple(int(i) for i in offset_split)
                print("Set offset: {}".format(offset))
            else:
                print("Offset ({}) should be given as 3 values (x, y, z)"
                      .format(arg_split[1]))
        elif arg_split[0] == ARG_CHANNEL:
            channel = int(arg_split[1])
        elif arg_split[0] == ARG_SUBSET:
            subset = int(arg_split[1])
        elif arg_split[0] == ARG_SIDES:
            sides_split = arg_split[1].split(",")
            if len(sides_split) >= 3:
                roi_size = tuple(int(i) for i in sides_split)
                print("Set roi_size: {}".format(roi_size))
            else:
                print("Sides ({}) should be given as 3 values (x, y, z)"
                      .format(arg_split[1]))

def start_jvm(heap_size="8G"):
    """Starts the JVM for Python-Bioformats.
    
    Args:
        heap_size: JVM heap size, defaulting to 8G.
    """
    jb.start_vm(class_path=bf.JARS, max_heap_size=heap_size)

def parse_ome(filename):
    """Parses metadata for image name and size information.
    
    Args:
        filename: Image file, assumed to have metadata in OME XML format.
    
    Returns:
        names: array of names of subsets within the file.
        sizes: array of tuples with dimensions for each subset. Dimensions
            will be given as (time, z, y, x, channels).
    """
    time_start = time()
    metadata = bf.get_omexml_metadata(filename)
    ome = bf.OMEXML(metadata)
    count = ome.image_count
    names, sizes = [], []
    for i in range(count):
        image = ome.image(i)
        names.append(image.Name)
        pixel = image.Pixels
        size = (pixel.SizeT, pixel.SizeZ, pixel.SizeY, pixel.SizeX, pixel.SizeC)
        sizes.append(size)
    print("names: {}\nsizes: {}".format(names, sizes))
    print('time for parsing OME XML: %f' %(time() - time_start))
    return names, sizes

def find_sizes(filename):
    """Finds image size information using the ImageReader.
    
    Args:
        filename: Image file, assumed to have metadata in OME XML format.
    
    Returns:
        sizes: array of tuples with dimensions for each subset. Dimensions
            will be given as (time, z, y, x, channels).
    """
    time_start = time()
    sizes = []
    with bf.ImageReader(filename) as rdr:
        format_reader = rdr.rdr
        count = format_reader.getSeriesCount()
        for i in range(count):
            size = ( format_reader.getSizeT(), format_reader.getSizeZ(), 
                     format_reader.getSizeY(), format_reader.getSizeX(), 
                     format_reader.getSizeC() )
            print(size)
            sizes.append(size)
    print('time for finding sizes: %f' %(time() - time_start))
    return sizes

def read_file(filename, save=True, load=True, z_max=-1, offset=None):
    """Reads in an imaging file.
    
    Can load the file from a saved Numpy array and also for only a subset
    of z-planes if asked.
    
    Args:
        filename: Image file, assumed to have metadata in OME XML format.
        save: True to save the resulting Numpy array (default).
        load: If True, attempts to load a Numpy array from the same 
            location and name except for ".npz" appended to the end 
            (default). The array can be accessed as "output['image5d']".
        z_max: Number of z-planes to load, or -1 if all should be loaded
            (default).
        offset: Tuple of offset given as (x, y, z) from which to start
            loading z-plane (x, y ignored for now). Defaults to 
            (0, 0, 0).
    
    Returns:
        image5d: array of image data.
        size: tuple of dimensions given as (time, z, y, x, channels).
    """
    filename_npz = filename + ".npz"
    if load:
        try:
            time_start = time()
            output = np.load(filename_npz)
            print('file opening time: %f' %(time() - time_start))
            image5d = output["image5d"]
            return image5d
        except IOError as err:
            print("Unable to load {}, will attempt to reload {}".format(filename_npz, filename))
    sizes = find_sizes(filename)
    rdr = bf.ImageReader(filename, perform_init=True)
    size = sizes[subset]
    nt, nz = size[:2]
    if z_max != -1:
        nz = z_max
    if offset == None:
    	offset = (0, 0, 0) # (x, y, z)
    channels = 3 if size[4] <= 3 else size[4]
    image5d = np.empty((nt, nz, size[2], size[3], channels), np.uint8)
    time_start = time()
    for t in range(nt):
        for z in range(nz):
            print("loading planes from [{}, {}]".format(t, z))
            image5d[t, z] = rdr.read(z=(z + offset[2]), t=t, series=subset, rescale=False)
    print('file import time: %f' %(time() - time_start))
    outfile = open(filename_npz, "wb")
    if save:
        time_start = time()
        # could use compression (savez_compressed), but much slower
        np.savez(outfile, image5d=image5d)
        outfile.close()
        print('file save time: %f' %(time() - time_start))
    return image5d

def denoise(roi):
    """Denoises an image.
    
    Args:
        roi: Region of interest.
    
    Returns:
        Denoised region of interest.
    """
    # saturating extreme values to maximize contrast
    vmin, vmax = stats.scoreatpercentile(roi, (0.5, 99.5))
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
    t3 = time()
    denoised = restoration.denoise_nl_means(denoised,
                        patch_size=5, patch_distance=7,
                        h=0.1, multichannel=False)
    t4 = time()
    print('time for non-local means denoising: %f' %(t4 - t3))
    '''
    
    # total variation denoising
    time_start = time()
    denoised = restoration.denoise_tv_chambolle(denoised, weight=0.2)
    print('time for total variation: %f' %(time() - time_start))
    
    return denoised

def segment_roi(roi, vis):
    """Segments an image, drawing contours around segmented regions.
    
    Args:
        roi: Region of interest to segment.
        vis: Visualization object on which to draw the contour.
    """
    print("segmenting...")
    # random-walker segmentation
    markers = np.zeros(roi.shape, dtype=np.uint8)
    markers[roi > 0.4] = 1
    markers[roi < 0.33] = 2
    walker = segmentation.random_walker(roi, markers, beta=1000., mode='cg_mg')
    walker = morphology.remove_small_objects(walker == 1, 200)
    labels = measure.label(walker, background=0)
    surf2 = vis.scene.mlab.contour3d(labels)

def show_roi(image5d, vis, offset=(0, 0, 0)):
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
    roi = image5d[0, cube_slices[2], cube_slices[1], cube_slices[0], channel]
    
    # thinner z
    #roi = image5d[0, slice(offset[2], offset[2] + 20), cube_slices[1], cube_slices[0], channel]
    
    # thin z, entire x-y
    #roi = image5d[0, slice(offset[2], offset[2] + 5), :, :, channel]
    #roi = image5d[0, 0:5, :, :, channel]
    
    # thin segment along x-axis to verify x-y orientation
    #roi = image5d[0, 0, cube_slices[1], :, channel]
    
    # thin segment along y-axis to verify x-y orientation
    #roi = image5d[0, 0, :, cube_slices[0], channel]
    
    # entire stack
    #roi = image5d[0, :, :, :, 1]
    
    roi = denoise(roi)
    
    # Plot in Mayavi
    #mlab.figure()
    vis.scene.mlab.clf()
    
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
    
    #mlab.show()
    return roi

def show_subplot(gs, subploti, offset):
    #ax = plt.subplot2grid((2, 7), (1, subploti))
    ax = plt.subplot(gs[1, subploti])
    #ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    size = image5d.shape
    z = offset[2]
    if z < 0 or z >= size[1]:
        print("skipping z-plane {}".format(z))
        plt.imshow(np.zeros(roi_size[0:2]))
    else:
        #i = subploti - 1
        #plt.imshow(image5d[0, i * nz//6, :, :, 0], cmap=cm.rainbow)
        plt.imshow(image5d[0, offset[2], 
                           slice(offset[1], offset[1] + roi_size[1]), 
                           slice(offset[0], offset[0] + roi_size[0]), channel], 
                   cmap=cm.rainbow)
   
def plot_2d_stack(offset):
    fig = plt.figure()
    z_planes = roi_size[2]
    if z_planes % 2 == 0:
        z_planes = z_planes + 1
    gs = gridspec.GridSpec(2, z_planes, wspace=0.0, hspace=0.0)
    half_z_planes = z_planes // 2
    ax = plt.subplot(gs[0, :half_z_planes])
    #ax = plt.subplot2grid((2, 7), (0, 0), colspan=4)
    #ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(image5d[0, offset[2], :, :, channel], 
               cmap=cm.rainbow)
    ax.add_patch(patches.Rectangle(offset[0:2], roi_size[0], roi_size[1], 
                                   fill=False, edgecolor="black"))
    z = offset[2]
    for i in range(z_planes):
    	show_subplot(gs, i, (offset[0], offset[1], z - half_z_planes + i))
    img3d = mlab.screenshot()
    ax = plt.subplot(gs[half_z_planes:z_planes])
    ax.imshow(img3d)
    _hide_axes(ax)
    gs.tight_layout(fig, pad=0)
    #plt.tight_layout()
    plt.ion()
    plt.show()

def _hide_axes(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

class VisHandler(Handler):
    """Simple handler for Visualization object events.
    
    Closes the JVM when the window is closed.
    """
    def closed(self, info, is_ok):
        jb.kill_vm()

class Visualization(HasTraits):
    """GUI for choosing a region of interest and segmenting it.
    
    TraitUI-based graphical interface for selecting dimensions of an
    image to view and segment.
    
    Attributes:
        x_low, x_high, ...: Low and high values for each offset.
        x_offset: Integer trait for x-offset.
        y_offset: Integer trait for y-offset.
        z_offset: Integer trait for z-offset.
        scene: The main scene
        btn_redraw_trait: Button editor for drawing the reiong of 
            interest.
        btn_segment_trait: Button editor for segmenting the ROI.
        roi: The ROI.
    """
    x_low = 0
    x_high = 100
    y_low = 0
    y_high = 100
    z_low = 0
    z_high = 100
    x_offset = Int
    y_offset = Int
    z_offset = Int
    scene = Instance(MlabSceneModel, ())
    btn_redraw_trait = Button("Redraw")
    btn_segment_trait = Button("Segment")
    btn_2d_trait = Button("2D Plots")
    roi = None
    
    def __init__(self):
        # Do not forget to call the parent's __init__
        HasTraits.__init__(self)
        # dimension max values in pixels
        size = image5d.shape
        self.z_high = size[1]
        self.y_high = size[2]
        self.x_high = size[3]
        curr_offset = offset
        # apply user-defined offsets
        if curr_offset is not None:
            self.x_offset = curr_offset[0]
            self.y_offset = curr_offset[1]
            self.z_offset = curr_offset[2]
        else:
            print("No offset, using standard one")
            curr_offset = self._curr_offset()
            #self.roi = show_roi(image5d, self, cube_len=cube_len)
        self.roi = show_roi(image5d, self, offset=curr_offset)
        #plot_2d_stack(curr_offset)
    
    @on_trait_change('x_offset,y_offset,z_offset')
    def update_plot(self):
        print("x: {}, y: {}, z: {}".format(self.x_offset, self.y_offset, self.z_offset))
    
    def _btn_redraw_trait_fired(self):
        # ensure that cube dimensions don't exceed array
        size = image5d.shape
        if self.x_offset + roi_size[0] > size[3]:
            self.x_offset = size[3] - roi_size[0]
        if self.y_offset + roi_size[1] > size[2]:
            self.y_offset = size[2] - roi_size[1]
        if self.z_offset + roi_size[2] > size[1]:
            self.z_offset = size[1] - roi_size[2]
        
        # show updated region of interest
        offset = self._curr_offset()
        print(offset)
        self.roi = show_roi(image5d, self, offset=offset)
    
    def _btn_segment_trait_fired(self):
        #print(Visualization.roi)
        segment_roi(self.roi, self)
    
    def _btn_2d_trait_fired(self):
        curr_offset = self._curr_offset()
        plot_2d_stack(curr_offset)
    
    def _curr_offset(self):
        return (self.x_offset, self.y_offset, self.z_offset)
    
    # the layout of the dialog created
    view = View(
        Item(
            'scene', 
            editor=SceneEditor(scene_class=MayaviScene),
            height=500, width=500, show_label=False
        ),
        VGroup(
            Item(
                'x_offset',
                editor=RangeEditor(
                    low_name="x_low",
                    high_name="x_high",
                    mode="slider")
            ),
            Item(
                'y_offset',
                editor=RangeEditor(
                    low_name="y_low",
                    high_name="y_high",
                    mode="slider")
            ),
            Item(
                'z_offset',
                editor=RangeEditor(
                    low_name="z_low",
                    high_name="z_high",
                    mode="slider")
            )
        ),
        HGroup(
            Item("btn_redraw_trait", show_label=False), 
            Item("btn_segment_trait", show_label=False), 
            Item("btn_2d_trait", show_label=False)
        ),
        handler=VisHandler(),
        title = "clrbrain",
        resizable = True
    )

# loads the image and GUI
start_jvm()
#names, sizes = parse_ome(filename)
#sizes = find_sizes(filename)
image5d = read_file(filename) #, z_max=cube_len)
visualization = Visualization()
visualization.configure_traits()
