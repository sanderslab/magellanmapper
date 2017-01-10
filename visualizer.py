#!/bin/bash

import javabridge as jb
import bioformats as bf
import numpy as np
import math
from time import time
from mayavi import mlab
from matplotlib import pyplot as plt, cm
from scipy import stats
from skimage import restoration
from skimage import exposure
from skimage import segmentation
from skimage import measure
from skimage import morphology
from scipy import ndimage

from traits.api import HasTraits, Range, Instance, \
                    on_trait_change, Button
from traitsui.api import View, Item, HGroup, VGroup, Handler
from tvtk.pyface.scene_editor import SceneEditor
from mayavi.tools.mlab_scene_model import \
                    MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene


filename = 'P21_L5_CONT_DENDRITE.czi'
filename = '../../Downloads/Rbp4cre_halfbrain_4-28-16_Subset3.czi'
subset = 1 # arbitrary series for demonstration
cube_len = 100
def start_jvm(heap_size="8G"):
    jb.start_vm(class_path=bf.JARS, max_heap_size=heap_size)

def parse_ome(filename):
    metadata = bf.get_omexml_metadata(filename)
    ome = bf.OMEXML(metadata)
    count = ome.image_count
    names, sizes = [], []
    for i in range(count):
        image = ome.image(i)
        names.append(image.Name)
        pixel = image.Pixels
        size = ( pixel.SizeT, pixel.SizeZ, pixel.SizeX, pixel.SizeY, pixel.SizeC )
        sizes.append(size)
    print("names: {}\nsizes: {}".format(names, sizes))
    return names, sizes

def read_file(filename, save=True, load=True, z_max=-1, offset=None):
    filename_npz = filename + ".npz"
    if load:
        try:
            time_start = time()
            output = np.load(filename_npz)
            print('file load time: %f' %(time() - time_start))
            return output["image5d"]
        except IOError as err:
            print("Unable to load {}, will attempt to reload {}".format(filename_npz, filename))
    rdr = bf.ImageReader(filename, perform_init=True)
    size = sizes[subset]
    nt, nz = size[:2]
    if z_max != -1:
        nz = z_max
    if offset == None:
    	offset = (0, 0, 0) # (z, x, y)
    image5d = np.empty((nt, nz, size[2], size[3], 3), np.uint8)
    #print(image5d.shape)
    time_start = time()
    for t in range(nt):
        for z in range(nz):
            print("loading planes from [{}, {}]".format(t, z))
            image5d[t, z] = rdr.read(z=(z + offset[0]), t=t, series=idx, rescale=False)
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
    t5 = time()
    denoised = restoration.denoise_tv_chambolle(denoised, weight=0.2)
    t6 = time()
    print('time for total variation: %f' %(t6 - t5))
    
    return denoised

def segment_roi(roi, vis):
    print("segmenting...")
    # random-walker segmentation
    markers = np.zeros(roi.shape, dtype=np.uint8)
    markers[roi > 0.4] = 1
    markers[roi < 0.33] = 2
    walker = segmentation.random_walker(roi, markers, beta=1000., mode='cg_mg')
    walker = morphology.remove_small_objects(walker == 1, 200)
    labels = measure.label(walker, background=0)
    surf2 = vis.scene.mlab.contour3d(labels)

def show_roi(image5d, vis, cube_len=100, offset=(0, 0, 0)):
    #offset = (10, 50, 200)
    cube_slices = []
    for i in range(len(offset)):
        cube_slices.append(slice(offset[i], offset[i] + cube_len))
    print(cube_slices)
    roi = image5d[0, cube_slices[0], cube_slices[1], cube_slices[2], 0]
    #roi = load_roi(image5d, cube_len, offset)
    roi = denoise(roi)
    
    # Plot in Mayavi
    #mlab.figure()
    vis.scene.mlab.clf()
    
    scalars = vis.scene.mlab.pipeline.scalar_field(roi)
    # appears to add some transparency to the cube
    contour = vis.scene.mlab.pipeline.contour(scalars)
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
    #mlab.show()
    return roi

class VisHandler(Handler):
    def closed(self, info, is_ok):
        jb.kill_vm()

class Visualization(HasTraits):
    x_offset = Range(0, 100,  0)
    y_offset = Range(0, 100, 0)
    z_offset = Range(0, 100, 0)
    scene = Instance(MlabSceneModel, ())
    btn_redraw_trait = Button("Redraw")
    btn_segment_trait = Button("Segment")
    roi = None
    
    def __init__(self):
        # Do not forget to call the parent's __init__
        HasTraits.__init__(self)
        self.roi = show_roi(image5d, self, cube_len=cube_len)
        #segment_roi(Visualization.roi, self)
    
    @on_trait_change('x_offset,y_offset,z_offset')
    def update_plot(self):
        print("x: {}, y: {}, z: {}".format(self.x_offset, self.y_offset, self.z_offset))
    
    def _btn_redraw_trait_fired(self):
        size = sizes[subset]
        offset=(math.floor(float(self.z_offset) / 100 * size[1]), # z
                math.floor(float(self.x_offset) / 100 * size[2]), # x
                math.floor(float(self.y_offset) / 100 * size[3])) # y
        print(offset)
        self.roi = show_roi(image5d, self, cube_len=cube_len, offset=offset)
    
    def _btn_segment_trait_fired(self):
        #print(Visualization.roi)
        segment_roi(self.roi, self)

    # the layout of the dialog created
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                    height=250, width=300, show_label=False),
                VGroup('x_offset', 'y_offset', 'z_offset'),
                HGroup(Item("btn_redraw_trait", show_label=False), 
                       Item("btn_segment_trait", show_label=False)),
                handler=VisHandler()
               )

start_jvm()
names, sizes = parse_ome(filename)
image5d = read_file(filename) #, z_max=cube_len)
visualization = Visualization()
visualization.configure_traits()
