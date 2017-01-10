#!/bin/bash

import javabridge as jb
import bioformats as bf
import numpy as np
from matplotlib import pyplot as plt, cm
import lifio

filename = 'P21_L5_CONT_DENDRITE.czi'
filename = '../../Downloads/Rbp4cre_halfbrain_4-28-16_Subset3.czi'

lifio.start()
metadata = bf.get_omexml_metadata(filename)
names, sizes, resolutions = lifio.parse_xml_metadata(metadata)
print("names: {}\nsizes: {}\nresolutions: {}".format(names, sizes, resolutions))

def show_subplot(subploti):
    ax = fig.add_subplot(2, 3, subploti)
    i = subploti - 1
    print("nt: {}, nz: {}".format(nt, nz))
    #plt.imshow(image5d[0, i * nz//6, :, :, 0], cmap=cm.rainbow)
    plt.imshow(image5d[0, i * nz//6, :nz, :nz, 0], cmap=cm.rainbow)
    
rdr = bf.ImageReader(filename, perform_init=True)
idx = 1 # arbitrary subset for demonstration
size = sizes[idx]
nt, nz = size[:2]
nz = 100 # TESTING: limit z-planes for faster loading
#image5d = np.empty(size, np.uint8)
image5d = np.empty((nt, nz, size[2], size[3], 3), np.uint8)
print(image5d.shape)
for t in range(nt):
    for z in range(nz):
        print("loading planes from [{}, {}]".format(t, z))
        image5d[t, z] = rdr.read(z=z, t=t, series=idx, rescale=False)
fig = plt.figure()
for i in range(6): # TESTING: show every nth plane
	show_subplot(i + 1)
#plt.imshow(image5d[0, 0, :, :, 0], cmap=cm.rainbow)
#ax = fig.add_subplot(222)
#plt.imshow(image5d[nt//2, nz//2, :, :, 0], cmap=cm.rainbow)
plt.show()


'''
image_np = lifio.read_image_series(filename)
plt.imshow(image_np)
plt.show()
'''

lifio.done()