# Stack manipulations
# Author: David Young, 2017
"""Imports and exports stacks in various formats
"""

import os
import glob

import numpy as np
from skimage import transform
from skimage import io
import matplotlib.pyplot as plt
from matplotlib import animation

from clrbrain import cli

def animated_gif(path):
    files = sorted(glob.glob(path))
    num_files = len(files)
    if num_files < 1:
        return None
    parent_path = os.path.dirname(files[0])
    image3d = []
    for f in files:
        print("importing {}".format(f))
        img = io.imread(f)
        img = transform.rescale(img, 0.2)
        image3d.append(img)
    
    # export to animated GIF
    fig = plt.figure()
    animation = animation.ArtistAnimation(fig, image3d, interval=100, repeat_delay=0)
    animation.save(parent_path + "_animation.gif", writer="imagemagick")
    plt.show()
    return image5d

if __name__ == "__main__":
    print("Clrbrain stack manipulations")
    cli.main(True)
    animated_gif(os.path.join(cli.filename, "*"))
