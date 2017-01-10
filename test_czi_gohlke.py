#matplotlib inline
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, './')

from czifile import CziFile

filename = 'P21_L5_CONT_DENDRITE.czi'
with CziFile(filename) as czi:
    image_arrays = czi.asarray(resize=False)

image_arrays.shape

image = image_arrays[0,1,0,0].T[0]
image.shape
plt.imshow(image)
plt.plot()
plt.show()

# TESTING:
exit()

# This indexing gives us full sized images in region 1
# Some playing was required to extract the correct data.
images = [image_arrays[0,0,0,index].T[0] for index in range(7)]
len(images)

# To fit on the screen in a nice way, we can arange the 
# z-stack in a grid of 4x3 on a large figure.
N_rows = 2
N_cols = 3
fig, ax_grid = plt.subplots(N_rows, N_cols, figsize=(N_cols*10,N_rows*10))

for row in range(N_rows):
    for col in range(N_cols): 
        image = images.pop()
        ax_grid[row][col].imshow(image)

## Count the number of cells
testImage = images[0]
plt.imshow(testImage)
plt.show()
