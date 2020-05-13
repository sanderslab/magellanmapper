# Pixel value display for figure footers
# Author: David Young, 2018, 2020
"""Customize values displayed in Matplotlib figure footers.
"""

import numpy as np


class PixelDisplay(object):
    """Custom image intensity display in :attr:``Axes.format_coord``.

    Attributes:
        imgs (List[:obj:`np.ndarray`]): Sequence of images whose intensity
            values will be displayed.
        ax_imgs (List[:obj:`matplotlib.image.AxesImage`]): Nested sequence of
            Matplotlib images corresponding to ``imgs``.
        downsample (float): Downsampling factor; defaults to 1.
        cmap_labels (:obj:`colormaps.DiscreteColormap`): Labels colormap
            to find the corresponding RGB value; defaults to None, in which
            case the corresponding colormap in ``ax_imgs`` will be used for
            the labels (index 1) will be used instead.
    """

    def __init__(self, imgs, ax_imgs, downsample=1, cmap_labels=None):
        self.imgs = imgs
        self.ax_imgs = ax_imgs
        self.downsample = downsample
        self.cmap_labels = cmap_labels

    def __call__(self, x, y):
        coord = (int(y), int(x))
        # re-upsample coordinates for any downsampling
        output = ["{}={}".format(a, coord[i] * self.downsample)
                  for i, a in enumerate(("x", "y"))]
        rgb = ""
        for i, img in enumerate(self.imgs):
            if x < 0 or y < 0 or x >= img.shape[1] or y >= img.shape[0]:
                # no corresponding px for the image
                z = "n/a"
            else:
                # get the corresponding intensity value, truncating floats
                z = img[coord]
                if i == 1:
                    # for the label image, get its RGB value
                    ax_img = self.ax_imgs[i][0]
                    if self.cmap_labels:
                        label_rgb = self.cmap_labels(
                            self.cmap_labels.convert_img_labels(z))
                    else:
                        label_rgb = ax_img.cmap(ax_img.norm(z))
                    rgb = "RGB for label {}: {}".format(
                        z, tuple(np.multiply(label_rgb[:3], 255).astype(int)))
                if isinstance(z, float): z = "{:.4f}".format(z)
            output.append("z(image{})={}".format(i, z))
        output.append(rgb)
        return ", ".join(output)
