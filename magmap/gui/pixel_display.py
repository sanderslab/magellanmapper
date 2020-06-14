# Pixel value display for figure footers
# Author: David Young, 2018, 2020
"""Customize values displayed in Matplotlib figure footers.
"""

import numpy as np

from magmap.io import libmag


class PixelDisplay(object):
    """Custom image intensity display in :attr:``Axes.format_coord``.

    Attributes:
        imgs (List[:obj:`np.ndarray`]): Sequence of images whose intensity
            values will be displayed.
        ax_imgs (List[:obj:`matplotlib.image.AxesImage`]): Nested sequence of
            Matplotlib images corresponding to ``imgs``.
        downsample (float): Downsampling factor; defaults to 1.
        offset (List[int], List[List[int]]): Coordinate offset given as
            a sequence of ``(y, x)`` or a nested sequence of offsets
            corresponding to each image in ``imgs``; defaults to None.
        cmap_labels (:obj:`colormaps.DiscreteColormap`): Labels colormap
            to find the corresponding RGB value; defaults to None, in which
            case the corresponding colormap in ``ax_imgs`` will be used for
            the labels (index 1) will be used instead.
    """

    def __init__(self, imgs, ax_imgs, downsample=1, offset=None,
                 cmap_labels=None):
        self.imgs = imgs
        self.ax_imgs = ax_imgs
        self.downsample = downsample
        self.offset = offset
        self.cmap_labels = cmap_labels

    def get_msg(self, event):
        """Get the pixel display message from a Matplotlib event.

        Args:
            event (:obj:`matplotlib.backend_bases.Event`): Matplotlib event.

        Returns:
            str: The message based on the data coordinates within the first
            axes in :attr:`ax_imgs`. None if `event` is not within these axes.

        """
        if event.inaxes != self.ax_imgs[0][0].axes:
            return None
        return self.__call__(event.xdata, event.ydata)

    def __call__(self, x, y):
        """Get the pixel display message.

        Args:
            x (int): x-data coordinate.
            y (int): y-data coordinate.

        Returns:
            str: Message showing ``x,y`` coordinates, intensity values,
            and corresponding RGB label for each overlaid image at the
            given location.

        """
        coord = (int(y), int(x))
        rgb = None
        output = []
        main_img_shape = self.imgs[0].shape[:2]
        for i, img in enumerate(self.imgs):
            # scale coordinates from axes space, based on main image, to
            # given image's space to get pixel from given image
            scale = np.divide(img.shape[:2], main_img_shape)
            coord_img = tuple(np.multiply(coord, scale).astype(int))
            if any(np.less(coord_img, 0)) or any(
                    np.greater_equal(coord_img, img.shape[:len(coord_img)])):
                # no corresponding px for the image
                px = "n/a"
            else:
                # get the corresponding intensity value, truncating floats
                px = img[coord_img]
                if i == 1:
                    # for the label image, get its RGB value
                    ax_img = self.ax_imgs[i][0]
                    if self.cmap_labels:
                        label_rgb = self.cmap_labels(
                            self.cmap_labels.convert_img_labels(px))
                    else:
                        label_rgb = ax_img.cmap(ax_img.norm(px))
                    rgb = "RGB for label {}: {}".format(
                        px, tuple(np.multiply(label_rgb[:3], 255).astype(int)))
                if isinstance(px, float): px = "{:.4f}".format(px)

            # re-upsample coordinates for any downsampling
            orig_coord = np.multiply(coord, self.downsample)
            if self.offset:
                # shift for a single offset
                off = self.offset
                if libmag.is_seq(self.offset[0]):
                    # use corresponding offset for the given image in case
                    # overlaid images have different offsets
                    off = self.offset[i]
                orig_coord = np.add(orig_coord, off)
            output.append(
                "Image {}: x={}, y={}, px={}"
                .format(i, orig_coord[1], orig_coord[0], px))

        # join output message
        if rgb:
            output.append(rgb)
        msg = "; ".join(output)
        return msg
