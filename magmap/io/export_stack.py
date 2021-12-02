# Stack manipulations
# Author: David Young, 2017, 2019
"""Import and export image stacks in various formats.
"""

from collections import OrderedDict
import os
import glob
from typing import List, Optional, Sequence

import numpy as np
from skimage import transform
from skimage import io
from matplotlib import animation
from matplotlib.image import AxesImage
from scipy import ndimage

from magmap.cv import chunking, cv_nd
from magmap.plot import colormaps
from magmap.settings import config
from magmap.io import importer
from magmap.io import libmag
from magmap.io import np_io
from magmap.plot import plot_3d
from magmap.plot import plot_support

_logger = config.logger.getChild(__name__)


class StackPlaneIO(chunking.SharedArrsContainer):
    """Worker class to export planes from a stack with support for 
    multiprocessing.
    
    Attributes:
        imgs: A list of images, with exact specification determined by 
            the calling function.
        images: Sequence of images. For import, each "image" is a path to 
            and image file. For export, each "image" is a sequence of 
            planes, with the first sequence assumed to an atlas, 
            followed by labels-based images, each consisting of 
            corresponding planes.
        fn_process: Function to process each image through multiprocessing, 
            where the function should take an index and image and return the 
            index and processed plane.
        rescale (float): Rescale factor; defaults to 1.
        cmaps_labels: Sequence of colormaps for labels-based images; 
            defaults to None. Length should be equal to that of 
            ``images`` - 1.
        start_planei (int): Index of start plane, used for labeling the
            plane; defaults to 0. The plane is only annotated when
            :attr:`config.plot_labels[config.PlotLabels.TEXT_POS]` is given
            to specify the position of the text in ``x,y`` relative to the
            axes.
    """
    imgs = None
    
    def __init__(self):
        super().__init__()
        self.images = None
        self.fn_process = None
        self.rescale = 1
        self.start_planei = 0
        self.origin = None
        self.aspect = None
        self.cmaps_labels = None
        self.img_slice = None
    
    @classmethod
    def set_data(cls, imgs):
        """Set data to be accessed by worker functions."""
        cls.imgs = imgs
    
    @classmethod
    def convert_imgs(cls):
        """Restore all shared arrays to a list of arrays."""
        if cls.imgs is None:
            cls.imgs = []
            for key in cls.shared_arrs.keys():
                cls.imgs.append(cls.convert_shared_arr(key))
    
    @classmethod
    def import_img(cls, i, rescale, multichannel):
        """Import and rescale an image.
        
        Assumes that :attr:``imgs`` is a list of paths to 2D images.
        
        Args:
            i: Index within :attr:``imgs`` to plot.
            rescale: Rescaling multiplier.
            multichannel: True if the images are multichannel.
        
        Returns:
            A tuple of ``i`` and a list of the processed images. The 
            processed image list has the same length as :attr:``imgs``, 
            or the number of image paths.
        """
        # TODO: consider removing since imported image is not saved;
        # should import first before exporting
        
        cls.convert_imgs()
        path = cls.imgs[i]
        print("importing {}".format(path))
        img = io.imread(path)
        img = transform.rescale(
            img, rescale, mode="reflect", multichannel=multichannel, 
            preserve_range=True, anti_aliasing=True)
        return i, img
    
    @classmethod
    def process_plane(cls, i, target_size, rotate=None):
        """Process corresponding planes from related images.
        
        Assumes that :attr:``imgs`` is a list of nested 2D image lists, 
        where the first nested list is assumed to be a sequence of 
        histology image planes, while subsequent images are 
        labels-based images.
        
        Args:
            i: Index within nested lists of :attr:``imgs`` to plot.
            target_size: Resize to this shape.
            rotate: Degrees by which to rotate; defaults to None.
        
        Returns:
            A tuple of ``i`` and a list of the processed images. The 
            processed image list has the same length as :attr:``imgs``, 
            or the number of nested lists.
        """
        print("processing plane {}".format(i))
        cls.convert_imgs()
        imgs_proc = []
        for j, img_stack in enumerate(cls.imgs):
            if j == 0:
                # atlas image
                img = transform.resize(
                    img_stack[i], target_size, mode="reflect",
                    preserve_range=True, anti_aliasing=True)
            else:
                # labels-based image, using nearest-neighbor interpolation
                img = transform.resize(
                    img_stack[i], target_size, mode="reflect",
                    preserve_range=True, anti_aliasing=False, order=0)
            imgs_proc.append(img)
        if rotate:
            # TODO: consider removing since not saved; should instead rotate
            # image using transform task
            
            # rotate, filling background with edge color
            for j, img in enumerate(imgs_proc):
                #img = img[10:-100, 10:-80] # manually crop out any border
                cval = np.mean(img[0, 0])
                img = ndimage.rotate(img, rotate, cval=cval)
                # additional cropping for oblique bottom edge after rotation
                #img = img[:-50]
                imgs_proc[j] = img
        return i, imgs_proc

    def build_stack(
            self, axs: List, scale_bar: bool = True, fit: bool = False
    ) -> Optional[List]:
        """Builds a stack of Matploblit 2D images.
        
        Uses multiprocessing to load or resize each image.
        
        Args:
            axs: Sub-plot axes.
            scale_bar: True to include scale bar; defaults to True.
            fit: True to fit the figure frame to the resulting image.
        
        Returns:
            :List[List[:obj:`matplotlib.image.AxesImage`]]: Nested list of 
            axes image objects. The first list level contains planes, and
            the second level are channels within each plane.
        
        """
        def handle_extracted_plane():
            # get sub-plot and hide x/y axes
            ax = axs
            if libmag.is_seq(ax):
                ax = axs[imgi]
            plot_support.hide_axes(ax)
    
            # multiple artists can be shown at each frame by collecting 
            # each group of artists in a list; overlay_images returns 
            # a nested list containing a list for each image, which in turn 
            # contains a list of artists for each channel
            ax_imgs = plot_support.overlay_images(
                ax, self.aspect, self.origin, imgs, None, cmaps_all,
                ignore_invis=True, check_single=True)
            if colorbar and len(ax_imgs) > 0 and len(ax_imgs[0]) > 0:
                # add colorbar with scientific notation if outside limits
                cbar = ax.figure.colorbar(ax_imgs[0][0], ax=ax, shrink=0.7)
                plot_support.set_scinot(cbar.ax, lbls=None, units=None)
            plotted_imgs[imgi] = np.array(ax_imgs).flatten()
            
            if libmag.is_seq(text_pos) and len(text_pos) > 1:
                # write plane index in axes rather than data coordinates
                text = ax.text(
                    *text_pos[:2], "{}-plane: {}".format(
                        plot_support.get_plane_axis(config.plane),
                        self.start_planei + imgi),
                    transform=ax.transAxes, color="w")
                plotted_imgs[imgi] = [*plotted_imgs[imgi], text]

            if scale_bar:
                plot_support.add_scale_bar(ax, 1 / self.rescale, config.plane)
        
        # number of image types (eg atlas, labels) and corresponding planes
        num_image_types = len(self.images)
        if num_image_types < 1: return None
        num_images = len(self.images[0])
        if num_images < 1: return None
        
        # import the images as Matplotlib artists via multiprocessing
        plotted_imgs: List = [None] * num_images
        img_shape = self.images[0][0].shape
        target_size = np.multiply(img_shape, self.rescale).astype(int)
        multichannel = self.images[0][0].ndim >= 3
        if multichannel:
            print("building stack for channel: {}".format(config.channel))
            target_size = target_size[:-1]

        # setup imshow parameters
        colorbar = config.roi_profile["colorbar"]
        cmaps_all = [config.cmaps, *self.cmaps_labels]
        text_pos = config.plot_labels[config.PlotLabels.TEXT_POS]
        
        StackPlaneIO.set_data(self.images)
        pool_results = None
        pool = None
        multiprocess = self.rescale != 1
        if multiprocess:
            # set up multiprocessing
            initializer = None
            initargs = None
            if not chunking.is_fork():
                # set up labels image as a shared array for spawned mode
                initializer, initargs = StackPlaneIO.build_pool_init(
                    OrderedDict([
                        (i, img) for i, img in enumerate(self.images)]))
    
            pool = chunking.get_mp_pool(initializer, initargs)
            pool_results = []

        for i in range(num_images):
            # add rotation argument if necessary
            args = (i, target_size)
            if pool is None:
                # extract and handle without multiprocessing
                imgi, imgs = self.fn_process(*args)
                handle_extracted_plane()
            else:
                # extract plane in multiprocessing
                pool_results.append(
                    pool.apply_async(self.fn_process, args=args))
        
        if multiprocess:
            # handle multiprocessing output
            for result in pool_results:
                imgi, imgs = result.get()
                handle_extracted_plane()
            pool.close()
            pool.join()

        if fit and plotted_imgs:
            # fit each figure to its first available image
            for ax_img in plotted_imgs:
                # images may be flattened AxesImage, array of AxesImage and
                # Text, or None if alpha set to 0
                if libmag.is_seq(ax_img):
                    ax_img = ax_img[0]
                if isinstance(ax_img, AxesImage):
                    plot_support.fit_frame_to_image(
                        ax_img.figure, ax_img.get_array().shape, self.aspect)
        
        return plotted_imgs


def animate_imgs(base_path, plotted_imgs, delay, ext=None, suffix=None):
    """Export to an animated image.
    
    Defaults to an animated GIF unless ``ext`` specifies otherwise.
    Requires ``FFMpeg`` for MP4 file format exports and ``ImageMagick`` for
    all other types of exports.
    
    Args:
        base_path (str): String from which an output path will be constructed.
        plotted_imgs (List[:obj:`matplotlib.image.AxesImage]): Sequence of
            images to include in the animation.
        delay (int): Delay between image display in ms. If None, the delay will
            defaul to 100ms.
        ext (str): Extension to use when saving, without the period. Defaults
            to None, in which case "gif" will be used.
        suffix (str): String to append to output path before extension;
            defaults to None to ignore.

    """
    # set up animation output path and time interval
    if ext is None: ext = "gif"
    out_path = libmag.combine_paths(base_path, "animated", ext=ext)
    if suffix: out_path = libmag.insert_before_ext(out_path, suffix, "_")
    libmag.backup_file(out_path)
    if delay is None:
        delay = 100
    if plotted_imgs and len(plotted_imgs[0]) > 0:
        fig = plotted_imgs[0][0].figure
    else:
        libmag.warn("No images available to animate")
        return
    
    # WORKAROUND: FFMpeg may give a "height not divisible by 2" error, fixed
    # by padding with a pixel
    # TODO: check if needed for width
    # TODO: account for difference in FFMpeg height and fig height
    for fn, size in {
            # fig.set_figwidth: fig.get_figwidth(),
            fig.set_figheight: fig.get_figheight()}.items():
        if size * fig.dpi % 2 != 0:
            fn(size + 1. / fig.dpi)
            print("Padded size with", fn, fig.get_figwidth(), "to new size of",
                  fig.get_figheight())
    
    # generate and save animation
    anim = animation.ArtistAnimation(
        fig, plotted_imgs, interval=delay, repeat_delay=0, blit=False)
    try:
        writer = "ffmpeg" if ext == "mp4" else "imagemagick"
        anim.save(out_path, writer=writer)
        print("saved animation file to {}".format(out_path))
    except ValueError as e:
        print(e)
        libmag.warn("No animation writer available for Matplotlib")


def _setup_labels_cmaps(imgs, cmaps_labels=None):
    """Set up labels colormaps for registered images.
    
    Args:
        imgs (List[:obj:`np.ndarray`]): Sequence of images where the first
            is assumed to be non-labels, the second is labels, and the
            third are label borders.
        cmaps_labels (List[List[
            :class:`magmap.plot.colormaps.DiscreteColormap`]]): 
            List of discrete colormaps corresponding to ``imgs[:1]``.
    
    Returns:
        list: List of colormaps for ``[labels_img, borders_img]``.
    
    """
    if cmaps_labels is None:
        cmaps_labels = []
    num_imgs = len(imgs)
    if num_imgs > 1:
        # get colormap for 2nd image, the main labels image
        cmaps_labels.append(config.cmap_labels)
    if num_imgs > 2:
        # subsequent image's colormap is based on first labels if possible
        cmaps_labels.append(
            colormaps.get_borders_colormap(
                imgs[2], imgs[1], cmaps_labels[0]))
    return cmaps_labels


def setup_stack(
        image5d: np.ndarray, path: Optional[str] = None,
        offset: Optional[Sequence[int]] = None,
        roi_size: Optional[Sequence[int]] = None,
        slice_vals: Optional[Sequence[int]] = None,
        rescale: Optional[float] = None,
        labels_imgs: Optional[Sequence[np.ndarray]] = None
) -> StackPlaneIO:
    """Set up a stack of images for export to file.
     
    Supports a stack of image files in a directory or a single volumetric image
    and associated labels images.
    
    Args:
        image5d: Images as a 4/5D Numpy array (t,z,y,x[c]). Can be None if 
            ``path`` is set.
        path: Path to an image directory from which all files will be imported 
            in Python sorted order, taking precedence over ``imaged5d``; 
            defaults to None.
        offset: Tuple of offset given in user order (x, y, z); defaults to 
            None. Requires ``roi_size`` to not be None.
        roi_size: Size of the region of interest in user order (x, y, z); 
            defaults to None. Requires ``offset`` to not be None.
        slice_vals: List from which to construct a slice object to 
            extract only a portion of the image. Defaults to None, which 
            will give the whole image. If ``offset`` and ``roi_size`` are 
            also given, ``slice_vals`` will only be used for its interval term.
        rescale: Rescaling factor for each image, performed on a plane-by-plane 
            basis; defaults to None, in which case 1.0 will be used.
        labels_imgs: Sequence of labels-based images as a Numpy z,y,x arrays, 
            typically including labels and borders images; defaults to None.
    
    Returns:
        Stack builder instance.
    
    """
    print("Starting image stack setup")
    
    # build "z" slice, which will be applied to the transposed image
    interval = 1  # default to export each plane
    if offset is not None and roi_size is not None:
        # extract planes based on ROI settings
        
        # transpose coordinates to given plane
        _, arrs_1d = plot_support.transpose_images(
            config.plane, arrs_1d=[offset[::-1], roi_size[::-1]])
        offset = arrs_1d[0][::-1]
        roi_size = arrs_1d[1][::-1]
        
        # ROI offset and size take precedence over slice vals except 
        # for use of the interval term
        if slice_vals is not None and len(slice_vals) > 2:
            interval = slice_vals[2]
        size = roi_size[2]
        img_sl = slice(offset[2], offset[2] + size, interval)
        if interval is not None and interval < 0:
            # reverse start/stop order to iterate backward
            img_sl = slice(img_sl.stop, img_sl.start, interval)
        print("using ROI offset {}, size {}, {}"
              .format(offset, size, img_sl))
    
    elif slice_vals:
        # build directly from slice vals, replacing start and step if None
        sl = slice(*slice_vals)
        sl = [sl.start, sl.stop, sl.step]
        if sl[0] is None:
            # default to start at beginning of stack
            sl[0] = 0
        if sl[2] is None:
            # default to interval/step of 1
            sl[2] = 1
        img_sl = slice(*sl)
    
    else:
        # default to take the whole image stack
        img_sl = slice(0, None, 1)
    
    if rescale is None:
        rescale = 1.0
    aspect = None
    origin = None
    cmaps_labels = []
    extracted_planes = []
    start_planei = 0
    if path and os.path.isdir(path):
        # builds animations from all files in a directory
        planes = sorted(glob.glob(os.path.join(path, "*")))[::interval]
        _logger.info("Importing images from %s: %s", path, planes)
        fnc = StackPlaneIO.import_img
        extracted_planes.append(planes)
    else:
        # load images from path and extract ROI based on slice parameters, 
        # assuming 1st image is atlas, 2nd and beyond are labels-based
        imgs = [image5d]
        if labels_imgs is not None:
            for img in labels_imgs:
                if img is not None: imgs.append(img[None])
            _setup_labels_cmaps(imgs, cmaps_labels)
        main_shape = None  # z,y,x shape of 1st image
        for img in imgs:
            sl = img_sl
            img_shape = img.shape[1:4]
            if main_shape:
                if main_shape != img_shape:
                    # scale slice bounds to the first image's shape
                    scaling = np.divide(img_shape, main_shape)
                    axis = plot_support.get_plane_axis(config.plane, True)
                    sl = libmag.scale_slice(
                        sl, scaling[axis], img_shape[axis])
            else:
                main_shape = img_shape
            planes, aspect, origin = plot_support.extract_planes(
                img, sl, plane=config.plane)
            if offset is not None and roi_size is not None:
                # get ROI using transposed coordinates on transposed planes; 
                # returns list
                planes = planes[
                    :, offset[1]:offset[1]+roi_size[1], 
                    offset[0]:offset[0]+roi_size[0]]
            extracted_planes.append(planes)
        fnc = StackPlaneIO.process_plane
        if img_sl.start:
            start_planei = img_sl.start
    
    # store in stack worker
    stacker = StackPlaneIO()
    stacker.images = extracted_planes
    stacker.fn_process = fnc
    stacker.rescale = rescale
    stacker.start_planei = start_planei
    stacker.origin = origin
    stacker.aspect = aspect
    stacker.cmaps_labels = cmaps_labels
    stacker.img_slice = img_sl
    
    return stacker


def stack_to_img(paths, roi_offset, roi_size, series=None, subimg_offset=None,
                 subimg_size=None, animated=False, suffix=None):
    """Build an image file from a stack of images in a directory or an 
    array, exporting as an animated GIF or movie for multiple planes or 
    extracting a single plane to a standard image file format.
    
    Writes the file to the parent directory of path.
    
    Args:
        paths (List[str]): Image paths, which can each be either an image 
            directory or a base path to a single image, including 
            volumetric images.
        roi_offset (Sequence[int]): Tuple of offset given in user order
            ``x,y,z``; defaults to None. Requires ``roi_size`` to not be None.
        roi_size (Sequence[int]): Size of the region of interest in user order 
            ``x,y,z``; defaults to None. Requires ``roi_offset`` to not be None.
        series (int): Image series number; defaults to None.
        subimg_offset (List[int]): Sub-image offset as (z,y,x) to load;
            defaults to None.
        subimg_size (List[int]): Sub-image size as (z,y,x) to load;
            defaults to None.
        animated (bool): True to export as an animated image; defaults to False.
        suffix (str): String to append to output path before extension; 
            defaults to None to ignore.

    """
    # set up figure layout for collages
    size = config.plot_labels[config.PlotLabels.LAYOUT]
    ncols, nrows = size if size else (1, 1)
    num_paths = len(paths)
    collage = num_paths > 1
    figs = {}
    
    for i in range(nrows):
        for j in range(ncols):
            n = i * ncols + j
            if n >= num_paths: break
            
            # load an image and set up its image stacker
            path_sub = paths[n]
            axs = []
            # TODO: test directory of images
            # TODO: consider not reloading first image
            np_io.setup_images(path_sub, series, subimg_offset, subimg_size)
            stacker = setup_stack(
                config.image5d, path_sub, offset=roi_offset,
                roi_size=roi_size, slice_vals=config.slice_vals, 
                rescale=config.transform[config.Transforms.RESCALE],
                labels_imgs=(config.labels_img, config.borders_img))

            # add sub-plot title unless groups given as empty string
            title = None
            if config.groups:
                title = libmag.get_if_within(config.groups, n)
            elif num_paths > 1:
                title = os.path.basename(path_sub)
            
            if not stacker.images: continue
            for k in range(len(stacker.images[0])):
                # create or retrieve fig; animation has only 1 fig
                planei = 0 if animated else (
                        stacker.img_slice.start + k * stacker.img_slice.step)
                fig_dict = figs.get(planei)
                if not fig_dict:
                    # set up new fig
                    fig, gs = plot_support.setup_fig(
                        nrows, ncols, config.plot_labels[config.PlotLabels.SIZE])
                    fig_dict = {"fig": fig, "gs": gs, "imgs": []}
                    figs[planei] = fig_dict
                ax = fig_dict["fig"].add_subplot(fig_dict["gs"][i, j])
                if title:
                    ax.title.set_text(title)
                axs.append(ax)

            # export planes
            plotted_imgs = stacker.build_stack(
                axs, config.plot_labels[config.PlotLabels.SCALE_BAR],
                size is None or ncols * nrows == 1)

            if animated:
                # store all plotted images in single fig
                fig_dict = figs.get(0)
                if fig_dict:
                    fig_dict["imgs"] = plotted_imgs
            else:
                # store one plotted image per fig; not used currently
                for fig_dict, img in zip(figs.values(), plotted_imgs):
                    fig_dict["imgs"].append(img)
    
    path_base = paths[0]
    for planei, fig_dict in figs.items():
        if animated:
            # generate animated image (eg animated GIF or movie file)
            animate_imgs(
                path_base, fig_dict["imgs"], config.delay, config.savefig,
                suffix)
        else:
            # generate single figure with axis and plane index in filename
            if collage:
                # output filename as a collage of images
                if not os.path.isdir(path_base):
                    path_base = os.path.dirname(path_base)
                path_base = os.path.join(path_base, "collage")
            
            # insert mod as suffix, then add any additional suffix;
            # can use config.prefix_out for make_out_path prefix
            mod = "_plane_{}{}".format(
                plot_support.get_plane_axis(config.plane), planei)
            out_path = libmag.make_out_path(path_base, suffix=mod)
            if suffix:
                out_path = libmag.insert_before_ext(out_path, suffix)
            plot_support.save_fig(
                out_path, config.savefig, fig=fig_dict["fig"])


def reg_planes_to_img(imgs, path=None, ax=None):
    """Export registered image single planes to a single figure.
    
    Simplified export tool taking a single plane from each registered image
    type, overlaying in a single figure, and exporting to file.
    
    Args:
        imgs (List[:obj:`np.ndarray`]): Sequence of image planes to display.
            The first image is assumed to be greyscale, the second is labels,
            and any subsequent images are borders.
        path (str): Output base path, which will be combined with
            :attr:`config.savefig`; defaults to None to not save.
        ax (:obj:`matplotlib.image.Axes`): Axes on which to plot; defaults
            to False, in which case a new figure and axes will be generated.

    """
    if ax is None:
        # set up new figure with single subplot
        fig, gs = plot_support.setup_fig(
            1, 1, config.plot_labels[config.PlotLabels.SIZE])
        ax = fig.add_subplot(gs[0, 0])
    stacker = StackPlaneIO()
    stacker.images = [img[None] for img in imgs]
    stacker.fn_process = StackPlaneIO.process_plane
    stacker.cmaps_labels = _setup_labels_cmaps(imgs)
    plotted_imgs = stacker.build_stack(ax, scale_bar=False)
    ax_img = plotted_imgs[0][0]
    aspect, origin = plot_support.get_aspect_ratio(config.plane)
    plot_support.fit_frame_to_image(
        ax_img.figure, ax_img.get_array().shape, aspect)
    if path:
        plot_support.save_fig(path, config.savefig)


def export_planes(image5d, ext, channel=None, separate_chls=False):
    """Export all planes of a 3D+ image into separate 2D image files.
    
    Unlike :meth:`stack_to_img`, this method simply exports all planes and
    each channel into separate files. Supports image rotation set in
    :attr:`magmap.settings.config.transform`.

    Args:
        image5d (:obj:`np.ndarray`): Image in ``t,z,y,x[,c]`` format.
        ext (str): Save format given as an extension without period.
        channel (int): Channel to save; defaults to None for all channels.
        separate_chls (bool): True to export all channels from each plane to
            a separate image; defaults to False. 

    """
    # set up output path
    suffix = "_export" if config.suffix is None else config.suffix 
    out_path = libmag.make_out_path(suffix=suffix)
    output_dir = os.path.dirname(out_path)
    basename = os.path.splitext(os.path.basename(out_path))[0]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # set up image and apply any rotation
    roi = image5d[0]
    multichannel, channels = plot_3d.setup_channels(roi, channel, 3)
    rotate = config.transform[config.Transforms.ROTATE]
    roi = cv_nd.rotate90(roi, rotate, multichannel=multichannel)
    
    num_planes = len(roi)
    num_digits = len(str(num_planes))
    for i, plane in enumerate(roi):
        # add plane to output path if more than one output file
        out_name = basename if num_planes <= 1 else "{}_{:0{}d}".format(
            basename, i, num_digits)
        path = os.path.join(output_dir, out_name)
        if separate_chls and multichannel:
            for chl in channels:
                # save each channel as separate file
                plane_chl = plane[..., chl]
                path_chl = "{}{}{}.{}".format(
                    path, importer.CHANNEL_SEPARATOR, chl, ext)
                print("Saving image plane {} to {}".format(i, path_chl))
                io.imsave(path_chl, plane_chl)
        else:
            # save single channel plane
            path = "{}.{}".format(path, ext)
            print("Saving image plane {} to {}".format(i, path))
            io.imsave(path, plane)


if __name__ == "__main__":
    print("MagellanMapper stack manipulations")
