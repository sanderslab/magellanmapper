# Numpy archive import/export.
# Author: David Young, 2019, 2020
"""Import/export for Numpy-based archives such as ``.npy`` and ``.npz`` formats.
"""
import os

import numpy as np
import pandas as pd

from magmap.plot import colormaps
from magmap.settings import config
from magmap.cv import detector
from magmap.io import importer
from magmap.io import libmag
from magmap.atlas import ontology
from magmap.io import sitk_io
from magmap.cv import stack_detect
from magmap.atlas import transformer
from magmap.plot import plot_3d


class Image5d:
    """Main image storage.
    
    Attributes:
        img (:obj:`np.ndarray`): 5D Numpy array in the format ``t,z,y,x,c``;
            defaults to None.
        path_img (str): Path from which ``img`` was loaded; defaults to None.
        path_meta (str): Path from which metadata for ``img`` was loaded;
            defaults to None.
        img_io (enum): I/O source for image5d array; defaults to None.
    
    """
    def __init__(self, img=None, path_img=None, path_meta=None, img_io=None):
        """Construct an Image5d object."""
        self.img = img
        self.path_img = path_img
        self.path_meta = path_meta
        self.img_io = img_io


def load_blobs(img_path, check_scaling=False, scaled_shape=None, scale=None):
    """Load blobs from an archive.
    
    Scaling can be computed to translate blob coordinates into another
    space, such as a heat map for a downsampled image.
    
    Args:
        img_path (str): Base path to blobs.
        check_scaling (bool): True to check scaling, in which case
            the scaling factor and scaled resolutions will be returned.
            Defaults to False.
        scaled_shape (List): Shape of image to calculate scaling factor if
            this factor cannot be found from a transposed file's metadata;
            defaults to None.
        scale (int, float): Scalar scaling factor, used to find a
            rescaled file; defaults to None. To find a resized file instead,
            set an atlas profile with the resizing factor.

    Returns:
        :obj:`magmap.cv.detector.Blobs`[, List, List]: Blobs object.
        If ``check_scaling`` is True, also returns sequence of scaling
        factors to a scaled or resized image, or None if not loaded or
        given, and the resolutions of the full-sized image in which the
        blobs were detected.

    """
    # load blobs and display counts
    path = libmag.combine_paths(img_path, config.SUFFIX_BLOBS)
    print("Loading blobs from", path)
    with np.load(path) as archive:
        info = read_np_archive(archive)
        blobs = detector.Blobs()
        if "segments" in info:
            blobs.blobs = info["segments"]
            print("Loaded {} blobs".format(len(blobs.blobs)))
            if config.verbose:
                detector.show_blobs_per_channel(blobs.blobs)
        if "colocs" in info:
            blobs.colocalizations = info["colocs"]
            if blobs.colocalizations is not None:
                print("Loaded blob co-localizations for {} channels"
                      .format(blobs.colocalizations.shape[1]))
        if config.verbose:
            print(info)
    if not check_scaling:
        return blobs

    # get scaling and resolutions from blob space to that of a down/upsampled
    # image space
    load_size = config.atlas_profile["target_size"]
    img_path_transposed = transformer.get_transposed_image_path(
        img_path, scale, load_size)
    scaling = None
    res = None
    if scale is not None or load_size is not None:
        # retrieve scaling from a rescaled/resized image
        _, img_info = importer.read_file(
            img_path_transposed, config.series, return_info=True)
        scaling = img_info["scaling"]
        res = np.multiply(config.resolutions[0], scaling)
        print("retrieved scaling from resized image:", scaling)
        print("rescaled resolution for full-scale image:", res)
    elif scaled_shape is not None:
        # fall back to scaling based on comparison to original image
        img5d = importer.read_file(img_path_transposed, config.series)
        scaling = importer.calc_scaling(
            img5d.img, None, scaled_shape=scaled_shape)
        res = config.resolutions[0]
        print("using scaling compared to full image:", scaling)
        print("resolution from full-scale image:", res)
    return blobs, scaling, res


def read_np_archive(archive):
    """Load Numpy archive file into a dictionary, skipping any values 
    that cannot be loaded.

    Args:
        archive: Loaded Numpy archive.

    Returns:
        Dictionary with keys and values corresponding to that of the 
        Numpy archive, skipping any values that could not be loaded 
        such as those that would require pickling when not allowed.
    """
    output = {}
    for key in archive.keys():
        try:
            output[key] = archive[key]
        except ValueError:
            print("unable to load {} from archive, will ignore".format(key))
    return output


def _check_np_none(val):
    """Checks if a value is either NoneType or a Numpy None object such as
    that returned from a Numpy archive that saved an undefined variable.
    
    Args:
        val: Value to check.
    
    Returns:
        The value if not a type of None, or a NoneType.
    """
    return None if val is None or np.all(np.equal(val, None)) else val


def setup_images(path=None, series=None, offset=None, size=None,
                 proc_mode=None, allow_import=True):
    """Sets up an image and all associated images and metadata.

    Paths for related files such as registered images will generally be
    constructed from ``path``. If :attr:`config.prefix` is set, it will
    be used in place of ``path`` for registered labels.
    
    Args:
        path (str): Path to image from which MagellanMapper-style paths will 
            be generated.
        series (int): Image series number; defaults to None.
        offset (List[int]): Sub-image offset given in z,y,x; defaults to None.
        size (List[int]): Sub-image shape given in z,y,x; defaults to None.
        proc_mode (str): Processing mode, which should be a key in 
            :class:`config.ProcessTypes`, case-insensitive; defaults to None.
        allow_import (bool): True to allow importing the image if it
            cannot be loaded; defaults to True.
    
    """
    def add_metadata():
        # override metadata set from command-line metadata args if available
        md = {
            config.MetaKeys.RESOLUTIONS: config.meta_dict[
                config.MetaKeys.RESOLUTIONS],
            config.MetaKeys.MAGNIFICATION: config.meta_dict[
                config.MetaKeys.MAGNIFICATION],
            config.MetaKeys.ZOOM: config.meta_dict[config.MetaKeys.ZOOM],
            config.MetaKeys.SHAPE: config.meta_dict[config.MetaKeys.SHAPE],
            config.MetaKeys.DTYPE: config.meta_dict[config.MetaKeys.DTYPE],
        }
        for key, val in md.items():
            if val is not None:
                # explicitly set metadata takes precedence over extracted vals
                import_md[key] = val
    
    # LOAD MAIN IMAGE
    
    # reset image5d
    config.image5d = None
    config.image5d_is_roi = False
    config.img5d = Image5d()
    load_subimage = offset is not None and size is not None
    config.resolutions = None
    
    # reset label images
    config.labels_img = None
    config.labels_img_sitk = None
    config.borders_img = None

    filename_base = importer.filename_to_base(path, series)
    subimg_base = None

    if load_subimage and not config.save_subimg:
        # load a saved sub-image file if available and not set to save one
        subimg_base = stack_detect.make_subimage_name(
            filename_base, offset, size)
        filename_subimg = libmag.combine_paths(
            subimg_base, config.SUFFIX_SUBIMG)

        try:
            # load sub-image if available
            config.image5d = np.load(filename_subimg, mmap_mode="r")
            config.image5d = importer.roi_to_image5d(config.image5d)
            config.image5d_is_roi = True
            config.img5d.img = config.image5d
            config.img5d.path_img = filename_subimg
            config.img5d.img_io = config.LoadIO.NP
            print("Loaded sub-image from {} with shape {}"
                  .format(filename_subimg, config.image5d.shape))

            # after loading sub-image, load original image's metadata
            # for essential data such as vmin/vmax; will only warn if
            # fails to load since metadata could be specified elsewhere
            _, orig_info = importer.make_filenames(path, series)
            print("load original image metadata from:", orig_info)
            importer.load_metadata(orig_info)
        except IOError:
            print("Ignored sub-image file from {} as unable to load"
                  .format(filename_subimg))

    proc_type = libmag.get_enum(proc_mode, config.ProcessTypes)
    if config.load_data[config.LoadData.BLOBS] or proc_type in (
            config.ProcessTypes.LOAD,
            config.ProcessTypes.COLOC_MATCH,
            config.ProcessTypes.EXPORT_ROIS,
            config.ProcessTypes.EXPORT_BLOBS,
            config.ProcessTypes.DETECT):
        # load a blobs archive
        try:
            if subimg_base:
                try:
                    # load blobs generated from sub-image
                    config.blobs = load_blobs(subimg_base)
                except (FileNotFoundError, KeyError):
                    # fallback to loading from full image blobs and getting
                    # a subset, shifting them relative to sub-image offset
                    print("Unable to load blobs file based on {}, will try "
                          "from {}".format(subimg_base, filename_base))
                    config.blobs = load_blobs(filename_base)
                    config.blobs.blobs, _ = detector.get_blobs_in_roi(
                        config.blobs.blobs, offset, size, reverse=False)
                    detector.shift_blob_rel_coords(
                        config.blobs.blobs, np.multiply(offset, -1))
            else:
                # load full image blobs
                config.blobs = load_blobs(filename_base)
        except (FileNotFoundError, KeyError) as e2:
            print("Unable to load blobs file")
            if proc_type in (
                    config.ProcessTypes.LOAD, config.ProcessTypes.EXPORT_BLOBS):
                # blobs expected but not found
                raise e2
    
    if path and config.image5d is None:
        # load or import the main image stack
        print("Loading main image")
        try:
            if path.endswith(sitk_io.EXTS_3D):
                # attempt to format supported by SimpleITK and prepend time axis
                config.image5d = sitk_io.read_sitk_files(path)[None]
                config.img5d.img = config.image5d
                config.img5d.path_img = path
                config.img5d.img_io = config.LoadIO.SITK
            else:
                # load or import from MagellanMapper Numpy format
                import_only = proc_type is config.ProcessTypes.IMPORT_ONLY
                img5d = None
                if not import_only:
                    # load previously imported image
                    img5d = importer.read_file(path, series)
                if allow_import:
                    # re-import over existing image or import new image
                    if os.path.isdir(path) and all(
                            [r is None for r in config.reg_suffixes.values()]):
                        # import directory of single plane images to single
                        # stack if no register suffixes are set
                        chls, import_md = importer.setup_import_dir(path)
                        add_metadata()
                        prefix = config.prefix
                        if not prefix:
                            prefix = os.path.join(
                                os.path.dirname(path),
                                importer.DEFAULT_IMG_STACK_NAME)
                        img5d = importer.import_planes_to_stack(
                            chls, prefix, import_md)
                    elif import_only or img5d is None:
                        # import multi-plane image
                        chls, import_path = importer.setup_import_multipage(
                            path)
                        prefix = config.prefix if config.prefix else import_path
                        import_md = importer.setup_import_metadata(
                            chls, config.channel, series)
                        add_metadata()
                        img5d = importer.import_multiplane_images(
                            chls, prefix, import_md, series,
                            channel=config.channel)
                if img5d is not None:
                    # set loaded main image in config
                    config.img5d = img5d
                    config.image5d = config.img5d.img
        except FileNotFoundError as e:
            print(e)
            print("Could not load {}, will fall back to any associated "
                  "registered image".format(path))
    
    if config.metadatas and config.metadatas[0]:
        # assign metadata from alternate file if given to supersede settings
        # for any loaded image5d
        # TODO: access metadata directly from given image5d's dict to allow
        # loading multiple image5d images simultaneously
        importer.assign_metadata(config.metadatas[0])
    
    # main image is currently required since many parameters depend on it
    atlas_suffix = config.reg_suffixes[config.RegSuffixes.ATLAS]
    if atlas_suffix is None and config.image5d is None:
        # fallback to atlas if main image not already loaded
        atlas_suffix = config.RegNames.IMG_ATLAS.value
        print("main image is not set, falling back to registered "
              "image with suffix", atlas_suffix)
    # use prefix to get images registered to a different image, eg a
    # downsampled version, or a different version of registered images
    path = config.prefix if config.prefix else path
    if path and atlas_suffix is not None:
        try:
            # will take the place of any previously loaded image5d
            config.image5d = sitk_io.read_sitk_files(
                path, reg_names=atlas_suffix)[None]
            config.img5d.img = config.image5d
            config.img5d.img_io = config.LoadIO.SITK
        except FileNotFoundError as e:
            print(e)
    
    annotation_suffix = config.reg_suffixes[config.RegSuffixes.ANNOTATION]
    if annotation_suffix is not None:
        try:
            # load labels image
            # TODO: need to support multichannel labels images
            config.labels_img, config.labels_img_sitk = sitk_io.read_sitk_files(
                path, reg_names=annotation_suffix, return_sitk=True)
        except FileNotFoundError as e:
            print(e)
            if config.image5d is not None:
                # create a blank labels images for custom annotation; colormap
                # can be generated for the original labels loaded below
                config.labels_img = np.zeros(
                    config.image5d.shape[1:4], dtype=int)
                print("Created blank labels image from main image")
        if config.image5d is not None and config.labels_img is not None:
            # set up scaling factors by dimension between intensity and
            # labels images
            config.labels_scaling = importer.calc_scaling(
                config.image5d, config.labels_img)
        try:
            if config.load_labels is not None:
                # load labels reference file
                labels_ref = ontology.load_labels_ref(config.load_labels)
                if isinstance(labels_ref, pd.DataFrame):
                    # parse CSV files loaded into data frame
                    config.labels_ref_lookup = ontology.create_lookup_pd(
                        labels_ref)
                else:
                    # parse dict from ABA JSON file
                    config.labels_ref_lookup = (
                        ontology.create_aba_reverse_lookup(labels_ref))
        except FileNotFoundError as e:
            print(e)
    
    borders_suffix = config.reg_suffixes[config.RegSuffixes.BORDERS]
    if borders_suffix is not None:
        # load borders image, which can also be another labels image
        try:
            config.borders_img = sitk_io.read_sitk_files(
                path, reg_names=borders_suffix)
        except FileNotFoundError as e:
            print(e)
    
    if (config.atlas_labels[config.AtlasLabels.ORIG_COLORS]
            and config.load_labels is not None):
        # load original labels image from same directory as ontology
        # file for consistent ID-color mapping, even if labels are missing
        try:
            config.labels_img_orig = sitk_io.load_registered_img(
                config.load_labels, config.RegNames.IMG_LABELS.value)
        except FileNotFoundError as e:
            print(e)
            libmag.warn(
                "could not load original labels image; colors may differ"
                "differ from it")
    
    load_rot90 = config.roi_profile["load_rot90"]
    if load_rot90 and config.image5d is not None:
        # rotate main image specified num of times x90deg after loading since 
        # need to rotate images output by deep learning toolkit
        config.image5d = np.rot90(config.image5d, load_rot90, (2, 3))

    if (config.image5d is not None and load_subimage
            and not config.image5d_is_roi):
        # crop full image to bounds of sub-image
        config.image5d = plot_3d.prepare_subimg(
            config.image5d, offset, size)[None]
        config.image5d_is_roi = True

    # add any additional image5d thresholds for multichannel images, such 
    # as those loaded without metadata for these settings
    colormaps.setup_cmaps()
    num_channels = get_num_channels(config.image5d)
    config.near_max = libmag.pad_seq(config.near_max, num_channels, -1)
    config.near_min = libmag.pad_seq(config.near_min, num_channels, 0)
    config.vmax_overview = libmag.pad_seq(
        config.vmax_overview, num_channels)
    colormaps.setup_colormaps(num_channels)


def get_num_channels(image5d):
    """Get the number of channels in a 5D image.

    Args:
        image5d (:obj:`np.ndarray`): Numpy arry in the order, `t,z,y,x[,c]`.

    Returns:
        int: Number of channels inferred based on the presence and length
        of the 5th dimension.

    """
    return 1 if image5d is None or image5d.ndim <= 4 else image5d.shape[4]


def write_raw_file(arr, path):
    """Write an array to a RAW data file.
    
    The array will be output directly to disk through a memmapped object.
    
    Args:
        arr (:obj:`np.ndarray`): Array to write.
        path (str): Output path.

    """
    print("Writing array of shape {}, type {} for {}"
          .format(arr.shape, arr.dtype, path))
    out_file = np.memmap(path, dtype=arr.dtype, mode="w+", shape=arr.shape)
    out_file[:] = arr[:]
    del out_file  # flushes to disk
    print("Finished writing", path)
