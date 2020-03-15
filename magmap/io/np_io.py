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


def load_blobs(img_path, check_scaling=False, scaled_shape=None, scale=None):
    """Load blobs from an archive and compute scaling.
    
    Args:
        img_path (str): Base path to blobs.
        check_scaling (bool): True to check scaling, in which case
            the scaling factor and scaled resolutions will be returned.
            Defaults to False.
        scaled_shape (List): Shape of image to calculate scaling factor
            this factor cannot be found from a transposed file's metadata;
            defaults to None.
        scale (int, float): Scalar scaling factor, used to find a
            transposed file; defaults to None.

    Returns:
        :obj:`np.ndarray`, List, List: Array of blobs If ``check_scaling``
        is True, also returns sequence of scaling
        factors to a scaled or resized image, or None if not loaded or given;
        and the resolutions of the full-sized image in which the blobs
        were detected. 

    """
    # load blobs and display counts
    path = libmag.combine_paths(img_path, config.SUFFIX_BLOBS)
    print("Loading blobs from", path)
    info = read_np_archive(np.load(path))
    blobs = info["segments"]
    print("loaded {} blobs".format(len(blobs)))
    if config.verbose:
        detector.show_blobs_per_channel(blobs)
        print(info)
    if not check_scaling:
        return blobs

    # get scaling from source image, which can be rescaled/resized image
    # since contains scaling image
    load_size = config.register_settings["target_size"]
    img_path_transposed = transformer.get_transposed_image_path(
        img_path, scale, load_size)
    scaling = None
    res = None
    if scale is not None or load_size is not None:
        _, img_info = importer.read_file(
            img_path_transposed, config.series, return_info=True)
        scaling = img_info["scaling"]
        res = np.multiply(config.resolutions[0], scaling)
        print("retrieved scaling from resized image:", scaling)
        print("rescaled resolution for full-scale image:", res)
    elif scaled_shape is not None:
        # fall back to scaling based on comparison to original image
        image5d = importer.read_file(
            img_path_transposed, config.series)
        scaling = importer.calc_scaling(
            image5d, None, scaled_shape=scaled_shape)
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
                 proc_mode=None):
    """Sets up an image and all associated images and metadata.
    
    Args:
        path (str): Path to image from which MagellanMapper-style paths will 
            be generated.
        series (int): Image series number; defaults to None.
        offset (List[int]): Sub-image offset given in z,y,x; defaults to None.
        size (List[int]): Sub-image shape given in z,y,x; defaults to None.
        proc_mode (str): Processing mode, which should be a key in 
            :class:`config.ProcessTypes`, case-insensitive; defaults to None.
    
    """
    # LOAD MAIN IMAGE
    
    # reset image5d
    config.image5d = None
    config.image5d_is_roi = False
    load_subimage = offset is not None and size is not None

    filename_base = importer.filename_to_base(path, series)
    filename_blobs = libmag.combine_paths(filename_base, config.SUFFIX_BLOBS)
    subimg_base = None

    if load_subimage:
        # load a saved sub-image file if available
        subimg_base = stack_detect.make_subimage_name(
            filename_base, offset, size)
        filename_subimg = libmag.combine_paths(
            subimg_base, config.SUFFIX_SUBIMG)

        try:
            # load sub-image if available
            config.image5d = np.load(filename_subimg, mmap_mode="r")
            config.image5d = importer.roi_to_image5d(config.image5d)
            config.image5d_is_roi = True
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
    if proc_type in (config.ProcessTypes.LOAD, config.ProcessTypes.EXPORT_ROIS,
                     config.ProcessTypes.EXPORT_BLOBS,
                     config.ProcessTypes.PROCESSING_MP):
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
                    config.blobs, _ = detector.get_blobs_in_roi(
                        config.blobs, offset, size, reverse=False)
                    detector.shift_blob_rel_coords(
                        config.blobs, np.multiply(offset, -1))
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
        if os.path.isdir(path):
            # import directory of TIFF images
            config.image5d = importer.import_dir(os.path.join(path, "*"))
        elif path.endswith(sitk_io.EXTS_3D):
            try:
                # attempt to format supported by SimpleITK and prepend time axis
                config.image5d = sitk_io.read_sitk_files(path)[None]
            except FileNotFoundError as e:
                print(e)
        else:
            # load or import from MagellanMapper Numpy format, using any path
            # changes during attempting ROI load
            load = proc_type is not config.ProcessTypes.IMPORT_ONLY  # re/import
            config.image5d = importer.read_file(
                path, series, channel=config.channel, load=load)
    
    if config.metadatas and config.metadatas[0]:
        # assign metadata from alternate file if given to supersede settings
        # for any loaded image5d
        # TODO: access metadata directly from given image5d's dict to allow
        # loading multiple image5d images simultaneously
        importer.assign_metadata(config.metadatas[0])

    if config.load_labels is not None:
        # load registered files including labels
        
        # main image is currently required since many parameters depend on it
        atlas_suffix = config.reg_suffixes[config.RegSuffixes.ATLAS]
        if atlas_suffix is None and config.image5d is None:
            # fallback to atlas if main image not already loaded
            atlas_suffix = config.RegNames.IMG_ATLAS.value
        if path and atlas_suffix is not None:
            # will take the place of any previously loaded image5d
            config.image5d = sitk_io.read_sitk_files(
                path, reg_names=atlas_suffix)[None]
        
        annotation_suffix = config.reg_suffixes[config.RegSuffixes.ANNOTATION]
        if annotation_suffix is not None:
            # load labels image, set up scaling, and load labels file, 
            # using prefix for registered files if given
            try:
                path = config.prefix if config.prefix else path
                # TODO: need to support multichannel labels images
                config.labels_img = sitk_io.read_sitk_files(
                    path, reg_names=annotation_suffix)
                if config.image5d is not None:
                    config.labels_scaling = importer.calc_scaling(
                        config.image5d, config.labels_img)
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
        
        if config.atlas_labels[config.AtlasLabels.ORIG_COLORS]:
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
    
    load_rot90 = config.process_settings["load_rot90"]
    if load_rot90 and config.image5d is not None:
        # rotate main image specified num of times x90deg after loading since 
        # need to rotate images output by deep learning toolkit
        config.image5d = np.rot90(config.image5d, load_rot90, (2, 3))

    if (config.image5d is not None and load_subimage
            and not config.image5d_is_roi):
        # crop full image to bounds of sub-image
        config.image5d = plot_3d.prepare_subimg(
            config.image5d, size, offset)[None]
        config.image5d_is_roi = True

    # add any additional image5d thresholds for multichannel images, such 
    # as those loaded without metadata for these settings
    colormaps.setup_cmaps()
    num_channels = (1 if config.image5d is None or config.image5d.ndim <= 4 
                    else config.image5d.shape[4])
    config.near_max = libmag.pad_seq(config.near_max, num_channels, -1)
    config.near_min = libmag.pad_seq(config.near_min, num_channels, 0)
    config.vmax_overview = libmag.pad_seq(
        config.vmax_overview, num_channels)
    config.cmaps = list(config.process_settings["channel_colors"])
    num_cmaps = len(config.cmaps)
    if num_cmaps < num_channels:
        # add colormap for each remaining channel, purposely inducing 
        # int wraparound for greater color contrast
        chls_diff = num_channels - num_cmaps
        colors = colormaps.discrete_colormap(
            chls_diff, alpha=255, prioritize_default=False, seed=config.seed, 
            min_val=150) / 255.0
        print("generating colormaps from RGBA colors:\n", colors)
        for color in colors:
            config.cmaps.append(colormaps.make_dark_linear_cmap("", color))
