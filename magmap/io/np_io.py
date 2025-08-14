# Numpy archive import/export.
# Author: David Young, 2019, 2020
"""Import/export for Numpy-based archives such as ``.npy`` and ``.npz`` formats.
"""
import os
import pathlib
import pprint
import re
from time import time
from typing import Any, Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Union, \
    cast

import numpy as np
import tifffile

from magmap.atlas import labels_meta, ontology, transformer
from magmap.cv import cv_nd, detector
from magmap.io import importer, libmag, naming, sitk_io
from magmap.plot import colormaps, plot_3d
from magmap.settings import config

if TYPE_CHECKING:
    try:
        from brainglobe_atlasapi import BrainGlobeAtlas
    except ImportError:
        from bg_atlasapi import BrainGlobeAtlas

_logger = config.logger.getChild(__name__)


class Image5d:
    """Main image storage.
    
    Attributes:
        img: 5D Numpy array in the format ``t,z,y,x,c``; defaults to None.
        path_img: Path from which ``img`` was loaded; defaults to None.
        path_meta: Path from which metadata for ``img`` was loaded;
            defaults to None.
        img_io: I/O source for image5d array; defaults to None.
        subimg_offset: Sub-image offset in ``z,y,x``; defaults to None.
        subimg_size: Sub-image size in ``z,y,x``; defaults to None.
        meta: Image metadata dictionary; defaults to None.
    
    """
    def __init__(
            self, img: Optional[np.ndarray] = None,
            path_img: Optional[str] = None,
            path_meta: Optional[str] = None,
            img_io: Optional[config.LoadIO] = None):
        """Construct an Image5d object."""
        # attributes assignable from args
        self.img = img
        self.path_img = path_img
        self.path_meta = path_meta
        self.img_io = img_io
        
        # additional attributes
        self.subimg_offset: Optional[Sequence[int]] = None
        self.subimg_size: Optional[Sequence[int]] = None
        self.meta: Optional[Dict[Union[str, config.MetaKeys], Any]] = None
        #: True if image is RGB(A); defaults to False. 
        self.rgb: bool = False


def img_to_blobs_path(path):
    """Get the blobs path associated with an image or user-supplied.
    
    The user-supplied blobs path stored in :attr:`magmap.io.config.load_data`
    takes precedence over ``path``.
    
    Args:
        path (str): Image base path, without extension or MagellanMapper
            suffixes.

    Returns:
        str: Default MagellanMapper blobs path based on image path, or
        the config path if it is a string.

    """
    path_blobs = config.load_data[config.LoadData.BLOBS]
    if isinstance(path_blobs, str):
        # user-supplied path takes precedence
        return path_blobs
    return libmag.combine_paths(path, config.SUFFIX_BLOBS)


def find_scaling(
        img_path: str, scaled_shape: Optional[Sequence[int]] = None,
        scale: float = None, load_size: Optional[Sequence[int]] = None
) -> Tuple[Sequence[float], Sequence[float]]:
    """Find scaling between two images.
    
    Scaling can be computed to translate blob coordinates into another
    space, such as a downsampled image. These compressed coordinates can be
    used to generate a heat map of blobs.
    
    Args:
        img_path: Base path to image.
        scaled_shape: Shape of image to calculate scaling factor if
            this factor cannot be found from a transposed file's metadata;
            defaults to None.
        scale: Scalar scaling factor, used to find a
            rescaled file; defaults to None. To find a resized file instead,
            set an atlas profile with the resizing factor.
        load_size: Size of image to load in ``x, y, z``, typically given by an
            atlas profile and used to identify the path of the scaled
            image to load; defaults to None.

    Returns:
        Tuple of sequence of scaling factors to a scaled
        or resized image, or None if not loaded or given, and the resolutions
        of the full-sized image found based on ``img_path``.

    """
    # path to image, which may have been resized
    img_path_transposed = transformer.get_transposed_image_path(
        img_path, scale, load_size)
    scaling = None
    res = None
    if scale is not None or load_size is not None:
        # retrieve scaling from a rescaled/resized image
        img_info = importer.read_file(img_path_transposed, config.series).meta
        scaling = img_info["scaling"]
        res = np.multiply(config.resolutions[0], scaling)
        _logger.info("Retrieved scaling from resized image: %s", scaling)
        _logger.info("Rescaled resolution for full-scale image: %s", res)
    
    elif scaled_shape is not None:
        # scale by comparing to original image
        img5d = importer.read_file(img_path_transposed, config.series)
        img5d_shape = None
        if img5d.img is not None:
            # get the shape from the original image
            img5d_shape = img5d.img.shape
        elif img5d.meta is not None:
            # get the shape from the original image's metadata
            img5d_shape = img5d.meta[config.MetaKeys.SHAPE][1:4]
        
        if img5d_shape is not None:
            # find the scaling factor using the original and resized image's
            # shapes 
            scaling = importer.calc_scaling(
                None, None, img5d_shape, scaled_shape)
            res = config.resolutions[0]
            _logger.info("Using scaling compared to full image: %s", scaling)
            _logger.info("Resolution from full-scale image: %s", res)
    
    return scaling, res


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


def setup_images(
        path: str,
        series: Optional[int] = None,
        offset: Optional[Sequence[int]] = None,
        size: Optional[Sequence[int]] = None,
        proc_type: Optional["config.ProcessTypes"] = None,
        allow_import: bool = True,
        fallback_main_img: bool = True,
        bg_atlas: Optional["BrainGlobeAtlas"] = None,
        labels_ref_path: Optional[str] = None):
    """Sets up an image and all associated images and metadata.

    Paths for related files such as registered images will generally be
    constructed from ``path``. If :attr:`config.prefix` is set, it will
    be used in place of ``path`` for registered labels.
    
    Args:
        path: Path to image from which MagellanMapper-style paths will 
            be generated.
        series: Image series number; defaults to None.
        offset: Sub-image offset given in z,y,x; defaults to None.
        size: Sub-image shape given in z,y,x; defaults to None.
        proc_type: Processing type.
        allow_import: True to allow importing the image if it
            cannot be loaded; defaults to True.
        fallback_main_img: True to fall back to loading a registered image
            if possible if the main image could not be loaded; defaults to True.
        bg_atlas: BrainGlobe atlas; defaults to None. If provided, the
            images and labels reference will be extracted from the atlas
            instead of loaded from ``path``.
        labels_ref_path: Path to labels reference file. Defaults to None,
            in which case :att:`config.load_labels` and any loaded labels
            metadata will be used.
    
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
        
        res = import_md[config.MetaKeys.RESOLUTIONS]
        if res is None:
            # default to 1 for x,y,z since image resolutions are required
            res = [1] * 3
            import_md[config.MetaKeys.RESOLUTIONS] = res
            _logger.warn("No image resolutions found. Defaulting to: %s", res)
    
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
    config.labels_img_orig = None
    config.borders_img = None
    config.labels_metadata = None
    config.labels_ref = None
    
    # reset blobs
    config.blobs = None

    filename_base = importer.filename_to_base(path, series)
    subimg_base = None
    blobs = None
    
    # registered images set to load
    atlas_suffix = config.reg_suffixes[config.RegSuffixes.ATLAS]
    annotation_suffix = config.reg_suffixes[config.RegSuffixes.ANNOTATION]
    borders_suffix = config.reg_suffixes[config.RegSuffixes.BORDERS]

    if load_subimage and not config.save_subimg:
        # load a saved sub-image file if available and not set to save one
        subimg_base = naming.make_subimage_name(
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
            config.img5d.subimg_offset = offset
            config.img5d.subimg_size = size
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

    if config.load_data[config.LoadData.BLOBS] or proc_type in (
            config.ProcessTypes.LOAD,
            config.ProcessTypes.COLOC_MATCH,
            config.ProcessTypes.EXPORT_ROIS,
            config.ProcessTypes.EXPORT_BLOBS):
        # load a blobs archive
        blobs = detector.Blobs()
        try:
            if subimg_base:
                try:
                    # load blobs generated from sub-image
                    config.blobs = blobs.load_blobs(
                        img_to_blobs_path(subimg_base))
                except (FileNotFoundError, KeyError):
                    # fallback to loading from full image blobs and getting
                    # a subset, shifting them relative to sub-image offset
                    print("Unable to load blobs file based on {}, will try "
                          "from {}".format(subimg_base, filename_base))
                    config.blobs = blobs.load_blobs(
                        img_to_blobs_path(filename_base))
                    blobs.blobs, _ = detector.get_blobs_in_roi(
                        blobs.blobs, offset, size, reverse=False)
                    detector.Blobs.shift_blob_rel_coords(
                        blobs.blobs, np.multiply(offset, -1))
            else:
                # load full image blobs
                config.blobs = blobs.load_blobs(
                    img_to_blobs_path(filename_base))
        except (FileNotFoundError, KeyError) as e2:
            print("Unable to load blobs file")
            if proc_type in (
                    config.ProcessTypes.LOAD, config.ProcessTypes.EXPORT_BLOBS):
                # blobs expected but not found
                raise e2
    
    if path and config.image5d is None and not atlas_suffix:
        # load or import the main image stack
        print("Loading main image")
        try:
            path_lower = path.lower()
            import_only = proc_type is config.ProcessTypes.IMPORT_ONLY
            if bg_atlas:
                # extract image from BrainGlobeAtlas object
                img5d = Image5d(
                    bg_atlas.reference[None], str(bg_atlas.root_dir),
                    img_io=config.LoadIO.BRAIN_GLOBE)
                config.resolutions = np.array([bg_atlas.resolution])
            
            elif path_lower.endswith(sitk_io.EXTS_3D):
                # load format supported by SimpleITK and prepend time axis;
                # if 2D, convert to 3D
                img5d = sitk_io.read_sitk_files(path, make_3d=True)
            elif not import_only and path_lower.endswith((".tif", ".tiff")):
                # load TIF file directly
                img5d, meta = read_tif(path)
                config.resolutions = meta[config.MetaKeys.RESOLUTIONS]
            else:
                # load or import from MagellanMapper Numpy format
                img5d = None
                if not import_only:
                    # load previously imported image
                    img5d = importer.read_file(path, series)
                if allow_import and (img5d is None or img5d.img is None):
                    # import image; will re-import over any existing image file 
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
                    elif import_only:
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
            _logger.exception(e)
            _logger.info("Could not load %s", path)
    
    if config.metadatas and config.metadatas[0]:
        # assign metadata from alternate file if given to supersede settings
        # for any loaded image5d
        # TODO: access metadata directly from given image5d's dict to allow
        # loading multiple image5d images simultaneously
        importer.assign_metadata(config.metadatas[0])
    
    # main image is currently required since many parameters depend on it
    if fallback_main_img and atlas_suffix is None and config.image5d is None:
        # fallback to atlas if main image not already loaded
        atlas_suffix = config.RegNames.IMG_ATLAS.value
        _logger.info(
            "Main image is not set, falling back to registered image with "
            "suffix %s", atlas_suffix)
    
    # use prefix to get images registered to a different image, eg a
    # downsampled version, or a different version of registered images
    path = config.prefix if config.prefix else path
    if path and atlas_suffix is not None:
        try:
            # will take the place of any previously loaded image5d
            config.img5d = sitk_io.read_sitk_files(
                path, atlas_suffix, make_3d=True)
            config.image5d = config.img5d.img
            
            if config.img5d.img is not None:
                # get near min/max across whole image
                config.near_min, config.near_max = \
                    importer.calc_intensity_bounds(config.img5d.img)
        except FileNotFoundError as e:
            print(e)
    
    # load metadata related to the labels image
    config.labels_metadata = labels_meta.LabelsMeta(
        f"{path}." if config.prefix else path).load()
    
    # load labels reference file
    if bg_atlas:
        # set up labels reference table from BrainGlobe atlas
        if hasattr(bg_atlas, "structures_path"):
            config.labels_ref = ontology.LabelsRef(
                str(bg_atlas.structures_path))
            config.labels_ref.loaded_ref = bg_atlas.lookup_df
            config.labels_ref.create_ref_lookup()
        
    else:
        # load labels reference file
        ref_paths = [
            labels_ref_path,  # given path
            config.load_labels,  # CLI path
            config.labels_metadata.path_ref  # path from metadata
        ]
        ref_paths = [p for p in ref_paths if p is not None]
        labels_ref = None
        for ref in ref_paths:
            try:
                # load labels reference file
                labels_ref = ontology.LabelsRef(ref).load()
                if labels_ref.ref_lookup is not None:
                    config.labels_ref = labels_ref
                    _logger.debug("Loaded labels reference file from %s", ref)
                    break
            except (FileNotFoundError, KeyError):
                pass
        if ref_paths and (labels_ref is None or labels_ref.ref_lookup is None):
            # warn if labels path given but none found
            _logger.warn(
                "Unable to load labels reference file from '%s', skipping",
                ref_paths)

    if annotation_suffix is not None or bg_atlas:
        if bg_atlas:
            # extract labels image from BrainGlobe atlas
            config.labels_img = bg_atlas.annotation
            config.labels_img_sitk = sitk_io.convert_img(config.labels_img)
        else:
            try:
                # load labels image
                # TODO: need to support multichannel labels images
                img5d, config.labels_img_sitk = sitk_io.read_sitk_files(
                    path, annotation_suffix, True, True)
                config.labels_img = img5d.img[0]
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
    
    if borders_suffix is not None:
        # load borders image, which can also be another labels image
        try:
            config.borders_img = sitk_io.read_sitk_files(
                path, borders_suffix, make_3d=True).img[0]
        except FileNotFoundError as e:
            print(e)
    
    if config.atlas_labels[config.AtlasLabels.ORIG_COLORS]:
        labels_orig_ids = config.labels_metadata.region_ids_orig
        if labels_orig_ids is None:
            if config.load_labels is not None:
                # load original labels image from same directory as ontology
                # file for consistent ID-color mapping, even if labels are missing
                try:
                    config.labels_img_orig = sitk_io.load_registered_img(
                        config.load_labels, config.RegNames.IMG_LABELS.value)
                except FileNotFoundError as e:
                    print(e)
            if config.labels_img is not None and config.labels_img_orig is None:
                _logger.warn(
                    "Could not load original labels image IDs; colors may "
                    "differ from the original image")
    
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
        config.img5d.img = config.image5d
        config.image5d_is_roi = True

    # add any additional image5d thresholds for multichannel images, such
    # as those loaded without metadata for these settings
    if not colormaps.CMAPS:
        colormaps.setup_cmaps()
    num_channels = get_num_channels(config.image5d)
    config.near_max = libmag.pad_seq(config.near_max, num_channels, -1)
    config.near_min = libmag.pad_seq(config.near_min, num_channels, 0)
    config.vmax_overview = libmag.pad_seq(
        config.vmax_overview, num_channels)
    colormaps.setup_colormaps(num_channels)
    
    if config.labels_img is not None:
        # make discrete colormap for labels image
        config.cmap_labels = colormaps.setup_labels_cmap(config.labels_img)
    
    if (blobs is not None and blobs.blobs is not None
            and config.img5d.img is not None and blobs.roi_size is not None):
        # scale blob coordinates to main image if shapes differ
        scaling = np.divide(config.img5d.img.shape[1:4], blobs.roi_size)
        # scale radius by mean of other dimensions' scaling
        scaling = np.append(scaling, np.mean(scaling))
        if not np.all(scaling == 1):
            _logger.debug("Scaling blobs to main image by factor: %s", scaling)
            blobs.blobs[:, :4] = ontology.scale_coords(
                blobs.blobs[:, :4], scaling)
        blobs.scaling = scaling
        
        if config.labels_img is not None:
            # assign blobs to regions
            coords = blobs.blobs[:, :3]
            regions = ontology.get_label_ids_from_position(
                coords.astype(int), config.labels_img)
            blobs.format_blobs()
            blobs.set_blob_col(blobs.blobs, blobs.Cols.REGION, regions)


def get_num_channels(img: np.ndarray, is_3d: bool = False) -> int:
    """Get the number of image channels based on expected dimensions.

    Args:
        img: Numpy array.
        is_3d: True if ``img` is a 3D+ array with ``z,y,x[,c]`` order.
            Otherwise, assumes that the image is in 5D (4D+channel),
            `t,z,y,x[,c]` order.

    Returns:
        Inferred number of channels.

    """
    chl_dim = 3 if is_3d else 4
    return 1 if img is None or img.ndim <= chl_dim else img.shape[chl_dim]


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


def read_tif(
        path: str, img5d: Image5d = None
) -> Tuple[Image5d, Dict[config.MetaKeys, Any]]:
    """Read TIF files with Tifffile with lazy access through memory mapping.

    Save this file to NPY format if :attr:`config.savefig` is set to "npy".
    
    Args:
        path: Path to file.
        img5d: Image5d storage class; defaults to None.

    Returns:
        Image5d storage instance and dictionary of extracted metadata.

    """
    if img5d is None:
        # set up a new storage instance
        img5d = Image5d()
    
    # extract metadata
    tif = tifffile.TiffFile(path)
    md = dict.fromkeys(config.MetaKeys)
    axes = tif.series[0].axes.lower()
    _logger.debug("TIF axes: %s", axes)
    nrot = 0
    if tif.ome_metadata:
        # read OME-XML metadata
        names, sizes, md = importer.parse_ome_raw(tif.ome_metadata)
        res = np.array(md[config.MetaKeys.RESOLUTIONS])
        _logger.debug("OME-TIF metadata: %s", tif.ome_metadata)
    else:
        # parse resolutions
        res = np.ones((1, 3))
        if tif.imagej_metadata:
            if config.verbose:
                _logger.debug(
                    "ImageJ TIF metadata:\n%s",
                    pprint.pformat(tif.imagej_metadata))
            
            if "spacing" in tif.imagej_metadata:
                # ImageJ format holds z-resolution as spacing
                res[0, 0] = tif.imagej_metadata["spacing"]
            
            if "Info" in tif.imagej_metadata:
                # extract rotation info
                info = tif.imagej_metadata["Info"]
                rotate = re.search(r"Rotate=[0-9]*", info)
                if rotate:
                    # TODO: use arbitrary degree rotation?
                    nrot = int(rotate.group().split("=")[1]) // -90
                    
        for i, name in enumerate(("YResolution", "XResolution")):
            try:
                # parse x/y-resolution from standard TIF metadata
                axis_res = tif.pages[0].tags[name].value
                if axis_res and len(axis_res) > 1 and axis_res[0]:
                    res[0, i + 1] = axis_res[1] / axis_res[0]
            except KeyError:
                # no resolution found, default to 1
                _logger.info(
                    "No %s tag found in TIF metadata, defaulting to 1", name)
    md[config.MetaKeys.RESOLUTIONS] = res
    
    # load TIFF by memory mapping
    tif_memmap = tifffile.memmap(path)
    ndim = len(tif_memmap.shape)
    if ndim < 4 or ndim == 4 and "c" in axes:
        # add a time dimension for 3D or 3D+C images to ensure TZYX(C) axes
        # TODO: add any time "t" axis is absent?
        tif_memmap = np.expand_dims(tif_memmap, axis=0)
    if "z" not in axes:
        # add a z-axis for 2D images
        tif_memmap = np.expand_dims(tif_memmap, axis=0)
    if axes[0] == "c":
        # move channel dimension to end
        tif_memmap = np.swapaxes(tif_memmap, 2, -1)
    
    if nrot:
        # apply 90 deg rotations; appears to need flipping; both return views
        # TODO: check scenarios requiring flipping
        tif_memmap = cv_nd.rotate90(tif_memmap, nrot, (2, 3), ndim >= 5)
        tif_memmap = tif_memmap[:, :, :, ::-1]
    md[config.MetaKeys.SHAPE] = tif_memmap.shape
    
    if config.verbose:
        _logger.debug("Parsed TIF metadata:\n%s", pprint.pformat(md))
        _logger.debug("Loaded TIF into shape: %s", tif_memmap.shape)
    
    # add image to Image5d instance
    img5d.img = tif_memmap
    img5d.path_img = path
    img5d.img_io = config.LoadIO.TIFFFILE

    if config.savefig and config.savefig.lower() == "npy":
        write_npy(tif_memmap, md, path)
    
    return img5d, md


def write_npy(
        image5d: np.ndarray, md: Dict[Union[str, config.MetaKeys], Any],
        path: Union[str, pathlib.Path], find_near_bounds: bool = True):
    """Write a NumPy array to NPY file.
    
    Args:
        image5d: NumPy array in ``t, z, y, x, c`` dimension order.
        md: Metadata dictionary with keys such as
            :attr:`magmap.settings.config.MetaKeys.RESOLUTIONS`.
        path: Base output path.
        find_near_bounds: True to find near min/max bounds for each channel.
    
    """
    
    # set up filenames for NPY and metadata files
    time_start = time()
    filename_image5d, filename_meta = importer.make_filenames(
        libmag.splitext(path)[0], keep_ext=True)
    _logger.info(
        "Saving image metadata to '%s' and image to '%s'...",
        filename_meta, filename_image5d)
    
    near_mins = []
    near_maxs = []
    if find_near_bounds:
        # find near min/max bounds for each channel
        lows = []
        highs = []
        for img in image5d[0]:
            # near max/min bounds per channel for the given plane
            low, high = importer.calc_intensity_bounds(img, dim_channel=2)
            lows.append(low)
            highs.append(high)
        near_mins, near_maxs = importer.calc_near_intensity_bounds(
            near_mins, near_maxs, lows, highs)
    else:
        near_mins.append(0.)
        near_maxs.append(0.)
    
    # save image5d metadata to file
    importer.save_image_info(
        filename_meta, [os.path.basename(path)], [image5d.shape],
        md[config.MetaKeys.RESOLUTIONS],
        md[config.MetaKeys.MAGNIFICATION],
        md[config.MetaKeys.ZOOM],
        near_mins, near_maxs)
    
    # save image5d to an NPY file using memory mapping
    img_mmap = np.lib.format.open_memmap(
        filename_image5d, mode="w+", dtype=image5d.dtype,
        shape=image5d.shape)
    img_mmap[:] = image5d[:]
    img_mmap.flush()
    _logger.info("...saved image")
    _logger.info("file save time: {}".format(time() - time_start))


def write_tif(
        img5d: "Image5d", path: Union[str, pathlib.Path], **kwargs: Any):
    """Write a NumPy array to TIF files.
    
    Each channel will be exported to a separate file.
    
    Args:
        img5d: ``Image5d`` object.
        path: Base output path. If ``image5d`` has multiple channels, they
            will be exported to files with ``_ch_<n>`` appended just before
            the extension.
        kwargs: Arguments passed to :meth:`tifffile.imwrite`.

    """
    image5d = img5d.img
    if image5d is None or img5d.meta is None:
        _logger.error("No image5d to write to TIF files")
        return
    
    nchls = get_num_channels(image5d)
    for i in range(nchls):
        # export the given channel to a separate file, adding the channel to
        # the filename if multiple channels exist
        img_chl = image5d if image5d.ndim <= 4 else image5d[..., i]
        out_path = pathlib.Path(libmag.make_out_path(
            f"{path}{f'_ch_{i}' if nchls > 1 else ''}.tif",
            combine_prefix=True)).resolve()
        pathlib.Path.mkdir(out_path.parent.resolve(), exist_ok=True)
        libmag.backup_file(out_path)
        
        if "imagej" in kwargs and kwargs["imagej"]:
            # ImageJ format assumes dimension order of TZCYXS
            img_chl = img_chl[:, :, np.newaxis]
            axes = "TZCYXS"
        else:
            # default to TZYXC order, without channel
            axes = "TZYX"
        
        # image metadata, currently only stored in "ImageDescription" tag
        res = img5d.meta.get(config.MetaKeys.RESOLUTIONS, None)
        if res is None:
            res = [1] * 3  # default to 1 for x,y,z
        else:
            res = libmag.pad_seq(res, 3, 1)  # ensure 3D resolution
        res = [float(n) for n in res]
        metadata = {
            "axes": axes,
            "PhysicalSizeZ": res[0],
            "PhysicalSizeY": res[1],
            "PhysicalSizeX": res[2],
        }
        _logger.debug("metadata for TIF: %s", metadata)

        # specify x/y-resolutions for TIF tag, which uses fractions
        res_tif = cast(
            Tuple[float, float], tuple(1 / r for r in res[2:0:-1]))
        _logger.debug("TIF resolutions (fractions): %s", res_tif)
        
        # write to TIF
        _logger.info(
            "Exporting image of shape %s to '%s'", img_chl.shape, out_path)
        tifffile.imwrite(
            out_path, img_chl, resolution=res_tif, metadata=metadata,
            photometric="minisblack", **kwargs)
