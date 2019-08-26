#!/bin/bash
# Image registration
# Author: David Young, 2017, 2019
"""Register images to one another.

The registration type can be given on the command-line (see :mod:``cli``) as 
outlined here. Each type can be coupled with additional arguments in ``cli``.
    * ``single``: Register a single image to a reference atlas. Filenames 
        should be given in :attr:``config.filenames``, where the first 
        filename is the fixed image (eg the experiment), and the 
        second image is the moving image (eg the atlas). The atlas rather 
        than the experimental image is typically moved here since the 
        experimental image is usually much larger and more difficult to move. 
        For faster registration, a downsampled image can be given as this 
        experiment image, and the third argument to ``filenames`` can 
        optionally be given as a name prefix corresponding to the original 
        experiment file. ``flip`` can also be given to flip the 
    * ``group``: Register multiple experimental images to one another via 
        groupwise registration. All the images given in 
        :attr:``config.filenames`` will be registered to one another, except 
        the last filename, which will be used as the output name prefix.
    * ``overlays``: Overlay the moving image on top of the fixed image to 
        visualize alignment. ``flip`` can be given to flip the moving image.
    * ``volumes``: Calculate volumes for each region in the experimental 
        image based on the corresponding registered labels image. ``labels`` 
        should be given to specify the labels file and level. ``rescale`` is 
        used to find the rescaled, typically downsampled images to use as a 
        much faster way to calculate volumes. ``no_show`` suppresses 
        graph display.
    *   ``densities``: Similar to ``volumes`` but include nuclei densities 
        per region by loading the processed stack file.
    *   ``export_vols``: Export regional volumes and related measures to CSV 
        file.
    *   ``export_regions``: Export atlas annotation region information such 
        as region ID to name CSV file and region network SIF file.
    *   ``new_atlas``: Based on "single" registration, outputting registered 
        files in the naming format of a new atlas and bypassing label 
        truncation.
"""

import os
import multiprocessing as mp
import shutil
from time import time

import pandas as pd
import numpy as np
import SimpleITK as sitk
from skimage import filters
from skimage import measure
from skimage import morphology
from skimage import transform

from clrbrain import atlas_refiner
from clrbrain import atlas_stats
from clrbrain import config
from clrbrain import detector
from clrbrain import importer
from clrbrain import lib_clrbrain
from clrbrain import ontology
from clrbrain import plot_2d
from clrbrain import plot_3d
from clrbrain import segmenter
from clrbrain import stats
from clrbrain import transformer
from clrbrain import vols
from clrbrain import export_regions
from clrbrain import sitk_io

SAMPLE_VOLS = "vols_by_sample"
SAMPLE_VOLS_LEVELS = SAMPLE_VOLS + "_levels"
SAMPLE_VOLS_SUMMARY = SAMPLE_VOLS + "_summary"

COMBINED_SUFFIX = "combined"
REREG_SUFFIX = "rereg"

# 3D format extensions to check when finding registered files
_SIGNAL_THRESHOLD = 0.01


def _translation_adjust(orig, transformed, translation, flip=False):
    """Adjust translation based on differences in scaling between original 
    and transformed images to allow the translation to be applied to the 
    original image.
    
    Assumes (x, y, z) order for consistency with SimpleITK since this method 
    operates on SimpleITK format images.
    
    Args:
        orig: Original image in SimpleITK format.
        transformed: Transformed image in SimpleITK format.
        translation: Translation in (x, y, z) order, taken from transform 
            parameters and scaled to the transformed images's spacing.
    
    Returns:
        The adjusted translation in (x, y, z) order.
    """
    if translation is None:
        return translation
    # TODO: need to check which space the TransformParameter is referring to 
    # and how to scale it since the adjusted translation does not appear to 
    # be working yet
    orig_origin = orig.GetOrigin()
    transformed_origin = transformed.GetOrigin()
    origin_diff = np.subtract(transformed_origin, orig_origin)
    print("orig_origin: {}, transformed_origin: {}, origin_diff: {}"
          .format(orig_origin, transformed_origin, origin_diff))
    orig_size = orig.GetSize()
    transformed_size = transformed.GetSize()
    size_ratio = np.divide(orig_size, transformed_size)
    print("orig_size: {}, transformed_size: {}, size_ratio: {}"
          .format(orig_size, transformed_size, size_ratio))
    translation_adj = np.multiply(translation, size_ratio)
    #translation_adj = np.add(translation_adj, origin_diff)
    print("translation_adj: {}".format(translation_adj))
    if flip:
        translation_adj = translation_adj[::-1]
    return translation_adj

def _show_overlays(imgs, translation, fixed_file, plane):
    """Shows overlays via :func:plot_2d:`plot_overlays_reg`.
    
    Args:
        imgs: List of images in Numpy format
        translation: Translation in (z, y, x) format for Numpy consistency.
        fixed_file: Path to fixed file to get title.
        plane: Planar transposition.
    """
    cmaps = ["Blues", "Oranges", "prism"]
    #plot_2d.plot_overlays(imgs, z, cmaps, os.path.basename(fixed_file), aspect)
    show = not config.no_show
    plot_2d.plot_overlays_reg(
        *imgs, *cmaps, translation, os.path.basename(fixed_file), plane, show)

def _handle_transform_file(fixed_file, transform_param_map=None):
    base_name = sitk_io.reg_out_path(fixed_file, "")
    filename = base_name + "transform.txt"
    param_map = None
    if transform_param_map is None:
        param_map = sitk.ReadParameterFile(filename)
    else:
        sitk.WriteParameterFile(transform_param_map[0], filename)
        param_map = transform_param_map[0]
    return param_map, None # TODO: not using translation parameters
    transform = np.array(param_map["TransformParameters"]).astype(np.float)
    spacing = np.array(param_map["Spacing"]).astype(np.float)
    len_spacing = len(spacing)
    #spacing = [16, 16, 20]
    translation = None
    # TODO: should parse the transforms into multiple dimensions
    if len(transform) == len_spacing:
        translation = np.divide(transform[0:len_spacing], spacing)
        print("transform: {}, spacing: {}, translation: {}"
              .format(transform, spacing, translation))
    else:
        print("Transform parameters do not match scaling dimensions")
    return param_map, translation


def smoothing_peak(df, thresh_label_loss=None, filter_size=None):
    """Extract the row of peak smoothing quality from the given data 
    frame matching the given criteria.
    
    Args:
        df: Data frame from which to extract.
        thresh_label_loss: Only check rows below or equal to this 
            fraction of label loss; defaults to None to ignore.
        filter_size: Only rows with the given filter size; defaults 
            to None to ignore.
    
    Returns:
        New data frame with the row having the peak smoothing quality 
        meeting criteria.
    """
    if thresh_label_loss is not None:
        df = df.loc[
            df[config.SmoothingMetrics.LABEL_LOSS.value] <= thresh_label_loss]
    if filter_size is not None:
        df = df.loc[
            df[config.SmoothingMetrics.FILTER_SIZE.value] == filter_size]
    sm_qual = df[config.SmoothingMetrics.SM_QUALITY.value]
    df_peak = df.loc[sm_qual == sm_qual.max()]
    return df_peak


def _load_numpy_to_sitk(numpy_file, rotate=False):
    """Load Numpy image array to SimpleITK Image object.
    
    Args:
        numpy_file: Path to Numpy archive file.
        rotate: True if the image should be rotated 180 deg; defaults to False.
    
    Returns:
        The image in SimpleITK format.
    """
    image5d = importer.read_file(numpy_file, config.series)
    roi = image5d[0, ...] # not using time dimension
    if rotate:
        roi = np.rot90(roi, 2, (1, 2))
    sitk_img = sitk.GetImageFromArray(roi)
    spacing = detector.resolutions[0]
    sitk_img.SetSpacing(spacing[::-1])
    # TODO: consider setting z-origin to 0 since image generally as 
    # tightly bound to subject as possible
    #sitk_img.SetOrigin([0, 0, 0])
    sitk_img.SetOrigin([0, 0, -roi.shape[0] // 2])
    #sitk_img.SetOrigin([0, 0, -roi.shape[0]])
    return sitk_img

def _curate_img(fixed_img, labels_img, imgs=None, inpaint=True, carve=True, 
                holes_area=None):
    """Curate images by in-painting where corresponding pixels are present in 
    fixed image but not labels or other images and removing pixels 
    present in those images but not the fixed image.
    
    Args:
        fixed_img: Image in SimpleITK format by which to curate other images.
        labels_img: Labels image in SimpleITK format, used to determine 
            missing pixels and measure overlap.
        imgs: Array of additional images to curate corresponding pixels 
            as those curated in ``labels_img``. Defaults to None.
        inpaint: True to in-paint; defaults to True.
        carve: True to carve based on ``fixed_img``; defaults to True. If 
            ``inpaint`` is True, ``carve`` should typically be True as well.
        holes_area: Maximum area of holes to fill when carving.
    
    Returns:
        A list of images in SimpleITK format that have been curated.
    """
    fixed_img_np = sitk.GetArrayFromImage(fixed_img)
    labels_img_np = sitk.GetArrayFromImage(labels_img)
    result_img = labels_img
    result_img_np = labels_img_np
    # ensure that labels image is first
    if imgs:
        imgs.insert(0, labels_img)
    else:
        imgs = [labels_img]
    
    # mask image showing where result is 0 but fixed image is above thresh 
    # to fill in with nearest neighbors
    thresh = filters.threshold_mean(fixed_img_np)
    print("thresh: {}".format(thresh))
    # fill only empty regions corresponding to filled pixels, but fills 
    # some with 0 from dist transform pointing to appropriately empty pixels
    #to_fill = np.logical_and(labels_img_np == 0, fixed_img_np > thresh)
    to_fill = labels_img_np == 0
    
    result_imgs = []
    for i in range(len(imgs)):
        # in-paint and remove pixels from result image where fixed image is 
        # below threshold
        img = imgs[i]
        result_img_np = sitk.GetArrayFromImage(img)
        if inpaint:
            result_img_np = plot_3d.in_paint(result_img_np, to_fill)
        if carve:
            _, mask = plot_3d.carve(fixed_img_np, thresh, holes_area)
            result_img_np[~mask] = 0
        result_img = sitk_io.replace_sitk_with_numpy(img, result_img_np)
        result_imgs.append(result_img)
        if i == 0:
            # check overlap based on labels images; should be 1.0 by def
            result_img_np[result_img_np != 0] = 2
            result_img_for_overlap = sitk_io.replace_sitk_with_numpy(
                img, result_img_np)
            atlas_refiner.measure_overlap(
                fixed_img, result_img_for_overlap, transformed_thresh=1)
    return result_imgs

def _transform_labels(transformix_img_filter, labels_img, truncation=None, 
                      flip=False):
    if truncation is not None:
        # truncate ventral and posterior portions since variable 
        # amount of tissue or quality of imaging in these regions
        labels_img_np = sitk.GetArrayFromImage(labels_img)
        truncation = list(truncation)
        if flip:
            # assume labels were rotated 180deg around the z-axis, so 
            # need to flip y-axis fracs
            truncation[1] = np.subtract(1, truncation[1])[::-1]
        atlas_refiner.truncate_labels(labels_img_np, *truncation)
        labels_img = sitk_io.replace_sitk_with_numpy(labels_img, labels_img_np)
    
    # apply atlas transformation to labels image
    labels_pixel_id = labels_img.GetPixelID() # now as signed int
    print("labels_pixel type: {}".format(labels_img.GetPixelIDTypeAsString()))
    transformix_img_filter.SetMovingImage(labels_img)
    transformix_img_filter.Execute()
    transformed_labels_img = transformix_img_filter.GetResultImage()
    transformed_labels_img = sitk.Cast(transformed_labels_img, labels_pixel_id)
    '''
    # remove bottom planes after registration; make sure to turn off 
    # bottom plane removal in mirror step
    img_np = sitk.GetArrayFromImage(labels_img)
    atlas_refiner._truncate_labels(img_np, z_frac=(0.2, 1.0))
    labels_img = sitk_io.replace_sitk_with_numpy(labels_img, img_np)
    '''
    print(transformed_labels_img)
    '''
    LabelStatistics = sitk.LabelStatisticsImageFilter()
    LabelStatistics.Execute(fixed_img, labels_img)
    count = LabelStatistics.GetCount(1)
    mean = LabelStatistics.GetMean(1)
    variance = LabelStatistics.GetVariance(1)
    print("count: {}, mean: {}, variance: {}".format(count, mean, variance))
    '''
    return transformed_labels_img

def _config_reg_resolutions(grid_spacing_schedule, param_map, ndim):
    if grid_spacing_schedule:
        # assume spacing given as single val for all dimensions
        param_map["GridSpacingSchedule"] = grid_spacing_schedule
        num_res = len(grid_spacing_schedule)
        res_set = grid_spacing_schedule[:ndim]
        if len(np.unique(res_set)) != ndim:
            # any repeated resolutions must mean that each value refers to 
            # a dimension rather than to all dimensions; note that schedules 
            # with completely different values per dimension won't be 
            # identified correctly
            num_res /= ndim
        param_map["NumberOfResolutions"] = [str(num_res)]


def register_duo(fixed_img, moving_img, path=None):
    """Register two images to one another using ``SimpleElastix``.
    
    Args:
        fixed_img: The image to be registered to in :class``SimpleITK.Image`` 
            format.
        moving_img: The image to register to ``fixed_img`` in 
            :class``SimpleITK.Image`` format.
        path: Path as string from whose parent directory the points-based
            registration files ``fix_pts.txt`` and ``mov_pts.txt`` will
            be found; defaults to None, in which case points-based
            reg will be ignored even if set.
    
    Returns:
        Tuple of the registered image in SimpleITK Image format and a 
        Transformix filter with the registration's parameters to 
        reapply them on other images.
    """
    # basic info from images just prior to SimpleElastix filtering for 
    # registration; to view raw images, show these images rather than merely 
    # turning all iterations to 0 since simply running through the filter 
    # will alter images
    print("fixed image (type {}):\n{}".format(
        fixed_img.GetPixelIDTypeAsString(), fixed_img))
    print("moving image (type {}):\n{}".format(
        moving_img.GetPixelIDTypeAsString(), moving_img))
    
    # set up SimpleElastix filter
    elastix_img_filter = sitk.ElastixImageFilter()
    elastix_img_filter.SetFixedImage(fixed_img)
    elastix_img_filter.SetMovingImage(moving_img)
    
    # set up parameter maps for translation, affine, and deformable regs
    settings = config.register_settings
    metric_sim = settings["metric_similarity"]
    param_map_vector = sitk.VectorOfParameterMap()
    
    # translation to shift and rotate
    param_map = sitk.GetDefaultParameterMap("translation")
    param_map["Metric"] = [metric_sim]
    param_map["MaximumNumberOfIterations"] = [settings["translation_iter_max"]]
    '''
    # TESTING: minimal registration
    param_map["MaximumNumberOfIterations"] = ["0"]
    '''
    param_map_vector.append(param_map)
    
    # affine to sheer and scale
    param_map = sitk.GetDefaultParameterMap("affine")
    param_map["Metric"] = [metric_sim]
    param_map["MaximumNumberOfIterations"] = [settings["affine_iter_max"]]
    param_map_vector.append(param_map)
    
    # bspline for non-rigid deformation
    param_map = sitk.GetDefaultParameterMap("bspline")
    param_map["Metric"] = [metric_sim, *param_map["Metric"][1:]]
    param_map["FinalGridSpacingInVoxels"] = [
        settings["bspline_grid_space_voxels"]]
    del param_map["FinalGridSpacingInPhysicalUnits"] # avoid conflict with vox
    param_map["MaximumNumberOfIterations"] = [settings["bspline_iter_max"]]
    _config_reg_resolutions(
        settings["grid_spacing_schedule"], param_map, fixed_img.GetDimension())
    if path is not None and settings["point_based"]:
        # point-based registration added to b-spline, which takes point sets 
        # found in name_prefix's folder; note that coordinates are from the 
        # originally set fixed and moving images, not after transformation up 
        # to this point
        fix_pts_path = os.path.join(os.path.dirname(path), "fix_pts.txt")
        move_pts_path = os.path.join(
            os.path.dirname(path), "mov_pts.txt")
        if os.path.isfile(fix_pts_path) and os.path.isfile(move_pts_path):
            metric = list(param_map["Metric"])
            metric.append("CorrespondingPointsEuclideanDistanceMetric")
            param_map["Metric"] = metric
            #param_map["Metric2Weight"] = ["0.5"]
            elastix_img_filter.SetFixedPointSetFileName(fix_pts_path)
            elastix_img_filter.SetMovingPointSetFileName(move_pts_path)
    
    param_map_vector.append(param_map)
    elastix_img_filter.SetParameterMap(param_map_vector)
    elastix_img_filter.PrintParameterMap()
    transform = elastix_img_filter.Execute()
    transformed_img = elastix_img_filter.GetResultImage()
    
    # prep filter to apply transformation to label files
    transform_param_map = elastix_img_filter.GetTransformParameterMap()
    transformix_img_filter = sitk.TransformixImageFilter()
    transformix_img_filter.SetTransformParameterMap(transform_param_map)
    
    return transformed_img, transformix_img_filter

def register(fixed_file, moving_file_dir, plane=None, flip=False, 
             show_imgs=True, write_imgs=True, name_prefix=None, 
             new_atlas=False):
    """Register an atlas and associated labels to a sample image 
    using the SimpleElastix library.
    
    Args:
        fixed_file: The image to register, given as a Numpy archive file to 
            be read by :importer:`read_file`.
        moving_file_dir: Directory of the atlas images, including the 
            main image and labels. The atlas was chosen as the moving file
            since it is likely to be lower resolution than the Numpy file.
        plane: Planar orientation to which the atlas will be transposed, 
            considering the atlas' original plane as "xy". Defaults to 
            None to avoid planar transposition.
        flip: True if the moving files (does not apply to fixed file) should 
            be flipped/rotated; defaults to False.
        show_imgs: True if the output images should be displayed; defaults to 
            True.
        write_imgs: True if the images should be written to file; defaults to 
            False.
        name_prefix: Path with base name where registered files will be output; 
            defaults to None, in which case the fixed_file path will be used.
        new_atlas: True to generate registered images that will serve as a 
            new atlas; defaults to False.
    """
    start_time = time()
    if name_prefix is None:
        name_prefix = fixed_file
    settings = config.register_settings
    
    # load fixed image, assumed to be experimental image
    fixed_img = _load_numpy_to_sitk(fixed_file)
    
    # preprocessing; store original fixed image for overlap measure
    fixed_img_orig = fixed_img
    if settings["preprocess"]:
        img_np = sitk.GetArrayFromImage(fixed_img)
        #img_np = plot_3d.saturate_roi(img_np)
        img_np = plot_3d.denoise_roi(img_np)
        fixed_img = sitk_io.replace_sitk_with_numpy(fixed_img, img_np)
    fixed_img_size = fixed_img.GetSize()
    
    # load moving image, assumed to be atlas
    moving_file = os.path.join(moving_file_dir, config.RegNames.IMG_ATLAS.value)
    moving_img = sitk.ReadImage(moving_file)
    
    # load labels image and match with atlas
    labels_img = sitk.ReadImage(os.path.join(
        moving_file_dir, config.RegNames.IMG_LABELS.value))
    moving_img, labels_img, _, _ = atlas_refiner.match_atlas_labels(
        moving_img, labels_img, flip)
    
    transformed_img, transformix_img_filter = register_duo(
        fixed_img, moving_img, name_prefix)
    # turn off to avoid overshooting the interpolation for the labels image 
    # (see Elastix manual section 4.3)
    transform_param_map = transformix_img_filter.GetTransformParameterMap()
    transform_param_map[-1]["FinalBSplineInterpolationOrder"] = ["0"]
    transformix_img_filter.SetTransformParameterMap(transform_param_map)
    
    # overlap stats comparing original and registered samples (eg histology)
    print("DSC of original and registered sample images")
    dsc_sample = atlas_refiner.measure_overlap(
        fixed_img_orig, transformed_img, 
        transformed_thresh=settings["atlas_threshold"])
    
    def make_labels(truncate):
        nonlocal transformed_img
        truncation = settings["truncate_labels"] if truncate else None
        img = _transform_labels(
            transformix_img_filter, labels_img, truncation=truncation, 
            flip=flip)
        print(img.GetSpacing())
        # WORKAROUND: labels img floating point vals may be more rounded 
        # than transformed moving img for some reason; assume transformed 
        # labels and moving image should match exactly, so replace labels 
        # with moving image's transformed spacing
        img.SetSpacing(transformed_img.GetSpacing())
        print(img.GetSpacing())
        print(fixed_img_orig.GetSpacing(), transformed_img.GetSpacing())
        dsc = None
        if settings["curate"]:
            img, transformed_img = _curate_img(
                fixed_img_orig, img, imgs=[transformed_img], inpaint=new_atlas)
            print("DSC of original and registered sample images after curation")
            dsc = atlas_refiner.measure_overlap(
                fixed_img_orig, transformed_img, 
                transformed_thresh=settings["atlas_threshold"])
        return img, dsc
    
    labels_img_full, dsc_sample_curated = make_labels(False)
    labels_img, _ = labels_img_full if new_atlas else make_labels(True)
    
    imgs_write = (fixed_img, transformed_img, labels_img_full, labels_img)
    if show_imgs:
        # show individual SimpleITK images in default viewer
        for img in imgs_write: sitk.Show(img)
    
    if write_imgs:
        # write atlas and labels files, transposed according to plane setting
        if new_atlas:
            imgs_names = (
                config.RegNames.IMG_ATLAS.value, 
                config.RegNames.IMG_LABELS.value)
            imgs_write = [transformed_img, labels_img]
        else:
            imgs_names = (
                config.RegNames.IMG_EXP.value, 
                config.RegNames.IMG_ATLAS.value, 
                config.RegNames.IMG_LABELS.value, 
                config.RegNames.IMG_LABELS_TRUNC.value)
        for i in range(len(imgs_write)):
            out_path = imgs_names[i]
            if new_atlas:
                out_path = os.path.join(os.path.dirname(name_prefix), out_path)
            else:
                out_path = sitk_io.reg_out_path(name_prefix, out_path)
            print("writing {}".format(out_path))
            sitk.WriteImage(imgs_write[i], out_path, False)

    # save transform parameters and attempt to find the original position 
    # that corresponds to the final position that will be displayed
    _, translation = _handle_transform_file(name_prefix, transform_param_map)
    translation = _translation_adjust(
        moving_img, transformed_img, translation, flip=True)
    
    # compare original atlas with registered labels taken as a whole
    dsc_labels = atlas_refiner.measure_overlap_combined_labels(
        fixed_img_orig, labels_img_full)
    
    # measure compactness of fixed image
    fixed_img_orig_np = sitk.GetArrayFromImage(fixed_img_orig)
    thresh_atlas = fixed_img_orig_np > filters.threshold_mean(fixed_img_orig_np)
    compactness = plot_3d.compactness(
        plot_3d.perimeter_nd(thresh_atlas), thresh_atlas)
    
    # save basic metrics in CSV file
    basename = lib_clrbrain.get_filename_without_ext(fixed_file)
    metrics = {
        config.AtlasMetrics.SAMPLE: [basename], 
        config.AtlasMetrics.REGION: config.REGION_ALL, 
        config.AtlasMetrics.CONDITION: [np.nan], 
        config.AtlasMetrics.DSC_ATLAS_SAMPLE: [dsc_sample], 
        config.AtlasMetrics.DSC_ATLAS_SAMPLE_CUR: [dsc_sample_curated], 
        config.AtlasMetrics.DSC_ATLAS_LABELS: [dsc_labels], 
        config.SmoothingMetrics.COMPACTNESS: [compactness]
    }
    df_path = lib_clrbrain.combine_paths(
        name_prefix, config.PATH_ATLAS_IMPORT_METRICS)
    print("\nImported {} whole atlas stats:".format(basename))
    stats.dict_to_data_frame(metrics, df_path, show="\t")

    '''
    # show 2D overlays or registered image and atlas last since blocks until 
    # fig is closed
    imgs = [
        sitk.GetArrayFromImage(fixed_img),
        sitk.GetArrayFromImage(moving_img), 
        sitk.GetArrayFromImage(transformed_img), 
        sitk.GetArrayFromImage(labels_img)]
    _show_overlays(imgs, translation, fixed_file, None)
    '''
    print("time elapsed for single registration (s): {}"
          .format(time() - start_time))

def register_reg(fixed_path, moving_path, reg_base=None, reg_names=None, 
                 plane=None, flip=False, prefix=None, suffix=None, show=True):
    """Using registered images including the unregistered copies of 
    the original image, register these images to another image.
    
    For example, registered images can be registered back to the atlas. 
    This method can also be used to move unregistered original images 
    that have simply been copied as ``config.RegNames.IMG_EXP.value`` 
    during registration. 
    This copy can be registered "back" to the atlas, reversing the 
    fixed/moving images in :meth:``register`` to move all experimental 
    images into the same space.
    
    Args:
        fixed_path: Path to he image to be registered to in 
            :class``SimpleITK.Image`` format.
        moving_path: Path to the image in :class``SimpleITK.Image`` format 
            to register to the image at ``fixed_path``.
        reg_base: Registration suffix to combine with ``moving_path``. 
            Defaults to None to use ``moving_path`` as-is, with 
            output name based on :const:``config.RegNames.IMG_EXP.value``.
        reg_names: List of additional registration suffixes associated 
            with ``moving_path`` to be registered using the same 
            transformation. Defaults to None.
        plane: Planar orientation to which the atlas will be transposed, 
            considering the atlas' original plane as "xy". Defaults to 
            None to avoid planar transposition.
        flip: True if the moving files (does not apply to fixed file) should 
            be flipped/rotated; defaults to False.
        prefix: Base path to use for output; defaults to None to 
            use ``moving_path`` instead.
        suffix: String to combine with ``moving_path`` to load images; 
            defaults to None.
        show: True to show images after registration; defaults to True.
    """
    fixed_img = sitk.ReadImage(fixed_path)
    mod_path = moving_path
    if suffix is not None:
        # adjust image path to load with suffix
        mod_path = lib_clrbrain.insert_before_ext(mod_path, suffix)
    if reg_base is None:
        # load the image directly from given path
        moving_img = sitk.ReadImage(mod_path)
    else:
        # treat the path as a base path to which a reg suffix will be combined
        moving_img = sitk_io.load_registered_img(
            mod_path, reg_base, get_sitk=True)
    
    # register the images and apply the transformation to any 
    # additional images previously registered to the moving path
    rotate = 2 if flip else 0
    moving_img = atlas_refiner.transpose_img(moving_img, plane, rotate)
    transformed_img, transformix_img_filter = register_duo(
        fixed_img, moving_img, prefix)
    reg_imgs = [transformed_img]
    names = [config.RegNames.IMG_EXP.value if reg_base is None else reg_base]
    if reg_names is not None:
        for reg_name in reg_names:
            img = sitk_io.load_registered_img(mod_path, reg_name, get_sitk=True)
            img = atlas_refiner.transpose_img(img, plane, rotate)
            transformix_img_filter.SetMovingImage(img)
            transformix_img_filter.Execute()
            img_result = transformix_img_filter.GetResultImage()
            reg_imgs.append(img_result)
            names.append(reg_name)
    
    # use prefix as base output path if given and append distiguishing string 
    # to differentiate from original files
    output_base = lib_clrbrain.insert_before_ext(
        moving_path if prefix is None else prefix, REREG_SUFFIX, "_")
    if suffix is not None:
        output_base = lib_clrbrain.insert_before_ext(output_base, suffix)
    imgs_write = {}
    for name, img in zip(names, reg_imgs):
        # use the same reg suffixes, assuming that output_base will give a 
        # distinct name to avoid overwriting previously registered images
        imgs_write[name] = img
    sitk_io.write_reg_images(imgs_write, output_base)
    if show:
        for img in imgs_write.values(): sitk.Show(img)


def _crop_image(img_np, labels_img, axis, eraser=None):
    """Crop image by removing the empty space at the start of the given axis.
    
    Args:
        img_np: 3D image array in Numpy format.
        labels_img: 3D image array in Numpy format of the same shape as 
            ``img_np``, typically a labels image from which to find the empty 
            region to crop.
        axis: Axis along which to crop.
        eraser: Erase rather than crop, changing pixels that would have been 
            cropped to the given intensity value instead; defaults to None.
    
    Returns:
        Tuple of ``img_crop, i``, where ``img_crop is the cropped image with 
        planes removed along the start of the given axis until the first 
        non-empty plane is reached, or erased if ``eraser`` is given, and 
        ``i`` is the index of the first non-cropped/erased plane.
    """
    # find the first non-zero plane in the labels image along the given axis, 
    # expanding slices to the include the rest of the image; 
    # TODO: consider using mask from given region in labels to zero out 
    # corresponding region in img_np
    slices = [slice(None)] * labels_img.ndim
    shape = labels_img.shape
    for i in range(shape[axis]):
        slices[axis] = i
        plane = labels_img[tuple(slices)]
        if not np.allclose(plane, 0):
            print("found first non-zero plane at i of {}".format(i))
            break
    
    # crop image if a region of empty planes is found at the start of the axis
    img_crop = img_np
    if i < shape[axis]:
        slices = [slice(None)] * img_np.ndim
        if eraser is None:
            slices[axis] = slice(i, shape[axis])
            img_crop = img_crop[tuple(slices)]
            print("cropped image from shape {} to {}"
                  .format(shape, img_crop.shape))
        else:
            slices[axis] = slice(0, i)
            img_crop[tuple(slices)] = eraser
            print("erased image outside of {} to {} intensity value"
                  .format(slices, eraser))
    else:
        print("could not find non-empty plane at which to crop")
    return img_crop, i

def register_group(img_files, flip=None, show_imgs=True, 
             write_imgs=True, name_prefix=None, scale=None):
    """Group registers several images to one another.
    
    Args:
        img_files: Paths to image files to register.
        flip: Boolean list corresponding to ``img_files`` flagging 
            whether to flip the image or not; defaults to None, in which 
            case no images will be flipped.
        show_imgs: True if the output images should be displayed; defaults to 
            True.
        write_imgs: True if the images should be written to file; defaults to 
            True.
        name_prefix: Path with base name where registered files will be output; 
            defaults to None, in which case the fixed_file path will be used.
        scale: Rescaling factor as a scalar value, used to find the rescaled, 
            smaller images corresponding to ``img_files``. Defaults to None.
    """
    start_time = time()
    if name_prefix is None:
        name_prefix = img_files[0]
    target_size = config.register_settings["target_size"]
    
    '''
    # TESTING: assuming first file is a raw groupwise registered image, 
    # import it for post-processing
    img = sitk.ReadImage(img_files[0])
    img_np = sitk.GetArrayFromImage(img)
    print("thresh mean: {}".format(filters.threshold_mean(img_np)))
    carve_threshold = config.register_settings["carve_threshold"]
    holes_area = config.register_settings["holes_area"]
    img_np, img_np_unfilled = plot_3d.carve(
        img_np, thresh=carve_threshold, holes_area=holes_area, 
        return_unfilled=True)
    sitk.Show(sitk_io.replace_sitk_with_numpy(img, img_np_unfilled))
    sitk.Show(sitk_io.replace_sitk_with_numpy(img, img_np))
    return
    '''
    
    img_vector = sitk.VectorOfImage()
    flip_img = False
    # image properties of 1st image, in SimpleITK format
    origin = None
    size_orig = None
    size_cropped = None
    start_y = None
    spacing = None
    img_np_template = None # first image, used as template for rest
    for i in range(len(img_files)):
        # load image, fipping if necessary and using tranpsosed img if specified
        img_file = img_files[i]
        img_file = transformer.get_transposed_image_path(
            img_file, scale, target_size)
        if flip is not None:
            flip_img = flip[i]
        img = _load_numpy_to_sitk(img_file, flip_img)
        size = img.GetSize()
        img_np = sitk.GetArrayFromImage(img)
        if img_np_template is None:
            img_np_template = np.copy(img_np)
        
        # crop y-axis based on registered labels to ensure that sample images 
        # have the same structures since variable amount of tissue posteriorly; 
        # cropping appears to work better than erasing for groupwise reg, 
        # preventing some images from being stretched into the erased space
        labels_img = sitk_io.load_registered_img(
            img_files[i], config.RegNames.IMG_LABELS_TRUNC.value)
        img_np, y_cropped = _crop_image(img_np, labels_img, 1)#, eraser=0)
        '''
        # crop anterior region
        rotated = np.rot90(img_np, 2, (1, 2))
        rotated, _ = _crop_image(rotated, np.rot90(labels_img, 2, (1, 2)), 1)
        img_np = np.rot90(rotated, 2, (1, 2))
        '''
        
        # force all images into same size and origin as first image 
        # to avoid groupwise registration error on physical space mismatch
        if size_cropped is not None:
            # use default interpolation, but should change to nearest neighbor 
            # if using for labels
            img_np = transform.resize(
                img_np, size_cropped[::-1], anti_aliasing=True, mode="reflect")
            print(img_file, img_np.shape)
        img = sitk_io.replace_sitk_with_numpy(img, img_np)
        if origin is None:
            origin = img.GetOrigin()
            size_orig = size
            size_cropped = img.GetSize()
            spacing = img.GetSpacing()
            start_y = y_cropped
            print("size_cropped: ", size_cropped, ", size_orig", size_orig)
        else:
            # force images into space of first image; may not be exactly 
            # correct but should be close since resized to match first image, 
            # and spacing of resized images and atlases largely ignored in 
            # favor of comparing shapes of large original and registered images
            img.SetOrigin(origin)
            img.SetSpacing(spacing)
        print("img_file: {}\n{}".format(img_file, img))
        img_vector.push_back(img)
        #sitk.Show(img)
    #sitk.ProcessObject.SetGlobalDefaultDirectionTolerance(1)
    #sitk.ProcessObject.SetGlobalDefaultCoordinateTolerance(100)
    img_combined = sitk.JoinSeries(img_vector)
    
    settings = config.register_settings
    elastix_img_filter = sitk.ElastixImageFilter()
    elastix_img_filter.SetFixedImage(img_combined)
    elastix_img_filter.SetMovingImage(img_combined)
    param_map = sitk.GetDefaultParameterMap("groupwise")
    param_map["FinalGridSpacingInVoxels"] = [
        settings["bspline_grid_space_voxels"]]
    del param_map["FinalGridSpacingInPhysicalUnits"] # avoid conflict with vox
    param_map["MaximumNumberOfIterations"] = [settings["groupwise_iter_max"]]
    # TESTING:
    #param_map["MaximumNumberOfIterations"] = ["0"]
    _config_reg_resolutions(
        settings["grid_spacing_schedule"], param_map, img_np_template.ndim)
    elastix_img_filter.SetParameterMap(param_map)
    elastix_img_filter.PrintParameterMap()
    transform_filter = elastix_img_filter.Execute()
    transformed_img = elastix_img_filter.GetResultImage()
    
    # extract individual 3D images from 4D result image
    extract_filter = sitk.ExtractImageFilter()
    size = list(transformed_img.GetSize())
    size[3] = 0 # set t to 0 to collapse this dimension
    extract_filter.SetSize(size)
    imgs = []
    num_images = len(img_files)
    for i in range(num_images):
        extract_filter.SetIndex([0, 0, 0, i]) # x, y, z, t
        img = extract_filter.Execute(transformed_img)
        img_np = sitk.GetArrayFromImage(img)
        # resize to original shape of first image, all aligned to position 
        # of subject within first image
        img_large_np = np.zeros(size_orig[::-1])
        img_large_np[:, start_y:start_y+img_np.shape[1]] = img_np
        if show_imgs:
            sitk.Show(sitk_io.replace_sitk_with_numpy(img, img_large_np))
        imgs.append(img_large_np)
    
    # combine all images by taking their mean
    img_mean = np.mean(imgs, axis=0)
    extend_borders = settings["extend_borders"]
    carve_threshold = settings["carve_threshold"]
    if extend_borders and carve_threshold:
        # merge in specified border region from first image for pixels below 
        # carving threshold to prioritize groupwise image
        slices = []
        for border in extend_borders[::-1]:
            slices.append(slice(*border) if border else slice(None))
        slices = tuple(slices)
        region = img_mean[tuple(slices)]
        region_template = img_np_template[tuple(slices)]
        mask = region < carve_threshold
        region[mask] = region_template[mask]
    img_raw = sitk_io.replace_sitk_with_numpy(transformed_img, img_mean)
    
    # carve groupwise registered image if given thresholds
    imgs_to_show = []
    imgs_to_show.append(img_raw)
    holes_area = settings["holes_area"]
    if carve_threshold and holes_area:
        img_mean, _, img_mean_unfilled = plot_3d.carve(
            img_mean, thresh=carve_threshold, holes_area=holes_area, 
            return_unfilled=True)
        img_unfilled = sitk_io.replace_sitk_with_numpy(
            transformed_img, img_mean_unfilled)
        transformed_img = sitk_io.replace_sitk_with_numpy(
            transformed_img, img_mean)
        # will show unfilled and filled in addition to raw image
        imgs_to_show.append(img_unfilled)
        imgs_to_show.append(transformed_img)
    
    if show_imgs:
        for img in imgs_to_show: sitk.Show(img)
    
    #transformed_img = img_raw
    if write_imgs:
        # write both the .mhd and Numpy array files to a separate folder to 
        # mimic the atlas folder format
        out_path = os.path.join(name_prefix, config.RegNames.IMG_GROUPED.value)
        if not os.path.exists(name_prefix):
            os.makedirs(name_prefix)
        print("writing {}".format(out_path))
        sitk.WriteImage(transformed_img, out_path, False)
        img_np = sitk.GetArrayFromImage(transformed_img)
        detector.resolutions = [transformed_img.GetSpacing()[::-1]]
        importer.save_np_image(img_np[None], out_path, config.series)
    
    print("time elapsed for groupwise registration (s): {}"
          .format(time() - start_time))

def register_labels_to_atlas(path_fixed):
    """Register annotation labels to its own atlas.
    
    Labels may not match their own underlying atlas image well, 
    particularly in the orthogonal directions in which the labels 
    were not constructed. To improve alignment between the labels 
    and the atlas itself, register the edge-detected versions of the 
    labels and its atlas.
    
    Edge files are assumed to have been generated by 
    :func:``make_edge_images``.
    
    Args:
        path_fixed: Path to the fixed file, typically the atlas file 
            with stained sections. The corresponding edge and labels 
            files will be loaded based on this path.
    """
    # load corresponding edge files
    fixed_sitk = sitk_io.load_registered_img(
        path_fixed, config.RegNames.IMG_ATLAS_EDGE.value, get_sitk=True)
    moving_sitk = sitk_io.load_registered_img(
        path_fixed, config.RegNames.IMG_LABELS_EDGE.value, get_sitk=True)
    
    # set up SimpleElastix filter
    elastix_img_filter = sitk.ElastixImageFilter()
    elastix_img_filter.SetFixedImage(fixed_sitk)
    elastix_img_filter.SetMovingImage(moving_sitk)
    
    # set up registration parameters
    settings = config.register_settings
    param_map_vector = sitk.VectorOfParameterMap()
    # bspline for non-rigid deformation
    param_map = sitk.GetDefaultParameterMap("bspline")
    param_map["FinalGridSpacingInVoxels"] = [
        settings["bspline_grid_space_voxels"]]
    del param_map["FinalGridSpacingInPhysicalUnits"] # avoid conflict with vox
    param_map["MaximumNumberOfIterations"] = [settings["bspline_iter_max"]]
    #param_map["MaximumNumberOfIterations"] = "0"
    _config_reg_resolutions(
        settings["grid_spacing_schedule"], param_map, fixed_sitk.GetDimension())
    param_map_vector.append(param_map)
    elastix_img_filter.SetParameterMap(param_map_vector)
    elastix_img_filter.PrintParameterMap()
    
    # perform the registration
    transform = elastix_img_filter.Execute()
    transformed_img = elastix_img_filter.GetResultImage()
    sitk.Show(transformed_img)
    
    # set up filter to apply the same transformation to label file, 
    # ensuring that no new labels are interpolated
    transform_param_map = elastix_img_filter.GetTransformParameterMap()
    transform_param_map[-1]["FinalBSplineInterpolationOrder"] = ["0"]
    transformix_img_filter = sitk.TransformixImageFilter()
    transformix_img_filter.SetTransformParameterMap(transform_param_map)
    
    # apply transformation, manually resetting spacing in case of rounding
    labels_sitk = sitk_io.load_registered_img(
        path_fixed, config.RegNames.IMG_LABELS.value, get_sitk=True)
    transformed_labels = _transform_labels(transformix_img_filter, labels_sitk)
    transformed_labels.SetSpacing(transformed_img.GetSpacing())
    #sitk.Show(transformed_labels)
    
    # write transformed labels file
    out_path_base = os.path.splitext(path_fixed)[0] + "_edgereg.mhd"
    imgs_write = {
        config.RegNames.IMG_LABELS.value: transformed_labels}
    sitk_io.write_reg_images(imgs_write, out_path_base)
    # copy original atlas metadata file to allow opening this atlas 
    # alongside new labels image for comparison
    shutil.copy(
        sitk_io.reg_out_path(path_fixed, config.RegNames.IMG_ATLAS.value), 
        out_path_base)

def _mirror_imported_labels(labels_img_np, start):
    # mirror labels that have been imported and transformed with x and z swapped
    labels_img_np = atlas_refiner.mirror_planes(
        np.swapaxes(labels_img_np, 0, 2), start, mirror_mult=-1, 
        check_equality=True)
    labels_img_np = np.swapaxes(labels_img_np, 0, 2)
    return labels_img_np

def _is_profile_mirrored():
    # check if profile is set for mirroring, though does not necessarily
    # mean that the image itself is mirrored; allows checking for 
    # simplification by operating on one half and mirroring to the other
    return (config.register_settings["extend_labels"]["mirror"] and
            config.register_settings["labels_mirror"] is not None)

def make_edge_images(path_img, show=True, atlas=True, suffix=None, 
                     path_atlas_dir=None):
    """Make edge-detected atlas and associated labels images.
    
    The atlas is assumed to be a sample (eg microscopy) image on which 
    an edge-detection filter will be applied. The labels image is 
    assumed to be an annotated image whose edges will be found by 
    obtaining the borders of all separate labels.
    
    Args:
        path_img: Path to the image atlas. The labels image will be 
            found as a corresponding, registered image, unless 
            ``path_atlas_dir`` is given.
        show_imgs: True if the output images should be displayed; defaults 
            to True.
        atlas: True if the primary image is an atlas, which is assumed 
            to be symmetrical. False if the image is an experimental/sample 
            image, in which case erosion will be performed on the full 
            images, and stats will not be performed.
        suffix: Modifier to append to end of ``path_img`` basename for 
            registered image files that were output to a modified name; 
            defaults to None.
        path_atlas_dir: Path to atlas directory to use labels from that 
            directory rather than from labels image registered to 
            ``path_img``, such as when the sample image is registered 
            to an atlas rather than the other way around. Typically 
            coupled with ``suffix`` to compare same sample against 
            different labels. Defaults to None.
    """
    
    # load atlas image, assumed to be a histology image
    if atlas:
        print("generating edge images for atlas")
        atlas_suffix = config.RegNames.IMG_ATLAS.value
    else:
        print("generating edge images for experiment/sample image")
        atlas_suffix = config.RegNames.IMG_EXP.value
    
    # adjust image path with suffix
    mod_path = path_img
    if suffix is not None:
        mod_path = lib_clrbrain.insert_before_ext(mod_path, suffix)
    
    labels_from_atlas_dir = path_atlas_dir and os.path.isdir(path_atlas_dir)
    if labels_from_atlas_dir:
        # load labels from atlas directory
        # TODO: consider applying suffix to labels dir
        path_atlas = path_img
        path_labels = os.path.join(
            path_atlas_dir, config.RegNames.IMG_LABELS.value)
        print("loading labels from", path_labels)
        labels_sitk = sitk.ReadImage(path_labels)
    else:
        # load labels registered to sample image
        path_atlas = mod_path
        labels_sitk = sitk_io.load_registered_img(
            mod_path, config.RegNames.IMG_LABELS.value, get_sitk=True)
    labels_img_np = sitk.GetArrayFromImage(labels_sitk)
    
    # load atlas image, set resolution from it
    atlas_sitk = sitk_io.load_registered_img(
        path_atlas, atlas_suffix, get_sitk=True)
    detector.resolutions = np.array([atlas_sitk.GetSpacing()[::-1]])
    atlas_np = sitk.GetArrayFromImage(atlas_sitk)
    
    # output images
    atlas_sitk_log = None
    atlas_sitk_edge = None
    labels_sitk_interior = None
    
    log_sigma = config.register_settings["log_sigma"]
    if log_sigma is not None and suffix is None:
        # generate LoG and edge-detected images for original image
        print("generating LoG edge-detected images with sigma", log_sigma)
        thresh = (config.register_settings["atlas_threshold"] 
                  if config.register_settings["log_atlas_thresh"] else None)
        atlas_log = plot_3d.laplacian_of_gaussian_img(
            atlas_np, sigma=log_sigma, labels_img=labels_img_np, thresh=thresh)
        atlas_sitk_log = sitk_io.replace_sitk_with_numpy(atlas_sitk, atlas_log)
        atlas_edge = plot_3d.zero_crossing(atlas_log, 1).astype(np.uint8)
        atlas_sitk_edge = sitk_io.replace_sitk_with_numpy(
            atlas_sitk, atlas_edge)
    else:
        # if sigma not set or if using suffix to compare two images, 
        # load from original image to compare against common image
        atlas_edge = sitk_io.load_registered_img(
            path_img, config.RegNames.IMG_ATLAS_EDGE.value)

    erode = config.register_settings["erode_labels"]
    if erode["interior"]:
        # make map of label interiors for interior/border comparisons
        print("Eroding labels to generate interior labels image")
        erosion = config.register_settings["marker_erosion"]
        erosion_frac = config.register_settings["erosion_frac"]
        interior = erode_labels(
            labels_img_np, erosion, erosion_frac, 
            atlas and _is_profile_mirrored())
        labels_sitk_interior = sitk_io.replace_sitk_with_numpy(
            labels_sitk, interior)
    
    # make labels edge and edge distance images
    dist_to_orig, labels_edge = edge_distances(
        labels_img_np, atlas_edge, spacing=atlas_sitk.GetSpacing()[::-1])
    dist_sitk = sitk_io.replace_sitk_with_numpy(atlas_sitk, dist_to_orig)
    labels_sitk_edge = sitk_io.replace_sitk_with_numpy(labels_sitk, labels_edge)
    
    # show all images
    imgs_write = {
        config.RegNames.IMG_ATLAS_LOG.value: atlas_sitk_log, 
        config.RegNames.IMG_ATLAS_EDGE.value: atlas_sitk_edge, 
        config.RegNames.IMG_LABELS_EDGE.value: labels_sitk_edge, 
        config.RegNames.IMG_LABELS_INTERIOR.value: labels_sitk_interior, 
        config.RegNames.IMG_LABELS_DIST.value: dist_sitk, 
    }
    if show:
        for img in imgs_write.values():
            if img: sitk.Show(img)
    
    # write images to same directory as atlas with appropriate suffix
    sitk_io.write_reg_images(imgs_write, mod_path)

def erode_labels(labels_img_np, erosion, erosion_frac=None, mirrored=True):
    """Erode labels image for use as markers or a map of the interior.
    
    Args:
        labels_img_np: Numpy image array of labels in z,y,x format.
        erosion: Filter size for erosion.
        erosion_frac: Target erosion fraction; defaults to None.
        mirrored: True if the primary image mirrored/symmatrical. False
            if otherwise, such as unmirrored atlases or experimental/sample
            images, in which case erosion will be performed on the full image.
    
    Returns:
        The eroded labels as a new array of same shape as that of 
        ``labels_img_np``.
    """
    labels_to_erode = labels_img_np
    if mirrored:
        # for atlases, assume that labels image is symmetric across the x-axis
        len_half = labels_img_np.shape[2] // 2
        labels_to_erode = labels_img_np[..., :len_half]
    
    # convert labels image into markers
    #eroded = segmenter.labels_to_markers_blob(labels_img_np)
    eroded = segmenter.labels_to_markers_erosion(
        labels_to_erode, erosion, erosion_frac)
    if mirrored:
        eroded = _mirror_imported_labels(eroded, len_half)
    
    return eroded

def edge_aware_segmentation(path_atlas, show=True, atlas=True, suffix=None):
    """Segment an atlas using its previously generated edge map.
    
    Labels may not match their own underlying atlas image well, 
    particularly in the orthogonal directions in which the labels 
    were not constructed. To improve alignment between the labels 
    and the atlas itself, register the labels to an automated, roughly 
    segmented version of the atlas. The goal is to improve the 
    labels' alignment so that the atlas/labels combination can be 
    used for another form of automated segmentation by registering 
    them to experimental brains via :func:``register``.
    
    Edge files are assumed to have been generated by 
    :func:``make_edge_images``.
    
    Args:
        path_atlas: Path to the fixed file, typically the atlas file 
            with stained sections. The corresponding edge and labels 
            files will be loaded based on this path.
        show_imgs: True if the output images should be displayed; defaults 
            to True.
        atlas: True if the primary image is an atlas, which is assumed 
            to be symmetrical. False if the image is an experimental/sample 
            image, in which case segmentation will be performed on the full 
            images, and stats will not be performed.
        suffix: Modifier to append to end of ``path_atlas`` basename for 
            registered image files that were output to a modified name; 
            defaults to None. If ``atlas`` is True, ``suffix`` will only 
            be applied to saved files, with files still loaded based on the 
            original path.
    """
    # adjust image path with suffix
    load_path = path_atlas
    mod_path = path_atlas
    if suffix is not None:
        mod_path = lib_clrbrain.insert_before_ext(mod_path, suffix)
        if atlas: load_path = mod_path
    
    # load corresponding files via SimpleITK
    atlas_sitk = sitk_io.load_registered_img(
        load_path, config.RegNames.IMG_ATLAS.value, get_sitk=True)
    atlas_sitk_edge = sitk_io.load_registered_img(
        load_path, config.RegNames.IMG_ATLAS_EDGE.value, get_sitk=True)
    labels_sitk = sitk_io.load_registered_img(
        load_path, config.RegNames.IMG_LABELS.value, get_sitk=True)
    labels_sitk_markers = sitk_io.load_registered_img(
        load_path, config.RegNames.IMG_LABELS_MARKERS.value, get_sitk=True)
    
    # get Numpy arrays of images
    atlas_img_np = sitk.GetArrayFromImage(atlas_sitk)
    atlas_edge = sitk.GetArrayFromImage(atlas_sitk_edge)
    labels_img_np = sitk.GetArrayFromImage(labels_sitk)
    markers = sitk.GetArrayFromImage(labels_sitk_markers)
    
    # segment image from markers
    if atlas:
        # segment only half of image, assuming symmetry
        len_half = atlas_img_np.shape[2] // 2
        labels_seg = segmenter.segment_from_labels(
            atlas_edge[..., :len_half], markers[..., :len_half], 
            labels_img_np[..., :len_half])
    else:
        labels_seg = segmenter.segment_from_labels(
            atlas_edge, markers, labels_img_np)
    
    smoothing = config.register_settings["smooth"]
    if smoothing is not None:
        # smoothing by opening operation based on profile setting
        atlas_refiner.smooth_labels(
            labels_seg, smoothing, config.SmoothingModes.opening)
    
    if atlas:
        # mirror back to other half
        labels_seg = _mirror_imported_labels(labels_seg, len_half)
    
    # expand background to smoothed background of original labels to 
    # roughly match background while still allowing holes to be filled
    crop = config.register_settings["crop_to_orig"]
    atlas_refiner.crop_to_orig(
        labels_img_np, labels_seg, crop)
    
    if labels_seg.dtype != labels_img_np.dtype:
        # watershed may give different output type, so cast back if so
        labels_seg = labels_seg.astype(labels_img_np.dtype)
    labels_sitk_seg = sitk_io.replace_sitk_with_numpy(labels_sitk, labels_seg)
    
    # show DSCs for labels
    print("\nMeasuring overlap of atlas and combined watershed labels:")
    atlas_refiner.measure_overlap_combined_labels(atlas_sitk, labels_sitk_seg)
    print("Measuring overlap of individual original and watershed labels:")
    atlas_refiner.measure_overlap_labels(labels_sitk, labels_sitk_seg)
    print("\nMeasuring overlap of combined original and watershed labels:")
    atlas_refiner.measure_overlap_labels(
        atlas_refiner.make_labels_fg(labels_sitk), 
        atlas_refiner.make_labels_fg(labels_sitk_seg))
    print()
    
    # show and write image to same directory as atlas with appropriate suffix
    sitk_io.write_reg_images(
        {config.RegNames.IMG_LABELS.value: labels_sitk_seg}, mod_path)
    if show: sitk.Show(labels_sitk_seg)
    return path_atlas

def merge_atlas_segmentations(img_paths, show=True, atlas=True, suffix=None):
    """Merge atlas segmentations for a list of files as a multiprocessing 
    wrapper for :func:``merge_atlas_segmentations``, after which 
    edge image post-processing is performed separately since it 
    contains tasks also performed in multiprocessing.
    
    Args:
        img_path: Path to image to load.
        show_imgs: True if the output images should be displayed; defaults 
            to True.
        atlas: True if the image is an atlas; defaults to True.
        suffix: Modifier to append to end of ``img_path`` basename for 
            registered image files that were output to a modified name; 
            defaults to None.
    """
    start_time = time()
    
    # erode all labels images into markers from which to grown via watershed
    erode = config.register_settings["erode_labels"]
    erosion = config.register_settings["marker_erosion"]
    erosion_frac = config.register_settings["erosion_frac"]
    mirrored = atlas and _is_profile_mirrored()
    for img_path in img_paths:
        mod_path = img_path
        if suffix is not None:
            mod_path = lib_clrbrain.insert_before_ext(mod_path, suffix)
        labels_sitk = sitk_io.load_registered_img(
            mod_path, config.RegNames.IMG_LABELS.value, get_sitk=True)
        print("Eroding labels to generate markers for atlas segmentation")
        if erode["markers"]:
            # use default minimal post-erosion size (not setting erosion frac)
            markers = erode_labels(
                sitk.GetArrayFromImage(labels_sitk), erosion, mirrored=mirrored)
            labels_sitk_markers = sitk_io.replace_sitk_with_numpy(
                labels_sitk, markers)
            sitk_io.write_reg_images(
                {config.RegNames.IMG_LABELS_MARKERS.value: labels_sitk_markers},
                mod_path)
    
    pool = mp.Pool()
    pool_results = []
    for img_path in img_paths:
        print("setting up atlas segmentation merge for", img_path)
        # convert labels image into markers
        pool_results.append(pool.apply_async(
            edge_aware_segmentation, args=(img_path, show, atlas, suffix)))
    for result in pool_results:
        # edge distance calculation and labels interior image generation 
        # are multiprocessed, so run them as post-processing tasks to 
        # avoid nested multiprocessing
        path = result.get()
        mod_path = path
        if suffix is not None:
            mod_path = lib_clrbrain.insert_before_ext(path, suffix)
        
        # make edge distance images and stats
        labels_sitk = sitk_io.load_registered_img(
            mod_path, config.RegNames.IMG_LABELS.value, get_sitk=True)
        labels_np = sitk.GetArrayFromImage(labels_sitk)
        dist_to_orig, labels_edge = edge_distances(
            labels_np, path=path, spacing=labels_sitk.GetSpacing()[::-1])
        dist_sitk = sitk_io.replace_sitk_with_numpy(labels_sitk, dist_to_orig)
        labels_sitk_edge = sitk_io.replace_sitk_with_numpy(
            labels_sitk, labels_edge)

        labels_sitk_interior = None
        if erode["interior"]:
            # make interior images from labels using given targeted 
            # post-erosion frac
            interior = erode_labels(
                labels_np, erosion, erosion_frac=erosion_frac, 
                mirrored=mirrored)
            labels_sitk_interior = sitk_io.replace_sitk_with_numpy(
                labels_sitk, interior)
        
        # write images to same directory as atlas
        imgs_write = {
            config.RegNames.IMG_LABELS_DIST.value: dist_sitk, 
            config.RegNames.IMG_LABELS_EDGE.value: labels_sitk_edge, 
            config.RegNames.IMG_LABELS_INTERIOR.value: labels_sitk_interior, 
        }
        sitk_io.write_reg_images(imgs_write, mod_path)
        if show:
            for img in imgs_write.values():
                if img: sitk.Show(img)
        print("finished {}".format(path))
    pool.close()
    pool.join()
    print("time elapsed for merging atlas segmentations:", time() - start_time)

def edge_distances(labels, atlas_edge=None, path=None, spacing=None):
    """Measure the distance between edge images.
    
    Args:
        labels: Labels image as Numpy array.
        atlas_edge: Image as a Numpy array of the atlas reduced to its edges. 
            Defaults to None to load from the corresponding registered 
            file path based on ``path``.
        path: Path from which to load ``atlas_edge`` if it is None.
        spacing: Grid spacing sequence of same length as number of image 
            axis dimensions; defaults to None.
    
    Returns:
        An image array of the same shape as ``labels_edge`` with 
        label edge values replaced by corresponding distance values.
    """
    if atlas_edge is None:
        atlas_edge = sitk_io.load_registered_img(
            path, config.RegNames.IMG_ATLAS_EDGE.value)
    
    # create distance map between edges of original and new segmentations
    labels_edge = vols.make_labels_edge(labels)
    dist_to_orig, _, _ = plot_3d.borders_distance(
        atlas_edge != 0, labels_edge != 0, spacing=spacing)
    
    return dist_to_orig, labels_edge

def make_density_image(img_path, scale=None, shape=None, suffix=None, 
                       labels_img_sitk=None):
    """Make a density image based on associated blobs.
    
    Uses the shape of the registered labels image by default to set 
    the voxel sizes for the blobs.
    
    Args:
        img_path: Path to image, which will be used to indentify the blobs file.
        scale: Rescaling factor as a scalar value to find the corresponding 
            full-sized image. Defaults to None to use the register 
            setting ``target_size`` instead if available, falling back 
            to load the full size image to find its shape if necessary.
        shape: Final shape size; defaults to None to use the shape of 
            the labels image.
        suffix: Modifier to append to end of ``img_path`` basename for 
            registered image files that were output to a modified name; 
            defaults to None.
        labels_img_sitk: Labels image as a SimpleITK ``Image`` object; 
            defaults to None, in which case the registered labels image file 
            corresponding to ``img_path`` with any ``suffix`` modifier 
            will be opened.
    
    Returns:
        Tuple of the density image as a Numpy array in the same shape as 
        the opened image; Numpy array of blob IDs; and the original 
        ``img_path`` to track such as for multiprocessing.
    """
    mod_path = img_path
    if suffix is not None:
        mod_path = lib_clrbrain.insert_before_ext(img_path, suffix)
    if labels_img_sitk is None:
        labels_img_sitk = sitk_io.load_registered_img(
            mod_path, config.RegNames.IMG_LABELS.value, get_sitk=True)
    labels_img = sitk.GetArrayFromImage(labels_img_sitk)
    # load blobs
    filename_base = importer.filename_to_base(
        img_path, config.series)
    info = np.load(filename_base + config.SUFFIX_INFO_PROC)
    blobs = info["segments"]
    print("loading {} blobs".format(len(blobs)))
    # get scaling from source image, which can be rescaled/resized image 
    # since contains scaling image
    load_size = config.register_settings["target_size"]
    img_path_transposed = transformer.get_transposed_image_path(
        img_path, scale, load_size)
    if scale is not None or load_size is not None:
        image5d, img_info = importer.read_file(
            img_path_transposed, config.series, return_info=True)
        scaling = img_info["scaling"]
    else:
        # fall back to scaling based on comparison to original image
        image5d = importer.read_file(
            img_path_transposed, config.series)
        scaling = importer.calc_scaling(image5d, labels_img)
    if shape is not None:
        # scale blobs to an alternative final size
        scaling = np.divide(shape, np.divide(labels_img.shape, scaling))
        labels_spacing = np.multiply(
            labels_img_sitk.GetSpacing()[::-1], 
            np.divide(labels_img.shape, shape))
        labels_img = np.zeros(shape, dtype=labels_img.dtype)
        labels_img_sitk.SetSpacing(labels_spacing[::-1])
    print("using scaling: {}".format(scaling))
    # annotate blobs based on position
    blobs_ids, coord_scaled = ontology.get_label_ids_from_position(
        blobs[:, :3], labels_img, scaling, 
        return_coord_scaled=True)
    print("blobs_ids: {}".format(blobs_ids))
    
    # build heat map to store densities per label px and save to file
    heat_map = plot_3d.build_heat_map(labels_img.shape, coord_scaled)
    out_path = sitk_io.reg_out_path(
        mod_path, config.RegNames.IMG_HEAT_MAP.value)
    print("writing {}".format(out_path))
    heat_map_sitk = sitk_io.replace_sitk_with_numpy(labels_img_sitk, heat_map)
    sitk.WriteImage(heat_map_sitk, out_path, False)
    return heat_map, blobs_ids, img_path

def make_density_images_mp(img_paths, scale=None, shape=None, suffix=None):
    """Make density images for a list of files as a multiprocessing 
    wrapper for :func:``make_density_image``
    
    Args:
        img_path: Path to image, which will be used to indentify the blobs file.
        scale: Rescaling factor as a scalar value. If set, the corresponding 
            image for this factor will be opened. If None, the full size 
            image will be used. Defaults to None.
        suffix: Modifier to append to end of ``img_path`` basename for 
            registered image files that were output to a modified name; 
            defaults to None.
    """
    start_time = time()
    pool = mp.Pool()
    pool_results = []
    for img_path in img_paths:
        print("making image", img_path)
        pool_results.append(pool.apply_async(
            make_density_image, args=(img_path, scale, shape, suffix)))
    heat_maps = []
    for result in pool_results:
        _, _, path = result.get()
        print("finished {}".format(path))
    pool.close()
    pool.join()
    print("time elapsed for making density images:", time() - start_time)

def make_sub_segmented_labels(img_path, suffix=None):
    """Divide each label based on anatomical borders to create a 
    sub-segmented image.
    
    The segmented labels image will be loaded, or if not available, the 
    non-segmented labels will be loaded instead.
    
    Args:
        img_path: Path to main image from which registered images will 
            be loaded.
        suffix: Modifier to append to end of ``img_path`` basename for 
            registered image files that were output to a modified name; 
            defaults to None.
    
    Returns:
        Sub-segmented image as a Numpy array of the same shape as 
        the image at ``img_path``.
    """
    # adjust image path with suffix
    mod_path = img_path
    if suffix is not None:
        mod_path = lib_clrbrain.insert_before_ext(mod_path, suffix)
    
    # load labels
    labels_sitk = sitk_io.load_registered_img(
        mod_path, config.RegNames.IMG_LABELS.value, get_sitk=True)
    
    # atlas edge image is associated with original, not modified image
    atlas_edge = sitk_io.load_registered_img(
        img_path, config.RegNames.IMG_ATLAS_EDGE.value)
    
    # sub-divide the labels and save to file
    labels_img_np = sitk.GetArrayFromImage(labels_sitk)
    labels_subseg = segmenter.sub_segment_labels(labels_img_np, atlas_edge)
    labels_subseg_sitk = sitk_io.replace_sitk_with_numpy(
        labels_sitk, labels_subseg)
    sitk_io.write_reg_images(
        {config.RegNames.IMG_LABELS_SUBSEG.value: labels_subseg_sitk}, mod_path)
    return labels_subseg

def merge_images(img_paths, reg_name, prefix=None, suffix=None, 
                 fn_combine=np.sum):
    """Merge images from multiple paths.
    
    Assumes that the images are relatively similar in size, but will resize 
    them to the size of the first image to combine the images.
    
    Args:
        img_paths: Paths from which registered paths will be found.
        reg_name: Registration suffix to load for the given paths 
            in ``img_paths``.
        prefix: Start of output path; defaults to None to use the first 
           path in ``img_paths`` instead.
        suffix: Portion of path to be combined with each path 
            in ``img_paths`` and output path; defaults to None.
        fn_combine: Function to apply to combine images with ``axis=0``. 
            Defaults to :func:``np.sum``. If None, each image will be 
            inserted as a separate channel.
    
    Returns:
        The combined image in SimpleITK format.
    """
    if len(img_paths) < 1: return None
    
    img_sitk = None
    img_np_base = None
    img_nps = []
    for img_path in img_paths:
        mod_path = img_path
        if suffix is not None:
            # adjust image path with suffix
            mod_path = lib_clrbrain.insert_before_ext(mod_path, suffix)
        print("loading", mod_path)
        get_sitk = img_sitk is None
        img = sitk_io.load_registered_img(mod_path, reg_name, get_sitk=get_sitk)
        if get_sitk:
            # use the first image as template
            img_np = sitk.GetArrayFromImage(img)
            img_np_base = img_np
            img_sitk = img
        else:
            # resize to first image
            img_np = transform.resize(
                img, img_np_base.shape, preserve_range=True, 
                anti_aliasing=True, mode="reflect")
        img_nps.append(img_np)
    
    # combine images and write single combo image
    if fn_combine is None:
        # combine raw images into separate channels
        img_combo = np.stack(img_nps, axis=img_nps[0].ndim)
    else:
        # merge by custom function
        img_combo = fn_combine(img_nps, axis=0)
    combined_sitk = sitk_io.replace_sitk_with_numpy(img_sitk, img_combo)
    # fallback to using first image's name as base
    output_base = img_paths[0] if prefix is None else prefix
    if suffix is not None:
        output_base = lib_clrbrain.insert_before_ext(output_base, suffix)
    output_reg = lib_clrbrain.insert_before_ext(
        reg_name, COMBINED_SUFFIX, "_")
    sitk_io.write_reg_images({output_reg: combined_sitk}, output_base)
    return combined_sitk

def overlay_registered_imgs(fixed_file, moving_file_dir, plane=None, 
                            flip=False, name_prefix=None, out_plane=None):
    """Shows overlays of previously saved registered images.
    
    Should be run after :func:`register` has written out images in default
    (xy) orthogonal orientation. Also output the Dice similiarity coefficient.
    
    Args:
        fixed_file: Path to the fixed file.
        moving_file_dir: Moving files directory, from which the original
            atlas will be retrieved.
        plane: Orthogonal plane to flip the moving image.
        flip: If true, will flip the fixed file first; defaults to False.
        name_prefix: Path with base name where registered files are located; 
            defaults to None, in which case the fixed_file path will be used.
        out_plane: Output plane to view.
    """
    # get the experiment file
    if name_prefix is None:
        name_prefix = fixed_file
    image5d = importer.read_file(fixed_file, config.series)
    roi = image5d[0, ...] # not using time dimension
    
    # get the atlas file and transpose it to match the orientation of the 
    # experiment image
    out_path = os.path.join(moving_file_dir, config.RegNames.IMG_ATLAS.value)
    print("Reading in {}".format(out_path))
    moving_sitk = sitk.ReadImage(out_path)
    moving_sitk = atlas_refiner.transpose_img(
        moving_sitk, plane, 2 if flip else 0)
    moving_img = sitk.GetArrayFromImage(moving_sitk)
    
    # get the registered atlas file, which should already be transposed
    transformed_sitk = sitk_io.load_registered_img(name_prefix, get_sitk=True)
    transformed_img = sitk.GetArrayFromImage(transformed_sitk)
    
    # get the registered labels file, which should also already be transposed
    labels_img = sitk_io.load_registered_img(
        name_prefix, config.RegNames.IMG_LABELS.value)
    
    # calculate the Dice similarity coefficient
    atlas_refiner.measure_overlap(
        _load_numpy_to_sitk(fixed_file), transformed_sitk)
    
    # overlay the images
    imgs = [roi, moving_img, transformed_img, labels_img]
    _, translation = _handle_transform_file(name_prefix)
    translation = _translation_adjust(
        moving_sitk, transformed_sitk, translation, flip=True)
    _show_overlays(imgs, translation, fixed_file, out_plane)


def get_scaled_regionprops(img_region, scaling):
    """Get the scaled regionprops for an image mask
    
    Args:
        img_region: Mask of the region.
        scaling: Scaling factor for the labels image size compared with the 
            experiment image.
    
    Returns:
        A tuple of ``props, bbox, centroid``. ``props`` is the ``regionprops`` 
        properties for the given region. ``bbox`` is the bounding box 
        of this region scaled back to the experiment image. ``centroid`` is 
        the centroid position also scaled to the experiment. Note that the 
        bounding box may encompass many pixels not included in the region 
        itself, including border pixels or even internal pixels in irregularly 
        shaped or sparsely scattered regions. If so, the centroid position 
        may in fact be outside of the region. To ensure that a coordinate 
        within the region tree for ``label_id`` is returned, use 
        :func:``ontology.get_region_middle`` instead.
    """
    # TODO: consider deprecating
    # TODO: use only deepest child?
    img_region = img_region.astype(np.int32)
    print("img_region 1's: {}".format(np.count_nonzero(img_region)))
    props = measure.regionprops(img_region)
    print("num props: {}".format(len(props)))
    scaling_inv = 1 / scaling
    if len(props) > 0:
        bbox_scaled = np.array(props[0].bbox)
        bbox_scaled[:3] = bbox_scaled[:3] * scaling_inv
        bbox_scaled[3:] = bbox_scaled[3:] * scaling_inv
        centroid_scaled = np.array(props[0].centroid) * scaling_inv
        bbox_scaled = np.around(bbox_scaled).astype(np.int)
        centroid_scaled = np.around(centroid_scaled).astype(np.int)
        print(props[0].bbox, props[0].centroid)
        print(bbox_scaled, centroid_scaled)
        return props, bbox_scaled, centroid_scaled
    return props, None, None


def volumes_by_id(img_paths, labels_ref_lookup, suffix=None, unit_factor=None, 
                  groups=None, max_level=None, combine_sides=True, 
                  extra_metrics=None):
    """Get volumes and additional label metrics for each single labels ID.
    
    Args:
        img_paths: Sequence of images.
        labels_ref_lookup: The labels reference lookup, assumed to be an 
            OrderedDict generated by :func:`ontology.create_reverse_lookup` 
            to look up by ID while preserving key order to ensure that 
            parents of any child will be reached prior to the child.
        suffix: Modifier to append to end of ``img_path`` basename for 
            registered image files that were output to a modified name; 
            defaults to None to use "original" as the condition.
        unit_factor: Factor by which volumes will be divided to adjust units; 
            defaults to None.
        groups: Dictionary of sample grouping metadata, where each 
            entry is a list with a values corresponding to ``img_paths``.
        max_level: Integer of maximum ontological level to measure. 
            Giving this value changes the measurement to volumes 
            by ontology, where all children of each label are included 
            in the label rather than taking the label at face value, 
            the default drawn label. Labels at levels up to and including 
            this value will be included. Defaults to None to take 
            labels at face value only.
        combine_sides: True to combine corresponding labels from opposite 
            sides of the sample; defaults to True. Corresponding positive 
            and negative numbers will always be included, and this 
            flag will only determine whether the sides are combined 
            rather than kept separate.
        extra_metrics (List[Enum]): List of enums from 
            :class:`config.MetricGroups` specifying additional stats; 
            defaults to None.
    
    Returns:
        Tuple of Pandas data frames with volume-related metrics for each 
        sample, the first with all each region for each sample in a 
        separate row, and the second with all regions combined in a 
        weighted fashion. This second data frame will be None if 
        ``max_levels`` is not None since subregions are already included 
        in each label.
    """
    start_time = time()
    
    # prepare data frame output path and condition column based on suffix
    out_path = SAMPLE_VOLS
    out_path_summary = SAMPLE_VOLS_SUMMARY
    if max_level is not None:
        out_path = SAMPLE_VOLS_LEVELS
    if suffix is None:
        condition = "original" 
    else:
        condition = config.suffix.replace("_", "")
        out_path += config.suffix
        out_path_summary += config.suffix
    
    # grouping metadata, which will be combined with groups
    grouping = {"Condition": condition}
    
    # setup labels
    label_ids = sitk_io.find_atlas_labels(
        config.load_labels, max_level, labels_ref_lookup)
    if not combine_sides:
        # include opposite side as separate labels; otherwise, defer to 
        # ontology (max_level flag) or labels metrics to get labels from 
        # opposite sides by combining them
        label_ids.extend([-1 * n for n in label_ids])
    if max_level is not None:
        # setup ontological labels
        ids_with_children = []
        for label_id in label_ids:
            label = labels_ref_lookup[abs(label_id)]
            label_level = label[ontology.NODE][config.ABAKeys.LEVEL.value]
            if label_level <= max_level:
                # get children (including parent first) up through level, 
                # including both pos and neg IDs within a super-label if 
                # combining sides
                ids_with_children.append(
                    ontology.get_children_from_id(
                        labels_ref_lookup, label_id, both_sides=combine_sides))
        label_ids = ids_with_children
    
    dfs = []
    dfs_all = []
    sort_cols = ["Region", "Sample", "Side"]
    for i, img_path in enumerate(img_paths):
        # adjust image path with suffix
        mod_path = img_path
        if suffix is not None:
            mod_path = lib_clrbrain.insert_before_ext(img_path, suffix)
        
        # load data frame if available
        df_path = "{}_volumes.csv".format(os.path.splitext(mod_path)[0])
        df_level_path = None
        df = None
        if max_level is not None:
            df_level_path = lib_clrbrain.insert_before_ext(
                df_path, "_level{}".format(max_level))
            if os.path.exists(df_path):
                df = pd.read_csv(df_path)
            else:
                lib_clrbrain.warn(
                    "Could not find raw stats for drawn labels from "
                    "{}, will measure stats for individual regions "
                    "repeatedly. To save processing time, consider "
                    "stopping and re-running first without levels"
                    .format(df_path))
        
        spacing = None
        img_np = None
        labels_img_np = None
        labels_edge = None
        dist_to_orig = None
        labels_interior = None
        heat_map = None
        subseg = None
        if (df is None or 
                extra_metrics and config.MetricGroups.SHAPES in extra_metrics):
            # open images registered to the main image, starting with the 
            # experimental image if available and falling back to atlas; 
            # avoid opening if data frame is available and not taking any 
            # stats requiring images
            try:
                img_sitk = sitk_io.load_registered_img(
                    mod_path, config.RegNames.IMG_EXP.value, get_sitk=True)
            except FileNotFoundError as e:
                print(e)
                lib_clrbrain.warn("will load atlas image instead")
                img_sitk = sitk_io.load_registered_img(
                    mod_path, config.RegNames.IMG_ATLAS.value, get_sitk=True)
            img_np = sitk.GetArrayFromImage(img_sitk)
            spacing = img_sitk.GetSpacing()[::-1]
            
            # load labels in order of priority: full labels > truncated labels; 
            # labels are required and will give exception if not found
            try:
                labels_img_np = sitk_io.load_registered_img(
                    mod_path, config.RegNames.IMG_LABELS.value)
            except FileNotFoundError as e:
                print(e)
                lib_clrbrain.warn(
                    "will attempt to load trucated labels image instead")
                labels_img_np = sitk_io.load_registered_img(
                    mod_path, config.RegNames.IMG_LABELS_TRUNC.value)
            
            # load labels edge and edge distances images
            try:
                labels_edge = sitk_io.load_registered_img(
                    mod_path, config.RegNames.IMG_LABELS_EDGE.value)
                dist_to_orig = sitk_io.load_registered_img(
                    mod_path, config.RegNames.IMG_LABELS_DIST.value)
            except FileNotFoundError as e:
                print(e)
                lib_clrbrain.warn("will ignore edge measurements")
            
            # load labels marker image
            try:
                labels_interior = sitk_io.load_registered_img(
                    mod_path, config.RegNames.IMG_LABELS_INTERIOR.value)
            except FileNotFoundError as e:
                print(e)
                lib_clrbrain.warn("will ignore label markers")
            
            # load heat map of nuclei per voxel if available
            try:
                heat_map = sitk_io.load_registered_img(
                    mod_path, config.RegNames.IMG_HEAT_MAP.value)
            except FileNotFoundError as e:
                print(e)
                lib_clrbrain.warn("will ignore nuclei stats")
            
            # load sub-segmentation labels if available
            try:
                subseg = sitk_io.load_registered_img(
                    mod_path, config.RegNames.IMG_LABELS_SUBSEG.value)
            except FileNotFoundError as e:
                print(e)
                lib_clrbrain.warn("will ignore labels sub-segmentations")
            print("tot blobs", np.sum(heat_map))
            
            mirror = config.register_settings["labels_mirror"]
            if mirror is None or mirror == -1: mirror = 0.5
            mirrori = int(labels_img_np.shape[2] * mirror)
            half_lbls = labels_img_np[:, :, mirrori:]
            if (np.sum(half_lbls < 0) == 0 and 
                    np.sum(half_lbls != 0) > np.sum(labels_img_np != 0) / 3):
                # unmirrored images may have bilateral labels that are all pos, 
                # while these metrics assume that R hem is neg; invert pos R 
                # labels if they are at least 1/3 of total labels and not just 
                # spillover from the L side into an otherwise unlabeled R
                print("less than half of labels in right hemisphere are neg; "
                      "inverting all pos labels in x >= {} (shape {}) for "
                      "sided metrics".format(mirrori, labels_img_np.shape))
                half_lbls[half_lbls > 0] *= -1
        
        # prepare sample name with original name for comparison across 
        # conditions and add an arbitrary number of metadata grouping cols
        sample = lib_clrbrain.get_filename_without_ext(img_path)
        if groups is not None:
            for key in groups.keys():
                grouping[key] = groups[key][i]
            
        # measure stats per label for the given sample; max_level already 
        # takes care of combining sides
        df, df_all = vols.measure_labels_metrics(
            sample, img_np, labels_img_np, 
            labels_edge, dist_to_orig, labels_interior, heat_map, subseg, 
            spacing, unit_factor, 
            combine_sides and max_level is None, 
            label_ids, grouping, df, extra_metrics)
        
        # output volume stats CSV to atlas directory and append for 
        # combined CSVs
        if max_level is None:
            stats.data_frames_to_csv([df], df_path, sort_cols=sort_cols)
        elif df_level_path is not None:
            stats.data_frames_to_csv([df], df_level_path, sort_cols=sort_cols)
        dfs.append(df)
        dfs_all.append(df_all)
    
    # combine data frames from all samples by region for each sample
    df_combined = stats.data_frames_to_csv(
        dfs, out_path, sort_cols=sort_cols)
    df_combined_all = None
    if max_level is None:
        # combine weighted combo of all regions per sample; 
        # not necessary for levels-based (ontological) volumes since they 
        # already accumulate from sublevels
        df_combined_all = stats.data_frames_to_csv(
            dfs_all, out_path_summary)
    print("time elapsed for volumes by ID: {}".format(time() - start_time))
    return df_combined, df_combined_all

def make_labels_diff_img(img_path, df_path, meas, fn_avg, prefix=None, 
                         show=False, level=None, meas_path_name=None, 
                         col_wt=None):
    """Replace labels in an image with the differences in metrics for 
    each given region between two conditions.
    
    Args:
        img_path: Path to the base image from which the corresponding 
            registered image will be found.
        df_path: Path to data frame with metrics for the labels.
        meas: Name of colum in data frame with the chosen measurement.
        fn_avg: Function to apply to the set of measurements, such as a mean. 
            Can be None if ``df_path`` points to a stats file from which 
            to extract metrics directly in :meth:``vols.map_meas_to_labels``.
        prefix: Start of path for output image; defaults to None to 
            use ``img_path`` instead.
        show: True to show the images after generating them; defaults to False.
        level: Ontological level at which to look up and show labels. 
            Assume that labels level image corresponding to this value 
            has already been generated by :meth:``make_labels_level_img``. 
            Defaults to None to use only drawn labels.
        meas_path_name: Name to use in place of `meas` in output path; 
            defaults to None.
        col_wt (str): Name of column to use for weighting; defaults to None.
    """
    # load labels image and data frame before generating map for the 
    # given metric of the chosen measurement
    print("Generating labels difference image for", meas, "from", df_path)
    reg_name = (config.RegNames.IMG_LABELS.value if level is None 
                else config.RegNames.IMG_LABELS_LEVEL.value.format(level))
    labels_sitk = sitk_io.load_registered_img(img_path, reg_name, get_sitk=True)
    labels_np = sitk.GetArrayFromImage(labels_sitk)
    df = pd.read_csv(df_path)
    labels_diff = vols.map_meas_to_labels(
        labels_np, df, meas, fn_avg, reverse=True, col_wt=col_wt)
    if labels_diff is None: return
    labels_diff_sitk = sitk_io.replace_sitk_with_numpy(labels_sitk, labels_diff)
    
    # save and show labels difference image using measurement name in 
    # output path or overriding with custom name
    meas_path = meas if meas_path_name is None else meas_path_name
    reg_diff = lib_clrbrain.insert_before_ext(
        config.RegNames.IMG_LABELS_DIFF.value, meas_path, "_")
    if fn_avg is not None:
        # add function name to output path if given
        reg_diff = lib_clrbrain.insert_before_ext(
            reg_diff, fn_avg.__name__, "_")
    imgs_write = {reg_diff: labels_diff_sitk}
    out_path = prefix if prefix else img_path
    sitk_io.write_reg_images(imgs_write, out_path)
    if show:
        for img in imgs_write.values():
            if img: sitk.Show(img)

def make_labels_level_img(img_path, level, prefix=None, show=False):
    """Replace labels in an image with their parents at the given level.
    
    Labels that do not fall within a parent at that level will remain in place.
    
    Args:
        img_path: Path to the base image from which the corresponding 
            registered image will be found.
        level: Ontological level at which to group child labels. 
        prefix: Start of path for output image; defaults to None to 
            use ``img_path`` instead.
        show: True to show the images after generating them; defaults to False.
    """
    # load original labels image and setup ontology dictionary
    labels_sitk = sitk_io.load_registered_img(
        img_path, config.RegNames.IMG_LABELS.value, get_sitk=True)
    labels_np = sitk.GetArrayFromImage(labels_sitk)
    ref = ontology.load_labels_ref(config.load_labels)
    labels_ref_lookup = ontology.create_aba_reverse_lookup(ref)
    
    ids = list(labels_ref_lookup.keys())
    for key in ids:
        keys = [key, -1 * key]
        for region in keys:
            if region == 0: continue
            # get ontological label
            label = labels_ref_lookup[abs(region)]
            label_level = label[ontology.NODE][config.ABAKeys.LEVEL.value]
            if label_level == level:
                # get children (including parent first) at given level 
                # and replace them with parent
                label_ids = ontology.get_children_from_id(
                    labels_ref_lookup, region)
                labels_region = np.isin(labels_np, label_ids)
                print("replacing labels within", region)
                labels_np[labels_region] = region
    labels_level_sitk = sitk_io.replace_sitk_with_numpy(labels_sitk, labels_np)
    
    # generate an edge image at this level
    labels_edge = vols.make_labels_edge(labels_np)
    labels_edge_sikt = sitk_io.replace_sitk_with_numpy(labels_sitk, labels_edge)
    
    # write and optionally display labels level image
    imgs_write = {
        config.RegNames.IMG_LABELS_LEVEL.value.format(level): labels_level_sitk, 
        config.RegNames.IMG_LABELS_EDGE_LEVEL.value.format(level): 
            labels_edge_sikt, 
    }
    out_path = prefix if prefix else img_path
    sitk_io.write_reg_images(imgs_write, out_path)
    if show:
        for img in imgs_write.values():
            if img: sitk.Show(img)


def _test_labels_lookup():
    """Test labels reverse dictionary creation and lookup.
    """
    
    # create reverse lookup dictionary
    ref = ontology.load_labels_ref(config.load_labels)
    lookup_id = 15565 # short search path
    #lookup_id = 126652058 # last item
    time_dict_start = time()
    id_dict = ontology.create_aba_reverse_lookup(ref)
    labels_img = sitk_io.load_registered_img(
        config.filename, config.RegNames.IMG_LABELS.value)
    max_labels = np.max(labels_img)
    print("max_labels: {}".format(max_labels))
    time_dict_end = time()
    
    # look up a single ID
    time_node_start = time()
    found = id_dict[lookup_id]
    time_node_end = time()
    print("found {}: {} with parents {}"
          .format(lookup_id, found[ontology.NODE]["name"], 
                  found[ontology.PARENT_IDS]))
    
    # brute-force query
    time_direct_start = time()
    node = ontology.get_node(ref["msg"][0], "id", lookup_id, "children")
    time_direct_end = time()
    #print(node)
    
    print("time to create id_dict (s): {}".format(time_dict_end - time_dict_start))
    print("time to find node (s): {}".format(time_node_end - time_node_start))
    print("time to find node directly (s): {}".format(time_direct_end - time_direct_start))
    
    # get a list of IDs corresponding to each blob
    blobs = np.array([[300, 5000, 3000], [350, 5500, 4500], [400, 6000, 5000]])
    image5d = importer.read_file(config.filename, config.series)
    scaling = importer.calc_scaling(image5d, labels_img)
    ids, coord_scaled = ontology.get_label_ids_from_position(
        blobs[:, 0:3], labels_img, scaling, return_coord_scaled=True)
    print("blob IDs:\n{}".format(ids))
    print("coord_scaled:\n{}".format(coord_scaled))

def _test_region_from_id():
    """Test finding a region by ID in a labels image.
    """
    if len(config.filenames) > 1:
        # unregistered, original labels image; assume that atlas directory has 
        # been given similarly to register fn
        path = os.path.join(
            config.filenames[1], config.RegNames.IMG_LABELS.value)
        labels_img = sitk.ReadImage(path)
        labels_img = sitk.GetArrayFromImage(labels_img)
        scaling = np.ones(3)
        print("loaded labels image from {}".format(path))
    else:
        # registered labels image and associated experiment file
        labels_img = sitk_io.load_registered_img(
            config.filename, config.RegNames.IMG_LABELS.value)
        if config.filename.endswith(".mhd"):
            img = sitk.ReadImage(config.filename)
            img = sitk.GetArrayFromImage(img)
            image5d = img[None]
        else:
            image5d = importer.read_file(config.filename, config.series)
        scaling = importer.calc_scaling(image5d, labels_img)
        print("loaded experiment image from {}".format(config.filename))
    ref = ontology.load_labels_ref(config.load_labels)
    id_dict = ontology.create_aba_reverse_lookup(ref)
    middle, img_region, region_ids = ontology.get_region_middle(
        id_dict, 16652, labels_img, scaling)
    atlas_label = ontology.get_label(
        middle, labels_img, id_dict, scaling, None, True)
    props, bbox, centroid = get_scaled_regionprops(img_region, scaling)
    print("bbox: {}, centroid: {}".format(bbox, centroid))

def _test_curate_img(path, prefix):
    fixed_img = _load_numpy_to_sitk(path)
    labels_img = sitk_io.load_registered_img(
        prefix, config.RegNames.IMG_LABELS.value, get_sitk=True)
    atlas_img = sitk_io.load_registered_img(
        prefix, config.RegNames.IMG_ATLAS.value, get_sitk=True)
    labels_img.SetSpacing(fixed_img.GetSpacing())
    holes_area = config.register_settings["holes_area"]
    result_imgs = _curate_img(
        fixed_img, labels_img, [atlas_img], inpaint=False, 
        holes_area=holes_area)
    sitk.Show(fixed_img)
    sitk.Show(labels_img)
    sitk.Show(result_imgs[0])
    sitk.Show(result_imgs[1])

def _test_smoothing_metric():
    img = np.zeros((6, 6, 6), dtype=int)
    img[1:5, 1:5, 1:5] = 1
    img_smoothed = morphology.binary_erosion(img)
    print("img:\n{}".format(img))
    print("img_smoothed:\n{}".format(img_smoothed))
    atlas_refiner.label_smoothing_metric(img, img_smoothed)

def main():
    """Handle registration processing tasks as specified in 
    :attr:``config.register_type``.
    """
    plot_2d.setup_style("default")
    # convert volumes to next larger prefix (eg um^3 to mm^3)
    unit_factor = np.power(1000.0, 3)
    
    # name prefix to use a different name from the input files, such as when 
    # registering transposed/scaled images but outputting paths corresponding 
    # to the original image
    if config.prefix is not None:
        print("Formatting registered filenames to match {}"
              .format(config.prefix))
    if config.suffix is not None:
        print("Modifying registered filenames with suffix {}"
              .format(config.suffix))
    flip = False
    if config.flip is not None:
        flip = config.flip[0]
    show = not config.no_show
    size = config.roi_sizes
    if size: size = size[0][:2]
    
    #_test_labels_lookup()
    #_test_region_from_id()
    #_test_curate_img(config.filenames[0], config.prefix)
    #_test_smoothing_metric()
    #os._exit(os.EX_OK)
    
    reg = config.RegisterTypes[config.register_type]
    if config.register_type is None:
        # explicitly require a registration type
        print("Please choose a registration type")
    
    elif reg in (
            config.RegisterTypes.single, config.RegisterTypes.new_atlas):
        # "single", basic registration of 1st to 2nd image, transposing the 
        # second image according to config.plane and config.flip_horiz; 
        # "new_atlas" registers similarly but outputs new atlas files
        new_atlas = reg is config.RegisterTypes.new_atlas
        register(*config.filenames[0:2], plane=config.plane, 
                 flip=flip, name_prefix=config.prefix, new_atlas=new_atlas, 
                 show_imgs=show)
    
    elif reg is config.RegisterTypes.group:
        # groupwise registration, which assumes that the last image 
        # filename given is the prefix and uses the full flip array
        prefix = config.filenames[-1]
        register_group(
            config.filenames[:-1], flip=config.flip, name_prefix=config.prefix, 
            scale=config.rescale, show_imgs=show)
    
    elif reg is config.RegisterTypes.overlays:
        # overlay registered images in each orthogonal plane
        for out_plane in config.PLANE:
            overlay_registered_imgs(
                *config.filenames[0:2], plane=config.plane, 
                flip=flip, name_prefix=config.prefix, 
                out_plane=out_plane)
    
    elif reg is config.RegisterTypes.export_regions:
        # export regions IDs to CSV files
        
        # use ABA levels up through the specified level to collect 
        # sub-regions to the given level
        if config.labels_level is None:
            levels = [None]
        else:
            levels = list(range(config.labels_level + 1))
        ref = ontology.load_labels_ref(config.load_labels)
        labels_ref_lookup = ontology.create_aba_reverse_lookup(ref)
        
        # export region IDs and parents at given level to CSV
        export_regions.export_region_ids(
            labels_ref_lookup, "region_ids", config.labels_level)
        # export region IDs to network file
        export_regions.export_region_network(
            labels_ref_lookup, "region_network")
    
    elif reg is config.RegisterTypes.import_atlas:
        # import original atlas, mirroring if necessary
        atlas_refiner.import_atlas(config.filename, show=show)
    
    elif reg is config.RegisterTypes.export_common_labels:
        # export common labels
        export_regions.export_common_labels(
            config.filenames, config.PATH_COMMON_LABELS)
    
    elif reg is config.RegisterTypes.convert_itksnap_labels:
        # convert labels from ITK-SNAP to CSV format
        df = ontology.convert_itksnap_to_df(config.filename)
        output_path = lib_clrbrain.combine_paths(
            config.filename, ".csv", sep="")
        stats.data_frames_to_csv([df], output_path)
    
    elif reg is config.RegisterTypes.export_metrics_compactness:
        # export data frame with compactness to compare:
        # 1) whole histology image and unsmoothed labels
        # 2) unsmoothed and selected smoothed labels
        # 1st config.filenames element should be atlas import stats, 
        # and 2nd element should be smoothing stats
        
        # load data frames
        df_stats = pd.read_csv(config.filename)  # atlas import stats
        df_smoothing = pd.read_csv(config.filenames[1])  # smoothing stats
        
        cols = [config.AtlasMetrics.SAMPLE.value,
                config.AtlasMetrics.REGION.value,
                config.AtlasMetrics.CONDITION.value,
                config.SmoothingMetrics.COMPACTNESS.value]
        
        # compare histology vs combined original labels
        df_histo_vs_orig, dfs_baseline = stats.filter_dfs_on_vals(
            [df_stats, df_smoothing], cols, 
            [None, (config.SmoothingMetrics.FILTER_SIZE.value, 0)])
        df_histo_vs_orig[config.GENOTYPE_KEY] = "Histo Vs Orig Labels"
        
        # compare combined original vs smoothed labels
        smooth = config.register_settings["smooth"]
        df_unsm_vs_sm, dfs_smoothed = stats.filter_dfs_on_vals(
            [dfs_baseline[1], df_smoothing], cols, 
            [None, (config.SmoothingMetrics.FILTER_SIZE.value, smooth)])
        df_unsm_vs_sm[config.GENOTYPE_KEY] = "Smoothing"

        # compare histology vs smoothed labels
        df_histo_vs_sm = pd.concat([dfs_baseline[0], dfs_smoothed[1]])
        df_histo_vs_sm[config.GENOTYPE_KEY] = "Vs Smoothed Labels"
        
        # export data frames
        output_path = lib_clrbrain.combine_paths(
            config.filename, "compactness.csv")
        df = pd.concat([df_histo_vs_orig, df_histo_vs_sm, df_unsm_vs_sm])
        df[config.AtlasMetrics.REGION.value] = "all"
        stats.data_frames_to_csv(df, output_path)
    
    elif reg is config.RegisterTypes.plot_smoothing_metrics:
        # plot smoothing metrics
        title = "{} Label Smoothing".format(
            lib_clrbrain.str_to_disp(
                os.path.basename(config.filename).replace(
                    config.PATH_SMOOTHING_METRICS, "")))
        lbls = ("Fractional Change", "Smoothing Filter Size")
        plot_2d.plot_lines(
            config.filename, config.SmoothingMetrics.FILTER_SIZE.value, 
            (config.SmoothingMetrics.COMPACTED.value,
             config.SmoothingMetrics.DISPLACED.value,
             config.SmoothingMetrics.SM_QUALITY.value), 
            ("--", "--", "-"), lbls, title, size, show, "_quality")
        plot_2d.plot_lines(
            config.filename, config.SmoothingMetrics.FILTER_SIZE.value, 
            (config.SmoothingMetrics.SA_VOL.value,
             config.SmoothingMetrics.LABEL_LOSS.value), 
            ("-", "-"), lbls, None, size, show, "_extras", ("C3", "C4"))
    
    elif reg is config.RegisterTypes.smoothing_peaks:
        # find peak smoothing qualities without label loss for a set 
        # of data frames and output to combined data frame
        dfs = []
        for path in config.filenames:
            df = pd.read_csv(path)
            dfs.append(smoothing_peak(df, 0, None))
        stats.data_frames_to_csv(dfs, "smoothing_peaks.csv")
    
    elif reg in (
            config.RegisterTypes.make_edge_images, 
            config.RegisterTypes.make_edge_images_exp):
        
        # convert atlas or experiment image and associated labels 
        # to edge-detected images; labels can be given as atlas dir from 
        # which labels will be extracted (eg import dir)
        atlas = reg is config.RegisterTypes.make_edge_images
        for img_path in config.filenames:
            make_edge_images(
                img_path, show, atlas, config.suffix, config.load_labels)
    
    elif reg is config.RegisterTypes.reg_labels_to_atlas:
        # register labels to its underlying atlas
        register_labels_to_atlas(config.filename)
    
    elif reg in (
        config.RegisterTypes.merge_atlas_segs, 
        config.RegisterTypes.merge_atlas_segs_exp):
        
        # merge various forms of atlas segmentations
        atlas = reg is config.RegisterTypes.merge_atlas_segs
        merge_atlas_segmentations(
            config.filenames, show=show, atlas=atlas, suffix=config.suffix)
    
    elif reg is config.RegisterTypes.vol_stats:
        # volumes stats
        ref = ontology.load_labels_ref(config.load_labels)
        labels_ref_lookup = ontology.create_aba_reverse_lookup(ref)
        groups = {}
        if config.groups is not None:
            groups[config.GENOTYPE_KEY] = [
                config.GROUPS_NUMERIC[geno] for geno in config.groups]
        # should generally leave uncombined for drawn labels to allow 
        # faster level building, where can combine sides
        combine_sides = config.register_settings["combine_sides"]
        extra_metric_groups = config.register_settings["extra_metric_groups"]
        volumes_by_id(
            config.filenames, labels_ref_lookup, suffix=config.suffix, 
            unit_factor=unit_factor, groups=groups, 
            max_level=config.labels_level, combine_sides=combine_sides, 
            extra_metrics=extra_metric_groups)
    
    elif reg is config.RegisterTypes.make_density_images:
        # make density images
        size = config.roi_sizes
        if size: size = size[0][::-1]
        make_density_images_mp(
            config.filenames, config.rescale, size, config.suffix)
    
    elif reg is config.RegisterTypes.make_subsegs:
        # make sub-segmentations for all images
        for img_path in config.filenames:
            make_sub_segmented_labels(img_path, config.suffix)

    elif reg is config.RegisterTypes.merge_images:
        # take mean of separate experiments from all paths using the 
        # given registered image type, defaulting to experimental images
        suffix = config.RegNames.IMG_EXP.value
        if config.reg_suffixes is not None:
            # use suffix assigned to atlas
            suffix_exp = config.reg_suffixes[config.RegSuffixes.ATLAS]
            if suffix_exp: suffix = suffix_exp
        merge_images(config.filenames, suffix, config.prefix, config.suffix)

    elif reg is config.RegisterTypes.merge_images_channels:
        # combine separate experiments from all paths into separate channels
        merge_images(
            config.filenames, config.RegNames.IMG_EXP.value, config.prefix, 
            config.suffix, fn_combine=None)

    elif reg is config.RegisterTypes.register_reg:
        # register a group of registered images to another image, 
        # such as the atlas to which the images were originally registered
        suffixes = None
        if config.reg_suffixes is not None:
            # get additional suffixes to register the same as for exp img
            suffixes = [config.reg_suffixes[key] 
                        for key, val in config.reg_suffixes.items() 
                        if config.reg_suffixes[key] is not None]
        register_reg(
            *config.filenames[:2], config.RegNames.IMG_EXP.value, suffixes, 
            config.plane, flip, config.prefix, config.suffix, show)

    elif reg is config.RegisterTypes.make_labels_level:
        # make a labels image grouped at the given level
        make_labels_level_img(
            config.filename, config.labels_level, config.prefix, show)
    
    elif reg is config.RegisterTypes.labels_diff:
        # generate labels difference images for various measurements 
        # and metrics, using the output from volumes measurements for 
        # drawn labels
        path_df = (SAMPLE_VOLS if config.labels_level is None 
                   else SAMPLE_VOLS_LEVELS) + ".csv"
        metrics = [
            (vols.LabelMetrics.CoefVarNuc.name, np.nanmean), 
            (vols.LabelMetrics.CoefVarNuc.name, np.nanmedian), 
            (vols.LabelMetrics.CoefVarIntens.name, np.nanmean), 
            (vols.LabelMetrics.CoefVarIntens.name, np.nanmedian), 
        ]
        for metric in metrics:
            col_wt = vols.get_metric_weight_col(metric[0])
            make_labels_diff_img(
                config.filename, path_df, *metric, config.prefix, show, 
                config.labels_level, col_wt=col_wt)

    elif reg is config.RegisterTypes.labels_diff_stats:
        # generate labels difference images for various measurements 
        # from a stats CSV generated by the R clrstats package
        metrics = (
            vols.LabelMetrics.EdgeDistSum.name, 
            vols.LabelMetrics.CoefVarNuc.name, 
            vols.LabelMetrics.CoefVarIntens.name, 
            #vols.MetricCombos.HOMOGENEITY.value[0], 
        )
        for metric in metrics:
            path_df = "{}_{}.csv".format("vols_stats", metric)
            if not os.path.exists(path_df): continue
            col_wt = vols.get_metric_weight_col(metric)
            make_labels_diff_img(
                config.filename, path_df, "vals.effect", None, config.prefix, 
                show, meas_path_name=metric, col_wt=col_wt)
    
    elif reg is config.RegisterTypes.combine_cols:
        # normalize the given columns to original values in a data frame 
        # and combine columns for composite metrics
        df = pd.read_csv(config.filename)
        df = stats.normalize_df(
            df, ["Sample", "Region"], "Condition", "original", 
            (vols.LabelMetrics.VarIntensity.name, 
             vols.LabelMetrics.VarIntensDiff.name, 
             vols.LabelMetrics.EdgeDistSum.name, 
             vols.LabelMetrics.VarNuclei.name), 
            (vols.LabelMetrics.VarIntensity.Volume.name, ))
        df = stats.combine_cols(
            df, (vols.MetricCombos.HOMOGENEITY, ))
        stats.data_frames_to_csv(
            df, lib_clrbrain.insert_before_ext(config.filename, "_norm"))

    elif reg is config.RegisterTypes.zscores:
        # measurea and export z-scores for the given metrics to a new 
        # data frame and display as a scatter plot
        metric_cols = (
            vols.LabelMetrics.VarIntensity.name, 
            #vols.LabelMetrics.VarIntensDiff.name,
            vols.LabelMetrics.VarNuclei.name,  
            vols.LabelMetrics.EdgeDistSum.name, 
        )
        extra_cols = ("Sample", "Condition", vols.LabelMetrics.Volume.name)
        atlas_stats.meas_plot_zscores(
            config.filename, metric_cols, extra_cols, 
            (vols.MetricCombos.HOMOGENEITY, ), size, show)
    
    elif reg is config.RegisterTypes.coefvar:
        # measure and export coefficient of variation for the given metrics 
        # to a new data frame and display as a scatter plot; note that these 
        # CVs are variation of intensity variation, for example, rather than 
        # the raw CVs of intensity measures in vols
        metric_cols = (
            vols.LabelMetrics.VarIntensity.name, 
            vols.LabelMetrics.VarIntensMatch.name,
            vols.LabelMetrics.VarNuclei.name,  
            vols.LabelMetrics.EdgeDistSum.name, 
        )
        combos = (
            vols.MetricCombos.COEFVAR_INTENS, vols.MetricCombos.COEFVAR_NUC
        )
        atlas_stats.meas_plot_coefvar(
            config.filename, ["Region"], "Condition", "original", metric_cols, 
            combos, vols.LabelMetrics.Volume.name, size, show)

    elif reg is config.RegisterTypes.melt_cols:
        # melt columns specified in "groups" using ID columns from 
        # standard atlas metrics
        id_cols = [
            config.AtlasMetrics.SAMPLE.value, config.AtlasMetrics.REGION.value, 
            config.AtlasMetrics.CONDITION.value]
        df = stats.melt_cols(
            pd.read_csv(config.filename), id_cols, config.groups, 
            config.GENOTYPE_KEY)
        stats.data_frames_to_csv(
            df, lib_clrbrain.insert_before_ext(config.filename, "_melted"))

    elif reg is config.RegisterTypes.plot_region_dev:
        # plot region development
        try:
            metric = vols.LabelMetrics[config.stats_type].name
            atlas_stats.plot_region_development(metric, size, show)
        except KeyError:
            lib_clrbrain.warn(
                "\"{}\" metric not found in {} for developmental plots"
                .format(config.stats_type, [e.name for e in vols.LabelMetrics]))
        
    elif reg is config.RegisterTypes.plot_lateral_unlabeled:
        # plot lateral edge unlabeled fractions as both lines and bars
        cols = (config.AtlasMetrics.LAT_UNLBL_VOL.value,
                config.AtlasMetrics.LAT_UNLBL_PLANES.value)
        atlas_stats.plot_unlabeled_hemisphere(config.filename, cols, size, show)


if __name__ == "__main__":
    print("Clrbrain image registration")
    from clrbrain import cli
    cli.main(True)
    main()
