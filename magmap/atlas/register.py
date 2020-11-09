#!/usr/bin/env python
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

from collections import OrderedDict
import os
import shutil
from time import time

import pandas as pd
import numpy as np
import SimpleITK as sitk
from skimage import filters, measure, morphology, transform

from magmap.atlas import atlas_refiner, edge_seg, ontology, transformer
from magmap.cv import cv_nd
from magmap.io import cli, df_io, export_regions, importer, libmag, sitk_io
from magmap.plot import plot_2d, plot_3d
from magmap.settings import config
from magmap.stats import atlas_stats, clustering, vols

SAMPLE_VOLS = "vols_by_sample"
SAMPLE_VOLS_LEVELS = SAMPLE_VOLS + "_levels"
SAMPLE_VOLS_SUMMARY = SAMPLE_VOLS + "_summary"

REREG_SUFFIX = "rereg"

# 3D format extensions to check when finding registered files
_SIGNAL_THRESHOLD = 0.01

# sort volume columns
_SORT_VOL_COLS = [
    config.AtlasMetrics.REGION.value,
    config.AtlasMetrics.SAMPLE.value,
    config.AtlasMetrics.SIDE.value,
]

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
    plot_2d.plot_overlays_reg(
        *imgs, *cmaps, translation, os.path.basename(fixed_file), plane,
        config.show)


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


def _load_numpy_to_sitk(numpy_file, rotate=False, channel=None):
    """Load Numpy image array to SimpleITK Image object.

    Use ``channel`` to extract a single channel before generating a
    :obj:`sitk.Image` object for many SimpleITK filters that require
    single-channel ("scalar" rather than "vector") images.
    
    Args:
        numpy_file: Path to Numpy archive file.
        rotate: True if the image should be rotated 180 deg; defaults to False.
        channel (int, Tuple[int]): Integer or sequence of integers specifying
            channels to keep.
    
    Returns:
        The image in SimpleITK format.
    """
    img5d = importer.read_file(numpy_file, config.series)
    image5d = img5d.img
    roi = image5d[0, ...]  # not using time dimension
    if channel is not None and len(roi.shape) >= 4:
        roi = roi[..., channel]
        print("extracted channel(s) for SimpleITK image:", channel)
    if rotate:
        roi = np.rot90(roi, 2, (1, 2))
    sitk_img = sitk.GetImageFromArray(roi)
    spacing = config.resolutions[0]
    sitk_img.SetSpacing(spacing[::-1])
    # TODO: consider setting z-origin to 0 since image generally as 
    # tightly bound to subject as possible
    #sitk_img.SetOrigin([0, 0, 0])
    sitk_img.SetOrigin([0, 0, -roi.shape[0] // 2])
    #sitk_img.SetOrigin([0, 0, -roi.shape[0]])
    return sitk_img


def _curate_img(fixed_img, labels_img, imgs=None, inpaint=True, carve=True, 
                thresh=None, holes_area=None):
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
        thresh (float): Threshold to use for carving; defaults to None to
            determine by taking the mean threshold of ``fixed_img``.
        holes_area: Maximum area of holes to fill when carving.
    
    Returns:
        A list of images in SimpleITK format that have been curated.
    """
    fixed_img_np = sitk.GetArrayFromImage(fixed_img)
    labels_img_np = sitk.GetArrayFromImage(labels_img)
    # ensure that labels image is first
    if imgs:
        imgs.insert(0, labels_img)
    else:
        imgs = [labels_img]
    
    # mask image showing where result is 0 but fixed image is above thresh 
    # to fill in with nearest neighbors
    if thresh is None:
        thresh = filters.threshold_mean(fixed_img_np)
    print("Carving thresh for curation: {}".format(thresh))
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
            result_img_np = cv_nd.in_paint(result_img_np, to_fill)
        if carve:
            _, mask = cv_nd.carve(fixed_img_np, thresh, holes_area)
            result_img_np[~mask] = 0
        result_img = sitk_io.replace_sitk_with_numpy(img, result_img_np)
        result_imgs.append(result_img)
        if i == 0:
            # check overlap based on labels images; should be 1.0 by def
            result_img_np[result_img_np != 0] = 2
            result_img_for_overlap = sitk_io.replace_sitk_with_numpy(
                img, result_img_np)
            atlas_refiner.measure_overlap(
                fixed_img, result_img_for_overlap, thresh_img2=1)
    return result_imgs


def _transform_labels(transformix_img_filter, labels_img, truncation=None):
    if truncation is not None:
        # truncate ventral and posterior portions since variable 
        # amount of tissue or quality of imaging in these regions
        labels_img_np = sitk.GetArrayFromImage(labels_img)
        truncation = list(truncation)
        rotate = config.transform[config.Transforms.ROTATE]
        if rotate and libmag.get_if_within(rotate, 0) >= 2:
            # assume labels were rotated 180deg around the z-axis, so 
            # need to flip y-axis fracs
            # TODO: take into account full transformations
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


def register_duo(fixed_img, moving_img, path=None, fixed_mask=None,
                 moving_mask=None):
    """Register two images to one another using ``SimpleElastix``.
    
    Args:
        fixed_img (:obj:`sitk.Image`): The image to be registered to.
        moving_img (:obj:`sitk.Image`): The image to register to ``fixed_img``.
        path (str): Path as string from whose parent directory the points-based
            registration files ``fix_pts.txt`` and ``mov_pts.txt`` will
            be found; defaults to None, in which case points-based
            reg will be ignored even if set.
        fixed_mask (:obj:`sitk.Image`): Mask for ``fixed_img``; defaults
            to None.
        moving_mask (:obj:`sitk.Image`): Mask for ``moving_img``; defaults
            to None.
    
    Returns:
        :obj:`SimpleITK.Image`, :obj:`sitk.TransformixImageFilter`: Tuple of
        the registered image and a Transformix filter with the registration's
        parameters to reapply them on other images.
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

    # add any masks
    if fixed_mask is not None:
        elastix_img_filter.SetFixedMask(fixed_mask)
    if moving_mask is not None:
        elastix_img_filter.SetMovingMask(moving_mask)
    
    # set up parameter maps for translation, affine, and deformable regs
    settings = config.atlas_profile
    param_map_vector = sitk.VectorOfParameterMap()
    for key in ("reg_translation", "reg_affine", "reg_bspline"):
        params = settings[key]
        if not params: continue
        max_iter = params["max_iter"]
        # TODO: consider removing since does not skip if "0" and need at least
        # one transformation for reg, even if 0 iterations
        if not max_iter: continue
        param_map = sitk.GetDefaultParameterMap(params["map_name"])
        similarity = params["metric_similarity"]
        if len(param_map["Metric"]) > 1:
            param_map["Metric"] = [similarity, *param_map["Metric"][1:]]
        else:
            param_map["Metric"] = [similarity]
        param_map["MaximumNumberOfIterations"] = [max_iter]
        grid_spacing_sched = params["grid_spacing_schedule"]
        if grid_spacing_sched:
            # fine tune the spacing for multi-resolution registration
            _config_reg_resolutions(
                grid_spacing_sched, param_map, fixed_img.GetDimension())
        else:
            # num of resolutions is automatically set by spacing sched
            param_map["NumberOfResolutions"] = [params["num_resolutions"]]
        grid_space_voxels = params["grid_space_voxels"]
        if grid_space_voxels:
            param_map["FinalGridSpacingInVoxels"] = [grid_space_voxels]
            # avoid conflict with voxel spacing
            if "FinalGridSpacingInPhysicalUnits" in param_map:
                del param_map["FinalGridSpacingInPhysicalUnits"]
        erode_mask = params["erode_mask"]
        if erode_mask:
            param_map["ErodeMask"] = [erode_mask]

        if path is not None and params["point_based"]:
            # point-based registration added to b-spline, which takes point sets
            # found in name_prefix's folder; note that coordinates are from the
            # originally set fixed and moving images, not after transformation
            # up to this point
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
    elastix_img_filter.Execute()
    transformed_img = elastix_img_filter.GetResultImage()
    
    # prep filter to apply transformation to label files; turn off final
    # bspline interpolation to avoid overshooting the interpolation for the
    # labels image (see Elastix manual section 4.3)
    transform_param_map = elastix_img_filter.GetTransformParameterMap()
    transformix_img_filter = sitk.TransformixImageFilter()
    transform_param_map[-1]["FinalBSplineInterpolationOrder"] = ["0"]
    transformix_img_filter.SetTransformParameterMap(transform_param_map)
    
    return transformed_img, transformix_img_filter


def register(fixed_file, moving_img_path, show_imgs=True, write_imgs=True,
             name_prefix=None, new_atlas=False):
    """Register an atlas and associated labels to a sample image 
    using the SimpleElastix library.
    
    Uses the first channel in :attr:`config.channel` or the first image channel.
    
    Args:
        fixed_file: The image to register, given as a Numpy archive file to 
            be read by :importer:`read_file`.
        moving_img_path (str): Moving image base path from which an intensity
            and a labels image will be loaded using registered image suffixes.
            Falls back to atlas volume for the intensity image and annotation
            for the labels image. The intensity image will be used for
            registration. The atlas is currently used as the moving file
            since it is likely to be lower resolution than the Numpy file.
        show_imgs: True if the output images should be displayed; defaults to 
            True.
        write_imgs: True if the images should be written to file; defaults to 
            False.
        name_prefix: Path with base name where registered files will be output; 
            defaults to None, in which case the fixed_file path will be used.
        new_atlas: True to generate registered images that will serve as a 
            new atlas; defaults to False.
    """
    def get_similarity_metric():
        # get the name of the similarity metric from the first found name
        # among the available transformations
        for trans in ("reg_translation", "reg_affine", "reg_bspline"):
            trans_prof = settings[trans]
            if trans_prof:
                sim = trans_prof["metric_similarity"]
                if sim:
                    return sim
        return None
    
    start_time = time()
    if name_prefix is None:
        name_prefix = fixed_file
    settings = config.atlas_profile
    
    # load fixed image, assumed to be experimental image
    chl = config.channel[0] if config.channel else 0
    fixed_img = _load_numpy_to_sitk(fixed_file, channel=chl)
    
    # preprocess fixed image; store original fixed image for overlap measure
    # TODO: assume fixed image is preprocessed before starting this reg?
    fixed_img_orig = fixed_img
    if settings["preprocess"]:
        img_np = sitk.GetArrayFromImage(fixed_img)
        #img_np = plot_3d.saturate_roi(img_np)
        img_np = plot_3d.denoise_roi(img_np)
        fixed_img = sitk_io.replace_sitk_with_numpy(fixed_img, img_np)
    
    # load moving images based on registered image suffixes, falling back to
    # atlas volume and labels suffixes
    moving_atlas_suffix = config.reg_suffixes[config.RegSuffixes.ATLAS]
    if not moving_atlas_suffix:
        moving_atlas_suffix = config.RegNames.IMG_ATLAS.value
    moving_img = sitk_io.load_registered_img(
        moving_img_path, moving_atlas_suffix, get_sitk=True)
    moving_labels_suffix = config.reg_suffixes[config.RegSuffixes.ANNOTATION]
    if not moving_labels_suffix:
        moving_labels_suffix = config.RegNames.IMG_LABELS.value
    labels_img = sitk_io.load_registered_img(
        moving_img_path, moving_labels_suffix, get_sitk=True)

    # TODO: implement mask option
    fixed_mask = None
    moving_mask = None

    # transform and preprocess moving images

    # transpose moving images
    moving_img = atlas_refiner.transpose_img(moving_img)
    labels_img = atlas_refiner.transpose_img(labels_img)

    # get Numpy arrays of moving images for preprocessing
    moving_img_np = sitk.GetArrayFromImage(moving_img)
    labels_img_np = sitk.GetArrayFromImage(labels_img)
    moving_mask_np = None
    if moving_mask is not None:
        moving_mask_np = sitk.GetArrayFromImage(moving_mask)

    crop_out_labels = config.atlas_profile["crop_out_labels"]
    if crop_out_labels is not None:
        # crop moving images to extent without given labels; note that
        # these labels may still exist within the cropped image
        mask = np.zeros_like(labels_img_np, dtype=np.uint8)
        mask[labels_img_np != 0] = 1
        mask[np.isin(labels_img_np, crop_out_labels)] = 0
        labels_img_np, moving_img_np, _ = cv_nd.crop_to_labels(
            labels_img_np, moving_img_np, mask, 0, 0)

    rotate = config.atlas_profile["rotate"]
    if rotate and rotate["rotation"] is not None:
        # more granular 3D rotation than in prior transposition
        moving_img_np = transformer.rotate_img(moving_img_np, rotate)
        labels_img_np = transformer.rotate_img(labels_img_np, rotate, 0)
        if moving_mask_np is not None:
            moving_mask_np = transformer.rotate_img(moving_mask_np, rotate, 0)

    # convert images back to sitk format
    labels_img = sitk_io.replace_sitk_with_numpy(labels_img, labels_img_np)
    moving_img = sitk_io.replace_sitk_with_numpy(moving_img, moving_img_np)
    moving_imgs = [moving_img, labels_img]
    if moving_mask is not None:
        moving_mask = sitk_io.replace_sitk_with_numpy(
            moving_mask, moving_mask_np)
        moving_imgs.append(moving_mask)

    rescale = config.atlas_profile["rescale"]
    if rescale:
        # rescale images as a factor of their spacing in case the scaling
        # transformation in the affine-based registration is insufficient
        moving_img_spacing = np.multiply(moving_img.GetSpacing(), rescale)
        for img in moving_imgs:
            img.SetSpacing(moving_img_spacing)
    
    # perform registration
    try:
        img_moved, transformix_filter = register_duo(
            fixed_img, moving_img, name_prefix, fixed_mask, moving_mask)
    except RuntimeError:
        libmag.warn("Could not perform registration. Will retry with"
                    "world info (spacing, origin, etc) set to that of the "
                    "fixed image.")
        # fall back to simply matching all world info
        # TODO: consider matching world info by default since output is same
        imgs = list(moving_imgs)
        if fixed_mask is not None:
            imgs.append(fixed_mask)
        for img in imgs:
            sitk_io.match_world_info(fixed_img, img)
        img_moved, transformix_filter = register_duo(
            fixed_img, moving_img, name_prefix, fixed_mask, moving_mask)

    # overlap stats comparing original and registered samples (eg histology)
    print("DSC of original and registered sample images")
    thresh_mov = settings["atlas_threshold"]
    dsc_sample = atlas_refiner.measure_overlap(
        fixed_img_orig, img_moved, thresh_img2=thresh_mov)
    fallback = settings["metric_sim_fallback"]
    metric_sim = get_similarity_metric()
    if fallback and dsc_sample < fallback[0]:
        # fall back to another atlas profile; update the current profile with
        # this new profile and re-set-up the original profile afterward
        print("Registration DSC below threshold of {}, will re-register "
              "using {} atlas profile".format(*fallback))
        atlas_prof_name_orig = settings[settings.NAME_KEY]
        cli.setup_atlas_profiles(fallback[1], reset=False)
        metric_sim = get_similarity_metric()
        img_moved, transformix_filter = register_duo(
            fixed_img, moving_img, name_prefix, fixed_mask, moving_mask)
        cli.setup_atlas_profiles(atlas_prof_name_orig)
        dsc_sample = atlas_refiner.measure_overlap(
            fixed_img_orig, img_moved, thresh_img2=thresh_mov)
    
    def make_labels(truncation=None):
        # apply the same transformation to labels via Transformix, with option
        # to truncate part of labels
        labels_trans = _transform_labels(
            transformix_filter, labels_img, truncation=truncation)
        print(labels_trans.GetSpacing())
        # WORKAROUND: labels img floating point vals may be more rounded 
        # than transformed moving img for some reason; assume transformed 
        # labels and moving image should match exactly, so replace labels 
        # with moving image's transformed spacing
        labels_trans.SetSpacing(img_moved.GetSpacing())
        dsc = None
        labels_trans_cur = None
        transformed_img_cur = None
        if settings["curate"]:
            thresh_carve = settings["carve_threshold"]
            if isinstance(thresh_carve, str):
                # get threshold from another setting, eg atlas_threshold_all
                thresh_carve = settings[thresh_carve]
            labels_trans_cur, transformed_img_cur = _curate_img(
                fixed_img_orig, labels_trans, [img_moved], inpaint=new_atlas,
                thresh=thresh_carve, holes_area=settings["holes_area"])
            print("DSC of original and registered sample images after curation")
            dsc = atlas_refiner.measure_overlap(
                fixed_img_orig, transformed_img_cur, 
                thresh_img2=settings["atlas_threshold"])
        return labels_trans, labels_trans_cur, transformed_img_cur, dsc

    # apply same transformation to labels, +/- curation to carve the moving
    # image where the fixed image does not exist or in-paint where it does
    labels_moved, labels_moved_cur, img_moved_cur, dsc_sample_cur = (
        make_labels())
    img_moved_precur = None
    if img_moved_cur is not None:
        # save pre-curated moved image
        img_moved_precur = img_moved
        img_moved = img_moved_cur
    labels_moved_precur = None
    if labels_moved_cur is not None:
        # save pre-curated moved labels
        labels_moved_precur = labels_moved
        labels_moved = labels_moved_cur
    imgs_write = {
        config.RegNames.IMG_EXP.value: fixed_img,
        config.RegNames.IMG_ATLAS.value: img_moved,
        config.RegNames.IMG_ATLAS_PRECUR.value: img_moved_precur,
        config.RegNames.IMG_LABELS.value: labels_moved,
        config.RegNames.IMG_LABELS_PRECUR.value: labels_moved_precur,
    }
    truncate_labels = settings["truncate_labels"]
    if truncate_labels is not None:
        labels_img_truc = make_labels(truncate_labels)[1]
        imgs_write[config.RegNames.IMG_LABELS_TRUNC.value] = labels_img_truc

    if show_imgs:
        # show individual SimpleITK images in default viewer
        for img in imgs_write.values():
            if img is not None:
                sitk.Show(img)
    
    if write_imgs:
        # write atlas and labels files
        write_prefix = name_prefix
        if new_atlas:
            # new atlases only consist of atlas and labels without fixed
            # image filename in path
            keys = (config.RegNames.IMG_ATLAS.value,
                    config.RegNames.IMG_LABELS.value)
            imgs_write = {k: imgs_write[k] for k in keys}
            write_prefix = os.path.dirname(name_prefix)
        sitk_io.write_reg_images(
            imgs_write, write_prefix, prefix_is_dir=new_atlas)

    # save transform parameters and attempt to find the original position 
    # that corresponds to the final position that will be displayed
    _, translation = _handle_transform_file(
        name_prefix, transformix_filter.GetTransformParameterMap())
    _translation_adjust(moving_img, img_moved, translation, flip=True)
    
    # compare original atlas with registered labels taken as a whole
    dsc_labels = atlas_refiner.measure_overlap_combined_labels(
        fixed_img_orig, labels_moved)
    
    # measure compactness of fixed image
    fixed_img_orig_np = sitk.GetArrayFromImage(fixed_img_orig)
    thresh_atlas = fixed_img_orig_np > filters.threshold_mean(fixed_img_orig_np)
    compactness, _, _ = cv_nd.compactness_3d(
        thresh_atlas, fixed_img_orig.GetSpacing()[::-1])
    
    # save basic metrics in CSV file
    basename = libmag.get_filename_without_ext(fixed_file)
    metrics = {
        config.AtlasMetrics.SAMPLE: [basename], 
        config.AtlasMetrics.REGION: config.REGION_ALL, 
        config.AtlasMetrics.CONDITION: [np.nan], 
        config.AtlasMetrics.SIMILARITY_METRIC: [metric_sim],
        config.AtlasMetrics.DSC_ATLAS_SAMPLE: [dsc_sample], 
        config.AtlasMetrics.DSC_ATLAS_SAMPLE_CUR: [dsc_sample_cur],
        config.AtlasMetrics.DSC_SAMPLE_LABELS: [dsc_labels],
        config.SmoothingMetrics.COMPACTNESS: [compactness],
    }
    df_path = libmag.combine_paths(
        name_prefix, config.PATH_ATLAS_IMPORT_METRICS)
    print("\nImported {} whole atlas stats:".format(basename))
    df_io.dict_to_data_frame(metrics, df_path, show="\t")

    '''
    # show 2D overlays or registered image and atlas last since blocks until 
    # fig is closed
    imgs = [
        sitk.GetArrayFromImage(fixed_img),
        sitk.GetArrayFromImage(moving_img), 
        sitk.GetArrayFromImage(img_moved), 
        sitk.GetArrayFromImage(labels_img)]
    _show_overlays(imgs, translation, fixed_file, None)
    '''
    print("time elapsed for single registration (s): {}"
          .format(time() - start_time))


def register_rev(fixed_path, moving_path, reg_base=None, reg_names=None,
                 plane=None, prefix=None, suffix=None, show=True):
    """Reverse registration from :meth:`register`, registering a sample
    image (moving image) to an atlas image (fixed image).

    Useful for registering a sample image and associated registered atlas
    images to another image. For example, registered images can be registered
    back to the atlas.

    This method can also be used to move unregistered original images 
    that have simply been copied as ``config.RegNames.IMG_EXP.value`` 
    during registration. This copy can be registered "back" to the atlas,
    reversing the fixed/moving images in :meth:``register`` to move all
    experimental images into the same space.
    
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
        mod_path = libmag.insert_before_ext(mod_path, suffix)
    if reg_base is None:
        # load the image directly from given path
        moving_img = sitk.ReadImage(mod_path)
    else:
        # treat the path as a base path to which a reg suffix will be combined
        moving_img = sitk_io.load_registered_img(
            mod_path, reg_base, get_sitk=True)
    
    # register the images and apply the transformation to any 
    # additional images previously registered to the moving path
    moving_img = atlas_refiner.transpose_img(moving_img, plane)
    transformed_img, transformix_img_filter = register_duo(
        fixed_img, moving_img, prefix)
    settings = config.atlas_profile
    print("DSC of original and registered sample images")
    dsc_sample = atlas_refiner.measure_overlap(
        fixed_img, transformed_img, thresh_img1=settings["atlas_threshold"])
    fallback = settings["metric_sim_fallback"]
    if fallback and dsc_sample < fallback[0]:
        print("reg DSC below threshold of {}, will re-register using {} "
              "similarity metric".format(*fallback))
        atlas_prof_name_orig = settings[settings.NAME_KEY]
        cli.setup_atlas_profiles(fallback[1], reset=False)
        transformed_img, transformix_img_filter = register_duo(
            fixed_img, moving_img, prefix)
        cli.setup_atlas_profiles(atlas_prof_name_orig)
    reg_imgs = [transformed_img]
    names = [config.RegNames.IMG_EXP.value if reg_base is None else reg_base]
    if reg_names is not None:
        for reg_name in reg_names:
            img = sitk_io.load_registered_img(mod_path, reg_name, get_sitk=True)
            img = atlas_refiner.transpose_img(img, plane)
            transformix_img_filter.SetMovingImage(img)
            transformix_img_filter.Execute()
            img_result = transformix_img_filter.GetResultImage()
            reg_imgs.append(img_result)
            names.append(reg_name)
    
    # use prefix as base output path if given and append distinguishing string 
    # to differentiate from original files
    output_base = libmag.insert_before_ext(
        moving_path if prefix is None else prefix, REREG_SUFFIX, "_")
    if suffix is not None:
        output_base = libmag.insert_before_ext(output_base, suffix)
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


def register_group(img_files, rotate=None, show_imgs=True,
                   write_imgs=True, name_prefix=None, scale=None):
    """Group registers several images to one another.
    
    Uses the first channel in :attr:`config.channel` or the first channel
    in each image.
    
    Args:
        img_files: Paths to image files to register.
        rotate (List[int]): List of number of 90 degree rotations for images
            corresponding to ``img_files``; defaults to None, in which
            case the `config.transform` rotate attribute will be used.
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
    if rotate is None:
        rotate = config.transform[config.Transforms.ROTATE]
    target_size = config.atlas_profile["target_size"]
    
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
    # image properties of 1st image, in SimpleITK format
    origin = None
    size_orig = None
    size_cropped = None
    start_y = None
    spacing = None
    img_np_template = None # first image, used as template for rest
    for i, img_file in enumerate(img_files):
        # load image, flipping if necessary and using transposed img if
        # specified, and extract only the channel from config setting
        img_file = transformer.get_transposed_image_path(
            img_file, scale, target_size)
        rot = rotate and libmag.get_if_within(rotate, i, 0) >= 2
        chl = config.channel[0] if config.channel else 0
        img = _load_numpy_to_sitk(img_file, rot, chl)
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
    
    settings = config.atlas_profile
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
        img_mean, _, img_mean_unfilled = cv_nd.carve(
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
        config.resolutions = [transformed_img.GetSpacing()[::-1]]
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
    :func:``edge_seg.make_edge_images``.
    
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
    settings = config.atlas_profile
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


def overlay_registered_imgs(fixed_file, moving_file_dir, plane=None, 
                            rotate=None, name_prefix=None, out_plane=None):
    """Shows overlays of previously saved registered images.
    
    Should be run after :func:`register` has written out images in default
    (xy) orthogonal orientation. Also output the Dice similiarity coefficient.
    
    Args:
        fixed_file: Path to the fixed file.
        moving_file_dir: Moving files directory, from which the original
            atlas will be retrieved.
        plane: Orthogonal plane to flip the moving image.
        rotate (int): Number of 90 degree rotations; defaults to None.
        name_prefix: Path with base name where registered files are located; 
            defaults to None, in which case the fixed_file path will be used.
        out_plane: Output plane to view.
    """
    # get the experiment file
    if name_prefix is None:
        name_prefix = fixed_file
    img5d = importer.read_file(fixed_file, config.series)
    image5d = img5d.img
    roi = image5d[0, ...]  # not using time dimension
    
    # get the atlas file and transpose it to match the orientation of the 
    # experiment image
    out_path = os.path.join(moving_file_dir, config.RegNames.IMG_ATLAS.value)
    print("Reading in {}".format(out_path))
    moving_sitk = sitk.ReadImage(out_path)
    moving_sitk = atlas_refiner.transpose_img(
        moving_sitk, plane, rotate)
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


def make_label_ids_set(labels_ref_lookup, max_level, combine_sides):
    """Make a set of label IDs for the given level and sides.
    
    Args:
        labels_ref_lookup: The labels reference lookup, assumed to be an 
            OrderedDict generated by :func:`ontology.create_reverse_lookup` 
            to look up by ID while preserving key order to ensure that 
            parents of any child will be reached prior to the child.
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

    Returns:
        :List[int]: List of IDs.

    """
    label_ids = sitk_io.find_atlas_labels(
        config.load_labels, max_level is None, labels_ref_lookup)
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
    return label_ids


def _setup_vols_df(df_path, max_level):
    # setup data frame and paths for volume metrics
    df_level_path = None
    df = None
    if max_level is not None:
        df_level_path = libmag.insert_before_ext(
            df_path, "_level{}".format(max_level))
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
        else:
            libmag.warn(
                "Could not find raw stats for drawn labels from {}, will "
                "measure stats for individual regions repeatedly. To save "
                "processing time, consider stopping and re-running first "
                "without levels".format(df_path))
    return df, df_path, df_level_path


def volumes_by_id(img_paths, labels_ref_lookup, suffix=None, unit_factor=None, 
                  groups=None, max_level=None, combine_sides=True, 
                  extra_metrics=None):
    """Get volumes and additional label metrics for each single labels ID.
    
    Atlas (intensity) and annotation (labels) images can be configured
    in :attr:`config.reg_suffixes`.
    
    Args:
        img_paths: Sequence of images.
        labels_ref_lookup: The labels reference lookup, assumed to be an 
            OrderedDict generated by :func:`ontology.create_reverse_lookup`.
        suffix: Modifier to append to end of ``img_path`` basename for 
            registered image files that were output to a modified name; 
            defaults to None to use "original" as the condition.
        unit_factor: Factor by which volumes will be divided to adjust units; 
            defaults to None.
        groups: Dictionary of sample grouping metadata, where each 
            entry is a list with a values corresponding to ``img_paths``.
        max_level: Integer of maximum ontological level to measure. 
            Defaults to None to take labels at face value only.
        combine_sides: True to combine corresponding labels from opposite 
            sides of the sample; defaults to True.
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
    grouping = OrderedDict()
    grouping[config.AtlasMetrics.SAMPLE.value] = None
    grouping[config.AtlasMetrics.CONDITION.value] = condition
    
    # setup labels
    label_ids = make_label_ids_set(labels_ref_lookup, max_level, combine_sides)
    
    dfs = []
    dfs_all = []
    for i, img_path in enumerate(img_paths):
        # adjust image path with suffix
        mod_path = img_path
        if suffix is not None:
            mod_path = libmag.insert_before_ext(img_path, suffix)
        
        # load data frame if available
        df_path = "{}_volumes.csv".format(os.path.splitext(mod_path)[0])
        df, df_path, df_level_path = _setup_vols_df(df_path, max_level)
        
        spacing = None
        img_np = None
        labels_img_np = None
        labels_edge = None
        dist_to_orig = None
        labels_interior = None
        heat_map = None
        blobs = None
        subseg = None
        if (df is None or 
                extra_metrics and config.MetricGroups.SHAPES in extra_metrics):
            # open images registered to the main image; avoid opening if data
            # frame is available and not taking any stats requiring images
            
            # open intensity image in priority: config > exp > atlas
            atlas_suffix = config.reg_suffixes[config.RegSuffixes.ATLAS]
            if not atlas_suffix:
                atlas_suffix = config.RegNames.IMG_EXP.value
            try:
                img_sitk = sitk_io.load_registered_img(
                    mod_path, atlas_suffix, get_sitk=True)
            except FileNotFoundError as e:
                print(e)
                libmag.warn("will load atlas image instead")
                img_sitk = sitk_io.load_registered_img(
                    mod_path, config.RegNames.IMG_ATLAS.value, get_sitk=True)
            img_np = sitk.GetArrayFromImage(img_sitk)
            spacing = img_sitk.GetSpacing()[::-1]
            
            # load labels in order of priority: config > full labels
            # > truncated labels; required so give exception if not found
            labels_suffix = config.reg_suffixes[config.RegSuffixes.ANNOTATION]
            if not labels_suffix:
                labels_suffix = config.RegNames.IMG_LABELS.value
            try:
                labels_img_np = sitk_io.load_registered_img(
                    mod_path, labels_suffix)
            except FileNotFoundError as e:
                print(e)
                libmag.warn(
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
                libmag.warn("will ignore edge measurements")
            
            # load labels marker image
            try:
                labels_interior = sitk_io.load_registered_img(
                    mod_path, config.RegNames.IMG_LABELS_INTERIOR.value)
            except FileNotFoundError as e:
                print(e)
                libmag.warn("will ignore label markers")
            
            # load heat map of nuclei per voxel if available
            try:
                heat_map = sitk_io.load_registered_img(
                    mod_path, config.RegNames.IMG_HEAT_MAP.value)
            except FileNotFoundError as e:
                print(e)
                libmag.warn("will ignore nuclei stats")

            if (extra_metrics and 
                    config.MetricGroups.POINT_CLOUD in extra_metrics):
                # load blobs with coordinates, label IDs, and cluster IDs
                # if available
                try:
                    blobs = np.load(libmag.combine_paths(
                        mod_path, config.SUFFIX_BLOB_CLUSTERS))
                    print(blobs)
                except FileNotFoundError as e:
                    print(e)

            # load sub-segmentation labels if available
            try:
                subseg = sitk_io.load_registered_img(
                    mod_path, config.RegNames.IMG_LABELS_SUBSEG.value)
            except FileNotFoundError as e:
                print(e)
                libmag.warn("will ignore labels sub-segmentations")
            print("tot blobs", np.sum(heat_map))
            
        # prepare sample name with original name for comparison across 
        # conditions and add an arbitrary number of metadata grouping cols
        sample = libmag.get_filename_without_ext(img_path)
        grouping[config.AtlasMetrics.SAMPLE.value] = sample
        if groups is not None:
            for key in groups.keys():
                grouping[key] = groups[key][i]
            
        # measure stats per label for the given sample; max_level already 
        # takes care of combining sides
        df, df_all = vols.measure_labels_metrics(
            img_np, labels_img_np, labels_edge, dist_to_orig, labels_interior,
            heat_map, blobs, subseg, spacing, unit_factor, 
            combine_sides and max_level is None, label_ids, grouping, df, 
            extra_metrics)
        
        # output volume stats CSV to atlas directory and append for 
        # combined CSVs
        if max_level is None:
            df_io.data_frames_to_csv([df], df_path, sort_cols=_SORT_VOL_COLS)
        elif df_level_path is not None:
            df_io.data_frames_to_csv(
                [df], df_level_path, sort_cols=_SORT_VOL_COLS)
        dfs.append(df)
        dfs_all.append(df_all)
    
    # combine data frames from all samples by region for each sample
    df_combined = df_io.data_frames_to_csv(
        dfs, out_path, sort_cols=_SORT_VOL_COLS)
    df_combined_all = None
    if max_level is None:
        # combine weighted combo of all regions per sample; 
        # not necessary for levels-based (ontological) volumes since they 
        # already accumulate from sublevels
        df_combined_all = df_io.data_frames_to_csv(
            dfs_all, out_path_summary)
    print("time elapsed for volumes by ID: {}".format(time() - start_time))
    return df_combined, df_combined_all


def volumes_by_id_compare(img_paths, labels_ref_lookup, unit_factor=None,
                          groups=None, max_level=None, combine_sides=True,
                          offset=None, roi_size=None):
    """Compare metrics for volumes metrics for each label ID between different
    sets of atlases.
    
    Label identities can be translated using CSV files specified in the
    :attr:`config.atlas_labels[config.AtlasLabels.TRANSLATE_LABELS` value
    to compare labels from different atlases or groups of labels that do
    not fit exclusively into a single super-structure. Each file will be
    mapped to the corresponding path in ``img_paths``.

    Args:
        img_paths: Sequence of images to compare.
        labels_ref_lookup: The labels reference lookup, assumed to be an 
            OrderedDict generated by :func:`ontology.create_reverse_lookup`.
        unit_factor: Factor by which volumes will be divided to adjust units; 
            defaults to None.
        groups: Dictionary of sample grouping metadata, where each 
            entry is a list with a values corresponding to ``img_paths``.
        max_level: Integer of maximum ontological level to measure. 
            Defaults to None to take labels at face value only.
        combine_sides: True to combine corresponding labels from opposite 
            sides of the sample; defaults to True.
        offset (List[int]): ROI offset in ``x,y,z``; defaults to None to use
            the whole image.
        roi_size (List[int]): ROI shape in ``x,y,z``; defaults to None to use
            the whole image.

    Returns:
        :obj:`pd.DataFrame`: Pandas data frame with volume-related metrics.
    """
    start_time = time()
    num_img_paths = len(img_paths)
    if num_img_paths < 2:
        libmag.warn(
            "need at least 2 images to compare, found {}".format(num_img_paths))
    
    # setup labels and load data frame if available
    label_ids = make_label_ids_set(labels_ref_lookup, max_level, combine_sides)
    df_path = "{}_volcompare.csv".format(os.path.splitext(img_paths[0])[0])
    df, df_path, df_level_path = _setup_vols_df(df_path, max_level)
    
    spacing = None
    labels_imgs = None
    heat_map = None
    if df is None:
        # open images for primary measurements rather than weighting from
        # data frame
        labels_imgs_sitk = [
            sitk_io.load_registered_img(
                p, config.RegNames.IMG_LABELS.value, get_sitk=True)
            for p in img_paths]
        labels_imgs = [sitk.GetArrayFromImage(img) for img in labels_imgs_sitk]
        spacing = labels_imgs_sitk[0].GetSpacing()[::-1]
        
        # load heat map of nuclei per voxel if available based on 1st path
        try:
            heat_map = sitk_io.load_registered_img(
                img_paths[0], config.RegNames.IMG_HEAT_MAP.value)
        except FileNotFoundError as e:
            libmag.warn("will ignore nuclei stats")
       
        if offset is not None and roi_size is not None:
            # extract an ROI from all images
            print("Comparing overlap within ROI given by offset {}, shape {} "
                  "(x,y,z)".format(offset, roi_size))
            labels_imgs = [plot_3d.prepare_roi(img, offset, roi_size, 3)
                           for img in labels_imgs]
            if heat_map is not None:
                heat_map = plot_3d.prepare_roi(heat_map, offset, roi_size, 3)
        
        if config.atlas_profile["crop_to_first_image"]:
            # an image may only contain a subset of labels or parts of labels;
            # allow cropping to compare only this extent in other images
            print("Crop all images to foreground of first image")
            mask = None
            for i, labels_img in enumerate(labels_imgs):
                if i == 0:
                    mask = labels_img > 0
                else:
                    labels_img[~mask] = 0
            if mask is not None and heat_map is not None:
                heat_map[~mask] = 0
        
        paths_translate = config.atlas_labels[
            config.AtlasLabels.TRANSLATE_LABELS]
        if paths_translate:
            # load data frames corresponding to each labels image to convert
            # label IDs, clearing all other labels
            if not libmag.is_seq(paths_translate):
                paths_translate = [paths_translate]
            for labels_img, path_translate in zip(labels_imgs, paths_translate):
                if os.path.exists(path_translate):
                    # translate labels based on the given data frame
                    ontology.replace_labels(
                        labels_img, pd.read_csv(path_translate), clear=True)
                elif path_translate:
                    # warn if path does not exist; empty string can skip image
                    libmag.warn("{} does not exist, skipping label translation"
                                .format(paths_translate))
    
    # sample metadata
    sample = libmag.get_filename_without_ext(img_paths[0])
    grouping = OrderedDict()
    grouping[config.AtlasMetrics.SAMPLE.value] = sample
    if groups is not None:
        for key in groups.keys():
            grouping[key] = groups[key]
    
    # measure stats per label for the given sample; max_level already 
    # takes care of combining sides
    df_out = vols.measure_labels_overlap(
        labels_imgs, heat_map, spacing, unit_factor, 
        combine_sides and max_level is None, label_ids, grouping, df)
    
    # output volume stats CSV to atlas directory and append for 
    # combined CSVs
    if max_level is None:
        df_out = df_io.data_frames_to_csv(
            df_out, df_path, sort_cols=_SORT_VOL_COLS)
    elif df_level_path is not None:
        df_out = df_io.data_frames_to_csv(
            df_out, df_level_path, sort_cols=_SORT_VOL_COLS)
    print("time elapsed for volumes compared by ID: {}"
          .format(time() - start_time))
    return df_out


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
    img5d = importer.read_file(config.filename, config.series)
    scaling = importer.calc_scaling(img5d.img, labels_img)
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
            img5d = importer.read_file(config.filename, config.series)
        scaling = importer.calc_scaling(img5d.img, labels_img)
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
    holes_area = config.atlas_profile["holes_area"]
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
    :attr:`magmap.config.register_type`.
    """
    # name prefix to use a different name from the input files, such as when 
    # registering transposed/scaled images but outputting paths corresponding 
    # to the original image
    if config.prefix is not None:
        print("Formatting registered filenames to match {}"
              .format(config.prefix))
    if config.suffix is not None:
        print("Modifying registered filenames with suffix {}"
              .format(config.suffix))
    size = config.roi_sizes
    if size: size = size[0][:2]
    # TODO: transition size -> fig_size
    fig_size = config.plot_labels[config.PlotLabels.SIZE]
    scaling = None
    if config.metadatas and config.metadatas[0]:
        output = config.metadatas[0]
        if "scaling" in output:
            scaling = output["scaling"]
    
    #_test_labels_lookup()
    #_test_region_from_id()
    #_test_curate_img(config.filenames[0], config.prefix)
    #_test_smoothing_metric()
    #os._exit(os.EX_OK)

    reg = libmag.get_enum(config.register_type, config.RegisterTypes)
    print("Performing register task:", reg)
    if config.register_type is None:
        # explicitly require a registration type
        print("Please choose a registration type")
    
    elif reg in (
            config.RegisterTypes.SINGLE, config.RegisterTypes.NEW_ATLAS):
        # "single", basic registration of 1st to 2nd image, transposing the 
        # second image according to config.plane and config.flip_horiz; 
        # "new_atlas" registers similarly but outputs new atlas files
        new_atlas = reg is config.RegisterTypes.NEW_ATLAS
        register(
            *config.filenames[:2], name_prefix=config.prefix,
            new_atlas=new_atlas, show_imgs=config.show)
    
    elif reg is config.RegisterTypes.GROUP:
        # groupwise registration, which assumes that the last image 
        # filename given is the prefix and uses the full flip array
        register_group(
            config.filenames[:-1], name_prefix=config.prefix,
            scale=config.transform[config.Transforms.RESCALE],
            show_imgs=config.show)
    
    elif reg is config.RegisterTypes.OVERLAYS:
        # overlay registered images in each orthogonal plane
        for out_plane in config.PLANE:
            overlay_registered_imgs(
                *config.filenames[0:2], plane=config.plane,
                name_prefix=config.prefix, out_plane=out_plane)
    
    elif reg is config.RegisterTypes.EXPORT_REGIONS:
        # export regions IDs to CSV files
        
        ref = ontology.load_labels_ref(config.load_labels)
        labels_ref_lookup = ontology.create_aba_reverse_lookup(ref)
        
        # export region IDs and parents at given level to CSV, using only
        # the atlas' labels if orig colors is true
        path = "region_ids"
        if config.filename:
            path = "{}_{}".format(path, config.filename)
        export_regions.export_region_ids(
            labels_ref_lookup, path, config.labels_level,
            config.atlas_labels[config.AtlasLabels.ORIG_COLORS])
        
        # export region IDs to network file
        export_regions.export_region_network(
            labels_ref_lookup, "region_network")
    
    elif reg is config.RegisterTypes.IMPORT_ATLAS:
        # import original atlas, mirroring if necessary
        atlas_refiner.import_atlas(config.filename, config.show, config.prefix)
    
    elif reg is config.RegisterTypes.EXPORT_COMMON_LABELS:
        # export common labels
        export_regions.export_common_labels(
            config.filenames, config.PATH_COMMON_LABELS)
    
    elif reg is config.RegisterTypes.CONVERT_ITKSNAP_LABELS:
        # convert labels from ITK-SNAP to CSV format
        df = ontology.convert_itksnap_to_df(config.filename)
        output_path = libmag.combine_paths(
            config.filename, ".csv", sep="")
        df_io.data_frames_to_csv([df], output_path)
    
    elif reg is config.RegisterTypes.EXPORT_METRICS_COMPACTNESS:
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
        df_histo_vs_orig, dfs_baseline = df_io.filter_dfs_on_vals(
            [df_stats, df_smoothing], cols, 
            [None, (config.SmoothingMetrics.FILTER_SIZE.value, 0)])
        df_histo_vs_orig[config.GENOTYPE_KEY] = "Histo Vs Orig Labels"
        
        # compare combined original vs smoothed labels
        smooth = config.atlas_profile["smooth"]
        df_unsm_vs_sm, dfs_smoothed = df_io.filter_dfs_on_vals(
            [dfs_baseline[1], df_smoothing], cols, 
            [None, (config.SmoothingMetrics.FILTER_SIZE.value, smooth)])
        df_unsm_vs_sm[config.GENOTYPE_KEY] = "Smoothing"

        # compare histology vs smoothed labels
        df_histo_vs_sm = pd.concat([dfs_baseline[0], dfs_smoothed[1]])
        df_histo_vs_sm[config.GENOTYPE_KEY] = "Vs Smoothed Labels"
        
        # export data frames
        output_path = libmag.combine_paths(
            config.filename, "compactness.csv")
        df = pd.concat([df_histo_vs_orig, df_histo_vs_sm, df_unsm_vs_sm])
        df[config.AtlasMetrics.REGION.value] = "all"
        df_io.data_frames_to_csv(df, output_path)
    
    elif reg is config.RegisterTypes.PLOT_SMOOTHING_METRICS:
        # plot smoothing metrics
        title = "{} Label Smoothing".format(
            libmag.str_to_disp(
                os.path.basename(config.filename).replace(
                    config.PATH_SMOOTHING_METRICS, "")))
        lbls = ("Fractional Change", "Smoothing Filter Size")
        plot_2d.plot_lines(
            config.filename, config.SmoothingMetrics.FILTER_SIZE.value, 
            (config.SmoothingMetrics.COMPACTION.value,
             config.SmoothingMetrics.DISPLACEMENT.value,
             config.SmoothingMetrics.SM_QUALITY.value), 
            ("--", "--", "-"), lbls, title, size, config.show, "_quality")
        plot_2d.plot_lines(
            config.filename, config.SmoothingMetrics.FILTER_SIZE.value, 
            (config.SmoothingMetrics.SA_VOL_FRAC.value,
             config.SmoothingMetrics.LABEL_LOSS.value), 
            ("-", "-"), lbls, None, size, config.show, "_extras", ("C3", "C4"))
    
    elif reg is config.RegisterTypes.SMOOTHING_PEAKS:
        # find peak smoothing qualities without label loss and at a given
        # filter size
        dfs = {}
        dfs_noloss = []
        key_filt = config.SmoothingMetrics.FILTER_SIZE.value
        for path in config.filenames:
            # load smoothing metrics CSV file and find peak smoothing qualities
            # without label loss across filter sizes
            df = pd.read_csv(path)
            df_peak = atlas_stats.smoothing_peak(df, 0, None)
            dfs_noloss.append(df_peak)
            
            # round extraneous decimals that may be introduced from floats
            df[key_filt] = df[key_filt].map(libmag.truncate_decimal_digit)
            filter_sizes = np.unique(
                df[config.SmoothingMetrics.FILTER_SIZE.value])
            for filter_size in filter_sizes:
                # extract smoothing quality at the given filter size
                dfs.setdefault(filter_size, []).append(
                    atlas_stats.smoothing_peak(df, None, filter_size))
        
        for key in dfs:
            # save metrics across atlases for given filter size
            df_io.data_frames_to_csv(
                dfs[key], "smoothing_filt{}.csv".format(key))
        
        # round peak filter sizes after extraction since sizes are now strings
        df_peaks = df_io.data_frames_to_csv(dfs_noloss)
        df_peaks[key_filt] = df_peaks[key_filt].map(
            libmag.truncate_decimal_digit)
        df_io.data_frames_to_csv(df_peaks, "smoothing_peaks.csv")

    elif reg is config.RegisterTypes.SMOOTHING_METRICS_AGGR:
        # re-aggregate smoothing metrics from raw stats
        df = pd.read_csv(config.filename)
        df_aggr = atlas_refiner.aggr_smoothing_metrics(df)
        df_io.data_frames_to_csv(df_aggr, config.PATH_SMOOTHING_METRICS)

    elif reg in (
            config.RegisterTypes.MAKE_EDGE_IMAGES,
            config.RegisterTypes.MAKE_EDGE_IMAGES_EXP):
        
        # convert atlas or experiment image and associated labels 
        # to edge-detected images; labels can be given as atlas dir from 
        # which labels will be extracted (eg import dir)
        atlas = reg is config.RegisterTypes.MAKE_EDGE_IMAGES
        for img_path in config.filenames:
            edge_seg.make_edge_images(
                img_path, config.show, atlas, config.suffix, config.load_labels)
    
    elif reg is config.RegisterTypes.REG_LABELS_TO_ATLAS:
        # register labels to its underlying atlas
        register_labels_to_atlas(config.filename)
    
    elif reg in (
            config.RegisterTypes.MERGE_ATLAS_SEGS,
            config.RegisterTypes.MERGE_ATLAS_SEGS_EXP):
        
        # merge various forms of atlas segmentations
        atlas = reg is config.RegisterTypes.MERGE_ATLAS_SEGS
        edge_seg.merge_atlas_segmentations(
            config.filenames, show=config.show, atlas=atlas,
            suffix=config.suffix)
    
    elif reg in (config.RegisterTypes.VOL_STATS,
                 config.RegisterTypes.VOL_COMPARE):
        # volumes stats
        labels_ref_lookup = ontology.create_aba_reverse_lookup(
            ontology.load_labels_ref(config.load_labels))
        groups = {}
        if config.groups is not None:
            groups[config.GENOTYPE_KEY] = [
                config.GROUPS_NUMERIC[geno] for geno in config.groups]
        # should generally leave uncombined for drawn labels to allow 
        # faster level building, where can combine sides
        combine_sides = config.atlas_profile["combine_sides"]
        if reg is config.RegisterTypes.VOL_STATS:
            # separate metrics for each sample
            extra_metric_groups = config.atlas_profile[
                "extra_metric_groups"]
            volumes_by_id(
                config.filenames, labels_ref_lookup, suffix=config.suffix, 
                unit_factor=config.unit_factor, groups=groups, 
                max_level=config.labels_level, combine_sides=combine_sides, 
                extra_metrics=extra_metric_groups)
        elif reg is config.RegisterTypes.VOL_COMPARE:
            # compare the given samples
            volumes_by_id_compare(
                config.filenames, labels_ref_lookup, 
                unit_factor=config.unit_factor, groups=groups,
                max_level=config.labels_level, combine_sides=combine_sides,
                offset=config.roi_offset, roi_size=config.roi_size)

    elif reg is config.RegisterTypes.MAKE_DENSITY_IMAGES:
        # make density images from blobs
        size = config.roi_sizes
        if size: size = size[0][::-1]
        export_regions.make_density_images_mp(
            config.filenames, config.transform[config.Transforms.RESCALE],
            size, config.suffix, config.channel)
    
    elif reg is config.RegisterTypes.MAKE_SUBSEGS:
        # make sub-segmentations for all images
        for img_path in config.filenames:
            edge_seg.make_sub_segmented_labels(img_path, config.suffix)

    elif reg is config.RegisterTypes.MERGE_IMAGES:
        # take mean of separate experiments from all paths using the 
        # given registered image type, defaulting to experimental images
        suffix = config.RegNames.IMG_EXP.value
        if config.reg_suffixes is not None:
            # use suffix assigned to atlas
            suffix_exp = config.reg_suffixes[config.RegSuffixes.ATLAS]
            if suffix_exp: suffix = suffix_exp
        sitk_io.merge_images(
            config.filenames, suffix, config.prefix, config.suffix)

    elif reg is config.RegisterTypes.MERGE_IMAGES_CHANNELS:
        # combine separate experiments from all paths into separate channels
        sitk_io.merge_images(
            config.filenames, config.RegNames.IMG_EXP.value, config.prefix, 
            config.suffix, fn_combine=None)

    elif reg is config.RegisterTypes.REGISTER_REV:
        # register a group of registered images to another image, 
        # such as the atlas to which the images were originally registered
        suffixes = None
        if config.reg_suffixes is not None:
            # get additional suffixes to register the same as for exp img
            suffixes = [config.reg_suffixes[key]
                        for key, val in config.reg_suffixes.items()
                        if config.reg_suffixes[key] is not None]
        register_rev(
            *config.filenames[:2], config.RegNames.IMG_EXP.value, suffixes, 
            config.plane, config.prefix, config.suffix, config.show)

    elif reg is config.RegisterTypes.MAKE_LABELS_LEVEL:
        # make a labels image grouped at the given level
        export_regions.make_labels_level_img(
            config.filename, config.labels_level, config.prefix, config.show)
    
    elif reg is config.RegisterTypes.LABELS_DIFF:
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
            export_regions.make_labels_diff_img(
                config.filename, path_df, *metric, config.prefix, config.show,
                config.labels_level, col_wt=col_wt)

    elif reg is config.RegisterTypes.LABELS_DIFF_STATS:
        # generate labels difference images for various measurements 
        # from a stats CSV generated by the R clrstats package
        metrics = (
            vols.LabelMetrics.EdgeDistSum.name,
            vols.LabelMetrics.CoefVarNuc.name,
            vols.LabelMetrics.CoefVarIntens.name,
            vols.LabelMetrics.NucCluster.name,
            vols.LabelMetrics.NucClusNoise.name,
            #vols.MetricCombos.HOMOGENEITY.value[0], 
        )
        for metric in metrics:
            path_df = "{}_{}.csv".format("vols_stats", metric)
            if not os.path.exists(path_df):
                print("{} does not exist for labels difference stats, skipping"
                      .format(metric))
                continue
            col_wt = vols.get_metric_weight_col(metric)
            export_regions.make_labels_diff_img(
                config.filename, path_df, "vals.effect", None, config.prefix, 
                config.show, meas_path_name=metric, col_wt=col_wt)
    
    elif reg is config.RegisterTypes.COMBINE_COLS:
        # normalize the given columns to original values in a data frame 
        # and combine columns for composite metrics
        df = pd.read_csv(config.filename)
        df = df_io.normalize_df(
            df, ["Sample", "Region"], "Condition", "original", 
            (vols.LabelMetrics.VarIntensity.name,
             vols.LabelMetrics.VarIntensDiff.name,
             vols.LabelMetrics.EdgeDistSum.name,
             vols.LabelMetrics.VarNuclei.name), 
            (vols.LabelMetrics.VarIntensity.Volume.name,))
        df = df_io.combine_cols(
            df, (vols.MetricCombos.HOMOGENEITY,))
        df_io.data_frames_to_csv(
            df, libmag.insert_before_ext(config.filename, "_norm"))

    elif reg is config.RegisterTypes.ZSCORES:
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
            (vols.MetricCombos.HOMOGENEITY,), size, config.show)
    
    elif reg is config.RegisterTypes.COEFVAR:
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
            combos, vols.LabelMetrics.Volume.name, size, config.show)

    elif reg is config.RegisterTypes.MELT_COLS:
        # melt columns specified in "groups" using ID columns from 
        # standard atlas metrics
        id_cols = [
            config.AtlasMetrics.SAMPLE.value, config.AtlasMetrics.REGION.value, 
            config.AtlasMetrics.CONDITION.value]
        df = df_io.melt_cols(
            pd.read_csv(config.filename), id_cols, config.groups, 
            config.GENOTYPE_KEY)
        df_io.data_frames_to_csv(
            df, libmag.insert_before_ext(config.filename, "_melted"))

    elif reg is config.RegisterTypes.PIVOT_CONDS:
        # pivot condition column to separate metric columns
        id_cols = [
            config.AtlasMetrics.SAMPLE.value, ]
        df = df_io.cond_to_cols_df(
            pd.read_csv(config.filename), id_cols, 
            config.AtlasMetrics.CONDITION.value, None, config.groups)
        df_io.data_frames_to_csv(
            df, libmag.insert_before_ext(config.filename, "_condtocol"))

    elif reg is config.RegisterTypes.PLOT_REGION_DEV:
        # plot region development
        try:
            metric = vols.LabelMetrics[config.df_task].name
            atlas_stats.plot_region_development(metric, size, config.show)
        except KeyError:
            libmag.warn(
                "\"{}\" metric not found in {} for developmental plots"
                .format(config.df_task, [e.name for e in vols.LabelMetrics]))
        
    elif reg is config.RegisterTypes.PLOT_LATERAL_UNLABELED:
        # plot lateral edge unlabeled fractions as both lines and bars
        cols = (config.AtlasMetrics.LAT_UNLBL_VOL.value,
                config.AtlasMetrics.LAT_UNLBL_PLANES.value)
        atlas_stats.plot_unlabeled_hemisphere(
            config.filename, cols, size, config.show)

    elif reg is config.RegisterTypes.PLOT_INTENS_NUC:
        # combine nuclei vs. intensity R stats and generate scatter plots
        labels = (config.plot_labels[config.PlotLabels.Y_LABEL],
                  config.plot_labels[config.PlotLabels.X_LABEL])
        # same label use for x/y and only in denominator
        unit = config.plot_labels[config.PlotLabels.X_UNIT]
        atlas_stats.plot_intensity_nuclei(
            config.filenames, labels, fig_size, config.show, unit)
    
    elif reg is config.RegisterTypes.MEAS_IMPROVEMENT:
        # measure summary improvement/worsening stats by label from R stats
        
        def meas(**args):
            atlas_stats.meas_improvement(
                config.filename, col_effect, "vals.pcorr", col_wt=col_wt,
                **args)
        
        col_idx = config.AtlasMetrics.REGION.value
        col_effect = "vals.effect"
        col_wt = config.plot_labels[config.PlotLabels.WT_COL]

        # improvement for all labels
        df = pd.read_csv(config.filename).set_index(col_idx)
        meas(df=df)
        
        # filter based on E18.5 atlas drawn labels
        df_e18 = df.loc[df[config.AtlasMetrics.LEVEL.value].isin([5, 7])]
        meas(suffix="_drawn", df=df_e18)
        
        if len(config.filenames) >= 2 and config.filenames[1]:
            # compare with reference stats from another image such as
            # the original (eg E18.5) atlas, filtering to keep only labels
            # that improved in the reference to see how many stats also
            # improve in the test, and likewise for worsened reference labels
            df_cp = pd.read_csv(config.filenames[1])
            df_cp_impr = df_cp[df_cp[col_effect] > 0].set_index(col_idx)
            df_cp_wors = df_cp[df_cp[col_effect] < 0].set_index(col_idx)
            for df_mode, suf in zip((df, df_e18), ("", "_drawn")):
                for df_cp_mode, suf_cp in zip(
                        (df_cp_impr, df_cp_wors), ("_e18_impr", "_e18_wors")):
                    meas(suffix="".join((suf, suf_cp)), 
                         df=df_mode.loc[df_mode.index.isin(df_cp_mode.index)])
    
    elif reg is config.RegisterTypes.PLOT_KNNS:
        # plot k-nearest-neighbor distances for multiple paths
        clustering.plot_knns(
            config.filenames, config.suffix, config.show,
            config.plot_labels[config.PlotLabels.LEGEND_NAMES])
    
    elif reg is config.RegisterTypes.CLUSTER_BLOBS:
        # cluster blobs and output to Numpy archive
        clustering.cluster_blobs(config.filename, config.suffix)
    
    elif reg is config.RegisterTypes.PLOT_CLUSTER_BLOBS:
        # show blob clusters for the given plane
        atlas_stats.plot_clusters_by_label(
            config.filename, config.roi_offsets[0][2], config.suffix,
            config.show, scaling)
    
    elif reg is config.RegisterTypes.LABELS_DIST:
        # measure distance between corresponding labels in two different
        # labels images
        if len(config.filenames) < 2:
            print("Please provide paths to 2 labels images")
            return
        labels_imgs = [sitk_io.read_sitk_files(p) for p in config.filenames[:2]]
        spacing = scaling
        if spacing is None and len(config.resolutions) > 0:
            # default to using loaded metadata
            spacing = config.resolutions[0]
        out_path = libmag.make_out_path(libmag.combine_paths(
            config.filename, "labelsdist.csv"))
        # base name of first image filename
        sample = libmag.get_filename_without_ext(config.filename)
        vols.labels_distance(*labels_imgs[:2], spacing, out_path, sample)

    else:
        print("Could not find register task:", reg)


if __name__ == "__main__":
    print("MagellanMapper image registration")
    cli.main(True)
    main()
