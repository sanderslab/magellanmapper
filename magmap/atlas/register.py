#!/usr/bin/env python
# Image registration
# Author: David Young, 2017, 2023
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
import dataclasses
import os
import shutil
from time import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

try:
    import itk
except ImportError:
    itk = None
import pandas as pd
import numpy as np
try:
    import SimpleITK as sitk
except ImportError:
    sitk = None
from skimage import filters, measure, morphology, transform

from magmap.atlas import atlas_refiner, edge_seg, labels_meta, ontology, \
    reg_tasks, transformer
from magmap.cv import cv_nd
from magmap.io import cli, df_io, export_regions, importer, libmag, sitk_io
from magmap.plot import plot_2d, plot_3d
from magmap.settings import atlas_prof, config
from magmap.stats import atlas_stats, clustering, vols

SAMPLE_VOLS = "vols_by_sample"
SAMPLE_VOLS_LEVELS = SAMPLE_VOLS + "_levels"

REREG_SUFFIX = "rereg"

# 3D format extensions to check when finding registered files
_SIGNAL_THRESHOLD = 0.01

# sort volume columns
_SORT_VOL_COLS = [
    config.AtlasMetrics.REGION.value,
    config.AtlasMetrics.SAMPLE.value,
    config.AtlasMetrics.SIDE.value,
]

_logger = config.logger.getChild(__name__)


@dataclasses.dataclass
class RegImgs:
    """Data class for tracking registered images."""
    #: Original experimental/fixed image.
    exp_orig: Optional[Union[np.ndarray, "sitk.Image", "itk.Image"]] = None
    #: Experimental/fixed image.
    exp: Optional[Union[np.ndarray, "sitk.Image", "itk.Image"]] = None
    #: Atlas/moving image.
    atlas: Optional[Union[np.ndarray, "sitk.Image", "itk.Image"]] = None
    #: Atlas/moving labels image.
    labels: Optional[Union[np.ndarray, "sitk.Image", "itk.Image"]] = None
    #: Labels markers image.
    labels_markers: Optional[Union[
        np.ndarray, "sitk.Image", "itk.Image"]] = None
    #: Borders image.
    borders: Optional[Union[np.ndarray, "sitk.Image", "itk.Image"]] = None
    #: Experimental/fixed mask image.
    exp_mask: Optional[Union[np.ndarray, "sitk.Image", "itk.Image"]] = None
    #: Atlas/moving mask image.
    atlas_mask: Optional[Union[np.ndarray, "sitk.Image", "itk.Image"]] = None
    
    @staticmethod
    def get_order(name: str) -> Optional[int]:
        """Get interpolation order for the given image type.
        
        Args:
            name: Image field name.

        Returns:
            0 if the image is a labels-related image, otherwise None.

        """
        return 0 if name in ("labels", "labels_markers", "borders") else None
    
    def update_fields(
            self, fn: Callable[
                [Optional[Union[np.ndarray, "sitk.Image", "itk.Image"]]],
                Optional[Union[np.ndarray, "sitk.Image", "itk.Image"]]]):
        """Update all fields.
        
        Args:
            fn: Function to apply to each field if not None.

        """
        for field in dataclasses.fields(RegImgs):
            reg_img = getattr(self, field.name)
            if reg_img is not None:
                reg_img = fn(reg_img)
            setattr(self, field.name, reg_img)


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
    # read or write transformation file
    base_name = sitk_io.reg_out_path(fixed_file, "")
    filename = base_name + "transform.txt"
    if transform_param_map is None:
        if sitk:
            param_map = sitk.ReadParameterFile(filename)
        else:
            param_map = itk.ParameterObject.New()
            param_map.ReadParameterFile(filename)
    else:
        if sitk:
            sitk.WriteParameterFile(transform_param_map[0], filename)
        else:
            param = itk.ParameterObject.New()
            param.WriteParameterFile(transform_param_map[0], filename)
        param_map = transform_param_map[0]
    return param_map, None # TODO: not using translation parameters
    transform = np.array(param_map["TransformParameters"]).astype(float)
    spacing = np.array(param_map["Spacing"]).astype(float)
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
    return param_map, None  # TODO: not using translation parameters


def curate_img(
        fixed_img: "sitk.Image", labels_img: "sitk.Image",
        imgs: Optional[List["sitk.Image"]] = None, inpaint: bool = True,
        carve: bool = True, thresh: Optional[float] = None,
        holes_area: Optional[int] = None) -> List["sitk.Image"]:
    """Curate an image by the foreground of another.
    
    In-painting where corresponding pixels are present in fixed image but
    not labels or other images and removing pixels
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
        thresh: Threshold to use for carving; defaults to None to
            determine by taking the mean threshold of ``fixed_img``.
        holes_area: Maximum area of holes to fill when carving.
    
    Returns:
        A list of images in SimpleITK format that have been curated, starting
        with the curated ``labels_img``, followed by the images in ``imgs``.
    
    """
    fixed_img_np = sitk_io.convert_img(fixed_img)
    labels_img_np = sitk_io.convert_img(labels_img)
    
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
        result_img_np = sitk_io.convert_img(img)
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


def register_repeat(
        transformix_img_filter: Union[
            "sitk.TransformixImageFilter", "itk.TransformixFilter"],
        img: Union["sitk.Image", "itk.Image"], preserve_idents: bool = False
) -> Union["sitk.Image", "itk.Image"]:
    """Transform labels to match a prior registration.
    
    Uses an Elastix Transformix filter to reproduce a transformation on
    a labels image. This filter can be from SimpleITK or ITK. Ensures that
    the output pixel type remains the same as the input.
    
    Args:
        transformix_img_filter: Filter generated from a prior registration.
        img: SimpleITK or ITK image.
        preserve_idents: True to ensure that identities in ``img`` are
            preserved. Typically used for label images. Defaults to False.

    Returns:
        Transformed image.

    """
    # use SimpleITK if it has Elastix and filter is from it
    is_sitk = sitk and "TransformixImageFilter" in dir(sitk) and isinstance(
        transformix_img_filter, sitk.TransformixImageFilter)
    
    # check if input image is sitk type
    is_sitk_img = sitk and isinstance(img, sitk.Image)
    
    if is_sitk:
        # store data type
        pixel_id = img.GetPixelID()
    else:
        if is_sitk_img:
            # convert to ITK Image
            img = sitk_io.sitk_to_itk_img(img)
        
        # cast to required data type for ITK transformation
        pixel_id = type(img)
        img = img.astype(itk.F)
    
    img_sitk = img
    img_unique = None
    if preserve_idents:
        # map values to indices in their unique array to minimize change of
        # rounding errors for large values
        img_np = sitk_io.convert_img(img_sitk)
        img_unique, img_inds = np.unique(img_np, return_inverse=True)
        
        # offset by 1 since background will be inserted as 0
        img_inds = img_inds.reshape(img_np.shape) + 1
        img_sitk = sitk_io.replace_sitk_with_numpy(
            img, img_inds.astype(np.float32))
    
    if is_sitk:
        # reapply transformation using sitk
        transformix_img_filter.SetMovingImage(img_sitk)
        transformix_img_filter.Execute()
        transf_img = transformix_img_filter.GetResultImage()
        transf_img = sitk.Cast(transf_img, pixel_id)
    else:
        # reapply transformation using ITK
        
        # set up new filter since it needs to be set up with the image
        # to support 3D images
        params = transformix_img_filter.GetTransformParameterObject()
        transformix_img_filter = itk.TransformixFilter.New(img_sitk)
        transformix_img_filter.SetTransformParameterObject(params)
        
        # perform the transformation
        transformix_img_filter.UpdateLargestPossibleRegion()
        transf_img = transformix_img_filter.GetOutput()
        cast_filter = itk.CastImageFilter[type(transf_img), pixel_id].New()
        cast_filter.SetInput(transf_img)
        transf_img = cast_filter.GetOutput()
    
    if preserve_idents:
        # map indices back to original values
        transf_inds = sitk_io.convert_img(transf_img)
        transf_inds = transf_inds.astype(int)
        bkgdi = np.nonzero(img_unique == 0)
        if len(bkgdi) > 0:
            # convert inserted background (0) to background index
            transf_inds[transf_inds == 0] = bkgdi[0] + 1
        transf_inds -= 1
        transf_np = img_unique[transf_inds]
        transf_img = sitk_io.replace_sitk_with_numpy(transf_img, transf_np)
    
    _logger.info(f"Transformed image:\n{transf_img}")
    '''
    LabelStatistics = sitk.LabelStatisticsImageFilter()
    LabelStatistics.Execute(fixed_img, labels_img)
    count = LabelStatistics.GetCount(1)
    mean = LabelStatistics.GetMean(1)
    variance = LabelStatistics.GetVariance(1)
    print("count: {}, mean: {}, variance: {}".format(count, mean, variance))
    '''
    
    if not is_sitk and is_sitk_img:
        transf_img = sitk_io.itk_to_sitk_img(transf_img)
    return transf_img


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


def register_duo(
        fixed_img: Union["sitk.Image", "itk.Image"],
        moving_img: Union["sitk.Image", "itk.Image"],
        path: Optional[str] = None,
        fixed_mask: Optional[Union["sitk.Image", "itk.Image"]] = None,
        moving_mask: Optional[Union["sitk.Image", "itk.Image"]] = None,
        regs: Optional[Union[
            Sequence[str], Sequence["atlas_prof.RegParamMap"]]] = None
) -> Tuple[Union["sitk.Image", "itk.Image"],
           Union["sitk.TransformixImageFilter", "itk.TransformixFilter"]]:
    """Register two images to one another using ``Elastix``.
    
    Supports Elastix provided by SimpleITK or ITK. If SimpleITK images are
    given with Elastix enabled in SimpleITK, registration will be performed
    in SimpleITK. If Elastix is not enabled, however, the images will be
    converted to ITK, and registration will use ITK-Elastix. If ITK images
    are given, ITK-Elastix is assumed to be present; no conversion will take
    place to SimpleITK.
    
    Args:
        fixed_img: The image to be registered to.
        moving_img: The image to register to ``fixed_img``.
        path: Path as string from whose parent directory the points-based
            registration files ``fix_pts.txt`` and ``mov_pts.txt`` will
            be found. If None, points-based reg will be ignored even if set.
        fixed_mask: Mask for ``fixed_img``, typically a uint8 image.
        moving_mask: Mask for ``moving_img``, typically a uint8 image.
        regs: Sequence of atlas profile registration keys or registration
            parameter objects. The default of None gives all three major
            registration types, "reg_translation", "reg_affine", "reg_bspline".
    
    Returns:
        Tuple of the registered image and a Transformix filter with the
        registration's parameters to reapply them on other images.
    
    """
    if regs is None:
        # default to perform all the major registration types
        regs = ("reg_translation", "reg_affine", "reg_bspline")
    
    # collect images
    reg_imgs = RegImgs(
        exp=fixed_img, atlas=moving_img, exp_mask=fixed_mask,
        atlas_mask=moving_mask)
    
    if not itk and not sitk:
        raise ImportError(
            config.format_import_err(
                "itk-elastix", "ITK-Elastix or SimpleITK with Elastix",
                "image registration"))
    
    is_sitk_img = sitk and isinstance(reg_imgs.exp, sitk.Image)
    if itk and isinstance(reg_imgs.exp, itk.Image):
        # use ITK-Elastix if images are of ITK type
        is_sitk = False
        
    elif is_sitk_img:
        # images are of SimpleITK type
        if "ElastixImageFilter" in dir(sitk):
            # use SimpleITK if Elastix is enabled
            is_sitk = True
        else:
            # fall back to ITK-Elastix
            is_sitk = False
            _logger.debug(
                "Converting SimpleITK to ITK images since Elastix was not "
                "found in the SimpleITK library")
            
            # convert sitk to ITK images
            reg_imgs.update_fields(
                lambda x: sitk_io.sitk_to_itk_img(x))
        
    else:
        raise TypeError(
            f"Images must be ITK or SimpleITK Image types, but 'fixed_img' is "
            f"of type {type(reg_imgs.exp)}")
    
    _logger.info(f"Fixed image:\n{reg_imgs.exp}")
    _logger.info(f"Moving image:\n{reg_imgs.atlas}")
    
    param_map_vector = None  # for sitk
    reg_params = None  # for ITK
    if is_sitk:
        # set up SimpleElastix filter
        _logger.info("Registering images using SimpleITK with Elastix")
        elastix_img_filter = sitk.ElastixImageFilter()
        elastix_img_filter.SetFixedImage(reg_imgs.exp)
        elastix_img_filter.SetMovingImage(reg_imgs.atlas)
        param_map_vector = sitk.VectorOfParameterMap()
        
        # add any masks
        if reg_imgs.exp_mask is not None:
            elastix_img_filter.SetFixedMask(reg_imgs.exp_mask)
        if reg_imgs.atlas_mask is not None:
            elastix_img_filter.SetMovingMask(reg_imgs.atlas_mask)
        
    else:
        # set up ITK-Elastix filter
        _logger.info("Registering images using ITK-Elastix")
        
        # main images must be float
        reg_imgs.exp = reg_imgs.exp.astype(itk.F)
        reg_imgs.atlas = reg_imgs.atlas.astype(itk.F)
        elastix_img_filter = itk.ElastixRegistrationMethod.New(
            reg_imgs.exp, reg_imgs.atlas)
        reg_params = itk.ParameterObject.New()
        
        # masks must be unsigned char
        if reg_imgs.exp_mask:
            reg_imgs.exp_mask = reg_imgs.exp_mask.astype(itk.UC)
            elastix_img_filter.SetFixedMask(reg_imgs.exp_mask)
        if reg_imgs.atlas_mask:
            reg_imgs.atlas_mask = reg_imgs.atlas_mask.astype(itk.UC)
            elastix_img_filter.SetMovingMask(reg_imgs.atlas_mask)
    
    # set up parameter maps for the included registration types
    settings = config.atlas_profile
    for reg in regs:
        # get registration parameters from profile
        params = reg if isinstance(
            reg, atlas_prof.RegParamMap) else settings[reg]
        if not params: continue
        
        max_iter = params["max_iter"]
        # TODO: Consider removing since does not skip if "0" and need at least
        #   one transformation for reg, even if 0 iterations. Also, note that
        #   turning all iterations to 0 since simply running through the filter
        #   will still alter images.
        if not max_iter: continue
        
        reg_name = params["map_name"]
        if reg_params:
            param_map = reg_params.GetDefaultParameterMap(reg_name)
        else:
            param_map = sitk.GetDefaultParameterMap(reg_name)
        
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
                grid_spacing_sched, param_map, reg_imgs.exp.GetDimension())
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
                # set point files if both exist
                metric = list(param_map["Metric"])
                metric.append("CorrespondingPointsEuclideanDistanceMetric")
                param_map["Metric"] = metric
                #param_map["Metric2Weight"] = ["0.5"]
                elastix_img_filter.SetFixedPointSetFileName(fix_pts_path)
                elastix_img_filter.SetMovingPointSetFileName(move_pts_path)
        
        if param_map_vector is not None:
            param_map_vector.append(param_map)
        elif reg_params is not None:
            reg_params.AddParameterMap(param_map)
    
    if is_sitk:
        # perform registration in sitk
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

    else:
        # perform registration in ITK-Elastix
        elastix_img_filter.SetParameterObject(reg_params)
        elastix_img_filter.SetLogToConsole(config.verbose)
        elastix_img_filter.UpdateLargestPossibleRegion()
        transformed_img = elastix_img_filter.GetOutput()
        transf_params = elastix_img_filter.GetTransformParameterObject()
        
        # set up transformix
        transf_params.SetParameter(
            transf_params.GetNumberOfParameterMaps() - 1,
            "FinalBSplineInterpolationOrder", "0")
        transformix_img_filter = itk.TransformixFilter.New()
        transformix_img_filter.SetTransformParameterObject(transf_params)
        
        if is_sitk_img:
            # convert back to sitk Image
            transformed_img = sitk_io.itk_to_sitk_img(transformed_img)
    
    return transformed_img, transformix_img_filter


def register(
        fixed_file: str, moving_img_path: str, show_imgs: bool = True,
        write_imgs: bool = True, name_prefix: Optional[str] = None,
        new_atlas: bool = False,
        transformix: Optional[Union[
            "sitk.TransformixImageFilter", "itk.TransformixFilter"]] = None
) -> Union["sitk.TransformixImageFilter", "itk.TransformixFilter"]:
    """Register an atlas to a sample image using the Elastix library.
    
    Loads the images, applies any transformations to the moving image, and
    registers the moving to the sample images. Applies the identical
    registration to the associated atlas labels image as well. The specific
    moving images can be adjusted through
    :attr:`magmap.settings.config.reg_suffixes`, where the "atlas" image is
    the intensity image used for registration, "annotation" is the atlas
    labels, "fixed_mask" is the mask for ``fixed_file`` (optional), and
    "moving_mask" is the mask for the "atlas" image (optional). If either
    mask is given, both should be given as required by Elastix. Multiple
    "annotation" images can be given, which will each be transformed
    identially to the "atlas" image.
    
    Uses the first channel in :attr:`config.channel`, or the first image
    channel if this value is None.
    
    Fallbacks:
        * If the registration does not complete, the moving image origin and
          direction will be changed to that of the fixed image
        * If no reg, the spacing will also be matched
        * If still no reg, the atlas will be output as-is
        * If the atlas profile ``fallback`` parameter sets a DSC threshold
          and alternate registration settings, the image will be re-registered
          under these new settings if below the threshold
    
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
        transformix: Elastix transformation filter from prior run;
            defaults to None.
    Returns:
        The Elastix transformation filter generated by the registration.
    
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
    rescale = config.atlas_profile["rescale"]
    
    # set up rotation by arbitrary degrees
    rotate = config.atlas_profile["rotate"]
    rotate_deg = None
    if rotate and rotate["rotation"]:
        # convert profile dict to kwargs for transpose fn; skip order field
        # since it depends on whether image is a label type
        rotate_deg = [
            dict(angle=a, axis=x, resize=rotate["resize"])
            for a, x in rotate["rotation"]]
    
    # load fixed image, assumed to be experimental image
    chl = config.channel[0] if config.channel else 0
    fixed_img = sitk_io.load_numpy_to_sitk(fixed_file, channel=chl)
    
    # preprocess fixed image; store original fixed image for overlap measure
    # TODO: assume fixed image is preprocessed before starting this reg?
    fixed_img_orig = fixed_img
    if settings["preprocess"]:
        img_np = sitk_io.convert_img(fixed_img)
        #img_np = plot_3d.saturate_roi(img_np)
        img_np = plot_3d.denoise_roi(img_np)
        fixed_img = sitk_io.replace_sitk_with_numpy(fixed_img, img_np)

    # load moving intensity image based on registered image suffixes, falling
    # back to atlas volume suffix
    moving_atlas_suffix = config.reg_suffixes[config.RegSuffixes.ATLAS]
    if not moving_atlas_suffix:
        moving_atlas_suffix = config.RegNames.IMG_ATLAS.value
    moving_img = sitk_io.load_registered_img(
        moving_img_path, moving_atlas_suffix, get_sitk=True)
    
    # load the moving labels images
    moving_labels_suffix = config.reg_suffixes[config.RegSuffixes.ANNOTATION]
    if not moving_labels_suffix:
        moving_labels_suffix = [config.RegNames.IMG_LABELS.value]
    if libmag.is_seq(moving_labels_suffix):
        # load all label images and set first as the labels_img
        moving_imgs = sitk_io.load_registered_imgs(
            moving_img_path, moving_labels_suffix, get_sitk=True)
        labels_img = tuple(moving_imgs.values())[0]
    else:
        # load single labels image
        labels_img = sitk_io.load_registered_img(
            moving_img_path, moving_labels_suffix, get_sitk=True)
        moving_imgs = {moving_labels_suffix: labels_img}

    # get image masks given as registered image suffixes relative to the
    # fixed image path or prefix
    fixed_mask = None
    moving_mask = None
    fixed_mask_suffix = config.reg_suffixes[config.RegSuffixes.FIXED_MASK]
    if fixed_mask_suffix:
        fixed_mask = sitk_io.load_registered_img(
            name_prefix, fixed_mask_suffix, get_sitk=True)
    moving_mask_suffix = config.reg_suffixes[config.RegSuffixes.MOVING_MASK]
    if moving_mask_suffix:
        moving_mask = sitk_io.load_registered_img(
            name_prefix, moving_mask_suffix, get_sitk=True)
        moving_mask = atlas_refiner.transpose_img(
            moving_mask, target_size_res=rescale, rotate_deg=rotate_deg,
            order=0)
    
    truncate_labels = settings["truncate_labels"]
    if truncate_labels is not None:
        # generate a truncated/cropped version of the labels image
        labels_trunc_np = sitk_io.convert_img(labels_img)
        atlas_refiner.truncate_labels(labels_trunc_np, *truncate_labels)
        labels_trunc = sitk_io.replace_sitk_with_numpy(
            labels_img, labels_trunc_np)
        moving_imgs["trunc"] = labels_trunc
    
    for key, img in moving_imgs.items():
        _logger.info("Transposing image: %s", key)
        moving_imgs[key] = atlas_refiner.transpose_img(
            img, target_size_res=rescale, rotate_deg=rotate_deg, order=0)
    
    # transform and preprocess moving images

    # transpose moving images
    # TODO: track all images in moving_imgs and transpose together for
    #   simplicity and to avoid redundant transformations
    moving_img = atlas_refiner.transpose_img(
        moving_img, target_size_res=rescale, rotate_deg=rotate_deg)
    labels_img = atlas_refiner.transpose_img(
        labels_img, target_size_res=rescale, rotate_deg=rotate_deg, order=0)

    # get Numpy arrays of moving images for preprocessing
    moving_img_np = sitk_io.convert_img(moving_img)
    labels_img_np = sitk_io.convert_img(labels_img)
    moving_mask_np = None
    if moving_mask is not None:
        moving_mask_np = sitk_io.convert_img(moving_mask)

    crop_out_labels = config.atlas_profile["crop_out_labels"]
    if crop_out_labels is not None:
        # crop moving images to extent without given labels; note that
        # these labels may still exist within the cropped image
        mask = np.zeros_like(labels_img_np, dtype=np.uint8)
        mask[labels_img_np != 0] = 1
        mask[np.isin(labels_img_np, crop_out_labels)] = 0
        labels_img_np, moving_img_np, _ = cv_nd.crop_to_labels(
            labels_img_np, moving_img_np, mask, 0, 0)

    # convert images back to sitk format
    labels_img = sitk_io.replace_sitk_with_numpy(labels_img, labels_img_np)
    moving_img = sitk_io.replace_sitk_with_numpy(moving_img, moving_img_np)
    moving_imgs["atlas"] = moving_img
    moving_imgs["annot"] = labels_img
    if moving_mask is not None:
        moving_mask = sitk_io.replace_sitk_with_numpy(
            moving_mask, moving_mask_np)
        moving_imgs["mask"] = moving_mask

    thresh_mov = settings["atlas_threshold"]
    if transformix:
        # re-transform using parameters from a prior registration
        transformix_filter = transformix
        img_moved = register_repeat(transformix_filter, moving_img)
        dsc_sample = atlas_refiner.measure_overlap(
            fixed_img_orig, img_moved, thresh_img2=thresh_mov)
        metric_sim = get_similarity_metric()
    
    else:
        # perform a new registration
        transformix_filter = None
        try:
            img_moved, transformix_filter = register_duo(
                fixed_img, moving_img, name_prefix, fixed_mask, moving_mask)
        except RuntimeError:
            # fall back to match some world info
            _logger.warn(
                "Could not perform registration. Will retry with origin and "
                "direction set to that of the fixed image.")
            imgs = list(moving_imgs.values())
            if fixed_mask is not None:
                imgs.append(fixed_mask)
            for img in imgs:
                sitk_io.match_world_info(fixed_img, img, spacing=False)
            try:
                img_moved, transformix_filter = register_duo(
                    fixed_img, moving_img, name_prefix, fixed_mask, moving_mask)
            except RuntimeError:
                # fall back to match all world info
                _logger.warn(
                    "Could not perform registration. Will retry with spacing, "
                    "origin and direction set to that of the fixed image.")
                for img in imgs:
                    sitk_io.match_world_info(fixed_img, img)
                try:
                    img_moved, transformix_filter = register_duo(
                        fixed_img, moving_img, name_prefix, fixed_mask, moving_mask)
                except RuntimeError:
                    # output atlas as-is, including world info matching
                    _logger.warn(
                        "Could not perform the registration despite matching world "
                        "info. Will output images as-is unless a fallback reg is "
                        "set in the atlas profile.")
                    img_moved = moving_img
    
        # overlap stats comparing original and registered samples (eg histology)
        _logger.info("DSC of original and registered sample images")
        thresh_mov = settings["atlas_threshold"]
        dsc_sample = atlas_refiner.measure_overlap(
            fixed_img_orig, img_moved, thresh_img2=thresh_mov)
        fallback = settings["metric_sim_fallback"]
        metric_sim = get_similarity_metric()
        if fallback and dsc_sample < fallback[0]:
            # fall back to another atlas profile; update the current profile with
            # this new profile and re-set-up the original profile afterward
            # TODO: consider whether to reset world info
            _logger.info(
                f"Registration DSC below threshold of {fallback[0]}, will "
                f"re-register using {fallback[1]} atlas profile")
            cli.setup_atlas_profiles(fallback[1], reset=False)
    
    def make_labels(lbls_img):
        # apply the same transformation to labels via Transformix
        
        if transformix_filter is None:
            return lbls_img, None, None, None
        # transform label
        labels_trans = register_repeat(transformix_filter, lbls_img, True)
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
            # curate labels based on thresholded intensity image
            # TODO: consider moving out curation to separate task
            thresh_carve = settings["carve_threshold"]
            if isinstance(thresh_carve, str):
                # get threshold from another setting, eg atlas_threshold_all
                thresh_carve = settings[thresh_carve]
            labels_trans_cur, transformed_img_cur = curate_img(
                fixed_img_orig, labels_trans, [img_moved], inpaint=new_atlas,
                thresh=thresh_carve, holes_area=settings["holes_area"])
            _logger.info(
                "DSC of original and registered sample images after curation")
            dsc = atlas_refiner.measure_overlap(
                fixed_img_orig, transformed_img_cur, 
                thresh_img2=settings["atlas_threshold"])
        return labels_trans, labels_trans_cur, transformed_img_cur, dsc

    # apply same transformation to labels, +/- curation to carve the moving
    # image where the fixed image does not exist or in-paint where it does
    labels_moved, labels_moved_cur, img_moved_cur, dsc_sample_cur = (
        make_labels(labels_img))
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
    
    if truncate_labels is not None:
        # transform cropped image +/- curation
        _logger.info("Transforming cropped/truncated labels image")
        truncted_imgs = make_labels(moving_imgs["trunc"])
        imgs_write[config.RegNames.IMG_LABELS_TRUNC.value] = truncted_imgs[0]
        imgs_write[config.RegNames.IMG_LABELS_TRUNC_PRECUR.value] = truncted_imgs[1]
    if len(moving_labels_suffix) > 1:
        print("transforming rest of labels", moving_labels_suffix)
        for suffix in moving_labels_suffix[1:]:
            if suffix not in moving_imgs: continue
            _logger.info("Transforming image: %s", moving_imgs[suffix])
            imgs_write[suffix] = make_labels(moving_imgs[suffix])[0]
            print("transforming label", suffix, imgs_write[suffix])

    if show_imgs and sitk:
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
        
        lbls_meta = labels_meta.LabelsMeta(moving_img_path).load()
        if os.path.exists(lbls_meta.save_path):
            # save labels metadata file to output directory
            lbls_meta.prefix = name_prefix
            lbls_meta.save()

    # compare original atlas with registered labels taken as a whole
    dsc_labels = atlas_refiner.measure_overlap_combined_labels(
        fixed_img_orig, labels_moved)
    
    # measure compactness of fixed image
    fixed_img_orig_np = sitk_io.convert_img(fixed_img_orig)
    thresh_atlas = fixed_img_orig_np > filters.threshold_mean(fixed_img_orig_np)
    compactness, _, _ = cv_nd.compactness_3d(
        thresh_atlas, tuple(fixed_img_orig.GetSpacing())[::-1])
    
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
    _logger.info("\nImported %s whole atlas stats:", basename)
    df_io.dict_to_data_frame(metrics, df_path, show="\t")

    '''
    # save transform parameters and attempt to find the original position
    # that corresponds to the final position that will be displayed
    if transformix_filter is not None:
        _, translation = _handle_transform_file(
            name_prefix, transformix_filter.GetTransformParameterMap())
        _translation_adjust(moving_img, img_moved, translation, flip=True)
    
    # show 2D overlays or registered image and atlas last since blocks until 
    # fig is closed
    imgs = [
        sitk.GetArrayFromImage(fixed_img),
        sitk.GetArrayFromImage(moving_img), 
        sitk.GetArrayFromImage(img_moved), 
        sitk.GetArrayFromImage(labels_img)]
    _show_overlays(imgs, translation, fixed_file, None)
    '''
    
    _logger.info(
        "Time elapsed for single registration (s): %s", time() - start_time)
    if transformix_filter is None:
        _logger.warn(
            "Unable to perform the registration. Atlas images were output "
            "with only pre-registration transformations.")
    return transformix_filter


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
    fixed_img = sitk_io.read_img(fixed_path)
    mod_path = moving_path
    if suffix is not None:
        # adjust image path to load with suffix
        mod_path = libmag.insert_before_ext(mod_path, suffix)
    if reg_base is None:
        # load the image directly from given path
        moving_img = sitk_io.read_img(mod_path)
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
    if show and sitk:
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


def register_group(
        img_files: Sequence[str], rotate: Optional[Sequence[int]] = None,
        show_imgs: bool = True, write_imgs: bool = True,
        name_prefix: Optional[str] = None, scale: Optional[float] = None):
    """Group registers several images to one another.
    
    Uses the first channel in :attr:`config.channel` or the first channel
    in each image.
    
    Registration parameters are assumed to be in a "b-spline"
    :class:`magmap.settings.atlas_prof.RegParamMap`.
    
    Performs registration using SimpleITK if the package is present, otherwise
    using ITK-Elastix.
    
    Args:
        img_files: Paths to image files to register. A minimum of 4 images
            is required for groupwise registration.
        rotate: List of number of 90 degree rotations for images
            corresponding to ``img_files``; defaults to None, in which
            case the `config.transform` rotate attribute will be used.
        show_imgs: True if the output images should be displayed.
        write_imgs: True if the images should be written to file.
        name_prefix: Path with base name where registered files will be output;
            defaults to None, in which case the fixed_file path will be used.
        scale: Rescaling factor as a scalar value, used to find the rescaled,
            smaller images corresponding to ``img_files``.
    
    """
    start_time = time()
    if name_prefix is None:
        name_prefix = img_files[0]
    if rotate is None:
        rotate = config.transform[config.Transforms.ROTATE]
    target_size = config.atlas_profile["target_size"]
    
    if not itk and not sitk:
        raise ImportError(
            config.format_import_err(
                "itk-elastix", "ITK-Elastix or SimpleITK with Elastix",
                "groupwise image registration"))
    
    imgs = None
    img_vector = None  # for sitk
    reg_params = None  # for ITK
    if sitk:
        # register with SimpleITK if present
        _logger.info("Groupwise registering images using SimpleITK with Elastix")
        img_vector = sitk.VectorOfImage()
        
    else:
        # set up ITK-Elastix parameters
        _logger.info("Groupwise registering images using ITK-Elastix")
        reg_params = itk.ParameterObject.New()
        imgs = []
    
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
        img = sitk_io.load_numpy_to_sitk(img_file, rot, chl)
        size = img.GetSize() if sitk else itk.size(img)
        img_np = sitk_io.convert_img(img)
        if img_np_template is None:
            img_np_template = np.copy(img_np)

        y_cropped = 0
        try:
            # crop y-axis based on registered labels so that sample images,
            # which appears to work better than erasing for groupwise reg by
            # preventing some images from being stretched into the erased space
            labels_img = sitk_io.load_registered_img(
                img_files[i], config.RegNames.IMG_LABELS_TRUNC.value)
            _logger.info("Cropping image based on labels trunction image")
            img_np, y_cropped = _crop_image(img_np, labels_img, 1)#, eraser=0)
        except FileNotFoundError:
            pass
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
            size_cropped = img.GetSize() if sitk else tuple(itk.size(img))
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
        
        if img_vector is not None:
            img_vector.push_back(img)
        else:
            imgs.append(img)
    
    #sitk.ProcessObject.SetGlobalDefaultDirectionTolerance(1)
    #sitk.ProcessObject.SetGlobalDefaultCoordinateTolerance(100)
    
    if img_vector is not None:
        # sitk provides an array to register images of different sizes
        img_combined = sitk.JoinSeries(img_vector)
        elastix_img_filter = sitk.ElastixImageFilter()
        elastix_img_filter.SetFixedImage(img_combined)
        elastix_img_filter.SetMovingImage(img_combined)
    else:
        # assume all images are of the same size for ITK
        imgs_np = [sitk_io.convert_img(m) for m in imgs]
        imgs_np = np.array(imgs_np)
        img_combined = sitk_io.convert_img(imgs_np, False).astype(itk.F)
        sitk_io.match_world_info(imgs[0], img_combined)
        elastix_img_filter = itk.ElastixRegistrationMethod.New(
            img_combined, img_combined)
    
    _logger.info("Images to groupwise register:\n%s", img_combined)
    
    # get custom parameters from b-spline registration profile
    settings = config.atlas_profile
    reg = settings["reg_bspline"]
    
    # get default parameters for groupwise registeration
    reg_name = "groupwise"
    if reg_params:
        param_map = reg_params.GetDefaultParameterMap(reg_name)
    else:
        param_map = sitk.GetDefaultParameterMap(reg_name)
    
    # change spacing to voxels
    param_map["FinalGridSpacingInVoxels"] = [reg["grid_space_voxels"]]
    del param_map["FinalGridSpacingInPhysicalUnits"]  # avoid conflict with vox
    
    # set iterations
    param_map["MaximumNumberOfIterations"] = [reg["max_iter"]]
    # TESTING:
    # param_map["MaximumNumberOfIterations"] = ["0"]
    
    # add a set of axis resolutions for each resolution level
    _config_reg_resolutions(
        reg["grid_spacing_schedule"], param_map, img_np_template.ndim)
    
    if sitk:
        # groupwise register images in sitk
        elastix_img_filter.SetParameterMap(param_map)
        elastix_img_filter.PrintParameterMap()
        elastix_img_filter.Execute()
        transformed_img = elastix_img_filter.GetResultImage()
        extract_filter = sitk.ExtractImageFilter()
        
        # extract individual 3D images from 4D result image
        size = list(transformed_img.GetSize())
        size[3] = 0  # set t to 0 to collapse this dimension
        extract_filter.SetSize(size)
        imgs = []
        num_images = len(img_files)
        for i in range(num_images):
            extract_filter.SetIndex([0, 0, 0, i])  # x, y, z, t
            img = extract_filter.Execute(transformed_img)
            img_np = sitk_io.convert_img(img)
            # resize to original shape of first image, all aligned to position
            # of subject within first image
            img_large_np = np.zeros(size_orig[::-1])
            img_large_np[:, start_y:start_y + img_np.shape[1]] = img_np
            if show_imgs and sitk:
                sitk.Show(sitk_io.replace_sitk_with_numpy(img, img_large_np))
            imgs.append(img_large_np)
    
    else:
        # groupwise register images in ITK
        reg_params.AddParameterMap(param_map)
        elastix_img_filter.SetParameterObject(reg_params)
        elastix_img_filter.SetLogToConsole(config.verbose)
        elastix_img_filter.UpdateLargestPossibleRegion()
        transformed_img = elastix_img_filter.GetOutput()
        imgs = sitk_io.convert_img(transformed_img)

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
    imgs_to_show = [img_raw]
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
    
    if show_imgs and sitk:
        for img in imgs_to_show: sitk.Show(img)
    
    #transformed_img = img_raw
    if write_imgs:
        # write both the .mhd and Numpy array files to a separate folder to
        # mimic the atlas folder format
        out_path = os.path.join(name_prefix, config.RegNames.IMG_GROUPED.value)
        if not os.path.exists(name_prefix):
            os.makedirs(name_prefix)
        sitk_io.write_img(transformed_img, out_path)
        img_np = sitk_io.convert_img(transformed_img)
        config.resolutions = [tuple(transformed_img.GetSpacing())[::-1]]
        importer.save_np_image(img_np[None], out_path, config.series)
    
    _logger.info(
        "Time elapsed for groupwise registration (s): %s", time() - start_time)


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
    moving_sitk = sitk_io.read_img(out_path)
    moving_sitk = atlas_refiner.transpose_img(
        moving_sitk, plane, rotate)
    moving_img = sitk_io.convert_img(moving_sitk)
    
    # get the registered atlas file, which should already be transposed
    transformed_sitk = sitk_io.load_registered_img(name_prefix, get_sitk=True)
    transformed_img = sitk_io.convert_img(transformed_sitk)
    
    # get the registered labels file, which should also already be transposed
    labels_img = sitk_io.load_registered_img(
        name_prefix, config.RegNames.IMG_LABELS.value)
    
    # calculate the Dice similarity coefficient
    atlas_refiner.measure_overlap(
        sitk_io.load_numpy_to_sitk(fixed_file), transformed_sitk)
    
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
        bbox_scaled = np.around(bbox_scaled).astype(int)
        centroid_scaled = np.around(centroid_scaled).astype(int)
        print(props[0].bbox, props[0].centroid)
        print(bbox_scaled, centroid_scaled)
        return props, bbox_scaled, centroid_scaled
    return props, None, None


def make_label_ids_set(
        labels_ref_path: str, labels_ref_lookup: Dict[int, Any],
        max_level: int, combine_sides: bool,
        label_ids: Optional[Sequence[int]] = None) -> List[int]:
    """Make a set of label IDs for the given level and sides.
    
    Args:
        labels_ref_path: Path to labels reference from which to load
            original labels if ``max_level`` is None. Can be None if
            ``labels_id`` is given.
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
        label_ids: Sequence of IDs; defaults to None to get using
            ``labels_ref_path``.

    Returns:
        List of label IDs.

    """
    drawn_only = max_level is None
    if label_ids is None or not drawn_only:
        # get label IDs from at atlas
        label_ids = sitk_io.find_atlas_labels(
            labels_ref_path, drawn_only, labels_ref_lookup)
    if not combine_sides and np.all(np.array(label_ids) >= 0):
        # include opposite side as separate labels; otherwise, defer to 
        # ontology (max_level flag) or labels metrics to get labels from 
        # opposite sides by combining them
        label_ids.extend([-1 * n for n in label_ids])
    if max_level is not None:
        # setup ontological labels
        ids_with_children = []
        for label_id in label_ids:
            label = labels_ref_lookup.get(abs(label_id))
            if label is None: continue
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


def volumes_by_id(
        img_paths: Sequence[str], labels_ref_path: Optional[str] = None,
        suffix: Optional[str] = None, unit_factor: Optional[str] = None,
        groups: Optional[Dict] = None, max_level: Optional[int] = None,
        combine_sides=True, extra_metrics: Optional[str] = None):
    """Get volumes and additional label metrics for each single labels ID.
    
    Atlas (intensity) and annotation (labels) images can be configured
    in :attr:`magmap.settings.config.reg_suffixes`.
    :attr:`magmap.settings.config.plot_labels` can be used
    to configure the condition field with the
    :attr:`magmap.settings.config.PlotLabels.CONDITION` key.
    
    Args:
        img_paths: Sequence of images.
        labels_ref_path (str): Labels reference path(s).
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
    
    if len(img_paths) < 1:
        return None, None
    
    # prepare data frame output paths and 
    out_base = SAMPLE_VOLS if max_level is None else SAMPLE_VOLS_LEVELS
    out_path = libmag.make_out_path(
        out_base, suffix=suffix, combine_prefix=True)
    out_path_summary = libmag.make_out_path(
        f"{out_base}_summary", suffix=suffix, combine_prefix=True)
    
    # prep condition column based on suffix and plot labels flag
    condition = "original" if suffix is None else suffix.replace("_", "")
    cond_arg = config.plot_labels[config.PlotLabels.CONDITION]
    if cond_arg:
        # override default condition or from suffix
        condition = cond_arg
    
    # grouping metadata, which will be combined with groups
    grouping = OrderedDict()
    grouping[config.AtlasMetrics.SAMPLE.value] = None
    grouping[config.AtlasMetrics.CONDITION.value] = condition
    
    # set up labels reference if available
    label_ids = None
    labels_ref = ontology.LabelsRef(labels_ref_path).load()
    ref_not_spec = labels_ref.ref_lookup is None
    df_regions = None
    
    # region columns to keep and mapping to convert column names
    region_col_conv = {
        config.ABAKeys.ABA_ID.value: config.AtlasMetrics.REGION.value,
        config.ABAKeys.NAME.value: config.AtlasMetrics.REGION_NAME.value,
        config.ABAKeys.LEVEL.value: config.AtlasMetrics.LEVEL.value,
        config.ABAKeys.ACRONYM.value: config.AtlasMetrics.REGION_ABBR.value,
    }
    
    dfs = []
    dfs_all = []
    for i, img_path in enumerate(img_paths):
        # adjust image path with suffix
        mod_path = img_path
        if suffix is not None:
            mod_path = libmag.insert_before_ext(img_path, suffix)

        labels_metadata = labels_meta.LabelsMeta(mod_path).load()
        labels_ref_exp = labels_ref
        if ref_not_spec:
            # load labels metadata from sample image if global ref not loaded
            labels_ref_exp = ontology.LabelsRef(labels_metadata.path_ref).load()
        label_ids_exp = labels_metadata.region_ids_orig
        if label_ids is None or ref_not_spec:
            # build IDs if not yet built, or always build if no global ref
            label_ids = make_label_ids_set(
                labels_ref_path, labels_ref_exp.ref_lookup, max_level,
                combine_sides, label_ids_exp)
            
            # extract region names into a separate data frame with selected,
            # renamed columns
            df_regions = labels_ref_exp.get_ref_lookup_as_df()
            cols = [c for c in region_col_conv.keys()
                    if c in df_regions.columns]
            df_regions = df_regions[cols]
            df_regions = df_regions.rename(region_col_conv, axis=1)
        
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
            img_np = sitk_io.convert_img(img_sitk)
            spacing = tuple(img_sitk.GetSpacing())[::-1]
            
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
            
            # load heat map of nuclei per voxel if available; use density
            # suffix if provided
            density_suffix = config.reg_suffixes[config.RegSuffixes.DENSITY]
            if not density_suffix:
                density_suffix = config.RegNames.IMG_HEAT_MAP.value
            try:
                heat_map = sitk_io.load_registered_img(mod_path, density_suffix)
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
        if df_regions is not None:
            # merge in region names
            df = df_io.join_dfs(
                (df_regions, df), config.AtlasMetrics.REGION.value, how="right")
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


def volumes_by_id_compare(img_paths, labels_ref_paths, unit_factor=None,
                          groups=None, max_level=None, combine_sides=True,
                          offset=None, roi_size=None):
    """Compare label volumes metrics between different sets of atlases.
    
    Label identities can be translated using CSV files specified in the
    :attr:`config.atlas_labels[config.AtlasLabels.TRANSLATE_LABELS` value
    to compare labels from different atlases or groups of labels that do
    not fit exclusively into a single super-structure. Each file will be
    mapped to the corresponding paths in the ``img_paths`` and
    ``labels_ref_paths`` sequences. The
    :attr:`config.atlas_labels[config.AtlasLabels.TRANSLATE_CHILDREN`
    allows specifying whether to include the children of each label.
    Labels default to use the last value in each of these sequences.

    Args:
        img_paths (Sequence[str]): Paths from which registered labels images
            will be loaded to compare.
        labels_ref_paths (Union[str, Sequence[str]): Labels reference path(s).
            If multiple paths are provided, the corresponding reference
            to the labels translation paths will be used for translation,
            and the last reference will be used to find labels. If fewer
            references than translation paths are provided, the last
            reference will be used for the remaining translation paths.
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
    
    # load labels references and label IDs based on the last reference
    labels_ref_paths = libmag.to_seq(labels_ref_paths)
    labels_ref_lookups = [
        ontology.LabelsRef(p).load().ref_lookup for p in labels_ref_paths]
    label_ids = make_label_ids_set(
        labels_ref_paths[-1], labels_ref_lookups[-1], max_level, combine_sides)
    
    # set up data frame path based on last image and load data frame if available
    df_path = libmag.make_out_path("{}_volcompare.csv".format(
        os.path.splitext(img_paths[-1])[0]))
    df, df_path, df_level_path = _setup_vols_df(df_path, max_level)
    
    spacing = None
    labels_imgs = None
    heat_map = None
    if df is None:
        # open images for primary measurements rather than weighting from
        # data frame
        
        # get labels image based on registered image suffix
        reg_labels = config.reg_suffixes[config.RegSuffixes.ANNOTATION]
        if reg_labels is None:
            reg_labels = config.RegNames.IMG_LABELS.value
        labels_imgs_sitk = []
        for img_path in img_paths:
            try:
                img = sitk_io.load_registered_img(
                    img_path, reg_labels, get_sitk=True)
            except FileNotFoundError:
                libmag.warn("{} not found for {}, will fall back to regular "
                            "annotations".format(reg_labels, img_path))
                img = sitk_io.load_registered_img(
                    img_path, config.RegNames.IMG_LABELS.value, get_sitk=True)
            labels_imgs_sitk.append(img)
        labels_imgs = [sitk_io.convert_img(img) for img in labels_imgs_sitk]
        spacing = tuple(labels_imgs_sitk[0].GetSpacing())[::-1]
        
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
        
        paths_translate = libmag.to_seq(config.atlas_labels[
            config.AtlasLabels.TRANSLATE_LABELS])
        if paths_translate:
            # load data frames corresponding to each labels image to convert
            # label IDs, clearing all other labels
            
            # match references to translation paths, defaulting to last ref
            # for any unpaired translation paths
            len_paths_tr = len(paths_translate)
            refs = list(labels_ref_lookups)
            libmag.pad_seq(refs, len_paths_tr, refs[-1])
            
            # also match to flags for translating children
            translate_chil = libmag.to_seq(config.atlas_labels[
                config.AtlasLabels.TRANSLATE_CHILDREN], non_none=False)
            translate_chil = libmag.pad_seq(
                translate_chil, len_paths_tr, translate_chil[-1])
            for labels_img, path_translate, ref, chil in zip(
                    labels_imgs, paths_translate, refs, translate_chil):
                if os.path.exists(path_translate):
                    # translate labels using the given path, including
                    # children of each label if flagged
                    ontology.replace_labels(
                        labels_img, pd.read_csv(path_translate), clear=True,
                        ref=ref if chil else None, combine_sides=combine_sides)
                elif path_translate:
                    # warn if path does not exist; empty string can skip image
                    libmag.warn("{} does not exist, skipping label translation"
                                .format(paths_translate))
        
        for img_path, labels_img_sitk, labels_img in zip(
                img_paths, labels_imgs_sitk, labels_imgs):
            # re-save labels image to inspect any cropping, translation
            imgs_write = {
                config.RegNames.IMG_LABELS_TRANS.value: (
                    sitk_io.replace_sitk_with_numpy(
                        labels_img_sitk, labels_img)),
            }
            sitk_io.write_reg_images(imgs_write, img_path)
    
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
    ref = ontology.LabelsRef(config.load_labels).load()
    lookup_id = 15565 # short search path
    #lookup_id = 126652058 # last item
    time_dict_start = time()
    labels_img = sitk_io.load_registered_img(
        config.filename, config.RegNames.IMG_LABELS.value)
    max_labels = np.max(labels_img)
    print("max_labels: {}".format(max_labels))
    time_dict_end = time()
    
    # look up a single ID
    time_node_start = time()
    found = ref.ref_lookup[lookup_id]
    time_node_end = time()
    print("found {}: {} with parents {}"
          .format(lookup_id, found[ontology.NODE]["name"],
                  found[ontology.PARENT_IDS]))
    
    # brute-force query
    time_direct_start = time()
    node = ref.get_node(ref["msg"][0], "id", lookup_id, "children")
    time_direct_end = time()
    print(node)
    
    print("time to create id_dict (s): {}".format(time_dict_end - time_dict_start))
    print("time to find node (s): {}".format(time_node_end - time_node_start))
    print("time to find node directly (s): {}".format(time_direct_end - time_direct_start))
    
    # get a list of IDs corresponding to each blob
    blobs = np.array([[300, 5000, 3000], [350, 5500, 4500], [400, 6000, 5000]])
    img5d = importer.read_file(config.filename, config.series)
    scaling = importer.calc_scaling(img5d.img, labels_img)
    coord_scaled = ontology.scale_coords(blobs[:, 0:3], scaling, labels_img.shape)
    label_ids = ontology.get_label_ids_from_position(coord_scaled, labels_img)
    print("blob IDs:\n{}".format(label_ids))
    print("coord_scaled:\n{}".format(coord_scaled))


def _test_region_from_id():
    """Test finding a region by ID in a labels image.
    """
    if len(config.filenames) > 1:
        # unregistered, original labels image; assume that atlas directory has 
        # been given similarly to register fn
        path = os.path.join(
            config.filenames[1], config.RegNames.IMG_LABELS.value)
        labels_img = sitk_io.read_img(path)
        labels_img = sitk_io.convert_img(labels_img)
        scaling = np.ones(3)
        print("loaded labels image from {}".format(path))
    else:
        # registered labels image and associated experiment file
        labels_img = sitk_io.load_registered_img(
            config.filename, config.RegNames.IMG_LABELS.value)
        if config.filename.endswith(".mhd"):
            img = sitk_io.read_img(config.filename)
            img = sitk_io.convert_img(img)
            image5d = img[None]
        else:
            img5d = importer.read_file(config.filename, config.series)
        scaling = importer.calc_scaling(img5d.img, labels_img)
        print("loaded experiment image from {}".format(config.filename))
    ref = ontology.LabelsRef(config.load_labels).load()
    middle, img_region, region_ids = ontology.get_region_middle(
        ref.ref_lookup, 16652, labels_img, scaling)
    atlas_label = ontology.get_label(
        middle, labels_img, ref.ref_lookup, scaling, None, True)
    props, bbox, centroid = get_scaled_regionprops(img_region, scaling)
    print("bbox: {}, centroid: {}".format(bbox, centroid))


def _test_curate_img(path, prefix):
    fixed_img = sitk_io.load_numpy_to_sitk(path)
    labels_img = sitk_io.load_registered_img(
        prefix, config.RegNames.IMG_LABELS.value, get_sitk=True)
    atlas_img = sitk_io.load_registered_img(
        prefix, config.RegNames.IMG_ATLAS.value, get_sitk=True)
    labels_img.SetSpacing(fixed_img.GetSpacing())
    holes_area = config.atlas_profile["holes_area"]
    result_imgs = curate_img(
        fixed_img, labels_img, [atlas_img], inpaint=False, 
        holes_area=holes_area)
    if sitk:
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
    """Handle registration processing tasks
    
    Tasks are specified in :attr:`magmap.config.register_type`.
    
    """
    if sitk and hasattr(sitk.ProcessObject, "SetGlobalDefaultThreader"):
        # manually set threader for SimpleITK >= 2 to avoid potential hangs
        # during Python multiprocessing; set by default in SimpleITK 2.1
        sitk.ProcessObject.SetGlobalDefaultThreader("Platform")
    
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
        # "single" (rather than groupwise) registration, or "new_atlas" to
        # register and output files as a new atlas
        new_atlas = reg is config.RegisterTypes.NEW_ATLAS
        transformix = register(
            *config.filenames[:2], name_prefix=config.prefix,
            new_atlas=new_atlas, show_imgs=config.show)
        
        if len(config.filenames) > 2:
            # apply the same transformation to additional atlas(es)
            prefix = config.prefix if config.prefix else config.filenames[0]
            for i, path in enumerate(config.filenames[2:]):
                # use additional config.prefix for output names if available
                name_prefix = libmag.get_if_within(config.prefixes, i + 1)
                if not name_prefix:
                    # fall back to combine first prefix or image path and atlas
                    name_prefix = (
                        f"{prefix}_"
                        f"{libmag.splitext(os.path.basename(path))[0]}")
                register(
                    config.filenames[0], path, name_prefix=name_prefix,
                    new_atlas=new_atlas, show_imgs=config.show,
                    transformix=transformix)
    
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

        ref = ontology.LabelsRef(config.load_labels).load()
        
        # export region IDs and parents at given level to CSV, using only
        # the atlas' labels if orig colors is true
        path = "region_ids"
        if config.filename:
            path = "{}_{}".format(path, config.filename)
        export_regions.export_region_ids(
            ref.ref_lookup, path, config.labels_level,
            config.atlas_labels[config.AtlasLabels.ORIG_COLORS])
        
        # export region IDs to network file
        export_regions.export_region_network(
            ref.ref_lookup, "region_network")
    
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
                dfs[key], libmag.make_out_path(
                    config.prefix, "", "smoothing_filt{}.csv".format(key)))
        
        # round peak filter sizes after extraction since sizes are now strings
        df_peaks = df_io.data_frames_to_csv(dfs_noloss)
        df_peaks[key_filt] = df_peaks[key_filt].map(
            libmag.truncate_decimal_digit)
        df_io.data_frames_to_csv(
            df_peaks, libmag.make_out_path(
                config.prefix, "", "smoothing_peaks.csv"))

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
                config.filenames, config.load_labels, suffix=config.suffix,
                unit_factor=config.unit_factor, groups=groups, 
                max_level=config.labels_level, combine_sides=combine_sides, 
                extra_metrics=extra_metric_groups)
        elif reg is config.RegisterTypes.VOL_COMPARE:
            # compare the given samples
            volumes_by_id_compare(
                config.filenames, config.load_labels,
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
        reg_tasks.build_labels_diff_images(config.filenames[1:])
    
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
        atlas_stats.meas_landmark_dist(config.filenames, scaling)

    else:
        print("Could not find register task:", reg)


if __name__ == "__main__":
    print("MagellanMapper image registration")
    cli.main(True)
    main()
