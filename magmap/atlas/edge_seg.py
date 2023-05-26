# Segmentation based on edge detection
# Author: David Young, 2019, 2023
"""Re-segment atlases based on edge detections.
"""
import os
from time import time
from typing import List, Optional

import numpy as np
import pandas as pd
try:
    import SimpleITK as sitk
except ImportError:
    sitk = None
from skimage import color

from magmap.atlas import atlas_refiner
from magmap.cv import chunking, cv_nd, segmenter
from magmap.settings import atlas_prof, config, profiles
from magmap.io import df_io, libmag, sitk_io
from magmap.stats import vols

_logger = config.logger.getChild(__name__)


def _mirror_imported_labels(labels_img_np, start, mirror_mult, axis):
    # mirror labels that have been imported and transformed may have had
    # axes swapped, requiring them to be swapped back
    labels_img_np = atlas_refiner.mirror_planes(
        np.swapaxes(labels_img_np, 0, axis), start, mirror_mult=mirror_mult,
        check_equality=True)
    labels_img_np = np.swapaxes(labels_img_np, 0, axis)
    return labels_img_np


def _is_profile_mirrored():
    # check if profile is set for mirroring, though does not necessarily
    # mean that the image itself is mirrored; allows checking for 
    # simplification by operating on one half and mirroring to the other
    mirror = config.atlas_profile["labels_mirror"]
    return (mirror and mirror[profiles.RegKeys.ACTIVE]
            and mirror["start"] is not None)


def _get_mirror_mult():
    # get the mirrored labels multiplier, which is -1 if set to neg labels
    # and 1 if otherwise
    mirror = config.atlas_profile["labels_mirror"]
    mirror_mult = -1 if mirror and mirror["neg_labels"] else 1
    return mirror_mult


def make_edge_images(path_img, show=True, atlas=True, suffix=None, 
                     path_atlas_dir=None):
    """Make edge-detected atlas and associated labels images.
    
    The atlas is assumed to be a sample (eg microscopy) image on which 
    an edge-detection filter will be applied. The labels image is 
    assumed to be an annotated image whose edges will be found by 
    obtaining the borders of all separate labels. Atlas and labels images
    can be set in :attr:`config.reg_suffixes`. If the labels image suffix is
    an empty string (:attr:`config.reg_suffixes[annotation=""]`) and
    ``path_atlas_dir`` is not used, no labels image will be used.
    
    Args:
        path_img: Path to the image atlas. The labels image will be 
            found as a corresponding, registered image, unless 
            ``path_atlas_dir`` is given.
        show (bool): True if the output images should be displayed; defaults
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
            different labels. Defaults to None, in which case the labels
            image is loaded from the registered labels image to ``path_img``.
    
    """
    
    # load intensity image from which to detect edges
    atlas_suffix = config.reg_suffixes[config.RegSuffixes.ATLAS]
    if not atlas_suffix:
        if atlas:
            # atlases default to using the atlas volume image
            print("generating edge images for atlas")
            atlas_suffix = config.RegNames.IMG_ATLAS.value
        else:
            # otherwise, use the experimental image
            print("generating edge images for experiment/sample image")
            atlas_suffix = config.RegNames.IMG_EXP.value
    
    # adjust image path with suffix
    mod_path = path_img
    if suffix is not None:
        mod_path = libmag.insert_before_ext(mod_path, suffix)
    
    # load labels image
    labels_from_atlas_dir = path_atlas_dir and os.path.isdir(path_atlas_dir)
    labels_suffix = config.reg_suffixes[config.RegSuffixes.ANNOTATION]
    if labels_suffix is None:
        labels_suffix = config.RegNames.IMG_LABELS.value
    labels_sitk = None
    path_atlas = mod_path
    if labels_from_atlas_dir:
        # load labels from atlas directory
        path_atlas = path_img
        path_labels = os.path.join(path_atlas_dir, labels_suffix)
        print("loading labels from", path_labels)
        labels_sitk = sitk_io.read_img(path_labels)
    elif labels_suffix:
        # load labels registered to sample image
        labels_sitk = sitk_io.load_registered_img(
            mod_path, labels_suffix, get_sitk=True)
    labels_img_np = None if labels_sitk is None else sitk_io.convert_img(
        labels_sitk)
    # load atlas image, set resolution from it
    atlas_sitk = sitk_io.load_registered_img(
        path_atlas, atlas_suffix, get_sitk=True)
    config.resolutions = np.array([tuple(atlas_sitk.GetSpacing())[::-1]])
    atlas_np = sitk_io.convert_img(atlas_sitk)
    
    if config.rgb:
        # convert RGB atlas image to grayscale for single channel
        atlas_np = color.rgb2gray(atlas_np)
    
    # output images
    atlas_sitk_log = None
    atlas_sitk_edge = None
    labels_sitk_interior = None
    
    log_sigma = config.atlas_profile["log_sigma"]
    if log_sigma is not None and suffix is None:
        # generate LoG and edge-detected images for original image
        print("generating LoG edge-detected images with sigma", log_sigma)
        thresh = (config.atlas_profile["atlas_threshold"]
                  if config.atlas_profile["log_atlas_thresh"] else None)
        atlas_log = cv_nd.laplacian_of_gaussian_img(
            atlas_np, sigma=log_sigma, labels_img=labels_img_np, thresh=thresh)
        atlas_sitk_log = sitk_io.replace_sitk_with_numpy(atlas_sitk, atlas_log)
        atlas_edge = cv_nd.zero_crossing(atlas_log, 1).astype(np.uint8)
        atlas_sitk_edge = sitk_io.replace_sitk_with_numpy(
            atlas_sitk, atlas_edge)
    else:
        # if sigma not set or if using suffix to compare two images, 
        # load from original image to compare against common image
        atlas_edge = sitk_io.load_registered_img(
            path_img, config.RegNames.IMG_ATLAS_EDGE.value)

    erode = config.atlas_profile["erode_labels"]
    if erode["interior"]:
        # make map of label interiors for interior/border comparisons
        print("Eroding labels to generate interior labels image")
        erosion = config.atlas_profile[
            profiles.RegKeys.EDGE_AWARE_REANNOTATION]
        erosion_frac = config.atlas_profile["erosion_frac"]
        interior, _ = erode_labels(
            labels_img_np, erosion, erosion_frac, 
            atlas and _is_profile_mirrored(), _get_mirror_mult())
        labels_sitk_interior = sitk_io.replace_sitk_with_numpy(
            labels_sitk, interior)
    
    # make labels edge and edge distance images
    dist_sitk = None
    labels_sitk_edge = None
    if config.atlas_profile["meas_edge_dists"]:
        dist_to_orig, labels_edge = edge_distances(
            labels_img_np, atlas_edge, spacing=tuple(atlas_sitk.GetSpacing())[::-1])
        dist_sitk = sitk_io.replace_sitk_with_numpy(atlas_sitk, dist_to_orig)
        labels_sitk_edge = sitk_io.replace_sitk_with_numpy(
            labels_sitk, labels_edge)
    
    # show all images
    imgs_write = {
        config.RegNames.IMG_ATLAS_LOG.value: atlas_sitk_log, 
        config.RegNames.IMG_ATLAS_EDGE.value: atlas_sitk_edge, 
        config.RegNames.IMG_LABELS_EDGE.value: labels_sitk_edge, 
        config.RegNames.IMG_LABELS_INTERIOR.value: labels_sitk_interior, 
        config.RegNames.IMG_LABELS_DIST.value: dist_sitk, 
    }
    if show and sitk:
        for img in imgs_write.values():
            if img: sitk.Show(img)
    
    # write images to same directory as atlas with appropriate suffix
    sitk_io.write_reg_images(imgs_write, mod_path)


def erode_labels(labels_img_np, erosion, erosion_frac=None, mirrored=True,
                 mirror_mult=-1):
    """Erode labels image for use as markers or a map of the interior.
    
    Args:
        labels_img_np (:obj:`np.ndarray`): Numpy image array of labels in
            z,y,x format.
        erosion (dict): Dictionary of erosion filter settings from
            :class:`profiles.RegKeys` to pass to
            :meth:`segmenter.labels_to_markers_erosion`.
        erosion_frac (int): Target erosion fraction; defaults to None.
        mirrored (bool): True if the primary image mirrored/symmatrical, in
            which case erosion will only be performed one symmetric half
            and mirrored to the other half. If False or no symmetry is
            found, such as unmirrored atlases or experimental/sample
            images, erosion will be performed on the full image.
        mirror_mult (int): Multiplier for mirrored labels; defaults to -1
            to make mirrored labels the inverse of their source labels.
    
    Returns:
        :obj:`np.ndarray`, :obj:`pd.DataFrame`: The eroded labels as a new
        array of same shape as that of ``labels_img_np`` and a data frame
        of erosion stats.
    """
    labels_to_erode = labels_img_np
    sym_axis = atlas_refiner.find_symmetric_axis(labels_img_np, mirror_mult)
    is_mirrored = mirrored and sym_axis >= 0
    len_half = None
    if is_mirrored:
        # if symmetric, erode only one symmetric half
        len_half = labels_img_np.shape[sym_axis] // 2
        slices = [slice(None)] * labels_img_np.ndim
        slices[sym_axis] = slice(len_half)
        labels_to_erode = labels_img_np[tuple(slices)]
    
    # convert labels image into markers
    #eroded = segmenter.labels_to_markers_blob(labels_img_np)
    eroded, df = segmenter.labels_to_markers_erosion(
        labels_to_erode, erosion[profiles.RegKeys.MARKER_EROSION],
        erosion_frac, erosion[profiles.RegKeys.MARKER_EROSION_MIN],
        skel_eros_filt_size=erosion[profiles.RegKeys.SKELETON_EROSION])
    if is_mirrored:
        # mirror changes onto opposite symmetric half
        eroded = _mirror_imported_labels(
            eroded, len_half, mirror_mult, sym_axis)
    
    return eroded, df


def edge_aware_segmentation(
        path_atlas: str, atlas_profile: atlas_prof.AtlasProfile,
        show: bool = True, atlas: bool = True, suffix: Optional[str] = None,
        exclude_labels: Optional[pd.DataFrame] = None, mirror_mult: int = -1):
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
        atlas_profile: Atlas profile.
        show: True if the output images should be displayed; defaults 
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
        exclude_labels: Sequence of labels to exclude from the
            segmentation; defaults to None.
        mirror_mult: Multiplier for mirrored labels; defaults to -1
            to make mirrored labels the inverse of their source labels.
    """
    # adjust image path with suffix
    load_path = path_atlas
    mod_path = path_atlas
    if suffix is not None:
        mod_path = libmag.insert_before_ext(mod_path, suffix)
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
    atlas_img_np = sitk_io.convert_img(atlas_sitk)
    atlas_edge = sitk_io.convert_img(atlas_sitk_edge)
    labels_img_np = sitk_io.convert_img(labels_sitk)
    markers = sitk_io.convert_img(labels_sitk_markers)
    
    # segment image from markers
    sym_axis = atlas_refiner.find_symmetric_axis(atlas_img_np)
    mirrorred = atlas and sym_axis >= 0
    len_half = None
    seg_args = {"exclude_labels": exclude_labels}
    edge_prof = atlas_profile[profiles.RegKeys.EDGE_AWARE_REANNOTATION]
    if edge_prof:
        edge_filt = edge_prof[profiles.RegKeys.WATERSHED_MASK_FILTER]
        if edge_filt and len(edge_filt) > 1:
            # watershed mask filter settings from atlas profile
            seg_args["mask_filt"] = edge_filt[0]
            seg_args["mask_filt_size"] = edge_filt[1]
    if mirrorred:
        # segment only half of image, assuming symmetry
        len_half = atlas_img_np.shape[sym_axis] // 2
        slices = [slice(None)] * labels_img_np.ndim
        slices[sym_axis] = slice(len_half)
        sl = tuple(slices)
        labels_seg = segmenter.segment_from_labels(
            atlas_edge[sl], markers[sl], labels_img_np[sl], **seg_args)
    else:
        # segment the full image, including excluded labels on the opposite side
        exclude_labels = exclude_labels.tolist().extend(
            (mirror_mult * exclude_labels).tolist())
        seg_args["exclude_labels"] = exclude_labels
        labels_seg = segmenter.segment_from_labels(
            atlas_edge, markers, labels_img_np, **seg_args)
    
    smoothing = atlas_profile["smooth"]
    smoothing_mode = atlas_profile["smoothing_mode"]
    cond = ["edge-aware_seg"]
    if smoothing is not None:
        # smoothing by opening operation based on profile setting
        meas_smoothing = atlas_profile["meas_smoothing"]
        cond.append("smoothing")
        df_aggr, df_raw = atlas_refiner.smooth_labels(
            labels_seg, smoothing, smoothing_mode,
            meas_smoothing, tuple(labels_sitk.GetSpacing())[::-1])
        df_base_path = os.path.splitext(mod_path)[0]
        if df_raw is not None:
            # write raw smoothing metrics
            df_io.data_frames_to_csv(
                df_raw, f"{df_base_path}_{config.PATH_SMOOTHING_RAW_METRICS}")
        if df_aggr is not None:
            # write aggregated smoothing metrics
            df_io.data_frames_to_csv(
                df_aggr, f"{df_base_path}_{config.PATH_SMOOTHING_METRICS}")
    
    if mirrorred:
        # mirror back to other half
        labels_seg = _mirror_imported_labels(
            labels_seg, len_half, mirror_mult, sym_axis)
    
    # expand background to smoothed background of original labels to 
    # roughly match background while still allowing holes to be filled
    crop = atlas_profile["crop_to_orig"]
    atlas_refiner.crop_to_orig(
        labels_img_np, labels_seg, crop)
    
    if labels_seg.dtype != labels_img_np.dtype:
        # watershed may give different output type, so cast back if so
        labels_seg = labels_seg.astype(labels_img_np.dtype)
    labels_sitk_seg = sitk_io.replace_sitk_with_numpy(labels_sitk, labels_seg)
    
    # show DSCs for labels
    _logger.info(
        "\nMeasuring overlap of individual original and watershed labels:")
    dsc_lbls_comb = atlas_refiner.measure_overlap_labels(
        labels_sitk, labels_sitk_seg)
    _logger.info(
        "\nMeasuring overlap of combined original and watershed labels:")
    dsc_lbls_indiv = atlas_refiner.measure_overlap_labels(
        atlas_refiner.make_labels_fg(labels_sitk), 
        atlas_refiner.make_labels_fg(labels_sitk_seg))
    _logger.info("")
    
    # measure and save whole atlas metrics
    metrics = {
        config.AtlasMetrics.SAMPLE: [os.path.basename(mod_path)],
        config.AtlasMetrics.REGION: config.REGION_ALL,
        config.AtlasMetrics.CONDITION: "|".join(cond),
        config.AtlasMetrics.DSC_LABELS_ORIG_NEW_COMBINED: dsc_lbls_comb,
        config.AtlasMetrics.DSC_LABELS_ORIG_NEW_INDIV: dsc_lbls_indiv,
    }
    df_metrics_path = libmag.combine_paths(
        mod_path, config.PATH_ATLAS_IMPORT_METRICS)
    atlas_refiner.measure_atlas_refinement(
        metrics, atlas_sitk, labels_sitk_seg, atlas_profile, df_metrics_path)

    # show and write image to same directory as atlas with appropriate suffix
    sitk_io.write_reg_images(
        {config.RegNames.IMG_LABELS.value: labels_sitk_seg}, mod_path)
    if show and sitk: sitk.Show(labels_sitk_seg)
    return path_atlas


def merge_atlas_segmentations(img_paths, show=True, atlas=True, suffix=None):
    """Merge atlas segmentations for a list of files as a multiprocessing 
    wrapper for :func:``merge_atlas_segmentations``, after which 
    edge image post-processing is performed separately since it 
    contains tasks also performed in multiprocessing.
    
    Args:
        img_paths (List[str]): Sequence of image paths to load.
        show (bool): True if the output images should be displayed; defaults 
            to True.
        atlas (bool): True if the image is an atlas; defaults to True.
        suffix (str): Modifier to append to end of ``img_path`` basename for 
            registered image files that were output to a modified name; 
            defaults to None.
    """
    start_time = time()
    
    # erode all labels images into markers for watershed; not multiprocessed
    # since erosion is itself multiprocessed
    erode = config.atlas_profile["erode_labels"]
    erosion = config.atlas_profile[profiles.RegKeys.EDGE_AWARE_REANNOTATION]
    erosion_frac = config.atlas_profile["erosion_frac"]
    mirrored = atlas and _is_profile_mirrored()
    mirror_mult = _get_mirror_mult()
    dfs_eros = []
    for img_path in img_paths:
        mod_path = img_path
        if suffix is not None:
            mod_path = libmag.insert_before_ext(mod_path, suffix)
        labels_sitk = sitk_io.load_registered_img(
            mod_path, config.RegNames.IMG_LABELS.value, get_sitk=True)
        print("Eroding labels to generate markers for atlas segmentation")
        df = None
        if erode["markers"]:
            # use default minimal post-erosion size (not setting erosion frac)
            markers, df = erode_labels(
                sitk_io.convert_img(labels_sitk), erosion,
                mirrored=mirrored, mirror_mult=mirror_mult)
            labels_sitk_markers = sitk_io.replace_sitk_with_numpy(
                labels_sitk, markers)
            sitk_io.write_reg_images(
                {config.RegNames.IMG_LABELS_MARKERS.value: labels_sitk_markers},
                mod_path)
            df_io.data_frames_to_csv(
                df, "{}_markers.csv".format(os.path.splitext(mod_path)[0]))
        dfs_eros.append(df)
    
    pool = chunking.get_mp_pool()
    pool_results = []
    for img_path, df in zip(img_paths, dfs_eros):
        print("setting up atlas segmentation merge for", img_path)
        # convert labels image into markers
        exclude = df.loc[
            np.isnan(df[config.SmoothingMetrics.FILTER_SIZE.value]),
            config.AtlasMetrics.REGION.value]
        print("excluding these labels from re-segmentation:\n", exclude)
        pool_results.append(pool.apply_async(
            edge_aware_segmentation,
            args=(img_path, config.atlas_profile, show, atlas, suffix, exclude,
                  mirror_mult)))
    for result in pool_results:
        # edge distance calculation and labels interior image generation 
        # are multiprocessed, so run them as post-processing tasks to 
        # avoid nested multiprocessing
        path = result.get()
        mod_path = path
        if suffix is not None:
            mod_path = libmag.insert_before_ext(path, suffix)
        
        dist_sitk = None
        labels_sitk_edge = None
        labels_sitk_interior = None
        meas_edge_dist = config.atlas_profile["meas_edge_dists"]
        erode_interior = erode["interior"]
        if meas_edge_dist or erode_interior:
            labels_sitk = sitk_io.load_registered_img(
                mod_path, config.RegNames.IMG_LABELS.value, get_sitk=True)
            labels_np = sitk_io.convert_img(labels_sitk)
            if meas_edge_dist:
                # make edge distance images and stats
                dist_to_orig, labels_edge = edge_distances(
                    labels_np, path=path,
                    spacing=tuple(labels_sitk.GetSpacing())[::-1])
                dist_sitk = sitk_io.replace_sitk_with_numpy(
                    labels_sitk, dist_to_orig)
                labels_sitk_edge = sitk_io.replace_sitk_with_numpy(
                    labels_sitk, labels_edge)
    
            if erode_interior:
                # make interior images from labels using given targeted 
                # post-erosion frac
                interior, _ = erode_labels(
                    labels_np, erosion, erosion_frac=erosion_frac, 
                    mirrored=mirrored, mirror_mult=mirror_mult)
                labels_sitk_interior = sitk_io.replace_sitk_with_numpy(
                    labels_sitk, interior)
        
        # write images to same directory as atlas
        imgs_write = {
            config.RegNames.IMG_LABELS_DIST.value: dist_sitk, 
            config.RegNames.IMG_LABELS_EDGE.value: labels_sitk_edge, 
            config.RegNames.IMG_LABELS_INTERIOR.value: labels_sitk_interior, 
        }
        sitk_io.write_reg_images(imgs_write, mod_path)
        if show and sitk:
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
    labels_edge = vols.LabelToEdge.make_labels_edge(labels)
    dist_to_orig, _, _ = cv_nd.borders_distance(
        atlas_edge != 0, labels_edge != 0, spacing=spacing)
    
    return dist_to_orig, labels_edge


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
        mod_path = libmag.insert_before_ext(mod_path, suffix)
    
    # load labels
    labels_sitk = sitk_io.load_registered_img(
        mod_path, config.RegNames.IMG_LABELS.value, get_sitk=True)
    
    # atlas edge image is associated with original, not modified image
    atlas_edge = sitk_io.load_registered_img(
        img_path, config.RegNames.IMG_ATLAS_EDGE.value)
    
    # sub-divide the labels and save to file
    labels_img_np = sitk_io.convert_img(labels_sitk)
    labels_subseg = segmenter.sub_segment_labels(labels_img_np, atlas_edge)
    labels_subseg_sitk = sitk_io.replace_sitk_with_numpy(
        labels_sitk, labels_subseg)
    sitk_io.write_reg_images(
        {config.RegNames.IMG_LABELS_SUBSEG.value: labels_subseg_sitk}, mod_path)
    return labels_subseg
