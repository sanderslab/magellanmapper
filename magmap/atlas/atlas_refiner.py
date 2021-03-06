# Atlas refinement
# Author: David Young, 2019
"""Refine atlases in 3D.
"""
import os
from collections import OrderedDict

import SimpleITK as sitk
import numpy as np
import pandas as pd
from skimage import filters
from skimage import measure
from skimage import morphology
from skimage import transform

from magmap.cv import chunking, cv_nd, segmenter
from magmap.io import df_io
from magmap.io import export_stack
from magmap.io import importer
from magmap.io import libmag
from magmap.io import np_io
from magmap.io import sitk_io
from magmap.plot import plot_3d, plot_support
from magmap.settings import config
from magmap.settings import profiles


def _get_bbox(img_np, threshold=10):
    """Get the bounding box for the largest object within an image.
    
    Args:
        img_np: Image as a Numpy array.
        threshold: Threshold level; defaults to 10. If None, assume 
            ``img_np`` is already binary.
    
    Returns:
        Bounding box of the largest object in the image.
    """
    props_sizes = cv_nd.get_thresholded_regionprops(
        img_np, threshold=threshold, sort_reverse=True)
    if props_sizes is None: return None
    labels_bbox = props_sizes[0][0].bbox
    #print("bbox: {}".format(labels_bbox))
    return labels_bbox


def truncate_labels(img_np, x_frac=None, y_frac=None, z_frac=None):
    """Truncate image by zero-ing out pixels outside of given bounds.
    
    Args:
        img_np: 3D image array in Numpy format.
        x_frac: 2D tuple of (x_start, x_end), given as fractions of the image. 
            Defaults to None, in which case the full image in that direction 
            will be preserved.
        y_frac: Same as ``x_frac`` but for y-axis.
        z_frac: Same as ``x_frac`` but for z-axis.
    
    Returns:
        The truncated image.
    """
    shape = img_np.shape
    bounds = (z_frac, y_frac, x_frac)
    axis = 0
    bounds_len = len(bounds)
    for bound in bounds:
        if bound is not None:
            # convert boundary fractions to absolute coordinates for the 
            # given axis, filling the other axes with full slices
            bound_abs = np.multiply(bound, shape[axis]).astype(np.int)
            slices = [slice(None)] * bounds_len
            slices[axis] = slice(0, bound_abs[0])
            img_np[tuple(slices)] = 0
            slices[axis] = slice(bound_abs[1], None)
            img_np[tuple(slices)] = 0
            print("truncated axis {} outside of bounds {}"
                  .format(axis, bound_abs))
        axis += 1
    return img_np


def mirror_planes(img_np, start, mirror_mult=1, resize=True, start_dup=None,
                  rand_dup=None, check_equality=False):
    """Mirror image across its sagittal midline.
    
    Args:
        img: Labels image in Numpy format, which will be edited directly 
            unless ``resize`` is True.
        start: Starting index at which to begin mirroring, inclusive.
        mirror_mult: Multiplier for mirrored portion of image, such as -1 
            when creating a labels map with one side distinct from the other; 
            defaults to 1.
        resize: True if the image should be resized to be symmetric in size 
            across ``start``; defaults to True.
        start_dup: Fraction at which to start duplicating planes before 
            mirroring, which may be useful for atlases where entired labeled 
            regions are past midline, where duplication of a few atlas 
            planes may simply expand rather than duplicating structures; 
            defaults to None.
        rand_dup: Multiplier for randomizer to choose duplicate planes from 
            a random selection within this given range prior to ``start_dup``. 
            Defaults to None, in which case the plane just prior to 
            the duplication starting plane will simply be duplicated 
            throughout the region.
        check_equality: True to check equality from one half to the other 
            along axis 0; defaults to False.
    
    Returns:
        The mirrored image in Numpy format.
    """
    if resize:
        # made mirror index midline by adjusting image shape
        shape = img_np.shape
        shape_resized = np.copy(shape)
        shape_resized[0] = start * 2
        print("original shape: {}, new shape: {}".format(shape, shape_resized))
        if shape_resized[0] > shape[0]:
            # shift image into larger array, adding extra z padding
            img_resized = np.zeros(shape_resized, dtype=img_np.dtype)
            img_resized[:shape[0]] = img_np
            img_np = img_resized
        else:
            # truncate original image to fit into smaller shape
            img_np = img_np[:shape_resized[0]]
    tot_planes = len(img_np)
    if start_dup is not None:
        # duplicate planes starting at this fraction of total planes, 
        # typically corresponding to the true midline
        n = int(start_dup * tot_planes)
        num_planes = start - n
        if rand_dup is not None:
            # randomly pick planes from 1 to rand_dup behind the starting plane
            np.random.seed(num_planes)
            dup = n - np.ceil(np.random.rand(num_planes) * rand_dup)
            dup = dup.astype(int)
            dup[dup < 0] = 0
        else:
            # duplicate the prior plane through the entire region
            dup = np.repeat(n - 1, num_planes)
        for i in range(num_planes):
            plane_i = n + i
            if plane_i > 0 and plane_i < tot_planes:
                print("duplicating plane {} from {}".format(plane_i, dup[i]))
                img_np[plane_i] = img_np[dup[i]]
    if start <= tot_planes and start >= 0:
        # if empty planes at end, fill the empty space with the preceding 
        # planes in mirrored fashion
        remaining_planes = tot_planes - start
        end = start - remaining_planes - 1
        if end < 0:
            end = None # to slice to beginning with neg increment
            remaining_planes = start
        print("start: {}, end: {}, remaining_planes: {}, tot_planes: {}"
              .format(start, end, remaining_planes, tot_planes))
        img_np[start:start+remaining_planes] = np.multiply(
            img_np[start-1:end:-1], mirror_mult)
    else:
        # skip mirroring if no planes are empty or only first plane is empty
        print("nothing to mirror")
    if check_equality:
        print("Checking labels symmetry after mirroring:")
        check_mirrorred(img_np, mirror_mult=mirror_mult)
    return img_np


def check_mirrorred(img_np, mirror_mult=1, axis=0):
    """Check whether a given image, typically a labels image, is symmetric.
    
    Args:
        img_np (:class:`numpy.ndarray`): Numpy array of image, typically a
            labels image, where symmetry will be detected based on equality
            of two halves split by the image's last dimension.
        mirror_mult (int): Number by which to divide the 2nd half before
            checking for symmetry; defaults to 1. Typically a number 
            used to generate the 2nd half when mirroring.
        axis (int): Axis along which to check for symmetry; defaults to 0.
    
    Returns:
        bool, bool: Values equality, True if the sides are equal, and
        unique labels equality, True if the sides have equal sets of
        unique values.
    """
    half_len = img_np.shape[axis] // 2
    slices = [slice(None)] * img_np.ndim
    slices[axis] = slice(0, half_len)
    half_before = img_np[tuple(slices)]
    slices[axis] = slice(img_np.shape[axis], half_len - 1, -1)
    half_after = img_np[tuple(slices)] / mirror_mult
    equality_vals = np.array_equal(half_before, half_after)
    print("halves symmetric along axis {}?".format(axis), equality_vals)
    equality_lbls = np.array_equal(
        np.unique(half_before), np.unique(half_after))
    print("same labels in each half?", equality_lbls)
    return equality_vals, equality_lbls


def find_symmetric_axis(img_np, mirror_mult=1):
    """Find the axis along which the image is symmetric.

    Args:
        img_np (:obj:`np.ndarray`): Array to check for symmetry.
        mirror_mult (int): Number by which to divide the 2nd half before
            checking for symmetry; defaults to 1. Typically a number
            used to generate the 2nd half when mirroring.

    Returns:
        int: The axis that is symmetric, or -1 if no symmetry is found.

    """
    for i in range(img_np.ndim):
        eq_val, _ = check_mirrorred(img_np, mirror_mult, i)
        if eq_val:
            return i
    return -1


def _curate_labels(img, img_ref, mirror=None, edge=None, expand=None, 
                   rotate=None, smooth=None, affine=None, resize=True):
    """Curate labels through extension, mirroring, and smoothing.
    
    Extension fills in missing edge labels by extrapolating edge planes 
    to the volume of the underlying reference image. Mirroring copies labels 
    from one half of the image to the other for symmetry. Smoothing 
    removes jagged edges from slightly misaligned labels.
    
    Assume that the image is in sagittal sections, typically consisting of 
    only one hemisphere, empty from the far z planes toward the middle but not 
    necessarily the exact middle of the image. The egde to extend are low z 
    planes. Mirroring takes place across a z-plane, ideally at the true 
    midline.

    ``mirror`` and ``edge`` are applied regardless of their
    :obj:`profiles.RegKeys.ACTIVE` status.
    
    Args:
        img: Labels image in SimpleITK format.
        img_ref: Reference atlas image in SimpleITK format.
        mirror (Dict[str, float]): Label mirroring parameters. The value from
            ``start`` specifies the fraction of planes at which to start
            mirroring, where None skips mirroring, and -1 will cause the
            mirror plane to be found automatically based on the first
            plane completely without labels, starting from the highest
            plane and working downward. Defaults to None to ignore mirroring.
        edge (Dict[str, float]): Lateral edge extension parameters passed 
            to :meth:`plot_3d.extend_edge`. The value from ``start`` 
            specifies the fraction of z-planes from which to start 
            extending the edge to lower z-planes, and -1 will cause the 
            edge plane to be found automatically based on the first 
            plane with any labels, starting from the lowest plane. 
            Defaults to None, in which case edge extension will be skipped.
        expand: Tuple of 
            ((x_pixels_start, end), (y, ...), ...), (next_region, ...)) 
            specifying slice boundaries for regions to expand the labels to 
            the size of the atlas. Defaults to None.
        rotate: Tuple of ((angle0, axis0), ...) by which to rotate the 
            labels. Defaults to None to not rotate.
        smooth: Filter size for smoothing, or sequence of sizes to test 
            various smoothing strengths via multiprocessing. Defaults to 
            None to not smooth..
        affine: Dictionary for selective affine transformation, passed 
            to :func:`plot_3d.affine_nd`. Defaults to None to not 
            affine transform.
        resize: True to resize the image during mirroring; defaults to True.
    
    Returns:
         Tuple: the mirrored image in Numpy format; a tuple of the 
         indices from which the edge was extended (labels used as template 
         from this plane index) and the image was mirrored (index where 
         mirrored hemisphere starts); a data frame of smoothing stats, 
         or None if smoothing was not performed; and a data frame of raw 
         smoothing stats, or None if smoothing was not performed.
    """
    edge_start = edge["start"] if edge else None
    mirror_start = None
    mirror_mult = 1
    if mirror:
        mirror_start = mirror["start"]
        if mirror["neg_labels"]:
            mirror_mult = -1
    rotation = rotate["rotation"] if rotate else None
    
    # cast to int that takes the full range of the labels image
    img_np = sitk.GetArrayFromImage(img)
    label_ids_orig = np.unique(img_np)
    try:
        signed = True if mirror_mult == -1 else None
        dtype = libmag.dtype_within_range(
            np.amin(img_np), np.amax(img_np), True, signed)
        if dtype != img_np.dtype:
            print("casting labels image to type", dtype)
            img_np = img_np.astype(dtype)
    except TypeError as e:
        # fallback to large signed data type
        print(e)
        img_np.astype(np.int32)
    img_ref_np = sitk.GetArrayFromImage(img_ref)
    tot_planes = len(img_np)
    
    # lateral edges of atlas labels (ie low z in sagittal orientation) are 
    # missing in many ABA developing mouse atlases, requiring extension
    edgei = 0
    edgei_first = None
    if edge_start is not None and edge_start >= 0:
        # set start of extension from fraction of total of planes
        edgei = int(edge_start * tot_planes)
        print("will extend near edge from plane {}".format(edgei))
    else:
        # default to finding the first non-zero plane; if edge_start is None,
        # will only use this val for metrics and cropping
        for plane in img_np:
            if not np.allclose(plane, 0):
                if edgei_first is None:
                    edgei_first = edgei
                elif edgei - edgei_first >= 1:
                    # require at least 2 contiguous planes with signal
                    edgei = edgei_first
                    break
            else:
                edgei_first = None
            edgei += 1
        print("found start of contiguous non-zero planes at {}".format(edgei))
    
    if edge and edge_start is not None:
        # extend labels from the lowest labeled z-plane to cover the rest
        # of lower planes with signal in the reference image
        save_steps = edge[profiles.RegKeys.SAVE_STEPS]
        if save_steps:
            # load original labels and setup colormaps
            np_io.setup_images()
        extend_edge(
            img_np, img_ref_np, config.atlas_profile["atlas_threshold"],
            None, edgei, edge["surr_size"], edge["smoothing_size"],
            edge["in_paint"], None, edge[profiles.RegKeys.MARKER_EROSION],
            edge[profiles.RegKeys.MARKER_EROSION_MIN],
            edge[profiles.RegKeys.MARKER_EROSION_USE_MIN], edge["wt_lat"],
            save_steps)
    
    if expand:
        # expand selected regions
        for expand_limits in expand:
            # get region from slices specified by tuple of (start, end) pixels
            expand_slices = tuple(
                slice(*limits) for limits in expand_limits[::-1])
            region = img_np[expand_slices]
            region_ref = img_ref_np[expand_slices]
            print("expanding planes in slices", expand_slices)
            for expandi in range(len(region_ref)):
                # find bounding boxes for labels and atlas within region
                bbox = _get_bbox(region[expandi], 0) # assume pos labels region
                shape, slices = cv_nd.get_bbox_region(bbox)
                plane_region = region[expandi, slices[0], slices[1]]
                bbox_ref = _get_bbox(region_ref[expandi])
                shape_ref, slices_ref = cv_nd.get_bbox_region(bbox_ref)
                # expand bounding box region of labels to that of atlas
                plane_region = transform.resize(
                    plane_region, shape_ref, preserve_range=True, order=0, 
                    anti_aliasing=False, mode="reflect")
                region[expandi, slices_ref[0], slices_ref[1]] = plane_region
    
    # find approximate midline by locating the last zero plane from far edge 
    # at which to start mirroring across midline
    mirrori = tot_planes
    for plane in img_np[::-1]:
        if not np.allclose(plane, 0):
            print("found last zero plane from far border at {}".format(mirrori))
            break
        mirrori -= 1
    
    if rotation:
        if mirror_start is not None:
            # mirroring labels with original values in case rotation will cause 
            # some labels to be cut off, then rotate for each specified axis
            for i in range(mirrori, tot_planes):
                img_np[i] = img_np[mirrori - 1]
        for rot in rotation:
            print("rotating by", rot)
            img_np = cv_nd.rotate_nd(
                img_np, rot[0], rot[1], order=0, resize=rotate["resize"])
    
    if affine:
        for aff in affine:
            print("performing affine of", aff)
            img_np = cv_nd.affine_nd(img_np, **aff)
    
    if mirror_start is not None and mirror_start != -1:
        # reset mirror based on fractional profile setting
        mirrori = int(mirror_start * tot_planes)
        print("will mirror starting at plane index {}".format(mirrori))
    
    df_sm = None
    df_sm_raw = None
    if smooth is not None:
        # minimize jaggedness in labels, often seen outside of the original 
        # orthogonal direction, using pre-mirrored slices only since rest will 
        # be overwritten
        img_smoothed = img_np[:mirrori]
        img_smoothed_orig = np.copy(img_smoothed)
        spacing = img.GetSpacing()[::-1]
        if libmag.is_seq(smooth):
            # test sequence of filter sizes via multiprocessing for metrics
            # only, in which case the images will be left unchanged
            df_sm, df_sm_raw = _smoothing_mp(
                img_smoothed, img_smoothed_orig, smooth, spacing)
        else:
            # smooth labels image with single filter size
            _, df_sm, df_sm_raw = _smoothing(
                img_smoothed, img_smoothed_orig, smooth, spacing)
    
    # check that labels will fit in integer type
    libmag.printv(
        "type: {}, max: {}, max avail: {}".format(
            img_np.dtype, np.max(img_np), np.iinfo(img_np.dtype).max))
    
    if mirror_start is None:
        print("Checking baseline labels symmetry without mirroring:")
        check_mirrorred(img_np)
    else:
        # mirror, check beforehand for labels that will be loss
        print("Labels that will be lost from mirroring:")
        find_labels_lost(np.unique(img_np), np.unique(img_np[:mirrori]))
        img_np = mirror_planes(
            img_np, mirrori, mirror_mult=mirror_mult, check_equality=True,
            resize=resize)
        print("total final labels: {}".format(np.unique(img_np).size))
    
    print("\nTotal labels lost:")
    find_labels_lost(label_ids_orig, np.unique(img_np))
    print()
    
    return img_np, (edgei, mirrori), df_sm, df_sm_raw


def extend_edge(region, region_ref, threshold, plane_region, planei,
                surr_size=0, smoothing_size=0, in_paint=False, edges=None,
                marker_erosion=0, marker_erosion_min=None,
                marker_erosion_use_min=False, wt_lat=0, save_steps=False,
                edge_bounds=None):
    """Recursively extend the nearest plane with labels based on the 
    underlying atlas histology.

    Labels in a given atlas may be incomplete, absent along the lateral 
    edges. To fill in these missing labels, the last labeled plane will be 
    used to extend labeling for all distinct structures in the histology 
    ("reference" image). The given reference region will be thresholded 
    to find distinct sub-regions, and each sub-region will be recursively 
    extended to fill each successive lateral plane with labels from the 
    prior plane resized to that of the reference region in the given plane.

    This approach assumes that each nearer plane is the same size or smaller 
    than the next farther plane is, such as the tapering edge of a specimen, 
    since each subsequent plane will be within the bounds of the prior plane. 
    The number of sub-regions to extend is set by the first plane, after 
    which only the largest object within each sub-region will be followed. 
    Labels will be cropped in this first plane to match the size of 
    each corresponding reference region and resized to the size of the 
    largest object in all subsequent planes.
    
    To improve correspondence with the underlying histology, edge-aware
    reannotation can be applied. This reannotation is a 2D/3D implementation,
    where the edge map is generated in 3D, but the reannotation occurs in
    serial 2D, with each generated plane becoming the template for the next
    plane to give smooth transitions from plane to plane. During the erosion
    step, the labels are allowed to disappear to emulate their tapering off.

    Args:
        region (:obj:`np.ndarray`): Labels volume region, which will be 
            extended along decreasing z-planes and updated in-place.
        region_ref (:obj:`np.ndarray`): Corresponding reference 
            (eg histology) region.
        threshold (int, float): Threshold intensity for `region_ref`.
        plane_region (:obj:`np.ndarray`): Labels 2D template that will be 
            resized for current plane; if None, a template will be cropped 
            from `region` at `planei`.
        planei (int): Plane index.
        surr_size (int): Structuring element size for dilating the labeled 
            area that will be considered foreground in `region_ref` 
            for finding regions to extend; defaults to 0 to not dilate.
        surr_size (int): Structuring element size for dilating the labeled 
            area that will be considered foreground in `region_ref` 
            for finding regions to extend; defaults to 0 to not dilate.
        smoothing_size (int): Structuring element size for 
            :func:`smooth_labels`; defaults to 0 to not smooth.
        in_paint (bool): True to in-paint ``region_ref`` foreground not
            present in ``plane_region``; defaults to False.
        edges (:obj:`np.ndarray`): Array of edges for watershed-based
            reannotation. Typically of same size as ``region``. Defaults
            to None to generate new edges from ``region_ref`` if 
            ``marker_erosion`` is > 0.
        marker_erosion (int): Structuring element size for label erosion to
            markers for watershed-based reannotation. Defaults to 0 to
            skip this reannotation.
        marker_erosion_min (int): Minimum size of erosion filter passed to
            :func:`segmenter.labels_to_markers_erosion`; defaults to None.
        marker_erosion_use_min (bool): Flag for using the minimum filter
            size if reached, passed to
            :func:`segmenter.labels_to_markers_erosion`; defaults to False.
        save_steps (bool): True to output intermediate steps as images,
            saving to the extension set in :attr:`config.savefig`; defaults
            to False.
        wt_lat (float): Weight lateral planes more heavily by reducing the
            weight of medial planes' erosion filter sizes by this fraction.
            Defaults to 0 to weight all planes equally. 1 causes the most
            medial plane to undergo no erosion, with erosion weight increasing
            linearly to 1 at the most lateral planes.
        edge_bounds (list[int]): Sequence of indices of lateral ``first, last``
            planes; defaults to None to determine based on edge map.
    
    """
    if planei < 0: return
    
    # find sub-regions in the reference image
    has_template = plane_region is not None
    region_ref_filt = region_ref[planei]
    if not has_template and surr_size > 0:
        # limit the reference image to the labels since when generating 
        # label templates since labels can only be extended from labeled areas; 
        # include padding by dilating slightly for unlabeled borders
        cv_nd.remove_bg_from_dil_fg(
            region_ref_filt, region[planei] != 0, morphology.disk(surr_size))
    # order extension from smallest to largest regions so largest have 
    # final say
    prop_sizes = cv_nd.get_thresholded_regionprops(
        region_ref_filt, threshold=threshold)
    if prop_sizes is None: return
    
    if has_template:
        # resize only largest property
        # TODO: could follow all props separately by generating new templates, 
        # though would need to decide when to crop new templates vs resize 
        num_props = len(prop_sizes)
        if num_props > 1:
            print("plane {}: ignoring smaller {} prop(s) of size(s) {}"
                  .format(planei, num_props - 1,
                          [p[1] for p in prop_sizes[1:]]))
        prop_sizes = prop_sizes[-1:]
    elif edges is None and marker_erosion > 0:
        log_sigma = config.atlas_profile["log_sigma"]
        if log_sigma is not None:
            # generate an edge map based on reference image
            thresh = (config.atlas_profile["atlas_threshold"]
                      if config.atlas_profile["log_atlas_thresh"] else None)
            atlas_log = cv_nd.laplacian_of_gaussian_img(
                region_ref, sigma=log_sigma, thresh=thresh)
            edges = cv_nd.zero_crossing(atlas_log, 1).astype(np.uint8)
    print("plane {}: extending {} props of sizes {}".format(
        planei, len(prop_sizes), [p[1] for p in prop_sizes]))
    
    for prop_size in prop_sizes:
        # get the region from the property
        _, slices = cv_nd.get_bbox_region(prop_size[0].bbox)
        prop_region_ref = region_ref[:, slices[0], slices[1]]
        prop_region = region[:, slices[0], slices[1]]
        edges_region = None
        if edges is not None:
            edges_region = edges[:, slices[0], slices[1]]
            if edge_bounds is None:
                # find first and last planes for extension; predict first plane
                # by presence of edge map, though may differ from true start if
                # sliced edge becomes None in a different plane
                edge_bounds = [0, planei]
                for edgei, edge in enumerate(edges_region):
                    if np.any(edge != 0):
                        edge_bounds[0] = edgei
                        break
        save_imgs = {}
        if not has_template:
            # crop to use corresponding labels as template for next planes
            print("plane {}: generating labels template of size {}"
                  .format(planei, np.sum(prop_region[planei] != 0)))
            prop_plane_region = prop_region[planei]
            if smoothing_size:
                # smooth to remove artifacts
                smooth_labels(prop_plane_region, smoothing_size)
            save_imgs["edge_template_plane{}".format(planei)] = [
                prop_region_ref[planei], prop_plane_region]
        else:
            # resize prior plane's labels to region's shape and replace region
            prop_plane_region = transform.resize(
                plane_region, prop_region[planei].shape, preserve_range=True,
                order=0, anti_aliasing=False, mode="reflect")
            print("plane {}: extending labels with template resized to {}, "
                  "in-painting set to {}"
                  .format(planei, np.sum(prop_plane_region != 0), in_paint))
            plane_add = prop_plane_region
            if in_paint:
                # in-paint to fill missing areas (eg ventricles that closed,
                # edges that don't align perfectly) based on thresholding, only
                # adding but not subtracting label pixels and retaining the
                # template plane for subsequent planes
                fg = plane_add != 0
                fg_thresh = prop_region_ref[planei] > threshold
                to_fill = np.logical_and(fg_thresh, ~fg)
                plane_add = cv_nd.in_paint(plane_add, to_fill)
            save_imgs["edge_resized_plane{}".format(planei)] = [
                    prop_region_ref[planei], plane_add]
            if edges_region is not None:
                # reannotate based on edge map; allow erosion to lose labels to
                # mimic tapering off of labels, preferentially eroding
                # centrally located labels
                perim = cv_nd.perimeter_nd(
                    plane_add != 0, largest_only=True)
                wt_dists = cv_nd.signed_distance_transform(~perim)
                filt_size = marker_erosion
                filt_size_min = marker_erosion_min
                if edge_bounds is not None and wt_lat:
                    # weight lateral planes more heavily by reducing weight
                    # of medial planes
                    wt_lat_frac = 1 - wt_lat * (
                            float(planei) - edge_bounds[0]) / (
                            edge_bounds[1] - edge_bounds[0])
                    filt_size *= wt_lat_frac
                    filt_size_min *= wt_lat_frac
                    print("planei", planei, "edge_bounds", edge_bounds,
                          "wt_lat", wt_lat_frac)
                markers, _ = segmenter.labels_to_markers_erosion(
                    plane_add, filt_size, -1, filt_size_min,
                    marker_erosion_use_min, wt_dists=wt_dists)
                plane_add = segmenter.segment_from_labels(
                    edges_region[planei], markers, plane_add)
                # make resulting plane the new template for smoother
                # transitions between planes
                prop_plane_region = plane_add
                save_imgs["edge_markers_plane{}".format(planei)] = [
                    prop_region_ref[planei], markers, edges_region[planei]]
                save_imgs["edge_annot_plane{}".format(planei)] = [
                    prop_region_ref[planei], prop_plane_region]
            prop_region[planei] = plane_add
        if save_steps:
            # export overlaid planes in single files
            for key, val in save_imgs.items():
                export_stack.reg_planes_to_img(val, key)
        # recursively call for each region to follow in next plane, but 
        # only get largest region for subsequent planes in case 
        # new regions appear, where the labels would be unknown
        extend_edge(
            prop_region, prop_region_ref, threshold, prop_plane_region,
            planei - 1, surr_size, smoothing_size, in_paint, edges_region,
            marker_erosion, marker_erosion_min, marker_erosion_use_min,
            wt_lat, save_steps, edge_bounds)


def crop_to_orig(labels_img_np_orig, labels_img_np, crop):
    """Crop new labels to extent of original labels.
    
    Args:
        labels_img_np_orig (:obj:`np.ndarray`): Original labels image array.
        labels_img_np (:obj:`np.ndarray`): Labels image array, which will
            be cropped in-place to ``labels_img_np_orig``.
        crop (bool): True to apply morphological opening to
            ``labels_img_np_orig`` before cropping.

    """
    print("cropping to original labels' extent with filter size of", crop)
    if crop is False: return
    mask = labels_img_np_orig == 0
    if crop > 0:
        # smooth mask
        mask = morphology.binary_opening(mask, morphology.ball(crop))
    labels_img_np[mask] = 0


def _smoothing(img_np, img_np_orig, filter_size, spacing=None):
    """Smooth image and calculate smoothing metric for use individually or 
    in multiprocessing.
    
    Args:
        img_np: Image as Numpy array, which will be directly updated.
        img_np_orig: Original image as Numpy array for comparison with 
            smoothed image in metric.
        filter_size: Structuring element size for smoothing.
        spacing: Voxel spacing corresponing to ``img_np`` dimensions; 
            defaults to None.
    
    Returns:
        Tuple of ``filter_size`` and a data frame of smoothing metrices.
    """
    smoothing_mode = config.atlas_profile["smoothing_mode"]
    smooth_labels(img_np, filter_size, smoothing_mode)
    df_metrics, df_raw = label_smoothing_metric(
        img_np_orig, img_np, filter_size, spacing)
    print("\nAggregated smoothing metrics, weighted by original volume")
    df_io.print_data_frame(df_metrics)
    
    # curate back to lightly smoothed foreground of original labels
    crop = config.atlas_profile["crop_to_orig"]
    crop_to_orig(img_np_orig, img_np, crop)
    
    print("\nMeasuring foreground overlap of labels after smoothing:")
    measure_overlap_labels(
        make_labels_fg(sitk.GetImageFromArray(img_np)), 
        make_labels_fg(sitk.GetImageFromArray(img_np_orig)))
    
    return filter_size, df_metrics, df_raw


def _smoothing_mp(img_np, img_np_orig, filter_sizes, spacing=None):
    """Calculate smoothing metrics for a list of smoothing strengths.
    
    Args:
        img_np: Image as Numpy array, which will be not be updated.
        img_np_orig: Original image as Numpy array for comparison with 
            smoothed image in metric.
        filter_sizes: Tuple or list of structuring element sizes.
        spacing: Voxel spacing corresponing to ``img_np`` dimensions; 
            defaults to None.
    
    Returns:
        Data frame of combined metrics from smoothing for each filter size.
    """
    pool = chunking.get_mp_pool()
    pool_results = []
    for n in filter_sizes:
        pool_results.append(
            pool.apply_async(
                _smoothing, args=(img_np, img_np_orig, n, spacing)))
    dfs_metrics = []
    dfs_raw = []
    for result in pool_results:
        filter_size, df_metrics, df_raw = result.get()
        dfs_metrics.append(df_metrics)
        dfs_raw.append(df_raw)
        print("finished smoothing with filter size {}".format(filter_size))
    pool.close()
    pool.join()
    return pd.concat(dfs_metrics), pd.concat(dfs_raw)


def find_labels_lost(label_ids_orig, label_ids, label_img_np_orig=None):
    """Find labels lost and optionally the size of each label.
    
    Args:
        label_ids_orig (List[int]): Sequence of original label IDs.
        label_ids (List[int]): Sequence of new label IDs.
        label_img_np_orig (:obj:`np.ndarray`): Original labels image 
            array to show the size of lost regions; defaults to None.

    Returns:
        List[int]: Sequence of missing labels.

    """
    print("number of labels changed from {} to {}"
          .format(label_ids_orig.size, label_ids.size))
    labels_lost = label_ids_orig[np.isin(
        label_ids_orig, label_ids, invert=True)]
    print("IDs of labels lost: {}".format(labels_lost))
    if label_img_np_orig is not None:
        for lost in labels_lost:
            region_lost = label_img_np_orig[label_img_np_orig == lost]
            print("size of lost label {}: {}".format(lost, region_lost.size))
    return labels_lost


def smooth_labels(labels_img_np, filter_size=3, mode=None):
    """Smooth each label within labels annotation image.
    
    Labels images created in one orthogonal direction may have ragged, 
    high-frequency edges when viewing in the other orthogonal directions. 
    Smooth these edges by applying a filter to each label.
    
    Args:
        labels_img_np: Labels image as a Numpy array.
        filter_size: Structuring element or kernel size; defaults to 3.
        mode: One of :attr:``config.SmoothingModes``, where ``opening`` applies 
            a morphological opening filter unless the size is severely 
            reduced, in which case a closing filter is applied instead;  
            ``gaussian`` applies a Gaussian blur; and ``closing`` applies 
            a closing filter only.
    """
    if mode is None: mode = config.SmoothingModes.opening
    print("Smoothing labels with filter size of {}, mode {}"
          .format(filter_size, mode))
    if filter_size == 0:
        print("filter size of 0, skipping")
        return
    
    # copy original for comparison
    labels_img_np_orig = np.copy(labels_img_np)
    fn_selem = cv_nd.get_selem(labels_img_np.ndim)
    
    # sort labels by size, starting from largest to smallest
    label_ids = np.unique(labels_img_np)
    label_sizes = {}
    for label_id in label_ids:
        label_sizes[label_id] = len(labels_img_np[labels_img_np == label_id])
    label_sizes_ordered = OrderedDict(
        sorted(label_sizes.items(), key=lambda x: x[1], reverse=True))
    label_ids_ordered = label_sizes_ordered.keys()
    
    for label_id in label_ids_ordered:
        # smooth by label
        print("smoothing label ID {}".format(label_id))
        
        # get bounding box for label region
        bbox = cv_nd.get_label_bbox(labels_img_np, label_id)
        if bbox is None: continue
        _, slices = cv_nd.get_bbox_region(
            bbox, np.ceil(2 * filter_size).astype(int), labels_img_np.shape)
        
        # get region, skipping if no region left
        region = labels_img_np[tuple(slices)]
        label_mask_region = region == label_id
        region_size = np.sum(label_mask_region)
        if region_size == 0:
            print("no pixels to smooth, skipping")
            continue
        
        # smoothing based on mode
        region_size_smoothed = 0
        if mode is config.SmoothingModes.opening:
            # smooth region with opening filter, reducing filter size for 
            # small volumes and changing to closing filter 
            # if region would be lost or severely reduced
            selem_size = filter_size
            if region_size < 5000:
                selem_size = selem_size // 2
                print("using a smaller filter size of {} for a small region "
                      "of {} pixels".format(selem_size, region_size))
            selem = fn_selem(selem_size)
            smoothed = morphology.binary_opening(label_mask_region, selem)
            region_size_smoothed = np.sum(smoothed)
            size_ratio = region_size_smoothed / region_size
            if size_ratio < 0.01:
                print("region would be lost or too small "
                      "(ratio {}), will use closing filter instead"
                      .format(size_ratio))
                smoothed = morphology.binary_closing(label_mask_region, selem)
                region_size_smoothed = np.sum(smoothed)
            
            # fill original label space with closest surrounding labels
            # to fill empty spaces that would otherwise remain after
            # replacing the original with the smoothed label
            region = cv_nd.in_paint(region, label_mask_region)
            
        elif mode is config.SmoothingModes.gaussian:
            # smoothing with gaussian blur
            smoothed = filters.gaussian(
                label_mask_region, filter_size, mode="nearest", 
                multichannel=False).astype(bool)
            region_size_smoothed = np.sum(smoothed)
            
        elif mode is config.SmoothingModes.closing:
            # smooth region with closing filter
            smoothed = morphology.binary_closing(
                label_mask_region, morphology.ball(filter_size))
            region_size_smoothed = np.sum(smoothed)
            
            # fill empty spaces with closest surrounding labels
            region = cv_nd.in_paint(region, label_mask_region)
        
        # replace smoothed volume within in-painted region
        region[smoothed] = label_id
        labels_img_np[tuple(slices)] = region
        print("changed num of pixels from {} to {}"
              .format(region_size, region_size_smoothed))
    
    # show label loss metric
    print("\nLabels lost from smoothing:")
    label_ids_smoothed = np.unique(labels_img_np)
    find_labels_lost(
        label_ids, label_ids_smoothed, label_img_np_orig=labels_img_np_orig)
    
    # show DSC for labels
    print("\nMeasuring overlap of labels:")
    measure_overlap_labels(
        sitk.GetImageFromArray(labels_img_np_orig), 
        sitk.GetImageFromArray(labels_img_np))
    
    # weighted pixel ratio metric of volume change
    weighted_size_ratio = 0
    tot_pxs = 0
    for label_id in label_ids_ordered:
        # skip background since not a "region"
        if label_id == 0: continue
        size_orig = np.sum(labels_img_np_orig == label_id)
        size_smoothed = np.sum(labels_img_np == label_id)
        weighted_size_ratio += size_smoothed
        tot_pxs += size_orig
    weighted_size_ratio /= tot_pxs
    print("\nVolume ratio (smoothed:orig) weighted by orig size: {}\n"
          .format(weighted_size_ratio))


def label_smoothing_metric(orig_img_np, smoothed_img_np, filter_size=None,
                           spacing=None):
    """Measure degree of appropriate smoothing, defined as smoothing that 
    retains the general shape and placement of the region.
    
    Compare the difference in compactness before and after the smoothing 
    algorithm, termed "compaction," while penalizing inappropriate smoothing, 
    the smoothed volume lying outside of the original broad volume, termed 
    "displacement."
    
    Args:
        orig_img_np: Unsmoothed labels image as Numpy array.
        smoothed_img_np: Smoothed labels image as Numpy array, which 
            should be of the same shape as ``original_img_np``.
        filter_size (float): Structuring element size for smoothing, used
            here only to include in output data frame; defaults to None.
        spacing (list[float]): Sequence of voxel spacing in same order
            as for ``img_np``; defaults to None.
    
    Returns:
        Tuple of a data frame of the smoothing metrics and another data 
        frame of the raw metric components.
    """
    def meas_compactness(img_np):
        # get the borders of the label and add them to a rough image
        region = img_np[tuple(slices)]
        mask = region == label_id
        if np.sum(mask) == 0:
            print("label {} warning: region missing".format(label_id))
            return mask, 0, 0, np.nan
        compactness, area, vol = cv_nd.compactness_3d(mask, spacing)
        return mask, area, vol, compactness
    
    pxs = {}
    spacing_prod = 1 if spacing is None else np.prod(spacing)
    label_ids = np.unique(orig_img_np)
    for label_id in label_ids:
        # calculate metric for each label
        if label_id == 0: continue
        
        # use bounding box that fits around label in both original and 
        # smoothed image to improve efficiency over filtering whole image
        label_mask = np.logical_or(
            orig_img_np == label_id, smoothed_img_np == label_id)
        props = measure.regionprops(label_mask.astype(np.int))
        if len(props) < 1 or props[0].bbox is None: continue
        _, slices = cv_nd.get_bbox_region(
            props[0].bbox, 2, orig_img_np.shape)
        
        # measure surface area for SA:vol and to get vol mask
        mask_orig, area_orig, vol_orig, compact_orig = meas_compactness(
            orig_img_np)
        mask_smoothed, area_sm, vol_sm, compact_sm = meas_compactness(
            smoothed_img_np)
        
        # "compaction": fraction of reduced compactness
        compaction = (compact_orig - compact_sm) / compact_orig
        
        # "displacement": fraction of volume that is displaced
        displ = (np.sum(np.logical_and(mask_smoothed, ~mask_orig))
                 * spacing_prod / vol_sm) if vol_sm > 0 else np.nan
        
        # "smoothing quality": difference of compaction and displacement
        sm_qual = compaction - displ
        
        # SA:vol metrics
        sa_to_vol_orig = area_orig / vol_orig
        sa_to_vol_smoothed = np.nan
        if vol_sm > 0:
            sa_to_vol_smoothed = area_sm / vol_sm
        sa_to_vol_ratio = sa_to_vol_smoothed / sa_to_vol_orig
        
        label_metrics = {
            config.AtlasMetrics.REGION: label_id,
            config.SmoothingMetrics.FILTER_SIZE: filter_size,
            config.SmoothingMetrics.COMPACTION: compaction, 
            config.SmoothingMetrics.DISPLACEMENT: displ, 
            config.SmoothingMetrics.SM_QUALITY: sm_qual, 
            config.SmoothingMetrics.VOL_ORIG: vol_orig, 
            config.SmoothingMetrics.VOL: vol_sm, 
            config.SmoothingMetrics.COMPACTNESS_ORIG: compact_orig,
            config.SmoothingMetrics.COMPACTNESS: compact_sm,
            config.SmoothingMetrics.SA_VOL_ORIG: sa_to_vol_orig,
            config.SmoothingMetrics.SA_VOL: sa_to_vol_smoothed,
            config.SmoothingMetrics.SA_VOL_FRAC: sa_to_vol_ratio, 
        }
        for key, val in label_metrics.items():
            pxs.setdefault(key, []).append(val)
        print("label: {}, compaction: {}, displacement: {}, "
              "smoothing quality: {}"
              .format(label_id, compaction, displ, sm_qual))
    
    # print raw stats and calculate aggregate stats
    df_pxs = df_io.dict_to_data_frame(pxs)
    print()
    df_io.print_data_frame(df_pxs)
    df_metrics = aggr_smoothing_metrics(df_pxs)
    return df_metrics, df_pxs


def aggr_smoothing_metrics(df_pxs):
    """Aggregate smoothing metrics from a data frame of raw stats by label.
    
    Stats generally compare original and smoothed versions, but when stats of 
    a single version, typically the smoothed version is given. Stats are 
    weighted by the original volume.
    
    Args:
        df_pxs (:class:`pandas.DataFrame`): Data frame with raw stats by label.

    Returns:
        :class:`pandas.DataFrame`: Data frame of aggregated stats.

    """
    keys = [
        config.SmoothingMetrics.FILTER_SIZE,
        config.SmoothingMetrics.COMPACTION,
        config.SmoothingMetrics.DISPLACEMENT,
        config.SmoothingMetrics.SM_QUALITY,
        config.SmoothingMetrics.COMPACTNESS,
        config.SmoothingMetrics.COMPACTNESS_SD,
        config.SmoothingMetrics.COMPACTNESS_CV,
        config.SmoothingMetrics.SA_VOL,
        config.SmoothingMetrics.SA_VOL_FRAC
    ]
    metrics = {}
    wts = df_pxs[config.SmoothingMetrics.VOL_ORIG.value]
    for key in keys:
        if key is config.SmoothingMetrics.COMPACTNESS_SD:
            # standard deviation
            sd, _ = df_io.weight_std(
                df_pxs[config.SmoothingMetrics.COMPACTNESS.value], wts)
            metrics[key] = [sd]
        elif key is config.SmoothingMetrics.COMPACTNESS_CV:
            # coefficient of variation, which assumes that SD and weighted 
            # mean have already been measured
            metrics[key] = [
                metrics[config.SmoothingMetrics.COMPACTNESS_SD][0]
                / metrics[config.SmoothingMetrics.COMPACTNESS][0]]
        else:
            # default to weighted mean
            metrics[key] = [df_io.weight_mean(df_pxs[key.value], wts)]
    # measure label loss based on number of labels whose smoothed vol is 0
    num_labels_orig = np.sum(wts > 0)
    if num_labels_orig > 0:
        label_loss = (num_labels_orig - np.sum(
            df_pxs[config.SmoothingMetrics.VOL.value] > 0)) / num_labels_orig
    else:
        label_loss = np.nan
    metrics[config.SmoothingMetrics.LABEL_LOSS] = [label_loss]
    return df_io.dict_to_data_frame(metrics)


def transpose_img(img_sitk, plane=None, rotate=None, target_size=None,
                  flipud=None):
    """Transpose a SimpleITK format image via Numpy and re-export to SimpleITK.
    
    Args:
        img_sitk (:obj:`sitk.Image`): Image in SimpleITK format.
        plane (str): One of :attr:`config.PLANES` elements, specifying the
            planar orientation in which to transpose the image. The current 
            orientation is taken to be "xy". Defaults to None, in which case
            the value from :attr:`config.plane` will be taken.
        rotate: Number of times to rotate by 90 degrees; defaults to None, in
            which case the value will be automatically determined based on
            :attr:`config.transform` and ``plane``.
        target_size (List[int]): Size of target image, typically one to which
            ``img_sitk`` will be registered, in (x,y,z, SimpleITK standard)
            ordering.
        flipud (bool): True to invert the z-axis after transposition;
            defaults to None, in which case it will be based on ``plane``.
    
    Returns:
        :obj:`sitk.Image`: The transposed image. If the original image
        will undergo no transformations, ``img_sitk`` is simply returned.

    """
    if plane is None:
        plane = config.plane
    if rotate is None:
        # default to getting first rotation number
        rotate = libmag.get_if_within(
            config.transform[config.Transforms.ROTATE], 0)
    rotate_num = rotate if rotate else 0
    if plane in config.PLANE[1:]:
        # assume rotation and inversion based on planar transposition
        # TODO: check if holds generally true for these planar transforms
        if rotate is None:
            rotate_num += 1
        if flipud is None:
            flipud = True
    if flipud is None:
        flipud = False
    rotate = rotate_num
    print("Image transformation settings: plane {}, num of rotations {}, "
          "target_size {}, z-axis inversion {}"
          .format(plane, rotate, target_size, flipud))
    if ((not plane or plane == config.PLANE[0]) and not rotate
            and target_size is None and not flipud):
        print("No transformations to apply, skipping")
        return img_sitk

    img = sitk.GetArrayFromImage(img_sitk)
    img_dtype = img.dtype
    spacing = img_sitk.GetSpacing()[::-1]
    origin = img_sitk.GetOrigin()[::-1]
    transposed = img
    if plane is not None and plane != config.PLANE[0]:
        # transpose planes and metadata
        arrs_3d, arrs_1d = plot_support.transpose_images(
            plane, [transposed], [spacing, origin])
        transposed = arrs_3d[0]
        spacing, origin = arrs_1d
    if flipud:
        # invert along z-axis
        transposed = np.flipud(transposed)
    if rotate:
        # rotate the final output image by 90 deg
        # TODO: need to change origin? make axes accessible (eg (0, 2) for 
        # horizontal rotation)
        transposed = np.rot90(transposed, rotate, (1, 2))
        if rotate % 2 != 0:
            spacing = libmag.swap_elements(spacing, 1, 2)
            origin = libmag.swap_elements(origin, 1, 2)
    resize_factor = config.atlas_profile["resize_factor"]
    if target_size is not None and resize_factor:
        # rescale based on xy dimensions of given and target image so that
        # they are not so far off from one another that scaling does not occur; 
        # assume that size discrepancies in z don't affect registration and 
        # for some reason may even prevent registration
        size_diff = np.divide(target_size[::-1][1:3], transposed.shape[1:3])
        rescale = np.mean(size_diff) * resize_factor
        if rescale > 0.5:
            print("rescaling image by {}x after applying resize factor of {}"
                  .format(rescale, resize_factor))
            transposed = transform.rescale(
                transposed, rescale, mode="constant", preserve_range=True, 
                multichannel=False, anti_aliasing=False, 
                order=0).astype(img_dtype)
            spacing = np.divide(spacing, rescale)
        # casted back since transpose changes data type even when 
        # preserving range
        print(transposed.dtype, np.min(transposed), np.max(transposed))
    transposed = sitk.GetImageFromArray(transposed)
    transposed.SetSpacing(spacing[::-1])
    transposed.SetOrigin(origin[::-1])
    return transposed


def match_atlas_labels(img_atlas, img_labels, metrics=None):
    """Apply register profile settings to labels and match atlas image 
    accordingly.
    
    Args:
        img_atlas (:obj:`sitk.Image`): Reference image, such as histology.
        img_labels (:obj:`sitk.Image`): Labels image.
        metrics (:obj:`dict`): Dictionary to store metrics; defaults to 
            None, in which case metrics will not be measured.
    
    Returns:
        Tuple: ``img_atlas``, the updated atlas; ``img_labels``, the 
        updated labels; ``img_borders``, a new (:obj:`sitk.Image`) of the 
        same shape as the prior images except an extra channels dimension 
        as given by :func:``_curate_labels``; ``df_sm``, a 
        data frame of smoothing stats, or None if smoothing was not performed; 
        and ``df_sm_raw``, a data frame of raw smoothing stats, or 
        None if smoothing was not performed.
    """
    pre_plane = config.atlas_profile["pre_plane"]
    mirror = config.atlas_profile["labels_mirror"]
    is_mirror = mirror and mirror[profiles.RegKeys.ACTIVE]
    edge = config.atlas_profile["labels_edge"]
    is_edge = edge and edge[profiles.RegKeys.ACTIVE]
    expand = config.atlas_profile["expand_labels"]
    rotate = config.atlas_profile["rotate"]
    rotation = rotate["rotation"] if rotate else None
    smooth = config.atlas_profile["smooth"]
    crop = config.atlas_profile["crop_to_labels"]
    affine = config.atlas_profile["affine"]
    far_hem_neg = config.atlas_profile["make_far_hem_neg"]
    
    if pre_plane:
        # images in the correct desired orientation may need to be 
        # transposed prior to label curation since mirroring assumes 
        # a sagittal orientation
        img_atlas_np = sitk.GetArrayFromImage(img_atlas)
        img_labels_np = sitk.GetArrayFromImage(img_labels)
        arrs_3d, _ = plot_support.transpose_images(
            pre_plane, [img_atlas_np, img_labels_np])
        img_atlas_np = arrs_3d[0]
        img_labels_np = arrs_3d[1]
        img_atlas = sitk_io.replace_sitk_with_numpy(img_atlas, img_atlas_np)
        img_labels = sitk_io.replace_sitk_with_numpy(img_labels, img_labels_np)
    
    # curate labels
    mask_lbls = None  # mask of fully extended/mirrored labels for cropping
    extis = None  # extension indices of 1st labeled, then unlabeled planes
    if is_mirror and is_edge:
        # include any lateral extension and mirroring if ACTIVE flag is on
        img_labels_np, extis, df_sm, df_sm_raw = (
            _curate_labels(
                img_labels, img_atlas, mirror, edge, expand, rotate, smooth, 
                affine))
    else:
        # turn off lateral extension and/or mirroring
        img_labels_np, _, df_sm, df_sm_raw = _curate_labels(
            img_labels, img_atlas, mirror if is_mirror else None, 
            edge if is_edge else None, expand, rotate, smooth,
            affine)
        if metrics or crop and (
                mirror["start"] is not None or edge["start"] is not None):
            # use edge and mirror settings even if ACTIVE is off, but only
            # for metrics and to get the mask for cropping
            print("\nCurating labels with extension/mirroring only "
                  "for measurements and any cropping:")
            resize = is_mirror and mirror["start"] is not None
            lbls_np_mir, extis, _, _ = _curate_labels(
                img_labels, img_atlas, mirror, edge, expand, rotate, None, 
                affine, resize)
            mask_lbls = lbls_np_mir != 0
            print()
    
    # adjust atlas with same settings
    img_atlas_np = sitk.GetArrayFromImage(img_atlas)
    if rotation:
        for rot in rotation:
            img_atlas_np = cv_nd.rotate_nd(
                img_atlas_np, rot[0], rot[1], resize=rotate["resize"])
    if affine:
        for aff in affine:
            img_atlas_np = cv_nd.affine_nd(img_atlas_np, **aff)
    if is_mirror and mirror["start"] is not None:
        if mirror["atlas_mirror"]:
            # mirror underlying intensity image
            # TODO: consider removing dup since not using
            dup = config.atlas_profile["labels_dup"]
            img_atlas_np = mirror_planes(
                img_atlas_np, extis[1], start_dup=dup)
        else:
            labels_len = len(img_labels_np)
            atlas_len = len(img_atlas_np)
            if labels_len != atlas_len:
                # pad image with fewer z-planes at end since mirroring may
                # increase or decrease number of labels planes
                if labels_len > atlas_len:
                    img_atlas_np = plot_3d.pad_img(
                        img_atlas_np, (0, 0, 0), (labels_len, ))
                else:
                    img_labels_np = plot_3d.pad_img(
                        img_labels_np, (0, 0, 0), (atlas_len, ))

    crop_offset = None
    if crop:
        # crop atlas to the mask of the labels with some padding
        img_labels_np, img_atlas_np, crop_sl = cv_nd.crop_to_labels(
            img_labels_np, img_atlas_np, mask_lbls)
        if crop_sl[0].start > 0:
            # offset extension indices and crop labels mask
            extis = tuple(n - crop_sl[0].start for n in extis)
            if mask_lbls is not None:
                mask_lbls = mask_lbls[tuple(crop_sl)]
        crop_offset = tuple(s.start for s in crop_sl)

    if far_hem_neg and np.all(img_labels_np >= 0):
        # unmirrored images typically have only pos labels for both
        # hemispheres, but metrics assume that the far hem is neg to
        # distinguish sides; to make those labels neg, invert pos labels
        # there if they are >=1/3 of total labels, not just spillover from
        # the near side; also convert to signed data type if necessary
        dtype = libmag.dtype_within_range(
            np.amin(img_atlas_np), np.amax(img_labels_np), signed=True)
        if dtype != img_labels_np.dtype:
            img_labels_np = img_labels_np.astype(dtype)
        half_lbls = img_labels_np[extis[1]:]
        if (np.sum(half_lbls < 0) == 0 and
                np.sum(half_lbls != 0) > np.sum(img_labels_np != 0) / 3):
            print("less than half of labels in right hemisphere are neg; "
                  "inverting all pos labels in x >= {} (shape {}) for "
                  "sided metrics".format(extis[1], img_labels_np.shape))
            half_lbls[half_lbls > 0] *= -1
    
    if metrics is not None:
        
        # meas DSC of labeled hemisphere, using the sagittal midline 
        # to define the hemispheric boundaries
        dsc = measure_overlap_combined_labels(
            sitk.GetImageFromArray(img_atlas_np[:extis[1]]), 
            sitk.GetImageFromArray(img_labels_np[:extis[1]]), 
            config.atlas_profile["overlap_meas_add_lbls"])
        metrics[config.AtlasMetrics.DSC_ATLAS_LABELS_HEM] = [dsc]
        
        # meas frac of hemisphere that is unlabeled using "mirror" bounds
        thresh = config.atlas_profile["atlas_threshold_all"]
        thresh_atlas = img_atlas_np > thresh
        # some edge planes may be partially labeled
        lbl_edge = np.logical_and(
            img_labels_np[:extis[0]] != 0, thresh_atlas[:extis[0]])
        # simply treat rest of hem as labeled to focus on unlabeled lat portion
        metrics[config.AtlasMetrics.LAT_UNLBL_VOL] = 1 - (
            (np.sum(lbl_edge) + np.sum(thresh_atlas[extis[0]:extis[1]])) 
            / np.sum(thresh_atlas[:extis[1]]))

        # meas frac of planes that are at least partially labeled, using 
        # mask labels since from atlas portion that should be labeled
        frac = 0   # mask labels fully covered, so fully labeled by this def
        if mask_lbls is not None:
            mask_lbls_unrot = mask_lbls
            lbls_unrot = img_labels_np
            if rotation:
                # un-rotate so sagittal planes are oriented as orig drawn
                resize = rotate["resize"]
                for rot in rotation[::-1]:
                    mask_lbls_unrot = cv_nd.rotate_nd(
                        mask_lbls_unrot, -rot[0], rot[1], 0, resize)
                    lbls_unrot = cv_nd.rotate_nd(
                        lbls_unrot, -rot[0], rot[1], 0, resize)
            planes_lbl = 0
            planes_tot = 0.
            for i in range(extis[1]):
                if not np.all(mask_lbls_unrot[i] == 0):
                    # plane that should be labeled
                    planes_tot += 1
                    if not np.all(lbls_unrot[i] == 0):
                        # plane is at least partially labeled
                        planes_lbl += 1
            if planes_tot > 0:
                frac = 1 - (planes_lbl / planes_tot)
        metrics[config.AtlasMetrics.LAT_UNLBL_PLANES] = frac

    imgs_np = (img_atlas_np, img_labels_np)
    if pre_plane:
        # transpose back to original orientation
        imgs_np, _ = plot_support.transpose_images(
            pre_plane, imgs_np, rev=True)

    imgs_sitk = (img_atlas, img_labels)
    imgs_sitk_replaced = []
    for img_np, img_sitk in zip(imgs_np, imgs_sitk):
        if img_np is not None:
            # convert back to sitk img
            img_sitk = sitk_io.replace_sitk_with_numpy(img_sitk, img_np)
            if crop_offset:
                # shift origin for any cropping
                img_sitk.SetOrigin(np.add(
                    img_sitk.GetOrigin(), crop_offset[::-1]))

            # perform any transpositions
            img_sitk = transpose_img(img_sitk)

        imgs_sitk_replaced.append(img_sitk)
    img_atlas, img_labels = imgs_sitk_replaced
    
    return img_atlas, img_labels, df_sm, df_sm_raw


def import_atlas(atlas_dir, show=True, prefix=None):
    """Import atlas from the given directory, processing it according 
    to the register settings specified at :attr:``config.register_settings``.
    
    The imported atlas will be saved to a directory of the same path as 
    ``atlas_dir`` except with ``_import`` appended to the end. DSC 
    will be calculated and saved as a CSV file in this directory as well.
    
    Args:
        atlas_dir (str): Path to atlas directory.
        show (bool): True to show the imported atlas.
        prefix (str): Output path; defaults to None to ignore. If ends with
            a file separator,``atlas_dir`` will still be used for the output
            filename; otherwise, the basename will be used for this filename.
    """
    # load atlas and corresponding labels
    img_atlas, path_atlas = sitk_io.read_sitk(
        os.path.join(atlas_dir, config.RegNames.IMG_ATLAS.value))
    img_labels, _ = sitk_io.read_sitk(
        os.path.join(atlas_dir, config.RegNames.IMG_LABELS.value))
    unit_factor = config.atlas_profile["unit_factor"]
    if unit_factor:
        # adjust spacing to a different unit
        print("Adjusing atlas spacing by a factor of", unit_factor)
        for img in (img_atlas, img_labels):
            img.SetSpacing(np.multiply(img.GetSpacing(), unit_factor))
    
    # prep export paths
    target_dir = atlas_dir + "_import"
    basename = os.path.basename(atlas_dir)
    if prefix:
        # split prefix into dir and filename template; prefix_basename is
        # empty if prefix ends with file separator
        target_dir = os.path.dirname(prefix)
        prefix_basename = os.path.basename(prefix)
        if prefix_basename:
            basename = prefix_basename
    df_base_path = os.path.join(target_dir, basename) + "_{}"
    df_metrics_path = df_base_path.format(config.PATH_ATLAS_IMPORT_METRICS)
    name_prefix = os.path.join(target_dir, basename) + "."
    
    # set up condition
    overlap_meas_add = config.atlas_profile["overlap_meas_add_lbls"]
    edge = config.atlas_profile["labels_edge"]
    mirror = config.atlas_profile["labels_mirror"]
    if (edge and edge[profiles.RegKeys.ACTIVE]
            or mirror and mirror[profiles.RegKeys.ACTIVE]):
        cond = "extended" 
    else:
        # show baseline DSC of atlas to labels before any processing
        cond = "original"
        print("\nRaw DSC before import:")
        measure_overlap_combined_labels(
            img_atlas, img_labels, overlap_meas_add)
    
    # prep metrics
    metrics = {
        config.AtlasMetrics.SAMPLE: [basename], 
        config.AtlasMetrics.REGION: config.REGION_ALL, 
        config.AtlasMetrics.CONDITION: cond, 
    }
    
    # match atlas and labels to one another
    img_atlas, img_labels, df_sm, df_sm_raw = match_atlas_labels(
        img_atlas, img_labels, metrics=metrics)
    
    truncate = config.atlas_profile["truncate_labels"]
    if truncate:
        # truncate labels
        img_labels_np = truncate_labels(
            sitk.GetArrayFromImage(img_labels), *truncate)
        img_labels = sitk_io.replace_sitk_with_numpy(img_labels, img_labels_np)
    
    # show labels
    print("labels output data type:", img_labels.GetPixelIDTypeAsString())
    img_labels_np = sitk.GetArrayFromImage(img_labels)
    label_ids = np.unique(img_labels_np)
    print("number of labels: {}".format(label_ids.size))
    print(label_ids)
    
    # DSC and total volumes of atlas and labels
    print("\nDSC after import:")
    dsc, atlas_mask, labels_mask = measure_overlap_combined_labels(
        img_atlas, img_labels, overlap_meas_add, return_masks=True)
    metrics[config.AtlasMetrics.DSC_ATLAS_LABELS] = [dsc]
    metrics[config.AtlasMetrics.VOL_ATLAS] = [np.sum(atlas_mask)]
    metrics[config.AtlasMetrics.VOL_LABELS] = [np.sum(labels_mask)]
    
    # compactness of whole atlas (non-label) image; use lower threshold for 
    # compactness measurement to minimize noisy surface artifacts
    img_atlas_np = sitk.GetArrayFromImage(img_atlas)
    thresh = config.atlas_profile["atlas_threshold_all"]
    thresh_atlas = img_atlas_np > thresh
    compactness, _, _ = cv_nd.compactness_3d(
        thresh_atlas, img_atlas.GetSpacing()[::-1])
    metrics[config.SmoothingMetrics.COMPACTNESS] = [compactness]
    
    # write images with atlas saved as MagellanMapper/Numpy format to 
    # allow opening as an image within MagellanMapper alongside the labels image
    imgs_write = {
        config.RegNames.IMG_ATLAS.value: img_atlas, 
        config.RegNames.IMG_LABELS.value: img_labels}
    sitk_io.write_reg_images(
        imgs_write, name_prefix, copy_to_suffix=True, 
        ext=os.path.splitext(path_atlas)[1])
    config.resolutions = [img_atlas.GetSpacing()[::-1]]
    img_ref_np = sitk.GetArrayFromImage(img_atlas)
    img_ref_np = img_ref_np[None]
    importer.save_np_image(img_ref_np, name_prefix, 0)

    if df_sm_raw is not None:
        # write raw smoothing metrics
        df_io.data_frames_to_csv(
            df_sm_raw, 
            df_base_path.format(config.PATH_SMOOTHING_RAW_METRICS))
        
    if df_sm is not None:
        # write smoothing metrics to CSV with identifier columns
        df_smoothing_path = df_base_path.format(config.PATH_SMOOTHING_METRICS)
        df_sm.insert(0, config.AtlasMetrics.SAMPLE.value, basename)
        df_sm.insert(0, config.AtlasMetrics.REGION.value, config.REGION_ALL)
        df_sm.insert(0, config.AtlasMetrics.CONDITION.value, "smoothed")
        df_sm.loc[
            df_sm[config.SmoothingMetrics.FILTER_SIZE.value] == 0,
            config.AtlasMetrics.CONDITION.value] = "unsmoothed"
        df_io.data_frames_to_csv(
            df_sm, df_smoothing_path, 
            sort_cols=config.SmoothingMetrics.FILTER_SIZE.value)

    print("\nImported {} whole atlas stats:".format(basename))
    df_io.dict_to_data_frame(metrics, df_metrics_path, show=" ")
    
    if show:
        sitk.Show(img_atlas)
        sitk.Show(img_labels)


def measure_overlap(img1, img2, thresh_img1=None, thresh_img2=None,
                    add_to_img1_mask=None, return_masks=False):
    """Measure the Dice Similarity Coefficient (DSC) between two foreground 
    of two images.
    
    Args:
        img1 (:obj:`sitk.Image`): First image.
        img2 (:obj:`sitk.Image`): Image to compare.
        thresh_img1 (float): Threshold to determine the foreground of ``img1``;
            defaults to None to determine by a mean threshold.
        thresh_img2 (float): Threshold to determine the foreground of ``img2``;
            defaults to None to determine by a mean threshold.
        add_to_img1_mask (:obj:`np.ndarray`): Boolean mask to add foreground
            mask of ``img1``; defaults to None. Useful to treat as foreground
            regions that would be thresholded as background but included in
            labels.
        return_masks (bool): True to return the thresholded, mask images
            used for the overlap calculate; defaults to False.
    
    Returns:
        float: The DSC of the foreground of the two given images. If
        ``return_masks`` is True, also returns the masked images as
        :obj:`np.ndarray` arrays.
    """
    # upper threshold does not seem be set with max despite docs for 
    # sitk.BinaryThreshold, so need to set with max explicitly
    img1_np = sitk.GetArrayFromImage(img1)
    thresh_img1_up = float(np.amax(img1_np))
    img2_np = sitk.GetArrayFromImage(img2)
    thresh_img2_up = float(np.amax(img2_np))
    
    # use threshold mean if lower thresholds not given
    if not thresh_img1 or thresh_img1 > thresh_img1_up:
        thresh_img1 = float(filters.threshold_mean(img1_np))
    if not thresh_img2 or thresh_img2 > thresh_img2_up:
        thresh_img2 = float(filters.threshold_mean(img2_np))
    print("Measuring overlap with thresholds of {} (img1) and {} (img2)"
          .format(thresh_img1, thresh_img2))
    
    # similar to simple binary thresholding via Numpy
    binary_img1 = sitk.BinaryThreshold(
        img1, thresh_img1, thresh_img1_up)
    binary_img1_np = sitk.GetArrayFromImage(binary_img1)
    if add_to_img1_mask is not None:
        # add mask to foreground of img1
        binary_img1_np[add_to_img1_mask] = True
        binary_img1 = sitk_io.replace_sitk_with_numpy(
            binary_img1, binary_img1_np)
    binary_img2 = sitk.BinaryThreshold(
        img2, thresh_img2, thresh_img2_up)

    # match world info in case of slight rounding that might prevent the
    # filter from executing
    sitk_io.match_world_info(binary_img1, binary_img2)
    overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_filter.Execute(binary_img1, binary_img2)
    total_dsc = overlap_filter.GetDiceCoefficient()
    print("foreground DSC: {}\n".format(total_dsc))
    if return_masks:
        return total_dsc, binary_img1_np, sitk.GetArrayFromImage(binary_img2)
    return total_dsc


def measure_overlap_labels(labels_img1, labels_img2):
    """Measure the mean Dice Similarity Coefficient (DSC) between two 
    labeled images.
    
    Args:
        labels_img1 (:obj:`sitk.Image`): Labels image 1.
        labels_img2 (:obj:`sitk.Image`): Labels image 2.
    
    Returns:
        float: The mean label-by-label DSC of the two given images.
    """
    overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_filter.Execute(labels_img1, labels_img2)
    mean_region_dsc = overlap_filter.GetDiceCoefficient()
    print("Mean label-by-label DSC: {}".format(mean_region_dsc))
    return mean_region_dsc


def make_labels_fg(labels_sitk):
    """Make a labels foreground image.
    
    Args:
        labels_sitk (:obj:`sitk.Image`): Labels image where
            0 = background, and all other values are considered foreground.
    
    Returns:
        :obj:`sitk.Image`: Labels foreground image.
    """
    fg_img = sitk.GetArrayFromImage(labels_sitk)
    fg_img[fg_img != 0] = 1
    fg_img_sitk = sitk_io.replace_sitk_with_numpy(labels_sitk, fg_img)
    return fg_img_sitk


def measure_overlap_combined_labels(base_img, labels_img, add_lbls=None,
                                    return_masks=False):
    """Measures the overlap of a combined labels images to another image

    Should be 1.0 by definition when using ``labels_img`` to curate another
    image.

    Args:
        base_img (:obj:`sitk.Image`): Base image.
        labels_img (:obj:`sitk.Image`): Labels image, where all non-zero
            values will be treated as foreground.
        add_lbls (List[int]): Sequence of labels in ``labels_img`` to add
            to the mask for ``base_img``; defaults to None.
        return_masks (bool): True to return the thresholded, mask images
            used for the overlap calculate; defaults to False.

    Returns:
        float: Dice Similarity Coefficient from :meth:`measure_overlap`.

    """
    lbls_fg = make_labels_fg(labels_img)
    mask = None
    if add_lbls is not None:
        # build mask from labels to add to fixed image's foreground, such 
        # as labeled ventricles; TODO: get children of labels rather than 
        # taking labels range, but would need to load labels reference; 
        # TODO: consider using "atlas_threshold_all" profile setting 
        # instead, but would need to ensure that fixed thresholds work 
        # for both atlas and sample histology
        labels_np_abs = np.absolute(sitk.GetArrayFromImage(labels_img))
        mask = np.zeros_like(labels_np_abs, dtype=bool)
        for lbl in add_lbls:
            print("adding abs labels within", lbl)
            mask[np.all([labels_np_abs >= lbl[0], 
                         labels_np_abs < lbl[1]], axis=0)] = True
    print("DSC of thresholded fixed image compared with combined labels:")
    return measure_overlap(
        base_img, lbls_fg, thresh_img2=1, add_to_img1_mask=mask,
        return_masks=return_masks)
