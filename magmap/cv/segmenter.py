# Segmentation methods
# Author: David Young, 2018, 2019
"""Segment regions based on blobs, labels, and underlying features.
"""

from multiprocessing import sharedctypes
from time import time
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage
from skimage import feature
from skimage import filters
from skimage import segmentation
from skimage import measure
from skimage import morphology

from magmap.settings import config
from magmap.cv import chunking, cv_nd, detector
from magmap.io import libmag
from magmap.plot import plot_3d
from magmap.io import df_io


def _markers_from_blobs(roi, blobs):
    # use blobs as seeds by converting blobs into marker image
    markers = np.zeros(roi.shape, dtype=np.uint8)
    coords = libmag.coords_for_indexing(blobs[:, :3].astype(int))
    markers[tuple(coords)] = 1
    markers = morphology.dilation(markers, morphology.ball(1))
    markers = measure.label(markers)
    return markers


def _carve_segs(roi, blobs):
    # carve out background from segmented area
    carved = roi
    if blobs is None:
        # clean up by using simple threshold to remove all background
        carved, _ = cv_nd.carve(carved)
    else:
        # use blobs as ellipsoids to identify background to remove; 
        # TODO: consider setting spacing in config since depends on 
        # microscopy characteristics, such as elongation from 
        # thick lightsheet
        thresholded = plot_3d.build_ground_truth(
            np.zeros(carved.shape, dtype=bool), blobs, ellipsoid=True)
        #thresholded = thresholded.astype(bool)
        carved[~thresholded] = 0
    return carved


def segment_rw(roi, channel, beta=50.0, vmin=0.6, vmax=0.65, remove_small=None, 
               erosion=None, blobs=None, get_labels=False):
    """Segments an image using the Random-Walker algorithm.
    
    Args:
        roi: Region of interest to segment.
        channel: Channel to pass to :func:``plot_3d.setup_channels``.
        beta: Random-Walker beta term.
        vmin: Values under which to exclude in markers; defaults to 0.6. 
            Ignored if ``blobs`` is given.
        vmax: Values above which to exclude in markers; defaults to 0.65. 
            Ignored if ``blobs`` is given.
        remove_small: Threshold size of small objects to remove; defaults 
            to None to ignore.
        erosion: Structuring element size for erosion; defaults 
            to None to ignore.
        blobs: Blobs to use for markers; defaults to None, in which 
            case markers will be determined based on ``vmin``/``vmax`` 
            thresholds.
        get_labels: True to measure and return labels from the 
            resulting segmentation instead of returning the segmentations 
            themselves; defaults to False.
    
    Returns:
        List of the Random-Walker segmentations for the given channels, 
        If ``get_labels`` is True, the measured labels for the segmented 
        regions will be returned instead of the segmentations themselves.
    """
    print("Random-Walker based segmentation...")
    labels = []
    walkers = []
    multichannel, channels = plot_3d.setup_channels(roi, channel, 3)
    for i in channels:
        roi_segment = roi[..., i] if multichannel else roi
        if blobs is None:
            # mark unknown pixels as 0 by distinguishing known background 
            # and foreground
            markers = np.zeros(roi_segment.shape, dtype=np.uint8)
            markers[roi_segment < vmin] = 2
            markers[roi_segment >= vmax] = 1
        else:
            # derive markers from blobs
            markers = _markers_from_blobs(roi_segment, blobs)
        
        # perform the segmentation; conjugate gradient with multigrid
        # preconditioner option (cg_mg), which is faster but req pyamg
        walker = segmentation.random_walker(
            roi_segment, markers, beta=beta, mode="cg_mg")
        
        # clean up segmentation
        
        #lib_clrbrain.show_full_arrays()
        walker = _carve_segs(walker, blobs)
        if remove_small:
            # remove artifacts
            walker = morphology.remove_small_objects(walker, remove_small)
        if erosion:
            # attempt to reduce label connections by eroding
            walker = morphology.erosion(walker, morphology.octahedron(erosion))
        
        if get_labels:
            # label neighboring pixels to segmented regions
            # TODO: check if necessary; useful only if blobs not given?
            label = measure.label(walker, background=0)
            labels.append(label)
            #print("label:\n", label)
        
        walkers.append(walker)
        #print("walker:\n", walker)
    
    if get_labels:
        return labels
    return walkers


def segment_ws(roi, channel, thresholded=None, blobs=None): 
    """Segment an image using a compact watershed, including the option 
    to use a 3D-seeded watershed approach.
    
    Args:
        roi: ROI as a Numpy array in (z, y, x) order.
        channel: Channel to pass to :func:``plot_3d.setup_channels``.
        thresholded: Thresholded image such as a segmentation into foreground/
            background given by Random-walker (:func:``segment_rw``). 
            Defaults to None, in which case Otsu thresholding will be performed.
        blobs: Blobs as a Numpy array in [[z, y, x, ...], ...] order, which 
            are used as seeds for the watershed. Defaults to None, in which 
            case peaks on a distance transform will be used.
    
    Returns:
        List of watershed labels for each given channel, with each set 
        of labels given as an image of the same shape as ``roi``.
    """
    labels = []
    labels_ws = None
    multichannel, channels = plot_3d.setup_channels(roi, channel, 3)
    for i in channels:
        roi_segment = roi[..., i] if multichannel else roi
        if thresholded is None:
            # Ostu thresholing and object separate based on local max 
            # rather than seeded watershed approach
            roi_thresh = filters.threshold_otsu(roi, 64)
            thresholded = roi_segment > roi_thresh
        else:
            # r-w assigned 0 values to > 0 val labels
            thresholded = thresholded[0] - 1
        
        if blobs is None:
            # default to finding peaks of distance transform if no blobs 
            # given, using an anisotropic footprint
            distance = ndimage.distance_transform_edt(thresholded)
            try:
                local_max = feature.peak_local_max(
                    distance, indices=False, footprint=np.ones((1, 3, 3)), 
                    labels=thresholded)
            except IndexError as e:
                print(e)
                raise e
            markers = measure.label(local_max)
        else:
            markers = _markers_from_blobs(thresholded, blobs)
        
        # watershed with slight increase in compactness to give basins with 
        # more regular, larger shape
        labels_ws = watershed_distance(thresholded, markers, compactness=0.1)
        
        # clean up segmentation
        labels_ws = _carve_segs(labels_ws, blobs)
        labels_ws = morphology.remove_small_objects(labels_ws, min_size=100)
        #print("num ws blobs: {}".format(len(np.unique(labels_ws)) - 1))
        labels_ws = labels_ws[None]
        labels.append(labels_ws)
    return labels_ws


def labels_to_markers_blob(labels_img):
    """Convert a labels image to markers as blobs.
    
    These markers can be used in segmentation algorithms such as 
    watershed.
    
    Args:
        labels_img: Labels image as an integer Numpy array, where each 
            unique int is a separate label.
    
    Returns:
        Image array of the same shape as ``img`` and the same number of 
        labels as in ``labels_img``, with labels reduced to smaller 
        markers.
    """
    blobs = {}
    labels_unique = np.unique(labels_img)
    #labels_unique = np.concatenate((labels_unique[:5], labels_unique[-5:]))
    for label_id in labels_unique:
        if label_id == 0: continue
        print("finding centroid for label ID {}".format(label_id))
        props = cv_nd.get_label_props(labels_img, label_id)
        if len(props) >= 1: 
            # get centroid and convert to ellipsoid marker
            blob = [int(n) for n in props[0].centroid]
            blob.append(5)
            blobs[label_id] = np.array(blob)
            print("storing centroid as {}".format(blobs[label_id]))
    
    # build markers from centroids
    spacing = detector.calc_scaling_factor()
    spacing = spacing / np.amin(spacing)
    markers = plot_3d.build_ground_truth(
        np.zeros_like(labels_img), np.array(list(blobs.values())), 
        ellipsoid=True, labels=list(blobs.keys()), spacing=spacing)
    return markers


class LabelToMarkerErosion(object):
    """Convert a label to an eroded marker for multiprocessing
    
    Uses class methods as an encapsulated way to use in forked multiprocessing
    without requirement for global variables. In non-forked multiprocessing
    (eg "spawn" on Windows), regions and weights should be pickled directly.
    
    Attributes:
        labels_img: Integer labels images as a Numpy array.
        wt_dists: Array of distances by which to weight the filter size.
        labels_img_shared: ``labels_img`` as a shared array.
        labels_img_shape: Shape of ``labels_img_shared``.
        labels_img_dtype: Data type of ``labels_img_shared``.
    """
    labels_img: np.ndarray = None
    wt_dists: np.ndarray = None
    
    labels_img_shared: sharedctypes.RawArray = None
    labels_img_shape: Tuple = None
    labels_img_dtype: np.dtype = None
    
    @classmethod
    def setup_labels_img_shared(cls, img, shape, dtype):
        """Set up shared labels image for reconstructing as ndarray.
        
        Args:
            img: Labels image as a shared array.
            shape: Shape of the image.
            dtype: Data type of the image.

        """
        cls.labels_img_shared = img
        cls.labels_img_shape = shape
        cls.labels_img_dtype = dtype
    
    @classmethod
    def set_labels_img(cls, labels_img: np.ndarray, wt_dists: np.ndarray):
        """Set the labels image.
        
        Args:
            labels_img: Labels image to set as class attribute.
            wt_dists: Distance weights image to set as class attribute.
        """
        cls.labels_img = labels_img
        cls.wt_dists = wt_dists
    
    @classmethod
    def meas_wt(
            cls, labels_img: np.ndarray, label_id: int, wt_dists: np.ndarray
    ) -> float:
        """Measure the weight for a label based on weighted distances.
        
        Args:
            labels_img: Labels image.
            label_id: Label ID.
            wt_dists: Array of distances by which to weight the filter size.

        Returns:
            Normalized weight for ``label_id``.

        """
        return np.median(wt_dists[labels_img == label_id]) / np.amax(wt_dists)
    
    @classmethod
    def erode_label(
            cls, label_id: int, filter_size: int, target_frac: float = None,
            min_filter_size: int = 1, use_min_filter: bool = False,
            skel_eros_filt_size: Union[int, bool] = False,
            wt: float = None) -> Tuple[
                Tuple[int, np.ndarray, np.ndarray, Any],
                Union[Optional[List[slice]], Any], Any]:
        """Convert a label to a marker as an eroded version of the label.
        
        By default, labels will be eroded with the given ``filter_size`` 
        as long as their final size is > 20% of the original volume. If 
        the eroded volume is below threshold, ``filter_size`` will be 
        progressively decreased until the filter cannot be reduced further.
        
        Skeletonization of the labels recovers some details by partially
        preserving the original labels' extent, including thin regions that
        would be eroded away, thus serving a similar function as that of
        adaptive morphological filtering. ``skel_eros_filt_size`` allows
        titrating the amount of the labels` extent to be preserved.
        
        If :attr:`wt_dists` is present, the label's distance will be used
        to weight the starting filter size.
        
        Args:
            label_id: ID of label to erode.
            filter_size: Size of structing element to start erosion.
            target_frac: Target fraction of original label to erode. 
                Erosion will start with ``filter_size`` and use progressively
                smaller filters until remaining above this target. Defaults
                to None to use a fraction of 0.2. Titrates the relative
                amount of erosion allowed.
            min_filter_size: Minimum filter size, below which the
                original, uneroded label will be used instead. Defaults to 1.
                Use 0 to erode at size 1 even if below ``target_frac``.
                Titrates the absolute amount of erosion allowed.
            use_min_filter: True to erode at ``min_filter_size`` if
                a smaller filter size would otherwise be required; defaults
                to False to revert to original, uneroded size if a filter
                smaller than ``min_filter_size`` would be needed.
            skel_eros_filt_size: Erosion filter size before
                skeletonization to balance how much of the labels' extent will
                be preserved during skeletonization. Increase to reduce the
                skeletonization. Defaults to 0, which will cause
                skeletonization to be skipped.
            wt: Multiplier weight for ``filter_size``. Defaults to None, in
                which case the weighte will be calculated from
                :attr:``wt_dists`` if available, or ignored if not.
        
        Returns:
            Tuple of stats,including ``label_id`` for reference and 
            sizes of labels; list of slices denoting where to insert 
            the eroded label; and the eroded label itself.
        
        Raises:
            ValueError: if ``region`` is None and :attr:`labels_img` is not
                available.
        
        """
        if cls.labels_img is None:
            # convert shared raw array to Numpy array for labels image
            cls.labels_img = np.frombuffer(
                cls.labels_img_shared, cls.labels_img_dtype).reshape(
                cls.labels_img_shape)

        if (wt is None and cls.wt_dists is not None and
                cls.labels_img is not None):
            # weight the filter size by the fractional distance from median
            # of label distance and max dist
            wt = cls.meas_wt(cls.labels_img, label_id, cls.wt_dists)
        if wt is not None:
            filter_size = int(filter_size * wt)
            print("label {}: distance weight {}, adjusted filter size to {}"
                  .format(label_id, wt, filter_size))
            if use_min_filter and filter_size < min_filter_size:
                filter_size = min_filter_size
        
        # get region as mask; assume that label exists and will yield a 
        # bounding box since labels here are generally derived from the 
        # labels image itself
        region, slices = cv_nd.extract_region(cls.labels_img, label_id)
        label_mask_region = region == label_id
        region_size = np.sum(label_mask_region)
        region_size_filtered = region_size
        fn_selem = cv_nd.get_selem(cls.labels_img.ndim)
        
        # erode the labels, starting with the given filter size and decreasing
        # if the resulting label size falls below a given size ratio
        chosen_selem_size = np.nan
        filtered = label_mask_region
        size_ratio = 1
        for selem_size in range(filter_size, -1, -1):
            if selem_size < min_filter_size:
                if not use_min_filter:
                    print("label {}: could not erode without dropping below "
                          "minimum filter size of {}, reverting to original "
                          "region size of {}"
                          .format(label_id, min_filter_size, region_size))
                    filtered = label_mask_region
                    region_size_filtered = region_size
                    chosen_selem_size = np.nan
                break
            # erode check size ratio
            filtered = morphology.binary_erosion(
                label_mask_region, fn_selem(selem_size))
            region_size_filtered = np.sum(filtered)
            size_ratio = region_size_filtered / region_size
            thresh = 0.2 if target_frac is None else target_frac
            chosen_selem_size = selem_size
            if region_size_filtered < region_size and size_ratio > thresh:
                # stop eroding if underwent some erosion but stayed above
                # threshold size; skimage erosion treats border outside image
                # as True, so images may not undergo erosion and should
                # continue until lowest filter size is taken (eg NaN)
                break

        if not np.isnan(chosen_selem_size):
            print("label {}: changed num of pixels from {} to {} "
                  "(size ratio {}), initial filter size {}, chosen {}"
                  .format(label_id, region_size, region_size_filtered, 
                          size_ratio, filter_size, chosen_selem_size))

        if skel_eros_filt_size and np.sum(filtered) > 0:
            # skeletonize the labels to recover details from erosion;
            # need another labels erosion before skeletonization to avoid
            # preserving too much of the original labels' extent
            label_mask_region = morphology.binary_erosion(
                label_mask_region, fn_selem(skel_eros_filt_size))
            filtered = np.logical_or(
                filtered, 
                morphology.skeletonize_3d(label_mask_region).astype(bool))
        
        stats_eros = (label_id, region_size, region_size_filtered,
                      chosen_selem_size)
        return stats_eros, slices, filtered


def _init_labels_to_markers(*args):
    """Initialize labels to markers class attributes in spawned multiprocessing.
    """
    LabelToMarkerErosion.setup_labels_img_shared(*args)


def labels_to_markers_erosion(labels_img, filter_size=8, target_frac=None,
                              min_filter_size=None, use_min_filter=False, 
                              skel_eros_filt_size=None, wt_dists=None):
    """Convert a labels image to markers as eroded labels via multiprocessing.
    
    These markers can be used in segmentation algorithms such as 
    watershed.
    
    Args:
        labels_img (:obj:`np.ndarray`): Labels image as an integer Numpy array,
            where each unique int is a separate label.
        filter_size (int): Size of structing element for erosion, which should
            be > 0; defaults to 8.
        target_frac (float): Target fraction of original label to erode,
            passed to :func:`LabelToMarkerErosion.erode_label`. Defaults
            to None.
        min_filter_size (int): Minimum erosion filter size; defaults to None
            to use half of ``filter_size``, rounded down.
        use_min_filter (bool): True to erode even if ``min_filter_size``
            is reached; defaults to False to avoid any erosion if this size
            is reached.
        skel_eros_filt_size (int): Erosion filter size before skeletonization
            in :func:`LabelToMarkerErosion.erode_labels`. Defaults to None to
            use the minimum filter size, which is half of ``filter_size``.
        wt_dists (:obj:`np.ndarray`): Array of distances by which to weight
            the filter size, such as a distance transform to the outer
            perimeter of ``labels_img`` to weight central labels more
            heavily. Defaults to None.
    
    Returns:
        :obj:`np.ndarray`: Image array of the same shape as ``img`` and the
        same number of labels as in ``labels_img``, with eroded labels.
    """
    start_time = time()
    markers = np.zeros_like(labels_img)
    labels_unique = np.unique(labels_img)
    if min_filter_size is None:
        min_filter_size = filter_size // 2
    if skel_eros_filt_size is None:
        skel_eros_filt_size = filter_size // 2
    #labels_unique = np.concatenate((labels_unique[:5], labels_unique[-5:]))
    sizes_dict = {}
    cols = (config.AtlasMetrics.REGION.value, "SizeOrig", "SizeMarker",
            config.SmoothingMetrics.FILTER_SIZE.value)
    
    # erode labels via multiprocessing
    print("Eroding labels to markers with filter size {}, min filter size {}, "
          "and target fraction {}"
          .format(filter_size, min_filter_size, target_frac))
    is_fork = chunking.is_fork()
    initializer = None
    initargs = None
    if is_fork:
        # share large images as class attributes in forked mode
        LabelToMarkerErosion.set_labels_img(labels_img, wt_dists)
    else:
        # set up labels image as a shared array for spawned mode
        initializer = _init_labels_to_markers
        initargs = (
            sharedctypes.RawArray(np.ctypeslib.as_ctypes_type(
                labels_img.dtype), labels_img.flatten()),
            labels_img.shape,
            labels_img.dtype,
        )
    
    pool = chunking.get_mp_pool(initializer, initargs)
    pool_results = []
    for label_id in labels_unique:
        if label_id == 0: continue
        # erode labels to generate markers, excluding labels small enough
        # that they would require a filter smaller than half of original size
        args = [label_id, filter_size, target_frac, min_filter_size,
                use_min_filter, skel_eros_filt_size]
        if not is_fork:
            # pickle distance weight directly in spawned mode
            if wt_dists is not None:
                args.append(LabelToMarkerErosion.meas_wt(
                    labels_img, label_id, wt_dists))
        pool_results.append(
            pool.apply_async(LabelToMarkerErosion.erode_label, args=args))
    
    for result in pool_results:
        stats_eros, slices, filtered = result.get()
        # can only mutate markers outside of mp for changes to persist
        markers[tuple(slices)][filtered] = stats_eros[0]
        for col, stat in zip(cols, stats_eros):
            sizes_dict.setdefault(col, []).append(stat)
    pool.close()
    pool.join()
    
    # show erosion stats
    df = df_io.dict_to_data_frame(sizes_dict, show=True)
    
    print("time elapsed to erode labels into markers:", time() - start_time)
    return markers, df


def mask_atlas(atlas, labels_img):
    """Generate a mask of an atlas by combining its thresholded image 
    with its associated labels image.
    
    The labels image may be insufficient to find the whole atlas foreground 
    if the labels have missing regions or around edges, while the 
    thresholded atlas may have many holes. As a simple workaround, 
    combine these foregrounds to obtain a more complete mask of the atlas.
    
    Args:
        img: Image as a Numpy array to segment.
        labels_img: Labels image of the same shape as ``img``, where all 
            values except 0 will be taken as an additional 
            part of the resulting mask.
    Returns:
        Boolean array the same shape as ``img`` with True for all 
        pixels above threshold in ``img`` or within the 
        foreground of ``labels_img``.
    """
    thresh = filters.threshold_otsu(atlas)
    mask = np.logical_or(atlas > thresh, labels_img != 0)
    return mask


def segment_from_labels(edges, markers, labels_img, atlas_img=None,
                        exclude_labels=None,
                        mask_filt=config.SmoothingModes.opening,
                        mask_filt_size=2):
    """Segment an image using markers from a labels image.
    
    Labels images may have been generally manually and thus may not 
    perfectly match the underlying image. As a way to check or 
    augment the label image, segment the underlying image using 
    the labels as the seeds to prescribe the number and initial 
    location of each label.
    
    Args:
        edges (:obj:`np.ndarray`): Image as a Numpy array to segment,
            typically an edge-detected image of the main atlas.
        markers (:obj:`np.ndarray`): Image as an integer Numpy array of same
            shape as ``img`` to use as seeds for the watershed segmentation.
            This array is generally constructed from an array similar to
            ``labels_img``.
        labels_img (:obj:`np.ndarray`): Labels image as Numpy array of same
            shape as ``img``, used to generate a mask for the watershed.
            If None, a mask will be generated from a thresholded version of
            ``atlas_img``, so should only be None if ``atlas_img`` is not None. 
        atlas_img (:obj:`np.ndarray`): Atlas image as a Numpy array to use
            for finding foreground; defaults to None. If both ``labels_img``
            and ``atlas_img`` are not None, their combined volume will be
            used as a mask.
        exclude_labels (List[int]): Sequence of labels to exclude from the
            segmentation; defaults to None.
        mask_filt (:obj:`config.SmoothingModes`): Enumeration for a filter
            mode to use for the watershed mask; defaults to
            :obj:`config.SmoothingModes.opening`. Ignored if ``atlas_img``
            or both ``atlas_img`` and ``labels_img`` are given to generate
            the mask.
        mask_filt_size (int): Size of structuring element for the filter
            specified by ``mask_filt``; defaults to 2.
    
    Returns:
        :obj:`np.ndarray`: Segmented image of the same shape as ``img`` with
        the same number of labels as in ``markers``.
    
    """
    # generate mask for watershed
    if atlas_img is not None and labels_img is not None:
        # broad mask from both atlas and labels
        mask = mask_atlas(atlas_img, labels_img)
    elif atlas_img is not None:
        # otsu seems to give more inclusive threshold for these atlases
        _, mask = cv_nd.carve(
            atlas_img, thresh=filters.threshold_otsu(atlas_img), 
            holes_area=5000)
    else:
        # default to using label foreground
        mask = labels_img != 0
        fn_mask = None
        if mask_filt is config.SmoothingModes.opening:
            # default filter opens the mask to prevent spillover across
            # artifacts that may bridge otherwise separate structures
            fn_mask = morphology.binary_opening
        elif mask_filt is config.SmoothingModes.closing:
            fn_mask = morphology.binary_closing
        if fn_mask and mask_filt_size:
            print("Filtering watershed mask with {}, size {}"
                  .format(fn_mask, mask_filt_size))
            mask = fn_mask(mask, cv_nd.get_selem(labels_img.ndim)(
                mask_filt_size))
    
    exclude = None
    if exclude_labels is not None:
        # remove excluded labels from mask
        exclude = np.isin(labels_img, exclude_labels)
        mask[exclude] = False
        # WORKAROUND: remove excluded markers from marker image itself for
        # apparent Scikit-image bug (see PR 3809, fixed in 0.15)
        markers[np.isin(markers, exclude_labels)] = 0
    
    watershed = watershed_distance(
        edges == 0, markers, compactness=0.005, mask=mask)
    if exclude is not None:
        # add excluded labels directly to watershed image
        watershed[exclude] = labels_img[exclude]
    return watershed


def watershed_distance(foreground, markers=None, num_peaks=np.inf, 
                       compactness=0, mask=None):
    """Perform watershed segmentation based on distance from foreground 
    to background.
    
    Args:
        foreground: Boolean array where True represents foreground. The 
            distances will be measured from foreground to the 
            nearest background.
        markers: Array of same size as ``foreground`` with seeds to 
            use for the watershed. Defaults to None, in which case 
            markers will be generated from local peaks in the 
            distance transform.
        num_peaks: Number of peaks to include when generating markers; 
            defaults to infinity.
        compactness (float): Compactness factor for watershed; defaults to 0.
        mask: Boolean or binary array of same size as ``foreground`` 
            where True or 1 pixels will be filled by the watershed; 
            defaults to None to fill the whole image.
    
    Returns:
        The segmented image as an array of the same shape as that of 
        ``foreground``.
    """
    distance = ndimage.distance_transform_edt(foreground)
    if markers is None:
        # generate a limited number of markers from local peaks in the 
        # distance transform if markers are not given
        local_max = feature.peak_local_max(
            distance, indices=False, num_peaks=num_peaks)
        markers = measure.label(local_max)
    watershed = morphology.watershed(
        -distance, markers, compactness=compactness, mask=mask)
    return watershed


class SubSegmenter(object):
    """Sub-segment a label based on anatomical boundaries.
    
    All images should be of the same shape.
    
    Attributes:
        labels_img_np: Integer labels image as a Numpy array.
        atlas_edge: Numpy array of atlas reduced to binary image of its edges.
    """
    labels_img_np = None
    atlas_edge = None
    
    @classmethod
    def set_images(cls, labels_img_np, atlas_edge):
        """Set the images."""
        cls.labels_img_np = labels_img_np
        cls.atlas_edge = atlas_edge
    
    @classmethod
    def sub_segment(cls, label_id, dtype):
        """Calculate metrics for a given label or set of labels.
        
        Wrapper to call :func:``measure_variation`` and 
        :func:``measure_edge_dist``.
        
        Args:
            label_id: Integer of the label in :attr:``labels_img_np`` 
                to sub-divide.
        
        Returns:
            Tuple of the given label ID, list of slices where the label 
            resides in :attr:``labels_img_np``, and an array in the 
            same shape of the original label, now sub-segmented. The base  
            value of this sub-segmented array is multiplied by 
            :const:``config.SUB_SEG_MULT``, with each sub-region 
            incremented by 1.
        """
        label_mask = cls.labels_img_np == label_id
        label_size = np.sum(label_mask)
        
        labels_seg = None
        slices = None
        if label_size > 0:
            props = measure.regionprops(label_mask.astype(np.int))
            _, slices = cv_nd.get_bbox_region(props[0].bbox)
            
            # work on a view of the region for efficiency
            labels_region = np.copy(cls.labels_img_np[tuple(slices)])
            label_mask_region = labels_region == label_id
            atlas_edge_region = cls.atlas_edge[tuple(slices)]
            #labels_region[atlas_edge_region != 0] = 0
            labels_region[~label_mask_region] = 0
            
            # segment from anatomic borders, limiting peaks to get only 
            # dominant regions
            labels_seg = watershed_distance(
                atlas_edge_region == 0, num_peaks=5, compactness=0.01)
            labels_seg[~label_mask_region] = 0
            #labels_seg = measure.label(labels_region)
            
            # ensure that sub-segments occupy at least a certain 
            # percentage of the total label
            labels_retained = np.zeros_like(labels_region, dtype=dtype)
            labels_unique = np.unique(labels_seg[labels_seg != 0])
            print("found {} subregions for label ID {}"
                  .format(labels_unique.size, label_id))
            i = 0
            for seg_id in labels_unique:
                seg_mask = labels_seg == seg_id
                size = np.sum(seg_mask)
                ratio = size / label_size
                if ratio > 0.1:
                    # relabel based on original label, expanded to 
                    # allow for sub-labels
                    unique_id = np.abs(label_id) * config.SUB_SEG_MULT + i
                    unique_id = int(unique_id * label_id / np.abs(label_id))
                    print("keeping subregion {} of size {} (ratio {}) within "
                          "label {}".format(unique_id, size, ratio, label_id))
                    labels_retained[seg_mask] = unique_id
                    i += 1
            
            retained_unique = np.unique(labels_retained[labels_retained != 0])
            print("labels retained within {}: {}"
                  .format(label_id, retained_unique))
            '''
            # find neighboring sub-labels to merge into retained labels
            neighbor_added = True
            done = []
            while len(done) < retained_unique.size:
                for seg_id in retained_unique:
                    if seg_id in done: continue
                    neighbor_added = False
                    seg_mask = labels_retained == seg_id
                    exterior = plot_3d.exterior_nd(seg_mask)
                    neighbors = np.unique(labels_seg[exterior])
                    for neighbor in neighbors:
                        mask = np.logical_and(
                            labels_seg == neighbor, labels_retained == 0)
                        if neighbor == 0 or np.sum(mask) == 0: continue
                        print("merging in neighbor {} (size {}) to label {}"
                              .format(neighbor, np.sum(mask), seg_id))
                        labels_retained[mask] = seg_id
                        neighbor_added = True
                    if not neighbor_added:
                        print("{} is done".format(seg_id))
                        done.append(seg_id)
                print(done, retained_unique)
            labels_seg = labels_retained
            '''
            if retained_unique.size > 0:
                # in-paint missing space from non-retained sub-labels
                labels_seg = cv_nd.in_paint(
                    labels_retained, labels_retained == 0)
                labels_seg[~label_mask_region] = 0
            else:
                # if no sub-labels retained, replace whole region with 
                # new label
                labels_seg[label_mask_region] = label_id * config.SUB_SEG_MULT
        
        return label_id, slices, labels_seg


def sub_segment_labels(labels_img_np, atlas_edge):
    """Sub-segment a labels image into sub-labels based on anatomical 
    boundaries.
    
    Args:
        labels_img_np: Integer labels image as a Numpy array.
        atlas_edge: Numpy array of atlas reduced to binary image of its edges.
    
    Returns:
        Image as a Numpy array of same shape as ``labels_img_np`` with 
        each label sub-segmented based on anatomical boundaries. Labels 
        in this image will correspond to the original labels 
        multiplied by :const:``config.SUB_SEG_MULT`` to make room for 
        sub-labels, which will each be incremented by 1.
    """
    start_time = time()
    
    # use a class to set and process the label without having to 
    # reference the labels image as a global variable
    SubSegmenter.set_images(labels_img_np, atlas_edge)
    
    pool = chunking.get_mp_pool()
    pool_results = []
    label_ids = np.unique(labels_img_np)
    max_val = np.amax(labels_img_np) * (config.SUB_SEG_MULT + 1)
    dtype = libmag.dtype_within_range(-max_val, max_val, True)
    subseg = np.zeros_like(labels_img_np, dtype=dtype)
    
    for label_id in label_ids:
        # skip background
        if label_id == 0: continue
        pool_results.append(
            pool.apply_async(
                SubSegmenter.sub_segment, args=(label_id, dtype)))
    
    for result in pool_results:
        label_id, slices, labels_seg = result.get()
        # can only mutate markers outside of mp for changes to persist
        labels_seg_mask = labels_seg != 0
        subseg[tuple(slices)][labels_seg_mask] = labels_seg[labels_seg_mask]
        print("finished sub-segmenting label ID {}".format(label_id))
    pool.close()
    pool.join()
    
    print("time elapsed to sub-segment labels image:", time() - start_time)
    return subseg
