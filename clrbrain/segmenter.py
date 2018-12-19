# Segmentation methods
# Author: David Young, 2018
"""Segment regions based on blobs, labels, and underlying features.
"""

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import feature
from skimage import filters
from skimage import segmentation
from skimage import measure
from skimage import morphology

from clrbrain import config
from clrbrain import detector
from clrbrain import lib_clrbrain
from clrbrain import plot_3d

def _markers_from_blobs(roi, blobs):
    # use blobs as seeds by converting blobs into marker image
    markers = np.zeros(roi.shape, dtype=np.uint8)
    coords = lib_clrbrain.coords_for_indexing(blobs[:, :3].astype(int))
    markers[tuple(coords)] = 1
    markers = morphology.dilation(markers, morphology.ball(1))
    markers = measure.label(markers)
    return markers

def _carve_segs(roi, blobs):
    # carve out background from segmented area
    carved = roi
    if blobs is None:
        # clean up by using simple threshold to remove all background
        carved, _ = plot_3d.carve(carved)
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
        
        # perform the segmentation
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
    multichannel, channels = plot_3d.setup_channels(roi, channel, 3)
    for i in channels:
        roi_segment = roi[..., i] if multichannel else roi
        if thresholded is None:
            # Ostu thresholing and object separate based on local max 
            # rather than seeded watershed approach
            roi_thresh = filters.threshold_otsu(roi, 64)
            thresholded = roi > roi_thresh
        else:
            # r-w assigned 0 values to > 0 val labels
            thresholded = thresholded[0] - 1
        
        # distance transform to find boundaries in thresholded image
        distance = ndimage.distance_transform_edt(thresholded)
        
        if blobs is None:
            # default to finding peaks of distance transform if no blobs 
            # given, using an anisotropic footprint
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
        labels_ws = morphology.watershed(-distance, markers, compactness=0.1)
        
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
        props = plot_3d.get_label_props(labels_img, label_id)
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

def labels_to_markers_erosion(labels_img, filter_size=8):
    """Convert a labels image to markers as eroded labels.
    
    These markers can be used in segmentation algorithms such as 
    watershed.
    
    Args:
        labels_img: Labels image as an integer Numpy array, where each 
            unique int is a separate label.
        filter_size: Size of structing element for erosion; defaults to 8.
    
    Returns:
        Image array of the same shape as ``img`` and the same number of 
        labels as in ``labels_img``, with eroded labels.
    """
    markers = np.zeros_like(labels_img)
    labels_unique = np.unique(labels_img)
    #labels_unique = np.concatenate((labels_unique[:5], labels_unique[-5:]))
    sizes_dict = {}
    cols = ("region", "size_orig", "size_marker", "filter_size")
    for label_id in labels_unique:
        if label_id == 0: continue
        print("eroding label ID {}".format(label_id))
        
        # get region as mask
        bbox = plot_3d.get_label_bbox(labels_img, label_id)
        if bbox is None: continue
        _, slices = plot_3d.get_bbox_region(bbox)
        region = labels_img[tuple(slices)]
        label_mask_region = region == label_id
        region_size = np.sum(label_mask_region)
        region_size_filtered = region_size
        
        # erode the labels, starting with the given filter size and 
        # decreasing if the resulting label size falls below a given 
        # threshold
        for selem_size in range(filter_size, -1, -1):
            if selem_size == 0:
                if size_ratio < 0.01:
                    print("could not erode without losing region, skipping")
                    filtered = label_mask_region
                break
            elif selem_size < filter_size:
                print("size ratio of {} after erosion filter size of {}, "
                      "will reduce filter size".format(size_ratio, selem_size))
            # erode check size ratio
            filtered = morphology.binary_erosion(
                label_mask_region, morphology.ball(selem_size))
            region_size_filtered = np.sum(filtered)
            size_ratio = region_size_filtered / region_size
            if size_ratio > 0.3: break
        
        # insert eroded region into markers image
        markers[tuple(slices)][filtered] = label_id
        print("changed num of pixels from {} to {}"
              .format(region_size, np.sum(filtered)))
        vals = (label_id, region_size, region_size_filtered, selem_size)
        for col, val in zip(cols, vals):
            sizes_dict.setdefault(col, []).append(val)
    
    df_sizes = pd.DataFrame(sizes_dict)
    print(df_sizes.to_csv(sep="\t", index=False))
    return markers

def mask_atlas(atlas, labels_img):
    """Mask an atlas with its thresholded image filled in with an 
    associated labels image.
    
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

def segment_from_labels(atlas_img, edges, labels_img, markers):
    """Segment an image using markers from a labels image.
    
    Labels images may have been generally manually and thus may not 
    perfectly match the underlying image. As a way to check or 
    augment the label image, segment the underlying image using 
    the labels as the seeds to prescribe the number and initial 
    location of each label.
    
    Args:
        atlas_img: Atlas image as a Numpy array to use for finding foreground.
        edges: Image as a Numpy array to segment, typically an edge-detected 
            image of the main atlas.
        labels_img: Labels image as Numpy array of same shape as ``img``, 
            used to generate a mask of the interior of ``img`` in which 
            to segment. If None, a mask be be generated from a thresholded 
            version of ``atlas_img``.
        markers: Image as an integer Numpy array of same shape as ``img`` 
            to use as seeds for the watershed segmentation. This array 
            is generally constructed from an array similar to ``labels_img``.
    
    Returns:
        Segmented image of the same shape as ``img`` with the same 
        number of labels as in ``markers``.
    """
    distance = ndimage.distance_transform_edt(edges == 0)
    watershed = morphology.watershed(-distance, markers, compactness=0.01)
    if labels_img is None:
        # otsu seems to give more inclusive threshold for these atlases
        _, mask = plot_3d.carve(
            atlas_img, thresh=filters.threshold_otsu(atlas_img), 
            holes_area=5000)
    else:
        # use labels if available; ideally should fully cover the atlas 
        # with the atlas only used to identify outside borders
        mask = mask_atlas(atlas_img, labels_img)
    watershed[~mask] = 0
    return watershed
