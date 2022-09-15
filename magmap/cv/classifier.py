"""Blob classifier."""
from typing import Tuple, Sequence

import numpy as np

from magmap.cv import detector
from magmap.plot import plot_3d
from magmap.settings import config

_logger = config.logger.getChild(__name__)


def extract_patches(roi: np.ndarray, blobs: np.ndarray, size: int = 16):
    """Extract image patches for blobs.
    
    Patches are 2D, centered on each blob but offset by one pixel in width
    and height for even-numbered patch dimensions.
    
    Args:
        roi: Image region of interst as a 3/4D array (``z, y, x, [c]``).
        blobs: 2D blobs array with each blob as a row in ``z, y, x, ...``.
        size: Patch size as an int for both width and height; defaults to 16.

    Returns:

    """
    # px backward from blob center
    size_back = size // 2
    # px forward
    size_fwd = -(size // -2)
    
    patches = []
    for blob in blobs[:, :3].astype(int):
        # extract 2D patch around blob
        blob = blob
        z = blob[0]
        y = blob[1]
        x = blob[2]
        patch = roi[z, y - size_back:y + size_fwd,
                    x - size_back:x + size_back, ...]
        
        # normalize patch to itself
        patch = patch / np.max(patch)
        patches.append(patch)
    
    # combine patches and add a channel axis
    x = np.stack(patches)
    shape = list(x.shape)
    shape.append(1)
    x = x.reshape(shape)
    
    return x


def classify_patches(model, x: np.ndarray, thresh: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Classify patches with a model.
    
    Args:
        model: Keras model.
        x: 2D array of image patches, each in ``y, x, c`` format.
        thresh: Score threshold to classify as 1, otherwise 0. Defaults to 0.5.

    Returns:
        Tuple of:
        - ``y_pred``: Integer array of class predictions.
        - ``y_score``: Float array of raw prediction scores.

    """
    # calculate prediction scores and assign predictions based on threshold
    y_score = model.predict(x).squeeze()
    y_pred = (y_score > thresh).astype(int).squeeze()
    
    return y_pred, y_score


def classify_blobs(
        path: str, image5d: np.ndarray, subimg_offset: Sequence[int],
        subimg_size: Sequence[int], channels: Sequence[int],
        blobs: "detector.Blobs", patch_size: int = 16):
    """Classify blobs based on surrounding image patches.
    
    Args:
        path: Path to Keras model.
        image5d: Image in ``t, z, y, x, [c]`` order.
        subimg_offset: Subimage offset in ``z, y, x``.
        subimg_size: Subimage size in ``z, y, x``.
        channels: Sequence of channels in ``image5d``.
        blobs: Blobs instance.
        patch_size: Patch size as an int for both width and height; defaults
            to 16.

    """
    
    # reduce subimage size if it would exceed image boundaries
    img_shape = image5d.shape[1:4]
    border_far_roi = np.add(subimg_offset, subimg_size)
    border_far_roi = np.where(
        np.greater_equal(border_far_roi, img_shape), img_shape, border_far_roi)
    subimg_size = border_far_roi - subimg_offset
    
    # initialize size of ROI border along each axis and offset
    border = (0, patch_size // 2, patch_size // 2)
    border_offset = np.subtract(subimg_offset, border)

    print("img_shape", img_shape)
    print("subimg_offset", subimg_offset)
    print("subimg_size", subimg_size)
    print("border", border)
    print("border_offset", border_offset)
    
    # set ROI bounding box by defining opposite corners, preventing them from
    # exceeding image boundaries
    border_near = np.where(border_offset < 0, 0, border_offset)
    border_far_full = border_far_roi + border
    border_far = np.where(
        border_far_full > img_shape, img_shape, border_far_full)
    roi_class = plot_3d.prepare_subimg(
        image5d, border_near, np.subtract(border_far, border_near))

    print("border_near", border_near)
    print("border_far_full", border_far_full)
    print("border_far", border_far)
    print("roi_class", roi_class.shape)
    
    # blobs ROI defaults to ROI without border, but any ROI border truncation
    # is converted to blob ROI padding to make up for the border loss
    blobs_near = np.where(border_offset < 0, -border_offset, subimg_offset)
    blobs_far = np.where(
        border_far_full > img_shape,
        np.multiply(img_shape, 2) - border_far_full, border_far_roi)
    
    # convert blob offsets to positions relative to ROI without border
    blobs_rel_offset = np.subtract(blobs_near, subimg_offset)
    blobs_size = np.subtract(blobs_far, blobs_near)
    blobs_shift = np.subtract(subimg_offset, border_near)
    blobs_roi, blobs_roi_mask = detector.get_blobs_in_roi(
        blobs.blobs, blobs_rel_offset, blobs_size, reverse=False)

    print("blobs_near", blobs_near)
    print("blobs_far", blobs_far)
    print("blobs_rel_offset", blobs_rel_offset)
    print("blobs_size", blobs_size)
    print("blobs_shift", blobs_shift)

    # load model with Keras
    from tensorflow.keras.models import load_model
    model = load_model(path)

    for chl in channels:
        _logger.debug("Classifying blobs in channel: %s", chl)
        
        # get blobs in channel for mask and to ensure blobs are present
        blobs_chl, blobs_chl_mask = blobs.blobs_in_channel(
            blobs.blobs, chl, True)
        if len(blobs_chl) < 1: continue
        
        # combine masks to set blobs array view without intermediate copy
        blobs_mask = np.logical_and(blobs_roi_mask, blobs_chl_mask)
        blobs_chl = np.add(blobs.blobs[blobs_mask, :3], blobs_shift)
        
        # classify blobs based on surrounding image patches
        roi_chl = (roi_class if roi_class.ndim < 4
                   else roi_class[..., chl])
        patches = extract_patches(roi_chl, blobs_chl, patch_size)
        y_pred, y_score = classify_patches(model, patches)
        blobs.set_blob_confirmed(blobs.blobs, y_pred, mask=blobs_mask)
        # blobs.set_blob_truth(blobs.blobs, y_score)
