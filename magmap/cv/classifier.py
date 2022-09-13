"""Blob classifier."""
from typing import Tuple

import numpy as np
from tensorflow.keras.models import load_model

from magmap.cv import detector


def extract_patches(roi: np.ndarray, blobs: "detector.Blobs", size: int = 16):
    """Extract image patches for blobs.
    
    Patches are 2D, centered on each blob but offset by one pixel in width
    and height for even-numbered patch dimensions.
    
    Args:
        roi: Image region of interst as a 3/4D array (``z, y, x, [c]``).
        blobs: Blobs instance.
        size: Patch size as an int for both width and height; defaults to 16.

    Returns:

    """
    # px backward from blob center
    size_back = size // 2
    # px forward
    size_fwd = -(size // -2)
    
    # get blob relative coordinates as ints and shift to account for image
    # padding
    blobs_shifted = blobs.blobs[:, :3].astype(int)
    blobs_shifted += size_back
    
    patches = []
    for blob in blobs_shifted:
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


def classify(
        path: str, x: np.ndarray, thresh: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Classify patches with a model.
    
    Args:
        path: Path to model.
        x: 2D array of image patches, each in ``y, x, c`` format.
        thresh: Score threshold to classify as 1, otherwise 0. Defaults to 0.5

    Returns:
        Tuple of:
        - ``y_pred``: Integer array of class predictions.
        - ``y_score``: Float array of raw prediction scores.

    """
    # load model with Keras
    model = load_model(path)
    
    # calculate prediction scores and assign predictions based on threshold
    y_score = model.predict(x).squeeze()
    y_pred = (y_score > thresh).astype(int).squeeze()
    
    return y_pred, y_score
