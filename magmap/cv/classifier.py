"""Blob classifier."""
from enum import Enum, auto
from time import time
from typing import Tuple, Sequence, Optional

import numpy as np

from magmap.cv import chunking, detector
from magmap.io import cli
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


def setup_classification_roi(
        image5d: np.ndarray, subimg_offset: Sequence[int],
        subimg_size: Sequence[int],
        blobs: "detector.Blobs", patch_size: int, blobs_relative: bool = False
) -> Tuple[np.ndarray, np.ndarray, Sequence[int]]:
    """Set up ROI for blob classification.
    
    Sets up an image ROI with a border to provide uniformly sized image
    patches around blobs, including blobs on the ROI edge. If the border
    would exceed an edge of the image, a padding area is created where
    blobs are excluded so that edge blobs can still have the full-sized
    patches.
    
    Args:
        image5d: 4/5D array as ``t, z, y, x[, c]``.
        subimg_offset: Subimage offset in ``z, y, x``.
        subimg_size: Subimage size in ``z, y, x``.
        blobs: Blobs instance.
        patch_size: Patch size as an int for both width and height.
        blobs_relative: True if ``blobs`` coordinates are relative; defaults
            to False.
    
    Returns:
        Tuple of:
        - ``roi``: region of interest as ``z, y, x, [c]``
        - ``blobs_roi_mask``: mask for ``blobs`` in the ROI.
        - ``blobs_shift``: Offset of blobs relative to ``subimg_offset``.

    """
    
    # reduce subimage size if it would exceed image boundaries
    img_shape = image5d.shape[1:4]
    border_far_roi = np.add(subimg_offset, subimg_size)
    border_far_roi = np.where(
        np.greater_equal(border_far_roi, img_shape), img_shape, border_far_roi)
    subimg_size = border_far_roi - subimg_offset
    border_far_roi = np.add(subimg_offset, subimg_size)
    
    # initialize size of ROI border along each axis and offset
    border = (0, patch_size // 2, patch_size // 2)
    border_offset = np.subtract(subimg_offset, border)
    
    # set ROI bounding box by defining opposite corners, preventing them from
    # exceeding image boundaries
    border_near = np.where(border_offset < 0, 0, border_offset)
    border_far_full = border_far_roi + border
    border_far = np.where(
        border_far_full > img_shape, img_shape, border_far_full)
    roi = plot_3d.prepare_subimg(
        image5d, border_near, np.subtract(border_far, border_near))
    
    # blobs ROI defaults to ROI without border, but any ROI border truncation
    # is converted to blob ROI padding to make up for the border loss
    blobs_near = np.where(border_offset < 0, -border_offset, subimg_offset)
    blobs_far = np.where(
        border_far_full > img_shape,
        np.multiply(img_shape, 2) - border_far_full, border_far_roi)
    
    # convert blob offsets to positions relative to ROI without border
    blobs_rel_offset = np.subtract(
        blobs_near, subimg_offset) if blobs_relative else blobs_near
    blobs_size = np.subtract(blobs_far, blobs_near)
    blobs_shift = np.subtract(subimg_offset, border_near)
    blobs_roi, blobs_roi_mask = detector.get_blobs_in_roi(
        blobs.blobs, blobs_rel_offset, blobs_size, reverse=False)
    
    return roi, blobs_roi_mask, blobs_shift


def classify_blobs(
        path: str, image5d: np.ndarray, subimg_offset: Sequence[int],
        subimg_size: Sequence[int], channels: Sequence[int],
        blobs: "detector.Blobs", patch_size: int = 16, blobs_relative=False
) -> Tuple[np.ndarray, Sequence[int]]:
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
        blobs_relative: True to treat blob coordinates as relative to
            ``subimg_offset``; defaults to False.
        
    Returns:
        Tuple of:
        - ``blobs_mask``: row mask for blobs in sub-image.
        - ``classifications``: corresponding blob classifications.
    
    Raises:
        `ModuleNotFoundError` if Tensorflow is not installed.

    """
    
    # set up image and blobs ROIs
    _logger.info("Setting up classification ROI")
    roi_class, blobs_roi_mask, blobs_shift = setup_classification_roi(
        image5d, subimg_offset, subimg_size, blobs, patch_size, blobs_relative)
    
    # load model with Keras
    try:
        from tensorflow.keras.models import load_model
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Tensorflow is required for classification. Please install, eg "
            "with 'pip install tensorflow'.")
    model = load_model(path)

    for chl in channels:
        _logger.info("Classifying blobs in channel: %s", chl)
        
        # get blobs in channel for mask and to ensure blobs are present
        blobs_chl, blobs_chl_mask = blobs.blobs_in_channel(
            blobs.blobs, chl, True)
        
        # combine masks to set blobs array view without intermediate copy
        blobs_mask = np.logical_and(blobs_roi_mask, blobs_chl_mask)
        if np.sum(blobs_mask) < 1: continue
        
        # shift blob coordinates by border or padding
        blobs_chl = np.add(blobs.blobs[blobs_mask, :3], blobs_shift)
        if not blobs_relative:
            # convert absolute to relative coordinates
            blobs_chl -= subimg_offset
        
        # classify blobs based on surrounding image patches
        roi_chl = (roi_class if roi_class.ndim < 4
                   else roi_class[..., chl])
        patches = extract_patches(roi_chl, blobs_chl, patch_size)
        y_pred, y_score = classify_patches(model, patches)
        blobs.set_blob_confirmed(blobs.blobs, y_pred, mask=blobs_mask)
        # blobs.set_blob_truth(blobs.blobs, y_score)
    
    classifications = blobs.get_blob_confirmed(blobs.blobs)[blobs_roi_mask]
    return blobs_roi_mask, classifications


class ClassifyImage:
    """Convert a label to an edge with class methods as an encapsulated 
    way to use in multiprocessing without requirement for global variables.

    """
    image5d = None
    blobs = None
    
    @classmethod
    def classify_whole_image(
            cls, model_path: Optional[str] = None,
            image5d: Optional[np.ndarray] = None,
            channels: Optional[Sequence[int]] = None,
            blobs: Optional["detector.Blobs"] = None, **kwargs):
        """Classify blobs in the whole image through multiprocessing
        
        Args:
            model_path: Path to Keras model. Defaults to None, in which case
                :attr:`magmap.settings.config.classifier` is accessed.
            image5d: Image in ``t, z, y, x, [c]`` order. Defaults to None, in which
                case :attr:`magmap.settings.config.img5d.img` is accessed.
            channels: Sequence of channels in ``image5d``. Defaults to None, in
                which case :attr:`magmap.settings.config.chanel` is accessed.
            blobs: Blobs instance. Defaults to None, in which case
                :attr:`magmap.settings.config.blobs` is accessed.
            kwargs: Additional arguments to :meth:`classify_blobs`.
        
        Raises:
            `FileNotFoundError` if a classifier model, image, or blobs are not
            found.
    
        """
        
        if not model_path:
            # get model path from config
            model_path = config.classifier.model
            if not model_path:
                raise FileNotFoundError("No classifier model found")
        
        if image5d is None and config.img5d is not None:
            # get main image from config
            image5d = config.img5d.img
            if image5d is None:
                raise FileNotFoundError("No image found")
        
        if channels is None:
            # set up channels from config
            channels = plot_3d.setup_channels(
                config.image5d, config.channel, 4)[1]
        
        if blobs is None:
            # get blobs from config
            blobs = config.blobs
            if blobs is None:
                raise FileNotFoundError("No blobs found")

        is_fork = chunking.is_fork()
        if is_fork:
            # share large objects as class variables in forked processes
            cls.image5d = image5d
            cls.blobs = blobs
        
        # set up multiprocessing
        start_time = time()
        pool = chunking.get_mp_pool()
        pool_results = []
        
        # chunk image simply by planes
        step = 100
        img_shape = image5d.shape[1:4]
        size = list(img_shape)
        size[0] = step
        nsteps = -(img_shape[0] // -step)
        
        _logger.info("Classifying whole image in %s steps", nsteps)
        for i in range(nsteps):
            # increment offset z-plane by step size
            offset = (i * step, 0, 0)
            pool_results.append(
                pool.apply_async(
                    cls.classify_chunk,
                    args=(model_path, offset, size, channels), kwds=kwargs))
        
        for result in pool_results:
            offset, blobs_mask, predictions = result.get()
            _logger.info(
                "Classified blobs at offset %s of %s", offset, img_shape)
            
            # update blobs since modifications in separate processes are lost
            blobs.set_blob_confirmed(blobs.blobs, predictions, blobs_mask)
            print(blobs.blobs[blobs_mask][:5])
        pool.close()
        pool.join()

        _logger.debug(
            "Time elapsed to classify whole image: %s", time() - start_time)
    
    @classmethod
    def classify_chunk(
            cls, model_path: str, subimg_offset: Sequence[int],
            subimg_size: Sequence[int], channels: Sequence[int], **kwargs):
        """Classify blobs in an image chunk.
        
        Args:
            model_path: Path to Keras model
            subimg_offset: Subimage offset in ``z, y, x``.
            subimg_size: Subimage size in ``z, y, x``.
            channels: Sequence of channels in :attr:`image5d`.
            kwargs: Additional arguments to :meth:`classify_blobs`.
        
        Returns:
            Tuple of:
            - ``subimg_offset``: the ``subimg_offset`` argument to track
              during multiprocessing.
            - ``blobs_mask``: row mask for blobs in sub-image.
            - ``classifications``: corresponding blob classifications.
    
        """
        
        if cls.image5d is None:
            # reopen main image in each spawned process
            cli.process_cli_args()
            cli.setup_image(config.filename)
            cls.image5d = config.img5d.img
            cls.blobs = config.blobs
        
        # classify blobs using model
        blobs_mask, classifications = classify_blobs(
            model_path, cls.image5d, subimg_offset, subimg_size, channels,
            cls.blobs, **kwargs)
        
        return subimg_offset, blobs_mask, classifications
        
