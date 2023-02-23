# Cell detection methods
# Author: David Young, 2017, 2020
"""Detects features within a 3D image stack.

Prunes duplicates and verifies detections against truth sets.

Attributes:
    CONFIRMATION: Dictionary of blob confirmation flags.
    OVERLAP_FACTOR: Pixel number multiplier for overlaps between adjacent ROIs.

"""

from enum import Enum
import math
import pprint
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from skimage.feature import blob_log

from magmap.cv import colocalizer, cv_nd
from magmap.io import libmag, np_io
from magmap.plot import plot_3d
from magmap.settings import config

# blob confirmation flags
CONFIRMATION: Dict[int, str] = {
    -1: "unverified",
    0: "no",
    1: "yes",
    2: "maybe"
}

# pixel number multiplier by scaling for max overlapping pixels per ROI
OVERLAP_FACTOR: int = 5

_logger = config.logger.getChild(__name__)


class Blobs:
    """Blob storage class.
    
    Attributes:
        blobs: 2D Numpy array of blobs in the format
            ``[[z, y, x, radius, ...], ...]``; defaults to None.
        blob_matches: Sequence of blob matches; defaults to None.
        colocalizations: 2D Numpy array of same length
            as ``blobs`` with a column for each channel, where 0 = no
            signal and 1 = signal at the corresponding blob's location
            in the given and channel; defaults to None.
        path: Path from which blobs were loaded; defaults to None.
        ver: Version number; defaults to :const:`BLOBS_NP_VER`.
        roi_offset: Offset in ``z,y,x`` from ROI in which blobs were detected.
        roi_size: Size in ``z,y,x`` from ROI in which blobs were detected.
        resolutions: Physical unit resolution sizes in ``[[z,y,x], ...]``;
            defaults to None.
        basename: Archive name, typically the filename without extension
            of the image file in which blobs were detected; defaults to None.
        scaling: Scaling factor from the blobs' space to the main image's
            space, in ``z,y,x``; defaults to ``[1, 1, 1]``.
    
    """

    #: Current blobs Numpy archive version number.
    #: 0: initial version
    #: 1: added resolutions, basename, offset, roi_size fields
    #: 2: added archive version number
    #: 3: added colocs
    #: 4: added "columns" for column names
    BLOBS_NP_VER: int = 4
    
    class Keys(Enum):
        """Numpy archive metadata keys as enumerations."""
        VER = "ver"
        BLOBS = "segments"
        COLOCS = "colocs"
        RESOLUTIONS = "resolutions"
        BASENAME = "basename"
        ROI_OFFSET = "offset"
        ROI_SIZE = "roi_size"
        COLS = "columns"
    
    class Cols(Enum):
        """Blob column names."""
        Z = "z"
        Y = "y"
        X = "x"
        RADIUS = "radius"
        CONFIRMED = "confirmed"
        TRUTH = "truth"
        CHANNEL = "channel"
        ABS_Z = "abs_z"
        ABS_Y = "abs_y"
        ABS_X = "abs_x"
    
    #: Dictionary of column types to column indices in :attr:`blobs`.
    col_inds: Dict["Cols", int] = {c: i for i, c in enumerate(Cols)}
    
    def __init__(
            self,
            blobs: Optional[np.ndarray] = None,
            blob_matches: Optional["colocalizer.BlobMatch"] = None,
            colocalizations: Optional[np.ndarray] = None,
            path: str = None):
        """Initialize blobs storage object."""
        self.blobs = blobs
        self.blob_matches = blob_matches
        self.colocalizations = colocalizations
        self.path = path
        
        # additional attributes
        self.ver: int = self.BLOBS_NP_VER
        self.roi_offset: Optional[Sequence[int]] = None
        self.roi_size: Optional[Sequence[int]] = None
        self.resolutions: Optional[Sequence[float]] = None
        self.basename: Optional[str] = None
        self.scaling: np.ndarray = np.ones(3)
        
        #: Blob columns loaded from metadata. Defaults to the first 6 columns
        #: in :class:`Cols`.
        self.cols: Sequence[str] = [c.value for c in self.Cols][:6]

    def load_blobs(self, path: str = None) -> "Blobs":
        """Load blobs from an archive.

        Also loads associated metadata from the archive.

        Args:
            path: Path to set :attr:`path`; defaults to None to use
                the existing path.

        Returns:
            Blobs object.

        """
        # load blobs and display counts
        if path is not None:
            self.path = path
        _logger.info("Loading blobs from: %s", self.path)
        
        with np.load(self.path) as archive:
            info = np_io.read_np_archive(archive)
            
            if config.verbose:
                _logger.debug(
                    "Blobs archive metadata:\n%s", pprint.pformat(info))

            if self.Keys.VER.value in info:
                # load archive version number
                self.basename = info[self.Keys.VER.value]
            
            if self.Keys.COLS.value in info:
                # load column indices into class attribute
                # TODO: convert to instance attribute after moving all blob
                # functions into this class
                self.cols = info[self.Keys.COLS.value]
                Blobs.col_inds = {c: None for c in self.Cols}
                for i, col in enumerate(self.cols):
                    try:
                        Blobs.col_inds[self.Cols(col)] = i
                    except ValueError:
                        _logger.warn(
                            "%s is not a valid Blobs column, skipping", col)
                _logger.debug(
                    "Loaded column indices:\n%s",
                    pprint.pformat(Blobs.col_inds))

            if self.Keys.BLOBS.value in info:
                # load blobs as a Numpy array
                self.blobs = info[self.Keys.BLOBS.value]
                _logger.info("Loaded %s blobs", len(self.blobs))
                if config.verbose:
                    self.show_blobs_per_channel(self.blobs)
            
            if self.Keys.COLOCS.value in info:
                # load intensity-based colocalizations
                self.colocalizations = info[self.Keys.COLOCS.value]
                if self.colocalizations is not None:
                    _logger.info(
                        "Loaded blob co-localizations for %s channels",
                        self.colocalizations.shape[1])
            
            if self.Keys.RESOLUTIONS.value in info:
                # load resolutions of image from which blobs were detected
                self.resolutions = info[self.Keys.RESOLUTIONS.value]

            if self.Keys.BASENAME.value in info:
                # load basename of image file from which blobs were detected
                self.basename = info[self.Keys.BASENAME.value]

            if self.Keys.ROI_OFFSET.value in info:
                # load offset of ROI from which blobs were detected
                self.roi_offset = info[self.Keys.ROI_OFFSET.value]
            
            if self.Keys.ROI_SIZE.value in info:
                # load size of ROI from which blobs were detected
                self.roi_size = info[self.Keys.ROI_SIZE.value]
            
        return self

    def save_archive(self, to_add=None, update=False):
        """Save the blobs Numpy archive file to :attr:`path`.
        
        Backs up any existing file before saving.
        
        Args:
            to_add (dict): Dictionary of items to add; defaults to None
                to use the current attributes.
            update (bool): True to load the Numpy archive at :attr:`path`
                and update it.

        Returns:
            dict: Dictionary saved to :attr:`path`.

        """
        if to_add is None:
            # save current attributes
            
            # filter out column indices that are None
            col_inds = {k: v for k, v in self.col_inds.items() if v is not None}
            
            blobs_arc = {
                Blobs.Keys.VER.value: self.ver,
                Blobs.Keys.BLOBS.value: self.blobs,
                Blobs.Keys.RESOLUTIONS.value: self.resolutions,
                Blobs.Keys.BASENAME.value: self.basename,
                Blobs.Keys.ROI_OFFSET.value: self.roi_offset,
                Blobs.Keys.ROI_SIZE.value: self.roi_size,
                Blobs.Keys.COLOCS.value: self.colocalizations,
                
                # save columns ordered by the col indices
                Blobs.Keys.COLS.value: [
                    k.value for k, v in sorted(
                        col_inds.items(), key=lambda e: e[1])],
            }
        else:
            blobs_arc = to_add
        
        if update:
            with np.load(self.path) as archive:
                # load archive, convert to dict, and update dict
                blobs_arc = np_io.read_np_archive(archive)
                blobs_arc.update(to_add)
        
        # back up any existing file
        libmag.backup_file(self.path)
        
        with open(self.path, "wb") as archive:
            # save as uncompressed zip Numpy archive file
            np.savez(archive, **blobs_arc)
            _logger.info("Saved blobs archive to: %s", self.path)
        
        if config.verbose:
            pprint.pprint(blobs_arc)
        return blobs_arc
    
    @classmethod
    def format_blobs(
            cls, blobs: np.ndarray,
            channel: Optional[Union[int, Sequence[int]]] = None) -> np.ndarray:
        """Format blobs with the full set of fields.
         

        Blobs in MagellanMapper can be assumed to start with ``z, y, x, radius``
        but should use this class's functions to manipulate other fields to 
        ensure that the correct columns are accessed. This function adds
        these fields: ``confirmation, truth, channel, abs z, abs y, abs x``.

        "Confirmed" is given as -1 = unconfirmed, 0 = incorrect, 1 = correct.

        "Truth" is given as -1 = not truth, 0 = not matched, 1 = matched, where 
        a "matched" truth blob is one that has a detected blob within a given 
        tolerance.

        Args:
            blobs: Numpy 2D array in ``[[z, y, x, radius, ...], ...]`` format.
            channel: Channel to set. Defaults to None, in which case the channel 
                will not be updated.

        Returns:
            Blobs array formatted as 
            ``[[z, y, x, radius, confirmation, truth, channel, 
              abs_z, abs_y, abs_x], ...]``.
        
        """
        # target num of cols minus current cols
        shape = blobs.shape
        extra_cols = len(cls.Cols) - shape[1]
        extras = np.ones((shape[0], extra_cols)) * -1
        blobs = np.concatenate((blobs, extras), axis=1)
        
        # map added col names to indices, assumed to be ordered as in Cols
        for i, col in enumerate(cls.Cols):
            if i < shape[1]: continue
            Blobs.col_inds[cls.Cols(col)] = i
        
        # copy relative to absolute coords
        blobs[:, cls._get_abs_inds()] = blobs[:, cls._get_rel_inds()]
        
        if channel is not None:
            # update channel
            cls.set_blob_channel(blobs, channel)
        
        return blobs

    @classmethod
    def get_blob_col(
            cls, blob: np.ndarray, col: Union[int, Sequence[int]]
    ) -> Union[int, float, np.ndarray]:
        """Get the value for the given column of a blob or blobs.

        Args:
            blob: 1D blob array or 2D array of blobs.
            col: Column index in ``blob``.

        Returns:
            Single value if ``blob`` is a single blob or array of values if
            it is an array of blobs.

        """
        is_multi_d = blob.ndim > 1
        if col is None:
            # no column indicates that this column has not been set up for the
            # blob, so return None or an empty ndarray
            return np.array([]) if is_multi_d else None
        
        if is_multi_d:
            return blob[..., col]
        return blob[col]
    
    @classmethod
    def get_blob_confirmed(
            cls, blob: np.ndarray) -> Union[int, float, np.ndarray]:
        """Get the confirmed value for blob or blobs.

        Args:
            blob: 1D blob array or 2D array of blobs.

        Returns:
            Single value if ``blob`` is a single blob or array of values if
            it is an array of blobs.

        """
        return cls.get_blob_col(blob, cls.col_inds[cls.Cols.CONFIRMED])
    
    @classmethod
    def get_blob_truth(cls, blob: np.ndarray) -> Union[int, float, np.ndarray]:
        """Get the truth value for blob or blobs.

        Args:
            blob: 1D blob array or 2D array of blobs.

        Returns:
            Single value if ``blob`` is a single blob or array of values if
            it is an array of blobs.

        """
        return cls.get_blob_col(blob, cls.col_inds[cls.Cols.TRUTH])

    @classmethod
    def get_blobs_channel(cls, blob: np.ndarray) -> np.ndarray:
        """Get the channel value for blob or blobs.
        
        Args:
            blob: 1D blob array or 2D array of blobs.

        Returns:
            Single value if ``blob`` is a single blob or array of values if
            it is an array of blobs.

        """
        # get the channel index from the class attribute of indices
        return cls.get_blob_col(blob, cls.col_inds[cls.Cols.CHANNEL])
    
    @classmethod
    def _get_rel_inds(cls) -> List[int]:
        """Get relative coordinate indices."""
        return [
            cls.col_inds[cls.Cols.Z],
            cls.col_inds[cls.Cols.Y],
            cls.col_inds[cls.Cols.X]
        ]

    @classmethod
    def _get_abs_inds(cls) -> List[int]:
        """Get absolute coordinate indices."""
        return [
            cls.col_inds[cls.Cols.ABS_Z],
            cls.col_inds[cls.Cols.ABS_Y],
            cls.col_inds[cls.Cols.ABS_X]
        ]
    
    @classmethod
    def get_blob_abs_coords(cls, blobs: np.ndarray) -> np.ndarray:
        """Get blob absolute coordinates.
        
        Args:
            blobs: 1D blob array or 2D array of blobs.

        Returns:
            Array of absolute coordinates, with dimensions corresponding to
            ``blobs``.

        """
        return cls.get_blob_col(blobs, cls._get_abs_inds())

    @classmethod
    def set_blob_col(
            cls, blob: np.ndarray, col: Union[int, Sequence[int]],
            val: Union[float, Sequence[float]],
            mask: Union[
                np.ndarray, np.lib.index_tricks.IndexExpression] = np.s_[:], **kwargs
    ) -> np.ndarray:
        """Set the value for the given column of a blob or blobs.

        Args:
            blob: 1D blob array or 2D array of blobs.
            col: Column index in ``blob``.
            val: New value. If ``blob`` is 2D, can be an array the length
                of ``blob``.
            mask: Mask for the first axis; defaults to an index expression
                for all values. Only used for multidimensional arrays.

        Returns:
            ``blob`` after modifications.

        """
        if blob.ndim > 1:
            # set value for col in last axis, applying mask for first axis
            blob[mask, ..., col] = val
        else:
            blob[col] = val
        return blob
    
    @classmethod
    def set_blob_confirmed(
            cls, blob: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Set the confirmed flag of a blob or blobs.

        Args:
            blob: 1D blob array or 2D array of blobs.
            args: Positional arguments passed to :meth:`set_blob_col`.
            kwargs: Named arguments passed to :meth:`set_blob_col`.

        Returns:
            ``blob`` after modifications.

        """
        return cls.set_blob_col(
            blob, cls.col_inds[cls.Cols.CONFIRMED], *args, **kwargs)
    
    @classmethod
    def set_blob_truth(
            cls, blob: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Set the truth flag of a blob or blobs.

        Args:
            blob: 1D blob array or 2D array of blobs.
            args: Positional arguments passed to :meth:`set_blob_col`.
            kwargs: Named arguments passed to :meth:`set_blob_col`.

        Returns:
            ``blob`` after modifications.

        """
        return cls.set_blob_col(
            blob, cls.col_inds[cls.Cols.TRUTH], *args, **kwargs)
    
    @classmethod
    def set_blob_channel(
            cls, blob: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Set the channel of a blob or blobs.

        Args:
            blob: 1D blob array or 2D array of blobs.
            args: Positional arguments passed to :meth:`set_blob_col`.
            kwargs: Named arguments passed to :meth:`set_blob_col`.

        Returns:
            ``blob`` after modifications.

        """
        return cls.set_blob_col(
            blob, cls.col_inds[cls.Cols.CHANNEL], *args, **kwargs)
    
    @classmethod
    def set_blob_abs_coords(
            cls, blobs: np.ndarray, coords: Sequence[int], *args, **kwargs
    ) -> np.ndarray:
        """Set blob absolute coordinates.
        
        Args:
            blobs: 2D array of blobs.
            coords: 2D array of absolute coordinates in the same order of
                coordinates as in ``blobs``.

        Returns:
            Modified ``blobs``.

        """
        cls.set_blob_col(blobs, cls._get_abs_inds(), coords, *args, **kwargs)
        return blobs
    
    @classmethod
    def shift_blobs(
            cls, blob: np.ndarray, cols: Union[int, Sequence[int]],
            fn: Callable,
            vals: Union[int, float, Sequence[Union[int, float]]],
            to_int: bool = False
    ) -> np.ndarray:
        """Shift blob columns with a function.

        Args:
            blob: 1D blob array or 2D array of blobs.
            cols: Column(s) to modify.
            fn: Function that takes a subset of ``blob`` and ``vals``.
            vals: Values by which to shift the corresponding elements of
                ``blob``.
            to_int: True to convert the shifted elements to int; defaults
                to False.

        Returns:
            ``blob`` shifted in-place.

        """
        if blob is not None:
            # shift elements in the given columns with the provided function
            is_1d = blob.ndim == 1
            sub = blob[cols] if is_1d else blob[..., cols]
            sub = fn(sub, vals)
            
            if to_int:
                # convert shifted values to ints
                sub = sub.astype(int)
            
            # replace values
            if is_1d:
                blob[cols] = sub
            else:
                blob[..., cols] = sub
        
        return blob

    @classmethod
    def shift_blob_rel_coords(
            cls, blob: np.ndarray, offset: Sequence[int]) -> np.ndarray:
        """Shift blob relative coordinates by offset.

        Args:
            blob: 1D blob array or 2D array of blobs.
            offset: Sequence of coordinates by which to shift
                the corresponding elements of ``blob``.

        Returns:
            ``blob`` shifted in-place.

        """
        return cls.shift_blobs(blob, cls._get_rel_inds(), np.add, offset)
    
    @classmethod
    def shift_blob_abs_coords(
            cls, blob: np.ndarray, offset: Sequence[int]) -> np.ndarray:
        """Shift blob absolute coordinates by offset.

        Args:
            blob: 1D blob array or 2D array of blobs.
            offset: Sequence of coordinates by which to shift
                the corresponding elements of ``blob``.

        Returns:
            ``blob`` shifted in-place.

        """
        return cls.shift_blobs(blob, cls._get_abs_inds(), np.add, offset)
    
    @classmethod
    def multiply_blob_rel_coords(
            cls, blob: np.ndarray,
            factor: Union[int, float, Sequence[Union[int, float]]]
    ) -> np.ndarray:
        """Multiply blob relative coordinates.

        Args:
            blob: 1D blob array or 2D array of blobs.
            factor: Factor by which to shift the corresponding elements of
                ``blob``.

        Returns:
            ``blob`` shifted in-place.

        """
        return cls.shift_blobs(
            blob, cls._get_rel_inds(), np.multiply, factor, True)
    
    @classmethod
    def multiply_blob_abs_coords(
            cls, blob: np.ndarray, factor: Union[int, float]) -> np.ndarray:
        """Multiply blob absolute coordinates.

        Args:
            blob: 1D blob array or 2D array of blobs.
            factor: Factor by which to shift the corresponding elements of
                ``blob``.

        Returns:
            ``blob`` shifted in-place.

        """
        return cls.shift_blobs(
            blob, cls._get_abs_inds(), np.multiply, factor, True)

    @classmethod
    def remove_abs_blob_coords(
            cls, blobs: np.ndarray, remove_extra: bool = False) -> np.ndarray:
        """Remove blob absolute coordinate columns.
        
        Args:
            blobs: 2D array of blobs.
            remove_extra: True to also remove any extra columns not in
                :attr:`cols_inds`; defaults to False.

        Returns:
            ``blob`` modified in-place.

        """
        inds = cls.col_inds.values() if remove_extra else slice(blobs.shape[1])
        return blobs[:, [i for i in inds if i not in cls._get_abs_inds()]]
    
    @classmethod
    def replace_rel_with_abs_blob_coords(cls, blobs: np.ndarray) -> np.ndarray:
        """Replace relative with absolute coordinates.
        
        Args:
            blobs: 2D array of blobs.

        Returns:
            ``blob`` modified in-place.

        """
        blobs[:, cls._get_rel_inds()] = blobs[:, cls._get_abs_inds()]
        return blobs

    @classmethod
    def blobs_in_channel(
            cls, blobs: np.ndarray, channel: Union[int, np.ndarray],
            return_mask: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Get blobs in the given channels.

        Args:
            blobs: 1D blob array or 2D array of blobs.
            channel: Sequence of channels to include.
            return_mask: True to return the mask of blobs in ``channel``.

        Returns:
            A view of the blobs in the channel, or all  blobs if ``channel``
            is None. If ``return_mask`` is True, also return the mask of
            blobs in the given channels.

        """
        blobs_chl = blobs
        mask = None
        if channel is not None:
            mask = np.isin(cls.get_blobs_channel(blobs), channel)
            blobs_chl = blobs[mask]
        if return_mask:
            return blobs_chl, mask
        return blobs_chl
    
    @classmethod
    def show_blobs_per_channel(cls, blobs: np.ndarray):
        """Show the number of blobs in each channel.

        Args:
            blobs: 1D blob array or 2D array of blobs.
        """
        channels = np.unique(cls.get_blobs_channel(blobs))
        for channel in channels:
            num_blobs = len(cls.blobs_in_channel(blobs, channel))
            _logger.info("- blobs in channel %s: %s", int(channel), num_blobs)
    
    @classmethod
    def blob_for_db(cls, blob: np.ndarray) -> np.ndarray:
        """Convert blob to absolute coordinates.
         
        Changes the blob format from that used within this module 
        to that used in :module:`sqlite`, where coordinates are absolute 
        rather than relative to the offset.

        Args:
            blob: Single blob.

        Returns:
            Blob in ``abs_z, abs_y, abs_x, rad, confirmed, truth, channel``
            format.
        """
        inds = [
            cls.col_inds[cls.Cols.RADIUS],
            cls.col_inds[cls.Cols.CONFIRMED],
            cls.col_inds[cls.Cols.TRUTH],
            cls.col_inds[cls.Cols.CHANNEL],
        ]
        return np.array([*blob[cls._get_abs_inds()], *blob[inds]])


def calc_scaling_factor():
    """Calculates the tolerance based on the resolutions, using the 
    first resolution.
    
    Return:
        Array of tolerance values in same shape as the first resolution.
    
    Raises:
        ``AttributeError`` if the :attr:``config.resolutions`` is None or has 
        less than one element.
    """
    if config.resolutions is None or len(config.resolutions) < 1:
        raise AttributeError(
            "Must load resolutions from file or set a resolution")
    factor = np.divide(1.0, config.resolutions[0])
    return factor


def calc_overlap(factor: Optional[int] = None):
    """Calculate overlap based on scaling factor and a factor.
    
    Args:
        factor: Overlap factor; defaults to None to use :const:``OVERLAP_FACTOR``

    Returns:
        Overlap as an array in the same shape and order as in 
        :attr:``resolutions``.
    
    """
    if factor is None:
        factor = OVERLAP_FACTOR
    return np.ceil(np.multiply(calc_scaling_factor(), factor)).astype(int)


def _blob_surroundings(blob, roi, padding, plane=False):
    rad = blob[3]
    start = np.subtract(blob[0:3], rad + padding).astype(int)
    start[start < 0] = 0
    end = np.add(blob[0:3], rad + padding).astype(int)
    shape = roi.shape
    for i in range(3):
        if end[i] >= shape[i]:
            end[i] = shape[i] - 1
    if plane:
        z = blob[0]
        if z < 0:
            z = 0
        elif z >= shape[0]:
            z = end[0]
        return roi[z, start[1]:end[1], start[2]:end[2]]
    else:
        return roi[start[0]:end[0], start[1]:end[1], start[2]:end[2]]


def show_blob_surroundings(blobs, roi, padding=1):
    print("showing blob surroundings")
    np.set_printoptions(precision=2, linewidth=200)
    for blob in blobs:
        print("{} surroundings:".format(blob))
        surroundings = _blob_surroundings(blob, roi, padding, True)
        print("{}\n".format(surroundings))
    np.set_printoptions()


def detect_blobs(
        roi: np.ndarray, channel: Sequence[int],
        exclude_border: Optional[Sequence[int]] = None) -> Optional[np.ndarray]:
    """Detects objects using 3D blob detection technique.
    
    Args:
        roi: Region of interest to segment.
        channel: Sequence of channels to select, which can
            be None to indicate all channels.
        exclude_border: Sequence of border pixels in x,y,z to exclude;
            defaults to None.
    
    Returns:
        Array of detected blobs, each given as
        ``z, row, column, radius, confirmation``.
    
    """
    shape = roi.shape
    multichannel, channels = plot_3d.setup_channels(roi, channel, 3)
    isotropic = config.get_roi_profile(channels[0])["isotropic"]
    if isotropic is not None:
        # interpolate for (near) isotropy during detection, using only the
        # first process settings since applies to entire ROI
        roi = cv_nd.make_isotropic(roi, isotropic)
    
    blobs_all = []
    for chl in channels:
        roi_detect = roi[..., chl] if multichannel else roi
        settings = config.get_roi_profile(chl)
        # scaling as a factor in pixel/um, where scaling of 1um/pixel
        # corresponds to factor of 1, and 0.25um/pixel corresponds to
        # 1 / 0.25 = 4 pixels/um; currently simplified to be based on
        # x scaling alone
        scale = calc_scaling_factor()
        scaling_factor = scale[2]
        
        # find blobs; sigma factors can be sequences by axes for anisotropic
        # detection in skimage >= 0.15, or images can be interpolated to
        # isotropy using the "isotropic" MagellanMapper setting
        min_sigma = settings["min_sigma_factor"] * scaling_factor
        max_sigma = settings["max_sigma_factor"] * scaling_factor
        num_sigma = settings["num_sigma"]
        threshold = settings["detection_threshold"]
        overlap = settings["overlap"]
        blobs_log = blob_log(
            roi_detect, min_sigma=min_sigma, max_sigma=max_sigma,
            num_sigma=num_sigma, threshold=threshold, overlap=overlap)
        if blobs_log.size < 1:
            _logger.debug("No blobs detected")
            continue
        blobs_log[:, 3] = blobs_log[:, 3] * math.sqrt(3)
        blobs = Blobs.format_blobs(blobs_log, chl)
        #print(blobs)
        blobs_all.append(blobs)
    if not blobs_all:
        return None
    blobs_all = np.vstack(blobs_all)
    if isotropic is not None:
        # if detected on isotropic ROI, need to reposition blob coordinates
        # for original, non-isotropic ROI
        isotropic_factor = cv_nd.calc_isotropic_factor(isotropic)
        blobs_all = Blobs.multiply_blob_rel_coords(
            blobs_all, 1 / isotropic_factor)
        blobs_all = Blobs.multiply_blob_abs_coords(
            blobs_all, 1 / isotropic_factor)
    
    if exclude_border is not None:
        # exclude blobs from the border in x,y,z
        blobs_all = get_blobs_interior(blobs_all, shape, *exclude_border)
    
    return blobs_all


def remove_duplicate_blobs(blobs, region):
    """Removes duplicate blobs.
    
    Args:
        blobs: The blobs, given as 2D array of [n, [z, row, column, radius]].
        region: Slice within each blob to check, such as slice(0, 2) to check
           for (z, row, column).
    
    Return:
        The blobs array with only unique elements.
    """
    # workaround while awaiting https://github.com/numpy/numpy/pull/7742
    # to become a reality, presumably in Numpy 1.13
    blobs_region = blobs[:, region]
    blobs_contig = np.ascontiguousarray(blobs_region)
    blobs_type = np.dtype(
        (np.void, blobs_region.dtype.itemsize * blobs_region.shape[1]))
    blobs_contig = blobs_contig.view(blobs_type)
    _, unique_indices = np.unique(blobs_contig, return_index=True)
    # TODO: should probably only look within blobs_region
    print("removed {} duplicate blobs"
          .format(blobs.shape[0] - unique_indices.size))
    return blobs[unique_indices]


def sort_blobs(blobs):
    """Sort blobs by their coordinates in priority of z,y,x.
    
    Args:
        blobs: Blobs as a 2D array of [n, [z, row, column, ...]].
    
    Returns:
        Tuple of the sorted blobs as a new array and an array of 
        sorting indices.
    """
    sort = np.lexsort(tuple(blobs[:, i] for i in range(2, -1, -1)))
    blobs = blobs[sort]
    return blobs, sort


def _find_close_blobs(blobs, blobs_master, tol):
    # creates a separate array for each blob in blobs_master to allow
    # comparison for each of its blobs with each blob to add
    blobs_diffs = np.abs(blobs_master[:, None, :3] - blobs[:, :3])
    close_master, close = np.nonzero((blobs_diffs <= tol).all(2))
    #print("close:\n{}\nclose_master:\n{}".format(close, close_master))
    return close_master, close


def remove_close_blobs(blobs, blobs_master, tol, chunk_size=1000):
    """Removes blobs that are close to one another.
    
    Args:
        blobs: The blobs to be checked for closeness and pruning, given as 2D 
            array of [n, [z, row, column, ...]].
        blobs_master: The list by which to check for close blobs, in the same
            format as blobs.
        tol: Tolerance to check for closeness, given in the same format
            as region. Blobs that are equal to or less than the the absolute
            difference for all corresponding parameters will be pruned in
            the returned array.
        chunk_size: Max size along first dimension for each blob array 
            to minimize memory consumption; defaults to 1000.
    
    Return:
        Tuple of the blobs array after pruning and ``blobs_master`` with 
        absolute coordinates updated with the average of any 
        corresponding duplicates.
    """
    num_blobs_check = len(blobs)
    num_blobs_master = len(blobs_master)
    if num_blobs_check < 1 or num_blobs_master < 1:
        # no blobs to remove if either array is empty
        return blobs, blobs_master
    
    # smallest type to hold blob coordinates, signed to use for diffs
    dtype = libmag.dtype_within_range(
        0, np.amax((np.amax(blobs[:, :3]), np.amax(blobs_master[:, :3]))), 
        True, True)
    match_check = None
    match_master = None
    
    # chunk both master and check array for consistent max array size; 
    # compare each master chunk to each check chunk and save matches 
    # to prune at end
    i = 0
    while i * chunk_size < num_blobs_master:
        start_master = i * chunk_size
        end_master = (i + 1) * chunk_size
        blobs_ref = blobs_master[start_master:end_master, :3].astype(dtype)
        j = 0
        while j * chunk_size < num_blobs_check:
            start_check = j * chunk_size
            end_check = (j + 1) * chunk_size
            blobs_check = blobs[start_check:end_check].astype(dtype)
            close_master, close = _find_close_blobs(blobs_check, blobs_ref, tol)
            # shift indices by offsets
            close += start_check
            close_master += start_master
            match_check = (close if match_check is None 
                           else np.concatenate((match_check, close)))
            match_master = (close_master if match_master is None 
                            else np.concatenate((match_master, close_master)))
            j += 1
        i += 1
    pruned = np.delete(blobs, match_check, axis=0)
    #if (len(close) > 0): print("{} removed".format(blobs[close][:, 0:4]))
    
    # shift close blobs to their mean values, storing values in the duplicated
    # coordinates and radius of the blob array after the confirmation value;
    # use the duplicated coordinates to work from any prior shifting; 
    # further duplicate testing will still be based on initial position to
    # allow detection of duplicates that occur in multiple ROI pairs
    abs_between = np.around(
        np.divide(
            np.add(Blobs.get_blob_abs_coords(blobs_master[match_master]), 
                   Blobs.get_blob_abs_coords(blobs[match_check])), 2))
    blobs_master[match_master] = Blobs.set_blob_abs_coords(
        blobs_master[match_master], abs_between)
    #print("blobs_master after shifting:\n{}".format(blobs_master[:, 5:9]))
    return pruned, blobs_master


def remove_close_blobs_within_array(blobs, region, tol):
    """Removes close blobs within a given array.
    
    Uses remove_close_blobs() to detect blobs close to one another inside
    the master array.
    
    Args:
        blobs: The blobs to add, given as 2D array of [n, [z, row, column, 
            radius]].
        region: Slice within each blob to check, such as slice(0, 2) to check
            for (z, row, column).
        tol: Tolerance to check for closeness, given in the same format
            as region. Blobs that are equal to or less than the the absolute
            difference for all corresponding parameters will be pruned in
            the returned array.
    
    Return:
        The blobs array without blobs falling inside the tolerance range.
    """
    if blobs is None:
        return None
    print("checking blobs for close duplicates:\n{}".format(blobs))
    blobs_all = None
    for blob in blobs:
        # check each blob against all blobs accepted thus far to ensure that
        # no blob is close to another blob
        #print("blob: {}".format(blob))
        if blobs_all is None:
            blobs_all = np.array([blob])
        else:
            # check an array of a single blob to add
            blobs_to_add, blobs_all = remove_close_blobs(
                np.array([blob]), blobs_all, region, tol)
            if blobs_to_add is not None:
                blobs_all = np.concatenate((blobs_all, blobs_to_add))
    return blobs_all


def meas_pruning_ratio(num_blobs_orig, num_blobs_after_pruning, num_blobs_next):
    """Measure blob pruning ratio.
    
    Args:
        num_blobs_orig: Number of original blobs, before pruning.
        num_blobs_after_pruning: Number of blobs after pruning.
        num_blobs_next: Number of a blobs in an adjacent segment, presumably 
            of similar size as that of the original blobs.
    
    Returns:
        Pruning ratios as a tuple of the original number of blobs, 
        blobs after pruning to original, and blobs after pruning to 
        the next region.
    """
    ratios = None
    if num_blobs_next > 0 and num_blobs_orig > 0:
        # calculate pruned:original and pruned:adjacent blob ratios
        print("num_blobs_orig: {}, blobs after pruning: {}, num_blobs_next: {}"
              .format(num_blobs_orig, num_blobs_after_pruning, num_blobs_next))
        ratios = (num_blobs_orig, num_blobs_after_pruning / num_blobs_orig, 
                  num_blobs_after_pruning / num_blobs_next)
    return ratios


def remove_close_blobs_within_sorted_array(blobs, tol):
    """Removes close blobs within a given array, first sorting the array by
    z, y, x.
    
    Args:
        blobs: The blobs to add, given as 2D array of [n, [z, row, column, 
            radius]].
        tol: Tolerance to check for closeness, given in the same format
            as region. Blobs that are equal to or less than the the absolute
            difference for all corresponding parameters will be pruned in
            the returned array.
    
    Return:
        Tuple of all blobs, a blobs array without blobs falling inside 
        the tolerance range.
    """
    if blobs is None:
        return None
    sort = np.lexsort((blobs[:, 2], blobs[:, 1], blobs[:, 0]))
    blobs = blobs[sort]
    #print("checking sorted blobs for close duplicates:\n{}".format(blobs))
    blobs_all = None
    for blob in blobs:
        # check each blob against all blobs accepted thus far to ensure that
        # no blob is close to another blob
        #print("blob: {}".format(blob))
        if blobs_all is None:
            # initialize array with first blob
            blobs_all = np.array([blob])
        else:
            # check each blob to add against blobs approved thus far
            i = len(blobs_all) - 1
            while i >= 0:
                blobs_diff = np.abs(np.subtract(
                    blob[:3], blobs_all[i, :3]))
                #print(blobs_diff)
                if (blobs_diff <= tol).all():
                    # remove duplicate blob and shift to mean of coords, 
                    # storing values in saved blob's abs coords; note that 
                    # this shift means that re-pruning the same region 
                    # may lead to further pruning
                    abs_between = np.around(
                        np.divide(
                            np.add(
                                Blobs.get_blob_abs_coords(blobs_all[i, None]), 
                                Blobs.get_blob_abs_coords(blob[None])), 2))
                    Blobs.set_blob_abs_coords(blobs_all[i, None], abs_between)
                    #print("updated blob:", blobs_all[i])
                    #print("removed blob:", blob)
                    break
                elif i == 0 or not (blobs_diff <= tol).any():
                    # add blob since at start of non-duplicate blobs list
                    # or no further chance for match within sorted list
                    blobs_all = np.concatenate((blobs_all, [blob]))
                    break
                i -= 1
    #print("blobs without close duplicates:\n{}".format(blobs_all))
    return blobs_all


def get_blobs_in_roi(
        blobs: np.ndarray, offset: Sequence[int], size: Sequence[int],
        margin: Sequence[int] = (0, 0, 0), reverse: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Get blobs within an ROI based on offset and size.
    
    Note that dimensions are in x,y,z for natural ordering but may change for
    consistency with z,y,x ordering used throughout MagellanMapper.
    
    Args:
        blobs: The blobs to retrieve, given as 2D array of
            ``[n, [z, row, column, radius, ...]]``.
        offset: Offset coordinates in ``x,y,z``.
        size: Size of ROI in ``x,y,z``.
        margin: Additional space outside the ROI to include in ``x,y,z``.
        reverse: True to reverse the order of ``offset`` and ``size``,
            assuming that they are in ``x,y,z`` rather than ``z,y,x`` order.
            Defaults to True for backward compatibility with the ROI
            convention used here.
    
    Returns:
        Tuple of:
        - ``segs_all`: Blobs within the ROI
        - ``mask``: the mask used to retrieve these blobs
    
    """
    if reverse:
        offset = offset[::-1]
        size = size[::-1]
    mask = np.all([
        blobs[:, 0] >= offset[0] - margin[0],
        blobs[:, 0] < offset[0] + size[0] + margin[0],
        blobs[:, 1] >= offset[1] - margin[1],
        blobs[:, 1] < offset[1] + size[1] + margin[1],
        blobs[:, 2] >= offset[2] - margin[2],
        blobs[:, 2] < offset[2] + size[2] + margin[2]], axis=0)
    segs_all = blobs[mask]
    return segs_all, mask


def get_blobs_interior(blobs, shape, pad_start, pad_end):
    """Get blobs within the interior of a region based on padding.
    
    Args:
        blobs: The blobs to retrieve, given as 2D array of 
            ``[n, [z, row, column, radius, ...]]``.
        shape: Shape of the region in z,y,x.
        pad_start: Offset of interior region in z,y,x to include blobs.
        pad_end: End offset in z,y,x.
    
    Returns:
        Blobs within the given interior.
    """
    return blobs[
        np.all([
            blobs[:, 0] >= pad_start[0], 
            blobs[:, 0] < shape[0] - pad_end[0],
            blobs[:, 1] >= pad_start[1], 
            blobs[:, 1] < shape[1] - pad_end[1],
            blobs[:, 2] >= pad_start[2], 
            blobs[:, 2] < shape[2] - pad_end[2]], axis=0)]


def _test_blob_duplicates():
    # tests blob duplication removal
    blobs = np.array([[1, 3, 4, 2.2342], [1, 8, 5, 3.13452], [1, 3, 4, 5.1234],
                      [1, 3, 5, 2.2342], [3, 8, 5, 3.13452]])
    print("sample blobs:\n{}".format(blobs))
    end = 3
    blobs_unique = remove_duplicate_blobs(blobs, slice(0, end))
    print("blobs_unique through first {} elements:\n{}"
          .format(end, blobs_unique))
    
    # tests removal of blobs within a given tolerance level
    tol = (1, 2, 2)
    blobs = remove_close_blobs_within_array(blobs, slice(0, end), tol)
    print("pruned sample blobs within tolerance {}:\n{}".format(tol, blobs))
    blobs_to_add = np.array([[1, 3, 5, 2.2342], [2, 10, 5, 3.13452], 
                             [2, 2, 4, 5.1234], [3, 3, 5, 2.2342]])
    print("blobs to add:\n{}".format(blobs_to_add))
    blobs_to_add = remove_close_blobs(blobs_to_add, blobs, slice(0, end), tol)
    print("pruned blobs to add:\n{}".format(blobs_to_add))


if __name__ == "__main__":
    print("Detector tests...")
