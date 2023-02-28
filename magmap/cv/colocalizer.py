# Object colocalization
# Copyright The MagellanMapper Contributors
"""Colocalize objects in an image, typically in separate channels."""

from enum import Enum
import multiprocessing as mp
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import numpy as np
from skimage import morphology

from magmap.cv import chunking, detector, stack_detect, verifier
from magmap.io import cli, df_io, libmag, sqlite
from magmap.settings import config

_logger = config.logger.getChild(__name__)


class BlobMatch:
    """Blob match storage class as a wrapper for a data frame of matches.

    Attributes:
        df: Data frame of matches with column names given by
            :class:`BlobMatch.Cols`; defaults to None.
        coords: Blob match coordinates, typically the mean coordinates of
            each blob pair; defaults to None.
        cmap: Colormap for each match; defaults to None.

    """
    
    class Cols(Enum):
        """Blob match column names."""
        MATCH_ID = "MatchID"
        ROI_ID = "RoiID"
        BLOB1_ID = "Blob1ID"
        BLOB1 = "Blob1"
        BLOB2_ID = "Blob2ID"
        BLOB2 = "Blob2"
        DIST = "Distance"
    
    def __init__(self, matches=None, match_id=None, roi_id=None, blob1_id=None,
                 blob2_id=None, df=None):
        """Initialize blob match object.

        Args:
            matches (list[list[
                :class:`numpy.ndarray`, :class:`numpy.ndarray`, float]]:
                List of blob match lists, which each contain,
                ``blob1, blob2, distance``. Defaults to None, which
                sets the data frame to None.
            match_id (Sequence[int]): Sequence of match IDs, which should be
                of the same length as ``matches``; defaults to None.
            roi_id (Sequence[int]): Sequence of ROI IDs, which should be
                of the same length as ``matches``; defaults to None.
            blob1_id (Sequence[int]): Sequence of blob 1 IDs, which should be
                of the same length as ``matches``; defaults to None.
            blob2_id (Sequence[int]): Sequence of blob2 IDs, which should be
                of the same length as ``matches``; defaults to None.
            df (:class:`pandas.DataFrame`): Pandas data frame to set in
                place of any other arguments; defaults to None.
        """
        self.df: Optional[pd.DataFrame] = None
        self.coords: Optional[np.ndarray] = None
        self.cmap: Optional[np.ndarray] = None
        
        if df is not None:
            # set data frame directly and ignore any other arguments
            self.df = df
            return
        if matches is None:
            # return since any other arguments must correspond to matches
            return
        
        matches_dict = {}
        for i, match in enumerate(matches):
            # assumes that all first sequences are of the same length
            vals = {
                BlobMatch.Cols.BLOB1: match[0],
                BlobMatch.Cols.BLOB2: match[1],
                BlobMatch.Cols.DIST: match[2],
            }
            if match_id is not None:
                vals[BlobMatch.Cols.MATCH_ID] = match_id[i]
            if roi_id is not None:
                vals[BlobMatch.Cols.ROI_ID] = roi_id[i]
            if blob1_id is not None:
                vals[BlobMatch.Cols.BLOB1_ID] = blob1_id[i]
            if blob2_id is not None:
                vals[BlobMatch.Cols.BLOB2_ID] = blob2_id[i]
            for key in BlobMatch.Cols:
                matches_dict.setdefault(key, []).append(
                    vals[key] if key in vals else None)
        self.df = df_io.dict_to_data_frame(matches_dict)
    
    def __repr__(self):
        """Format the underlying data frame."""
        if self.df is None:
            return "Empty blob matches"
        return df_io.print_data_frame(self.df, show=False, max_rows=10)
    
    def get_blobs(self, n: int) -> Optional[np.ndarray]:
        """Get blobs as a numpy array.

        Args:
            n: 1 for blob1, otherwise blob 2.

        Returns:
            Numpy array of the given blob type, or
            None if the :attr:`df` is None or the blob column does not exist.

        """
        col = BlobMatch.Cols.BLOB1 if n == 1 else BlobMatch.Cols.BLOB2
        if self.df is None or col.value not in self.df:
            return None
        df = self.df[col.value]
        if len(df) == 0:
            return None
        return np.vstack(df)
    
    def get_blobs_all(self) -> Optional[List[np.ndarray]]:
        """Get all blobs in the blob matches.
        
        Returns:
            Tuple of ``(blobs1, blobs2)``, or None if either are None.

        """
        blobs_all = []
        for n in (1, 2):
            blobs = self.get_blobs(n)
            if blobs is None:
                return None
            blobs_all.append(blobs)
        return blobs_all
    
    def update_blobs(self, fn, *args):
        """Update all blobs with the given function.

        Args:
            fn (func): Function that accepts the output of :meth:`get_blobs`
                separately for each set of blobs.
            *args (Any): Additional arguments to ``fn``.

        """
        if self.df is None: return
        for i, col in enumerate((BlobMatch.Cols.BLOB1, BlobMatch.Cols.BLOB2)):
            blobs = self.get_blobs(i + 1)
            if blobs is not None:
                self.df[col.value] = fn(blobs, *args).tolist()
    
    def get_mean_coords(self):
        """Get mean value of each pair of matched blobs.
        
        Returns:
            ``[n, 3]`` array of ``n`` blob pairs, also set to :attr:`coords`.

        """
        blobs = self.get_blobs_all()
        if any([b is None for b in blobs]):
            return None
        self.coords = np.mean([b[:, :3] for b in blobs], axis=0)
        return self.coords


class StackColocalizer(object):
    """Colocalize blobs in blocks based on matching blobs across channels.
    
    Support shared memory for spawned multiprocessing, with fallback to
    pickling in forked multiprocessing.

    """
    blobs: Optional["detector.Blobs"] = None
    match_tol: Optional[Sequence[float]] = None
    #: Channels to match; defaults to None.
    channels: Optional[Sequence[int]] = None
    
    @classmethod
    def colocalize_block(
            cls, coord: Sequence[int], offset: Sequence[int],
            shape: Sequence[int], blobs: Optional["detector.Blobs"] = None,
            tol: Optional[Sequence[float]] = None, setup_cli: bool = False,
            channels: Optional[Sequence[int]] = None
    ) -> Tuple[Sequence[int], Dict[Tuple[int, int], "BlobMatch"]]:
        """Colocalize blobs from different channels within a block.

        Args:
            coord: Block coordinate.
            offset: Block offset within the full image in z,y,x.
            shape: Block shape in z,y,x.
            blobs: Blobs; defaults to None, which will use :attr:`blobs`.
            tol: Tolerance for colocalizing blobs; defaults
                to None to use :attr:`match_tol`.
            setup_cli: True to set up CLI arguments, typically for
                a spawned (rather than forked) environment; defaults to False.
            channels: Channels to match; defaults to None, where
                :attr:`channels` will be used.

        Returns:
            Tuple of:
            - ``coord``: ``coord``, for tracking multiprocessing
            - ``matches``: dictionary of matches

        """
        if blobs is None:
            blobs = cls.blobs
        if tol is None:
            tol = cls.match_tol
        if channels is None:
            channels = cls.channels
        if setup_cli:
            # reload command-line parameters
            cli.process_cli_args()
        _logger.debug(
            "Match-based colocalizing blobs in ROI at offset %s, size %s",
            offset, shape)
        matches = colocalize_blobs_match(
            blobs, offset[::-1], shape[::-1], tol, channels=channels)
        return coord, matches
    
    @classmethod
    def colocalize_stack(
            cls, shape: Sequence[int], blobs: "detector.Blobs",
            channels: Optional[Sequence[int]] = None
    ) -> Dict[Tuple[int, int], "BlobMatch"]:
        """Entry point to colocalizing blobs within a stack.

        Args:
            shape: Image shape in z,y,x.
            blobs: Blobs.
            channels: Channels to match; defaults to None.

        Returns:
            Dictionary of matches, where keys are tuples of the channel pairs,
            and values are blob match objects.

        """
        chls = ("each pair of channels" if channels is None
                else f"channels: {channels}")
        _logger.info(
            "Colocalizing blobs based on matching blobs in %s", chls)
        # scale match tolerance based on block processing ROI size
        blocks = stack_detect.setup_blocks(config.roi_profile, shape)
        match_tol = np.multiply(
            blocks.overlap_base, config.roi_profile["verify_tol_factor"])
        
        # re-split stack with ROI size adjusted to the inner padding plus
        # the raw overlap size
        inner_pad = verifier.setup_match_blobs_roi(match_tol)[2]
        inner_pad = np.add(inner_pad, blocks.overlap_base)
        sub_roi_slices, sub_rois_offsets = chunking.stack_splitter(
            shape, blocks.max_pixels, inner_pad[::-1])
        
        is_fork = chunking.is_fork()
        if is_fork:
            # set shared data in forked multiprocessing
            cls.blobs = blobs
            cls.match_tol = match_tol
        pool = mp.Pool(processes=config.cpus)
        pool_results = []
        for z in range(sub_roi_slices.shape[0]):
            for y in range(sub_roi_slices.shape[1]):
                for x in range(sub_roi_slices.shape[2]):
                    coord = (z, y, x)
                    offset = sub_rois_offsets[coord]
                    slices = sub_roi_slices[coord]
                    shape = [s.stop - s.start for s in slices]
                    if is_fork:
                        # use variables stored as class attributes
                        pool_results.append(pool.apply_async(
                            StackColocalizer.colocalize_block,
                            args=(coord, offset, shape)))
                    else:
                        # pickle full set of variables
                        blobs_roi = detector.Blobs(detector.get_blobs_in_roi(
                            blobs.blobs, offset, shape)[0])
                        pool_results.append(pool.apply_async(
                            StackColocalizer.colocalize_block,
                            args=(coord, offset, shape, blobs_roi, match_tol,
                                  True, channels)))
        
        # dict of channel combos to blob matches data frame
        matches_all = {}
        for result in pool_results:
            coord, matches = result.get()
            count = 0
            for key, val in matches.items():
                matches_all.setdefault(key, []).append(val.df)
                count += len(val.df)
            _logger.info(
                "Adding %s matches from block at %s of %s",
                count, coord, np.add(sub_roi_slices.shape, -1))
        
        pool.close()
        pool.join()
        
        # prune duplicates by taking matches with shortest distance
        for key in matches_all.keys():
            matches_all[key] = pd.concat(matches_all[key])
            if matches_all[key].size > 0:
                for blobi in (BlobMatch.Cols.BLOB1, BlobMatch.Cols.BLOB2):
                    # convert blob column to ndarray to extract coords by column
                    matches = matches_all[key]
                    matches_uniq, matches_i, matches_inv, matches_cts = np.unique(
                        np.vstack(matches[blobi.value])[:, :3], axis=0,
                        return_index=True, return_inverse=True,
                        return_counts=True)
                    if np.sum(matches_cts > 1) > 0:
                        # prune if at least one blob has been matched to
                        # multiple other blobs
                        singles = matches.iloc[matches_i[matches_cts == 1]]
                        dups = []
                        for i, ct in enumerate(matches_cts):
                            # include non-duplicates to retain index
                            if ct <= 1: continue
                            # get indices in orig matches at given unique array
                            # index and take match with lowest dist
                            matches_mult = matches.loc[matches_inv == i]
                            dists = matches_mult[BlobMatch.Cols.DIST.value]
                            min_dist = np.amin(dists)
                            num_matches = len(matches_mult)
                            if num_matches > 1:
                                _logger.debug(
                                    "Pruning from %s matches of dist: %s",
                                    num_matches, dists)
                            matches_mult = matches_mult.loc[dists == min_dist]
                            # take first in case of any ties
                            dups.append(matches_mult.iloc[[0]])
                        matches_all[key] = pd.concat((singles, pd.concat(dups)))
                _logger.info(
                    "Colocalization matches for channels %s: %s",
                    key, len(matches_all[key]))
            _logger.debug(
                "Blob matches for %s after pruning:\n%s", key, matches_all[key])
            # store data frame in BlobMatch object
            matches_all[key] = BlobMatch(df=matches_all[key])
        
        return matches_all


def colocalize_blobs(roi, blobs, thresh=None):
    """Co-localize blobs from different channels based on surrounding
    intensities.
    
    Thresholds for detection are first identified in each channel by taking
    the blobs in the given channel, finding the surrounding intensities,
    and taking a low (5th) percentile. Then for each channel, the
    surrounding intensities of blobs in that channel are compared with
    the thresholds in the other channels. Blobs exceeding any given
    threshold are considered to co-localize in that channel.
    
    Args:
        roi (:obj:`np.ndarray`): Region of interest as a 3D+channel array.
        blobs (:obj:`np.ndarray`): Blobs as a 2D array in the format
            ``[[z, y, x, radius, confirmation, truth, channel...], ...]``.
        thresh (int, float, str): Threshold percentile of intensities from
            pixels surrounding each blob in the given channel. Use "min"
            to instead take the mininimum average intensity of all blobs
            in the channel. Defaults to None to use "min".

    Returns:
        :obj:`np.ndarray`: 2D Numpy array of same length as ``blobs`` with
        a column for each channel where 1 indicates that the corresponding
        blob has signal is present in the given channels at the blob's
        location, and 0 indicates insufficient signal.

    """
    if blobs is None or roi is None or len(roi.shape) < 4:
        return None
    if thresh is None:
        thresh = "min"
    print("Colocalizing blobs based on image intensity across channels")
    threshs = []
    selem = morphology.ball(2)
    
    # find only blobs in ROI since blobs list may include blobs from immediate
    # surrounds, but ROI is not available for them
    blobs_roi, blobs_roi_mask = detector.get_blobs_in_roi(
        blobs, (0, 0, 0), roi.shape[:3], reverse=False)
    blobs_chl = detector.Blobs.get_blobs_channel(blobs_roi)
    blobs_range_chls = []
    
    # get labeled masks of blobs for each channel and threshold intensities
    mask_roi = np.ones(roi.shape[:3], dtype=int)
    mask_roi_chls = []
    for chl in range(roi.shape[3]):
        # label a mask with blob indices surrounding each blob
        blobs_chl_mask = np.isin(blobs_chl, chl)
        blobs_range = np.where(blobs_chl_mask)[0]
        blobs_range_chls.append(blobs_range)
        mask = np.copy(mask_roi) * -1
        mask[tuple(libmag.coords_for_indexing(
            blobs_roi[blobs_chl_mask, :3].astype(int)))] = blobs_range
        mask = morphology.dilation(mask, selem=selem)
        mask_roi_chls.append(mask)
        
        if thresh == "min":
            # set minimum average surrounding intensity of all blobs as thresh
            threshs.append(
                None if len(blobs_range) == 0 else np.amin([
                    np.mean(roi[mask == b, chl]) for b in blobs_range]))
        else:
            # set a percentile of intensities surrounding all blobs in channel
            # as threshold for that channel, or the whole ROI if no blobs
            mask_blobs = mask >= 0
            roi_mask = roi if np.sum(mask_blobs) < 1 else roi[mask_blobs, chl]
            threshs.append(np.percentile(roi_mask, thresh))

    channels = np.unique(detector.Blobs.get_blobs_channel(blobs_roi)).astype(int)
    colocs_roi = np.zeros((blobs_roi.shape[0], roi.shape[3]), dtype=np.uint8)
    for chl in channels:
        # get labeled mask of blobs in the given channel
        mask = mask_roi_chls[chl]
        blobs_range = blobs_range_chls[chl]
        for chl_other in channels:
            if threshs[chl_other] is None: continue
            for blobi in blobs_range:
                # find surrounding intensity of blob in another channel
                mask_blob = mask == blobi
                blob_avg = np.mean(roi[mask_blob, chl_other])
                if config.verbose:
                    print(
                        blobi, detector.Blobs.get_blobs_channel(
                            blobs_roi[blobi]),
                        blobs_roi[blobi, :3], blob_avg, threshs[chl_other])
                if blob_avg >= threshs[chl_other]:
                    # intensities in another channel around blob's position
                    # is above that channel's threshold
                    colocs_roi[blobi, chl_other] = 1
    
    # create array for all blobs including those outside ROI
    colocs = np.zeros((blobs.shape[0], roi.shape[3]), dtype=np.uint8)
    colocs[blobs_roi_mask] = colocs_roi
    if config.verbose:
        for i, (blob, coloc) in enumerate(zip(blobs_roi, colocs)):
            print(i, detector.Blobs.get_blobs_channel(blob), blob[:3], coloc)
    return colocs


def colocalize_blobs_match(
        blobs: "detector.Blobs", offset: Sequence[int], size: Sequence[int],
        tol: Sequence[float], inner_padding: Optional[Sequence[int]] = None,
        channels: Optional[Sequence[int]] = None
) -> Optional[Dict[Tuple[int, int], "BlobMatch"]]:
    """Co-localize blobs in separate channels but the same ROI by finding
    optimal blob matches.

    Args:
        blobs: Blobs.
        offset: ROI offset given as x,y,z.
        size: ROI shape given as x,y,z.
        tol: Tolerances for matching given as x,y,z
        inner_padding: ROI padding given as x,y,z; defaults
            to None to use the padding based on ``tol``.
        channels: Indices of channels to colocalize. Defaults to None, which
            will use all channels in ``blobs``.

    Returns:
        Dictionary where keys are tuples of the two channels compared and
        values are blob matches objects, or None if ``blobs`` is None.

    """
    if blobs is None:
        return None
    thresh, scaling, inner_pad, resize, blobs_roi = \
        verifier.setup_match_blobs_roi(tol, blobs)
    if inner_padding is None:
        inner_padding = inner_pad
    matches_chls = {}
    
    # get channels in blobs
    blob_chls = np.unique(blobs.get_blobs_channel(blobs_roi)).astype(int)
    if channels is not None:
        # filter blob channels to only include specified channels
        blob_chls = [c for c in blob_chls if c in channels]
    
    for chl in blob_chls:
        # pair channels
        blobs_chl = blobs.blobs_in_channel(blobs_roi, chl)
        for chl_other in blob_chls:
            # prevent duplicates by skipping other channels below given channel
            if chl >= chl_other: continue
            # find colocalizations between blobs from one channel to blobs
            # in another channel
            blobs_chl_other = blobs.blobs_in_channel(blobs_roi, chl_other)
            blobs_inner_plus, blobs_truth_inner_plus, offset_inner, \
                size_inner, matches = verifier.match_blobs_roi(
                    blobs_chl_other, blobs_chl, offset, size, thresh, scaling,
                    inner_padding, resize)
            
            # reset truth and confirmation blob flags in matches
            chl_combo = (chl, chl_other)
            matches.update_blobs(blobs.set_blob_truth, -1)
            matches.update_blobs(blobs.set_blob_confirmed, -1)
            matches_chls[chl_combo] = matches
    
    return matches_chls


def _get_roi_id(
        db: "sqlite.ClrDB", offset: Sequence[int], shape: Sequence[int],
        exp_name: Optional[str] = None) -> int:
    """Get database ROI ID for the given ROI position within the main image5d.
    
    Args:
        db: Database object.
        offset: ROI offset in z,y,x.
        shape: ROI shape in z,y,x.
        exp_name: Name of experiment; defaults to None to attempt
            discovery through any image loaded to :attr:`config.img5d`.

    Returns:
        int: ROI ID or found or inserted ROI.

    """
    if exp_name is None:
        exp_name = sqlite.get_exp_name(
            config.img5d.path_img if config.img5d else None)
    exp_id = db.select_or_insert_experiment(exp_name, None)
    roi_id = sqlite.select_or_insert_roi(
        db.conn, db.cur, exp_id, config.series, offset, shape)[0]
    return roi_id


def insert_matches(db, matches):
    """Insert matches into database for a whole image.
    
    Args:
        db (:obj:`sqlite.ClrDB`): Database object.
        matches (dict[tuple[int, int], :class:`BlobMatch`):
            Dictionary of channel combo tuples to blob match objects.

    """
    # use size of 0 for each dimension for whole-image ROI, which avoids
    # the need to discover the image size
    roi_id = _get_roi_id(db, (0, 0, 0), (0, 0, 0))
    
    for chl_matches in matches.values():
        # insert blobs and matches for the given channel combo
        blobs = chl_matches.get_blobs_all()
        if blobs is not None:
            sqlite.insert_blobs(db.conn, db.cur, roi_id, np.vstack(blobs))
            config.db.insert_blob_matches(roi_id, chl_matches)


def select_matches(db, channels, offset=None, shape=None, exp_name=None):
    """Select blob matches for the given region from a database.
    
    Blob matches are assumed to have been processed from the whole image.
    To retrieve matches from a selected ROI, use
    :meth:`magmap.io.sqlite.ClrDB.select_blob_matches` instead.
    
    Args:
        db (:obj:`sqlite.ClrDB`): Database object.
        channels (list[int]): List of channels.
        offset (list[int]): ROI offset in z,y,x; defaults to None to use
            ``(0, 0, 0)``.
        shape (list[int]): ROI shape in z,y,x; defaults to None to use
            ``(0, 0, 0)``.
        exp_name (str): Name of experiment in ``db``.

    Returns:
        dict[tuple[int, int], list[:obj:`BlobMatch`]: Dictionary where
        keys are tuples of the two channels compared and values are a list
        of blob matches. None if no blob matches are found.

    """
    # get ROI for whole image, where whole-image matches are stored
    roi_id = _get_roi_id(db, (0, 0, 0), (0, 0, 0), exp_name)
    blob_matches = db.select_blob_matches(roi_id, offset, shape)
    if blob_matches.df is None or len(blob_matches.df) == 0:
        _logger.info("No blob matches found")
        return None
    
    matches = {}
    for chl in channels:
        # pair channels
        for chl_other in channels:
            if chl >= chl_other: continue
            # select matches for blobs in the first channel of the channel pair,
            # assuming chls were paired this way during insertion
            blobs1 = blob_matches.get_blobs(1)
            if blobs1 is None: continue
            chl_matches = BlobMatch(df=blob_matches.df.loc[
                detector.Blobs.get_blobs_channel(blobs1) == chl])
            
            # also extract only blob1's paired to blob2's in the 2nd channel 
            blobs2 = chl_matches.get_blobs(2)
            if blobs2 is None: continue
            chl_matches.df = chl_matches.df.loc[
                detector.Blobs.get_blobs_channel(blobs2) == chl_other]
            
            # store the matches with chl1-chl2 as key
            matches[(chl, chl_other)] = chl_matches
    return matches
