# Object colocalization
# Copyright The MagellanMapper Contributors
"""Colocalize objects in an image, typically in separate channels."""

import multiprocessing as mp

import numpy as np
from skimage import morphology

from magmap.cv import chunking, detector, stack_detect
from magmap.io import cli, libmag, sqlite
from magmap.settings import config


class StackColocalizer(object):
    """Colocalize blobs from different channels in a full image stack
    with multiprocessing.

    """
    blobs = None
    match_tol = None
    
    @classmethod
    def colocalize_block(cls, coord, offset, shape, blobs=None,
                         tol=None, setup_cli=False):
        """Colocalize blobs from different channels within a block.

        Args:
            coord (Tuple[int]): Block coordinate.
            offset (List[int]): Block offset within the full image in z,y,x.
            shape (List[int]): Block shape in z,y,x.
            blobs (:obj:`np.ndarray`): 2D blobs array; defaults to None to
                use :attr:`blobs`.
            tol (List[float]): Tolerance for colocalizing blobs; defaults
                to None to use :attr:`match_tol`.
            setup_cli (bool): True to set up CLI arguments, typically for
                a spawned (rather than forked) environment; defaults to False.

        Returns:
            Tuple[int], dict[Tuple[int], Tuple]: ``coord`` for tracking
            multiprocessing and the dictionary of matches.

        """
        if blobs is None:
            blobs = cls.blobs
        if tol is None:
            tol = cls.match_tol
        if setup_cli:
            cli.main(True, True)
        matches = colocalize_blobs_match(blobs, offset[::-1], shape[::-1], tol)
        return coord, matches
    
    @classmethod
    def colocalize_stack(cls, shape, blobs):
        """Entry point to colocalizing blobs within a stack.

        Args:
            shape (List[int]): Image shape in z,y,x.
            blobs (:obj:`np.ndarray`): 2D Numpy array of blobs.

        Returns:
            dict[Tuple[int], Tuple]: The dictionary of matches.

        """
        # set up ROI blocks from which to select blobs in each block
        sub_roi_slices, sub_rois_offsets, _, _, _, overlap_base, _, _ \
            = stack_detect.setup_blocks(config.roi_profile, shape)
        match_tol = np.multiply(
            overlap_base, config.roi_profile["verify_tol_factor"])
        
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
                        pool_results.append(pool.apply_async(
                            StackColocalizer.colocalize_block,
                            args=(coord, offset, shape,
                                  detector.get_blobs_in_roi(
                                      blobs, offset, shape)[0], match_tol,
                                  True)))
        
        # store blobs in dict with channel combos as keys
        matches_all = {}
        for result in pool_results:
            coord, matches = result.get()
            count = 0
            for key, val in matches.items():
                if key in matches_all:
                    matches_all[key].extend(val)
                else:
                    matches_all[key] = val
                count += len(val)
            print("adding {} matches from block at {} of {}"
                  .format(count, coord, np.add(sub_roi_slices.shape, -1)))
        
        pool.close()
        pool.join()
        
        # prune duplicates by taking matches with shortest distance
        for matchi in range(2):
            for key in matches_all.keys():
                # convert matches for channel-combo to ndarray to extract
                # blobs by column
                matches = np.array(matches_all[key])
                matches_uniq, matches_i, matches_inv, matches_cts = np.unique(
                    np.array(matches[:, matchi].tolist(), dtype=int)[:, :3],
                    axis=0, return_index=True, return_inverse=True,
                    return_counts=True)
                if np.sum(matches_cts > 1) > 0:
                    # prune if at least one blob has been matched to multiple
                    # other blobs
                    singles = matches[matches_i[matches_cts == 1]]
                    dups = []
                    for i, ct in enumerate(matches_cts):
                        # include non-duplicates to retain index
                        if ct <= 1: continue
                        # get indices in orig matches at given uniq array index
                        # and take match with lowest dist
                        matches_mult = matches[matches_inv == i]
                        min_dist = np.amin(matches_mult[:, 2])
                        num_matches = len(matches_mult)
                        if config.verbose and num_matches > 1:
                            print("pruning from", num_matches,
                                  "matches of dist:", matches_mult[:, 2])
                        matches_mult = matches_mult[
                            matches_mult[:, 2] == min_dist]
                        dups.append(matches_mult[0])
                    matches_all[key] = np.vstack((singles, dups))
                if config.verbose:
                    print("Colocalization matches for channels", key)
                    for match in matches_all[key]:
                        print("Blob1 {}, Blob2 {}, dist {}".format(
                            match[0][:3], match[1][:3], match[2]))
        
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
    threshs = []
    selem = morphology.ball(2)
    
    # find only blobs in ROI since blobs list may include blobs from immediate
    # surrounds, but ROI is not available for them
    blobs_roi, blobs_roi_mask = detector.get_blobs_in_roi(
        blobs, (0, 0, 0), roi.shape[:3], reverse=False)
    blobs_chl = detector.get_blobs_channel(blobs_roi)
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

    channels = np.unique(detector.get_blobs_channel(blobs_roi)).astype(int)
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
                    print(blobi, detector.get_blob_channel(blobs_roi[blobi]),
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
            print(i, detector.get_blob_channel(blob), blob[:3], coloc)
    return colocs


def colocalize_blobs_match(blobs, offset, size, tol, inner_padding=None):
    """Co-localize blobs in separate channels but the same ROI by finding
    optimal blob matches.

    Args:
        blobs (:obj:`np.ndarray`): Blobs from separate channels.
        offset (List[int]): ROI offset given as x,y,z.
        size (List[int]): ROI shape given as x,y,z.
        tol (List[float]): Tolerances for matching given as x,y,z
        inner_padding (List[int]): ROI padding given as x,y,z; defaults
            to None to use the padding based on ``tol``.

    Returns:
        Dict[Tuple, List]: Dictionary where keys are tuples of the two
        channels compared and values are a list of blob matches.

    """
    if blobs is None:
        return None
    thresh, scaling, inner_pad, resize, blobs = detector.setup_match_blobs_roi(
        blobs, tol)
    if inner_padding is None:
        inner_padding = inner_pad
    matches_chls = {}
    channels = np.unique(detector.get_blobs_channel(blobs)).astype(int)
    for chl in channels:
        # pair channels
        blobs_chl = detector.blobs_in_channel(blobs, chl)
        for chl_other in channels:
            # prevent duplicates by skipping other channels below given channel
            if chl >= chl_other: continue
            # find colocalizations between blobs from one channel to blobs
            # in another channel
            blobs_chl_other = detector.blobs_in_channel(blobs, chl_other)
            blobs_inner_plus, blobs_truth_inner_plus, offset_inner, \
                size_inner, matches = detector.match_blobs_roi(
                    blobs_chl_other, blobs_chl, offset, size, thresh, scaling,
                    inner_padding, resize)
            
            # reset truth and confirmation flags from matcher
            chl_combo = (chl, chl_other)
            for match in matches:
                for i, c in enumerate(chl_combo):
                    detector.set_blob_truth(match[i], -1)
                    detector.set_blob_confirmed(match[i], -1)
            matches_chls[chl_combo] = matches
    return matches_chls


def _get_roi_id(db, offset, shape):
    """Get database ROI ID for the given ROI position within the main image5d.
    
    Args:
        db (:obj:`sqlite.ClrDB`): Database object.
        offset (List[int]): ROI offset in z,y,x.
        shape (List[int]): ROI shape in z,y,x.

    Returns:
        int: ROI ID or found or inserted ROI.

    """
    exp_name = sqlite.get_exp_name(
        config.img5d.path_img if config.img5d else None)
    exp_id = sqlite.select_or_insert_experiment(
        db.conn, db.cur, exp_name, None)
    roi_id = sqlite.select_or_insert_roi(
        db.conn, db.cur, exp_id, config.series, offset, shape)[0]
    return roi_id


def insert_matches(db, offset, shape, matches):
    """Insert matches into database.
    
    Args:
        db (:obj:`sqlite.ClrDB`): Database object.
        offset (List[int]): ROI offset in z,y,x.
        shape (List[int]): ROI shape in z,y,x.
        matches (Dict, Tuple): Dictionary of matches.

    """
    roi_id = _get_roi_id(db, offset, shape[::-1])
    for match in matches.values():
        blobs = [b for m in match for b in m[:2]]
        sqlite.insert_blobs(db.conn, db.cur, roi_id, blobs)
        config.db.insert_blob_matches(roi_id, match)


def select_matches(db, offset, shape, channels):
    """Select blob matches for the given region from a database.
    
    Args:
        db (:obj:`sqlite.ClrDB`): Database object.
        offset (List[int]): ROI offset in z,y,x.
        shape (List[int]): ROI shape in z,y,x.
        channels (List[int]): List of channels.

    Returns:
        Dict[str, List[:obj:`magmap.io.sqlit.BlobMatch`]: Dictionary where
        keys are tuples of the two channels compared and values are a list
        of blob matches.

    """
    roi_id = _get_roi_id(db, (0, 0, 0), config.image5d.shape[3:0:-1])
    blobs, blob_ids = db.select_blobs_by_position(
        roi_id, offset[::-1], shape[::-1])
    blob_ids = np.array(blob_ids)
    matches = {}
    for chl in channels:
        # pair channels
        for chl_other in channels:
            if chl >= chl_other: continue
            # select matches for blobs in the given first channel of the pair
            # of channels, assuming chls were paired this way during insertion
            match = db.select_blob_matches_by_blob_id(
                roi_id, 1, blob_ids[detector.get_blobs_channel(blobs) == chl])
            match = [m for m in match
                     if detector.get_blob_channel(m.blob2) == chl_other]
            matches[(chl, chl_other)] = match
    return matches
