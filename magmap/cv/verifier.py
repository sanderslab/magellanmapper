# Blob verifications against ground truth
import os

import numpy as np
from scipy import optimize
from scipy.spatial import distance

from magmap.cv import colocalizer, detector
from magmap.io import df_io, libmag, sqlite
from magmap.settings import config
from magmap.stats import atlas_stats, mlearn

_logger = config.logger.getChild(__name__)


def _match_blobs(blobs, blobs_master, close, close_master, dists):
    """Group matches between blobs.
    
    Args:
        blobs (:obj:`np.ndarray`): Blobs as a 2D array of
            ``[n, [z, row, column, ...]]``.
        blobs_master (:obj:`np.ndarray`): Array in same format as ``blobs``
            from a master list.
        close (List[int]): Sequence of indices of ``blobs``.
        close_master (List[int]): Sequence of indices of ``blobs_master``
            matching the corresponding values in ``close``.
        dists (List[float]): Sequence of distances corresponding to the
            matches in ``close`` and ``close_master``.

    Returns:
        List[List]]: Sequence of matches, which each consist of
        ``blob_master, blob, distance``.

    """
    # show sorted list of matches between blobs and master blobs
    found_master = blobs_master[close_master]
    found_master, sort = detector.sort_blobs(found_master)
    found = blobs[close][sort]
    matches = []
    for f, fm, d in zip(found, found_master, dists[sort]):
        match = (fm, f, d)
        matches.append(match)
    return matches


def find_closest_blobs_cdist(blobs, blobs_master, thresh=None, scaling=None):
    """Find the closest blobs within a given tolerance using the 
    Hungarian algorithm to find blob matches.
    
    Args:
        blobs: Blobs as a 2D array of [n, [z, row, column, ...]].
        blobs_master: Array in same format as ``blobs``.
        thresh: Threshold distance beyond which blob pairings are excluded; 
            defaults to None to include all matches.
        scaling: Sequence of scaling factors by which to multiply the 
            blob coordinates before computing distances, used to 
            scale coordinates from an anisotropic to isotropic 
            ROI before computing distances, which assumes isotropy. 
            Defaults to None.
    
    Returns:
        Tuple of ``rowis`` and ``colis``, arrays of row and corresponding 
        column indices of the closest matches; and ``dists_closest``, an 
        array of corresponding distances for these matches. Only matches 
        within the given tolerance will be included.
    """
    blobs_scaled = blobs
    blobs_master_scaled = blobs_master
    if scaling is not None:
        # scale blobs and tolerance by given factor, eg for isotropy
        len_scaling = len(scaling)
        blobs_scaled = np.multiply(blobs[:, :len_scaling], scaling)
        blobs_master_scaled = np.multiply(blobs_master[:, :len_scaling], scaling)
    
    # find Euclidean distances between each pair of points and determine 
    # the optimal assignments using the Hungarian algorithm
    dists = distance.cdist(blobs_scaled, blobs_master_scaled)
    rowis, colis = optimize.linear_sum_assignment(dists)
    
    dists_closest = dists[rowis, colis]
    if thresh is not None:
        # filter out matches beyond the given threshold distance
        dists_in = dists_closest < thresh
        if config.verbose:
            print("only keeping blob matches within threshold distance of",
                  thresh)
            for blob, blob_sc, blob_base, blob_base_sc, dist, dist_in in zip(
                    blobs[rowis], blobs_scaled[rowis], blobs_master[colis],
                    blobs_master_scaled[colis], dists_closest,
                    dists_in):
                print("blob: {} (scaled {}), base: {} ({}), dist: {}, in? {}"
                      .format(blob[:3], blob_sc[:3], blob_base[:3],
                              blob_base_sc[:3], dist, dist_in))
        rowis = rowis[dists_in]
        colis = colis[dists_in]
        dists_closest = dists_closest[dists_in]
    
    return rowis, colis, dists_closest


def setup_match_blobs_roi(blobs, tol):
    """Set up tolerances for matching blobs in an ROI.
    
    Args:
        blobs (:obj:`np.ndarray`): Sequence of blobs to resize if the
            first ROI profile (:attr:`config.roi_profiles`) ``resize_blobs``
            value is given.
        tol (List[int, float]): Sequence of tolerances.

    Returns:
        float, List[float], List[float], List[float], :obj:`np.ndarray`:
        Distance map threshold, scaling normalized by ``tol``, ROI padding
        shape, resize sequence retrieved from ROI profile, and ``blobs``
        after any resizing.

    """
    # convert tolerance seq to scaling and single number distance
    # threshold for point distance map, which assumes isotropy; use
    # custom tol rather than calculating isotropy since may need to give
    # greater latitude along a given axis, such as poorer res in z
    thresh = np.amax(tol)  # similar to longest radius from the tol bounding box
    scaling = thresh / tol
    # casting to int causes improper offset import into db
    inner_padding = np.floor(tol[::-1])
    libmag.printv(
        "verifying blobs with tol {} leading to thresh {}, scaling {}, "
        "inner_padding {}".format(tol, thresh, scaling, inner_padding))
    
    # resize blobs based only on first profile
    resize = config.get_roi_profile(0)["resize_blobs"]
    if resize:
        blobs = detector.multiply_blob_rel_coords(blobs, resize)
        libmag.printv("resized blobs by {}:\n{}".format(resize, blobs))
    
    return thresh, scaling, inner_padding, resize, blobs


def match_blobs_roi(blobs, blobs_base, offset, size, thresh, scaling,
                    inner_padding, resize=None):
    """Match blobs from two sets of blobs in an ROI, prioritizing the inner
    portion of ROIs to avoid missing detections because of edge effects
    while also adding matches between a blob in the inner ROI and another
    blob in the remaining portion of the ROI.
    
    Args:
        blobs (:obj:`np.ndarray`): The blobs to be matched against
            ``blobs_base``, given as 2D array of
            ``[[z, row, column, radius, ...], ...]``.
        blobs_base (:obj:`np.ndarray`): The blobs to which ``blobs`` will
            be matched, in the same format as ``blobs``.
        offset (List[int]): ROI offset from which to select blobs in x,y,z.
        size (List[int]): ROI size in x,y,z.
        thresh (float): Distance map threshold
        scaling (List[float]): Scaling normalized by ``tol``.
        inner_padding (List[float]): ROI padding shape.
        resize (List[float]): Resize sequence retrieved from ROI profile;
            defaults to None.
    
    Returns:
        :class:`numpy.ndarray`, :class:`numpy.ndarray`, list[int], list[int],
        list[list[:class:`numpy.ndarray`, :class:`numpy.ndarray`, float]]:
        Array of blobs from ``blobs``; corresponding array from ``blobs_base``
        matching blobs in ``blobs``; offset of the inner portion of the ROI
        in absolute coordinates of x,y,z; shape of this inner portion of the
        ROI; and list of blob matches, each given as a list of
        ``blob_master, blob, distance``.
    
    """
    # get all blobs in inner and total ROI
    offset_inner = np.add(offset, inner_padding)
    size_inner = np.subtract(size, inner_padding * 2)
    libmag.printv(
        "offset: {}, offset_inner: {}, size: {}, size_inner: {}"
        .format(offset, offset_inner, size, size_inner))
    blobs_roi, _ = detector.get_blobs_in_roi(blobs, offset, size)
    if resize is not None:
        # TODO: doesn't align with exported ROIs
        padding = config.plot_labels[config.PlotLabels.PADDING]
        libmag.printv("shifting blobs in ROI by offset {}, border {}"
                      .format(offset, padding))
        blobs_roi = detector.shift_blob_rel_coords(blobs_roi, offset)
        if padding:
            blobs_roi = detector.shift_blob_rel_coords(blobs_roi, padding)
    blobs_inner, blobs_inner_mask = detector.get_blobs_in_roi(
        blobs_roi, offset_inner, size_inner)
    blobs_base_roi, _ = detector.get_blobs_in_roi(blobs_base, offset, size)
    blobs_base_inner, blobs_base_inner_mask = detector.get_blobs_in_roi(
        blobs_base_roi, offset_inner, size_inner)
    
    # compare blobs from inner region of ROI with all base blobs,
    # prioritizing the closest matches
    found, found_base, dists = find_closest_blobs_cdist(
        blobs_inner, blobs_base_roi, thresh, scaling)
    blobs_inner[:, 4] = 0
    blobs_inner[found, 4] = 1
    blobs_base_roi[blobs_base_inner_mask, 5] = 0
    blobs_base_roi[found_base, 5] = 1
    
    # add any base blobs missed in the inner ROI by comparing with
    # test blobs from outer ROI
    blobs_base_inner_missed = blobs_base_roi[blobs_base_roi[:, 5] == 0]
    blobs_outer = blobs_roi[np.invert(blobs_inner_mask)]
    found_out, found_base_out, dists_out = find_closest_blobs_cdist(
        blobs_outer, blobs_base_inner_missed, thresh, scaling)
    blobs_base_inner_missed[found_base_out, 5] = 1
    
    # combine inner and outer groups
    blobs_truth_inner_plus = np.concatenate(
        (blobs_base_roi[blobs_base_roi[:, 5] == 1],
         blobs_base_inner_missed))
    blobs_outer[found_out, 4] = 1
    blobs_inner_plus = np.concatenate((blobs_inner, blobs_outer[found_out]))

    matches_inner = _match_blobs(
        blobs_inner, blobs_base_roi, found, found_base, dists)
    matches_outer = _match_blobs(
        blobs_outer, blobs_base_inner_missed, found_out,
        found_base_out, dists_out)
    matches = colocalizer.BlobMatch([*matches_inner, *matches_outer])
    if config.verbose:
        '''
        print("blobs_roi:\n{}".format(blobs_roi))
        print("blobs_inner:\n{}".format(blobs_inner))
        print("blobs_base_inner:\n{}".format(blobs_base_inner))
        print("blobs_base_roi:\n{}".format(blobs_base_roi))
        print("found inner:\n{}"
              .format(blobs_inner[found]))
        print("truth found:\n{}"
              .format(blobs_base_roi[found_base]))
        print("blobs_outer:\n{}".format(blobs_outer))
        print("blobs_base_inner_missed:\n{}"
              .format(blobs_base_inner_missed))
        print("truth blobs detected by an outside blob:\n{}"
              .format(blobs_base_inner_missed[found_base_out]))
        print("all those outside detection blobs:\n{}"
              .format(blobs_roi_extra))
        print("blobs_inner_plus:\n{}".format(blobs_inner_plus))
        print("blobs_truth_inner_plus:\n{}".format(blobs_truth_inner_plus))
        '''

        print("Closest matches found (truth, detected, distance):")
        msgs = ("\n- Inner ROI:", "\n- Outer ROI:")
        for msg, matches_sub in zip(msgs, (matches_inner, matches_outer)):
            print(msg)
            for match in matches_sub:
                print(
                    "Blob1:", match[0][:3], "chl",
                    detector.get_blob_channel(match[0]), "Blob2:", match[1][:3],
                    "chl", detector.get_blob_channel(match[1]),
                    "dist:", match[2])
        print()
    
    return blobs_inner_plus, blobs_truth_inner_plus, offset_inner, size_inner, \
        matches


def verify_rois(rois, blobs, blobs_truth, tol, output_db, exp_id, exp_name,
                channel):
    """Verify blobs in ROIs by comparing detected blobs with truth sets
    of blobs stored in a database.
    
    Save the verifications to a separate database with a name in the same
    format as saved processed files but with "_verified.db" at the end.
    Prints basic statistics on the verification.
    
    Note that blobs are found from ROI parameters rather than loading from 
    database, so blobs recorded within these ROI bounds but from different 
    ROIs will be included in the verification.
    
    Args:
        rois: Rows of ROIs from sqlite database.
        blobs (:obj:`np.ndarray`): The blobs to be checked for accuracy,
            given as 2D array of ``[[z, row, column, radius, ...], ...]``.
        blobs_truth (:obj:`np.ndarray`): The list by which to check for
            accuracy, in the same format as blobs.
        tol: Tolerance as z,y,x of floats specifying padding for the inner
            ROI and used to generate a single tolerance distance within
            which a detected and ground truth blob will be considered
            potential matches.
        output_db: Database in which to save the verification flags, typical
            the database in :attr:``config.verified_db``.
        exp_id: Experiment ID in ``output_db``.
        exp_name (str): Name of experiment to store as the sample name for
            each row in the output data frame.
        channel (List[int]): Filter ``blobs_truth`` by this channel.
    
    Returns:
        tuple[int, int, int], str, :class:`pandas.DataFrame`: Tuple of
        ``pos, true_pos, false_pos`` stats, feedback message, and accuracy
        metrics in a data frame.
    
    """
    blobs_truth = detector.blobs_in_channel(blobs_truth, channel)
    blobs_truth_rois = None
    blobs_rois = None
    rois_falsehood = []
    thresh, scaling, inner_padding, resize, blobs = setup_match_blobs_roi(
        blobs, tol)
    
    # set up metrics dict for accuracy metrics of each ROI
    metrics = {}
    cols = (
        config.AtlasMetrics.SAMPLE,
        config.AtlasMetrics.CHANNEL,
        config.AtlasMetrics.OFFSET,
        config.AtlasMetrics.SIZE,
        mlearn.GridSearchStats.POS,
        mlearn.GridSearchStats.TP,
        mlearn.GridSearchStats.FP,
        mlearn.GridSearchStats.FN,
    )
    
    for roi in rois:
        # get ROI from database for ground truth blobs
        offset = (roi["offset_x"], roi["offset_y"], roi["offset_z"])
        size = (roi["size_x"], roi["size_y"], roi["size_z"])
        series = roi["series"]
        
        # find matches between truth and detected blobs
        blobs_inner_plus, blobs_truth_inner_plus, offset_inner, size_inner, \
            matches = match_blobs_roi(
                blobs, blobs_truth, offset, size, thresh, scaling,
                inner_padding, resize)
        
        # store blobs in separate verified DB
        roi_id, _ = sqlite.insert_roi(output_db.conn, output_db.cur, exp_id,
                                      series, offset_inner, size_inner)
        sqlite.insert_blobs(output_db.conn, output_db.cur, roi_id,
                            blobs_inner_plus)
        sqlite.insert_blobs(output_db.conn, output_db.cur, roi_id,
                            blobs_truth_inner_plus)
        output_db.insert_blob_matches(roi_id, matches)
        
        # compute accuracy metrics for the ROI
        pos = len(blobs_truth_inner_plus)  # condition pos
        true_pos = np.sum(blobs_inner_plus[:, 4] == 1)
        false_pos = np.sum(blobs_inner_plus[:, 4] == 0)
        false_neg = len(blobs_truth_inner_plus) - true_pos
        if false_neg > 0 or false_pos > 0:
            rois_falsehood.append((offset_inner, false_pos, false_neg))
        vals = (exp_name, channel[0] if channel else 0,
                tuple(offset_inner.astype(int)), tuple(size_inner.astype(int)),
                pos, true_pos, false_pos, pos - true_pos)
        for key, val in zip(cols, vals):
            metrics.setdefault(key, []).append(val)
        
        # combine blobs into total lists for stats across ROIs
        if blobs_truth_rois is None:
            blobs_truth_rois = blobs_truth_inner_plus
        else:
            blobs_truth_rois = np.concatenate(
                (blobs_truth_inner_plus, blobs_truth_rois))
        if blobs_rois is None:
            blobs_rois = blobs_inner_plus
        else:
            blobs_rois = np.concatenate((blobs_inner_plus, blobs_rois))
    
    # generate and show data frame of accuracy metrics for each ROI
    df = df_io.dict_to_data_frame(metrics, show=" ")
    
    # show accuracy metrics of blobs combined across ROIs
    true_pos = df[mlearn.GridSearchStats.TP.value].sum()
    false_pos = df[mlearn.GridSearchStats.FP.value].sum()
    pos = df[mlearn.GridSearchStats.POS.value].sum()
    false_neg = pos - true_pos
    print("Automated verification using tol {}:\n".format(tol))
    fdbk = "Accuracy metrics for channel {}:\n{}".format(
        channel, atlas_stats.calc_sens_ppv(
            pos, true_pos, false_pos, false_neg)[2])
    print(fdbk)
    print("ROIs with falsehood:\n{}".format(rois_falsehood))
    return (pos, true_pos, false_pos), fdbk, df


def verify_stack(filename_base, subimg_path_base, settings, segments_all,
                 channels, overlap_base):
    db_path_base = os.path.basename(subimg_path_base)
    stats_detection = None
    fdbk = None
    try:
        # Truth databases are any database stored with manually
        # verified blobs and loaded at command-line with the
        # `--truth_db` flag or loaded here. While all experiments
        # can be stored in a single database, this verification also
        # supports experiments saved to separate databases in the
        # software root directory and named as a sub-image but with
        # the `sqlite.DB_SUFFIX_TRUTH` suffix. Experiments in the
        # database are also assumed to be named based on the full
        # image or the sub-image filename, without any directories.
        
        # load ROIs from previously loaded truth database or one loaded
        # based on sub-image filename
        exp_name, rois = _get_truth_db_rois(
            subimg_path_base, filename_base,
            db_path_base if config.truth_db is None else None)
        if rois is None:
            # load alternate truth database based on sub-image filename
            print("Loading truth ROIs from experiment:", exp_name)
            exp_name, rois = _get_truth_db_rois(
                subimg_path_base, filename_base, db_path_base)
        if config.truth_db is None:
            raise LookupError(
                "No truth database found for experiment {}, will "
                "skip detection verification".format(exp_name))
        if rois is None:
            raise LookupError(
                "No truth set ROIs found for experiment {}, will "
                "skip detection verification".format(exp_name))
        
        # verify each ROI and store results in a separate database
        exp_id = sqlite.insert_experiment(
            config.verified_db.conn, config.verified_db.cur,
            exp_name, None)
        verify_tol = np.multiply(
            overlap_base, settings["verify_tol_factor"])
        stats_detection, fdbk, df_verify = verify_rois(
            rois, segments_all, config.truth_db.blobs_truth,
            verify_tol, config.verified_db, exp_id, exp_name,
            channels)
        df_io.data_frames_to_csv(df_verify, libmag.combine_paths(
            exp_name, "verify.csv"))
    except FileNotFoundError:
        libmag.warn("Could not load truth DB from {}; "
                    "will not verify ROIs".format(db_path_base))
    except LookupError as e:
        libmag.warn(str(e))
    return stats_detection, fdbk


def meas_detection_accuracy(blobs, verified=False, treat_maybes=0):
    """Measure detection accuracy based on blob confirmation statuses.

    Args:
        blobs (:obj:`np.ndarray`): 2D array of blobs in the format,
            ``[[z, row, column, radius, confirmation, truth, ...], ...]``.
        verified (bool): True to assume that blobs have undergone verification.
        treat_maybes (int): 0 to ignore maybes; 1 to treat maybes as correct,
            and 1 to treat maybes as incorrect.

    Returns:
        float, float, str: Sensivity, positive predictive value (PPV), and
        summary of stats as a string. If ``blobs`` is None or empty, returns
        None for each of these values.

    """
    if blobs is None or len(blobs) < 1:
        return None, None, None
    
    # basic stats based on confirmation status, ignoring maybes; "pos"
    # here means actual positives, whereas "true pos" means correct
    # detection, where radius <= 0 indicates that the blob was manually
    # added rather than detected; "false pos" are incorrect detections
    if verified:
        # basic stats based on confirmation status, ignoring maybes
        blobs_pos = blobs[blobs[:, 5] >= 0]  # all truth blobs
        blobs_detected = blobs[blobs[:, 5] == -1]  # all non-truth blobs
        blobs_true_detected = blobs_detected[blobs_detected[:, 4] == 1]
        blobs_false = blobs[blobs[:, 4] == 0]
    else:
        blobs_pos = blobs[blobs[:, 4] == 1]
        # TODO: consider just checking > 0
        blobs_true_detected = blobs_pos[blobs_pos[:, 3] >= config.POS_THRESH]
        blobs_false = blobs[blobs[:, 4] == 0]

    # calculate sensitivity and PPV; no "true neg" detection so no
    # specificity measurement
    all_pos = blobs_pos.shape[0]
    true_pos = blobs_true_detected.shape[0]
    false_pos = blobs_false.shape[0]
    if verified or treat_maybes == 0:
        # ignore maybes
        maybe_msg = "(ignoring maybes)"
    else:
        blobs_maybe = blobs[blobs[:, 4] == 2]
        blobs_maybe_detected = blobs_maybe[
            blobs_maybe[:, 3] >= config.POS_THRESH]
        num_maybe_detected = len(blobs_maybe_detected)
        if treat_maybes == 1:
            # most generous, where detections that are maybes are treated as
            # true pos, and missed blobs that are maybes are treated as ignored
            all_pos += num_maybe_detected
            true_pos += num_maybe_detected
            maybe_msg = "(treating maybes as correct)"
        else:
            # most conservative, where detections that are maybes are treated
            # as false pos, and missed blobs that are maybes are treated as pos
            all_pos += len(blobs_maybe) - num_maybe_detected
            false_pos += num_maybe_detected
            maybe_msg = "(treating maybes as incorrect)"

    # measure stats
    false_neg = all_pos - true_pos  # not detected but should have been
    sens, ppv, msg = atlas_stats.calc_sens_ppv(
        all_pos, true_pos, false_pos, false_neg)
    msg = "Detection stats {}:\n{}".format(maybe_msg, msg)
    return sens, ppv, msg


def _find_closest_blobs(blobs, blobs_master, tol):
    """Finds the closest matching blobs between two arrays. Each entry will 
    have no more than one match, and the total number of matches will be 
    the size of the shortest list.
    
    Args:
        blobs: The blobs to be checked for closeness, given as 2D 
            array of at least [n, [z, row, column, ...]].
        blobs_master: The list by which to check for close blobs, in the same
            format as blobs.
        tol: Tolerance to check for closeness, given in the same format
            as region. Blobs that are equal to or less than the the absolute
            difference for all corresponding parameters will be considered
            a potential match.
    
    Returns:
        close_master: Indices of blobs_master that are the closest match to
            the corresponding blobs in "blobs", in order of closest to farthest
            match.
        close: Indices of the corresponding blobs in the "blobs" array.
    """
    # NOTE: does not scale well with large numbers of blobs, presumably 
    # because of the need to check blobs individually with a continually 
    # increasing number of accepted blobs
    
    # TESTING: sorting changes blob matches; use find_closest_blobs_cdist 
    # to avoid this issue
    #blobs, sort = sort_blobs(blobs)
    
    close_master = []
    close = []
    # compare each element for differences, weighting based on tolerance; 
    # TODO: incorporate radius
    blobs_diffs_init = np.abs(
        blobs_master[:, :3][:, None] - blobs[:, :3])
    normalize_factor = np.divide(np.max(tol), tol)
    tol = np.multiply(tol, normalize_factor)
    far = np.max(tol) + 1
    #print("weighted tol: {}".format(tol))
    
    # matches limited by length of smallest list
    num_matches = min(len(blobs_master), len(blobs))
    for i in range(num_matches):
        # normalize the diffs
        blobs_diffs = np.multiply(blobs_diffs_init, normalize_factor)
        # sum to find smallest diff
        diffs_sums = np.sum(blobs_diffs, blobs_diffs.ndim - 1)
        #print("blobs_diffs:\n{}".format(blobs_diffs))
        #print("diffs_sums:\n{}".format(diffs_sums))
        for j in range(len(diffs_sums)):
            # iterate up to the number of master blobs
            
            # get indices of minimum differences
            min_master, min_blob = np.where(diffs_sums == diffs_sums.min())
            #print("diffs_sums: {}, min: {}".format(diffs_sums, diffs_sums.min()))
            found = False
            for k in range(len(min_master)):
                blob_master_closest = min_master[k]
                blob_closest = min_blob[k]
                diff = blobs_diffs[blob_master_closest, blob_closest]
                #print("min_master: {}, min_blob: {}".format(min_master, min_blob))
                #print("compare {} to {}".format(diff, tol))
                if (diff <= tol).all():
                    # only keep the match if within tolerance
                    close_master.append(blob_master_closest)
                    close.append(blob_closest)
                    # replace row/column corresponding to each picked blob with  
                    # distant values to ensure beyond tol
                    blobs_diffs_init[blob_master_closest, :, :] = far
                    blobs_diffs_init[:, blob_closest, :] = far
                    found = True
                    break
                elif (diff <= tol).any():
                    # closest value still to far, so set to distant 
                    # value to can check next min diff; 
                    # TODO: add far rather than simply assigning to it?
                    diffs_sums[blob_master_closest] = far
                else:
                    # no match if none of array dims within tolerance of min diff
                    break
            if found:
                break
    if config.verbose:
        # show sorted list of matches to compare between runs
        _match_blobs(
            blobs, blobs_master, close, close_master, np.zeros(len(blobs)))
    return np.array(close_master, dtype=int), np.array(close, dtype=int)


def _test_blob_verification(a, b, tol):
    # test verifying blobs by checking for closest matches within a tolerance
    print("test (b):\n{}".format(b))
    print("master (a):\n{}".format(a))
    #found_truth, detected = _find_closest_blobs(b, a, tol)
    #dists = np.zeros(len(blobs)
    detected, found_truth, dists = find_closest_blobs_cdist(b, a, tol)
    df_io.dict_to_data_frame(
        {"Testi": detected, "Masteri": found_truth, "Dist": dists}, show=True)


def _test_blob_autoverifier():
    _test_blob_close_sorted()
    a = np.array([[0, 1, 1], [1, 1, 1], [2, 1, 1]])
    b = np.array([[1, 1, 1], [2, 1, 1], [3, 1, 1], [4, 2, 0]])
    _test_blob_verification(a, b, 1)
    print()
    a = np.array([[24, 52, 346], [20, 55, 252]])
    b = np.array([[24, 54, 351]])
    _test_blob_verification(a, b, 6)


def _test_blob_close_sorted():
    a = np.ones((3, 3))
    a[:, 0] = [0, 1, 2]
    b = np.copy(a)
    b[:, 0] += 1
    print("test (b):\n{}".format(b))
    print("master (a):\n{}".format(a))
    blobs = np.concatenate((a, b))
    sort = np.lexsort((blobs[:, 2], blobs[:, 1], blobs[:, 0]))
    blobs = blobs[sort]
    blobs = detector.remove_close_blobs_within_sorted_array(blobs, (1, 2, 2))
    print("pruned:\n{}".format(blobs))


def _get_truth_db_rois(subimg_path_base, filename_base, db_path_base=None):
    """Get ROIs from a truth database.
    
    Args:
        subimg_path_base (str): Base path with sub-image.
        filename_base (str): Base path without sub-image to find the
            experiment, used only if an experiment cannot be found based on
            ``subimg_path_base``.
        db_path_base (str): Path to database to load; defaults to None
            to use :attr:`config.truth_db`.

    Returns:
        str, list[:class:`sqlite3.Row`]: Found experiment name and
        list of database ROI rows in that experiment, or None for each
        if the ROIs are not found.

    """
    name = None
    exp_rois = None
    if db_path_base:
        # load truth DB
        _logger.debug(
            "Loading truth db for verifications from '%s'", db_path_base)
        sqlite.load_truth_db(db_path_base)
    if config.truth_db is not None:
        # load experiment and ROIs from truth DB using the sub-image-based
        # name; series not included in exp name since in ROI
        name = sqlite.get_exp_name(subimg_path_base)
        _logger.debug("Loading truth ROIs from experiment '%s'", name)
        exp_rois = config.truth_db.get_rois(name)
        if exp_rois is None:
            # exp may have been named without sub-image
            old_name = name
            name = sqlite.get_exp_name(filename_base)
            _logger.debug(
                "'%s' experiment name not found, will try without any "
                "sub-image offset/size: '%s'", old_name, name)
            exp_rois = config.truth_db.get_rois(name)
        if exp_rois is None:
            # exp may differ from image name all together
            _logger.debug(
                "'%s' experiment name not found, will try first "
                "available experiment", name)
            exps = config.truth_db.select_experiment()
            if exps:
                name = exps[0]["name"]
                _logger.debug(
                    "Loading ROIs from first available experiment: '%s'", name)
                exp_rois = config.truth_db.get_rois(name)
    if not exp_rois:
        _logger.warn("No matching experiments found in the truth database")
    return name, exp_rois
