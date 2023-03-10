# Detect blobs within a chunked stack through multiprocessing
# Author: David Young, 2019, 2020
"""Stack blob detector.

Detect blobs within a stack that has been chunked to allow parallel 
processing.
"""
from enum import Enum
import os
from time import time
from typing import NamedTuple, Sequence, TYPE_CHECKING, Optional, Tuple

import numpy as np
import pandas as pd

from magmap.cv import chunking, classifier, colocalizer, detector, verifier
from magmap.io import cli, df_io, importer, libmag, naming
from magmap.plot import plot_3d
from magmap.settings import config, roi_prof

if TYPE_CHECKING:
    from magmap.settings import profiles

_logger = config.logger.getChild(__name__)


class StackTimes(Enum):
    """Stack processing durations."""
    DETECTION = "Detection"
    PRUNING = "Pruning"
    TOTAL = "Total_stack"


class StackDetector(object):
    """Detect blobs within a stack in a way that allows multiprocessing 
    without global variables.
    
    Attributes:
        img (:obj:`np.ndarray`): Full image array.
        last_coord (:obj:`np.ndarray`): Indices of last sub-ROI given as
            coordinates in z,y,x.
        denoise_max_shape (Tuple[int]): Maximum shape of each unit within
            each sub-ROI for denoising.
        exclude_border (bool): Sequence of border pixels in x,y,z to exclude;
            defaults to None.
        coloc (bool): True to perform blob co-localizations; defaults to False.
        channel (Sequence[int]): Sequence of channels; defaults to None to
            detect in all channels.
    """
    img = None
    last_coord = None
    denoise_max_shape = None
    exclude_border = None
    coloc = False
    channel = None
    
    @classmethod
    def detect_sub_roi_from_data(cls, coord, sub_roi_slices, offset):
        """Perform 3D blob detection within a sub-ROI using data stored
        as class attributes for forked multiprocessing.

        Args:
            coord (Tuple[int]): Coordinate of the sub-ROI in the order z,y,x.
            sub_roi_slices (Tuple[slice]): Sequence of slices within
                :attr:``img`` defining the sub-ROI.
            offset (Tuple[int]): Offset of the sub-ROI within the full ROI,
                in z,y,x.

        Returns:
            Tuple[int], :obj:`np.ndarray`: The coordinate given back again to
            identify the sub-ROI position and an array of detected blobs.

        """
        return cls.detect_sub_roi(
            coord, offset, cls.last_coord,
            cls.denoise_max_shape, cls.exclude_border, cls.img[sub_roi_slices],
            cls.channel, coloc=cls.coloc)

    @classmethod
    def detect_sub_roi(
            cls, coord: Sequence[int], offset: Sequence[int],
            last_coord: Sequence[int], denoise_max_shape: Sequence[int],
            exclude_border: bool, sub_roi: np.ndarray, channel: Sequence[int],
            img_path: Optional[str] = None, coloc: bool = False
    ) -> Tuple[Sequence[int], np.ndarray]:
        """Perform 3D blob detection within a sub-ROI without accessing
        class attributes, such as for spawned multiprocessing.
        
        Args:
            coord: Coordinate of the sub-ROI in the order z,y,x.
            offset: Offset of the sub-ROI within the full ROI, in z,y,x.
            last_coord: See attributes.
            denoise_max_shape: See attributes.
            exclude_border: See attributes.
            sub_roi: Array in which to perform detections.
            img_path: Path from which to load metadatat; defaults to None.
                If given, the command line arguments will be reloaded to
                set up the image and processing parameters.
            coloc: True to perform blob co-localizations; defaults to False.
            channel: Sequence of channels, where None detects in all channels.
        
        Returns:
            Tuple of:
            - ``coord``: the coordinate given back again to identify the
              sub-ROI position
            - ``blobs``: array of detected blobs

        """
        if img_path:
            # reload command-line parameters and image metadata, which is
            # required if run from a spawned (not forked) process
            cli.process_cli_args()
            _, orig_info = importer.make_filenames(img_path)
            importer.load_metadata(orig_info)
        _logger.debug(
            "Detecting blobs in sub-ROI at %s of %s", coord, last_coord)
        
        if denoise_max_shape is not None:
            # further split sub-ROI for preprocessing locally
            denoise_roi_slices, _ = chunking.stack_splitter(
                sub_roi.shape, denoise_max_shape)
            for z in range(denoise_roi_slices.shape[0]):
                for y in range(denoise_roi_slices.shape[1]):
                    for x in range(denoise_roi_slices.shape[2]):
                        # extract sub-sub-ROI
                        denoise_coord = (z, y, x)
                        denoise_roi = sub_roi[denoise_roi_slices[denoise_coord]]
                        _logger.debug(
                            f"Preprocessing sub-sub-ROI {denoise_coord} of "
                            f"{np.subtract(denoise_roi_slices.shape, 1)}")
                        
                        # preprocess all channels, even those not detected in
                        # case they are used for spectral unmixing
                        denoise_roi = plot_3d.saturate_roi(denoise_roi)
                        denoise_roi = plot_3d.denoise_roi(denoise_roi)
                        
                        # replace slices with denoised ROI
                        denoise_roi_slices[denoise_coord] = denoise_roi
            
            # re-merge back into the sub-ROI
            merged_shape = chunking.get_split_stack_total_shape(
                denoise_roi_slices)
            merged = np.zeros(
                tuple(merged_shape), dtype=denoise_roi_slices[0, 0, 0].dtype)
            chunking.merge_split_stack2(denoise_roi_slices, None, 0, merged)
            sub_roi = merged
        
        if exclude_border is None:
            exclude = None
        else:
            exclude = np.array([exclude_border, exclude_border])
            exclude[0, np.equal(coord, 0)] = 0
            exclude[1, np.equal(coord, last_coord)] = 0
        segments = detector.detect_blobs(sub_roi, channel, exclude)
        if coloc and segments is not None:
            # co-localize blobs and append to blobs array
            colocs = colocalizer.colocalize_blobs(sub_roi, segments)
            segments = np.hstack((segments, colocs))
        #print("segs before (offset: {}):\n{}".format(offset, segments))
        if segments is not None:
            # shift both coordinate sets (at beginning and end of array) to
            # absolute positioning, using the latter set to store shifted
            # coordinates based on duplicates and the former for initial
            # positions to check for multiple duplicates
            detector.Blobs.shift_blob_rel_coords(segments, offset)
            detector.Blobs.shift_blob_abs_coords(segments, offset)
            #print("segs after:\n{}".format(segments))
        return coord, segments
    
    @classmethod
    def detect_blobs_sub_rois(cls, img, sub_roi_slices, sub_rois_offsets,
                              denoise_max_shape, exclude_border, coloc,
                              channel):
        """Process blobs in chunked sub-ROIs via multiprocessing.

        Args:
            img (:obj:`np.ndarray`): Array in which to detect blobs.
            sub_roi_slices (:obj:`np.ndarray`): Numpy object array containing
                chunked sub-ROIs within a stack.
            sub_rois_offsets (:obj:`np.ndarray`): Numpy object array of the same
                shape as ``sub_rois`` with offsets in z,y,x corresponding to
                each sub-ROI. Offets are used to transpose blobs into
                absolute coordinates.
            denoise_max_shape (Tuple[int]): Maximum shape of each unit within
                each sub-ROI for denoising.
            exclude_border (Tuple[int]): Sequence of border pixels in x,y,z to
                exclude; defaults to None.
            coloc (bool): True to perform blob co-localizations; defaults to
                False.
            channel (Sequence[int]): Sequence of channels, where None detects
                in all channels.

        Returns:
            :obj:`np.ndarray`: Numpy object array of blobs corresponding to
            ``sub_rois``, with each set of blobs given as a Numpy array in the
            format, ``[n, [z, row, column, radius, ...]]``, including additional
            elements as given in :meth:``StackDetect.detect_sub_roi``.
        
        """
        # detect nuclei in each sub-ROI, passing an index to access each
        # sub-ROI to minimize pickling
        is_fork = chunking.is_fork()
        last_coord = np.subtract(sub_roi_slices.shape, 1)
        if is_fork:
            # set data as class attributes for direct access during forked
            # multiprocessing
            cls.img = img
            cls.last_coord = last_coord
            cls.denoise_max_shape = denoise_max_shape
            cls.exclude_border = exclude_border
            cls.coloc = coloc
            cls.channel = channel
        
        _logger.info("Starting blob detections for channel(s) %s with "
                     "co-localizations set to %s", channel, coloc)
        pool = chunking.get_mp_pool()
        pool_results = []
        for z in range(sub_roi_slices.shape[0]):
            for y in range(sub_roi_slices.shape[1]):
                for x in range(sub_roi_slices.shape[2]):
                    coord = (z, y, x)
                    if is_fork:
                        # use variables stored in class
                        pool_results.append(pool.apply_async(
                            StackDetector.detect_sub_roi_from_data,
                            args=(coord, sub_roi_slices[coord],
                                  sub_rois_offsets[coord])))
                    else:
                        # pickle full set of variables including sub-ROI and
                        # filename from which to load image parameters
                        pool_results.append(pool.apply_async(
                            StackDetector.detect_sub_roi,
                            args=(coord, sub_rois_offsets[coord], last_coord,
                                  denoise_max_shape, exclude_border,
                                  img[sub_roi_slices[coord]], channel,
                                  config.filename, coloc)))
    
        # retrieve blobs and assign to object array corresponding to sub_rois
        seg_rois = np.zeros(sub_roi_slices.shape, dtype=object)
        for result in pool_results:
            coord, segments = result.get()
            num_blobs = 0 if segments is None else len(segments)
            _logger.info(
                "Adding %s blobs from sub_roi at %s of %s",
                num_blobs, coord, np.add(sub_roi_slices.shape, -1))
            seg_rois[coord] = segments
    
        pool.close()
        pool.join()
        return seg_rois


class Blocks(NamedTuple):
    """Block processing chunk tuples."""
    #: Object array of tuples containing slices of each block.
    sub_roi_slices: np.ndarray
    # Similar Numpy array but with tuples of offsets.
    sub_rois_offsets: np.ndarray
    #: Int array of max shape for each sub-block used for denoising.
    denoise_max_shape: np.ndarray
    #: List of ints for border pixels to exclude in ``z, y, x``.
    exclude_border: Sequence[int]
    #: match tolerance as a Numpy float array in ``z, y, x``.
    tol: np.ndarray
    #: Float array of overlapping pixels in ``z, y, x``.
    overlap_base: np.ndarray
    #: Similar overlap array but modified by the border exclusion.
    overlap: np.ndarray
    #: Similar overlap array but for padding beyond the overlap.
    overlap_padding: np.ndarray
    #: Max pixels for each side in ``z, y, x`` order.
    max_pixels: np.ndarray


def setup_blocks(
        settings: "profiles.SettingsDict", shape: Sequence[int]) -> Blocks:
    """Set up blocks for block processing
    
    Each block is a chunk of a larger image processed sequentially or in
    parallel to optimize resource usage.
    
    Args:
        settings: Settings dictionary that defines that the blocks.
        shape: Shape of full image in ``z, y, x``.

    Returns:
        A named tuple of the block parameters.
        
    """
    scaling_factor = detector.calc_scaling_factor()
    print("microsope scaling factor based on resolutions: {}"
          .format(scaling_factor))
    denoise_size = settings["denoise_size"]
    denoise_max_shape = None
    if denoise_size:
        # further subdivide each sub-ROI for local preprocessing
        denoise_max_shape = np.ceil(
            np.multiply(scaling_factor, denoise_size)).astype(int)

    # overlap sub-ROIs to minimize edge effects
    overlap_base = detector.calc_overlap()
    tol = np.multiply(overlap_base, settings["prune_tol_factor"]).astype(int)
    overlap_padding = np.copy(tol)
    overlap = np.copy(overlap_base)
    exclude_border = settings["exclude_border"]
    if exclude_border is not None:
        # exclude border to avoid blob detector edge effects, where blobs
        # often collect at the faces of the sub-ROI;
        # ensure that overlap is greater than twice the border exclusion per
        # axis so that no plane will be excluded from both overlapping sub-ROIs
        exclude_border_thresh = np.multiply(2, exclude_border)
        overlap_less = np.less(overlap, exclude_border_thresh)
        overlap[overlap_less] = exclude_border_thresh[overlap_less]
        excluded = np.greater(exclude_border, 0)
        overlap[excluded] += 1  # additional padding
        overlap_padding[excluded] = 0  # no need to prune past excluded border
    print("sub-ROI overlap: {}, pruning tolerance: {}, padding beyond "
          "overlap for pruning: {}, exclude borders: {}"
          .format(overlap, tol, overlap_padding, exclude_border))
    max_pixels = np.ceil(np.multiply(
        scaling_factor, settings["segment_size"])).astype(int)
    print("preprocessing max shape: {}, detection max pixels: {}"
          .format(denoise_max_shape, max_pixels))
    sub_roi_slices, sub_rois_offsets = chunking.stack_splitter(
        shape, max_pixels, overlap)
    return Blocks(
        sub_roi_slices, sub_rois_offsets, denoise_max_shape,
        exclude_border, tol, overlap_base, overlap, overlap_padding, max_pixels)


def detect_blobs_blocks(
        filename_base: str, image5d: np.ndarray,
        offset: Optional[Sequence[int]] = None,
        size: Optional[Sequence[int]] = None,
        channels: Optional[Sequence[int]] = None, verify: bool = False,
        save_dfs: bool = True, full_roi: bool = False, coloc: bool = False
) -> Tuple[Tuple[int, int, int], str, "detector.Blobs"]:
    """Detect blobs by block processing of a large image.
    
    All channels are processed in the same blocks.
    
    Args:
        filename_base: Base path to use file output.
        image5d: Large image to process as a Numpy array of t,z,y,x,[c]
        offset: Sub-image offset given as coordinates in z,y,x.
        size: Sub-image shape given in z,y,x.
        channels: Sequence of channels, where None detects in all channels.
        verify: True to verify detections against truth database; defaults
            to False.
        save_dfs: True to save data frames to file; defaults to True.
        full_roi: True to treat ``image5d`` as the full ROI; defaults to False.
        coloc: True to perform blob co-localizations; defaults to False.
    
    Returns:
        Tuple of:
        - ``stats_detection``: accuracy metrics from
          :class:`magmap.cv.detector.verify_rois`
        - ``fdbk``: feedback message from this same function
        - ``blobs``: detected blobs
    
    """
    time_start = time()
    subimg_path_base = filename_base
    if size is None or offset is None:
        # uses the entire stack if no size or offset specified
        size = image5d.shape[1:4]
        offset = (0, 0, 0)
    else:
        # get base path for sub-image
        subimg_path_base = naming.make_subimage_name(
            filename_base, offset, size)
    filename_blobs = libmag.combine_paths(subimg_path_base, config.SUFFIX_BLOBS)
    
    # get ROI for given region, including all channels
    if full_roi:
        # treat the full image as the ROI
        roi = image5d[0]
    else:
        roi = plot_3d.prepare_subimg(image5d, offset, size)
    num_chls_roi = 1 if len(roi.shape) < 4 else roi.shape[3]
    if num_chls_roi < 2:
        coloc = False
        _logger.info("Unable to co-localize as image has only 1 channel")
    
    # prep chunking ROI into sub-ROIs with size based on segment_size, scaling
    # by physical units to make more independent of resolution; use profile
    # from first channel to be processed for block settings
    time_detection_start = time()
    settings = config.get_roi_profile(channels[0])
    _logger.info("Profile for block settings: %s", settings[settings.NAME_KEY])
    blocks = setup_blocks(settings, roi.shape)
    
    # TODO: option to distribute groups of sub-ROIs to different servers
    # for blob detection
    seg_rois = StackDetector.detect_blobs_sub_rois(
        roi, blocks.sub_roi_slices, blocks.sub_rois_offsets,
        blocks.denoise_max_shape, blocks.exclude_border, coloc, channels)
    detection_time = time() - time_detection_start
    _logger.info("Blob detection time (s): %s", detection_time)
    
    # prune blobs in overlapping portions of sub-ROIs
    time_pruning_start = time()
    segments_all, df_pruning = StackPruner.prune_blobs_mp(
        roi, seg_rois, blocks.overlap, blocks.tol, blocks.sub_roi_slices,
        blocks.sub_rois_offsets, channels, blocks.overlap_padding)
    pruning_time = time() - time_pruning_start
    _logger.info("Blob pruning time (s): %s", pruning_time)
    #print("maxes:", np.amax(segments_all, axis=0))
    
    # get weighted mean of ratios
    if df_pruning is not None:
        _logger.info("\nBlob pruning ratios:")
        path_pruning = "blob_ratios.csv" if save_dfs else None
        df_pruning_all = df_io.data_frames_to_csv(
            df_pruning, path_pruning, show=" ")
        cols = df_pruning_all.columns.tolist()
        blob_pruning_means = {}
        if "blobs" in cols:
            blobs_unpruned = df_pruning_all["blobs"]
            num_blobs_unpruned = np.sum(blobs_unpruned)
            for col in cols[1:]:
                blob_pruning_means["mean_{}".format(col)] = [
                    np.sum(np.multiply(df_pruning_all[col], blobs_unpruned))
                    / num_blobs_unpruned]
            path_pruning_means = "blob_ratios_means.csv" if save_dfs else None
            df_pruning_means = df_io.dict_to_data_frame(
                blob_pruning_means, path_pruning_means, show=" ")
        else:
            _logger.info("No blob ratios found")
    
    '''# report any remaining duplicates
    np.set_printoptions(linewidth=500, threshold=10000000)
    print("all blobs (len {}):".format(len(segments_all)))
    sort = np.lexsort(
        (segments_all[:, 2], segments_all[:, 1], segments_all[:, 0]))
    blobs = segments_all[sort]
    print(blobs)
    print("checking for duplicates in all:")
    print(detector.remove_duplicate_blobs(blobs, slice(0, 3)))
    '''
    
    stats_detection = None
    fdbk = None
    colocs = None
    blobs = detector.Blobs(segments_all, path=filename_blobs)
    if segments_all is not None:
        # remove the duplicated elements that were used for pruning
        blobs.replace_rel_with_abs_blob_coords(segments_all)
        if coloc:
            colocs = segments_all[:, 10:10+num_chls_roi].astype(np.uint8)
        # remove absolute coordinate and any co-localization columns
        segments_all = blobs.remove_abs_blob_coords(segments_all, True)
        
        # compare detected blobs with truth blobs
        # TODO: assumes ground truth is relative to any ROI offset,
        #   but should make customizable
        if verify:
            stats_detection, fdbk = verifier.verify_stack(
                filename_base, subimg_path_base, settings, segments_all,
                channels, blocks.overlap_base)
    
    if config.save_subimg:
        subimg_base_path = libmag.combine_paths(
            subimg_path_base, config.SUFFIX_SUBIMG)
        if (isinstance(config.image5d, np.memmap) and
                config.image5d.filename == os.path.abspath(subimg_base_path)):
            # file at sub-image save path may have been opened as a memmap
            # file, in which case saving would fail
            libmag.warn("{} is currently open, cannot save sub-image"
                        .format(subimg_base_path))
        else:
            # write sub-image, which is in ROI (3D) format
            with open(subimg_base_path, "wb") as f:
                np.save(f, roi)

    # update blobs array and add metadata
    # TODO: consider separating into blobs and blobs metadata archives
    blobs.blobs = segments_all
    blobs.colocalizations = colocs
    blobs.resolutions = config.resolutions
    blobs.basename = os.path.basename(config.filename)
    blobs.roi_offset = offset
    blobs.roi_size = size
    
    # whole image benchmarking time
    times = (
        [detection_time],
        [pruning_time],
        time() - time_start)
    times_dict = {}
    for key, val in zip(StackTimes, times):
        times_dict[key] = val
    if segments_all is None:
        _logger.info("\nNo blobs detected")
    else:
        _logger.info("\nTotal blobs found: %s", len(segments_all))
        blobs.show_blobs_per_channel(segments_all)
    _logger.info("\nTotal detection processing times (s):")
    path_times = "stack_detection_times.csv" if save_dfs else None
    df_io.dict_to_data_frame(times_dict, path_times, show=" ")
    
    return stats_detection, fdbk, blobs


def detect_blobs_stack(
        filename_base: str, subimg_offset: Optional[Sequence[int]] = None,
        subimg_size: Optional[Sequence[int]] = None, coloc: bool = False
) -> Tuple[Tuple[int, int, int], str, "detector.Blobs"]:
    """Detect blobs in a full stack, such as a whole large image.
    
    Process channels in separate sets of blocks if their profiles specify
    different block sizes.
    
    Args:
        filename_base: Base image filename, for saving files.
        subimg_offset: Sub-image offset as ``z,y,x`` to load
            from :attr:`config.image5d`; defaults to None.
        subimg_size: Sub-image size as ``z,y,x`` to load
            from :attr:`config.image5d`; defaults to None.
        coloc: True to also detect blob-colocalizations based on image
            intensity; defaults to False. For match-based colocalizations,
            use the ``coloc_match`` task
            (:meth:`magmap.colocalizer.StackColocalizer.colocalize_stack`)
            instead.
    
    Returns:
        Tuple of:
        - ``stats_detection``: combined accuracy metrics from
          :class:`magmap.cv.detector.verify_rois`
        - ``fdbk``: feedback message from this same function
        - ``blobs``: detected blobs across all channels in
          :attr:`magmap.settings.config.channel`

    """
    channels = plot_3d.setup_channels(config.image5d, config.channel, 4)[1]
    if roi_prof.ROIProfile.is_identical_settings(
            [config.get_roi_profile(c) for c in channels],
            roi_prof.ROIProfile.BLOCK_SIZES):
        _logger.info("Will process channels together in the same blocks")
        # combine channels into nested list
        channels = [channels]
    else:
        _logger.info(
            "Will process channels in separate blocks defined by profiles")
    
    cols = ("stats", "fdbk", "blobs")
    detection_out = {}
    for chl in channels:
        # detect blobs in each channel separately unless all channels
        # are combined in a single list
        if not libmag.is_seq(chl):
            chl = [chl]
        blobs_out = detect_blobs_blocks(
            filename_base, config.image5d, subimg_offset, subimg_size,
            chl, config.truth_db_mode is config.TruthDBModes.VERIFY,
            not config.grid_search_profile, config.image5d_is_roi, coloc)
        for col, val in zip(cols, blobs_out):
            detection_out.setdefault(col, []).append(val)
        _logger.info(f"\n{'-' * 80}")
    
    stats = None
    fdbk = None
    blobs_all = None
    if "blobs" in detection_out and detection_out["blobs"]:
        # join blobs and colocalizations from all channels and save archive
        blobs_all = detection_out["blobs"][0]
        blobs_all.blobs = libmag.combine_arrs(
            [b.blobs for b in detection_out["blobs"]
             if b.blobs is not None])
        if blobs_all.blobs is None:
            _logger.info("No blobs found across channels")
        else:
            _logger.info("Total blobs found across channels: %s",
                         len(blobs_all.blobs))
            detector.Blobs.show_blobs_per_channel(blobs_all.blobs)
        blobs_all.colocalizations = libmag.combine_arrs(
            [b.colocalizations for b in detection_out["blobs"]
             if b.colocalizations is not None])

        if channels:
            try:
                # classify blobs if model is set in config
                classifier.ClassifyImage.classify_whole_image(
                    channels=channels[0], blobs=blobs_all)
            except FileNotFoundError as e:
                _logger.debug(e)
        
        blobs_all.save_archive()
        _logger.info("\n")
        
        # combine verification stats and feedback messages
        stats = libmag.combine_arrs(
            detection_out["stats"], fn=np.sum, axis=0)
        fdbk = "\n".join(
            [f for f in detection_out["fdbk"] if f is not None])
    return stats, fdbk, blobs_all


class StackPruner(object):
    """Prune blobs within a stack in a way that allows multiprocessing 
    without global variables.
    
    Attributes:
        blobs_to_prune: List of tuples to be passed to 
            :meth:``detector.remove_close_blobs``. The final colums should
            have the coordinates of the sub-ROI in ``z, y, x`` order as
            given by :meth:`chunking.merge_blobs`.
    
    """
    blobs_to_prune = None
    
    @classmethod
    def prune_overlap_by_index(cls, i):
        """Prune an overlapping region.
        
        Args:
            i (int): Index in :attr:``blobs_to_prune``.
        
        Returns:
            The results from :meth:`prune_overlap`.
        """
        return cls.prune_overlap(i, cls.blobs_to_prune[i])

    @classmethod
    def prune_overlap(cls, i, pruner):
        """Prune overlapping blobs.
        
        Args:
            i (int): Index of :attr:``blobs_to_prune``.
            pruner: Corresponding value from :attr:``blobs_to_prune``.

        Returns:
            :obj:`np.ndarray`, tuple: Blobs after pruning and pruning ratio
            metrics.

        """
        blobs, axis, tol, blobs_next = pruner
        #print("orig blobs in axis {}, i {}\n{}".format(axis, i, blobs))
        if blobs is None: return None, None
        
        # assume that final columns hold sub-ROI coordinates in z,y,x order
        axis_col = blobs.shape[1] - 3 + axis
        num_blobs_orig = len(blobs)
        print("num_blobs_orig in axis {}, {}: {}"
              .format(axis, i, num_blobs_orig))
        blobs_master = blobs[blobs[:, axis_col] == i]
        blobs = blobs[blobs[:, axis_col] == i + 1]
        #print("blobs_master in axis {}, i {}\n{}".format(axis, i, blobs_master))
        #print("blobs to check in axis {}, next i ({})\n{}".format(axis, i + 1, blobs))
        pruned, blobs_master = detector.remove_close_blobs(
            blobs, blobs_master, tol)
        blobs_after_pruning = np.concatenate((blobs_master, pruned))
        #blobs_after_pruning = detector.remove_close_blobs_within_sorted_array(blobs, tol)
        pruning_ratios = None
        if blobs_next is not None:
            pruning_ratios = detector.meas_pruning_ratio(
                num_blobs_orig, len(blobs_after_pruning), len(blobs_next))
        return blobs_after_pruning, pruning_ratios
    
    @classmethod
    def prune_blobs_mp(cls, img, seg_rois, overlap, tol, sub_roi_slices,
                       sub_rois_offsets, channels, overlap_padding=None):
        """Prune close blobs within overlapping regions by checking within
        entire planes across the ROI in parallel with multiprocessing.
        
        Args:
            img (:obj:`np.ndarray`): Array in which to detect blobs.
            seg_rois (:obj:`np.ndarray`): Blobs from each sub-region.
            overlap: 1D array of size 3 with the number of overlapping pixels 
                for each image axis.
            tol: Tolerance as (z, y, x), within which a segment will be 
                considered a duplicate of a segment in the master array and
                removed.
            sub_roi_slices (:obj:`np.ndarray`): Object array of ub-regions, used
                to check size.
            sub_rois_offsets: Offsets of each sub-region.
            channels (Sequence[int]): Sequence of channels; defaults to None
                to detect in all channels.
            overlap_padding: Sequence of z,y,x for additional padding beyond
                ``overlap``. Defaults to None to use ``tol`` as padding.
        
        Returns:
            :obj:`np.ndarray`, :obj:`pd.DataFrame`: All blobs as a Numpy array
            and a data frame of pruning stats, or None for both if no blobs are
            in the ``seg_rois``.
        
        """
        # collect all blobs in master array to group all overlapping regions,
        # with sub-ROI coordinates as last 3 columns
        blobs_merged = chunking.merge_blobs(seg_rois)
        if blobs_merged is None:
            return None, None
        print("total blobs before pruning:", len(blobs_merged))
        
        print("pruning with overlap: {}, overlap tol: {}, pruning tol: {}"
              .format(overlap, overlap_padding, tol))
        blobs_all = []
        blob_ratios = {}
        cols = ("blobs", "ratio_pruning", "ratio_adjacent")
        if overlap_padding is None: overlap_padding = tol
        for i in channels:
            # prune blobs from each channel separately to avoid pruning based on 
            # co-localized channel signals
            blobs = detector.Blobs.blobs_in_channel(blobs_merged, i)
            for axis in range(3):
                # prune planes with all the overlapping regions within a given axis,
                # skipping if this axis has no overlapping sub-regions
                num_sections = sub_rois_offsets.shape[axis]
                if num_sections <= 1:
                    continue
                
                # multiprocess pruning by overlapping planes
                blobs_all_non_ol = None # all blobs from non-overlapping regions
                blobs_to_prune = []
                coord_last = tuple(np.subtract(sub_roi_slices.shape, 1))
                for j in range(num_sections):
                    # build overlapping region dimensions based on size of 
                    # sub-region in the given axis
                    coord = np.zeros(3, dtype=int)
                    coord[axis] = j
                    print("** setting up blob pruning in axis {}, section {} "
                          "of {}".format(axis, j, num_sections - 1))
                    offset = sub_rois_offsets[tuple(coord)]
                    sub_roi = img[sub_roi_slices[tuple(coord)]]
                    size = sub_roi.shape
                    _logger.debug(f"offset: {offset}, size: {size}")
                    
                    # overlapping region: each region but the last extends 
                    # into the next region, with the overlapping volume from 
                    # the end of the region, minus the overlap and a tolerance 
                    # space, to the region's end plus this tolerance space; 
                    # non-overlapping region: the region before the overlap, 
                    # after any overlap with the prior region (n > 1) 
                    # to the start of the overlap (n < last region)
                    blobs_ol = None
                    blobs_ol_next = None
                    blobs_in_non_ol = []
                    shift = overlap[axis] + overlap_padding[axis]
                    offset_axis = offset[axis]
                    if j < num_sections - 1:
                        bounds = [offset_axis + size[axis] - shift,
                                  offset_axis + size[axis]
                                  + overlap_padding[axis]]
                        libmag.printv(
                            "axis {}, boundaries: {}".format(axis, bounds))
                        blobs_ol = blobs[np.all([
                            blobs[:, axis] >= bounds[0], 
                            blobs[:, axis] < bounds[1]], axis=0)]
                        
                        # get blobs from immediatley adjacent region of the same 
                        # size as that of the overlapping region; keep same 
                        # starting point with or without overlap_tol
                        start = offset_axis + size[axis] + tol[axis]
                        bounds_next = [
                            start,
                            start + overlap[axis] + 2 * overlap_padding[axis]]
                        shape = np.add(
                            sub_rois_offsets[coord_last], sub_roi.shape[:3])
                        libmag.printv(
                            "axis {}, boundaries (next): {}, max bounds: {}"
                            .format(axis, bounds_next, shape[axis]))
                        if np.all(np.less(bounds_next, shape[axis])):
                            # ensure that next overlapping region is within ROI
                            blobs_ol_next = blobs[np.all([
                                blobs[:, axis] >= bounds_next[0], 
                                blobs[:, axis] < bounds_next[1]], axis=0)]
                        # non-overlapping region extends up this overlap
                        blobs_in_non_ol.append(blobs[:, axis] < bounds[0])
                    else:
                        # last non-overlapping region extends to end of region
                        blobs_in_non_ol.append(
                            blobs[:, axis] < offset_axis + size[axis])
                    
                    # get non-overlapping area
                    start = offset_axis
                    if j > 0:
                        # shift past overlapping part at start of region
                        start += shift
                    blobs_in_non_ol.append(blobs[:, axis] >= start)
                    blobs_non_ol = blobs[np.all(blobs_in_non_ol, axis=0)]
                    # collect all non-overlapping region blobs
                    if blobs_all_non_ol is None:
                        blobs_all_non_ol = blobs_non_ol
                    elif blobs_non_ol is not None:
                        blobs_all_non_ol = np.concatenate(
                            (blobs_all_non_ol, blobs_non_ol))
    
                    blobs_to_prune.append((blobs_ol, axis, tol, blobs_ol_next))
    
                is_fork = chunking.is_fork()
                if is_fork:
                    # set data as class variables to share across forks
                    cls.blobs_to_prune = blobs_to_prune
                pool = chunking.get_mp_pool()
                pool_results = []
                for j in range(len(blobs_to_prune)):
                    if is_fork:
                        # prune blobs in overlapping regions by multiprocessing,
                        # using a separate class to avoid pickling input blobs
                        pool_results.append(pool.apply_async(
                            StackPruner.prune_overlap_by_index, args=(j, )))
                    else:
                        # for spawned methods, need to pickle the blobs
                        pool_results.append(pool.apply_async(
                            StackPruner.prune_overlap, args=(
                                j, blobs_to_prune[j])))
                
                # collect all the pruned blob lists
                blobs_all_ol = None
                for result in pool_results:
                    blobs_ol_pruned, ratios = result.get()
                    if blobs_all_ol is None:
                        blobs_all_ol = blobs_ol_pruned
                    elif blobs_ol_pruned is not None:
                        blobs_all_ol = np.concatenate(
                            (blobs_all_ol, blobs_ol_pruned))
                    if ratios:
                        for col, val in zip(cols, ratios):
                            blob_ratios.setdefault(col, []).append(val)
                
                # recombine blobs from the non-overlapping with the pruned  
                # overlapping regions from the entire stack to re-prune along 
                # any remaining axes
                pool.close()
                pool.join()
                if blobs_all_ol is None:
                    blobs = blobs_all_non_ol
                elif blobs_all_non_ol is None:
                    blobs = blobs_all_ol
                else:
                    blobs = np.concatenate((blobs_all_non_ol, blobs_all_ol))
            # build up list from each channel
            blobs_all.append(blobs)
        
        # merge blobs into Numpy array and remove sub-ROI coordinate columns
        blobs_all = np.vstack(blobs_all)[:, :-3]
        print("total blobs after pruning:", len(blobs_all))
        
        # export blob ratios as data frame
        df = pd.DataFrame(blob_ratios)
        
        return blobs_all, df
