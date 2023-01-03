# Region labels export to data frames and CSV files
# Author: David Young, 2019
"""Region labels export to data frames and CSV files.

Convert regions from ontology files or atlases to data frames.
"""
import itertools
import os
import csv
from collections import OrderedDict
from time import time
from typing import Dict, Optional, Sequence, Tuple

import SimpleITK as sitk
import numpy as np
import pandas as pd

from magmap.io import libmag
from magmap.io import np_io
from magmap.atlas import ontology
from magmap.cv import chunking, colocalizer, cv_nd, detector
from magmap.io import df_io, sitk_io, sqlite
from magmap.plot import colormaps
from magmap.settings import config, atlas_prof
from magmap.stats import vols

_logger = config.logger.getChild(__name__)


def export_region_ids(labels_ref_lookup, path, level=None,
                      drawn_labels_only=False):
    """Export region IDs from annotation reference reverse mapped dictionary 
    to CSV and Excel files.

    Use a ``level`` of None to export labels only for the currently loaded
    atlas. The RGB values used for the currently loaded atlas will also be
    shown, with cell colors corresponding to these values in the Excel file.
    
    Args:
        labels_ref_lookup: The labels reference lookup, assumed to be an 
            OrderedDict generated by :func:`ontology.create_reverse_lookup` 
            to look up by ID while preserving key order to ensure that 
            parents of any child will be reached prior to the child.
        path: Path to output CSV file; if does not end with ``.csv``, it will 
            be added.
        level: Level at which to find parent for each label; defaults to None
            to get the immediate parent.
        drawn_labels_only (bool): True to export only the drawn labels for
            atlas labels in the same folder as ``labels_ref_lookup``.
            Defaults to False to use the full set of labels in
            ``labels_ref_lookup``
    
    Returns:
        Pandas data frame of the region IDs and corresponding names.
    """
    def color_cells(s):
        # convert RGB to hex values since Pandas Excel export only supports
        # named colors or hex (as of v0.22)
        css = ["background-color: #{:02x}{:02x}{:02x}".format(*c) for c in s]
        return css

    ext = ".csv"
    path_csv = path if path.endswith(ext) else path + ext
    
    # find ancestor for each label at the given level
    label_parents = ontology.labels_to_parent(
        labels_ref_lookup, level, allow_parent_same_level=True)
    
    cols = [config.AtlasMetrics.REGION.value,
            config.AtlasMetrics.REGION_ABBR.value,
            config.AtlasMetrics.REGION_NAME.value,
            config.AtlasMetrics.LEVEL.value,
            config.AtlasMetrics.PARENT.value]
    data = OrderedDict()
    label_ids = sitk_io.find_atlas_labels(
        config.load_labels, drawn_labels_only, labels_ref_lookup)
    cm = colormaps.get_labels_discrete_colormap(None, 0, use_orig_labels=True)
    rgbs = cm.cmap_labels
    if rgbs is not None:
        cols.append("RGB")
    for i, key in enumerate(label_ids):
        # get label dict
        label = labels_ref_lookup.get(key)
        if label is None: continue
        
        # ID of parent at label_parents' level
        parent = label_parents[key]
        vals = [key, label[ontology.NODE][config.ABAKeys.ACRONYM.value],
                label[ontology.NODE][config.ABAKeys.NAME.value],
                label[ontology.NODE][config.ABAKeys.LEVEL.value], parent]
        if rgbs is not None:
            vals.append(rgbs[i, :3])
        for col, val in zip(cols, vals):
            data.setdefault(col, []).append(val)
    df = df_io.dict_to_data_frame(data, path_csv)
    if rgbs is not None:
        df = df.style.apply(color_cells, subset="RGB")
    path_xlsx = "{}.xlsx".format(os.path.splitext(path)[0])
    df.to_excel(path_xlsx)
    print("exported regions to styled spreadsheet: \"{}\"".format(path_xlsx))
    return df


def export_region_network(labels_ref_lookup, path):
    """Export region network file showing relationships among regions 
    according to the SIF specification.
    
    See http://manual.cytoscape.org/en/stable/Supported_Network_File_Formats.html#sif-format
    for file format information.
    
    Args:
        labels_ref_lookup: The labels reference lookup, assumed to be an 
            OrderedDict generated by :func:`ontology.create_reverse_lookup` 
            to look up by ID while preserving key order to ensure that 
            parents of any child will be reached prior to the child.
        path: Path to output SIF file; if does not end with ``.sif``, it will 
            be added.
    """
    ext = ".sif"
    if not path.endswith(ext): path += ext
    network = {}
    for key in labels_ref_lookup.keys():
        if key < 0: continue  # only use original, non-neg region IDs
        label = labels_ref_lookup[key]
        parents = label.get(ontology.PARENT_IDS)
        if parents:
            for parent in parents[::-1]:
                # work backward since closest parent listed last
                #print("{} looking for parent {} in network".format(key, parent))
                network_parent = network.get(parent)
                if network_parent is not None:
                    # assume that all parents will have already been entered 
                    # into the network dict since the keys were entered in 
                    # hierarchical order and maintain their order of entry
                    network_parent.append(key)
                    break
        # all regions have a node, even if connected to no one
        network[key] = []
    
    with open(path, "w", newline="") as csv_file:
        stats_writer = csv.writer(csv_file, delimiter=" ")
        # each region will have a line along with any of its immediate children
        for key in network.keys():
            children = network[key]
            row = [str(key)]
            if children:
                row.extend(["pp", *children])
            stats_writer.writerow(row)
    print("exported region network: \"{}\"".format(path))


def export_common_labels(img_paths, output_path):
    """Export data frame combining all label IDs from the given atlases, 
    showing the presence of labels in each atlas.
    
    Args:
        img_paths: Image paths from which to load the corresponding 
            labels images.
        output_path: Path to export data frame to .csv.
    
    Returns:
        Data frame with label IDs as indices, column for each atlas, and 
        cells where 1 indicates that the given atlas has the corresponding 
        label.
    """
    labels_dict = {}
    for img_path in img_paths:
        name = libmag.get_filename_without_ext(img_path)
        labels_np = sitk_io.load_registered_img(
            img_path, config.RegNames.IMG_LABELS.value)
        # only use pos labels since assume neg labels are merely mirrored
        labels_unique = np.unique(labels_np[labels_np >= 0])
        labels_dict[name] = pd.Series(
            np.ones(len(labels_unique), dtype=int), index=labels_unique)
    df = pd.DataFrame(labels_dict)
    df.sort_index()
    df.to_csv(output_path)
    print("common labels exported to {}".format(output_path))
    return df


def make_density_image(
        img_path: str, scale: Optional[float] = None,
        shape: Optional[Sequence[int]] = None, suffix: Optional[str] = None, 
        labels_img_sitk: Optional[sitk.Image] = None,
        channel: Optional[Sequence[int]] = None,
        matches: Dict[Tuple[int, int], "colocalizer.BlobMatch"] = None,
        atlas_profile: Optional["atlas_prof.AtlasProfile"] = None,
        include: Sequence[int] = None
) -> Tuple[np.ndarray, str]:
    """Make a density image based on associated blobs.
    
    Uses the size and/or resolutions of the original image stored in the blobs
    if available to determine scaling between the blobs and the output image.
    Otherwise, attempts to load the original image or at least its metadata.
    The voxel sizes for the blobs is determined by giving an output image
    or shape.
    
    If ``matches`` is given, a heat map will be generated for each set
    of channels given in the dictionary. Otherwise, if the loaded blobs
    file has intensity-based colocalizations, a heat map will be generated
    for each combination of channels.
    
    Args:
        img_path: Path to image, which will be used to indentify the blobs file.
        scale: Scaling factor between the blobs' space and the output space;
            defaults to None to use the register. Scaling is found by
            :meth:`magmap.np_io.find_scaling`.
        shape: Output shape. Defaults to None, in which case the shape will
            match ``labels_img_sitk``.
        suffix: Modifier to append to end of ``img_path`` basename for
            registered image files that were output to a modified name;
            defaults to None.
        labels_img_sitk: Labels image. Defaults to None, in which case a
            registered labels image will be loaded.
        channel: Sequence of channels to include in density image. For
            multiple channels, blobs from all these channels are combined
            into one heatmap.  Defaults to None to use all channels.
        matches: Dictionary of channel combinations to blob matches; defaults
            to None.
        atlas_profile: Atlas profile, used for scaling; defaults to None.
        include: Sequence of blob ``confirmed`` flags to include; defaults
            to None, in which case all flags will be included.
    
    Returns:
        Tuple of the density image as a Numpy array in the
        same shape as the opened image and the original and ``img_path``
        to track such as for multiprocessing.
    
    """
    def make_heat_map():
        # build heat map to store densities per label px and save to file
        coord_scaled = ontology.scale_coords(
            blobs_chl[:, :3], scaling, labels_img.shape)
        _logger.debug("Scaled coords:\n%s", coord_scaled)
        return cv_nd.build_heat_map(labels_img.shape, coord_scaled)
    
    # set up paths and get labels image
    _logger.info("\n\nGenerating heat map from blobs")
    mod_path = img_path
    if suffix is not None:
        mod_path = libmag.insert_before_ext(img_path, suffix)
    
    # load blobs
    blobs = detector.Blobs().load_blobs(np_io.img_to_blobs_path(img_path))
    
    # prepare output image
    is_2d = False
    if (shape is not None and blobs.roi_size is not None
            and blobs.resolutions is not None):
        # use target shape provided directly; extract image size stored in
        # blobs archive, assuming ROI size is full the full image
        scaling = np.divide(shape, blobs.roi_size)
        labels_spacing = np.divide(blobs.resolutions[0], scaling)
        labels_img = np.zeros(shape, dtype=np.uint8)
        labels_img_sitk = sitk.GetImageFromArray(labels_img)
        labels_img_sitk.SetSpacing(labels_spacing[::-1])
    
    else:
        # default to use labels image as the size of the output image
        if labels_img_sitk is None:
            labels_img_sitk = sitk_io.load_registered_img(
                mod_path, config.RegNames.IMG_LABELS.value, get_sitk=True)
        labels_img = sitk.GetArrayFromImage(labels_img_sitk)
        
        is_2d = labels_img.ndim == 2
        labels_res = list(labels_img_sitk.GetSpacing()[::-1])
        if is_2d:
            # temporarily convert 2D images to 3D
            labels_img = labels_img[None]
            labels_res.insert(0, 1)
        
        if blobs.resolutions is not None:
            # find scaling based on blob to labels image resolution ratio
            scaling = np.divide(blobs.resolutions[0], labels_res)
        
        else:
            # find the scaling between the blobs and the labels image
            # TODO: remove target_size since it can be set by shape?
            target_size = (
                None if atlas_profile is None else atlas_profile["target_size"])
            scaling = np_io.find_scaling(
                img_path, labels_img.shape, scale, target_size)[0]
        
        if shape is not None:
            # scale blob coordinates and heat map to an alternative final shape
            scaling = np.divide(shape, np.divide(labels_img.shape, scaling))
            labels_spacing = np.multiply(
                labels_res, np.divide(labels_img.shape, shape))
            labels_img = np.zeros(shape, dtype=labels_img.dtype)
            labels_img_sitk.SetSpacing(labels_spacing[::-1])
    _logger.debug("Using image scaling: {}".format(scaling))
    
    blobs_chl = blobs.blobs
    _logger.info("Initial number of blobs: %s", len(blobs_chl))
    if channel is not None:
        # filter blobs by channel
        _logger.info(
            "Using blobs from channel(s), combining if multiple channels: %s",
            channel)
        blobs_chl = blobs_chl[np.isin(detector.Blobs.get_blobs_channel(
            blobs_chl), channel)]
        _logger.info("Number of remaining blobs: %s", len(blobs_chl))

    if include is not None:
        # filter blobs by confirmation flag
        _logger.info("Using blobs with confirmed flag(s): %s", include)
        blobs_chl = blobs_chl[np.isin(detector.Blobs.get_blob_confirmed(
            blobs_chl), include)]
        _logger.info("Number of remaining blobs: %s", len(blobs_chl))
    
    # annotate blobs based on position
    heat_map = make_heat_map()
    if is_2d:
        # convert back to 2D
        heat_map = heat_map[0]
    imgs_write = {
        config.RegNames.IMG_HEAT_MAP.value:
            sitk_io.replace_sitk_with_numpy(labels_img_sitk, heat_map)}
    
    heat_colocs = None
    if matches:
        # create heat maps for match-based colocalization combos
        heat_colocs = []
        for chl_combo, chl_matches in matches.items():
            _logger.info(
                "Generating match-based colocalization heat map "
                "for channel combo: %s", chl_combo)
            # use blobs in first channel of each channel pair for simplicity
            blobs_chl = chl_matches.get_blobs(1)
            heat_colocs.append(make_heat_map())
    
    elif blobs.colocalizations is not None:
        # create heat map for each intensity-based colocalization combo
        # as a separate channel in output image
        blob_chls = range(blobs.colocalizations.shape[1])
        blob_chls_len = len(blob_chls)
        if blob_chls_len > 1:
            # get all channel combos that include given channels
            combos = []
            chls = blob_chls if channel is None else channel
            for r in range(2, blob_chls_len + 1):
                combos.extend(
                    [tuple(c) for c in itertools.combinations(blob_chls, r)
                     if all([h in c for h in chls])])
            
            heat_colocs = []
            for combo in combos:
                _logger.info(
                    "Generating intensity-based colocalization heat map "
                    "for channel combo: %s", combo)
                blobs_chl = blobs.blobs[np.all(np.equal(
                    blobs.colocalizations[:, combo], 1), axis=1)]
                heat_colocs.append(make_heat_map())
    
    if heat_colocs is not None:
        # combine heat maps into single image
        heat_colocs = np.stack(heat_colocs, axis=3)
        if is_2d:
            # convert back to 2D
            heat_colocs = heat_colocs[0]
        imgs_write[config.RegNames.IMG_HEAT_COLOC.value] = \
            sitk_io.replace_sitk_with_numpy(labels_img_sitk, heat_colocs, True)
    
    # write images to file
    sitk_io.write_reg_images(imgs_write, mod_path)
    return heat_map, img_path


def make_density_images_mp(img_paths, scale=None, shape=None, suffix=None,
                           channel=None):
    """Make density images for a list of files as a multiprocessing 
    wrapper for :func:``make_density_image``
    
    Args:
        img_paths (List[str]): Sequence of image paths, which will be used to
            indentify the blob files.
        scale (int, float): Rescaling factor as a scalar value. If set,
            the corresponding image for this factor will be opened. If None,
            the full size  image will be used. Defaults to None.
        shape (List[int]): Sequence of target shape defining the voxels for
            the density map; defaults to None.
        suffix (str): Modifier to append to end of ``img_path`` basename for
            registered image files that were output to a modified name; 
            defaults to None.
        channel (List[int]): Sequence of channels to include in density image;
            defaults to None to use all channels.
    """
    start_time = time()
    pool = chunking.get_mp_pool()
    pool_results = []
    for img_path in img_paths:
        print("Making density image from blobs related to:", img_path)
        if config.channel:
            # get blob matches for the given channels if available; must load
            # from db outside of multiprocessing to avoid MemoryError
            matches = colocalizer.select_matches(
                config.db, config.channel,
                exp_name=sqlite.get_exp_name(img_path))
        else:
            matches = None
        pool_results.append(pool.apply_async(
            make_density_image,
            args=(img_path, scale, shape, suffix, None, channel, matches,
                  config.atlas_profile,
                  config.classifier.include)))
    for result in pool_results:
        _, path = result.get()
        print("finished {}".format(path))
    pool.close()
    pool.join()
    print("time elapsed for making density images:", time() - start_time)


def make_labels_diff_img(img_path, df_path, meas, fn_avg, prefix=None, 
                         show=False, level=None, meas_path_name=None, 
                         col_wt=None):
    """Replace labels in an image with the differences in metrics for 
    each given region between two conditions.
    
    Args:
        img_path: Path to the base image from which the corresponding 
            registered image will be found.
        df_path: Path to data frame with metrics for the labels.
        meas: Name of colum in data frame with the chosen measurement.
        fn_avg: Function to apply to the set of measurements, such as a mean. 
            Can be None if ``df_path`` points to a stats file from which 
            to extract metrics directly in :meth:``vols.map_meas_to_labels``.
        prefix: Start of path for output image; defaults to None to 
            use ``img_path`` instead.
        show: True to show the images after generating them; defaults to False.
        level: Ontological level at which to look up and show labels. 
            Assume that labels level image corresponding to this value 
            has already been generated by :meth:``make_labels_level_img``. 
            Defaults to None to use only drawn labels.
        meas_path_name: Name to use in place of `meas` in output path; 
            defaults to None.
        col_wt (str): Name of column to use for weighting; defaults to None.
    """
    # load labels image and data frame before generating map for the 
    # given metric of the chosen measurement
    print("Generating labels difference image for", meas, "from", df_path)
    reg_name = (config.RegNames.IMG_LABELS.value if level is None 
                else config.RegNames.IMG_LABELS_LEVEL.value.format(level))
    labels_sitk = sitk_io.load_registered_img(img_path, reg_name, get_sitk=True)
    labels_np = sitk.GetArrayFromImage(labels_sitk)
    df = pd.read_csv(df_path)
    labels_diff = vols.map_meas_to_labels(
        labels_np, df, meas, fn_avg, col_wt=col_wt)
    if labels_diff is None: return
    labels_diff_sitk = sitk_io.replace_sitk_with_numpy(labels_sitk, labels_diff)
    
    # save and show labels difference image using measurement name in 
    # output path or overriding with custom name
    meas_path = meas if meas_path_name is None else meas_path_name
    reg_diff = libmag.insert_before_ext(
        config.RegNames.IMG_LABELS_DIFF.value, meas_path, "_")
    if fn_avg is not None:
        # add function name to output path if given
        reg_diff = libmag.insert_before_ext(
            reg_diff, fn_avg.__name__, "_")
    imgs_write = {reg_diff: labels_diff_sitk}
    out_path = prefix if prefix else img_path
    sitk_io.write_reg_images(imgs_write, out_path)
    if show:
        for img in imgs_write.values():
            if img: sitk.Show(img)


def make_labels_level_img(
        img_path: Optional[str], level: int, prefix: Optional[str] = None,
        show: bool = False
) -> Dict[str, sitk.Image]:
    """Replace labels in an image with their parents at the given level.
    
    Labels that do not fall within a parent at that level will remain in place.
    
    Args:
        img_path: Path to the base image from which the corresponding
            registered image will be found. Can be None, where the globally
            set up image stored in :attr:`magmap.settings.config` will be used
            instead. If so, `prefix` must be given to specify the output path.
        level: Ontological level at which to group child labels.
        prefix: Start of path for output image; defaults to None to
            use ``img_path`` instead.
        show: True to show the images after generating them; defaults to False.
    
    Returns:
        Dictionary of registered image suffix to SimpleITK image.
    
    Raises:
        `ValueError` if `img_path` and `prefix` are both None.
    
    """
    if img_path is None:
        if not prefix:
            raise ValueError("Must set 'prefix' if 'img_path' is None")
        # use the globally set up image stored in config
        labels_sitk = config.labels_img_sitk
        ref = config.labels_ref
    else:
        # load original labels image and setup ontology dictionary
        labels_sitk = sitk_io.load_registered_img(
            img_path, config.RegNames.IMG_LABELS.value, get_sitk=True)
        ref = ontology.LabelsRef(config.load_labels).load()
    labels_np = sitk.GetArrayFromImage(labels_sitk)
    
    # remap labels to given level
    labels_np = ontology.make_labels_level(labels_np, ref, level)
    labels_level_sitk = sitk_io.replace_sitk_with_numpy(labels_sitk, labels_np)
    
    # generate an edge image at this level
    labels_edge = vols.LabelToEdge.make_labels_edge(labels_np)
    labels_edge_sitk = sitk_io.replace_sitk_with_numpy(labels_sitk, labels_edge)
    
    # write and optionally display labels level image
    imgs_write = {
        config.RegNames.IMG_LABELS_LEVEL.value.format(level): labels_level_sitk,
        config.RegNames.IMG_LABELS_EDGE_LEVEL.value.format(level):
            labels_edge_sitk,
    }
    out_path = prefix if prefix else img_path
    sitk_io.write_reg_images(imgs_write, out_path)
    if show:
        for img in imgs_write.values():
            if img: sitk.Show(img)
    
    return imgs_write
