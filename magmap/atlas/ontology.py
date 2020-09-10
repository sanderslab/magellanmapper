# Anatomical ontology management
# Author: David Young, 2019
"""Handle ontology lookup.
"""

import os
from collections import OrderedDict
from enum import Enum
import json

import numpy as np
import pandas as pd

from magmap.settings import config
from magmap.io import libmag

NODE = "node"
PARENT_IDS = "parent_ids"
MIRRORED = "mirrored"
RIGHT_SUFFIX = " (R)"
LEFT_SUFFIX = " (L)"


class LabelColumns(Enum):
    """Label data frame columns enumeration."""
    FROM_LABEL = "FromLabel"
    TO_LABEL = "ToLabel"


def load_labels_ref(path):
    """Load labels from a reference JSON or CSV file.
    
    Args:
        path: Path to labels reference.
    
    Returns:
        A JSON decoded object (eg dictionary) if the path has a JSON 
        extension, or a ``Pandas`` object otherwise.
    """
    labels_ref = None
    path_split = os.path.splitext(path)
    if path_split[1] == ".json":
        with open(path, "r") as f:
            labels_ref = json.load(f)
    else:
        labels_ref = pd.read_csv(path)
    return labels_ref


def convert_itksnap_to_df(path):
    """Convert an ITK-SNAP labels description file to a CSV file 
    compatible with MagellanMapper.
    
    Args:
        path: Path to description file.
    
    Returns:
        Pandas data frame of the description file.
    """
    # load description file and convert contiguous spaces to separators, 
    # remove comments, and add headers
    df = pd.read_csv(
        path, sep="\s+", comment="#", 
        names=[e.value for e in config.ItkSnapLabels])
    return df


def get_node(nested_dict, key, value, key_children):
    """Get a node from a nested dictionary by iterating through all 
    dictionaries until the specified value is found.
    
    Args:
        nested_dict: A dictionary that contains a list of dictionaries in
            the key_children entry.
        key: Key to check for the value.
        value: Value to find, assumed to be unique for the given key.
        key_children: Name of the children key, which contains a list of 
            further dictionaries but can be empty.
    
    Returns:
        The node matching the key-value pair, or None if not found.
    """
    try:
        #print("checking for key {}...".format(key), end="")
        found_val = nested_dict[key]
        #print("found {}".format(found_val))
        if found_val == value:
            return nested_dict
        children = nested_dict[key_children]
        for child in children:
            result = get_node(child, key, value, key_children)
            if result is not None:
                return result
    except KeyError as e:
        print(e)
    return None


def create_aba_reverse_lookup(labels_ref):
    """Create a reverse lookup dictionary for Allen Brain Atlas style
    ontology files.
    
    Args:
        labels_ref: The ontology file as a parsed JSON dictionary.
    
    Returns:
        Reverse lookup dictionary as output by 
        :func:`ontology.create_reverse_lookup`.
    """
    return create_reverse_lookup(
        labels_ref["msg"][0], config.ABAKeys.ABA_ID.value, 
        config.ABAKeys.CHILDREN.value)


def create_reverse_lookup(nested_dict, key, key_children, id_dict=OrderedDict(), 
                          parent_list=None):
    """Create a reveres lookup dictionary with the values of the original 
    dictionary as the keys of the new dictionary.
    
    Each value of the new dictionary is another dictionary that contains 
    "node", the dictionary with the given key-value pair, and "parent_ids", 
    a list of all the parents of the given node. This entry can be used to 
    track all superceding dictionaries, and the node can be used to find 
    all its children.
    
    Args:
        nested_dict: A dictionary that contains a list of dictionaries in
            the key_children entry.
        key: Key that contains the values to use as keys in the new dictionary. 
            The values of this key should be unique throughout the entire 
            nested_dict and thus serve as IDs.
        key_children: Name of the children key, which contains a list of 
            further dictionaries but can be empty.
        id_dict: The output dictionary as an OrderedDict to preserve key 
            order (though not hierarchical structure) so that children 
            will come after their parents; if None is given, an empty 
            dictionary will be created.
        parent_list: List of values for the given key in all parent 
            dictionaries.
    
    Returns:
        A dictionary with the original values as the keys, which each map 
        to another dictionary containing an entry with the dictionary 
        holding the given value and another entry with a list of all parent 
        dictionary values for the given key.
    """
    value = nested_dict[key]
    sub_dict = {NODE: nested_dict}
    if parent_list is not None:
        sub_dict[PARENT_IDS] = parent_list
    id_dict[value] = sub_dict
    try:
        children = nested_dict[key_children]
        parent_list = [] if parent_list is None else list(parent_list)
        parent_list.append(value)
        for child in children:
            #print("parents: {}".format(parent_list))
            create_reverse_lookup(
                child, key, key_children, id_dict, parent_list)
    except KeyError as e:
        print(e)
    return id_dict


def create_lookup_pd(df):
    """Create a lookup dictionary from a Pandas data frame.
    
    Args:
        df: Pandas data frame, assumed to have at least columns 
            corresponding to :const:``config.ABAKeys.ABA_ID`` and 
            :const:``config.ABAKeys.ABA_NAME``.
    
    Returns:
        Dictionary similar to that generated from 
        :meth:``create_reverse_lookup``, with IDs as keys and values 
        corresponding of another dictionary with :const:``NODE`` and 
        :const:``PARENT_IDS`` as keys. :const:``NODE`` in turn 
        contains a dictionary with entries for each Enum in 
        :const:``config.ABAKeys``.
    """
    id_dict = OrderedDict()
    ids = df[config.ABAKeys.ABA_ID.value]
    for region_id in ids:
        region = df[df[config.ABAKeys.ABA_ID.value] == region_id]
        region_dict = region.to_dict("records")[0]
        if config.ABAKeys.NAME.value not in region_dict:
            region_dict[config.ABAKeys.NAME.value] = str(region_id)
        region_dict[config.ABAKeys.LEVEL.value] = 1
        region_dict[config.ABAKeys.CHILDREN.value] = []
        region_dict[config.ABAKeys.ACRONYM.value] = ""
        sub_dict = {NODE: region_dict, PARENT_IDS: []}
        id_dict[region_id] = sub_dict
    return id_dict


def _get_children(labels_ref_lookup, label_id, children_all=[]):
    """Recursively get the children of a given non-negative atlas ID.
    
    Used as a helper function to :func:``get_children_from_id``.
    
    Args:
        labels_ref_lookup: The labels reference lookup, assumed to be 
            generated by :func:`create_reverse_lookup` to look up by ID.
        label_id: ID of the label to find, assumed to be >= 0 since 
            IDs in ``labels_ref_lookup`` are generally non-negative.
        children_all: List of all children of this ID, used recursively; 
            defaults to an empty list. To include the ID itself, pass in a 
            list with this ID alone.
    
    Returns:
        A list of all children of the given ID, in order from highest 
        (numerically lowest) level to lowest.
    """
    label = labels_ref_lookup.get(label_id)
    if label:
        # recursively gather the children of the label
        children = label[NODE][config.ABAKeys.CHILDREN.value]
        for child in children:
            child_id = child[config.ABAKeys.ABA_ID.value]
            #print("child_id: {}".format(child_id))
            children_all.append(child_id)
            _get_children(labels_ref_lookup, child_id, children_all)
    return children_all


def _mirror_label_ids(label_ids, combine=False):
    """Mirror label IDs, assuming that a "mirrored" ID is the negative
    of the given ID.
    
    Args:
        label_ids (Union[int, List[int]]): Single ID or sequence of IDs.
        combine (bool): True to return a list of ``label_ids`` along with
            their mirrored IDs; defaults to False to return on the mirrored IDs.

    Returns:
        Union[int, List[int]]: A single mirrored ID if ``label_ids`` is
        one ID and ``combine`` is False, or a list of IDs.

    """
    if libmag.is_seq(label_ids):
        mirrored = [-1 * n for n in label_ids]
        if combine:
            mirrored = list(label_ids).extend(mirrored)
    else:
        mirrored = -1 * label_ids
        if combine:
            mirrored = [label_ids, mirrored]
    return mirrored


def get_children_from_id(labels_ref_lookup, label_id, incl_parent=True, 
                         both_sides=False):
    """Get the children of a given atlas ID.
    
    Args:
        labels_ref_lookup: The labels reference lookup, assumed to be 
            generated by :func:`create_reverse_lookup` to look up by ID.
        label_id: ID of the label to find, which can be negative.
        incl_parent: True to include ``label_id`` itself in the list of 
            children; defaults to True.
        both_sides: True to include both sides, ie positive and negative 
            values of each ID. Defaults to False.
    
    Returns:
        A list of all children of the given ID, in order from highest 
        (numerically lowest) level to lowest.
    """
    id_abs = abs(label_id)
    children_all = [id_abs] if incl_parent else []
    region_ids = _get_children(labels_ref_lookup, id_abs, children_all)
    if both_sides:
        region_ids.extend(_mirror_label_ids(region_ids))
    elif label_id < 0:
        region_ids = _mirror_label_ids(region_ids)
    #print("region IDs: {}".format(region_ids))
    return region_ids


def labels_to_parent(labels_ref_lookup, level=None,
                     allow_parent_same_level=False):
    """Generate a dictionary mapping label IDs to parent IDs at a given level.
    
    Parents are considered to be "below" (numerically lower level) their
    children, or at least at the same level if ``allow_parent_same_level``
    is True.
    
    Args:
        labels_ref_lookup (dict): The labels reference lookup, assumed to be an
            OrderedDict generated by :func:`ontology.create_reverse_lookup` 
            to look up by ID while preserving key order to ensure that 
            parents of any child will be reached prior to the child.
        level (int): Level at which to find parent for each label; defaults to
            None to get the parent immediately below the given label.
        allow_parent_same_level (bool): True to allow selecting a parent at
            the same level as the label; False to require the parent to be
            at least one level below. Defaults to False.
    
    Returns:
        dict: Dictionary of label IDs to parent IDs at the given level. Labels
        at the given level will be assigned to their own ID, and labels below
        or without a parent at the level will be given a default level of 0.
    
    """
    # similar to volumes_dict_level_grouping but without checking for neg 
    # keys or grouping values
    label_parents = {}
    ids = list(labels_ref_lookup.keys())
    for label_id in ids:
        parent_at_level = 0
        label = labels_ref_lookup[label_id]
        
        # find ancestor above (numerically below) label's level
        label_level = label[NODE][config.ABAKeys.LEVEL.value]
        target_level = label_level - 1 if level is None else level
        if label_level == target_level:
            # use label's own ID if at target level
            parent_at_level = label_id
        elif label_level > target_level:
            parents = label.get(PARENT_IDS)
            if parents:
                for parent in parents[::-1]:
                    # assume that parents are ordered by decreasing
                    # (numerically higher) level
                    parent_level = labels_ref_lookup[
                        parent][NODE][config.ABAKeys.LEVEL.value]
                    if (parent_level <= target_level
                            or allow_parent_same_level
                            and parent_level == label_level):
                        # use first parent below (or at least at) target level
                        parent_at_level = parent
                        break
            else:
                print("No parents at level", label_level, "for label", label_id)
        
        parent_ref = label[NODE][config.ABAKeys.PARENT_ID.value]
        try:
            # check for discrepancies between parent listed in ontology file
            # and derived from parsed parent IDs
            assert parent_ref == parent_at_level
        except AssertionError:
            print("Parent at level {} or lower for label {} does not match"
                  "parent listed in reference file, {}"
                  .format(target_level, label_id, parent_at_level, parent_ref))
        label_parents[label_id] = parent_at_level
    return label_parents


def get_label_item(label, item_key, key=NODE):
    """Convenience function to get the item from the sub-label.

    Args:
        label (dict): The label dictionary. Assumes that ``label`` is a
            nested dictionary.
        item_key (str): Key for item to retrieve from within ``label[key]``.
        key (str): First level key; defaults to :const:`NODE`.

    Returns:
        The label item, or None if not found.
    """
    item = None
    try:
        if label is not None:
            sub = label[key]
            if sub is not None:
                item = sub[item_key]
    except KeyError as e:
        print(e, item_key)
    return item


def get_label_name(label, side=False):
    """Get the atlas region name from the label.
    
    Args:
        label (dict): The label dictionary.
        side (bool):
    
    Returns:
        The atlas region name, or None if not found.
    """
    name = None
    try:
        if label is not None:
            node = label[NODE]
            if node is not None:
                name = node[config.ABAKeys.NAME.value]
                print("name: {}".format(name), label[MIRRORED])
                if side:
                    if label[MIRRORED]:
                        name += LEFT_SUFFIX
                    else:
                        name += RIGHT_SUFFIX
    except KeyError as e:
        print(e, name)
    return name


def get_label_side(label_id):
    """Convert label IDs into side strings.

    The convention used here is that positive values = right, negative
    values = left.

    TODO: consider making pos/neg side correspondence configurable.
    
    Args:
        label_id (int, List[int]): Label ID or sequence of IDs to convert,
            where all negative labels are considered right, all positive
            are left, and any mix of pos, neg, or zero are both.

    Returns:
        :str: Value of corresponding :class:`config.HemSides` enum.

    """
    if np.all(np.greater(label_id, 0)):
        return config.HemSides.RIGHT.value
    elif np.all(np.less(label_id, 0)):
        return config.HemSides.LEFT.value
    return config.HemSides.BOTH.value


def get_label_ids_from_position(coord, labels_img, scaling=None, rounding=False, 
                                return_coord_scaled=False):
    """Get the atlas label IDs for the given coordinates.
    
    Args:
        coord: Coordinates of experiment image in (z, y, x) order. Can be an 
            [n, 3] array of coordinates.
        labels_img: The registered image whose intensity values correspond to 
            label IDs.
        scaling: Scaling factor for the labels image size compared with the 
            experiment image; defaults to None.
        rounding: True to round coordinates after scaling, which should be 
            used rounding to reverse coordinates that were previously scaled 
            inversely to avoid size degredation with repeated scaling. 
            Typically rounding is False (default) so that coordinates fall 
            evenly to their lowest integer, without exceeding max bounds.
        return_coord_scaled: True to return the array of scaled coordinates; 
            defaults to False.
    
    Returns:
        An array of label IDs corresponding to ``coord``, or a scalar of 
        one ID if only one coordinate is given. If ``return_coord_scaled`` is 
        True, also returns a Numpy array of the same shape as ``coord`` 
        scaled based on ``scaling``.
    """
    libmag.printv(
        "getting label IDs from coordinates using scaling", scaling)
    coord_scaled = coord
    if scaling is not None:
        # scale coordinates to atlas image size
        coord_scaled = np.multiply(coord, scaling)
    if rounding: 
        # round when extra precision is necessary, such as during reverse 
        # scaling, which requires clipping so coordinates don't exceed labels 
        # image shape
        coord_scaled = np.around(coord_scaled).astype(np.int)
        coord_scaled = np.clip(
            coord_scaled, None, np.subtract(labels_img.shape, 1))
    else:
        # typically don't round to stay within bounds
        coord_scaled = coord_scaled.astype(np.int)
    
    # index blob coordinates into labels image by int array indexing to 
    # get the corresponding label IDs
    coordsi = libmag.coords_for_indexing(coord_scaled)
    label_ids = labels_img[tuple(coordsi)][0]
    if return_coord_scaled:
        return label_ids, coord_scaled
    return label_ids


def get_label(coord, labels_img, labels_ref, scaling, level=None, 
              rounding=False):
    """Get the atlas label for the given coordinates.
    
    Args:
        coord: Coordinates of experiment image in (z, y, x) order.
        labels_img: The registered image whose intensity values correspond to 
            label IDs.
        labels_ref: The labels reference lookup, assumed to be generated by 
            :func:`ontology.create_reverse_lookup` to look up by ID.
        scaling: Scaling factor for the labels image size compared with the 
            experiment image.
        level: The ontology level as an integer to target; defaults to None. 
            If None, level will be ignored, and the exact matching label 
            to the given coordinates will be returned. If a level is given, 
            the label at the highest (numerically lowest) level encompassing 
            this region will be returned.
        rounding: True to round coordinates after scaling (see 
            :func:``get_label_ids_from_position``); defaults to False.
    
    Returns:
        The label dictionary at those coordinates, or None if no label is 
        found.
    """
    label_id = get_label_ids_from_position(coord, labels_img, scaling, rounding)
    libmag.printv("found label_id: {}".format(label_id))
    mirrored = label_id < 0
    if mirrored:
        label_id = -1 * label_id
    label = None
    try:
        label = labels_ref[label_id]
        if level is not None and label[
            NODE][config.ABAKeys.LEVEL.value] > level:
            
            # search for parent at "higher" (numerically lower) level 
            # that matches the target level
            parents = label[PARENT_IDS]
            label = None
            if label_id < 0:
                parents = np.multiply(parents, -1)
            for parent in parents:
                parent_label = labels_ref[parent]
                if parent_label[NODE][config.ABAKeys.LEVEL.value] == level:
                    
                    label = parent_label
                    break
        if label is not None:
            label[MIRRORED] = mirrored
            libmag.printv(
                "label ID at level {}: {}".format(level, label_id))
    except KeyError as e:
        libmag.printv(
            "could not find label id {} or its parent (error {})"
            .format(label_id, e))
    return label


def get_region_middle(labels_ref_lookup, label_id, labels_img, scaling, 
                      both_sides=False, incl_children=True):
    """Approximate the middle position of a region by taking the middle 
    value of its sorted list of coordinates.
    
    The region's coordinate sorting prioritizes z, followed by y, etc, meaning
    that the middle value will be closest to the middle of z but may fall
    be slightly away from midline in the other axes if this z does not
    contain y/x's around midline. Getting the coordinate at the middle
    of this list rather than another coordinate midway between other values
    in the region ensures that the returned coordinate will reside within
    the region itself, including non-contingous regions that may be
    intermixed with coordinates not part of the region.
    
    Args:
        labels_ref_lookup (Dict[int, Dict]): The labels reference lookup,
            assumed to be  generated by :func:`ontology.create_reverse_lookup`
            to look up by ID.
        label_id (int, List[int]): ID of the label to find, or sequence of IDs.
        labels_img (:obj:`np.ndarray`): The registered image whose intensity
            values correspond to label IDs.
        scaling (:obj:`np.ndarray`): Scaling factors as a Numpy array in z,y,x
            for the labels image size compared with the experiment image.
        both_sides (bool, List[bool]): True to include both sides, or
            sequence of booleans corresponding to ``label_id``; defaults
            to False.
        incl_children (bool): True to include children of ``label_id``,
            False to include only ``label_id``; defaults to True.
    
    Returns:
        List[int], :obj:`np.ndarray`, List[int]: ``coord``, the middle value
        of a list of all coordinates in the region at the given ID;
        ``img_region``, a boolean mask of the region within ``labels_img``;
        and ``region_ids``, a list of the IDs included in the region.
        If ``labels_ref_lookup`` is None, all values are None.
    
    """
    if not labels_ref_lookup:
        return None, None, None
    
    # gather IDs for label, including children and opposite sides
    if not libmag.is_seq(label_id):
        label_id = [label_id]
    if not libmag.is_seq(both_sides):
        both_sides = [both_sides]
    region_ids = []
    for region_id, both in zip(label_id, both_sides):
        if incl_children:
            # add children of the label +/- both sides
            region_ids.extend(get_children_from_id(
                labels_ref_lookup, region_id, incl_parent=True,
                both_sides=both))
        else:
            # add the label +/- its mirrored version
            region_ids.append(region_id)
            if both:
                region_ids.append(_mirror_label_ids(region_id))
    
    # get a list of all the region's coordinates to sort
    img_region = np.isin(labels_img, region_ids)
    region_coords = np.where(img_region)
    #print("region_coords:\n{}".format(region_coords))
    
    def get_middle(coords):
        # recursively get value at middle of list for each axis
        sort_ind = np.lexsort(coords[::-1])  # last axis is primary key
        num_coords = len(sort_ind)
        if num_coords > 0:
            mid_ind = sort_ind[int(num_coords / 2)]
            mid = coords[0][mid_ind]
            if len(coords) > 1:
                # shift to next axis in tuple of coords
                mask = coords[0] == mid
                coords = tuple(c[mask] for c in coords[1:])
                return (mid, *get_middle(coords))
            return (mid, )
        return None
    
    coord = None
    coord_labels = get_middle(region_coords)
    if coord_labels:
        print("coord_labels (unscaled): {}".format(coord_labels))
        print("ID at middle coord: {} (in region? {})"
              .format(labels_img[coord_labels], img_region[coord_labels]))
        coord = tuple(np.around(coord_labels / scaling).astype(np.int))
    print("coord at middle: {}".format(coord))
    return coord, img_region, region_ids


def rel_to_abs_ages(rel_ages, gestation=19):
    """Convert sample names to ages.
    
    Args:
        rel_ages (List[str]): Sequence of strings in the format, 
            ``[stage][relative_age_in_days]``, where stage
            is either "E" = "embryonic" or "P" = "postnatal", such as 
            "E3.5" for 3.5 days after conception, or "P10" for 10 days 
            after birth.
        gestation (int): Number of days from conception until birth.

    Returns:
        Dictionary of ``{name: age_in_days}``.

    """
    ages = {}
    for val in rel_ages:
        age = float(val[1:])
        if val[0].lower() == "p":
            age += gestation
        ages[val] = age
    return ages


def replace_labels(labels_img, df, clear=False):
    """Replace labels based on a data frame.
    
    Args:
        labels_img (:obj:`np.ndarray`): Labels image array whose values
            will be converted in-place.
        df (:obj:`pd.DataFrame`): Pandas data frame with from and to columns
            specified by :class:`LabelColumns` values.
        clear (bool): True to clear all other label values.

    Returns:
        :obj:`np.ndarray`: ``labels_img`` with values replaced in-place.

    """
    labels_img_orig = labels_img
    if clear:
        # clear all labels, replacing based on copy
        labels_img_orig = np.copy(labels_img)
        labels_img[:] = 0
    from_labels = df[LabelColumns.FROM_LABEL.value]
    to_labels = df[LabelColumns.TO_LABEL.value]
    for to_label in to_labels.unique():
        # replace all labels matching the given target label
        to_convert = from_labels.loc[to_labels == to_label]
        print("Converting labels from {} to {}"
              .format(to_convert.values, to_label))
        labels_img[np.isin(labels_img_orig, to_convert)] = to_label
    return labels_img
