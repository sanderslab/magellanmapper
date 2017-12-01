#!/bin/bash
# Image registration
# Author: David Young, 2017
"""Register images to one another.
"""

import os
import copy
import json
from collections import OrderedDict
from pprint import pprint
from time import time
import SimpleITK as sitk
import numpy as np

from clrbrain import cli
from clrbrain import config
from clrbrain import detector
from clrbrain import importer
from clrbrain import lib_clrbrain
from clrbrain import plot_2d

IMG_ATLAS = "atlasVolume.mhd"
IMG_LABELS = "annotation.mhd"

NODE = "node"
PARENT_IDS = "parent_ids"
MIRRORED = "mirrored"
RIGHT_SUFFIX = " (R)"
LEFT_SUFFIX = " (L)"
ABA_ID = "id"
ABA_PARENT = "parent_structure_id"
ABA_LEVEL = "st_level"

def _reg_out_path(file_path, reg_name):
    """Generate a path for a file registered to another file.
    
    Args:
        file_name: Full path of file registered to.
        reg_name: Filename alone of registered file.
    
    Returns:
        Full path with the registered filename including extension at the end.
    """
    file_path_base = importer.filename_to_base(file_path, cli.series)
    return file_path_base + "_" + reg_name

def _translation_adjust(orig, transformed, translation, flip=False):
    """Adjust translation based on differences in scaling between original 
    and transformed images to allow the translation to be applied to the 
    original image.
    
    Assumes (x, y, z) order for consistency with SimpleITK since this method 
    operates on SimpleITK format images.
    
    Args:
        orig: Original image in SimpleITK format.
        transformed: Transformed image in SimpleITK format.
        translation: Translation in (x, y, z) order, taken from transform 
            parameters and scaled to the transformed images's spacing.
    
    Returns:
        The adjusted translation in (x, y, z) order.
    """
    if translation is None:
        return translation
    # TODO: need to check which space the TransformParameter is referring to 
    # and how to scale it since the adjusted translation does not appear to 
    # be working yet
    orig_origin = orig.GetOrigin()
    transformed_origin = transformed.GetOrigin()
    origin_diff = np.subtract(transformed_origin, orig_origin)
    print("orig_origin: {}, transformed_origin: {}, origin_diff: {}"
          .format(orig_origin, transformed_origin, origin_diff))
    orig_size = orig.GetSize()
    transformed_size = transformed.GetSize()
    size_ratio = np.divide(orig_size, transformed_size)
    print("orig_size: {}, transformed_size: {}, size_ratio: {}"
          .format(orig_size, transformed_size, size_ratio))
    translation_adj = np.multiply(translation, size_ratio)
    #translation_adj = np.add(translation_adj, origin_diff)
    print("translation_adj: {}".format(translation_adj))
    if flip:
        translation_adj = translation_adj[::-1]
    return translation_adj

def _show_overlays(imgs, translation, fixed_file):
    """Shows overlays via :func:plot_2d:`plot_overlays_reg`.
    
    Args:
        imgs: List of images in Numpy format
        translation: Translation in (z, y, x) format for Numpy consistency.
        fixed_file: Path to fixed file to get title.
    """
    cmaps = ["Blues", "Oranges", "prism"]
    #plot_2d.plot_overlays(imgs, z, cmaps, os.path.basename(fixed_file), aspect)
    plot_2d.plot_overlays_reg(
        *imgs, *cmaps, translation, os.path.basename(fixed_file))

def _handle_transform_file(fixed_file, transform_param_map=None):
    base_name = _reg_out_path(fixed_file, "")
    filename = base_name + "transform.txt"
    param_map = None
    if transform_param_map is None:
        param_map = sitk.ReadParameterFile(filename)
    else:
        sitk.WriteParameterFile(transform_param_map[0], filename)
        param_map = transform_param_map[0]
    return param_map, None # TODO: not using translation parameters
    transform = np.array(param_map["TransformParameters"]).astype(np.float)
    spacing = np.array(param_map["Spacing"]).astype(np.float)
    len_spacing = len(spacing)
    #spacing = [16, 16, 20]
    translation = None
    # TODO: should parse the transforms into multiple dimensions
    if len(transform) == len_spacing:
        translation = np.divide(transform[0:len_spacing], spacing)
        print("transform: {}, spacing: {}, translation: {}"
              .format(transform, spacing, translation))
    else:
        print("Transform parameters do not match scaling dimensions")
    return param_map, translation

def _mirror_labels(img):
    """Mirror labels across the z plane.
    
    Assumes that the image is empty from the far z planes toward the middle 
    but not necessarily the exact middle. Finds the first plane that doesn't 
    have any intensity values and sets this position as the mirror plane.
    
    Args:
        img: Image in SimpleITK format.
    
    Returns:
        The mirrored image in the same dimensions, origin, and spacing as the 
        original image.
    """
    # TODO: check to make sure values don't get wrapped around if np.int32
    # max value is less than data max val
    img_np = sitk.GetArrayFromImage(img).astype(np.int32)
    tot_planes = len(img_np)
    i = tot_planes
    # need to work backward since the starting z-planes may also be empty
    for plane in img_np[::-1]:
        if not np.allclose(plane, 0):
            break
        i -= 1
    print("type: {}, max: {}, max avail: {}".format(
        img_np.dtype, np.max(img_np), np.iinfo(img_np.dtype).max))
    if i <= tot_planes and i >= 0:
        # if a empty planes at end, fill the empty space with the preceding 
        # planes in mirrored fashion
        remaining_planes = tot_planes - i
        end = i - remaining_planes
        if end < 0:
            end = 0
            remaining_planes = i
        print("i: {}, end: {}, remaining_planes: {}, tot_planes: {}"
              .format(i, end, remaining_planes, tot_planes))
        img_np[i:i+remaining_planes] = np.multiply(img_np[i-1:end-1:-1], -1)
    else:
        # skip mirroring if no planes are empty or only first plane is empty
        print("nothing to mirror")
        return img
    img_reflected = sitk.GetImageFromArray(img_np)
    img_reflected.SetSpacing(img.GetSpacing())
    img_reflected.SetOrigin(img.GetOrigin())
    return img_reflected

def transpose_img(img_sitk, plane, flip_horiz):
    img = sitk.GetArrayFromImage(img_sitk)
    spacing = img_sitk.GetSpacing()
    origin = img_sitk.GetOrigin()
    transposed = img
    if plane is not None and plane != plot_2d.PLANE[0]:
        # swap z-y to get (y, z, x) order for xz orientation
        transposed = np.swapaxes(transposed, 0, 1)
        # sitk convension is opposite of numpy with (x, y, z) order
        spacing = lib_clrbrain.swap_elements(spacing, 1, 2)
        origin = lib_clrbrain.swap_elements(origin, 1, 2)
        if plane == plot_2d.PLANE[1]:
            # rotate
            transposed = transposed[..., ::-1]
            transposed = np.swapaxes(transposed, 1, 2)
        elif plane == plot_2d.PLANE[2]:
            # swap new y-x to get (x, z, y) order for yz orientation
            transposed = np.swapaxes(transposed, 0, 2)
            spacing = lib_clrbrain.swap_elements(spacing, 0, 2)
            origin = lib_clrbrain.swap_elements(origin, 0, 2)
            # rotate
            transposed = np.swapaxes(transposed, 1, 2)
            spacing = lib_clrbrain.swap_elements(spacing, 0, 1)
        if plane == plot_2d.PLANE[1] or plane == plot_2d.PLANE[2]:
            # flip upside-down
            transposed[:] = np.flipud(transposed[:])
            if flip_horiz:
                transposed = transposed[..., ::-1]
        else:
            transposed[:] = transposed[:]
    transposed = sitk.GetImageFromArray(transposed)
    transposed.SetSpacing(spacing)
    transposed.SetOrigin(origin)
    return transposed

def register(fixed_file, moving_file_dir, flip_horiz=False, show_imgs=True, 
             write_imgs=False, name_prefix=None):
    """Registers two images to one another using the SimpleElastix library.
    
    Args:
        fixed_file: The image to register, given as a Numpy archive file to 
            be read by :importer:`read_file`.
        moving_file_dir: Directory of the atlas images, including the 
            main image and labels. The atlas was chosen as the moving file
            since it is likely to be lower resolution than the Numpy file.
    """
    if name_prefix is None:
        name_prefix = fixed_file
    image5d = importer.read_file(fixed_file, cli.series)
    roi = image5d[0, ...] # not using time dimension
    if flip_horiz:
        roi = roi[..., ::-1]
    fixed_img = sitk.GetImageFromArray(roi)
    spacing = detector.resolutions[0]
    #print("spacing: {}".format(spacing))
    fixed_img.SetSpacing(spacing[::-1])
    fixed_img.SetOrigin([0, 0, -roi.shape[0] // 2])
    #fixed_img = sitk.RescaleIntensity(fixed_img)
    #fixed_img = sitk.Cast(fixed_img, sitk.sitkUInt32)
    #sitk.Show(transpose_img(fixed_img, plot_2d.plane, flip_horiz))
    
    moving_file = os.path.join(moving_file_dir, IMG_ATLAS)
    moving_img = sitk.ReadImage(moving_file)
    
    print(fixed_img)
    print(moving_img)
    
    elastix_img_filter = sitk.ElastixImageFilter()
    elastix_img_filter.SetFixedImage(fixed_img)
    elastix_img_filter.SetMovingImage(moving_img)
    param_map_vector = sitk.VectorOfParameterMap()
    # translation to shift and rotate
    param_map = sitk.GetDefaultParameterMap("translation")
    param_map["MaximumNumberOfIterations"] = ["2048"]
    '''
    # TESTING: minimal registration
    param_map["MaximumNumberOfIterations"] = ["0"]
    '''
    param_map_vector.append(param_map)
    # affine to sheer and scale
    param_map = sitk.GetDefaultParameterMap("affine")
    param_map["MaximumNumberOfIterations"] = ["1024"]
    param_map_vector.append(param_map)
    # bspline for non-rigid deformation
    param_map = sitk.GetDefaultParameterMap("bspline")
    param_map["FinalGridSpacingInVoxels"] = ["50"]
    del param_map["FinalGridSpacingInPhysicalUnits"] # avoid conflict with vox
    #param_map["MaximumNumberOfIterations"] = ["512"]
    
    param_map_vector.append(param_map)
    elastix_img_filter.SetParameterMap(param_map_vector)
    elastix_img_filter.PrintParameterMap()
    transform = elastix_img_filter.Execute()
    transformed_img = elastix_img_filter.GetResultImage()
    
    # apply transformation to label files
    transform_param_map = elastix_img_filter.GetTransformParameterMap()
    transformix_img_filter = sitk.TransformixImageFilter()
    transformix_img_filter.SetTransformParameterMap(transform_param_map)
    img_files = (IMG_LABELS, )
    imgs_transformed = []
    for img_file in img_files:
        img = sitk.ReadImage(os.path.join(moving_file_dir, img_file))
        # ABA only gives half of atlas so need to mirror one side to other
        img = _mirror_labels(img)
        transformix_img_filter.SetMovingImage(img)
        transformix_img_filter.Execute()
        result_img = transformix_img_filter.GetResultImage()
        result_img = sitk.Cast(result_img, img.GetPixelID())
        imgs_transformed.append(result_img)
        print(result_img)
        '''
        LabelStatistics = sitk.LabelStatisticsImageFilter()
        LabelStatistics.Execute(fixed_img, result_img)
        count = LabelStatistics.GetCount(1)
        mean = LabelStatistics.GetMean(1)
        variance = LabelStatistics.GetVariance(1)
        print("count: {}, mean: {}, variance: {}".format(count, mean, variance))
        '''
    
    if show_imgs:
        # show individual SimpleITK images in default viewer
        sitk.Show(fixed_img)
        sitk.Show(moving_img)
        sitk.Show(transformed_img)
        for img in imgs_transformed:
            sitk.Show(img)
    
    if write_imgs:
        # write atlas and labels files, transposed according to plane setting
        imgs_names = (IMG_ATLAS, IMG_LABELS)
        imgs_write = [transformed_img, imgs_transformed[0]]
        for i in range(len(imgs_write)):
            out_path = _reg_out_path(name_prefix, imgs_names[i])
            img = transpose_img(imgs_write[i], plot_2d.plane, flip_horiz)
            print("writing {}".format(out_path))
            sitk.WriteImage(img, out_path, False)

    # show 2D overlay for registered images
    imgs = [
        roi, 
        sitk.GetArrayFromImage(moving_img), 
        sitk.GetArrayFromImage(transformed_img), 
        sitk.GetArrayFromImage(imgs_transformed[0])]
    # save transform parameters and attempt to find the original position 
    # that corresponds to the final position that will be displayed
    _, translation = _handle_transform_file(name_prefix, transform_param_map)
    translation = _translation_adjust(
        moving_img, transformed_img, translation, flip=True)
    
    # overlap stats
    overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
    '''
    # mean Dice Similarity Coefficient (DSC) of labeled regions;
    # not really applicable here since don't have moving labels;
    # fixed_img is 64-bit float (double), while transformed_img is 32-bit
    overlap_filter.Execute(sitk.Cast(fixed_img, sitk.sitkFloat32), transformed_img)
    mean_region_dsc = overlap_filter.GetDiceCoefficient()
    '''
    # Dice Similarity Coefficient (DSC) of total brain volume by applying 
    # simple binary mask for estimate of background vs foreground
    fixed_binary_img = sitk.BinaryThreshold(fixed_img, 0.01)
    transformed_binary_img = sitk.BinaryThreshold(transformed_img, 10.0)
    overlap_filter.Execute(fixed_binary_img, transformed_binary_img)
    #sitk.Show(fixed_binary_img)
    #sitk.Show(transformed_binary_img)
    total_dsc = overlap_filter.GetDiceCoefficient()
    #print("Mean regional DSC: {}".format(mean_region_dsc))
    print("Total DSC: {}".format(total_dsc))
    
    # show overlays last since blocks until fig is closed
    _show_overlays(imgs, translation, fixed_file)
    
def overlay_registered_imgs(fixed_file, moving_file_dir, flip_horiz=False, 
                            name_prefix=None):
    """Shows overlays of previously saved registered images.
    
    Should be run after :func:`register` has written out images in default
    (xy) orthogonal orientation.
    
    Args:
        fixed_file: Path to the fixed file.
        moving_file_dir: Moving files directory, from which the original
            atlas will be retrieved.
        flip_horiz: If true, will flip the fixed file horizontally first; 
            defaults to False.
        name_prefix: Path with base name where registered files are located; 
            defaults to None, in which case the fixed_file path will be used.
    """
    if name_prefix is None:
        name_prefix = fixed_file
    image5d = importer.read_file(fixed_file, cli.series)
    roi = image5d[0, ...] # not using time dimension
    if flip_horiz:
        roi = roi[..., ::-1]
    out_path = os.path.join(moving_file_dir, IMG_ATLAS)
    print("Reading in {}".format(out_path))
    moving_sitk = sitk.ReadImage(out_path)
    moving_img = sitk.GetArrayFromImage(moving_sitk)
    out_path = _reg_out_path(name_prefix, IMG_ATLAS)
    print("Reading in {}".format(out_path))
    transformed_sitk = sitk.ReadImage(out_path)
    transformed_img = sitk.GetArrayFromImage(transformed_sitk)
    out_path = _reg_out_path(name_prefix, IMG_LABELS)
    print("Reading in {}".format(out_path))
    labels_img = sitk.GetArrayFromImage(sitk.ReadImage(out_path))
    imgs = [roi, moving_img, transformed_img, labels_img]
    _, translation = _handle_transform_file(name_prefix)
    translation = _translation_adjust(
        moving_sitk, transformed_sitk, translation, flip=True)
    _show_overlays(imgs, translation, fixed_file)

def load_labels(fixed_file):
    labels_path = _reg_out_path(fixed_file, IMG_LABELS)
    labels_img = sitk.ReadImage(labels_path)
    print("loaded labels image from {}".format(labels_path))
    return sitk.GetArrayFromImage(labels_img)

def reg_scaling(image5d, reg):
    shape = image5d.shape
    if image5d.ndim >=4:
        shape = shape[1:4]
    scaling = np.divide(reg.shape[0:3], shape[0:3])
    print("registered image scaling compared to image5d: {}".format(scaling))
    return scaling

def load_labels_ref(path):
    labels_ref = None
    with open(path, "r") as f:
        labels_ref = json.load(f)
        #pprint(labels_ref)
    return labels_ref

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

def mirror_reverse_lookup(labels_ref, offset, name_modifier):
    # NOT CURRENTLY USED: replaced with neg values for mirrored side
    keys = list(labels_ref.keys())
    for key in keys:
        mirrored_key = key + offset
        mirrored_val = copy.deepcopy(labels_ref[key])
        node = mirrored_val[NODE]
        node[ABA_ID] = mirrored_key
        node[config.ABA_NAME] += name_modifier
        parent = node[ABA_PARENT]
        if parent is not None:
            node[ABA_PARENT] += offset
        try:
            parent_ids = mirrored_val[PARENT_IDS]
            parent_ids = np.add(parent_ids, offset).tolist()
        except KeyError as e:
            pass
        labels_ref[mirrored_key] = mirrored_val

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
        Reverse lookup dictionary as output by :func:`create_reverse_lookup`.
    """
    return create_reverse_lookup(labels_ref["msg"][0], ABA_ID, "children")

def get_label_ids_from_position(coord, labels_img, scaling):
    """Get the atlas label IDs for the given coordinates.
    
    Args:
        coord: Coordinates of experiment image in (z, y, x) order. Can be an 
            [n, 3] array of coordinates.
        labels_img: The registered image whose intensity values correspond to 
            label IDs.
        scaling: Scaling factor for the labels image size compared with the 
            experiment image.
    
    Returns:
        An array of label IDs corresponding to ``coords``, or a scalar of 
        one ID if only one coordinate is given.
    """
    coord_scaled = np.around(np.multiply(coord, scaling)).astype(np.int)
    #print(labels_img[tuple(coord_scaled[0])])
    #coord_scaled = coord_scaled[:, 1]#range(coord_scaled.shape[1])]
    coord_scaled = np.split(np.transpose(coord_scaled), coord_scaled.shape[0])
    '''
    if len(coord_scaled) > 1:
        coord_scaled = [row for row in np.transpose(coord_scaled)]
    '''
    print("coord_scaled: {}".format(coord_scaled))
    return labels_img[coord_scaled][0]

def get_label(coord, labels_img, labels_ref, scaling, level=None):
    """Get the atlas label for the given coordinates.
    
    Args:
        coord: Coordinates of experiment image in (z, y, x) order.
        labels_img: The registered image whose intensity values correspond to 
            label IDs.
        labels_ref: The labels reference lookup, assumed to be generated by 
            :func:`create_reverse_lookup` to look up by ID.
        scaling: Scaling factor for the labels image size compared with the 
            experiment image.
    
    Returns:
        The label dictionary at those coordinates, or None if no label is 
        found.
    """
    label_id = get_label_ids_from_position(coord, labels_img, scaling)
    print("label_id: {}".format(label_id))
    mirrored = label_id < 0
    if mirrored:
        label_id = -1 * label_id
    label = None
    try:
        label = labels_ref[label_id]
        if level is not None and label[NODE][ABA_LEVEL] > level:
            parents = label[PARENT_IDS]
            label = None
            if label_id < 0:
                parents = np.multiply(parents, -1)
            for parent in parents:
                parent_label = labels_ref[parent]
                if parent_label[NODE][ABA_LEVEL] == level:
                    label = parent_label
                    break
        if label is not None:
            label[MIRRORED] = mirrored
            print("label: {}".format(label_id))
    except KeyError as e:
        print("could not find label id {} or it parent (error {})"
              .format(label_id, e))
    return label

def get_label_name(label):
    """Get the atlas region name from the label.
    
    Args:
        label: The label dictionary.
    
    Returns:
        The atlas region name, or None if not found.
    """
    name = None
    try:
        if label is not None:
            node = label[NODE]
            if node is not None:
                name = node[config.ABA_NAME]
                print("name: {}".format(name))
                if label[MIRRORED]:
                    name += LEFT_SUFFIX
                else:
                    name += RIGHT_SUFFIX
    except KeyError as e:
        print(e, name)
    return name

def volumes_by_id(labels_img, labels_ref, scaling, resolution, level=None):
    """Get volumes by labels IDs.
    
    Args:
        labels_img: The registered image whose intensity values correspond to 
            label IDs.
        labels_ref: The labels reference lookup, assumed to be an OrderedDict 
            generated by :func:`create_reverse_lookup` to look up by ID 
            while preserving key order to ensure that parents of any child 
            will be reached prior to the child.
        scaling: Scaling factor for the labels image size compared with the 
            experiment image.
        resolutions: The image resolutions as an array in (z, y, x) order, 
            typically an element from :attr:`detector.resolutions`.
        level: The ontology level as an integer; defaults to None. If None, 
            volumes for all ontology levels will be returned. If a level is 
            given, only regions from that level will be returned, while 
            children will be collapsed into the parent at that level, and 
            regions above this level will be ignored.
    
    Returns:
        Nested dictionary of {ID: {:attr:`config.ABA_NAME`: name, 
        :attr:`config.VOL_KEY`: volume}}, where volume is in the cubed units of 
        :attr:`detector.resolutions`.
    """
    ids = list(labels_ref.keys())
    #print("ids: {}".format(ids))
    volumes_dict = {}
    scaling_res = np.multiply(scaling, resolution)
    scaling_vol = scaling_res[0] * scaling_res[1] * scaling_res[2]
    for key in ids:
        label_ids = [key, -1 * key]
        for label_id in label_ids:
            label = labels_ref[key] # always use pos val
            region = labels_img[labels_img == label_id]
            vol = len(region) * scaling_vol
            #print("checking id {} with vol {}".format(label_id, vol))
            if level is None or label[NODE][ABA_LEVEL] == level:
                region_dict = {
                    config.ABA_NAME: label[NODE][config.ABA_NAME],
                    config.VOL_KEY: vol
                }
                volumes_dict[label_id] = region_dict
            else:
                parents = label.get(PARENT_IDS)
                if parents is not None:
                    if label_id < 0:
                        parents = np.multiply(parents, -1)
                    for parent in parents:
                        region_dict = volumes_dict.get(parent)
                        if region_dict is not None:
                            region_dict[config.VOL_KEY] += vol
                            print("added vol {} from {} (id {}) to {}".format(
                                  vol, label[NODE][config.ABA_NAME], label_id, 
                                  region_dict[config.ABA_NAME]))
    for key in volumes_dict.keys():
        if key >= 0:
            print("{} (id {}), volume{}: {}, volume{}: {}, ".format(
                volumes_dict[key][config.ABA_NAME], key, 
                RIGHT_SUFFIX, volumes_dict[key][config.VOL_KEY], 
                LEFT_SUFFIX, volumes_dict[-1 * key][config.VOL_KEY]))
    return volumes_dict

def _test_labels_lookup():
    """Test labels reverse dictionary creation and lookup.
    """
    
    # create reverse lookup dictionary
    ref = load_labels_ref(config.load_labels)
    #pprint(ref)
    lookup_id = 15565 # short search path
    #lookup_id = 126652058 # last item
    time_dict_start = time()
    id_dict = create_aba_reverse_lookup(ref)
    labels_img = load_labels(cli.filename)
    max_labels = np.max(labels_img)
    print("max_labels: {}".format(max_labels))
    #mirror_reverse_lookup(id_dict, max_labels, " (R)")
    #pprint(id_dict)
    time_dict_end = time()
    time_node_start = time()
    found = id_dict[lookup_id]
    time_node_end = time()
    print("found {}: {} with parents {}".format(lookup_id, found[NODE]["name"], found[PARENT_IDS]))
    
    # brute-force query
    time_direct_start = time()
    node = get_node(ref["msg"][0], "id", lookup_id, "children")
    time_direct_end = time()
    #print(node)
    
    print("time to create id_dict (s): {}".format(time_dict_end - time_dict_start))
    print("time to find node (s): {}".format(time_node_end - time_node_start))
    print("time to find node directly (s): {}".format(time_direct_end - time_direct_start))
    
    # get volumes for each ID
    print("labels_img shape: {}".format(labels_img.shape))
    scaling = np.ones(3) * 0.05
    volumes_dict = volumes_by_id(labels_img, id_dict, scaling, [4.935,  0.913, 0.913], level=2)
    plot_2d.plot_volumes(volumes_dict, ignore_empty=True)
    
    # get a list of IDs corresponding to each blob
    blobs = np.array([[300, 5000, 8000], [350, 5500, 4500], [400, 6000, 5000]])
    ids = get_label_ids_from_position(blobs[:, 0:3], labels_img, scaling)
    print("blob IDs:\n{}".format(ids))

if __name__ == "__main__":
    print("Clrbrain image registration")
    cli.main(True)
    # run with --plane xy to generate non-transposed images before comparing 
    # orthogonal views in overlay_registered_imgs, then run with --plane xz
    # to re-transpose to original orientation for mapping locations
    prefix = None
    if len(cli.filenames) >= 3:
        prefix = cli.filenames[2]
    flip = config.flip_horiz
    register(*cli.filenames[0:2], flip_horiz=flip, write_imgs=True, name_prefix=prefix)
    #register(*cli.filenames[0:2], flip_horiz=flip, show_imgs=False)
    for plane in plot_2d.PLANE:
        plot_2d.plane = plane
        #overlay_registered_imgs(*cli.filenames[0:2], flip_horiz=flip, name_prefix=prefix)
    #_test_labels_lookup()
