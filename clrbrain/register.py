#!/bin/bash
# Image registration
# Author: David Young, 2017, 2018
"""Register images to one another.
"""

import os
import copy
import json
import multiprocessing as mp
from collections import OrderedDict
from pprint import pprint
from time import time
try:
    import SimpleITK as sitk
except ImportError as e:
    print(e)
    print("WARNING: SimpleElastix could not be found, so there will be error "
          "when attempting to register images or load registered images")
import numpy as np
from skimage import measure
from skimage import transform

from clrbrain import cli
from clrbrain import config
from clrbrain import detector
from clrbrain import importer
from clrbrain import lib_clrbrain
from clrbrain import plot_2d

IMG_ATLAS = "atlasVolume.mhd"
IMG_LABELS = "annotation.mhd"
IMG_GROUPED = "grouped.mhd"

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

def _show_overlays(imgs, translation, fixed_file, plane):
    """Shows overlays via :func:plot_2d:`plot_overlays_reg`.
    
    Args:
        imgs: List of images in Numpy format
        translation: Translation in (z, y, x) format for Numpy consistency.
        fixed_file: Path to fixed file to get title.
        plane: Planar transposition.
    """
    cmaps = ["Blues", "Oranges", "prism"]
    #plot_2d.plot_overlays(imgs, z, cmaps, os.path.basename(fixed_file), aspect)
    plot_2d.plot_overlays_reg(
        *imgs, *cmaps, translation, os.path.basename(fixed_file), plane)

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

def _get_bbox(img_np, threshold=10):
    labels, walker = detector.segment_rw(img_np, vmin=threshold, vmax=threshold)
    labels_props = measure.regionprops(labels, walker)
    if len(labels_props) < 1:
        return None
    labels_bbox = labels_props[0].bbox
    #print("bbox: {}".format(labels_bbox))
    return labels_bbox

def _get_bbox_region(bbox):
    shape = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    slices = (slice(bbox[0], bbox[0] + shape[0]),
              slice(bbox[1], bbox[1] + shape[1]))
    #print("shape: {}, slices: {}".format(shape, slices))
    return shape, slices

def _mirror_labels(img, img_ref):
    """Mirror labels across sagittal midline and add lateral edges.
    
    Assume that the image is in sagittal sections and consists of only one 
    hemisphere, empty from the far z planes toward the middle but not 
    necessarily the exact middle of the image. Find the first plane that 
    doesn't have any intensity values and set this position as the mirror 
    plane.
    
    Also assume that the later edges of the image are also missing. Build 
    edges that match the size of the reference image on one side and mirror 
    over to the other side.
    
    Args:
        img: Labels image in SimpleITK format.
        img: Reference atlas image in SimpleITK format.
    
    Returns:
        The mirrored image in the same dimensions, origin, and spacing as the 
        original image.
    """
    # TODO: check to make sure values don't get wrapped around if np.int32
    # max value is less than data max val
    img_np = sitk.GetArrayFromImage(img).astype(np.int32)
    img_ref_np = sitk.GetArrayFromImage(img_ref).astype(np.int32)
    tot_planes = len(img_np)
    
    # find the first non-zero plane
    i = 0
    for plane in img_np:
        if not np.allclose(plane, 0):
            print("found first non-zero plane at i of {}".format(i))
            break
        i += 1
    
    # find the bounds of the reference image in the given plane and resize 
    # the corresponding section of the labels image to the bounds of the 
    # reference image in the next plane closer to the edge; essentially 
    # extends the last edge labels plane, but 
    # TODO: remove ventricular from this last edge before extending
    bbox_last = _get_bbox(img_ref_np[i])
    while i > 0 and len(img_ref_np[i - 1] >= 10) > 0 and bbox_last is not None:
        shape_last, slices_last = _get_bbox_region(bbox_last)
        plane_region = img_np[i, slices_last[0], slices_last[1]]
        #print("plane_region max: {}".format(np.max(plane_region)))
        bbox = _get_bbox(img_ref_np[i - 1])
        if bbox is not None:
            shape, slices = _get_bbox_region(bbox)
            # assume that the reference image background is about < 10, the 
            # default threshold
            plane_region = transform.resize(
                plane_region, shape, preserve_range=True)
            #print("plane_region max: {}".format(np.max(plane_region)))
            img_np[i - 1, slices[0], slices[1]] = plane_region
        bbox_last = bbox
        i -= 1
    
    # find the last non-zero plane
    i = tot_planes
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

def transpose_img(img_sitk, plane, rotate=False):
    """Transpose a SimpleITK format image via Numpy and re-export to SimpleITK.
    
    Args:
        img_sitk: Image in SimpleITK format.
        plane: One of :attr:``plot_2d.PLANES`` elements, specifying the 
            planar orientation in which to transpose the image. The current 
            orientation is taken to be "xy".
        rotate: Rotate the final output image by 180 degrees; defaults to False.
    
    Returns:
        Transposed image in SimpleITK format.
    """
    img = sitk.GetArrayFromImage(img_sitk)
    spacing = img_sitk.GetSpacing()
    origin = img_sitk.GetOrigin()
    transposed = img
    if plane is not None and plane != plot_2d.PLANE[0]:
        # swap z-y to get (y, z, x) order for xz orientation
        transposed = np.swapaxes(transposed, 0, 1)
        # sitk convention is opposite of numpy with (x, y, z) order
        spacing = lib_clrbrain.swap_elements(spacing, 1, 2)
        origin = lib_clrbrain.swap_elements(origin, 1, 2)
        if plane == plot_2d.PLANE[1]:
            # rotate
            transposed = transposed[..., ::-1]
            transposed = np.swapaxes(transposed, 1, 2)
            spacing = lib_clrbrain.swap_elements(spacing, 0, 1)
            origin = lib_clrbrain.swap_elements(origin, 0, 1)
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
        else:
            transposed[:] = transposed[:]
    if rotate:
        # rotate the final output image by 180 deg
        # TODO: need to change origin? make axes accessible (eg (0, 2) for 
        # horizontal rotation)
        transposed = np.rot90(transposed, 2, (1, 2))
    transposed = sitk.GetImageFromArray(transposed)
    transposed.SetSpacing(spacing)
    transposed.SetOrigin(origin)
    return transposed

def _load_numpy_to_sitk(numpy_file, rotate=False, size=None):
    image5d = importer.read_file(numpy_file, cli.series)
    roi = image5d[0, ...] # not using time dimension
    if size is not None:
        roi = transform.resize(roi, size)#, anti_aliasing=True)
    if rotate:
        roi = np.rot90(roi, 2, (1, 2))
    sitk_img = sitk.GetImageFromArray(roi)
    spacing = detector.resolutions[0]
    #print("spacing: {}".format(spacing))
    sitk_img.SetSpacing(spacing[::-1])
    sitk_img.SetOrigin([0, 0, -roi.shape[0] // 2])
    return sitk_img

def register(fixed_file, moving_file_dir, plane=None, flip=False, 
             show_imgs=True, write_imgs=True, name_prefix=None):
    """Registers two images to one another using the SimpleElastix library.
    
    Args:
        fixed_file: The image to register, given as a Numpy archive file to 
            be read by :importer:`read_file`.
        moving_file_dir: Directory of the atlas images, including the 
            main image and labels. The atlas was chosen as the moving file
            since it is likely to be lower resolution than the Numpy file.
        plane: Planar orientation to which the atlas will be transposed, 
            considering the atlas' original plane as "xy".
        flip: True if the atlas should be flipped/rotated; defaults to False.
        show_imgs: True if the output images should be displayed; defaults to 
            True.
        write_imgs: True if the images should be written to file; defaults to 
            False.
        name_prefix: Path with base name where registered files are located; 
            defaults to None, in which case the fixed_file path will be used.
    """
    if name_prefix is None:
        name_prefix = fixed_file
    fixed_img = _load_numpy_to_sitk(fixed_file)
    '''
    # for some reason cannot load the .mhd file directly for the fixed file 
    # or else get "MovingImage is not present" error, whereas loading the 
    # fixed file from a Numpy array avoids the error
    if isinstance(fixed_file, str):
        if fixed_file.endswith(".mhd"):
            fixed_img = sitk.ReadImage(fixed_file)
        else:
            fixed_img = _load_numpy_to_sitk(fixed_file)
    else:
        fixed_img = fixed_file
    '''
    moving_file = os.path.join(moving_file_dir, IMG_ATLAS)
    moving_img = sitk.ReadImage(moving_file)
    moving_img = transpose_img(moving_img, plane, flip)
    
    print("fixed image from {}:\n{}".format(fixed_file, fixed_img))
    print("moving image from {}:\n{}".format(moving_file, moving_img))
    
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
    img_files = (IMG_LABELS, )
    transform_param_map = elastix_img_filter.GetTransformParameterMap()
    transformix_img_filter = sitk.TransformixImageFilter()
    transformix_img_filter.SetTransformParameterMap(transform_param_map)
    imgs_transformed = []
    for img_file in img_files:
        img = sitk.ReadImage(os.path.join(moving_file_dir, img_file))
        # ABA only gives half of atlas so need to mirror one side to other
        img = _mirror_labels(img)
        img = transpose_img(img, plot_2d.plane, flip)
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
        '''
        # TESTING: transpose saved images to new orientation
        img = sitk.ReadImage(_reg_out_path(name_prefix, IMG_LABELS))
        imgs_names = ("test.mhd", )
        imgs_write = [img]
        '''
        for i in range(len(imgs_write)):
            out_path = _reg_out_path(name_prefix, imgs_names[i])
            print("writing {}".format(out_path))
            sitk.WriteImage(imgs_write[i], out_path, False)

    # show 2D overlay for registered images
    imgs = [
        sitk.GetArrayFromImage(fixed_img),
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
    _show_overlays(imgs, translation, fixed_file, None)
    
def register_group(img_files, flip=None, show_imgs=True, 
             write_imgs=True, name_prefix=None):
    """Group registers several images to one another.
    
    Args:
        img_files: Paths to image files to register.
        flip: Boolean list corresponding to ``img_files`` flagging 
            whether to flip the image or not; defaults to None, in which 
            case no images will be flipped.
        show_imgs: True if the output images should be displayed; defaults to 
            True.
        write_imgs: True if the images should be written to file; defaults to 
            True.
        name_prefix: Path with base name where registered files are located; 
            defaults to None, in which case the fixed_file path will be used.
    """
    if name_prefix is None:
        name_prefix = img_files[0]
    img_vector = sitk.VectorOfImage()
    flip_img = False
    origin = None
    size = None
    for i in range(len(img_files)):
        img_file = img_files[i]
        if flip is not None:
            flip_img = flip[i]
        # force all images into same size and origin as first image 
        # to avoid groupwise registration error on physical space mismatch
        img = _load_numpy_to_sitk(img_file, flip_img, size)
        if origin is None:
            origin = img.GetOrigin()
            size = img.GetSize()[::-1]
        else:
            img.SetOrigin(origin)
        print("img_file:\n{}".format(img))
        img_vector.push_back(img)
        #sitk.Show(img)
    #sitk.ProcessObject.SetGlobalDefaultDirectionTolerance(1)
    #sitk.ProcessObject.SetGlobalDefaultCoordinateTolerance(100)
    img_combined = sitk.JoinSeries(img_vector)
    
    elastix_img_filter = sitk.ElastixImageFilter()
    elastix_img_filter.SetFixedImage(img_combined)
    elastix_img_filter.SetMovingImage(img_combined)
    param_map = sitk.GetDefaultParameterMap("groupwise")
    param_map["FinalGridSpacingInVoxels"] = ["50"]
    del param_map["FinalGridSpacingInPhysicalUnits"] # avoid conflict with vox
    # TESTING:
    #param_map["MaximumNumberOfIterations"] = ["0"]
    elastix_img_filter.SetParameterMap(param_map)
    elastix_img_filter.PrintParameterMap()
    transform = elastix_img_filter.Execute()
    transformed_img = elastix_img_filter.GetResultImage()
    
    if show_imgs:
        sitk.Show(transformed_img)
    
    if write_imgs:
        # write both the .mhd and Numpy array files
        out_path = name_prefix + IMG_GROUPED
        print("writing {}".format(out_path))
        sitk.WriteImage(transformed_img, out_path, False)
        img_np = sitk.GetArrayFromImage(transformed_img)
        importer.save_np_image(img_np, out_path, cli.series)
    
def overlay_registered_imgs(fixed_file, moving_file_dir, plane=None, 
                            flip=False, name_prefix=None, out_plane=None):
    """Shows overlays of previously saved registered images.
    
    Should be run after :func:`register` has written out images in default
    (xy) orthogonal orientation.
    
    Args:
        fixed_file: Path to the fixed file.
        moving_file_dir: Moving files directory, from which the original
            atlas will be retrieved.
        flip: If true, will flip the fixed file first; defaults to False.
        name_prefix: Path with base name where registered files are located; 
            defaults to None, in which case the fixed_file path will be used.
    """
    # get the experiment file
    if name_prefix is None:
        name_prefix = fixed_file
    image5d = importer.read_file(fixed_file, cli.series)
    roi = image5d[0, ...] # not using time dimension
    
    # get the atlas file and transpose it to match the orientation of the 
    # experiment image
    out_path = os.path.join(moving_file_dir, IMG_ATLAS)
    print("Reading in {}".format(out_path))
    moving_sitk = sitk.ReadImage(out_path)
    moving_sitk = transpose_img(moving_sitk, plane, flip)
    moving_img = sitk.GetArrayFromImage(moving_sitk)
    
    # get the registered atlas file, which should already be transposed
    out_path = _reg_out_path(name_prefix, IMG_ATLAS)
    print("Reading in {}".format(out_path))
    transformed_sitk = sitk.ReadImage(out_path)
    transformed_img = sitk.GetArrayFromImage(transformed_sitk)
    
    # get the registered labels file, which should also already be transposed
    out_path = _reg_out_path(name_prefix, IMG_LABELS)
    print("Reading in {}".format(out_path))
    labels_img = sitk.GetArrayFromImage(sitk.ReadImage(out_path))
    
    # overlay the images
    imgs = [roi, moving_img, transformed_img, labels_img]
    _, translation = _handle_transform_file(name_prefix)
    translation = _translation_adjust(
        moving_sitk, transformed_sitk, translation, flip=True)
    _show_overlays(imgs, translation, fixed_file, out_plane)

def load_labels(fixed_file, get_sitk=False):
    labels_path = _reg_out_path(fixed_file, IMG_LABELS)
    labels_img = sitk.ReadImage(labels_path)
    print("loaded labels image from {}".format(labels_path))
    if get_sitk:
        return labels_img
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
    # scale coordinates to atlas image size
    coord_scaled = np.multiply(coord, scaling).astype(np.int)
    
    # split coordinates into lists by dimension to index the labels image
    # at once
    coord_scaled = np.transpose(coord_scaled)
    coord_scaled = np.split(coord_scaled, coord_scaled.shape[0])
    #print("coord_scaled: {}".format(coord_scaled))
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

def volumes_by_id(labels_img, labels_ref_lookup, resolution, level=None, 
                  blobs_ids=None):
    """Get volumes by labels IDs.
    
    Args:
        labels_img: The registered image whose intensity values correspond to 
            label IDs.
        labels_ref_lookup: The labels reference lookup, assumed to be an 
            OrderedDict generated by :func:`create_reverse_lookup` to look up 
            by ID while preserving key order to ensure that parents of any 
            child will be reached prior to the child.
        resolution: The image resolutions as an array in (z, y, x) order, 
            typically the spacing of the SimpleITK image, not the 
            original image's resolution.
        level: The ontology level as an integer; defaults to None. If None, 
            volumes for all ontology levels will be returned. If a level is 
            given, only regions from that level will be returned, while 
            children will be collapsed into the parent at that level, and 
            regions above this level will be ignored.
        blobs_ids: List of label IDs for blobs. If None, blob densities will 
            not be calculated.
    
    Returns:
        Nested dictionary of {ID: {:attr:`config.ABA_NAME`: name, 
        :attr:`config.VOL_KEY`: volume, 
        :attr:`config.BLOBS_KEY`: number of blobs}}, where volume is in the 
        cubed units of :attr:`detector.resolutions`.
    """
    ids = list(labels_ref_lookup.keys())
    #print("ids: {}".format(ids))
    volumes_dict = {}
    scaling_vol = np.prod(resolution)
    for key in ids:
        label_ids = [key, -1 * key]
        for label_id in label_ids:
            label = labels_ref_lookup[key] # always use pos val
            region = labels_img[labels_img == label_id]
            vol = len(region) * scaling_vol
            blobs = None
            blobs_len = 0
            if blobs_ids is not None:
                blobs = blobs_ids[blobs_ids == label_id]
                blobs_len = len(blobs)
            #print("checking id {} with vol {}".format(label_id, vol))
            label_level = label[NODE][ABA_LEVEL]
            name = label[NODE][config.ABA_NAME]
            # include region in volumes dict if at the given level, no level 
            # specified, or at the default (organism) level, which is used 
            # to catch all children without a parent at the given level
            if level is None or label_level == level or label_level == -1:
                region_dict = {
                    config.ABA_NAME: label[NODE][config.ABA_NAME],
                    config.VOL_KEY: vol,
                    config.BLOBS_KEY: blobs_len
                }
                volumes_dict[label_id] = region_dict
                print("inserting region {} (id {}) with {} vol and {} blobs "
                      .format(name, label_id, vol, blobs_len))
            else:
                parents = label.get(PARENT_IDS)
                if parents is not None:
                    if label_id < 0:
                        parents = np.multiply(parents, -1)
                    # start from last parent to avoid level -1 unless no 
                    # parent found and stop checking as soon as parent found
                    found_parent = False
                    for parent in parents[::-1]:
                        region_dict = volumes_dict.get(parent)
                        if region_dict is not None:
                            region_dict[config.VOL_KEY] += vol
                            region_dict[config.BLOBS_KEY] += blobs_len
                            print("added {} vol and {} blobs from {} (id {}) "
                                  "to {}".format(vol, blobs_len, name, 
                                  label_id, region_dict[config.ABA_NAME]))
                            found_parent = True
                            break
                    if not found_parent:
                        print("could not find parent for {} with blobs {}"
                              .format(label_id, blobs_len))
    # blobs summary
    blobs_tot = 0
    for key in volumes_dict.keys():
        # all blobs matched to a region at the given level; 
        # TODO: find blobs that had a label (ie not 0) but did not match any 
        # parent, including -1
        if key >= 0:
            blobs_side = volumes_dict[key][config.BLOBS_KEY]
            blobs_mirrored = volumes_dict[-1 * key][config.BLOBS_KEY]
            print("{} (id {}), {}: volume {}, blobs {}; {}: volume {}, blobs {}"
                .format(volumes_dict[key][config.ABA_NAME], key, 
                RIGHT_SUFFIX, volumes_dict[key][config.VOL_KEY], blobs_side, 
                LEFT_SUFFIX, volumes_dict[-1 * key][config.VOL_KEY], 
                blobs_mirrored))
            blobs_tot += blobs_side + blobs_mirrored
    # all unlabeled blobs
    blobs_unlabeled = blobs_ids[blobs_ids == 0]
    blobs_unlabeled_len = len(blobs_unlabeled)
    print("unlabeled blobs (id 0): {}".format(blobs_unlabeled_len))
    blobs_tot += blobs_unlabeled_len
    print("total blobs accounted for: {}".format(blobs_tot))
    return volumes_dict

def get_volumes_dict_path(img_path, level):
    return "{}_volumes_level{}.json".format(os.path.splitext(img_path)[0], level)

def register_volumes(img_path, labels_ref_lookup, level, densities=False):
    """Register volumes and densities.
    
    If a volumes dictionary from the path generated by 
    :func:``get_volumes_dict_path`` exists, this dictionary will be loaded 
    instead
    
    Args:
        img_path: Path to the original image file.
        labels_path: Path to the registered labels image file.
        level: Ontology level at which to show volumes and densities.
        densities: True if densities should be displayed; defaults to False.
    
    Returns:
        The volumes dictionary in the format of {[ID as int]: 
        {name: [name as str], volume: [volume as float], blobs: 
        [blobs as int]}, ...}.
    """
    volumes_dict = None
    json_path = get_volumes_dict_path(img_path, level)
    if os.path.isfile(json_path):
        # reload saved volumes dictionary
        with open(json_path, "r") as fp:
            print("reloading saved volumes dict from {}".format(json_path))
            volumes_dict = json.load(
                fp, object_hook=lib_clrbrain.convert_indices_to_int)
            print(volumes_dict)
    else:
        # load labels image and setup labels dictionary
        labels_img_sitk = load_labels(img_path, get_sitk=True)
        scaling = labels_img_sitk.GetSpacing()
        labels_img = sitk.GetArrayFromImage(labels_img_sitk)
        print("labels_img shape: {}".format(labels_img.shape))
        
        # load blob densities by region if flagged
        blobs_ids = None
        if densities:
            # load blobs
            filename_base = importer.filename_to_base(img_path, cli.series)
            info = np.load(filename_base + cli.SUFFIX_INFO_PROC)
            blobs = info["segments"]
            print("loading {} blobs".format(len(blobs)))
            # load image just to get resolutions
            image5d = importer.read_file(img_path, cli.series)
            # annotate blobs based on position
            blobs_ids = get_label_ids_from_position(
                blobs[:, 0:3], labels_img, reg_scaling(image5d, labels_img))
            print(blobs_ids)
        
        # calculate and plot volumes and densities
        volumes_dict = volumes_by_id(
            labels_img, labels_ref_lookup, scaling, level=level, 
            blobs_ids=blobs_ids)
        print(volumes_dict)
        with open(json_path, "w") as fp:
            json.dump(volumes_dict, fp)
    return volumes_dict, json_path

def register_volumes_mp(img_paths, labels_ref_lookup, level, densities=False):
    start_time = time()
    '''
    for img_path in img_paths:
        register_volumes(img_path, labels_path, level, densities)
    '''
    pool = mp.Pool()
    pool_results = []
    for img_path in img_paths:
        pool_results.append(pool.apply_async(
            register_volumes, 
            args=(img_path, labels_ref_lookup, level, densities)))
    vols = []
    paths = []
    for result in pool_results:
        vol_dict, path = result.get()
        vols.append(vol_dict)
        paths.append(path)
        print("finished {}".format(path))
    pool.close()
    pool.join()
    print("time elapsed for volumes by ID: {}".format(time() - start_time))
    return vols, paths

def group_volumes(labels_ref_lookup, vol_dicts):
    """Group volumes from multiple volume dictionaries.
    
    Args:
        labels_ref_lookup: The labels reference lookup, assumed to be an 
            OrderedDict generated by :func:`create_reverse_lookup` to look up 
            by ID while preserving key order to ensure that parents of any 
            child will be reached prior to the child.
        vol_dicts: List of volume dictionaries, from which values from 
            identical IDs will be pooled.
    Returns:
        Grouped volumes dictionaries with a set of IDs containing all of the 
        unique IDs in the individual volume dictionaries and lists of values 
        in places of individual values.
    """
    ids = list(labels_ref_lookup.keys())
    grouped_vol_dict = {}
    for key in ids:
        # check all IDs, including negative versions for mirrored labels
        label_ids = [key, -1 * key]
        for label_id in label_ids:
            # entry associated with the ID, which should be identical for 
            # each dictionary except the numerical values
            entry_group = None
            for vol_dict in vol_dicts:
                entry_vol = vol_dict.get(label_id)
                if entry_vol is not None:
                    if entry_group is None:
                        # shallow copy since only immutable values
                        entry_group = dict(entry_vol)
                        entry_group[config.VOL_KEY] = [
                            entry_group[config.VOL_KEY]]
                        entry_group[config.BLOBS_KEY] = [
                            entry_group[config.BLOBS_KEY]]
                        grouped_vol_dict[label_id] = entry_group
                    else:
                        # append numerical values to existing lists
                        entry_group[config.VOL_KEY].append(
                            entry_vol[config.VOL_KEY])
                        entry_group[config.BLOBS_KEY].append(
                            entry_vol[config.BLOBS_KEY])
    print(grouped_vol_dict)
    return grouped_vol_dict

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
    
    # get a list of IDs corresponding to each blob
    blobs = np.array([[300, 5000, 8000], [350, 5500, 4500], [400, 6000, 5000]])
    ids = get_label_ids_from_position(blobs[:, 0:3], labels_img, np.ones(3) * config.labels_scaling)
    print("blob IDs:\n{}".format(ids))

def _test_mirror_labels(moving_file_dir):
    img = sitk.ReadImage(os.path.join(moving_file_dir, IMG_LABELS))
    img_ref = sitk.ReadImage(os.path.join(moving_file_dir, IMG_ATLAS))
    img = _mirror_labels(img, img_ref)
    sitk.Show(img)

if __name__ == "__main__":
    print("Clrbrain image registration")
    cli.main(True)
    # name prefix to use a different name from the input files, such as when 
    # registering transposed/scaled images but outputting paths corresponding 
    # to the original image
    prefix = None
    if len(cli.filenames) >= 3:
        prefix = cli.filenames[2]
        print("Formatting registered filenames to match {}".format(prefix))
    flip = False
    if config.flip is not None:
        flip = config.flip[0]
    
    #_test_labels_lookup()
    #_test_mirror_labels(cli.filenames[1])
    #os._exit(os.EX_OK)
    
    if config.register_type is None:
        # explicitly require a registration type
        print("Please choose a registration type")
    elif config.register_type == config.REGISTER_TYPES[0]:
        # "single", basic registration of 1st to 2nd image, transposing the 
        # second image according to plot_2d.plane and config.flip_horiz
        register(*cli.filenames[0:2], plane=plot_2d.plane, 
                 flip=flip, name_prefix=prefix)
    elif config.register_type == config.REGISTER_TYPES[1]:
        # groupwise registration, which assumes that the last image 
        # filename given is the prefix and uses the full flip array
        prefix = cli.filenames[-1]
        register_group(cli.filenames[:-1], flip=config.flip, name_prefix=prefix)
    elif config.register_type == config.REGISTER_TYPES[2]:
        # overlay registered images in each orthogonal plane
        for out_plane in plot_2d.PLANE:
            overlay_registered_imgs(
                *cli.filenames[0:2], plane=plot_2d.plane, 
                flip=flip, name_prefix=prefix, 
                out_plane=out_plane)
    elif config.register_type in (
        config.REGISTER_TYPES[3], config.REGISTER_TYPES[4]):
        # compute grouped volumes/densities by ontology level
        densities = config.register_type == config.REGISTER_TYPES[4]
        ref = load_labels_ref(config.load_labels)
        labels_ref_lookup = create_aba_reverse_lookup(ref)
        vol_dicts, json_paths = register_volumes_mp(
            cli.filenames, labels_ref_lookup, 
            config.labels_level, densities)
        # experiment identifiers, assumed to be at the start of the image 
        # filename, separated by a "-"; if no dash, will use the whole name
        show = not config.no_show
        exps = []
        for vol, path in zip(vol_dicts, json_paths):
            exp_name = os.path.basename(path)
            plot_2d.plot_volumes(
                vol, ignore_empty=True, 
                title=os.path.splitext(exp_name)[0], 
                densities=densities, show=show)
            exps.append(exp_name.split("-")[0])
        group_vol_dict = group_volumes(labels_ref_lookup, vol_dicts)
        plot_2d.plot_volumes(
            group_vol_dict, ignore_empty=True, 
            title="Volume Means from {} at Level {}".format(
                ", ".join(exps), config.labels_level), 
            densities=densities, show=show, multiple=True)
