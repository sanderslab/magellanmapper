#!/bin/bash
# Image registration
# Author: David Young, 2017
"""Register images to one another.
"""

import os
import SimpleITK as sitk
import numpy as np

from clrbrain import cli
from clrbrain import detector
from clrbrain import importer
from clrbrain import lib_clrbrain
from clrbrain import plot_2d

IMG_ATLAS = "atlasVolume.mhd"
IMG_LABELS = "annotation.mhd"
_REG_MOD = "_reg"

def _reg_out_path(base_path, base_name):
    return os.path.join(
        base_path, lib_clrbrain.insert_before_ext(base_name, _REG_MOD))

def _translation_adjust(orig, transformed, translation):
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
    translation = None # TODO: not using translation parameters for now
    plot_2d.plot_overlays_reg(*imgs, *cmaps, translation, os.path.basename(fixed_file))

def _handle_transform_file(fixed_file, transform_param_map=None):
    filename = fixed_file.rsplit(".", 1)[0] + "_transform.txt"
    param_map = None
    if transform_param_map is None:
        param_map = sitk.ReadParameterFile(filename)
    else:
        sitk.WriteParameterFile(transform_param_map[0], filename)
        param_map = transform_param_map[0]
    transform = np.array(param_map["TransformParameters"]).astype(np.float)
    spacing = np.array(param_map["Spacing"]).astype(np.float)
    #spacing = [16, 16, 20]
    translation = np.divide(transform, spacing)
    print("transform: {}, spacing: {}, translation: {}"
          .format(transform, spacing, translation))
    return param_map, translation

def register(fixed_file, moving_file_dir, flip_horiz=False, show_imgs=True, 
             write_imgs=False):
    """Registers two images to one another using the SimpleElastix library.
    
    Args:
        fixed_file: The image to register, given as a Numpy archive file to 
            be read by :importer:`read_file`.
        moving_file_dir: Directory of the atlas images, including the 
            main image and labels. The atlas was chosen as the moving file
            since it is likely to be lower resolution than the Numpy file.
    """
    image5d = importer.read_file(fixed_file, cli.series)
    roi = image5d[0, ...] # not using time dimension
    if flip_horiz:
        roi = roi[..., ::-1]
    fixed_img = sitk.GetImageFromArray(roi)
    spacing = detector.resolutions[0]
    fixed_img.SetSpacing(spacing)
    fixed_img.SetOrigin([0, 0, -roi.shape[0] // 2])
    fixed_img = sitk.RescaleIntensity(fixed_img)
    
    moving_file = os.path.join(moving_file_dir, IMG_ATLAS)
    moving_img = sitk.ReadImage(moving_file)
    
    elastix_img_filter = sitk.ElastixImageFilter()
    elastix_img_filter.SetFixedImage(fixed_img)
    elastix_img_filter.SetMovingImage(moving_img)
    param_map_vector = sitk.VectorOfParameterMap()
    param_map = sitk.GetDefaultParameterMap("translation")
    #param_map["AutomaticScalesEstimation"] = ["True"]
    #param_map["Transform"] = ["SimilarityTransform"]
    param_map["MaximumNumberOfIterations"] = ["2048"]
    param_map_vector.append(param_map)
    param_map = sitk.GetDefaultParameterMap("affine")
    #param_map["MaximumNumberOfIterations"] = ["512"]
    param_map_vector.append(param_map)
    elastix_img_filter.SetParameterMap(param_map_vector)
    elastix_img_filter.PrintParameterMap()
    transform = elastix_img_filter.Execute()
    transformed_img = elastix_img_filter.GetResultImage()
    
    fixed_dir = os.path.dirname(fixed_file)
    if write_imgs:
        out_path = _reg_out_path(fixed_dir, IMG_ATLAS)
        sitk.WriteImage(transformed_img, out_path, False)
    
    transform_param_map = elastix_img_filter.GetTransformParameterMap()
    _, translation = _handle_transform_file(fixed_file, transform_param_map)
    translation = _translation_adjust(moving_img, transformed_img, translation)
    
    transformix_img_filter = sitk.TransformixImageFilter()
    transformix_img_filter.SetTransformParameterMap(transform_param_map)
    img_files = (IMG_LABELS, )
    imgs_transformed = []
    for img_file in img_files:
        img = sitk.ReadImage(os.path.join(moving_file_dir, img_file))
        #sitk.Show(img)
        transformix_img_filter.SetMovingImage(img)
        transformix_img_filter.Execute()
        result_img = transformix_img_filter.GetResultImage()
        imgs_transformed.append(result_img)
        if write_imgs:
            out_path = _reg_out_path(fixed_dir, img_file)
            sitk.WriteImage(result_img, out_path, False)
    
    if show_imgs:
        sitk.Show(fixed_img)
        sitk.Show(moving_img)
        sitk.Show(transformed_img)
        for img in imgs_transformed:
            sitk.Show(img)
    
    # show 2D overlay for registered images
    imgs = [
        roi, 
        sitk.GetArrayFromImage(moving_img), 
        sitk.GetArrayFromImage(transformed_img), 
        sitk.GetArrayFromImage(imgs_transformed[0])]
    _show_overlays(imgs, translation[::-1], fixed_file)

def overlay_registered_imgs(fixed_file, moving_file_dir, flip_horiz=False):
    image5d = importer.read_file(fixed_file, cli.series)
    roi = image5d[0, ...] # not using time dimension
    if flip_horiz:
        roi = roi[..., ::-1]
    moving_sitk = sitk.ReadImage(os.path.join(moving_file_dir, IMG_ATLAS))
    moving_img = sitk.GetArrayFromImage(moving_sitk)
    fixed_dir = os.path.dirname(fixed_file)
    out_path = _reg_out_path(fixed_dir, IMG_ATLAS)
    transformed_sitk = sitk.ReadImage(out_path)
    transformed_img = sitk.GetArrayFromImage(transformed_sitk)
    out_path = _reg_out_path(fixed_dir, IMG_LABELS)
    labels_img = sitk.GetArrayFromImage(sitk.ReadImage(out_path))
    imgs = [roi, moving_img, transformed_img, labels_img]
    _, translation = _handle_transform_file(fixed_file)
    translation = _translation_adjust(moving_sitk, transformed_sitk, translation)
    _show_overlays(imgs, translation[::-1], fixed_file)

if __name__ == "__main__":
    print("Clrbrain image registration")
    cli.main(True)
    #register(cli.filenames[0], cli.filenames[1], flip_horiz=True, write_imgs=True)
    #register(cli.filenames[0], cli.filenames[1], flip_horiz=True, show_imgs=False)
    for plane in plot_2d.PLANE:
        plot_2d.plane = plane
        overlay_registered_imgs(cli.filenames[0], cli.filenames[1], flip_horiz=True)
