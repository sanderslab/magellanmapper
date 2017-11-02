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

def _show_overlays(imgs, z, fixed_file):
    cmaps = ["Blues", "Oranges", "prism"]
    aspect = detector.resolutions[0, 1] / detector.resolutions[0, 0]
    print("aspect: {}".format(aspect))
    #plot_2d.plot_overlays(imgs, z, cmaps, os.path.basename(fixed_file), aspect)
    plot_2d.plot_overlays_reg(*imgs, z, *cmaps, os.path.basename(fixed_file), aspect)

def register(fixed_file, moving_file_dir, flip_horiz=False, write_imgs=False):
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
    fixed_img.SetSpacing(detector.resolutions[0])
    #print("roi.shape: {}".format(roi.shape))
    fixed_img.SetOrigin([0, 0, -roi.shape[0] // 2])
    fixed_img = sitk.RescaleIntensity(fixed_img)
    #print("spacing: {}".format(fixed_img.GetSpacing()))
    
    moving_file = os.path.join(moving_file_dir, IMG_ATLAS)
    moving_img = sitk.ReadImage(moving_file)
    
    '''
    print(fixed_img)
    print(moving_img)
    '''
    sitk.Show(fixed_img)
    sitk.Show(moving_img)
    
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
    elastix_img_filter.Execute()
    
    transformed_img = elastix_img_filter.GetResultImage()
    sitk.Show(transformed_img)
    
    fixed_dir = os.path.dirname(fixed_file)
    if write_imgs:
        out_path = _reg_out_path(fixed_dir, IMG_ATLAS)
        sitk.WriteImage(transformed_img, out_path, False)
    
    transform_param_map = elastix_img_filter.GetTransformParameterMap()
    transformix_img_filter = sitk.TransformixImageFilter()
    transformix_img_filter.SetTransformParameterMap(transform_param_map)
    img_files = (IMG_LABELS, )
    imgs_transformed = []
    result_img = None
    for img_file in img_files:
        img = sitk.ReadImage(os.path.join(moving_file_dir, img_file))
        #sitk.Show(img)
        #print(min(img), max(img))
        transformix_img_filter.SetMovingImage(img)
        transformix_img_filter.Execute()
        result_img = transformix_img_filter.GetResultImage()
        #result_img = sitk.RescaleIntensity(result_img, 0, 1.769e+04)
        imgs_transformed.append(result_img)
        sitk.Show(result_img)
        if write_imgs:
            out_path = _reg_out_path(fixed_dir, img_file)
            sitk.WriteImage(result_img, out_path, False)
        '''
        result_img = sitk.RescaleIntensity(result_img)
        result_img = sitk.Cast(result_img, sitk.sitkUInt32)
        #sitk.Show(sitk.LabelOverlay(transformed_img, result_img))
        sitk.Show(sitk.LabelToRGB(result_img))
        '''
    
    imgs = [
        roi, 
        sitk.GetArrayFromImage(moving_img), 
        sitk.GetArrayFromImage(transformed_img), 
        sitk.GetArrayFromImage(imgs_transformed[0])]
    _show_overlays(imgs, roi.shape[0] // 3, fixed_file)

def overlay_registered_imgs(fixed_file, moving_file_dir, flip_horiz=False):
    image5d = importer.read_file(fixed_file, cli.series)
    roi = image5d[0, ...] # not using time dimension
    if flip_horiz:
        roi = roi[..., ::-1]
    moving_img_orig = sitk.GetArrayFromImage(
        sitk.ReadImage(os.path.join(moving_file_dir, IMG_ATLAS)))
    fixed_dir = os.path.dirname(fixed_file)
    out_path = _reg_out_path(fixed_dir, IMG_ATLAS)
    moving_img = sitk.GetArrayFromImage(sitk.ReadImage(out_path))
    out_path = _reg_out_path(fixed_dir, IMG_LABELS)
    labels_img = sitk.GetArrayFromImage(sitk.ReadImage(out_path))
    imgs = [roi, moving_img_orig, moving_img, labels_img]
    _show_overlays(imgs, roi.shape[0] // 3, fixed_file)

if __name__ == "__main__":
    print("Clrbrain image registration")
    cli.main(True)
    #register(cli.filenames[0], cli.filenames[1], flip_horiz=True, write_imgs=True)
    overlay_registered_imgs(cli.filenames[0], cli.filenames[1], flip_horiz=True)
