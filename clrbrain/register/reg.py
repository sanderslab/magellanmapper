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

IMG_ATLAS = "atlasVolume.mhd"
IMG_LABELS = "annotation.mhd"

def register(fixed_file, moving_file_dir, flip_horiz=False):
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
        roi = image5d[..., ::-1]
    fixed_img = sitk.GetImageFromArray(roi)
    fixed_img.SetSpacing(detector.resolutions[0])
    #print("roi.shape: {}".format(roi.shape))
    fixed_img.SetOrigin([0, 0, -roi.shape[0] // 2])
    fixed_img = sitk.RescaleIntensity(fixed_img)
    #print("spacing: {}".format(fixed_img.GetSpacing()))
    
    sitk.Show(fixed_img)
    moving_file = os.path.join(moving_file_dir, IMG_ATLAS)
    moving_img = sitk.ReadImage(moving_file)
    sitk.Show(moving_img)
    
    print(fixed_img)
    print(moving_img)
    
    elastix_img_filter = sitk.ElastixImageFilter()
    elastix_img_filter.SetFixedImage(fixed_img)
    elastix_img_filter.SetMovingImage(moving_img)
    param_map_vector = sitk.VectorOfParameterMap()
    param_map = sitk.GetDefaultParameterMap("translation")
    #param_map["AutomaticScalesEstimation"] = ["True"]
    #param_map["Transform"] = ["SimilarityTransform"]
    param_map_vector.append(param_map)
    param_map_vector.append(sitk.GetDefaultParameterMap("rigid"))
    elastix_img_filter.SetParameterMap(param_map_vector)
    elastix_img_filter.PrintParameterMap()
    elastix_img_filter.Execute()
    
    result_img = elastix_img_filter.GetResultImage()
    sitk.Show(result_img)
    #sitk.WriteImage(result_img, "output.tiff", False)
    
    transform_param_map = elastix_img_filter.GetTransformParameterMap()
    transformix_img_filter = sitk.TransformixImageFilter()
    transformix_img_filter.SetTransformParameterMap(transform_param_map)
    img_files = (os.path.join(moving_file_dir, IMG_LABELS), )
    for img_file in img_files:
        img = sitk.ReadImage(img_file)
        transformix_img_filter.SetMovingImage(img)
        transformix_img_filter.Execute()
        sitk.Show(transformix_img_filter.GetResultImage())
    
if __name__ == "__main__":
    print("Clrbrain image registration")
    cli.main(True)
    register(cli.filenames[0], cli.filenames[1], flip_horiz=False)
