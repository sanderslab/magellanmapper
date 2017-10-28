#!/bin/bash
# Image registration
# Author: David Young, 2017
"""Register images to one another.
"""

import SimpleITK as sitk
import numpy as np

from clrbrain import cli
from clrbrain import importer

def register(fixed_file, moving_file):
    """Registers two images to one another using the SimpleElastix library.
    
    Args:
        fixed_file: The image to register
        moving_file: The reference image.
    """
    #fixed_img = sitk.ReadImage(fixed_file)
    image5d = np.load(fixed_file)
    fixed_img = sitk.GetImageFromArray(image5d[0, :, :, :])
    print(fixed_img)
    moving_img = sitk.ReadImage(moving_file)
    print(moving_img)
    
    #resultImage = sitk.Elastix(fixed_img, moving_img)
    elastix_img_filter = sitk.ElastixImageFilter()
    elastix_img_filter.SetFixedImage(fixed_img)
    elastix_img_filter.SetMovingImage(moving_img)
    param_map = sitk.GetDefaultParameterMap("translation")
    elastix_img_filter.SetParameterMap(param_map)
    elastix_img_filter.PrintParameterMap()
    elastix_img_filter.Execute()
    
    result_img = elastix_img_filter.GetResultImage()
    sitk.Show(result_img)
    #sitk.WriteImage(result_img, "output.tiff", False)
    transform_param_map = elastix_img_filter.GetTransformParameterMap()

if __name__ == "__main__":
    print("Clrbrain image registration")
    cli.main(True)
    fixed_img = (importer.filename_to_base(cli.filenames[0], cli.series) 
                 + importer.SUFFIX_IMAGE5D)
    register(fixed_img, cli.filenames[1])