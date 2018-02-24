#!/bin/bash
# File exporter for Clrbrain
# Author: David Young, 2017, 2018
"""Image exporter for Clrbrain.

Convert images and corresponding database entries into formats for 
machine learning algorithms or other applications.
"""

import os
import glob

import numpy as np
from matplotlib import pyplot as plt
try:
    import SimpleITK as sitk
except ImportError as e:
    print(e)
    print("WARNING: SimpleElastix could not be found, so there will be error "
          "when attempting to export images to non-Numpy formats")

from clrbrain import config
from clrbrain import chunking
from clrbrain import lib_clrbrain
from clrbrain import sqlite
from clrbrain import plot_2d
from clrbrain import plot_3d

def make_roi_paths(path, roi_id, channel):
    path_base = "{}_roi{}".format(path, str(roi_id).zfill(5))
    path_img = "{}_ch{}.npy".format(path_base, channel)
    path_img_nifti = "{}_ch{}.nii.gz".format(path_base, channel)
    path_blobs = "{}_blobs.npy".format(path_base)
    path_img_annot = "{}_ch{}_annot.npy".format(path_base, channel)
    path_img_annot_nifti = "{}_ch{}_annot.nii.gz".format(path_base, channel)
    return path_base, path_img, path_img_nifti, path_blobs, path_img_annot, \
        path_img_annot_nifti

def export_rois(db, image5d, channel, path, border):
    """Export all ROIs from database.
    
    Args:
        db: Database from which to export.
        image5d: The image with the ROIs.
        channel: Channel to export.
        path: Path with filename base from which to save the exported files.
        border: Border dimensions in (x,y,z) order to not include in the ROI; 
            can be None.
    """
    if border is not None:
        border = np.array(border)
    exps = sqlite.select_experiment(db.cur, None)
    for exp in exps:
        rois = sqlite.select_rois(db.cur, exp["id"])
        for roi in rois:
            # get ROI as a small image
            size = sqlite.get_roi_size(roi)
            offset = sqlite.get_roi_offset(roi)
            img3d = plot_3d.prepare_roi(image5d, channel, size, offset)
            
            # get blobs, keep only confirmed ones, and change confirmation 
            # flag to avoid confirmation color in 2D plots
            roi_id = roi["id"]
            blobs = sqlite.select_blobs(db.cur, roi_id)
            blobs = blobs[blobs[:, 4] == 1]
            blobs[:, 4] = -1
            
            # adjust ROI size and offset if border set
            if border is not None:
                size = np.subtract(img3d.shape[::-1], 2 * border)
                img3d = plot_3d.prepare_roi(
                    img3d, channel, size, border)
                blobs[:, 0:3] = np.subtract(
                    blobs[:, 0:3], np.add(offset, border)[::-1])
            print("exporting ROI of shape {}".format(img3d.shape))
            
            # export ROI and 2D plots
            path_base, path_img, path_img_nifti, path_blobs, path_img_annot, \
                path_img_annot_nifti = make_roi_paths(path, roi_id, channel)
            np.save(path_img, img3d)
            # WORKAROUND: for some reason SimpleITK gives a conversion error 
            # when converting from uint16 (>u2) Numpy array
            img3d = img3d.astype(np.float64)
            img3d_sitk = sitk.GetImageFromArray(img3d)
            '''
            print(img3d_sitk)
            print("orig img:\n{}".format(img3d[0]))
            img3d_back = sitk.GetArrayFromImage(img3d_sitk)
            print(img3d.shape, img3d.dtype, img3d_back.shape, img3d_back.dtype)
            print("sitk img:\n{}".format(img3d_back[0]))
            '''
            sitk.WriteImage(img3d_sitk, path_img_nifti, False)
            plot_2d.plot_roi(img3d, blobs, channel, show=False, title=path_base)
            lib_clrbrain.show_full_arrays()
            
            # export image and blobs, stripping blob flags and adjusting 
            # user-added segments' radii
            blobs = blobs[:, 0:4]
            # prior to v.0.5.0, user-added segments had a radius of 0.0
            blobs[np.isclose(blobs[:, 3], 0), 3] = 5.0
            # as of v.0.5.0, user-added segments have neg radii whose abs
            # value corresponds to the displayed radius
            blobs[:, 3] = np.abs(blobs[:, 3])
            # make more rounded since near-integer values appear to give 
            # edges of 5 straight pixels
            # https://github.com/scikit-image/scikit-image/issues/2112
            #blobs[:, 3] += 1E-1
            blobs[:, 3] -= 0.5
            lib_clrbrain.printv("blobs:\n{}".format(blobs))
            np.save(path_blobs, blobs)
            
            # convert blobs to ground truth
            img3d_truth = plot_3d.build_ground_truth(size, blobs)
            print("exporting truth ROI of shape {}".format(img3d_truth.shape))
            np.save(path_img_annot, img3d_truth)
            #print(img3d_truth)
            sitk.WriteImage(
                sitk.GetImageFromArray(img3d_truth), path_img_annot_nifti, 
                False)
            # avoid smoothing interpolation, using "nearest" instead
            with plt.style.context(config.rc_params_mpl2_img_interp):
                plot_2d.plot_roi(
                    img3d_truth, None, channel, show=False, 
                    title=os.path.splitext(path_img_annot)[0])
            
            print("exported {}".format(path_base))
    '''
    _test_loading_rois(db, channel, path)
    '''

def load_roi_files(db, path):
    path_base, path_img, path_blobs = make_roi_paths(path, "*")
    img_paths = sorted(glob.glob(path_img))
    img_blobs_paths = sorted(glob.glob(path_blobs))
    imgs = []
    img_blobs = []
    for img, blobs in zip(img_paths, img_blobs_paths):
        img3d = np.load(img)
        imgs.append(img3d)
        print(img3d.shape)
        blobs = np.load(blobs)
        blobs = np.insert(blobs, blobs.shape[1], -1, axis=1)
        #print("blobs:\n{}".format(blobs))
        img_blobs.append(blobs)
    return path_base, imgs, img_blobs

def _test_loading_rois(db, channel, path):
    path_base, imgs, img_blobs = load_roi_files(db, path)
    for img, blobs in zip(imgs, img_blobs):
        plot_2d.savefig = None
        plot_2d.plot_roi(img, blobs, channel, show=True, title=path_base)

if __name__ == "__main__":
    print("Clrbrain exporter")
