#!/bin/bash
# ROI exporter for Clrbrain
# Author: David Young, 2017, 2019
"""ROI exporter for Clrbrain.

Convert images and corresponding database entries into formats for 
machine learning algorithms or other applications.
"""

import os
import glob

import numpy as np
from matplotlib import pyplot as plt
import SimpleITK as sitk

from clrbrain import config
from clrbrain import detector
from clrbrain import lib_clrbrain
from clrbrain import sqlite
from clrbrain import plot_3d
from clrbrain import roi_editor
 
def make_roi_paths(path, roi_id, channel, make_dirs=False):
    path_base = "{}_roi{}".format(path, str(roi_id).zfill(5))
    path_dir_nifti = "{}_nifti".format(path_base)
    name_base = os.path.basename(path_base)
    path_img = os.path.join(path_base, "{}_ch{}.npy".format(name_base, channel))
    path_img_nifti = os.path.join(
        path_dir_nifti, "{}_ch{}.nii.gz".format(name_base, channel))
    path_blobs = os.path.join(path_base, "{}_blobs.npy".format(name_base))
    path_img_annot = os.path.join(
        path_base, "{}_ch{}_annot.npy".format(name_base, channel))
    path_img_annot_nifti = os.path.join(
        path_dir_nifti, "{}_ch{}_annot.nii.gz".format(name_base, channel))
    if make_dirs:
        if not os.path.exists(path_base):
            os.makedirs(path_base)
        if not os.path.exists(path_dir_nifti):
            os.makedirs(path_dir_nifti)
    return path_base, path_dir_nifti, path_img, path_img_nifti, path_blobs, \
        path_img_annot, path_img_annot_nifti


def export_rois(db, image5d, channel, path, border):
    """Export all ROIs from database.
    
    If the current processing profile includes isotropic interpolation, the 
    ROIs will be resized to make isotropic according to this factor.
    
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
            img3d = plot_3d.prepare_roi(image5d, size, offset)
            
            # get blobs, keep only confirmed ones, and change confirmation 
            # flag to avoid confirmation color in 2D plots
            roi_id = roi["id"]
            blobs = sqlite.select_blobs(db.cur, roi_id)
            blobs = blobs[detector.get_blob_confirmed(blobs) == 1]
            blobs[:, 4] = -1
            
            # adjust ROI size and offset if border set
            if border is not None:
                size = np.subtract(img3d.shape[::-1], 2 * border)
                img3d = plot_3d.prepare_roi(img3d, size, border)
                blobs[:, 0:3] = np.subtract(
                    blobs[:, 0:3], np.add(offset, border)[::-1])
            print("exporting ROI of shape {}".format(img3d.shape))
            
            isotropic = config.process_settings["isotropic"]
            blobs_orig = blobs
            if isotropic is not None:
                # interpolation for isotropy if set in first processing profile
                img3d = plot_3d.make_isotropic(img3d, isotropic)
                isotropic_factor = plot_3d.calc_isotropic_factor(isotropic)
                blobs_orig = np.copy(blobs)
                blobs = detector.multiply_blob_rel_coords(
                    blobs, isotropic_factor)
            
            # export ROI and 2D plots
            path_base, path_dir_nifti, path_img, path_img_nifti, path_blobs, \
                path_img_annot, path_img_annot_nifti = make_roi_paths(
                    path, roi_id, channel, make_dirs=True)
            np.save(path_img, img3d)
            print("saved 3D image to {}".format(path_img))
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
            roi_ed = roi_editor.ROIEditor()
            roi_ed.plot_roi(
                img3d, blobs, channel, show=False, 
                title=os.path.splitext(path_img)[0])
            lib_clrbrain.show_full_arrays()
            
            # export image and blobs, stripping blob flags and adjusting 
            # user-added segments' radii; use original rather than blobs with 
            # any interpolation since the ground truth will itself be 
            # interpolated
            blobs = blobs_orig
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
            img3d_truth = plot_3d.build_ground_truth(
                np.zeros(size[::-1], dtype=np.uint8), blobs)
            if isotropic is not None:
                img3d_truth = plot_3d.make_isotropic(img3d_truth, isotropic)
                # remove fancy blending since truth set must be binary
                img3d_truth[img3d_truth >= 0.5] = 1
                img3d_truth[img3d_truth < 0.5] = 0
            print("exporting truth ROI of shape {}".format(img3d_truth.shape))
            np.save(path_img_annot, img3d_truth)
            #print(img3d_truth)
            sitk.WriteImage(
                sitk.GetImageFromArray(img3d_truth), path_img_annot_nifti, 
                False)
            # avoid smoothing interpolation, using "nearest" instead
            with plt.style.context(config.rc_params_mpl2_img_interp):
                roi_ed.plot_roi(
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
    roi_ed = roi_editor.ROIEditor()
    for img, blobs in zip(imgs, img_blobs):
        config.savefig = None
        roi_ed.plot_roi(img, blobs, channel, show=True, title=path_base)


def blobs_to_csv(blobs, path):
    """Exports blob coordinates and radius to CSV file, compressed with GZIP.
    
    Args:
        blobs: Blobs array, assumed to be in [[z, y, x, radius, ...], ...] 
            format.
        path: Path to blobs file. The CSV file will be the same as this path 
            except replacing the extension with ``.csv.gz``.
    """
    path_out = "{}.csv.gz".format(os.path.splitext(path)[0])
    header = "z,y,x,r"
    np.savetxt(path_out, blobs[:, :4], delimiter=",", header=header)


if __name__ == "__main__":
    print("Clrbrain exporter")
