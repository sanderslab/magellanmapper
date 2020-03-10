# MagellanMapper v1.1.3 Release Notes

## Changes

GUI
- Open multiple Atlas Editors at the same time, including synchronized annotation updates
- Fixed inability to open an ROI Editor when a non-editable window is open
- Fixed error when attempting to show 3D blob locations without any blobs
- Fixed error when showing some ROI Editor overview images

Nuclei detection
- Grid search hyperparameters groups have been reorganized into selectable profiles
- Fixed running grid searches/ROC curves with Pandas 1.0
- Unsharp filtering and erosion can be turned off during image preprocessing
- The `vmin` saturation settings is now configurable, similar to the `vmax` setting
- The lower threshold factor for max scaling is now configurable to reduce false detections in low signal areas
- Microscope profiles for minimal preprocessing and low resolution images
