# MagellanMapper v1.4 Release Notes

## MagellanMapper v1.4.0

### Changes

Installation

GUI
- Label selection options to include both sides and child labels
- Fixed error when looking up atlas label without a loaded reference file

CLI

Atlas refinement
- Set an alternative intensity image for edge detection using the registration suffixes atlas flag (`--reg_suffixes [atlas]`)
- Added a `watershed_mask_filter` setting in the `edge_aware_reannotation` atlas profile group to set the filter type and size for the watershed mask
- Added a `crop_to_first_image` option to compare matching volumes between two images by cropping the second image to the size of the first image
- Fixed to exclude labels that were not eroded from undergoing watershed-based reannotation

Atlas registration

Volumetric image processing
- Volume comparisons: include raw pixel and volume counts
- Option to compare volumes of only ROIs within a whole image

I/O

Server pipelines

Python stats and plots
- Perform arithmetic operations on data frame columns using `--df sum_cols`, `subtract_cols`, `multiply_cols`, `divide_cols`
- Data frame task to replace values (`--df replace_vals`)
- Added `--plot_labels x_scale` and `y_scale` parameters to set axis scaling, such as `log` for log-scaling
- Fixed matching label rows when weighting metrics

R stats and plots
- Update usage of `addTextLabels` to its successor package, `basicPlotterR`
- Provide feedback when plots fail to display

Code base and docs

### Dependency Updates

#### Python Dependency Changes

#### R Dependency Changes

#### Server dependency Changes
