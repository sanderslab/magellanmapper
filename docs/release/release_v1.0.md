# MagellanMapper v1.0 Release Notes

## MagellanMapper v1.0.0

### Changes

Installation
- Updated requirements and environment scripts

GUI
- More zoom levels in ROI editor
- Customize position of ROI with overview plots

CLI
- More CSV manipulations by command-line: append columns to a CSV, normalize metrics
- Generate generic line plots including error bars

Atlas refinement
- Increase default smoothing for P14 atlas

Atlas registration
- Increase bspline grid size when using correlation coefficient

I/O
- Specify alternate metadata files by `--meta` command-line argument
- Load blobs directly rather than through image setup

Python stats and plots
- Nuclei clustering stats using DBSCAN and k-nearest-neighbors from scikit-learn
- further customize generic bar plots
- Option to draw a horizontal line on bar plots based on a summary function
- Option to show labels as a binary image
- Fix DSC measurements for labels

R stats and plots
- Record n for each subgroup
- Profile for basic stats

### Dependency Updates

#### Python Dependency Changes

- Scikit-learn for clustering
