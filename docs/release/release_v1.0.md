# MagellanMapper v1.0 Release Notes

## MagellanMapper v1.0.0

### Changes

Installation
- updated requirements and environment scripts

GUI
- more zoom levels in ROI editor
- customize position of ROI with overview plots

CLI
- more CSV manipulations by command-line: append columns to a CSV, normalize metrics
- generate generic line plots including error bars

Atlas refinement
- increase default smoothing for P14 atlas

Atlas registration
- increase bspline grid size when using correlation coefficient

I/O
- specify alternate metadata files by ""--meta"" command-line argument
- load blobs directly rather than through image setup

Python stats and plots
- nuclei clustering stats using DBSCAN and k-nearest-neighbors from scikit-learn
- further customize generic bar plots
- option to draw a horizontal line on bar plots based on a summary function
- option to show labels as a binary image
- fix DSC measurements for labels

R stats and plots
- record n for each subgroup
- profile for basic stats

### Dependency Updates

#### Python Dependency Changes

- Scikit-learn for clustering
