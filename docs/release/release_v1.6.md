# MagellanMapper v1.6 Release Notes

## MagellanMapper v1.6.0

### Highlights

### Changes

#### Installation

#### GUI

- "Blend" option in the image adjustment panel to visualize alignment in overlaid images
- Drag and remove loaded profiles in the Profiles tab table

#### CLI

- The `--proc export_tif` task exports an NPY file to TIF format

#### Atlas refinement

#### Atlas registration

#### Cell detection

- Fixed blob segmentation for newer versions of Scikit-image (#91)

#### Volumetric image processing

#### I/O

- The `--proc export_planes` task can export a subset of image planes specified by `--slice`, or an ROI specified by `--offset` and `--size`

#### Server pipelines

#### Python stats and plots

#### R stats and plots

#### Code base and docs

### Dependency Updates

#### Python Dependency Changes

- `Tifffile` is now a direct dependency, previously already installed as a sub-dependency of other required packages

#### R Dependency Changes

#### Server dependency Changes
