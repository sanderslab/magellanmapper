# MagellanMapper v1.6 Release Notes

## MagellanMapper v1.6.0

### Highlights

### Changes

#### Installation

#### GUI

- Drag and remove loaded profiles in the Profiles tab table

#### CLI

- The `--proc export_tif` task exports an NPY file to TIF format

#### Atlas refinement

#### Atlas registration

#### Cell detection

#### Volumetric image processing

#### I/O

- Some TIF files can be loaded directly, without importing the file first (#90)
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
