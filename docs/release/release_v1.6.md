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

- Auto-scrolls to the selected annotation
- `ctrl+[n]+click` to add a channel now sets the channel directly to `n` rather than to the `n`th seleted channel
- Fixed blob segmentation for newer versions of Scikit-image (#91)

#### Volumetric image processing

- Fixed 3D surface area measurement with Scikit-image >= v0.19

#### I/O

- The `--proc export_planes` task can export a subset of image planes specified by `--slice`, or an ROI specified by `--offset` and `--size`

#### Server pipelines

#### Python stats and plots

#### R stats and plots

#### Code base and docs

### Dependency Updates

#### Python Dependency Changes

- Python 3.8 is the default version now that Python 3.6 has reached End-of-Life
- `Tifffile` is now a direct dependency, previously already installed as a sub-dependency of other required packages

#### R Dependency Changes

#### Server dependency Changes
