# MagellanMapper v1.6 Release Notes

## MagellanMapper v1.6.0

### Highlights

### Changes

#### Installation

#### GUI

- "Blend" option in the image adjustment panel to visualize alignment in overlaid images
- Drag and remove loaded profiles in the Profiles tab table
- "Help" buttons added to open the online documentation (#109)
- Default confirmation labels can be set before detection (#115)
- Fixed to reset the ROI selector when redrawing (#115)
- Fixed to reorient the camera after clearing the 3D space (#121)

#### CLI

- The `--proc export_tif` task exports an NPY file to TIF format
- Fixed to only remove the final extension from image paths, and paths given by the `--prefix <path>` CLI argument do not undergo any stripping (#115)

#### Atlas refinement

#### Atlas registration

#### Cell detection

##### Detection

- Detection channels can be selected, including those not in the currently displayed image or loaded from a saved blobs file (#121)
- Auto-scrolls the detections table to the selected annotation (#109)
- `ctrl+[n]+click` to add a channel now sets the channel directly to `n` rather than to the `n`th seleted channel (#109)
- Added a slider to choose the fraction of 3D blobs to display (#121)
- Improved blob size slider range and readability (#121)
- Fixed to scale blobs' radii when viewing blobs detections on a downsampled image (#121)
- Fixed getting atlas colors for blobs for ROIs inside the main image (#121)
- Fixed blob segmentation for newer versions of Scikit-image (#91)
- Fixed verifying and resaving blobs

##### Colocalization

- Match-based colocalization can run without the main image, using just its metadata instead (#117)
- These colocalizations are now displayed in the 3D viewer (#121)
- Fixed match-based colocalizations when no matches are found (#117, #120)
- Fixed slow loading of match-based colocalizations (#119)

#### Volumetric image processing

- Match-based colocalizations use larger processing blocks to avoid gaps (#120)
- Fixed 3D surface area measurement with Scikit-image >= v0.19

#### I/O

- Some TIF files can be loaded directly, without importing the file first (#90)
- The `--proc export_planes` task can export a subset of image planes specified by `--slice`, or an ROI specified by `--offset` and `--size`
- Image metadata is stored in the `Image5d` image object (#115)
- Fixed re-importing an image after loading it (#117)

#### Server pipelines

#### Python stats and plots

- Fixed alignment of headers and columns in data frames printed to console (#109)

#### R stats and plots

#### Code base and docs

### Dependency Updates

#### Python Dependency Changes

- Python 3.8 is the default version now that Python 3.6 has reached End-of-Life
- `Tifffile` is now a direct dependency, previously already installed as a sub-dependency of other required packages
- Updated to use the `axis_channel` parameter in Scikit-image's `transform.rescale` function (#115)

#### R Dependency Changes

#### Server dependency Changes
