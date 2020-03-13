# MagellanMapper v1.1 Release Notes

## MagellanMapper v1.1.4

### Changes

GUI
- Theming capabilities, including default and dark themes
- Workaround for error if 3D points contain invalid values
- Workaround to show labels when the first image displayed has no labels
- Detection sphere slider is larger by making the max value compact
- Fixed settings for overlaid multichannel images, allowing scalars in addition to sequences
- Larger radius to select each ROI Editor draggable circle
- Customize the number of serial 2D plot columns using the `--plot_labels layout` option
- ROIs without any blobs can be saved
- Truth blobs in verified plots are shown with their actual radius rather than as a fixed size and placed below detection blobs

CLI
- Added "".py"" extension to run script to allow cross-platofrm launch by double-click
- Fixed detecting and activating Conda and Venv environments in the run script
- Explicitly set truth database takes priority over the default path

Atlas registration
- No longer specifies left/right for each label since the laterality may change with mirroring
- Test profile without any registration iterations
- Alternate output paths can be given as directories in addition to full paths
- Fixed regression in generating a new atlas

Nuclei detection
- Simplified setup for varying values at a single index during a grid search of a hyperparameter as an array
- Total variation denoising weight is configurable

I/O
- Default zoom and magnification values are now 1 rather than -1, and total magnification is always shown as a non-negative value

Server pipelines
- Option to specify the plane orientation

Python stats and plots
- Shift ROC legends to the lower right to reduce the chance of obscuring data points

## MagellanMapper v1.1.3

### Changes

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

## MagellanMapper v1.1.2

### Changes

Installation
- No longer requires Git, C compiler, or preinstalled Java
- Startup script attempts to activate an environment if necessary, allowing the script to work from a double-click
- Moved installation docs to a separate file and added table of specialized dependency requirements
- Fixed Venv environment setup script
- Fixed SimpleElastix Windows build script to use the Conda environment Python executable if available

GUI
- Performance enhancement when viewing labeled images, especially for images with many colors
- Shows path of image loaded during startup
- Allows loading the ROI Selector without any image
- Fixed VTK error window display on Windows

CLI
- Removed unnecessary pipelines script options
- Fixed shutdown error on Windows and explicitly shut down the JVM if necessary
Atlas refinement:
- Label fill function now interpolates more smoothly and no longer adjusts the originally edited planes

Nuclei detection
- Fixed error when no blobs are found during verification against truth sets

I/O
- Applies metadata after importing image to make it immediately available for use
- Fixed regression in loading registered images

### Dependency Updates

#### Python Dependency Changes

- OpenJDK 8 is installed through the Conda pathway
- Uses prebuilt Javabridge

#### Server dependency Changes

- Git is no longer required since not accessing Javabridge from Git
- C compiler is no longer necessary as long as prebuilt Javabridge works
- Java is no longer needs to be preinstalled if following Conda pathway

## MagellanMapper v1.1.1

### Changes

GUI
- show the file path in the Atlas Editor window
- fixed loss of label visibility after setting the opacity to 0 and changing planes

Atlas refinement
- profile for Allen CCFv3
- use the resolution of the fixed image during registration if the initial registration fails to start

I/O
- use FFMpeg for export to MP4 files for smaller files that can be opened by QuickTime
- fixed TIFF import to match only files with TIFF extensions (eg `.tif` or `.tiff`)
- fixed loading images by SimpleITK to select a single channel

Python stats and plots
- option for image padding size

## MagellanMapper v1.1.0

### Changes

Installation
- Changed name from "Clrbrain" to "MagellanMapper"
- Default virtual environment name is `mag`
- Support new style Conda initialization during Conda environment setup
- Update reference Conda environment and Pip package specs

Atlas refinement
- Default to use symmetric colormaps rather than separate colors for labels on opposite hemispheres
- Export RGB values with colored cells in Excel file of region IDs for a given atlas
Atlas registration:
- Fix b-spline grid voxels setting in NCC profile

Python stats and plots
- Merge Excel files into separate sheets of single Excel file

Code base
- Reorganized all source modules into subpackages
- Moved all shell scripts in root folder into `bin` folder
