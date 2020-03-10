# MagellanMapper v1.1 Release Notes

## MagellanMapper v1.1.2 Release Notes

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

## MagellanMapper v1.1.2 Release Notes

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
