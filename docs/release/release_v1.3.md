# MagellanMapper v1.3 Release Notes

## MagellanMapper v1.3.3

This is a bug fix release for Conda setup.

### Changes

Installation
- Fix Conda setup when `conda` is not found, including new installations
- Fix setting up a Conda environment when another matching environment is found from a different Conda installation

I/O
- Fix import of multi-file, single-channel RAW images

Python stats and plots
- Option to drop duplicates when joining data frames using the new `--plot_labels drop_dups=<0|1>` command-line argument

## MagellanMapper v1.3.2 (beta)

This beta release streamlines scripts to install and run MagellanMapper through a file browser, without requiring a terminal. Registered images can now be loaded and overlaid through the GUI. The image import panel auto-populates available metadata, and RAW format images can be imported. Image intensity controls better adapt to the current image.

### Changes

Installation
- Conda setup script for Bash (eg Mac/Linux)
  - Creates a `MagellanMapper` launch script that auto-detects more `python` installations to launch `run.py` (ie when only `python3` is available or if `python` is only available through Conda)
  - Better able to find existing Conda installations before attempting to install Conda
- `run_cli.py` has been re-integrated into `run.py` and removed

GUI
- Select registered images and labels in the ROI panel
  - Control the main intensity image, showing all available images registered to the current image
  - Selector for a registered labels/annotation image
  - Selector a labels reference file
- Import panel
  - Brings the user to the import panel when unable to load a file
  - Auto-populates metadata when available from the image file
  - Added output image shape and data type fields 
  - Turn on all channel after when displaying a newly imported image
  - Fixed re-displaying an image re-imported to the same output path
- Image adjusment panel
  - Adapts the intensity range to the current image to allow finer adjustments
  - Option for auto-intensities
  - Fixed settings to persist when scrolling among planes
  - Fixed slider sizes by capping the size of number labels
  - Fixed intensity slider ranges for images that were not imported into Numpy format
- Blob detector controls have been moved to a separate "Detect" panel
- Fixed crash when selecting the 3D Viewer tab or detecting blobs without a loaded image
- Fixed updating ROI size when loading an image through the GUI
- Fixed decimal point display in integere

I/O
- Support for importing RAW image format files
- Export images to RAW format (`--proc export_raw`)
- Support for both TIFF and non-TIFF format files through Bioformats
- Fixed support for incomplete metadata
- Fixed importing images while skipping channels
- Fixed import when unable to load the main image

Server pipelines
- Fixed to not attempt download from S3 if an S3 directory was not set


## MagellanMapper v1.3.1 (beta)

This beta release contains multiple new control panels to adjust profiles, brightness/contrast, and image file import. These controls allow users to control MagellanMapper more graphically and reduces the need to restart for new settings.

### Changes

Installation
- Keep Windows setup script open after installing by double-click to view instructions and any error messages
- MacOS/Linux Conda setup script renamed to allow double-click launch on MacOS and to further distinguish from the Windows setup script

GUI
- Control panels
  - Profiles panel: select, refresh, and reload profiles
  - Image adjustment panel: change brightness, contrast, and opacity
  - Image import panel: view and adjust matched files before import, add metadata, load immediately after import
- Select subset of channels instead of only all or single channels
- Zoom and pane functions in the ROI Editor overview image plots
- Adapts to high resolution (HiDPI) screens
- Consistent dark theme applied automatically when the system dark theme is used (depends on PyQt detection)
- Fixed crash when opening the Atlas Editor tab without an image loaded
- Fixed performance regression in ROI Editor

CLI
- `--channel` command-line parameter accepts multiple arguments
- Sample commands script prioritizes finding Numpy image files
- More error output from run scripts
- Fixed attempting to run MagellanMapper in the Conda base environment


## MagellanMapper v1.3.0

This release brings many changes to streamline both the graphical and command-line interfaces. Please note that several options have changed, which may require updating custom scripts. The sample commands script (`bin/sample_cmds.sh`) has been updated to illustrate this usage.

Summary of usage changes:

| Old | New | Purpose |
| --- | --- | --- |
| `python -m magmap.xx.yy` | `run_cli.py` | All command-line based entry points can be accessed through the CLI using this script |
| Use atlas profile for registration | No longer needed | Atlases should be fully imported before image registration, and the atlas's profile should no longer be typically given when registering an image |
| `--stats` | `--df` | Run data-frame (eg CSV file) tasks |
| `--roc` | `--grid_search <name1>[,name2]` | Its main task is to perform Grid Search based hyperparameter tuning; specify profile names or YAML files |
| `--rescale` | `--transform rescale=x` | Grouped with other transformation tasks |
| `--microscope <name1>[_name2]` | `--roi_profile <name1>[,name2]` | Specifies profiles to process by regions of interest; delimit by `,` to allow underscores especially in file paths |
| `--reg_profile <name1>[_name2]` | `--atlas_profile <name1>[,name2]` | Specifies profiles for atlases; delimit by `,` to allow underscores especially in file paths |
| `--saveroi` | `--save_subimg` | Consistency with "sub-images" as parts of images that can contain ROIs |
| `--chunk_size` | None | Obsolete |
| `finer` atlas profile | None | Its settings are now default |
| `--res` | `--set_meta resolutions=x,y,z` | Grouped custom metadata settings into `--set_meta` |
| `--mag` | `--set_meta magnification=x.y` | Grouped custom metadata settings into `--set_meta` |
| `--zoom` | `--set_meta zoom=x,y,z` | Grouped custom metadata settings into `--set_meta` |
| `--no_show` | `--show 0` | Show with `1` |
| `--no_scale_bar` | `--plot_labels scale_bar=1` | Grouped with other plot labels controls |
| `--padding_2d` | `--plot_labels margin=x,y,z` | Grouped with other plot labels controls, adding `margin` as space outside the ROI |
| `--border` | `--plot_labels padding=x,y,z` | Duplicated by the `padding` argument |
| `--channel c1` | `--channel c1 [c2...]` | Accepts multiple channels (v1.3.1) |

### Changes

Installation
- New Windows install script
- The install scripts are now the recommended installation pathway
- Install scripts perform silent Miniconda installs after prompting
- Fixed the URL for Miniconda download
- Fixed the run script to include command-line arguments

GUI
- All new integrated graphical interface with unified ROI Editor, Atlas Editor, and 3D viewer in separate tabs alongside the controls panel
- ROI Editor
  - Overview plot zooming scales to the size of the ROI
  - Orange border highlights the ROI z-plane corresponding to the overview plots
  - Shows similar pixel information as in the Atlas Editor
  - Layout now respects labels
  - Title now specifies axes
  - Title and empty ROI plots compatible with dark theme
  - Halve the size of truth blobs in verification plots to avoid obscuring the underlying image
  - Easier to flag detections in the ROI Editor
- GUI image loading
  - Load sub-images through the GUI
  - Fixed loading the GUI without an image
  - Fixed loading images through the GUI
  - Fixed image coordinate limits after loading an image
- Region selection
  - Option to select multiple regions, separating IDs by `,`
  - Selecting a region ID shows its basic measurements
  - Fixed potential for label boundaries to exceed image boundaries
- Isotropic rescaling is turned on by default for 3D visualizations and incorporates image resolution
- `NaN` values can be used for invisible pixels, which reduces opacification when highlighting an atlas label in the ROI Editor
- 3D surface rendering is now default
- Unified save button for all viewers
- Auto-select a channel when restoring an ROI based on saved blobs' channel
- Fixed hang when opening the Atlas Editor for large images (downsamples images if necessary)
- Fixed error when showing an image with the z-offset set to the maximum value
- Fixed clearing picked colors during atlas painting
- Fixed the aspect ratio for images rotated 90 degrees
- Fixed display of RGB values for labels
- Fixed over darking some text boxes when hovering
- Fixed blob alignment in 3D surface rendering

CLI
- The CLI now serves as a unified entry point to the Command Line Interface, incuding `register`, `plot_2d`, and other tasks,
- The CLI can be accessed through the `run_cli.py` script, which benefits from the environment setup in `run.py` without loading a window
- `--proc preprocess` option to preprocess whole images by tasks specified in a `preprocess` ROI profile setting
- Many more sample commands for common tasks
- Better support for using the sample commands script without modification
- Help information added for command-line arguments
- Command-line arguments are checked for valid options when available
- Task to export image planes to separate files (`--proc export_planes`)
- Fixed sample commands for sub-images
- Fixed sample commands path setup for older versions of Bash (< 4.3)

Atlas refinement
- Resize images using the `--size` argument as an alternative to a profile setting
- Atlas operations expecting symmetry have been generalized across any axis
- Apply adaptive histogram equalization (access as a preprocessing task)
- Records total volumes of atlas and labels during atlas import
- Fixed rotation with resizing for non-z axes
- Fixed storing image plane boundaries for label contour interpolation

Atlas registration
- The atlas is assumed to be pre-imported, which avoids redundant atlas import tasks
- Support for more pre-registration atlas pre-processing tasks, such as 3D rotation, inversion, cropping, and rescaling
- Saves a truncated labels and pre-curated images only if the corresponding options are set
- Option to rescale units (eg mm to microns)
- Settings are customizable for each registration transformation task (eg translation, affine) rather than globally
- Defaults to increased b-spline iterations
- Fixed display of images through SimpleITK after registration

Volumetric image processing
- Grid Search profiles
  - Configurable as YAML files
  - Fixed Grid Searches without sub-image parameters
- Provides basic detection accuracy stats when saving blobs
- Option for whole-image contrast limited adaptive histogram equalization (using scikit-image)
- Fixed retrieving saved ROIs from the database

I/O
- Import from a directory of images
  - Option to convert RGB images to grayscale when importing images from a directory
  - Import multi-channel images
  - Fixed importing large images
- Option to specify output paths when importing a image directory or multi-page TIFF
- Can load multi-page TIFF files without channel indicator
- Defaults to saving figures as PNG
- Library functions for listing, downloading, and uploading files in AWS S3
- Library functions for de/compressing and testing files using ZSTD
- Allows loading the main image from a registered image path
- Avoids loading a sub-image when saving it to avoid a hang
- Sample YAML profiles for blob detection, registration
- `profiles` folder is checked automatically when loading YAML profiles

Python stats and plots
- Command-line options for configuring markers (`--plot_labels marker`) and annotation columns (`--plot_labels annot_col`)

R stats and plots
- Simple R script to load and run stats

Code base and docs
- Licensed under the [BSD-3 open source license](../../LICENSE.txt)
- Moved AWS cloud management to a [separate document](../cloud_aws.md)

### Dependency Updates

#### Python Dependency Changes

- Javabridge custom binary updated for import fix on MacOS
- Removed redundant PyQt5 installation during Conda installs
- Workaround for VTK 9 incompatibility with currently Mayavi dependency
