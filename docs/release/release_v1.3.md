# MagellanMapper v1.3 Release Notes

## MagellanMapper v1.3.0

This release brings many changes to streamline the command-line interface (CLI). Please note that several options have changed, which may require updating custom scripts. The sample commands script (`bin/sample_cmds.sh`) has been updated to illustrate this usage.

Summary of usage changes:

| Old | New | Purpose |
| --- | --- | --- |
| `python -m magmap.xx.yy` | `run_cli.py` | All command-line based entry points can be accessed through the CLI using this script |
| Use atlas profile for registration | No longer needed | Atlases should be fully imported before image registration, and the atlas's profile should no longer be typically given when registering an image |
| `--stats` | `--df` | Run data-frame (eg CSV file) tasks |
| `--roc` | `--grid_search` | Its main task is to perform Grid Search based hyperparameter tuning |
| `--rescale` | `--transform rescale=x` | Grouped with other transformation tasks |
| `--microscope` | `--roi_profile` | Specifies profiles to process by regions of interest |
| `--reg_profile` | `--atlas_profile` | Specifies profiles for atlases |

### Changes

Installation
- Fixed the run script to include command-line arguments

GUI
- Isotropic rescaling is turned on by default for 3D visualizations and incorporates image resolution
- Easier to flag detections in the ROI Editor
- Fixed hang when opening the Atlas Editor for large images (downsamples images if necessary)
- Fixed error when showing an image with the z-offset set to the maximum value
- Fixed the aspect ratio for images rotated 90 degrees

CLI
- The CLI now serves as a unified entry point to the Command Line Interface, incuding `register`, `plot_2d`, and other tasks,
- The CLI can be accessed through the `run_cli.py` script, which benefits from the environment setup in `run.py` without loading a window
- `--proc preprocess` option to preprocess whole images by tasks specified in a `preprocess` ROI profile setting
- Many more sample commands for common tasks
- Better support for using the sample commands script without modification
- Help information added for command-line arguments
- Command-line arguments are checked for valid options when available
- Fixed sample commands for sub-images
- Fixed sample commands path setup for older versions of Bash (< 4.3)

Atlas refinement
- Resize images using the `--size` argument as an alternative to a profile setting
- Atlas operations expecting symmetry have been generalized across any axis
- Apply adaptive histogram equalization (access as a preprocessing task)
- Fixed rotation with resizing for non-z axes

Atlas registration
- The atlas is assumed to be pre-imported, which avoids redundant atlas import tasks
- Support for more pre-registration atlas pre-processing tasks, such as 3D rotation, inversion, cropping, and rescaling
- Saves a truncated labels and pre-curated images only if the corresponding options are set
- Option to rescale units (eg mm to microns)
- Settings are customizable for each registration transformation task (eg translation, affine) rather than globally

Volumetric image processing
- Option for whole-image contrast limited adaptive histogram equalization (using scikit-image)
- Fixed retrieving saved ROIs from the database

I/O
- Library functions for listing, downloading, and uploading files in AWS S3
- Library functions for de/compressing and testing files using ZSTD

R stats and plots
- Simple R script to load and run stats

### Dependency Updates

#### Python Dependency Changes

#### R Dependency Changes

#### Server dependency Changes
