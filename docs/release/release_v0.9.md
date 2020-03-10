# MagellanMapper v0.9 Release Notes

## MagellanMapper v0.9.9

### Changes

GUI
- Atlas Editor label names readability: word-wrapped, transluscent background for better color contrast
- Keeps ROI within the zoomed overview plot in the ROI Editor when possible
- Stretch the color value range in discete colormaps while avoiding fully dark RGB colors
- Experimental discrete colormap based on evenly spaced RGB values

CLI
- Translates "none" into a `None` object for default settings
- Profile modifier dictionary settings update rather than overwrite prior dictionaries

Atlas refinement:
- Incorporated skeletonization along with erosion and watershed as a form of adaptive morphology to retain thin structures
- Lateral extension models the underlying histology using a serial 2D version of the edge-aware watershed and tapering off labels, preferentially central labels
- Smooths labels during lateral extension to remove artifacts
- In-paints rather than uses closing filter to close ventricles for more gradual, smoother filling
- Option to export intermediate steps during lateral edge extension
- Fixed reported erosion to marker filter and size values

I/O
- Simple export of overlaid registered files using a list of image paths
- Fixed making all values below vmin transparent
- Fixed exporting a stack when the first image is turne off
Server pipelines:
- Fixed reloading swap files from NVMe devices

R stats and plots
- Fixed p-value adjustment/correction to only apply to stats with 2 or more values

## MagellanMapper v0.9.8

### Changes

GUI
- Fix edge interpolation between planes after multiple edits in the same plane
- Allow overlaying downsized labels on the full resolution image
- Broader support for image flipping (eg ROI Editor flips ROI box in addition to images)

Atlas refinement
- Measure label-by-label improvement as percent of labels improved
- Measure DSCs by label

Atlas registration
- Profile setting for a fallback similarity metric if DSC is below a threshold

I/O
- No longer require the full image for stack detection or ROI export, falling back to loading an ROI image
- Fix loading ROI image metadata in Numpy >= 1.17

Python stats and plots
- Scatter plots: customize point transparency, use different markers for each group
- Join CSV files by column to allow storing experiment metadata in CSV

R stats and plots
- Jitter plots: option to annotate with sample names
- Configure p-value correction/alternative methods
- Customize the column used to identify the main groups for comparison
- Support separate comparison by sex and laterality

## MagellanMapper v0.9.7

### Changes

Atlas refinement
- Replaced simple boundary count for surface area metric with 3D marching cubes

Atlas registration
- Save pre-curated registered atlas image for inspection
- Overlay multiple sample images on top of one another to assess registration accuracy

Nuclei detection
- Export ROI stats such as intensity and nuclei counts along

Python stats and plots
- Weighted arithmetic mean library function
- Output a collage of multiple sample images
- Library function to merge data frames by columns

R stats and plots
- Fixed regression in regression models for multiple genotypes
- Option to filter regions by condition
- Added median and standard deviation output by group
- Fixed total number of groups in jitter plots

Code base
- Continued refactoring tuple constants to enums
- Bash scripts clean-up
- Restart Sphinx API docs generation

## MagellanMapper v0.9.6

### Changes

Installation
- Provided Conda environment full pinned specs for Win/Mac/Lin
- Option to specify a Conda environment spec during setup
- `requirements.txt` is now based on a full pip install and directly installable
- Assume that SimpleElastix is installed now that binaries are provided
- Fixed finding SimpleElastix binaries during Venv setup

Atlas refinement
- Option to specify right hemisphere label inversion to fix inadvertent inversion for rotated images

Atlas registration
- Customize the similarity metric used for registration

I/O
- Fixed image stack export from a directory of images

Server pipelines
- File size output and space between number and time unit in Slack notification

Python stats and plots
- Fixed regression in normalized developmental plots
- Fixed showing empty unit parentheses in axis labels
- Customize bar plot x-tick labels and rotation
- Option to weight effect sizes in labels difference images
- Option to customize vertical span labels

Code base
- Fefactor `register` main tasks
- Refactor non-registration tasks from `register` into separate modules
- Bash scripts cleanup

### Dependency Updates

#### Python Dependency Changes

- `setup.py` specifies Python >= 3.6 since this version is the lowest tested
- Custom PyPi is an extra rather than primary URL
- Official rather than forked SimpleElastix repository by default but also option to specify an alternative local repo directory

## MagellanMapper v0.9.5

### Changes

Installation
- Installs SimpleElastix binaries from a custom respository instead of SimpleITK
- Provides option to install optional dependency groups during in setup.py
- Venv setup script defaults to installing all optional dependencies
- Conda setup script uses `clr` as the default name again since the name can be customized
- Matplotlib ScaleBar is no longer required

Atlas refinement
- Updated profile modifiers for new pipeline nomenclature
- Adds metric for DSC of the labeled hemisphere
- Saves volume level stats to the imported atlas directory for easier access
- Stats output removes NaN columns for smaller files
- Labels interior-border images are now optional

Atlas registration
- Option to specify similarity metric

I/O
- Fixed missing interval when exporting a stack of separate image files

Server pipelines
- Finds NVMe devices assigned to a given device name

Python stats and plots
- Fixed colorbar labeling from command-line
- Fixed line plots with more line groups than styles and prioritize a solid line for the final group

### Dependency Updates

#### Python Dependency Changes

- SimpleElastix installed from custom PyPi repository on GemFury instead of SimpleITK from official PyPi
- awscli and boto3 installed by default in Conda environment spec
- Default Conda environment name is changed back to `clr` when using the setup script
- Added PyQt5 and SimpleITK as required dependencies in setup.py, with SimpleITK version specifieid to get custom SimpleElastix in place of the standard SimpleITK, assuming extra-url flag is added to pip install to find respository
- Optional dependencies in setup.py for AWS, import, and miscellaneous graphics
- Matplotlib ScaleBar now optional
- Venv setup scripti installs dependencies based on setup.py rather than requirements.txt
- Updated `requirements.txt` to current snapshot and added an equivalent `environment_full.yml` Conda spec

#### Server dependency Changes

- nvme-cli required to map EBS volumes in NVMe-based instances to the correct drive path 

## MagellanMapper v0.9.4

### Changes

Installation
- Ease Windows compatibility: SimpleElastix can be built with a batch script, docs on setup requirements, Bash setup scripts optional
- Allow a complete Conda environment installation from a .yml file
- Script for Venv setup with dependencies installation by requirements.txt file
- Specified required dependencies in setup.py
- Python-Bioformats/Javabridge and thus Java are now optional
- Shift most dependencies to Conda to make compiler optional

GUI
- Option to keep colormaps consistent with the original labels image
- Fixed intensity normalization for newer versions of Mayavi (>= 4.7)
Atlas refinement:
- Fixed lateral edge extension to extend only from labeled areas to avoid extending artifacts
- Decreased ADMBA E13.5 and E15.5 atlas thresholds now that artifacts are not being followed
- Minimize closing filter used to close ventricles in several ADMBA atlases

I/O
- Fixed error when displaying metadata after image import to Numpy format with newer Numpy versions (eg >=1.16)

Pipelines
- Option to avoid showing log output in nohup wrapper script
- Fixed potential to initiate a server command for the next rather than the current run script

Nuclei detection
- Removed deprecated anisotropic kernel settings

Python stats
- Make scatter plots cleaner by not splitting on different labels and giving an option to show only a single value from arrays in scatter plots to simply ROC curve labels
- Allow additional region metric groups, including those performed across whole super-structures rather than through weighted averaging
- Plot regional volumes and compactness across development
- Fixed line plots with more than 10 lines
- Extended scientific notation to line plots, format units with math text, use y-axis command-line-specified labels in bar plots, and allow turning off command-line labels for specific axes
- Quantify unlabeled hemispheres by volume and fraction of sagittal planes"

### Dependency Updates

#### Python Dependency Changes

- Conda installs from Conda-Forge over defaults, with strict channel preference if using the setup script
- `setup.py` specifies required dependencies
- `setuptools` essentially required for `setup.py` (attempts to fallback to disutils but will fail when trying to find packages)

## MagellanMapper v0.9.3

### Changes

I/O
- Support `--prefix` when loading registered image files

Nuclei detection
- Replace verification method with Hungarian algorithm, which gives: 1) better optimized pairing of detection and ground truth, 2) fixes dependence on blob order, 3) is based on Euclidean distance rather than merely a bounding box
- Fix grid search/ROC formatting for arrays

ROI editor
- Refactor into separate module and classes
- Restore white color for all draggable circles to improve their visibility
- `bone` profile for bluish-greyscale high-contrast colormaps
- Fix display of blob verification matches above and below the ROI

Atlas editor
- edit button to toggle editing mode without a keyboard (`w` shortcut to toggle; `alt+click` no longer required/supported for editing)
- text box to input new label colors and provide feedback on selected colors

Image transformations
- `--transform` command-line parameter to specify image rotation and flipping
- Fix applying `--flip` argument to image export

## MagellanMapper v0.9.2

### Changes

Atlasing
- Enabled save button only with unsaved edits
- Tool to convert ITK-SNAP labels description format to CSV importable by Clrbrain
- Option to treat specific labels' corresponding histology regions as foreground for DSC to account for ventricular labels in E11.5
- Option to turn off lateral extension/mirroring completely while still using them to crop the output atlas

Image registration
- Save basic stats such as DSC after image registration

Nuclei detection
- Added `4xnuc` profile with size parameters for 4x microscopy

Python stats
- Bar plots: option to weight bars, more robust layout tightening and label placement
- Scatter plots: fixed grouping by multiple columns
- Saved peak smoothing qualities to CSV
- Added total intensity metric

R stats
- Jitter plots: allow different number of subgroups per group, fixed displaying multiple sets of paired data in the same plot
- Grouped metadata columns in output and simplified adding new metadata
- Fixed p-value extraction for correlation coefficient matrices"

## MagellanMapper v0.9.1

### Changes

Visualization
- Automatically generate colormaps for all channels
- Equalize transparency for all channels
- Invert 3D visualization along the x-axis for same "handedness" as in Matplotlib images

Image import
- Fixed regression in importing TIFF images to Numpy/Clrbrain format

Atlasing
- Option to merge images into separate channels
- No longer display images with registration suffixes unless explicitly specified
- Fixed regressions in atlas rotation during import

General stats
- Measure coefficient of variation
- Option to reverse order of conditions

Python stats
- Use nuclei for weighted averaging of nuclei-based measurements
- Library function for generating probability plots
- Configurable bar plots labels, vertical spans to further group bars

R stats
- Fixed regressions in jitter plots with mean summaries, single genotypes, unitless labels

Clean-up
- Removed obsolete volume and density functions
- No longer merge distance images

## MagellanMapper v0.9.0

### Changes

Server management
- Support swap file setup and loading
- Remove default NVMe device paths since they are not stable

Blob detections
- Reduce default blob chunk size to 1000 to reduce memory requirement

Atlasing
- Decrease erosion fraction to 50:50 for interior-border match metrics

R stats
- Correlation coefficient calculation and matrix visualization
- Normality testing"

### Dependency Updates

#### R Dependency Changes

- Hmisc
- corrplot
