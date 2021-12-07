# MagellanMapper v1.5 Release Notes

## MagellanMapper v1.5.1

### Highlights

### Changes

#### Installation

#### GUI

#### CLI

#### Atlas refinement

#### Atlas registration

#### Cell detection

#### Volumetric image processing

#### I/O

#### Server pipelines

#### Python stats and plots

#### R stats and plots

#### Code base and docs

### Dependency Updates

#### Python Dependency Changes

#### R Dependency Changes

#### Server dependency Changes


## MagellanMapper v1.5.0

### Highlights

This release brings many enhancements aimed at creating a smoother user experience. Key changes include:
- We overhauled the profile panel to preview each profile and show the currently loaded settings
- Window sizes and positions are now saved between sessions
- Image brightness-related settings are saved while scrolling through image planes
- On Windows platforms, we now support all major atlas refinement tasks
- Atlas smoothing supports adaptive kernel sizes per region and improves stat output
- Newly generated atlases now collect labels metadata to make the atlases more portable
- Image import supports single-plane RAW images and provides more feedback

### Changes

#### Installation

#### GUI

- Profile panel overhaul (#66)
    - Preview profiles before adding them
    - Skip the "Load" step; profiles are automaticaly loaded when added
    - View the complete settings for all loaded profiles
    - Clearer and more compact labels and button arrangement
    - Settings panel for resetting preferences and finding version information
- Window size and position are saved as user preferences
- The default window size is smaller to fit into 720p displays
- The default focus is no longer the main image file path to avoid accidental file loading
- Registered image selectors are now more compact
- Fixed image intensity values and auto-adjustment to persist for each channel while scrolling and switching among channels (#76)
- Fixed to retain opacity settings for borders images while scrolling (#79)

#### CLI

See the [table of CLI changes](../cli.md#changes-in-magellanmapper-v15) for a summary of all changes in v1.5

- Multiple processing tasks can be given in the same command; eg `--proc detect coloc_match` (#30)
- Image preprocessing tasks have been integrated into `--proc`, no longer requiring a separate ROI profile; eg `--proc preprocess=rotate`
- The new `--prefix_out` flag allows customizing output paths when `--prefix` is used to modify input paths (#73)

#### Atlas refinement

- Adaptive kernel sizes can be used for smoothing operations (#53)
- Metadata for labels images are saved when importing an atlas and registering the atlas to another image so that the original atlas no longer needs to be available (and `--labels` argument does not to be given) when loading the atlas or registered image (#65, #67)
- Smoothing metrics are output during the `--register merge_atlas_segs` task (#51, #54)
- The atlas profile settings `meas_edge_dists` and `meas_smoothing` turn off these metrics to save time during atlas generation, and the profile `fewerstats` turns off both these settings (#31)
- Multiprocessing is turned off for lateral extension for better performance
- Labels difference images, which map metrics for labels onto the labels themselves for data visualization, can now take specific metric files and columns from the included R package (#73)
- Labels referencs loaded from CSV files now support level, abbreviation, and parent ID columns (#61)
- Fixed multiprocessing tasks with SimpleElastix 2.0
- Fixed DSC metrics between the atlas and its new labels, and more DSC metrics are saved (#57)
- Fixed exporting labels reference files when a label ID is missing (#61)

#### Atlas registration

- Image masks can be set to focus the field for image registration. Use the new `--reg_suffixes fixed_mask=<suffix-or-abs-path> moving_mask=<suffix-path>` command-line sub-arguments to load these mask files. (#40)
- Register multiple atlases at a time, applying the same transformation to each of them. Specify additional atlase after the first, with output paths specified as prefixes, eg: `./run.py <sample-path> <atlas1> <atlas2> --prefix <atlas1-output-path> <atlas2-output-path>`. (#69)
- When registration fails, it will attempt to match more image parameters such as spacing and direction as fallbacks (#71)

#### Cell detection

- Previously saved blobs are no longer loaded prior to re-detection
- More flexibility when loading databases with blob truth sets
- Grid searches support output path modifiers
- Fixed blob segmentation and showing labels when none of either are present
- Fixed exporting ROIs

#### Volumetric image processing

#### I/O

- Improvements to loading registered images
    - The main image is no longer loaded if a registered `atlas` image is given
    - Images can be specified as absolute paths using `--reg_suffixes` to load any image, including those registered to another image (#36)
    - Images loaded for edge detection can be configured using `--reg_suffixes`
    - Files with two extensions (eg `.nii.gz`) are supported
    - Files modified by `--prefix` can now also be found in the registered image dropdowns
    - More support for CSV format reference files
- Improvements to image import
    - Single plane RAW images can be loaded when importing files from a directory, in addition to multiplane RAW files (#32)
    - Skips single plane files that give errors (eg non-image files in the input directory) (#83)
    - Provides import error feedback in the GUI (#83)
    - The known parts of the import image shape are populated even if the full shape is not known
    - The Bio-Formats library has been updated to support more file formats (from Bio-Formats 5.1.8 to 6.6.0 via Python-Bioformats 1.1.0 to 4.0.5, respectively) (#70)
    - Fixed to disable the import directory button when metadata is insufficient
    - Fixed to create parent directories when importing images (#44)
    - Fixed to create default resolutions even when none are specified (#44)
- Improvements to exporting image stacks
    - Images can be exported to multiple separate figures (#68)
    - Sub-plots are labeled
    - Image rotation arguments are applied (#50)
    - Plane index is only added when exporting multiple planes
- Fixed to update metadata files when loaded through the `--meta` flag (#35)
- Fixed error when unable to load a profile `.yml` file

#### Server pipelines

- Continuous integration has been implemented through GitHub Actions to improve quality control (#55)

#### Python stats and plots

- Scatter plot updates
    - Annotation columns can be names of index columns
    - Colors can be specified

#### R stats and plots

#### Code base and docs

- Multiprocessing tasks are now more widely supported in Windows (`spawn` start mode), including the `--register import_atlas`, `make_edge_images`, `merge_atlas_segs`, `vol_stats`, and `make_density_images` tasks (#60, #68)
- Type hints are now being integrated, replacing docstring types for better typing info and debugging (#46)

### Dependency Updates

#### Python Dependency Changes

- Python-Bioformats has been upgraded to 4.0.5 and uses a custom package that uses the existing Javabridge rather than Python-Javabridge to avoid a higher NumPy version requirement
- Workaround for failure to install Mayavi because of a newer VTK, now pinned to 9.0.1
- Matplotlib >= 3.2 is now required
- 