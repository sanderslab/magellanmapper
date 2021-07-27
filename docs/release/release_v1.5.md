# MagellanMapper v1.5 Release Notes

## MagellanMapper v1.5.0

### Highlights

### Changes

#### Installation

#### GUI

- Profile panel overhaul
    - Preview profiles befor adding
    - Skip the "Load" step; profiles are automaticaly loaded when added
    - View the complete settings for all loaded profiles
    - Clearer and more compact labels and button arrangement
    - Settings panel for resetting preferences and finding version information
- Window size and position are saved as user prferences
- The default window size is smaller to fit into 720p displays
- The default focus is no longer the main image file path to avoid accidental file loading
- Registered image selectors are now more compact

#### CLI

See the [table of CLI changes](../cli.md#changes-in-magellanmapper-v15) for a summary of all changes in v1.5

- Multiple processing tasks can be given in the same command; eg `--proc detect coloc_match`
- Image preprocessing tasks have been integrated into `--proc`, no longer requiring a separate ROI profile; eg `--proc preprocess=rotate`
- The new `--prefix_out` flag allows customizing output paths when `--prefix` is used to modify input paths

#### Atlas refinement

- Adaptive kernel sizes can be used for smoothing operations
- Metadata for labels images are saved when importing an atlas and registering the atlas to another image so that the original atlas no longer needs to be available (and `--labels` argument does not to be given) when loading the atlas or registered image
- Smoothing metrics are output during the `--register merge_atlas_segs` task
- The atlas profile settings `meas_edge_dists` and `meas_smoothing` turn off these metrics to save time during atlas generation, and the profile `fewerstats` turns off both these settings
- Multiprocessing is turned off for lateral extension for better performance
- Fixed multiprocessing tasks with SimpleElastix 2.0
- Fixed DSC metrics between the atlas and its new labels, and more DSC metrics are saved

#### Atlas registration

- Image masks can be set to focus the field for image registration. Use the new `--reg_suffixes fixed_mask=<suffix-or-abs-path> moving_mask=<suffix-path>` command-line sub-arguments to load these mask files.
- Register multiple atlases at a time, applying the same transformation to each of them. Specify additional atlase after the first, with output paths specified as prefixes, eg: `./run.py <sample-path> <atlas1> <atlas2> --prefix <atlas1-output-path> <atlas2-output-path>`.

#### Cell detection

- Previously saved blobs are no longer loaded prior to re-detection
- Fixed blob segmentation and showing labels when none of either are present

#### Volumetric image processing

#### I/O

- Improvements to loading registered images
    - The main image is no longer loaded if a registered `atlas` image is given
    - Images can be specified as absolute paths to load any image, including those registered to another image
    - Images loaded for edge detection can be configured using `--reg_suffixes`
    - Files with two extensions (eg `.nii.gz`) are supported
    - Files modified by `--prefix` can now also be found in the registered image dropdowns
    - More support for CSV format reference files
- Improvements to exporting image stacks
    - Images can be exported to multiple separate figures
    - Sub-plots are labeled
    - Image rotation arguments are applied
    - Plane index is only added when exporting multiple planes
- Improvements to image import
    - Single plane RAW images can be loaded when importing files from a directory, in addition to multiplane RAW files
    - The known parts of the import image shape are populated even if the full shape is not known
    - The Bio-Formats library has been updated to support more file formats (from Bio-Formats 5.1.8 to 6.6.0 via Python-Bioformats 1.1.0 to 4.0.5, respectively)
    - Fixed to disable the import directory button when metadata is insufficient
    - Fixed to create parent directories when importing images
    - Fixed to create default resolutions even when none are specified
- Fixed to update metadata files when loaded through the `--meta` flag
- Fixed error when unable to load a profile `.yml` file

#### Server pipelines

- Continuous integration has been implemented through GitHub Actions to improve quality control

#### Python stats and plots

#### R stats and plots

#### Code base and docs

- Multiprocessing tasks are now more widely supported in Windows (`spawn` start mode), including the `--register import_atlas`, `make_edge_images`, `merge_atlas_segs`, `vol_stats`, and `make_density_images` tasks
- Type hints are now being integrated, replacing docstring types for better typing info and debugging

### Dependency Updates

#### Python Dependency Changes

- Python-Bioformats has been upgraded to 4.0.5 and uses a custom package that uses the existing Javabridge rather than Python-Javabridge to avoid a higher NumPy version requirement
- Workaround for failure to install Mayavi because of a newer VTK, now pinned to 9.0.1

#### R Dependency Changes

#### Server dependency Changes
