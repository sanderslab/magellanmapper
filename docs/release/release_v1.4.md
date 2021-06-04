# MagellanMapper v1.4 Release Notes

## MagellanMapper v1.4.0

### Highlights

This release brings many usability enhancements, foremost of which is our new standalone installers. Install and run MagellanMapper through point-and-click (no terminal required!) and open new files through your file browser.

We also added new options to the editors such as ROI centering/zooming and labels toggle, and we reorganized the tool panel for a cleaner, clearer look. Several tools are more responsive, such as Maximum Intensity Projections and Atlas Editor positioning.

Under the hood, we added blob co-localization across channels. Image volumes can be compared using different atlases. Imported images store metadata in a YAML file for readability as plain text files. Python support is extended to 3.6-3.8, and the R stats package supports a basic command-line interface and profiles.

For a complete list of command-line changes, please see this [table](../cli.md#changes-in-magellanmapper-v14).

For debugging, logs are now saved to `out.log` located in:
- Windows: `C:\Users\<your-username>\AppData\Local\MagellanMapper`
- macOS: `/Users/<your-username>Library/Application Support/MagellanMapper`
- Linux: `/home/<your-username>/.local/share/MagellanMapper`

### Changes

#### Installation

- Windows, macOS, and Linux standalone packages are now provided!
    - MagellanMapper can now be installed without the command-line
    - The Windows installer allows launching the application from the Start Menu
    - The macOS app can be dragged to the Applications to access from Launchpad
- Environment setup scripts support package updates
    - As we added a [new dependency](#python-dependency-changes), we made it easier to update existing environments
    - Re-running `bin\setup_conda.bat` on Windows updates as `bin/setup_conda` has on Mac/Linux
    - `bin/venv.sh` can now also be re-run to update Venv environments
- Python version support has been expanded to 3.6-3.8 now that we have [built](#python-dependency-changes) custom dependencies for these Python versions

#### GUI

- Reorganized options to group by viewer
- More tooltips (hover mouse over labels)
- Option to treat the ROI offset as the center of the ROI
- Atlas Editor
    - Zoom in to the ROI in the Atlas Editor
    - Paintbrush only appears in Edit mode
    - Annotating an image without labels will generate a new labels image file
    - Option to move the editor planes when moving the ROI sliders (on by default)
    - Option to turn off crosslines
- Atlas labels
    - Option to turn off atlas labels
    - Label selection options to include both sides and child labels
- 3D visualization
    - Adjust 3D surface opacities to look inside structures
    - Overlay blobs detected in full resolution images onto downsampled images
    - Blobs can be selected to view an ROI around specific blobs
    - 3D Atlas regions and ROIs can be added sequentially (keep the new "clear" option unchecked) 
    - Fixed shadow panes for multichannel images, isotropic visualization, and z-axis inversion
- Maximum intensity projections
    - Support added to the Atlas Editor
    - Automatically applied to both the ROI and Atlas Editor when toggled
- Color styles for blobs! Color by atlas labels, channel, or unique per blob (original style)
- Registered images selections are grouped into dropdowns for a cleaner look
- Refreshes viewers when the ROI changes in more cases
- Fixed error when looking up atlas label without a loaded reference file
- Fixed the size of the ROI outline after detecting blobs

#### CLI

- Unrecognized arguments are simply ignored rather than giving an error
- The new `--load` parameter replaces `--proc load` as a more flexible way to specify data to load, including `--load blobs` and `--load blobs blob_matches`
- Output of profiles settings is now pretty printed for readability

#### Atlas refinement

- Option to increase tapering during labels lateral extension by weighting label erosion with lateral distance, set by the `wt_lat` atlas profile setting
- Set an alternative intensity image for edge detection using the registration suffixes atlas flag (`--reg_suffixes [atlas]`)
- Added a `watershed_mask_filter` setting in the `edge_aware_reannotation` atlas profile group to set the filter type and size for the watershed mask
- `atlas_mirror` profile setting to toggle mirroring the intensity image across hemispheres during atlas curation
- Fixed to exclude labels that were not eroded from undergoing watershed-based reannotation
- Fixed saving edited images loaded through the GUI (#11)

#### Atlas registration

- Customize the atlas images used during image registration by using the `--reg_suffixes` CLI parameter
- Measure the distance from labels to specified landmarks before and after registration through the `--register labels_dist` task
- The `carve_threshold` and `holes_area` atlas profile settings are also applied to regular (non-groupwise) registration
- The similarity metric used for registration is included in the summary CSV file
- Fixed smoothing metrics for non-existent labels

#### Cell detection

- Blob co-localization
    - Detected blobs can now be co-localized two ways:
        1. Intensity-based: intensities above threshold at each blob's location in the remaining channels are considered co-localized signal
        2. Match-based: blobs from different channels are matched to find spatially overlapping blobs, similarly to automated blob verification against ground truth
    - The co-localization method can be set in the GUI when detecting blobs for a given ROI, shown as overlaid channel numbers (intensity-based) or corresponding blob numbers (match-based)
    - The `--proc detec_coloc` task performs intensity-based co-localization during whole image detections
    - The `--proc coloc_match` task performs match-based co-localization after detections were completed
    - Load blob matches with `--load blob_matches`
- Block processing settings can be set per channel rather than using the same settings for all channels; any block setting difference compared with other channels' profiles will trigger processing in separate blocks
- Accuracy metrics for each ROI are saved to CSV file
- Compare atlases translated to labels from different references and children

#### Volumetric image processing

- Volume comparisons: include raw pixel and volume counts
- Compare volumes registered to different atlases
    - Translate atlas labels IDs in one image to the IDs used in another image
        - `--atlas_labels translate_labels=<translation.csv>`, where `translation.csv` is a CSV file with `FromLabel` and `ToLabel` columns
        - `--atlas_labels translate_children=1` causes children of the given labels to be translated to the ID as well
        - Multiple translation files can be given (separate paths by `,`) to translate IDs in each image file
    - Option to compare volumes of only ROIs within a whole image using the `crop_to_first_image` option to compare matching volumes between two images by cropping the second image to the size of the first image
- Option to specify the registered images used for volume metrics through `--reg_suffixes`
- Option to specify channel(s) to include in heatmaps
- Blobs positions are scaled to the main image

#### I/O

- Open image files through file browsers (eg macOS Finder, Windows Explorer) (#18)
    - Open files in key supported file formats (eg `.npy`, `.mhd`, `.nii.gz`) by double-clicking or using "Open with...", or drag-n-drop onto the application icon in macOS
    - If MagellanMapper already has a loaded image, another application instance will be opened with the new image
- Open files on macOS through URIs: `open magmap:<path>`
- Image file metadata now uses YAML format for human-readable files; NPZ files are still supported for backward-compatibility
- Logging
    - Logging using the in-built Python `logging` facility is now supported, including output to log files
    - The `--verbose level=<n> log_path=<path>` flag specifies the log level from 1 (`DEBUG`, most verbose) to 5 (`CRITICAL`, least verbose) and log output path
    - Unhandled exceptions are now logged (saved to a temp file if caught before logging is set up) (#17)
- PDF export
    - Use nearest neighbor interpolation for consistency with 2D image export to other formats
    - Avoid merging layers by turning off image compositing
- Matplotlib style is set more consistently to "default"
- Intensity-based co-localizations are stored in the blobs archive
- Database
    - New table for blob matches
    - Support foreign keys
- Atlas labels export to CSV can output the immediate parent of each label to reconstruct label hierarchy by using `--register export_regions --labels level=None orig_colors=1`, where `level=None` gets the parent rather than labels only up to that level, and `orig_colors=1` gets only labels present in the image itself
- `--proc export_planes` now exports multi-channel images combined into single planes (eg RGB images), while the new `--proc export_planes_channels` exports each image to a separate channel
- Animations can display the plane number by using the `--plot_labels text_pos=<x,y>` to specify where to place the label
- The `--series` flag is now supported for import in the GUI
- Fixed to reset blobs when loading a new image

#### Python stats and plots

- Perform arithmetic operations on data frame columns using `--df sum_cols`, `subtract_cols`, `multiply_cols`, `divide_cols`
- Data frame task to replace values (`--df replace_vals`)
- Added `--plot_labels x_scale` and `y_scale` parameters to set axis scaling, such as `log` for log-scaling
- Support mixed Enum and non-Enum column names in Pandas data frames
- Generate parent directories if necessary before saving a data frame
- Option to label plot lines at right edge rather than in legend
- Figures are saved by default to PNG format, even if no extension is given
- Fixed matching label rows when weighting metrics
- Fixed unnecessary decimal numbers for integers in scatter plot annotations
- Fixed error when saving a figure to an unsupported file format

#### R stats and plots

- A basic command-line interface has been integrated through `run.R`, including path, profile, and measurement configuration
- Use the `tryCatchLog` package to assist with stacktraces for debugging
- Update usage of `addTextLabels` to its successor package, `basicPlotterR`
- Provide feedback when plots fail to display
- Option to load custom profiles from `.R` files
- Profile parameter to customize y-axis limits
- Wilcoxon Signed Rank test now uses a standardized effect size, using the Z-statistic computed by the `rcompanion` package
- Log-scaled volcano plots use a log-modulus transform, which fixes transforms when the minimum absolute value is 0
- Fixed to generate plots in both interactive and non-interactive environments

#### Code base and docs

- Python APIs
    - Previously Python APIs compatible with both Python 2 and 3 have been used when possible, but much of the package requires Python 3, and testing has been on Python >= 3.6
    - For a more consistent and modern codebase, we are initiating use of Python 3 APIs such as `pathlib` and specifically 3.6+ features such as f-strings
- Command-line arguments are now documented in a [table](../cli.md#command-line-argument-reference) for easier reference
- More links to external packages in API docs
- Instructions on building the API docs (#3)
- Readme cleanup (#2) and tabular format for Atlas Editor shortcuts (#5)
- `Blobs` and `Image5d` are being migrated to class structures for better encapsulation and additional metadata
- Initial unit testing, starting with `libmag` (#7)

### Dependency Updates

#### Python Dependency Changes

- The `appdirs` [package](https://github.com/ActiveState/appdirs) has been added to store user application files in standard operating system-dependent locations, allowing the application to be stored in locations without standard write access (see [settings](../settings.md#user-application-folder) for more details)
  - The database (`magmap.db`) is generated and stored there
  - Any existing `magmap.db` in the project root folder is left in place; copy it to the user application folder if you would like to use it
- Custom wheels have been built for SimpleElastix and Javabridge on Python 3.6-3.9 (#9)
  - Wheels are compatible with macOS 10.9+, Windows 10, and Linux glibc 2.23 (eg Ubuntu 16.04)
  - Python 3.9 is not yet supported for MagellanMapper because VTK 9 currently does not support this Python version
