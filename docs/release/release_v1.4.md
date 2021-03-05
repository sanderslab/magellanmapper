# MagellanMapper v1.4 Release Notes

## MagellanMapper v1.4.0

### Changes

Installation

GUI
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

CLI
- The new `--load` parameter replaces `--proc load` as a more flexible way to specify data to load, including `--load blobs` and `--load blobs blob_matches`
- Output of profiles settings is now pretty printed for readability

Atlas refinement
- Option to increase tapering during labels lateral extension by weighting label erosion with lateral distance, set by the `wt_lat` atlas profile setting
- Set an alternative intensity image for edge detection using the registration suffixes atlas flag (`--reg_suffixes [atlas]`)
- Added a `watershed_mask_filter` setting in the `edge_aware_reannotation` atlas profile group to set the filter type and size for the watershed mask
- `atlas_mirror` profile setting to toggle mirroring the intensity image across hemispheres during atlas curation
- Fixed to exclude labels that were not eroded from undergoing watershed-based reannotation
- Fixed incorrect color mapping for some corresponding labels (ie same region in opposite hemispheres)

Atlas registration
- Customize the atlas images used during image registration by using the `--reg_suffixes` CLI parameter
- Measure the distance from labels to specified landmarks before and after registration through the `--register labels_dist` task
- The `carve_threshold` and `holes_area` atlas profile settings are also applied to regular (non-groupwise) registration
- Specify a full fallback atlas profile rather than only a fallback similarity metric if the post-registration DSC falls below threshold (`metric_sim_fallback` setting)
- The similarity metric used for registration is included in the summary CSV file
- Fixed smoothing metrics for non-existent labels

Cell detection
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
- Fixed applying the first channel's profile setting for image saturation to all channels during blob detection

Volumetric image processing
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

I/O
- PDF export
    - Use nearest neighbor interpolation for consistency with 2D image export to other formats
    - Avoid merging layers by turning off image compositing
- Matplotlib style is set more consistently to "default"
- Intensity-based co-localizations are stored in the blobs archive
- Database
    - New table for blob matche
    - Support foreign keys
- Atlas labels export to CSV can output the immediate parent of each label to reconstruct label hierarchy by using `--register export_regions --labels level=None orig_colors=1`, where `level=None` gets the parent rather than labels only up to that level, and `orig_colors=1` gets only labels present in the image itself
- `--proc export_planes` now exports multi-channel images combined into single planes (eg RGB images), while the new `--proc export_planes_channels` exports each image to a separate channel
- Animations can display the plane number by using the `--plot_labels text_pos=<x,y>` to specify where to place the label
- The `--series` flag is now supported for import in the GUI
- Fixed reading image size and resolution metadata when values for some dimensions are missing
- Fixed import RGB images
- Fixed redundant channel import for some formats (eg some OME-TIFF files)
- Fixed to reset blobs when loading a new image

Server pipelines

Python stats and plots
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

R stats and plots
- A basic command-line interface has been integrated through `run.R`, including path, profile, and measurement configuration
- Use the `tryCatchLog` package to assist with stacktraces for debugging
- Update usage of `addTextLabels` to its successor package, `basicPlotterR`
- Provide feedback when plots fail to display
- Option to load custom profiles from `.R` files
- Profile parameter to customize y-axis limits
- Wilcoxon Signed Rank test now uses a standardized effect size, using the Z-statistic computed by the `rcompanion` package
- Log-scaled volcano plots use a log-modulus transform, which fixes transforms when the minimum absolute value is 0
- Fixed to generate plots in both interactive and non-interactive environments

Code base and docs
- More links to external packages in API docs
- Instructions on building the API docs
- `Blobs` and `Image5d` are being migrated to class structures for better encapsulation and additional metadata

### Dependency Updates

#### Python Dependency Changes

#### R Dependency Changes

#### Server dependency Changes
