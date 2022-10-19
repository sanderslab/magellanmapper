# MagellanMapper v1.6 Release Notes

## MagellanMapper v1.6.0

### Highlights

- Available as binary wheel to install without requiring the source code
- Simpler entry point to launch MagellanMapper: `mm`
- Atlases can be downloaded directly through [`BrainGlobe`](https://github.com/brainglobe/bg-atlasapi) (see the new "Atlases" panel)
- Atlas regions can be searched (see "Atlases > Region")
- Detection channels can be selected independently of the loaded image to overlay prior detections or compare channels (see "Detect > Chl")
- Images can be viewed as RGB (see "ROI > Channels")
- [Jupyter Notebook tutorial](https://github.com/sanderslab/magellanmapper/blob/master/bin/sample_cmds_bash.ipynb) for running common tasks

### Changes

#### Installation

- Available as binary wheel to install without requiring the source code (#166)
- Entry point added to launch the app by: `mm` (#166)
- Conda install simplified to use Pip primarily, which reduces disk space usage (#166)
- Removed Conda `environment_light.yml` and OS-specific specs (#187)

#### GUI

- More preferences are saved, such as the figure save location and ROI Editor layout (#138, #201)
- Drag and remove loaded profiles in the Profiles tab table
- "Help" buttons added to open the online documentation (#109)
- Default confirmation labels can be set before detection (#115)
- Resets the labels reference file path when reloading an image in the GUI (#139)
- BrainGlobe panel: access atlases hosted by BrainGlobe directly from the GUI (#75)
- Registered image suffixes with variable endings (eg `annotationEdgeLevel<n>`) now show up in the dropdown boxes (#142)
- Registered image and region names are truncated in the middle to prevent expanding the sidebar for long names (#147)
- "Show all" in the Regions section of the ROI panel shows names for all labels (#145)
- Atlas Editor planes can be reordered or turned off (#180)
- New viewer that displays each blob separately to verify blob classifications (#193)
- Image adjustment
  - "Blend" option in the image adjustment panel to visualize alignment in overlaid images (#89)
  - Image adjustment channels are radio buttons for easier selection (#212)
  - Fixed synchronization between the ROI Editor and image adjustment controls after initialization (#142)
- Fixed to reset the ROI selector when redrawing (#115)
- Fixed to reorient the camera after clearing the 3D space (#121)
- Fixed to turn off the minimum intensity slider's auto setting when manually changing the slider (#126) 
- Fixed error window when moving the atlas level slider before a 3D image has been rendered (#139)
- Fixed saving blobs in an ROI using the first displayed ROI or after moving the sliders without redrawing (#139)
- Fixed browsing for files or directories in some environments (#201)
- Fixed clearing the import path (#201)

#### CLI

- The `--proc export_tif` task exports an NPY file to TIF format
- the `--transform interpolation=<n>` configures the type of interpolation when resizing images during stack export (#127)
- Any axis can be flipped through `--transform flip=<axis>`, where `axis = 0` for the z-axis, 1 for the y-axis, and 2 for the x-axis (#147)
- Density/heat maps can be specified through `--reg_suffixes density=<suffix>` (#129)
- Write point files for corresponding-point-based registration (#195)
- Fixed to only remove the final extension from image paths, and paths given by the `--prefix <path>` CLI argument do not undergo any stripping (#115)

#### Atlas refinement

- The atlas transformer (`atlas_refiner.transpose_img`) provides a more comprehensive set of typical transformations before atlas refinement or registration, such as rotation to any angle, flipping along any axis, and resizing (#195, #214)

#### Atlas registration

- Image registration now supports multiple labels images given as `--reg_suffixes annotation=<suffix1>,<suffix2>,...`, which will apply the same transformation to each of these images (#147)
- Landmark distance measurements save the raw distances and no longer require spacing (#147)

#### Cell detection

##### Detection

- Detection channels can be selected, including those not in the currently displayed image or loaded from a saved blobs file (#121)
- Auto-scrolls the detections table to the selected annotation (#109)
- `ctrl+[n]+click` to add a channel now sets the channel directly to `n` rather than to the `n`th seleted channel (#109)
- Added a slider to choose the fraction of 3D blobs to display (#121)
- Improved blob size slider range and readability (#121)
- Blob columns can be customized, including excluding or reordering columns (#133, #216)
- Existing blob archives are backed up before saving (#216) 
- Fixed to scale blobs' radii when viewing blobs detections on a downsampled image (#121)
- Fixed getting atlas colors for blobs for ROIs inside the main image (#121)
- Fixed blob segmentation for newer versions of Scikit-image (#91)
- Fixed verifying and resaving blobs
- Fixed loading blobs in the GUI with no blobs in the ROI or channels selected (#216)

##### Colocalization

- Match-based colocalization can run without the main image, using just its metadata instead (#117)
- These colocalizations are now displayed in the 3D viewer (#121)
- Fixed match-based colocalizations when no matches are found (#117, #120)
- Fixed slow loading of match-based colocalizations (#119, #123)

#### Volumetric image processing

- Match-based colocalizations use larger processing blocks to avoid gaps (#120)
- Voxel density maps no longer require a registered image, or it can be used in place of full-size image metadata (#125, #226)
- Grid search profiles are now layered on top of one another rather than applied in sequential runs for consistency with ROI and atlas profiles (#138)
- Fixed 3D surface area measurement with Scikit-image >= v0.19

#### I/O

- Images can be viewed as RGB(A) using the `RGB` button or the `--rgb` CLI argument (#142)
- Some TIF files can be loaded directly, without importing the file first (#90, #213)
- The `--proc export_planes` task can export a subset of image planes specified by `--slice`, or an ROI specified by `--offset` and `--size`
- Image metadata is stored in the `Image5d` image object (#115)
- Better 2D image support
  - Extended zero-crossing detection to 2D cases (#142)
  - Unit factor conversions adapts to image dimensions (eg 2D vs 3D) (#132)
- Multiple multiplane image files can be selected directly instead of relying on file auto-detection (#201)
- Fixed re-importing an image after loading it (#117)
- Fixed to store the image path when loading a registered image as the main image, which fixes saving the experiment name used when saving blobs (#139)

#### Server pipelines

#### Python stats and plots

- Generate swarm plots in Seaborn (#137)
- Color bars can be configured in ROI profiles using [settings in Matplotlib](https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.colorbar.html), update dynamically, and no longer repeat in animations (#128)
- New plot label sub-arguments (#135):
  - `--plot labels err_col_abs=<col>`: plot error bars with a column of absolute rather than relative values, now that Clrstats gives absolute values for effect sizes
  - `--plot_labels background=<color>`: change plot background color with a Matplotlib color string
  - `--plot_labels vspan_col=<col> vspan_format=<str>`: column denoting vertical span groups and string format for them, respectively (#135, 137)
- The figure save wrapper (`plot_support.save_fig`) is more flexible (#215)
- Discrete colormaps can use [Matplotlib named colors](https://matplotlib.org/stable/gallery/color/named_colors.html) and use them for symmetric colors (#226)
- Fixed errors when generating labels difference heat maps, and conditions can be set through `--plot_labels condition=cond1,cond2,...` (#132)
- Fixed alignment of headers and columns in data frames printed to console (#109)

#### R stats and plots

- Specify models with the `--model <stats.model>` CLI argument (#135)
- Effect size confidence intervals are now absolute rather than relative values for clarity (#135)
- Region IDs file is no longer required since volume stats output from the Python pipeline already includes this region metadata (#132)
- Added the `diff.means` stats model to simply give the difference of means between conditions (#135) 
- The `revpairedstats` profile is now `revconds` since it applies to reversing conditions in general, not just for paired stats (#132)
- Stats errors are caught rather than stopping the pipeline (#132)
- The labels reference path has been moved to an environment variable, which can be configured through `--labels <path>` (#147)
- The Shapiro-Wilks test has been implemented in `meansModel` for consistent table output (#164)
- Fixed t-test, which also provides Cohen's d as a standardized effect size through the `effectsize` package (#135)
- Fixed jitter plot box plots to avoid covering labels (#147)

#### Code base and docs

- Blob column accessors are encapsulated in the `Blobs` class, which allows for flexibility in column inclusion and order (#133)
- Settings profiles are being migrated from dictionaries to data classes to document and configure settings more easily (#138)
- Jupyter Notebook as a tutorial for running various tasks in the CLI (#122)
- Documentation is now [hosted on ReadTheDocs](https://magellanmapper.readthedocs.io/en/latest/index.html), using the Furo theme (#225)

### Dependency Updates

#### Python Dependency Changes

- Python 3.8 is the default version now that Python 3.6 has reached End-of-Life
- The BrainGlobe Atlas API package (`bg-atlasapi`) dependency has been added to access a suite of cloud-based atlases (#75)
- The `dataclasses` backport is installed for Python < 3.7
- `Tifffile` is now a direct dependency, previously already installed as a sub-dependency of other required packages
- `Imagecodecs` is optional for `tifffile` but required for its uses here as of `tifffile v2022.7.28` and thus added as a dependency (#153)
- Updated to use the `axis_channel` parameter in Scikit-image's `transform.rescale` function (#115)
- Seaborn as an optional dependency for additional plot support (currently only swarm plots, #137)
- Scikit-learn is an optional rather than a required dependency (#150)
- The AWS-related dependencies (`boto3`, `awscli`) are also no longer installed in Conda environments (#150)
- The `jupyter` install group installs packages for running the Jupyter sample commands notebook in a Bash kernel (#122)
- Missing dependencies are starting to use more consistent error messages and instructions (#226)
- Python 3.6 and 3.7 have separate pinned dependencies (`envs/requirements_py3<n>`) (#232)

#### R Dependency Changes

- `effectsize` is a suggested dependency for Cohen's d, used in t-tests (#135)

#### Server dependency Changes
