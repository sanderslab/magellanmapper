# MagellanMapper v1.6 Release Notes

## MagellanMapper v1.6.0

### Highlights

- Smoother, faster interactions with main plots, including atlas label name display, label editing, and pan and zoom navigation
- Available as binary wheel to install without requiring the source code
- Faster, lighter installation with fewer required dependency packages
- Extends suppor to Python 3.11
- Simpler entry point to launch MagellanMapper: `mm`
- Supports ITK-Elastix for image registration
- Atlases can be downloaded directly through [`BrainGlobe`](https://github.com/brainglobe/bg-atlasapi) (see the new "Atlases" panel)
- Atlas regions can be searched (see "Atlases > Region")
- Atlas Editor planes can be reordered or turned off
- Detection channels can be selected independently of the loaded image to overlay prior detections or compare channels (see "Detect > Chl")
- Images can be viewed as RGB (see "ROI > Channels") or merged channels
- [Jupyter Notebook tutorial](https://github.com/sanderslab/magellanmapper/blob/master/bin/sample_cmds_bash.ipynb) for running common tasks

### Changes

#### Installation

- Available as binary wheel to install without requiring the source code (#166)
- Entry point added to launch the app by: `mm` (#166)
- Conda install simplified to use Pip primarily, which reduces disk space usage (#166)
- Removed Conda `environment_light.yml` (no GUI) and OS-specific specs (#187)
- No-GUI (headless) install is now by default, with a new `gui` install group to include the GUI (#317)
- Fixed instructions for installing by Pip in ZSH terminals (#485)

#### GUI

- More preferences are saved, such as the figure save location and ROI Editor layout (#138, #201, #575)
- Drag and remove loaded profiles in the Profiles tab table
- "Help" buttons added to open the online documentation (#109)
- Default confirmation labels can be set before detection (#115)
- Resets the labels reference file path when reloading an image in the GUI (#139)
- BrainGlobe panel: access atlases hosted by BrainGlobe directly from the GUI (#75)
- Registered image suffixes with variable endings (eg `annotationEdgeLevel<n>`) now show up in the dropdown boxes (#142)
- Registered image and region names are truncated in the middle to prevent expanding the sidebar for long names (#147)
- "Show all" in the Regions section of the ROI panel shows names for all labels (#145)
- Atlas Editor planes can be reordered or turned off (#180)
- EXPERIMENTAL: New viewer that displays each blob separately to verify blob classifications (#193)
- Image adjustment
  - Select colormaps for each channel (#574)
  - Settings are preserved when redrawing the image (#613)
  - "Merge" option in the ROI panel to merge channels using additive blending (#492, #552)
  - "Blend" option in the image adjustment panel to visualize alignment in overlaid images (#89, #450, #607)
  - Synced range of "filtered" ROI and overview images (#613)
  - Image adjustment channels are radio buttons for easier selection (#212)
  - Fixed synchronization with image adjustment controls (#142, #576)
  - Fixed redundant triggers when adjusting the displayed image (#474)
  - Fixed intensity sliders to cover the full range (#572, #576, #606, #613)
- Images are rotated by dynamic transformation (#214, #471, #505)
- Smoother, faster interactions with main plots, including atlas label name display, label editing, and pan and zoom navigation (#317, #335, #359, #367)
- Atlas labels adapt better in zoomed images to stay within each plot (#317)
- Fixed to reset the ROI selector when redrawing (#115)
- Fixed to reorient the camera after clearing the 3D space (#121)
- Fixed to turn off the minimum intensity slider's auto setting when manually changing the slider (#126) 
- Fixed error window when moving the atlas level slider before a 3D image has been rendered (#139)
- Fixed saving blobs in an ROI using the first displayed ROI or after moving the sliders without redrawing (#139)
- Fixed browsing for files or directories in some environments (#201)
- Fixed clearing the import path (#201)
- Fixed saving additional, old figures when using the save shortcut (#256)
- Fixed lag in zoom direction change (#359)
- Fixed conflict between shortcut to add blob and jumping to ROI plane (`ctrl+click`) by changing the jump shortcut to `j+click` (#456)

#### CLI

- The `--proc export_tif` task exports an NPY file to TIF format
- the `--transform interpolation=<n>` configures the type of interpolation when resizing images during stack export (#127)
- Any axis can be flipped through `--transform flip=<axis>`, where `axis = 0` for the z-axis, 1 for the y-axis, and 2 for the x-axis (#147)
- Density/heat maps can be specified through `--reg_suffixes density=<suffix>` (#129)
- Write point files for corresponding-point-based registration (#195)
- Quieter console output by default (#335)
- Fixed to only remove the final extension from image paths, and paths given by the `--prefix <path>` CLI argument do not undergo any stripping (#115)

#### Atlas refinement

- The atlas transformer (`atlas_refiner.transpose_img`) provides a more comprehensive set of typical transformations before atlas refinement or registration, such as rotation to any angle, flipping along any axis, and resizing (#195, #214)
- Edge/perimeter thickness can be customized (#307)
- Fixed groupwise registration for current atlas profiles, turned off default cropping (#444)

#### Atlas registration

- Image registration now supports multiple labels images given as `--reg_suffixes annotation=<suffix1>,<suffix2>,...`, which will apply the same transformation to each of these images (#147)
- Landmark distance measurements save the raw distances and no longer require spacing (#147)
- Masks with angle planes can be constructed (#252)
- `register.RegImgs` is a data class to track registered images (#335)
- Supports image I/O and registration through ITK
  - `ITK-Elastix` support has been added as an alternative to `SimpleITK` (#495, #500, #504)
  - Both libraries are now optional rather than required dependencies (#497)
  - Better support for image direction metadata (#497)
- Fixed changes to large label IDs during registration (#303)

#### Cell detection

##### Detection

- Detection channels can be selected, including those not in the currently displayed image or loaded from a saved blobs file (#121, #449)
- Auto-scrolls the detections table to the selected annotation (#109)
- `ctrl+[n]+click` to add a channel now sets the channel directly to `n` rather than to the `n`th seleted channel (#109)
- Added a slider to choose the fraction of 3D blobs to display (#121)
- Improved blob size slider range and readability (#121)
- Blob columns can be customized, including excluding or reordering columns (#133, #216, #449, #475)
- Existing blob archives are backed up before saving (#216)
- Basic spectral unmixing through channel subtraction (#458)
- Fixed to scale blobs' radii when viewing blobs detections on a downsampled image (#121)
- Fixed getting atlas colors for blobs for ROIs inside the main image (#121)
- Fixed blob segmentation for newer versions of Scikit-image (#91)
- Fixed verifying and resaving blobs
- Fixed loading blobs in the GUI with no blobs in the ROI or channels selected (#216)

##### Colocalization

- Match-based colocalization can run without the main image, using just its metadata instead (#117)
- These colocalizations are now displayed in the 3D viewer (#121)
- Specific match-based colocalizations channels can be set, eg `--channels 0 2` (#451)
- Fixed match-based colocalizations when no matches are found (#117, #120)
- Fixed slow loading of match-based colocalizations (#119, #123)

#### Block processing

- Match-based colocalizations use larger processing blocks to avoid gaps (#120)
- Voxel density maps no longer require a registered image, or it can be used in place of full-size image metadata (#125, #226)
- Grid search profiles are now layered on top of one another rather than applied in sequential runs for consistency with ROI and atlas profiles (#138)
- Fixed 3D surface area measurement with Scikit-image >= v0.19
- Fixed preprocessing to use the same blocks even if `verify_tol_factor` differs (#603)

#### I/O

- Images can be viewed and exported as RGB(A) using the `RGB` button or the `--rgb` CLI argument (#142, #445)
- EXPERIMENTAL: Some TIF files can be loaded directly, without importing the file first (#90, #213, #242, #445)
- The `--proc export_planes` task can export a subset of raw image planes without processing, specified by `--slice`, or an ROI specified by `--offset` and `--size`
- Image metadata is stored in the `Image5d` image object (#115)
- Better 2D image support
  - Extended zero-crossing detection to 2D cases (#142)
  - Unit factor conversions adapts to image dimensions (eg 2D vs 3D) (#132)
  - Fixed ROI padding during blob verification and match-based colocalization for 2D images (#380)
- Multiple multiplane image files can be selected directly instead of relying on file auto-detection (#201)
- `openpyxl` package is now optional during region export (#445)
- Use core fonts in PDF/PS file exports to keep vector text (#486)
- Fixed re-importing an image after loading it (#117)
- Fixed to store the image path when loading a registered image as the main image, which fixes saving the experiment name used when saving blobs (#139)
- Fixed parsing some metadata when importing files with Bio-Formats (#502)
- Fixed `--proc extract` for extracting a slice of multiple planes (#611)

#### Server pipelines

#### Python stats and plots

- Generate swarm and category plots in Seaborn (#137, #253, #612)
- Color bars can be configured in ROI profiles using [settings in Matplotlib](https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.colorbar.html), update dynamically, and no longer repeat in animations (#128)
- New plot label sub-arguments (#135):
  - `--plot labels err_col_abs=<col>`: plot error bars with a column of absolute rather than relative values, now that Clrstats gives absolute values for effect sizes
  - `--plot_labels background=<color>`: change plot background color with a Matplotlib color string
  - `--plot_labels vspan_col=<col> vspan_format=<str>`: column denoting vertical span groups and string format for them, respectively (#135, 137)
  - `--plot_labels rotation=<deg>`: change rotation in degrees (#445)
- 2D plot tasks
  - `--plot_2d decorate_plot` to add plot decorations such as title and axis labels (#457)
  - Tasks can be run programmatically as: `plot_2d.main(plot_2d_type=<config.Plot2DTypes>)` (#457)
- The figure save wrapper (`plot_support.save_fig`) is more flexible (#215)
- 2D plots can be set not to save (#445)
- Discrete colormaps can use [Matplotlib named colors](https://matplotlib.org/stable/gallery/color/named_colors.html) and use them for symmetric colors (#226)
- Vertical span labels adapt to the axes rather than data limits (#472, #612)
- Scatter plots support jitter and x-tick rotation (#486)
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
- One-sample t-tests and Wilxocon tests and the Shapiro-Wilks test have been implemented in `meansModel` (#164, #470)
- A basic `NAMESPACE` file is provided to fix installation and exporting functions (#303)
- Linear regression intercept term can be toggled using the `Intercept` environment field, and r<sup>2</sup> and intercept are exported (#445)
- Fixed t-test, which also provides Cohen's d as a standardized effect size through the `effectsize` package (#135)
- Fixed jitter plot box plots to avoid covering labels (#147)
- Fixed model fitting (#240, #304)
- Fixed plotting in PyCharm (#304)

#### Server pipelines

- The server setup script now accepts `-m [mount-path]` to set a custom mount path target (#469)

#### Code base and docs

- Blob column accessors are encapsulated in the `Blobs` class, which allows for flexibility in column inclusion and order (#133)
- Settings profiles are being migrated from dictionaries to data classes to document and configure settings more easily (#138)
- Jupyter Notebook as a tutorial for running various tasks in the CLI (#122)
- Documentation is now [hosted on ReadTheDocs](https://magellanmapper.readthedocs.io/en/latest/index.html), using the Furo theme (#225, #563)
- Default arguments are documented in API auto-docs (#485)
- Expand continuous integration testing to both pinned and fresh dependencies across Python 3.6-3.11 (#75, #101, #252, #342, #538)

### Dependency Updates

#### Python Dependency Changes

- Python 3.10-3.11 are now supported (#379, #517)
- Python 3.9 is the default version now that Python 3.6 has reached End-of-Life, and NumPy no longer supports Python 3.8 (#559, #563)
- Python 3.6-8 have been deprecated for removal in MM v1.7 and have separate pinned dependencies (`envs/requirements_py3<n>`) (#232, #379)
- Custom dependency binaries are now built for Python 3.8-3.11 (#379)
- The BrainGlobe Atlas API package (`bg-atlasapi`) dependency has been added to access a suite of cloud-based atlases (#75, #443, #498)
- The `dataclasses` backport is installed for Python < 3.7
- `Tifffile` is now a direct dependency, previously already installed as a sub-dependency of other required packages
- `Imagecodecs` is optional for `tifffile` but required for its uses here as of `tifffile v2022.7.28` and thus added as a dependency (#153)
- Updated to use the `axis_channel` parameter in Scikit-image's `transform.rescale` function (#115)
- Seaborn as an optional dependency for additional plot support (currently only swarm plots, #137)
- Scikit-learn is an optional rather than a required dependency (#150)
- The AWS-related dependencies (`boto3`, `awscli`) are now optional, installed in the `aws` group (#150, #379)
- Mayavi/VTK are now optional, installed in the `3d` group (#455, #618)
- The `jupyter` install group installs packages for running the Jupyter sample commands notebook in a Bash kernel (#122)
- Missing dependencies are starting to use more consistent error messages and instructions (#226)
- The SimpleElastix custom binaries are now built on SimpleITK with Elastix and is no longer a required dependency (#379, #501)
- ITK-Elastix is now supported for image registration (#501)
- Supports TraitsUI v8 (#510)
- Fixed error on deprecated NumPy data type aliases (#364)
- Fixed `qt4 backend` error by avoiding PyQt v5.15.8 (#431)

#### R Dependency Changes

- `effectsize` is a suggested dependency for Cohen's d, used in t-tests (#135)

#### Server dependency Changes
