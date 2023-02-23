# Command-Line Interface for MagellanMapper

## Command-Line argument reference

Argument | Sub-argument | Function | Ver Added | Last updated
--- | --- | --- | --- | ---
`--img` | `<path> [path2] ...` | Image paths. `--img` can be omitted if paths are given as the first argument. Can be given as the original filename (eg `myvolume.tiff`), the imported name, (`myvolume_image5d.npy`), a sub-image name (`myvolume_(x,y,z)x(x,y,z)_subimg.npy`), or a registered image (`myvolume_atlasVolume.mhd`). | v1.0.0 | [v1.4.0](#changes-in-magellanmapper-v14)
`--meta` | `<metadata-path> [path2] ...` | Metadata paths corresponding to images from `--img`. | v1.0.0 | v1.0.0
`--prefix` | `<path1> [path2] ...` | Path input/output prefix(es), typically used for the start of output paths. May also modify input paths, eg with `--img`. Takes precedence over `--img` for `--reg_suffixes`. <ul><li>*Since [v1.5.0](#changes-in-magellanmapper-v15):* Multiple paths can be given.</li></ul> | v1.0.0 | [v1.5.0](#changes-in-magellanmapper-v15)
`--prefix_out` | `<path1> [path2] ...` | Path output prefix(es), typically used for the start of output paths when `--prefix` is used for input paths. <ul><li>*Added in [v1.5.0](#changes-in-magellanmapper-v15)*</li></ul> | v1.5.0 | [v1.5.0](#changes-in-magellanmapper-v15)
`--rgb` | None | Open images as RGB(A). <ul><li>*Added in [v1.6.0](#changes-in-magellanmapper-v16)*</li></ul> | [v1.6.0](#changes-in-magellanmapper-v16)
`--suffix` | `<path>` | Path suffix, typically used for the end of output paths. <ul><li>*Since [v1.5.0](#changes-in-magellanmapper-v15):* Multiple paths can be given.</li></ul> | v1.0.0 | v1.0.0
`--channel` | `<n> [n2] ...` | Indices of channels to include, starting from 0. | v1.0.0 | v1.0.0
`--series` | `<n>` | Index of image series such as image tile, starting from 0. | v1.0.0 | v1.0.0
`--subimg_offset` | `<x,y,z> [x2,y2,z2] ...` | Sub-image offset coordinates given as `x, y, z`. Sub-images are 3D or 3D+channel subsets of full images that can be saved as separate files. Multiple sets of coordinates can be given to load different sub-images. | v1.2.0 | v1.2.0
`--subimg_size` | `<x,y,z> [x2,y2,z2] ...` | Sub-image size given as `x, y, z`. | v1.2.0 | v1.2.0
`--offset` | `<x,y,z> [x2,y2,z2] ...` | ROI (region of interest) offset coordinates given as `x, y, z`. ROIs are regions zoomed into within larger images. Multiple sets of coordinates can be given to set different ROIs, such as an ROI per loaded image. | v1.0.0 | v1.0.0
`--size` | `<x,y,z> [x2,y2,z2] ...` | ROI size given as `x, y, z`. | v1.0.0 | v1.0.0
`--db` | `<path>` | The main to the main SQlite database. | v1.0.0 | v1.0.0
`--cpus` | `<n>` | The number of CPUs to use during multiprocessing tasks. `none` (default) can be given to use the max number of processors. | v1.3.6 | [v1.3.6](#changes-in-magellanmapper-v13)
`--cpus` | `<n>` | The number of CPUs to use during multiprocessing tasks. `none` (default) can be given to use the max number of processors. | v1.3.6 | v1.3.6
`--load` | `blobs=<path> blob_matches=<path>` | Paths to load data. `blobs` are objects detected as blobs. `blob_matches` are co-localized blobs among channels. | v1.4.0 | [v1.4.0](#changes-in-magellanmapper-v14)
`--proc` | `<task1>=[sub-task1,...] <task2>` | Image processing tasks; see `config.ProcessTypes` | v1.0.0 | [v1.5.0](#changes-in-magellanmapper-v15)
`--register` | `<task>` | Registration and other atlas-related tasks; see `config.RegisterTypes` | v1.0.0 | [v1.4.0](#changes-in-magellanmapper-v14)
`--df` | `<task>` | Pandas data frame tasks; see `config.DFTasks` | v1.0.0 | [v1.3.0](#changes-in-magellanmapper-v13)
`--plot_2d` | `<task>` | 2D plot tasks; see `config.Plot2DTypes` | v1.0.0 | v1.0.0
`--ec2_start` | `<tag_name> <ami_id> <instance_type> <subnet_id> <sec_group> <key_name> <ebs> <max_count> <snapshot_ids>` | Start an EC2 instance. | v1.0.0 | v1.0.0
`--ec2_list` | `<state> <image_id>` | List EC2 instances. | v1.0.0 | v1.0.0
`--ec2_terminate` | `<instance1> [instance2] ...` | Terminate EC2 instances by instance ID. | v1.0.0 | v1.0.0
`--notify` | `<URL> <message> <attachment>` | Post a message to a URL. ``attachment`` can be a longer message string. | v1.0.0 | v1.0.0
`--grid_search` | `<profile>` | Perform a Grid Search hyperparameter tuning task for blob detection with the given profile name or file. | v1.0.0 | [v1.3.0](#changes-in-magellanmapper-v13)
`--theme` | `<theme1> [theme2] ...` | User interface themes; see `config.Themes`. | v1.1.4 | v1.1.4
`--truth_db` | `[mode=<mode>] [path=<path>]` | Truth database; see `config.TruthDBModes` for available modes. | v1.0.0 | v1.0.0
`--labels` | `[path_ref="""] [level=0] [ID=0] [orig_colors=1] [symmetric_colors=1] [binary=<backround>,<foreground>] [translate_labels=<path>"] [translate_children=0]` | Atlas label settings; see `config.AtlasLabels`. <ul><li>`path_ref`: Path to the labels reference file. Should have at least these columns: `Region` or `id` for region IDs, and `RegionName` or `name` for corresponding names. Atlases generated in >=v1.5.0 with this flag will copy this file into the atlas directory so that this argument is not needed when loading the atlas and images registered to it.</li> <li>`level`: Ontology level. Structures in sub-levels will be grouped at this level for volume stats. | v1.0.0 | v1.0.0
`--transform` | `[rotate=0] [flip_vert=0] [flip_horiz=0] [flip=axis] [rescale=0] [interpolation=1]` | Image transformations; see `config.Transforms`. <ul><li>`interpolation`: Interpolation order `n` for the main (intensity) image, where `n` is passed to the `order` argument in Scikit-image's [`transform.resize`](https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize) function</li> <li>*Since [v1.6.0](#changes-in-magellanmapper-v16):* `flip`: Flip the selected axis, where `axis` = 0 for the z-axis, 1 for the y-axis, and 2 for the x-axis</li></ul> | v1.0.0 | v1.6.0
`--reg_suffixes` | `[atlas=atlasVolume.mhd,...] [annotation=annotation.mhd,...] [borders=<path>]` | Suffixes of registered images to load; see `config.RegSuffixes` for suffixes used throughout the package. Atlases and regenerated and registered with output paths based on the path given by `--img` or `--prefix`. For example:<ul><li>"Base" path (from `--img` or `--prefix`): `/home/me/my_image`</li><li>Intensity/histology/"atlas" image: `/home/me/my_image_atlasVolume.nii.gz`</li><li>Annotation/labels image: `/home/me/my_image_annotation.nii.gz`</li><li>Other registered images, eg atlas edges: `/home/me/my_image_atlasEdge.nii.gz`</li></ul>To load intensity and annotations image: `./run.py --img /home/me/my_image --reg_suffixes atlasVolume.nii.gz annotation.nii.gz`. <ul><li>*Since [v1.5.0](#changes-in-magellanmapper-v15):* Suffixes can be also given as an absolute path, such as a labels image not registered to the main image.</li> <li>*Since [v1.6.0](#changes-in-magellanmapper-v16):* Multiple labels suffixes can be given, separated by commas.</li></ul> | v1.0.0 | [v1.5.0](#changes-in-magellanmapper-v15)
`--plot_labels` | `[title=<title>] [err_col_abs=<col>] [background=<color>] [vspan_col=<col>] [vspan_format=<str>] [rotation=<deg>] ...` | Plot labels; see `config.PlotLabels` for available parameters.<ul><li>*Since [v1.6.0](#changes-in-magellanmapper-v16):* Added several new sub-arguments.</li></ul> | v1.0.0 | [v1.6.0](#changes-in-magellanmapper-v16)
`--set_meta` | `[resolutions=<x,y,z>] [magnification=<n>] [zoom=<n>] [shape=<c,x,y,z,...>] [dtype=<data-type>]` | Metadata to set when importing an image; see `config.MetaKeys`. | v1.0.0 | [v1.3.0](#changes-in-magellanmapper-v13)
`--plane` | `<xy\|xz\|yz>` | Transpose to the given planar orientation | v1.0.0 | v1.0.0
`--show` | `<0\|1>` | Show images after generating them in atlas pipelines. `0` to not show, `1` to show. | v1.0.0 | [v1.3.0](#changes-in-magellanmapper-v13)
`--alphas` | `<n,...>` | Alpha (opacity) levels for the main image; can give as a comma-delimited for multiple channels (eg `0.5,0.3`) | v1.0.0 | v1.0.0
`--vmin` | `<n,...>` | Minimum intensity levels; can give as a comma-delimited for multiple channels (eg `0.1,0.2`) | v1.0.0 | v1.0.0
`--vmax` | `<n,...>` | Maximum intensity levels; can give as a comma-delimited for multiple channels (eg `0.1,0.2`) | v1.0.0 | v1.0.0
`--seed` | `<n>` | Random number generator seed. | v1.0.0 | v1.0.0
`--save_subimg` | None | Save a sub-image (given by `--subimg_offset` and `--subimg_size`) as a separate `<base>_subimg.npy` file | v1.0.0 | [v1.3.0](#changes-in-magellanmapper-v13)
`--slice` | `<start[,stop,step]>` | Slice range similar to the Python `slice` function. | v1.0.0 | v1.0.0
`--delay` | `<n>` | Time delay in ms, used for animations. | v1.0.0 | v1.0.0
`--savefig` | `<ext>` | Extension to use when saving figures, without period. When unset, defaults to PNG when exporting images. | v1.0.0 | [v1.3.0](#changes-in-magellanmapper-v13)
`--groups` | `[group1] [group2] ...` | Group ID corresponding to each image in `--img`. | v1.0.0 | v1.0.0
`--verbose`, `-v` | `[level=<n>] [path=<path>]` | Verbose output. `level` can range from `1` (`DEBUG`) to 5 (`CRITICAL`); defaults to debug if `level` is not set, or info if `-v` is not given. `path` is the output log path. | v1.0.0 | [v1.4.0](#changes-in-magellanmapper-v14)

## Changes in MagellanMapper v1.6

Old | New | Version | Purpose of Change |
--- | --- | --- | ---
None | `--rgb` | v1.6a1 | Open images in RGB(a) mode.
`--plot_labels ...` | `--plot_labels err_col_abs=<col> ...` | v1.6a1 | Plot error bars with a column of absolute rather than relative values, now that Clrstats gives absolute values for effect sizes.
` ` |  `--plot_labels background=<color>` | v1.6a1 | Change plot background color with a Matplotlib color string.
` ` |  `--plot_labels vspan_col=<col> vspan_format=<str>` | v1.6a1 | Column denoting vertical span groups and string format for them, respectively.
` ` |  `--plot_labels rotation=<deg>` | v1.6a2 | Change rotation in degrees.
`--ref_suffixes annotation=` | `--transform [flip=axis] ...` | v1.6a1 | Flip the specified axis.
`--transform ...` | `--transform [interpolation=n] ...` | v1.6a1 | Interpolation order can be specified when exporting the main image.
`--transform ...` | `--transform [flip=axis] ...` | v1.6a1 | Flip the specified axis.

## Changes in MagellanMapper v1.5

Old | New | Version | Purpose of Change |
--- | --- | --- | ---
`--prefix <path>` | `--prefix <path1> [path2] ...` | v1.5.0 | Multiple prefixes can be given
`--proc <task>` | `--proc <task1>=[sub-task1,...] [task2] ...` | v1.5.0 | Multiple processing tasks can be given as well as sub-tasks
None | `--proc export_tif` | v1.5.0 | Export NPY files to TIF format.
Specified in ROI profiles | `--proc preprocess=[rotate,saturate,...]` | v1.5.0 | Pre-processing tasks are integrated as sub-processing tasks; see `config.PreProcessKeys` for task names
`--reg_suffixes [atlas=<suffix1>] ... [fixed_mask=<suffix2>] [moving_mask=<suffix3>]` | Unchanged | v1.5.0 | Suffixes can now be given as an absolute path to load directly from the path, eg a labels image not registered to the main image. Image masks for registration can also be given as `fixed_mask` and `moving_mask`.
`--suffix <path>` | `--suffix <path1> [path2] ...` | v1.5.0 | Multiple suffixes can be given

## Changes in MagellanMapper v1.4

Old | New | Version | Purpose of Change |
--- | --- | --- | ---
None | `--atlas_labels translate_labels=<translation.csv> translate_children=1` | v1.4.0 | Translate labels, where `translation.csv` is a CSV file with `FromLabel` and `ToLabel` columns, and `translate_children` is `1` to also translate all children of each label
None | `--df sum_cols`, `subtract_cols`, `multiply_cols`, `divide_cols` | v1.4.0 | Arithmetic operations on data frame columns
None | `--df replace_vals` | v1.4.0 | Replace values
`--img <orig-name>.` | Can be given as `--img <orig-name>_image5d.npy`, or `python run.py <path1> [path2] ...` | v1.4.0 | The full imported filename can be given, and the `--img` can be omitted when the paths are given as first arguments
None | `--plot_labels text_pos=<x,y>` | v1.4.0 | Add plane number annotation at the `x,y` position in stack animations
None | `--plot_labels x_scale`, `y_scale` | v1.4.0 | Set plot axis scaling, such as `log` for log-scaling
`--proc detect --roi_profile <profile0> <profile1> <profile2> --channel 1 2` | `--proc detect --roi_profile "" <profile1> <profile2> --channel 1 2` | v1.4.0 | Block processing is performed based on each profile, rather than only on `profile0`; in this case, `profile0` is ignored since channel 0 is skipped and can be given as empty quotes
None | `--proc detec_coloc` | v1.4.0 | Intensity-based co-localizations during blob detection
None | `--proc coloc_match` | v1.4.0 | Match-based co-localizations after blob detection
None | `--proc export_planes` | v1.4.0 | Export all channels together into each plane
`--proc export_planes` | `--proc export_planes_channels` | v1.4.0 | Export each channel into a separate plane
`--proc load` | `--load [data0], [data1]=<path>, ...` | v1.4.0 | Load different data types, such as `blobs`, `blob_matches`, including custom path
`--register export_regions --labels level=None` | `--register export_regions --labels level=None orig_colors=1` | v1.4.0 | Use `orig_colors=1` rather than `level` to export only labels present in the image itself; `level=None` gets the parent rather than labels only up to that level
None | `--register labels_dist` | v1.4.0 | Measure the distance from labels to specified landmarks
`--verbose` | `--verbose level=<n> log_path=<path>` | v1.4.0 | Log level from 1 (`DEBUG`, most verbose) to 5 (`CRITICAL`, least verbose) and log output path

## Changes in MagellanMapper v1.3

Old | New | Version | Purpose of Change |
--- | --- | --- | ---
`--border` | `--plot_labels padding=x,y,z` | v1.3.0 | Duplicated by the `padding` argument
`--channel c1` | `--channel c1 [c2...]` | v1.3.1 | Accepts multiple channels
`--chunk_size` | None | v1.3.0 | Obsolete by block processing improvements
None | `--cpus <n>` | v1.3.6 | Specify the maximum number of CPUs to use for multiprocessing tasks
`finer` atlas profile | None | v1.3.0 | Its settings are now default
`-h` | Same | v1.3.7 | Sub-arguments are now included along with main arguments in help
`--mag` | `--set_meta magnification=x.y` | v1.3.0 | Grouped custom metadata settings into `--set_meta`
`--microscope <name1>[_name2]` | `--roi_profile <name1>[,name2]` | v1.3.0 | Specifies profiles to process by regions of interest; delimit by `,` to allow underscores especially in file paths
`--no_scale_bar` | `--plot_labels scale_bar=1` | v1.3.0 | Grouped with other plot labels controls
`--no_show` | `--show 0` | v1.3.0 | Show with `1`
`--padding_2d` | `--plot_labels margin=x,y,z` | v1.3.0 | Grouped with other plot labels controls, adding `margin` as space outside the ROI
None | `--proc export_raw` | v1.3.2 | Export images to RAW format
None | `--proc preprocess` | v1.3.0 | Pre-processing tasks using `profiles.PreProcessKeys` specified in ROI profiles
None | `--plot_labels dpi=<n>` | v1.3.7 | Configure DPI of saved images
None | `--plot_labels drop_dups=<0|1>` | v1.3.3 | Option to drop duplicates when joining data frames
`--atlas_labels binary=<color>` | `--plot_labels nan_color=<color>` | v1.3.7 | Group with `--plot_labels` to specify colors for NaN values
None | `--img <dir> --reg_suffixes [atlas-img] [annotation-img]` | v1.3.5 | Load only registered images by specifying a directory rather than an image file path as the main image
`--reg_profile <name1>[_name2]` | `--atlas_profile <name1>[,name2]` | v1.3.0 | Specifies profiles for atlases; delimit by `,` to allow underscores especially in file paths
`--rescale` | `--transform rescale=x` | v1.3.0 | Grouped with other transformation tasks
`--res` | `--set_meta resolutions=x,y,z` | v1.3.0 | Grouped custom metadata settings into `--set_meta`
`--roc` | `--grid_search <name1>[,name2]` | v1.3.0 | Its main task is to perform Grid Search based hyperparameter tuning; specify profile names or YAML files
`python -m magmap.xx.yy` | `run_cli.py` | v1.3.0 | All command-line based entry points can be accessed through the CLI using this script
`run_cli.py` | `run.py` | v1.3.2 | `run_cli.py` functionality was integrated into the `run.py` script and removed
`--savefig <ext>` | Optional | v1.3.7 | Defaults to PNG when exporting images even if `--savefig` is not set
`--saveroi` | `--save_subimg` | v1.3.0 | Consistency with "sub-images" as parts of images that can contain ROIs
`--stats` | `--df` | v1.3.0 | Run data-frame (eg CSV file) tasks
`--zoom` | `--set_meta zoom=x,y,z` | v1.3.0 | Grouped custom metadata settings into `--set_meta`
