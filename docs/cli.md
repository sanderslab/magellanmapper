# Command-Line Interface for MagellanMapper

## Command-Line argument reference

Argument | Sub-argument | Function | Ver Added | Last updated
--- | --- | --- | --- | ---
`--img` | `<path> [path2] ...` | Image paths. `--img` can be omitted if paths are given as the first argument. Can be given as the original filename (eg `myvolume.tiff`), the imported name, (`myvolume_image5d.npy`), a sub-image name (`myvolume_(x,y,z)x(x,y,z)_subimg.npy`), or a registered image (`myvolume_atlasVolume.mhd`). | v1.0.0 | [v1.4.0](#changes-in-magellanmapper-v14)
`--meta` | `<metadata-path> [path2] ...` | Metadata paths corresponding to images from `--img`. | v1.0.0 | v1.0.0
`--prefix` | `<path>` | Path prefix, typically used for the start of output paths. | v1.0.0 | v1.0.0
`--suffix` | `<path>` | Path suffix, typically used for the end of output paths. | v1.0.0 | v1.0.0
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
TODO: finish arguments

## Changes in MagellanMapper v1.5

Old | New | Version | Purpose of Change |
--- | --- | --- | ---
`--proc <task>` | `--proc <task1>=[sub-task1,...] <task2>` | v1.5.0 | Multiple processing tasks can be given as well as sub-tasks
Specified in ROI profiles | `--proc preprocess=[rotate,saturate,...]` | v1.5.0 | Pre-processing tasks are integrated as sub-processing tasks; see `config.PreProcessKeys` for task names

## Changes in MagellanMapper v1.4

Old | New | Version | Purpose of Change |
--- | --- | --- | ---
None | `--atlas_labels translate_labels=<translation.csv> translate_children=1` | v1.4.0 | Translate labels, where `translation.csv` is a CSV file with `FromLabel` and `ToLabel` columns, and `translate_children` is `1` to also translate all children of each label
None | `--df sum_cols`, `subtract_cols`, `multiply_cols`, `divide_cols` | v1.4.0 | Arithmetic operations on data frame columns
None | `--df replace_vals` | v1.4.0 | Replace values
`--img <orig-name>.` | Can be given as `--img <orig-name>_image5d.npy`, or `python run.py <path1> [path2] ...` | v1.4.0 | The full imported filename can be given, and the `--img` can be omitted when the paths are given as first arguments
None | `--plot_labels text_pos=<x,y>` | v1.4.0 | Add plane number annotation at the `x,y` position in stack animations
None | `--plot_labels x_scale`, `y_scale` | v1.4.0 | Set plot axis scaling, such as `log` for log-scaling
`--proc detec --roi_prof <profile0> <profile1> <profile2> --channel 1 2` | `--proc detec --roi_prof "" <profile1> <profile2> --channel 1 2` | v1.4.0 | Block processing is performed based on each profile, rather than only on `profile0`; in this case, `profile0` is ignored since channel 0 is skipped and can be given as empty quotes
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
