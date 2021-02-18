# Command-Line Interface for MagellanMapper

## Command-Line argument reference

Argument | Function | Version Added
--- | --- | ---

## Changes in MagellanMapper v1.4

Old | New | Version | Purpose of Change |
--- | --- | --- | ---
`--proc detec --roi_prof <profile0> <profile1> <profile2> --channel 1 2` | `--proc detec --roi_prof "" <profile1> <profile2> --channel 1 2` | v1.4.0 | Block processing is performed based on each profile, rather than only on `profile0`; in this case, `profile0` is ignored since channel 0 is skipped and can be given as empty quotes
None | `--df sum_cols`, `subtract_cols`, `multiply_cols`, `divide_cols` | v1.4.0 | Arithmetic operations on data frame columns
None | `--df replace_vals` | v1.4.0 | Replace values
None | `--plot_labels x_scale`, `y_scale` | v1.4.0 | Set plot axis scaling, such as `log` for log-scaling

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
