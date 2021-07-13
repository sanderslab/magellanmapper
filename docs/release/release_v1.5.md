# MagellanMapper v1.5 Release Notes

## MagellanMapper v1.5.0

### Highlights

### Changes

#### Installation

#### GUI

- Window size and position are saved as user prferences
- The default window size is smaller to fit into 720p displays
- The default focus is no longer the main image file path to avoid accidental file loading

#### CLI

See the [table of CLI changes](../cli.md#changes-in-magellanmapper-v15) for a summary of all changes in v1.5

- Multiple processing tasks can be given in the same command; eg `--proc detect coloc_match`
- Image preprocessing tasks have been integrated into `--proc`, no longer requiring a separate ROI profile; eg `--proc preprocess=rotate`

#### Atlas refinement

- Metadata for labels images are saved when importing an atlas and registering the atlas to another image so that the original atlas no longer needs to be available (and `--labels` argument does not to be given) when loading the atlas or registered image
- Smoothing metrics are output during the `--register merge_atlas_segs` task
- The atlas profile settings `meas_edge_dists` and `meas_smoothing` turn off these metrics to save time during atlas generation, and the profile `fewerstats` turns off both these settings

#### Atlas registration

- Image masks can be set to focus the field for image registration; use the new `--reg_suffixes fixed_mask=<suffix-or-abs-path> moving_mask=<suffix-path>` command-line sub-arguments to load these mask files

#### Cell detection

- Previously saved blobs are no longer loaded prior to re-detection

#### Volumetric image processing

#### I/O

- Single plane RAW images can be loaded when importing files from a directory, in addition to multiplane RAW files
- Fixed enabling the import directory without sufficient metadata
- Fixed to update metadata files when loaded through the `--meta` flag
- Fixed error when unable to load a profile `.yml` file

#### Server pipelines

#### Python stats and plots

#### R stats and plots

#### Code base and docs

### Dependency Updates

#### Python Dependency Changes

#### R Dependency Changes

#### Server dependency Changes
