# MagellanMapper v1.5 Release Notes

## MagellanMapper v1.5.0

### Highlights

### Changes

#### Installation

#### GUI

#### CLI

See the [table of CLI changes](../cli.md#changes-in-magellanmapper-v15) for a summary of all changes in v1.5

- Multiple processing tasks can be given in the same command; eg `--proc detect coloc_match`
- Image preprocessing tasks have been integrated into `--proc`, no longer requiring a separate ROI profile; eg `--proc preprocess=rotate`

#### Atlas refinement

- The atlas profile `meas_edge_dists` entry can be used to turn off edge distance measuring

#### Atlas registration

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
