# MagellanMapper v1.2 Release Notes

## MagellanMapper v1.2.1

### Changes

GUI
- Atlas Editor uses command-line specified alpha values

CLI
- Option to load microscope and atlas profiles through YAML files

Atlas refinement
- Option to turn off negative labeling used to distinguish hemispheres
- The origin will update after atlas cropping
- Cleaned up profile settings for label edge extension and mirroring

I/O
- Fixed image stack export to use the new sub-image settings
- Applied workaround for exporting image labels where the first labeled plane is empty

### Dependency Updates

#### Python Dependency Changes

- PyYaml (already included with other dependencies)

## MagellanMapper v1.2.0

### Changes

Installation
- These release notes are now included in the repository

GUI
- Setting transparency in overlaid images is more configurable

CLI
- Multiprocessing supported cross-platform by adding `spawn` support for Windows, which is also available and the new default in Mac; currently implemented for blob detection and image downsampling
- Fixed shebang for most Python files

I/O
- File naming simplifications: removed series string (eg `00000`), fixed some extensions (eg NPY instead of NPZ for single value archives), clearer suffixes (eg `blobs` instead of `info_proc`)
- Distinguished sub-images (specify by `--subimg_offset` and `--subimg_size`) vs ROIs (`--offset` and `--size`), which fixes many issues with saving/loading image and blob subsets

Server pipelines
- Tile stitching script uses `JAVA_HOME` for Java by default
- Fixed attempting to upload to S3 when directory is not set
