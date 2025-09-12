# MagellanMapper v1.7 Release Notes

## MagellanMapper v1.7.0

### Highlights

- Javabridge/Bioformats has been replaced with multipage TIF file support
- Minimum supported Python version is now 3.10 to reduce maintenance burden and improve testing on more recent versions of Python

### Changes

#### Installation

- `project.toml` installation configuration replaces `setup.py` (#741)

#### GUI

- Fixed display of label RGB values and coordinates in figure filenames (#753)

#### CLI

#### Atlas refinement

#### Atlas registration

#### Cell detection

#### Block processing

#### I/O

- TIF files can be imported without Javabridge/Bioformats in the GUI ("Import" tab) or by loading a TIF image directly with the flag, `--savefig npy` (#738, #753, #756)
- Read and write `PhysicalSpacingX`-style TIF resolutions (#753)
- Exports to TIF are now multichannel TIF files (#756)
- Fixed issues with loading certain TIF files' metadata (#738, #754)
- Fixed saving/loading rescaled images using Numpy 2 (#738)

#### Server pipelines

#### Python stats and plots

- Make normalizing data frames more flexible (#746)
- Fixed to apply plot setting across shared axes (#740)

#### R stats and plots

#### Code base and docs

- Better encapsulated the main image to pave the way for loading multiple images in the same session (#754)
- Integration tests for loading images and large stack blob detection (#754)

### Dependency Updates

#### Python Dependency Changes

- Bumped minimum supported Python version to 3.10 (#712)
- `import` group packages (Javabridge/Bioformats) are deprecated and no longer included in the `most` group install (#745)

#### R Dependency Changes

#### Server dependency Changes
