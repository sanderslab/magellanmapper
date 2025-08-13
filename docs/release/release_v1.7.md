# MagellanMapper v1.7 Release Notes

## MagellanMapper v1.7.0

### Highlights

- Better multipage TIF file support, including import to NPY format without Javabridge/Bioformats
- Minimum supported Python version is now 3.10 to reduce maintenance burden and improve testing on more recent versions of Python

### Changes

#### Installation

- `project.toml` installation configuration replaces `setup.py` (#741)

#### GUI

#### CLI

#### Atlas refinement

#### Atlas registration

#### Cell detection

#### Block processing

#### I/O

- TIF files can be imported without Javabridge/Bioformats by loading a TIF image directly with the flag, `--savefig npy` (#738)
- Fixed loading TIF files without resolution metadata (#738)
- Fixed saving/loading rescaled images using Numpy 2 (#738)

#### Server pipelines

#### Python stats and plots

- Fixed to apply plot setting across shared axes (#740)

#### R stats and plots

#### Code base and docs

### Dependency Updates

#### Python Dependency Changes

- Bumped minimum supported Python version to 3.10 (#712)
- `import` group packages (Javabridge/Bioformats) are deprecated and no longer included in the `most` group install (#745)

#### R Dependency Changes

#### Server dependency Changes
