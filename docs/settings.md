# Settings in MagellanMapper

MagellanMapper supports settings configuration at several different levels. Starting from the most transient to more permanent levels, these settings are:

1. Settings within the graphical user interface (GUI)
1. Command-line arguments
1. Profiles

## GUI Parameters

The ROI selector provides many options for displaying 2D and 3D regions of interest. Users can select these parameters through checkboxes, sliders, text input, or other controls. Currently these settings are reset with each program launch.

## Command-Line Arguments

The command-line interface (CLI) provides many options to configure how MagellanMapper displays and processes images. While these settings are transient, one may apply them repeatedly by saving the command in a script such as a Bash or Batch file.

To see the full list of commands available, use this command:

```
./run.py --help
```

Commands-line arguments are typically given in the usual way, either as an argument alone or with a values associated to that argument:

```
./run.py --verbose # display verbose output for debugging
./run.py --img my_image.tif # load an image file
./run.py --img my_image.tif another_img.tif # load 2 image files
```

Some arguments further subdivide values into separate sub-arguments. Each sub-argument is designated with an equals (`=`) sign or based on the order of sub-arguments, similar to a Python function call:

```
# --reg_suffixes is in the order: exp (intensity), annotation (labels), border files
./run.py --reg_suffixes exp.mhd # load experiment as intensity file
./run.py --reg_suffixes exp.mhd annotation.mhd # load exp and label files
./run.py --reg_suffixes annotation=annotationEdge.mhd # replace label with edge file
```

## Profiles

Profile are collections are settings that are typically applied repeatedly. For example, the settings for a given atlas are grouped into a single profile to apply those settings each time the atlas is used. Profiles may be combined to cover multiple scenarios at once, such as the atlas profile along with a profile adjusting the output format for any given atlas.

Currently two types of profiles exist in MagellanMapper: 1) microscope profiles, typically used for processing of microscopy images, and 2) register profiles, typically for atlas registration and processing. Each type of profile has its own Python class which defines its settings as a dictionary and set of built-in profiles.

To apply a profile, use the appropriate command-line argument followed by the profile name. To combine profiles, string these names together, separating each name with an underscore (`_`):

```
./run.py --microscope lightsheet # apply the lightsheet profile
./run.py --microscope lightsheet_4xnuc # modify the lightsheet profile with the 4xnuc profile
./run.py --microscope lightsheet lightsheet_cytoplasm # ch0: lightsheet, ch1: lightsheet modified by cytoplasm profile
./run.py --reg_profile abae11pt5 # ABA E11.5 atlas profile
```

Microscope profiles can be distinct for each channel. If only one profile is given for a multi-channel, that profile will be used for all channels.

Users may wish to design their own custom profiles. Profiles can be saved in YAML format using the same setting names and loaded by their paths:

```yaml
---
clip_vmin: 0
clip_vmax: 100
clip_max: 0.4
...
```

Settings that use Enums will be translated automatically:

```yaml
---
labels_mirror:
  RegKeys.ACTIVE: True
...
```

To load the files, include their paths as if they were names:

```
./run.py --microscope mypath.yaml # only load custom profile
./run.py --microscope lightsheet_mypath.yaml # apply on top of lightsheet
./run.py --microscope mypath.yaml_lightsheet # apply before lightsheet
```

Note that because of the separator used between profiles, profile filenames cannot have underscores. Paths may include separate folders, however:

```
./run.py --microscope lightsheet_profiles/mypath.yaml
```
