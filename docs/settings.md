# Settings in MagellanMapper

MagellanMapper supports settings configuration at several different levels of "permanence." Starting from the most transient to more enduring settings, these settings are:

1. Graphical user interface (GUI) controls, configurable during runtime
1. Command-line arguments, set at launch time
1. Profiles, groups of settings saved in a file and set at launch time or loaded through the GUI

## GUI Parameters

The ROI selector provides many options for displaying 2D and 3D regions of interest. Users can select these parameters through checkboxes, sliders, text input, or other controls. Currently these settings are reset with each program launch.

## Command-Line Arguments

The command-line interface (CLI) provides many options to configure how MagellanMapper displays and processes images. While these settings are transient, one may apply them repeatedly by saving the command in a script such as a Bash or Batch file.

To see the full list of commands available, use this command:

```
./run.py --help
```

Commands-line arguments are typically given either as an argument alone or with associated values:

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

Currently two types of profiles exist in MagellanMapper: 1) region of interest (ROI) profiles, used for processing higher resolution images, and 2) atlas profiles, used for atlas registration and processing. Each type of profile is defined by a custon Python dictionary and includes a set of built-in profiles. You can create custom profiles in YAML format.

### Specifying Profiles

#### By Command-Line

To apply a profile, use the appropriate command-line argument followed by the profile name. To combine profiles, string these names together, separating each name with a comma (`,`):

```
./run.py --roi_profile lightsheet # apply the lightsheet profile
./run.py --roi_profile lightsheet,4xnuc # modify the lightsheet profile with the 4xnuc profile
./run.py --roi_profile lightsheet lightsheet,cytoplasm # ch0: lightsheet, ch1: lightsheet modified by cytoplasm profile
./run.py --atlas_profile abae11pt5 # ABA E11.5 atlas profile
```

ROI profiles can be distinct for each channel. If only one profile is given for a multi-channel image, that profile will be used for all channels. For example, given the profile and channel setup for whole image blob detection:

```
./run.py --img <myimage> --proc detec --roi_prof profile0 profile1 profile2 profile3 --channel 1 2 
```

- `profile0`:
  -  Typically used for channel 0 detections, but skipped here because the channel arguments skips channel 0
  -  Used as the "default" profile to set up block processing parameters (eg `segment_size`, `denoise_size` profile settings) and any channels where a profile is not specified
  - **Changed in v1.4**: Since this channel is skipped, the profile is ignored as well for block processing. Each detection channel's profile is used separately to set up block processing specific to that channel. If all channels have the same block processing settings, blocks will be shared across channels to reduce processing time.
- `profile1` is used for channel 1 detections (even though channel 0 is skipped)
- `profile2` is used for channel 2
- `profile3` is skipped, since no detections are done for channel 3

#### In the GUI

Profiles can also be loaded in the "Profiles" tab in the graphical interface. Built-in profiles and files in the `profiles` folder can be selected from the dropdown box and applied to the specified channels.

If you have added new files to the `profiles` folder (see [below](#custom-profiles), press the "Load Profiles" button to repopulate the dropdown with your files.

### Custom Profiles

Users may wish to design their own custom profiles. Profiles can be saved in YAML format using the same setting names and loaded by their paths. Pressing "Refresh" in the ROI tab or "Detect" in the Detection tab will reload custom profile files on-the-fly if they have changed. This way, you can test new detection settings each time you press "Detect."

Example profile file:

```yaml
---
clip_vmin: 0
clip_vmax: 100
clip_max: 0.4
...
```

Example settings that use Enums will be translated automatically:

```yaml
---
labels_mirror:
  RegKeys.ACTIVE: True
...
```

To load the files, include their paths as if they were names:

```
./run.py --roi_profile mypath.yaml # only load custom profile
./run.py --roi_profile lightsheet,mypath.yaml # apply on top of lightsheet
./run.py --roi_profile mypath.yaml,lightsheet # apply before lightsheet
```

Note that because of the separator used between profiles, profile filenames cannot have commas. Paths may include separate folders, however:

```
./run.py --roi_profile lightsheet,profiles/mypath.yaml
```

As a simplification, profiles stored in the `profiles` folder can be specified without the folder name. For example, the above command can be shorted to:

```
./run.py --roi_profile lightsheet,mypath.yaml
```
