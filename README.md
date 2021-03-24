# MagellanMapper

MagellanMapper is a graphical imaging informatics suite for 3D reconstruction and automated analysis of whole specimens and atlases. Its design philosophy is to make the raw 3D images as accessible as possible, simplify annotation from nuclei to atlases, and scale from the laptop or desktop to the cloud in cross-platform environments.

![ROI Editor and Atlas Editor screenshots](https://user-images.githubusercontent.com/1258953/83934132-f699aa00-a7e0-11ea-932c-0e58366d5061.png)

<img align="right" width="200" src="https://currentprotocols.onlinelibrary.wiley.com/cms/asset/14209f75-4368-40c9-b83b-1bb866991a8b/cpns76-gra-0001-m.jpg" title="Current Protocols cover image">

- For more information on the methods used for 3D atlas construction, please see: https://elifesciences.org/articles/61408
- For step-by-step instructions on using the software, please see: https://currentprotocols.onlinelibrary.wiley.com/doi/abs/10.1002/cpns.104


## Installation

1. Download MagellanMapper
    - Download and extract the [latest release.](https://github.com/sanderslab/magellanmapper/releases/latest)
    - Or clone the git repo: `git clone https://github.com/sanderslab/magellanmapper.git`
1. Install MagellanMapper using the following script in the `magellanmapper` folder.
    - On Mac or Linux: `bin/setup_conda`
    - On Windows: `bin\setup_conda.bat`

The script will also install the Conda package manager if it is not already installed. This process may take up to 5 minutes, depending on internet connection speed.

### Install Notes
- On Mac, it may be necessary to right-click and "Open with" the Terminal app.
- On Linux, it may be necessary to go to "Preferences" in the file browser (eg the Files app), select the "Behavior" tab, and choose to "Run" or "Ask" when executing text files.
- See [Installation](docs/install.md) for more details, including installation without Conda, using Pip or Venv+Pip instead.
  
## Run MagellanMapper

### From a file browser

**On Mac or Linux**: Double-click the `MagellanMapper icon created during Conda setup. This Unix executable should open with Terminal by default on Mac and after the file browser preference change described above on Linux.

**On Windows**: Run `run.py` through Python.
- It may be necessary to right-click, choose "Open with", and browse to the Conda `pythonw.exe` file to open `run.py`
- If a security warning displays, click on "More info" and "Run anyway" to launch the file

Note that during the first run, there may be a delay of up to several minutes from antivirus scanning for the new Python interpreter location in the new environment. Subsequent launches are typically much faster.

### From a terminal

```
conda activate mag
python <path-to-magellanmapper>/run.py
```

This approach is recommended when running command-line tasks or for debugging output. Replace `mag` if you gave the environment a different name.

MagellanMapper can be run as a GUI as described above or headlessly for automated tasks. [`sample_cmds.sh`](bin/sample_cmds.sh) is a script that shows examples of common commands. You can also use [`pipelines.sh`](bin/pipelines.sh), a script to run many automated pipelines within MagellanMapper, such as whole volume nuclei detection and image transposition. See [Settings](docs/settings.md) for how to customize parameters for your image analysis.

### Image file import
In the "Import" tab, you can select files, view and update metadata, and import the files.

Medical imaging formats such as `.mha` (or `.mhd/.raw`) and `.nii` (or `.nii.gz`) can be opened with the SimpleITK/SimpleElastix Library and do not require separate import. Standard image formats such as TIFF or proprietary microscopy formats such as CZI can be imported by MagellanMapper into an industry standard Numpy format, which allows on-the-fly loading to reduce memory requirements and initial loading time.



### Sample 3D data

To try out functions with sample images, download any of these practice files:

- [Sample region of nuclei at 4x (`sample_region.zip`)](https://github.com/sanderslab/magellanmapper/releases/download/v1.1.3/sample_region.zip)
- [Sample downsampled tissue cleared whole brain (`sample_brain.zip`)](https://github.com/sanderslab/magellanmapper/releases/download/v1.1.3/sample_brain.zip)
- [Allen Developing Mouse Brain Atlas E18.5 (`ADMBA-E18pt5.zip`)](https://github.com/sanderslab/magellanmapper/releases/download/v1.1.3/ADMBA-E18pt5.zip)

## 3D viewer

The main MagellanMapper GUI displays a 3D viewer and region of interest (ROI) selection controls. MagellanMapper uses the Mayavi data visualizer for 3D voxel or surface rendering.

From the ROI selection controls, two different 2D editors can be opened. All but the last `2D styles` option open various forms of the Nuclei Annotation Editor. The final option opens the Atlas Editor, a 2D/3D viewer.

## Nuclei Annotation Editor

The multi-level 2D plotter is geared toward simplifying annotation for nuclei. Select the `ROI Editor` tab to view the editor. Press the `Redraw` button to redraw the editor at the selected ROI. To detect and display nuclei in the ROI, select the `Detect` tab and press the `Detect` button.

| To Do...        | Shortcut            |
| ---------------- | :------------------: |
| Cycle between the 3 nuclei detection flags | Click within the dotted circles; incorrect (red), correct (green), or questionable (yellow) |
| Move the circle's position | `shift+click` and drag (note that the original position will remain as a solid circle) |
| Resize the circle's radius | `Alt+click` (`option+click` on Mac) and drag |
| Copy the circle | `"c"+click` |
| Duplicate a circle to the same position in anothe z-plane | `"v"+click` on the corresponding position in the z-plane to which the circle will be duplicated |
| Cut the circle | `"x"+click` |
| Delete the circle | `"d"+click` |
| Increase/decrease the overview plots' z-plane | Arrow `up/right` to increase and `down/left` to decrease |
| Jump to a z-plane in the overview plots corresponding to an ROI plane | `Right-click` on the the corresponding ROI plane |
| Preview the ROI at a certain position | `Left-click` in the overview plot |
| Redraw the editor at the chosen ROI settings | Double `right-click` in any overview plot |


## Atlas Editor

The multi-planar image plotter allows simplified viewing and editing of annotation labels for an atlas. Existing labels can be painted into adjacent areas, and synchronized planar viewing allows realtime visualization of changes in each plane.

To view the editor, select the `Atlas Editor` tab. The `Redraw` button in the `ROI` tab of the left panel will redraw the editor if necessary. The `Registered images` section allows selecting any available annotations and label reference files to overlay.

| To Do...       | Shortcut                    |
| ------------- |:-------------------------: |
| See the region name | Mouseover over any label |
| Move the crosshairs and the corresponding planes | `Left-click` |
| Move planes in the current plot | Scroll or arrow `up`/`down`|
| Zoom | `Right-click` or `Ctrl+left-click` while moving the mouse up/down |
| Pan | `Middle-click` or `Shift+left-click` while dragging the mouse |
| Toggle between 0 and full labels alpha (opacity) | `a` |
| Halve alpha | `shift+a` |
| Return to original alpha | press `a` twice |

Press on the "Edit" button to start painting labels using these controls:

| To Do...        | Shortcut                    |
| ------------- |:-------------------------:|
|Paint over a new area | `Left-click`, pick a color, then drag over image |
| Use the last picked color to paint over a new area | `Alt+Left-click`(option-click on Mac) |
| Make the paintbrush smaller/bigger | `[`/`]` (brackets) |
| Halve the increment of the paintbrush size | `[`/`]` and add `shift` |


Use the save button in the main window with the atlas window still open to resave


## Start a processing pipeline

Automated processing will attempt to scale based on your system resources but may require some manual intervention. This pipeline has been tested on a Macbook Pro laptop and AWS EC2 Linux (RHEL and Amazon Linux based) instances.

Optional dependencies:

- ImageJ/Fiji with the BigStitcher plugin: required for tile stitching; downloaded automatically onto a server when running `deploy.sh`
- ImageMagick: required for exporting a stack of planes to an animated GIF file
- FFMpeg: required to export a stack to a movie format such as MP4
- [Slack incoming webhook](https://api.slack.com/incoming-webhooks): to notify when tile stitching alignment is ready for verification and pipeline has completed

### Local
Run a pipeline in `pipelines.sh`.

For example, load a `.czi` file and display in the GUI, which will import the file into a Numpy format for faster future loading:

```
bin/pipelines.sh -i data/HugeImage.czi
```

To sitch a multi-tile image and perform cell detection on the entire image, which will load BigStitcher in ImageJ/Fiji for tile stitching:

```
bin/pipelines.sh -i data/HugeImage.czi -p full
```

See `bin/pipelines.sh` for additional sample commands for common scenarios, such as cell detection on a small region of interest. The file can be edited directly to load the same image, for example.

### Server

You can launch a standard server, deploy MagellanMapper code, and run a pipeline. See [tools for AWS cloud management](cloud_aws.sh) for more details. 

Licensed under the open-source [BSD-3 license](LICENSE.txt)

Author: David Young, 2017, 2021, Stephan Sanders Lab
