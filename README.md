# MagellanMapper

MagellanMapper is a graphical imaging informatics suite and pipeline for high-throughput, automated analysis of whole specimen. Its design philosophy is to make the raw 3D images as accessible as possible, simplify annotation from nuclei to atlases, and scale from the laptop to the cloud in cross-platform environments.

![ROI Editor and Atlas Editor screenshots](https://user-images.githubusercontent.com/1258953/83934132-f699aa00-a7e0-11ea-932c-0e58366d5061.png)

## Installation

1. Download and extract the MagellanMapper
  * Get the [latest release](https://github.com/sanderslab/magellanmapper/releases/latest)
  * Or clone this repo: `git clone https://github.com/sanderslab/magellanmapper.git`
1. Navigate to the MagellanMapper directory: `cd <path-to-magellanmapper>`
1. Install MagellanMapper
  * On Mac or Linux: `bin/setup_conda.sh`
  * On Windows: `bin\setup_conda.bat`
  
See [Installation](docs/install.md) for more details, including installing without Conda such as by Pip or Venv+Pip.
  
## Run MagellanMapper

MagellanMapper can be run as a GUI or headlessly for desktop or server tasks, respectively. To start MagellanMapper, run (assuming a Conda environment named `mag`):

```
conda activate mag
./run.py --img <path-to-your-image>
```

Proprietary image formats such as `.czi` will be imported automatically via Bioformats into a Numpy array format before loading it in the GUI. This format allows on-the-fly loading to reduce memory requirements and initial loading time. Medical imaging formats such as `.mha` (or `.mhd/.raw`) and `.nii` (or `.nii.gz`) are opened with SimpleITK/SimpleElastix and do not require separate import.

You can also use [`pipelines.sh`](bin/pipelines.sh), a script to run many automated pipelines within MagellanMapper, such as whole volume nuclei detection and image transposition. See below for more details. [`sample_cmds.sh`](bin/sample_cmds.sh) is a script that shows examples of common commands. It can also be modified and called directly.

See [Settings](docs/settings.md) for how to customize settings to your image files.

### Sample 3D data

To try out functions with some sample images, download any of these files:

- [Sample region of nuclei at 4x](https://github.com/sanderslab/magellanmapper/releases/download/v1.1.3/sample_region.zip)
- [Sample downsampled tissue cleared whole brain](https://github.com/sanderslab/magellanmapper/releases/download/v1.1.3/sample_brain.zip)
- [Allen Developing Mouse Brain Atlas E18.5](https://github.com/sanderslab/magellanmapper/releases/download/v1.1.3/ADMBA-E18pt5.zip)

## 3D viewer

The main MagellanMapper GUI displays a 3D viewer and region of interest (ROI) selection controls. MagellanMapper uses Mayavi for 3D voxel or surface rendering.

From the ROI selection controls, two different 2D editors can be opened. All but the last `2D styles` option open various forms of the Nuclei Annotation Editor. The final option opens the Atlas Editor, a 2D/3D viewer.

## Nuclei Annotation Editor

The multi-level 2D plotter is geared toward simplifying annotation for nuclei. Press on `Detect` to detect nuclei in the current ROI, then `Plot 2D` to open the figure.

- Click on dotted lines to cycle the nuclei detection flags from incorrect (red), correct (green), or questionable (yellow)
- `Shift+click` and drag to move the circle's position
- `Alt+click` and drag to resize the circle's radius
- `"c"+click` to copy the circle
- `"v"+click` in another z-plane to duplicate that circle in the corresponding position in that plane
- `"x"+click` to cut the circle
- `"d"+click` to delete the circle
- Arrow `up/down` to change the overview plots' z-plane
- `Right-arrow+click` (or just `right-arrow` in some cases) on an ROI plot to jump to the corresponding z-plane in the overview plots

## Atlas Editor

The multi-planar image plotter allows simplified viewing and editing of annotation labels for an atlas. Existing labels can be painted into adjacent areas, and synchronized planar viewing allows visualization of changes in each plane with realtime updates.

The atlas image must have an associated annotation image. Use the `--labels` flage to specify a labels `.json` file. Change the `2D plot styles` dropdown to `Atlas editor` and press `Plot 2D` to open the editor.

- Mouseover over any label to see the region name
- `Left-click` to move the crosshairs and the corresponding planes
- Scroll or arrow `up`/`down` to move planes in the current plot
- `Right-click` or `Ctrl+left-click` + mouse-up/down to zoom
- `Middle-click` or `Shift+left-click` + mouse drag to pan
- `a` to toggle between 0 and full labels alpha (opacity)
- `shift+a` to halve alpha (press `a` twice to return to original alpha)

Press on the "Edit" button to start painting labels:

- `Left-click` to pick a color, then drag to paint over a new area
- `Alt+Left-click` to use the last picked color instead
- `[`/`]` (brackets) to make the paintbrush smaller/bigger; add `shift` to halve the increment
- Use the save button in the main window with the atlas window still open to resave


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

Author: David Young, 2017, 2020, Stephan Sanders Lab
