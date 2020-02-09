# MagellanMapper
MagellanMapper is an imaging informatics GUI and pipeline for high-throughput, automated analysis of whole organs. Its design philosophy is to make the raw, original images as accessible as possible; simplify annotation from nuclei to atlases; and scale from the laptop to the cloud in cross-platform environments.

Author: David Young, 2017, 2020, Stephan Sanders Lab

- Currently access is limited to a private Git repo. Our eventual plan is to make MagellanMapper available via Anaconda and Pip.
- Contact the project managers about loading your public key to access the project repository

## Installation

1. Download and unarchive MagellanMapper, or clone this repo (`git clone git@bitbucket.org:psychcore/magellanmapper.git`)
1. Install MagellanMapper in a new Conda environment:
  
  ```
  conda env create -n mag magellanmapper/environment.yml
  ```
  
  On Mac or Linux, you can alternatively use our convenient setup script, which will also install Minconda if necessary:
  
  ```
  magellanmapper/bin/setup_conda.sh
  ```
  
  See [Installation](docs/install.md) for more details, including installs without Conda such as Pip or Venv+Pip.
  
## Run MagellanMapper

MagellanMapper can be run as a GUI or headlessly for desktop or server tasks, respectively. To start MagellanMapper, run (assuming a Conda environment named `mag`):

```
source activate mag
./run --img [path_to_your_image]
```

Proprietary image formats such as `.czi` will be imported automatically via Bioformats into a Numpy array format before loading it in the GUI. This format allows on-the-fly loading to reduce memory requirements and initial loading time. Medical imaging formats such as `.mha` (or `.mhd/.raw`) and `.nii` (or `.nii.gz`) are opened with SimpleITK/SimpleElastix and do not require separate import.

You can also use `pipelines.sh`, a script to run many automated pipelines within MagellanMapper, such as whole volume nuclei detection and image transposition. See below for more details.

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
- `Right` arrow to jump the overview plots to the same z-plane as the current mouseover

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

#### Dependencies

- `awscli`: AWS Command Line Interface for basic up/downloading of images and processed files S3. Install via Pip.
- `boto3`: AWS Python client to manage EC2 instances.

#### Launch a server

You can launch a standard server, deploy MagellanMapper code, and run a pipeline. Note that typically login with graphical support (eg via `vncserver`) is required during installation for Mayavi and stitching in the standard setup, but you can alternatively run a lightweight install without GUI (see above).

If you already have an AMI with MagellanMapper installed, you can launch a new instance of it via MagellanMapper:

```
python -u -m magmap.io.aws --ec2_start "Name" "ami-xxxxxxxx" "m5.4xlarge" \
  "subnet-xxxxxxxx" "sg-xxxxxxxx" "UserName" 50,2000 [2]
```

- `Name` is your name of choice
- `ami` is your previously saved AMI with MagellanMapper
- `m5.4xlarge` is the instance type, which can be changed depending on your performance requirements
- `subnet` is your subnet group
- `sg` is your security group
- `UserName` is the user name whose security key will be uploaded for SSH access
- `50,2000` creates a 50GB swap and 2000GB data drive, which can be changed depending on your needs
- `2` starts two instances (optional, defaults to 1)

#### Setup server with MagellanMapper

Deploy the MagellanMapper folder and supporting files:

```
bin/deploy.sh -p [path_to_your_aws_pem] -i [server_ip] \
    -d [optional_file0] -d [optional_file1]
```

- This script by default will:
  - Archive the MagellanMapper Git directory and `scp` it to the server, using your `.pem` file to access it
  - Download and install ImageJ/Fiji onto the server
  - Update Fiji and install BigStitcher for image stitching
- To only update an existing MagellanMapper directory on the server, add `-u`
- To add multiple files or folders such as `.aws` credentials, use the `-d` option as many times as you'd like

Setup drives on a new server instance:

```
bin/setup_server.sh -d [path_to_data_device] -w [path_to_swap_device] \
    -f [size_of_swap_file] -u [username]
```

- Format and mount data and swap drives
- Create swap files

#### Run MagellanMapper on server

Log into your instance and run the MagellanMapper pipeline of choice.

- SSH into your server instance, typically with port forwarding to allow VNC access:

```
ssh -L 5900:localhost:5900 -i [your_aws_pem] ec2-user@[your_server_ip]
```

- If necessary, start a graphical server (eg `vncserver`) to run ImageJ/Fiji for stitching or for Mayavi dependency setup
- Setup drives: `bin/setup_server.sh -s`, where the `-s` flag can be removed on subsequent launches if the drives are already initialized
- If MagellanMapper has not been installed, install it with `bin/setup_conda.sh` as above
- Activate the Conda environment set up during installation
- Run a pipeline, such as this command to fully process a multi-tile image with tile stitching, import to Numpy array, and cell detection, with AWS S3 import/export and Slack notifications along the way, followed by server clean-up/shutdown:

```
bin/process_nohup.sh -d "out_experiment.txt" -o -- bin/pipelines.sh \
  -i "/data/HugeImage.czi" -a "my/s3/bucket" -n \
  "https://hooks.slack.com/services/my/incoming/webhook" -p full -c
```

## Troubleshooting

### Installation on Windows

Currently MagellanMapper uses many Bash scripts, which require Cygwin or more recently Windows Subsystem for Linux (WSL) to run. Theoretically MagellanMapper most likely could run without them, which we will need to test.

In the meantime, here are instructions for either Linux-like layer:

#### WSL

After loading a WSL terminal, setup the MagellanMapper environment using the same steps as for Mac. SimpleElastix can be built during or after the setup as above.

Running in WSL requires setting up an X Server since WSL does not provide graphical support out of the box. In our experience, the easiest option is to use [MobaXTerm](https://mobaxterm.mobatek.net/), which supports HiDPI and OpenGL.

An alternative X Server is Cygwin/X, which requires the following modifications:

- Change the XWin Server startup shortcut to include `/usr/bin/startxwin -- -listen tcp +iglx -nowgl` to use indirect OpenGL software rendering (see [here](https://x.cygwin.com/docs/ug/using-glx.html))
- For HiDPI screens, run `export QT_AUTO_SCREEN_SCALE_FACTOR=0` and `export QT_SCALE_FACTOR=2` to increase window/font size (see [here](https://wiki.archlinux.org/index.php/HiDPI#Qt_5))

#### Cygwin

As an alternative to WSL, Cygwin itself can be used to build MagellanMapper and run without requiring an X server. Many dependencies must be built, however, using Cygwin's own `gcc`. At least as of 2019-03, VTK is not available for Cygwin.

#### MSYS2

As an alternative to Cygwin, MSYS2 can use binaries for many dependencies, minimizing build time. It can also use the MS Visual Studio compiler to build the dependencies that do require compilation. Note that `vcvars64.bat` or equivalent scripts do not appear to be required for these compilations.


### Qt/Mayavi/VTK errors

```
Numpy is required to build Mayavi correctly, please install it first
```

- During installation via `pip install -r requirements.txt`, the Mayavi package [may fail to install](https://github.com/enthought/mayavi/issues/782)
- Rerunning this command appears to allow Mayavi to find Numpy now that it has been installed


```
QXcbConnection: Could not connect to display
qt.qpa.xcb: could not connect to display
```

- As of at least 2018-01-05, Mayavi installation requires a GUI so will not work directly in headless cloud instances
- For servers, use RDP or an X11 forwarding instead
- For non-graphical setups such as WSL, start an X11 server (eg in Windows)
- `setup_conda.sh -s environment_light.yml` will setup a lightweight environment without Mayavi, which allows non-interactive whole image processing

```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

- Leads to `ImportError: No module named 'vtkRenderingOpenGL2Python'`
- Install the `libgl1-mesa-glx` package in Ubuntu (or similar in other distros)

```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
```

- A number of missing libraries may prevent Qt from loading
- Run `export QT_DEBUG_PLUGINS=1` to check error messages during startup, which may display the following missing packages:
  - `libxkbcommon-x11.so.0`: `xkbcommon` was removed from Qt [starting in 5.12.1](https://code.qt.io/cgit/qt/qtbase.git/tree/dist/changes-5.12.1?h=5.12.1); install `libxkbcommon-x11-0`
  - `libfontconfig.so.1`: Install `libfontconfig1`
  - `libXrender.so.1`: Install `libxrender1`

Additional errors:

- An error with VTK has prevented display of 3D images at least as of VTK 8.1.2 on RHEL 7.5, though the same VTK version works on Ubuntu 18.04
- PyQt5 5.12 may give an `FT_Get_Font_Format` error, requiring manual downgrade to 5.11.3, though 5.12 works on Ubuntu 18.04

### Long load times

- In our experience, initial load times can be very slow on MacOS 10.15 (Catalina) but improve on subsequent loads
- Images often take longer to display when first displayed because of time reading from disk, but the same part of the image shows faster on subsequent loads during the same session
- Viewing an image from the `yz` plane can be very slow because of how the image is accessed from disk

### Java installation for Python-Bioformats/Javabridge

- Double-check that the Java SDK has truly been installed since the MagellanMapper setup script may not catch all missing installations
- You may need to set up the JAVA\_HOME and JDK\_HOME environment variables in your `~/.bash_profile` or `~/.bashrc` files, such as:

```
# for a specific JDK installation
export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk1.8.0_111.jdk/Contents/Home
# or for the latest JDK you have installed
export JAVA_HOME="$(/usr/libexec/java_home)"
# then add to JDK_HOME and your path
export JDK_HOME="$JAVA_HOME"
export "PATH=$JAVA_HOME/bin:$PATH"
```

- Or add to the Windows Environment Variables "Path"
- Java 9 [changed](http://openjdk.java.net/jeps/220) the location of `libjvm.so`, fixed [here](https://github.com/LeeKamentsky/python-javabridge/pull/141) in the Python-Javabridge 1.0.18
- Java 12 no longer allows source <= 6, fixed in Python-Javabridge >1.0.18
- `setup_conda.sh` does not detect when Mac wants to install its own Java so will try to continue installation but fail at the Javabridge step; if you don't know whether Java is installed, run `java` from the command-line to check and install any Java 8+ (eg from [OpenJDK](http://openjdk.java.net/), not the default Mac installation) if necessary

### Command Line Tools setup (Mac)

- `setup_conda.sh` will attempt to detect whether the required Command Line Tools package on Mac is installed and activated. If you get:

```
xcrun: error: invalid active developer path (/Library/Developer/CommandLineTools), \
missing xcrun at: /Library/Developer/CommandLineTools/usr/bin/xcrun
```

- The Command Line Tools package on Mac may need to be installed or updated. Try `xcode-select --install` to install Xcode. If you get an error (eg "Can't install the software because it is not currently available from the Software Update server"), try downloading [Xcode directly](https://developer.apple.com/download/), then run `sudo xcodebuild -license` to accept the license agreement.

### Image Stitching

- Image stitching is run through ImageJ/Fiji
  - ImageJ itself also depends on Java but does not work well on Java > 8 (as of 2019-01-29)
  - As of MagellanMapper v0.8.3, an argument can be given to `pipelines.sh` and `stitch.sh` to specify the Java home specifically for ImageJ, which should be a typical path exported as `JAVA_HOME` but here passed as an argument to ImageJ, eg:

```
bin/pipelines.sh -j /Library/Java/JavaVirtualMachines/jdk1.8.0_131.jdk/Contents/Home
```

- Two ImageJ stitching plugins are available, which MagellanMapper runs as ImageJ scripts in minimize the need for intervention:
  - The original stitcher, `Stitching`, requires a large amount of RAM/swap space and runs single-threaded, taking days to stitch a multi-tile image
  - The new, recommended stitcher, `BigStitcher`, uses RAM much more efficiently through an HDF5 format and utilizes multiprocessing
- BigStitcher currently requires a graphical environment, which is also recommended for manual verification of tile alignment
- The threshold for links between tiles is set high to minimize false links, falling back on metadata, but still may give false alignments, so manual inspection of stitched images is recommended
- To fix alignments in BigStitcher:
 - Copy the `.xml~2` file to `fix.xml` to obtain the state just before the optimization step and open this file in BigStitcher
 - Use its Link Explorer to remove inappropriate links
 - Run the global optimizer again with two round and metadata fallback
 - If necessary, right-click in the Stitching Explorer to access the `Arrange views > Manually translate views` tool to move specific tiles

### International setup
- If you get a Python locale error, add these lines to your `~/.bash_profile` file:

```
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
```

### Additional tips

- If you get an `syntax error near unexpected token (` error, the run script may have been formatted incorrectly, eg through the Mac Text Editor program. Try `dos2unix [pipelines.sh]` (replace with your run script filename).

## Obsolete Issues

### Windowing responsiveness

- Controls (eg buttons, dropdowns) fail to update on PyQt5 5.10.1 on MacOS 10.10 until switching to another window and back again
- Workaround was to `pip uninstall PyQT5` and `conda install pyqt` to get the previously tested working PyQt version, 5.6.0, instead; newer versions such as 5.11.3 also work
- Some text may not update in PyQT 5.10.1 on later Mac versions until switching windows
