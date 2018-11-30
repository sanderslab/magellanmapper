# Clrbrain
Clrbrain is an imaging informatics GUI and pipeline for high-throughput, automated analysis of whole organs. Its design philosophy is to make the raw, original images as accessible as possible; simplify annotation with a focus on nuclei; and scale from the laptop to the cloud in cross-platform environments.

Author: David Young, 2017-2018, Stephan Sanders Lab

## Download

Currently access is limited to a private Git repo. Our eventual plan is to make Clrbrain available via Anaconda and Pip.

- Contact the project managers about loading your public key to access the project repository
- Download the repo: `git clone git@bitbucket.org:psychcore/clrbrain.git`

## Installation
```
./setup_env.sh
```
The setup script will install the following:

- [Miniconda](https://conda.io/miniconda.html), a light version of the Anaconda package and environment manager for Python, will be installed if an existing installation isn't found
- A `clr3` environment with Python 3
- Scipy, Numpy, Matplotlib stack
- Mayavi and TraitsUI stack for GUI and 3D visualization; note that Mayavi currently requires a graphical environment to install
- Scikit-Image for image processing
- Pandas for stats I/O
- SimpleITK or [SimpleElastix](https://github.com/SuperElastix/SimpleElastix), a fork with Elastix integrated (see below)

On occasion, Python dependencies with required updates that have not been released will be downloaded as shallow Git clones into the parent folder of Clrbrain (ie alongside rather than inside Clrbrain) and pip installed.

To install/run without a GUI, run a lightweight setup, `./setup_env.sh -l` ("L" arg), which avoids the Mayavi stack.

### SimpleITK/SimpleElastix dependency

SimpleElastix is used for registration tasks in Clrbrain. The library is not currently available in Pip and must be built manually.

Running the environment setup with `./setup_env.sh -s` will attempt to build and install SimpleElastix during setup. Without the `-s` flag, setup will fallback to installing SimpleITK, from which SimpleElastix is derived, to allow opening many 3D image formats such as `.mhd/.raw` and `.nii` files.

To build and install SimpleElastix manually, the `build_se.sh` script can be called directly. Be sure to uninstall SimpleITK from the environment (`pip uninstall SimpleITK`) before installing SimpleElastix to avoid a conflict.

### Additional Build/Runtime Dependencies

#### Required

- Java SDK, tested on v8-11, for importing image files from proprietary formats (eg `.czi`)

#### Optional

- GCC or related compilers for compiling Mayavi and SimpleElastix
- Git for downloading unreleased dependencies as above
- R for statistical models
- Zstd (fallback to Zip) for compression on servers
- MeshLab for 3D surface clean-up

### Tested Platforms

Clrbrain has been tested to build and run on:

- MacOS, tested on 10.11+
- Linux, tested on RHEL 7.4+
- Windows, tested on Windows 10 (see below for details):
  - Via built-in Windows Subsystem for Linux (WSL), tested on Ubuntu 18.04 and an X Server
  - Via Cygwin, tested on Cygwin 2.10+

## Run Clrbrain

Clrbrain can be run as a GUI or headlessly for desktop or server tasks, respectively. To start Clrbrain:

- Open a new terminal if you just installed Miniconda
- Run the script:

```
source activate clr3
./runclrbrain.sh -i [path_to_your_image]
```

`runclrbrain.sh` is a script to run many pipelines within Clrbrain, such as whole volume nuclei detection and image transposition. The default pipeline will open the Clrbrain GUI.

Proprietary image formats such as `.czi` will be imported automatically via Bioformats into a Numpy array format before loading it in the GUI. Overlaid images in formats that SimpleITK can open do not require import (we are currently working to open these files as primary images without requiring any import).

## 3D viewer

The main Clrbrain GUI displays a 3D viewer and ROI selection controls. Clrbrain uses Mayavi for 3D voxel or surface rendering.

From the ROI selection controls, two different 2D editors can be opened. All but the last `2D styles` options open various forms of the Nuclei Annotation Editor. The final option opens the Atlas Editor, a 2D/3D viewer.

## Nuclei Annotation Editor

The multi-level 2D plotter is geared toward simplifying annotation for nuclei. Press on "Detect" to detect nuclei in the current ROI, then "Plot 2D" to open the figure.

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

The atlas image must have an associated annotation image. Use the `--labels` flage to specify a labels `.json` file. Change the "2D plot styles" dropdown to "Atlas editor" and press "Plot 2D" to open the editor.

- Mouseover over any label to see the region name
- Shift+mouseover to move the crosshairs and the corresponding planes
- Scroll or arrow up/down to move planes in the current plot
- Right-click or Ctrl+click + mouse-up/down to zoom
- Middle-click or Alt+click + mouse drag to pan
- Click on the color you want to extend, then drag onto the area you want to paint with that color
- `[` (left bracket) to make the paintbrush smaller
- `]` (right bracket) to make it bigger
- Use the save button in the main window with the atlas window still open to resave


## Start a processing pipeline

Automated processing will attempt to scale based on your system resources but may require some manual intervention. This pipeline has been tested on a Macbook Pro laptop and AWS EC2 Linux (RHEL and Amazon Linux based) instances.

Optional dependencies:

- ImageJ/Fiji with the BigStitcher plugin: required for tile stitching; downloaded automatically onto a server when running `deploy.sh`
- ImageMagick: required for stack animation
- [Slack incoming webhook](https://api.slack.com/incoming-webhooks): to notify when tile stitching alignment is ready for verification and pipeline has completed

### Local
Run a pipeline in `runclrbrain.sh`.

For example, load a `.czi` file and display in the GUI, which will import the file into a Numpy format for faster future loading:

```
./runclrbrain.sh -i data/HugeImage.czi
```

To sitch a multi-tile image and perform cell detection on the entire image, which will load BigStitcher in ImageJ/Fiji for tile stitching:

```
./runclrbrain.sh -i data/HugeImage.czi -p full
```

See `runclrbrain.sh` for additional sample commands for common scenarios, such as cell detection on a small region of interest. The file can be edited directly to load the same image, for example.

### Server

Optional dependencies:

- `awscli`: AWS Command Line Interface for basic up/downloading of images and processed files S3. Install via Pip.
- `boto3`: AWS Python client to manage EC2 instances. 

#### Launch a server

You can launch a standard server, deploy Clrbrain code, and run a pipeline. Note that typically login with graphical support (eg via `vncserver`) is required during installation for Mayavi and stitching in the standard setup, but you can alternatively run a lightweight install without GUI (see above).

If you already have an AMI with Clrbrain installed, you can launch a new instance of it via Clrbrain:

```
python -u -m clrbrain.aws --ec2_start "Name" "ami-xxxxxxxx" "m5.4xlarge" "subnet-xxxxxxxx" "sg-xxxxxxxx" "UserName" 50,2000 [2]
```

- `Name` is your name of choice
- `ami` is your previously saved AMI with Clrbrain
- `m5.4xlarge` is the instance type, which can be changed depending on your performance requirements
- `subnet` is your subnet group
- `sg` is your security group
- `UserName` is the user name whose security key will be uploaded for SSH access
- `50,2000` creates a 50GB swap and 2000GB data drive, which can be changed depending on your needs
- `2` starts two instances (optional, defaults to 1)

#### Setup server with Clrbrain

Deploy the Clrbrain folder and supporting files:

```
./deploy.sh -p [path_to_your_aws_pem] -i [server_ip] \
    -d [optional_file0] -d [optional_file1]
```

- This script by default will:
  - Archive the Clrbrain Git directory and `scp` it to the server, using your `.pem` file to access it
  - Download and install ImageJ/Fiji onto the server
  - Update Fiji and install BigStitcher for image stitching
- To only update an existing Clrbrain directory on the server, add `-u`
- To add multiple files or folders such as `.aws` credentials, use the `-d` option as many times as you'd like

#### Run Clrbrain on server

Log into your instance and run the Clrbrain pipeline of choice.

- SSH into your server instance, typically with port forwarding to allow VNC access:
```
ssh -L 5900:localhost:5900 -i [path_to_your_aws_pem] ec2-user@[your_server_ip]
```
- If necessary, start a graphical server (eg `vncserver`) to run ImageJ/Fiji for stitching or for Mayavi dependency setup
- Setup drives: `clrbrain/setup_server.sh -s`, where the `-s` flag can be removed on subsequent launches if the drives are already initialized
- If Clrbrain has not been installed, install it with `clrbrain/setup_env.sh` as above
- Activate the Conda environment set up during installation
- Run a pipeline, such as this command to fully process a multi-tile image with tile stitching, import to Numpy array, and cell detection, with AWS S3 import/export and Slack notifications along the way, followed by server clean-up/shutdown:
```
clrbrain/process_nohup.sh -d "out_experiment.txt" -o -- ./runclrbrain.sh -i "/data/HugeImage.czi" -a "my/s3/bucket" -n "https://hooks.slack.com/services/my/incoming/webhook" -p full -c
```

## Troubleshooting

### Java installation

- Tested on Java 8-11 SE
- Double-check that the Java SDK has truly been installed since the Clrbrain setup script may not catch all missing installations
- You may need to set up the JAVA_HOME environment variable, such as `JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk1.8.0_111.jdk/Contents/Home`, and add this variable to your PATH in `~/.bash_profile`
- Java 9 [changed](http://openjdk.java.net/jeps/220) the location of `libjvm.so`, fixed [here](https://github.com/LeeKamentsky/python-javabridge/pull/141)
- Java 11 similarly changed locations, also fixed in Python-Javabridge
- `setup_env.sh` does not detect when Mac wants to install its own Java so will try to continue installation but fail at the Javabridge step

### Xcode setup (Mac)

- `xcrun: error: invalid active developer path (/Library/Developer/CommandLineTools), missing xcrun at: /Library/Developer/CommandLineTools/usr/bin/xcrun` error: The Command Line Tools package on Mac may need to be installed or updated. Try `xcode-select --install` to install Xcode. If you get an error (eg "Can't install the software because it is not currently available from the Software Update server"), try downloading Xcode directly from https://developer.apple.com/download/, then run `sudo xcodebuild -license` to accept the license agreement.

### Installation on Windows

Currently Clrbrain uses many Bash scripts, which require Cygwin or more recently Windows Subsystem for Linux (WSL) to run. Theoretically Clrbrain most likely could run without them, which we will need to test.

In the meantime, here are instructions for either Linux-like layer:

#### WSL

After loading a WSL terminal, setup the Clrbrain environment using the same steps as for Mac. SimpleElastix can be built during or after the setup as above.

Running in WSL requires setting up an X Server since WSL does not provide graphical support out of the box. In our experience, the easiest option is to use [MobaXTerm](https://mobaxterm.mobatek.net/), which supports HiDPI and OpenGL.

An alternative X Server is Cygwin/X, which requires the following modifications:

- Change the XWin Server startup shortcut to include `/usr/bin/startxwin -- -listen tcp +iglx -nowgl` to use indirect OpenGL software rendering (see [here](https://x.cygwin.com/docs/ug/using-glx.html))
- For HiDPI screens, run `export QT_AUTO_SCREEN_SCALE_FACTOR=0` and `export QT_SCALE_FACTOR=2` to increase window/font size (see [here](https://wiki.archlinux.org/index.php/HiDPI#Qt_5))

#### Cygwin

As an alternative to WSL, Cygwin itself can be used to build Clrbrain and run without an X server.

Building SimpleElastix on Windows is more complicated, however, requiring the following:

- Install Microsoft Visual Studio Build Tools 2017 with Windows SDK to build Mayavi and Javabridge
- Build SimpleElastix with VS 2017, though this compilation has not worked at least in our experience because of [this issue](https://github.com/SuperElastix/SimpleElastix/issues/126)

Clrbrain will default to installing SimpleITK, which may be sufficient if registration tasks are not required.

### International setup
- If you get a Python locale error, add these lines to your `~/.bash_profile` file:
```
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
```

### Sciki-image installation
- If you continue getting errors during Scikit-image compilation, delete the folder and restart its compilation from scratch
- After updating any Scikit-image Cython files, run `python setup.py build_ext -i` as per https://github.com/scikit-image/scikit-image. If older version of extensions remain, run `git clean -dxf` to completely clear the working directory (check for any working files you need!) before rerunning the extension builder.

### Mayavi installation
- As of at least 2018-01-05, Mayavi installation requires a GUI so will not work on headless cloud instances, giving a `QXcbConnection: Could not connect to display` error; use RDP or an X11 forwarding instead
- As of v.0.6.6 (2018-05-10), `setup_env.sh -l` will setup a lightweight environment without Mayavi, which allows non-interactive whole image processing

### Windowing responsiveness
- Controls (eg buttons, dropdowns) fail to update on PyQt5 5.10.1 on MacOS 10.10 until switching to another window and back again
- Workaround is to `pip uninstall PyQT5` and `conda install pyqt` to get PyQt 5.6.0 instead
- Some text may not update in PyQT 5.10.1 on later Mac versions until switching windows

### Image Stitching
- The original stitcher, `Stitching`, requires a large amount of RAM/swap space and runs single-threaded, taking days to stitch a multi-tile image
- The new, recommended stitcher, `BigStitcher`, uses RAM much more efficiently through an HDF5 format and utilizes multiprocessing
- Clrbrain runs these stitchers as ImageJ scripts in an attempt to require minimal intervention
- BigStitcher throws an exception without a graphical environment, however
- The threshold for links between tiles is set high to minimize false links, falling back on metadata, but still may give false alignments, so manual inspection of stitched images is recommended
- To fix alignments in BigStitcher:
 - Copy the `.xml~2` file to `fix.xml` to obtain the state just before the optimization step
 - Use its Link Explorer to remove inappropriate links
 - Run the global optimizer again with two round and metadata fallback
 - If necessary, use the Manually Align tool to move specific tiles

### Additional tips
- If you get an `syntax error near unexpected token (` error, the run script may have been formatted incorrectly, eg through the Mac Text Editor program. Try `dos2unix [runclrbrain.sh]` (replace with your run script filename) or re-copying from `runclrbrain.sh`.