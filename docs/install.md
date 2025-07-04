# Installation of MagellanMapper

## Table of Contents

- [Installation options](#installation-options)
- [Update MagellanMapper](#update-magellanmapper)
- [Building dependencies](#dependencies)
- [Uninstallation](#uninstallation)
- [Build documentation](#build-documentation)
- [Tested platforms](#tested-platforms)
- [Troubleshooting](#troubleshooting)

## Installation Options

MagellanMapper supports several Python setups.

### Quick Install

Install MagellanMapper with its graphical interface and registration tools:

```shell
pip install "magellanmapper[gui,itk]"
```

Then launch MagellanMapper:

```shell
mm
```

See [below](#dependencies) for supported Python versions and adding install groups.

See our [vignette](https://github.com/sanderslab/magellanmapper/blob/master/bin/sample_cmds_bash.ipynb) for getting started on MM!

### Full install using Conda

If you use Conda (available [here](https://docs.conda.io/en/latest/miniconda.html)), you can install MagellanMapper into a new environment named `mag` (or replace with desired name):

```shell
conda env create -n mag -f https://raw.githubusercontent.com/sanderslab/magellanmapper/master/envs/environment_rel.yml
```

To run, activate the environment and launch MagellanMapper by `mm`:

```shell
conda activate mag
mm
```

Conda will also install Java, which we use to read proprietary image formats.

The `mm` entry points was added in v1.6.0 to facilitate launching from installed packages.

### Full install using Pip

Install using Pip with Python >= 3.6 (see [Python versions](#python-version-support); Python >= 3.9 and [virtual environment](https://realpython.com/python-virtual-environments-a-primer/) recommended):

```shell
pip install "magellanmapper[most]" --extra-index-url https://pypi.fury.io/dd8/
```

The `most` group installs the GUI and file import tools (see [optional dependencies below](#optional-installation-groups)). The extra index accesses a few [customized dependencies](#custom-packages) for MagellanMapper.

Java will need to be installed to support more image formats (eg from [here](https://www.azul.com/downloads/?package=jdk)).

### Developer installs

You can install directly from the source code for the latest updates.

First, download the repo:

```shell
git clone https://github.com/sanderslab/magellanmapper.git
```

Next, install it:
- For Conda:

```shell
conda env create -n mag -f magellanmapper/environment.yml
```

- Or Pip:

```shell
pip install -e "magellanmapper[most]" --extra-index-url https://pypi.fury.io/dd8/
```

MagellanMapper can be run using `mm` and `mm-cli` as above, or through the run script:

```shell
python magellanmapper/run.py
```

### Installer packages

***Note**: We're in the process of determining how useful these are for the community. If you've liked them, please let us know! (And feedback welcome if you've run into any issues with them.)*

The easiest way to install MagellanMapper is using one of the [installers](https://github.com/sanderslab/magellanmapper/releases) now available for Windows, macOS, and Linux.

Windows users: The installer is not yet signed, meaning that Windows will still give some security warnings. If the Edge browser blocks the download, click the Downloads button -> the `...` button on the right of the file entry -> "Keep" -> "Show more" -> "Keep anyway". In the blue window after opening the file, click "More info" -> "Run anyway" to start the installer.

To run:
- **Mac**: run the MagellanMapper app
- **Windows**: in the Start Menu, go to "MagallanMapper v.x.y.z" and run "MagellanMapper"
- **Linux**: in a file browser, double-click on `MagellanMapper/MagellanMapper`

Windows users: The installer is not yet signed, meaning that Windows will still give some security warnings. If the Edge browser blocks the download, click the Downloads button -> the `...` button on the right of the file entry -> "Keep" -> "Show more" -> "Keep anyway". In the blue window after opening the file, click "More info" -> "Run anyway" to start the installer.

On Windows and Mac, you can also use "Open with" on supported file types (eg `.npy`, `.mhd`, `.nii.gz`) to open them in MagellanMapper.

### Installer scripts

We have also provided scripts to take care of installing Miniconda (if necessary), creating an environment, and installing MagellanMapper, without requiring command-line/terminal experience.

#### Conda installer script

Conda simplifies installation by managing all supporting packages such as Java and others that would otherwise need to be compiled. Conda's virtual environment also keeps these packages separate from other Python package installations that may be on your system.

1. Download MagellanMapper by cloning the git repo (or download the [latest release](https://github.com/sanderslab/magellanmapper/releases/latest)):
```
git clone https://github.com/sanderslab/magellanmapper.git
```
1. Install MagellanMapper using the following script in the `magellanmapper` folder
    - On Mac or Linux: `bin/setup_conda`
    - On Windows: `bin\setup_conda.bat`

1. Run by double-clicking on `MagellanMapper` in the main folder (macOS/Linux) or running `run.py` with Python (Windows).

- Installation may take up to 5 minutes, depending on internet connection speed.
- The script will also install the Conda package manager if not already installed.
- To update the environment, rerun the appropriate `setup_conda` script above.
- On Mac, it may be necessary to right-click and "Open with" the Terminal app.
- On Linux, it may be necessary to go to "Preferences" in the file browser (eg the Files app), select the "Behavior" tab, and choose to "Run" or "Ask" when executing text files.
- See [Installation](docs/install.md) for more details and install options.

#### Run from a file browser

**On Mac or Linux**: Double-click the MagellanMapper icon created during Conda setup. This Unix executable should open with Terminal by default on Mac and after the file browser preference change described above on Linux.

**On Windows**: Run `run.py` through Python.
- It may be necessary to right-click, choose "Open with", and browse to the Conda `pythonw.exe` file to open `run.py`
- If a security warning displays, click on "More info" and "Run anyway" to launch the file

Note that during the first run, there may be a delay of up to several minutes from antivirus scanning for the new Python interpreter location in the new environment. Subsequent launches are typically much faster.

#### Run from a terminal

See [above](#install-using-conda) for running MagellanMapper from a terminal, which is recommended when running command-line tasks or for debugging output.

#### Venv+Pip installer script

Venv is a virtual environment manager included with Python. We have provided a convenient script to set up a new environment and install all dependencies using Pip.

```shell
bin/setup_venv.sh [-n name]
```

This setup script will check and install the following dependencies:

- Checks for an existing compatible Python version
- Creates a new virtual environment
- Performs a Pip install of MagellanMapper and all dependencies

On Windows, the [Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019](https://visualstudio.microsoft.com/downloads/#microsoft-visual-c-redistributable-for-visual-studio-2019) (same package for all three years) is required.

## Update MagellanMapper

### Update the MagellanMapper package

If installed from a Python package, enter your virtual environment and run:

```shell
pip install -U magellanmapper --extra-index-url https://pypi.fury.io/dd8/
```

If installed from source:
- For a cloned Git repo: run `git pull` to pull in all software updates
- From a source code release: download the desired [release](https://github.com/sanderslab/magellanmapper/releases), extract it, and run MagellanMapper from there

### Update the Conda or Venv environment

Sometimes a virtual environment update is required for new dependencies.
- To update a Conda environment, rerun the `bin/setup_conda` (macOS/Linux) or `bin\setup_conda.bat` (Windows) script
- To update a Venv environment, rerun the `bin/setup_venv.sh` (macOS/Linux) or `bin\setup_venv.bat` (Windows) script

## Dependencies

### Python version support

| MagellanMapper Version | Python Versions Supported | Notes |
|-----|-----|-----|
| 1.7a1 | 3.10-3.13 | Defaults to 3.10. 3.6-3.9 support removed. |
| >=1.6b1 | 3.6-3.11 | Defaults to 3.9. GUI support added for 3.10-3.11 in MM 1.6b2, 3.12 in MM 1.6b5. |
| 1.6a1-a3 | 3.6-3.9 | GUI support added for 3.9; MM 1.6a2 base group no longer installs GUI |
| 1.4-1.5 | 3.6-3.9 | No GUI support in 3.9 |
| < 1.4 | 3.6 | Initial releases |

As of MM 1.6a2, the GUI can be excluded by installing the base group, eg without `[gui]` or `[most]`.

### Pinned packages

We've provided a few sets of pinned dependency versions:
- Python >= 3.10: `envs/requirements.txt`
- Python 3.8 (deprecated): `envs/requirements_py38`
- Python 3.7 (deprecated): `envs/requirements_py37`
- Python 3.6 (deprecated): `envs/requirements_py36`

These package versions are used for automated testing (continuous integration).

### Optional installation groups

| Group | Packages | Collection |
|-----|-----|-----|
| `most` | Import, GUI, and registration tools | Has `import`, `gui`, `itk` |
| `all` | All groups plus `seaborn`, `scikit-learn` | Has all below |
| `3D` | 3D rendering | |
| `aws` | Tools for accessing AWS | |
| `classifer` | Tensorflow | |
| `docs` | Tools for building docs | |
| `gui` | Main graphical interface | |
| `import` | Imports proprietary image formats | |
| `itk` | ITK-Elastix | |
| `jupyter` | Running Notebooks | |
| `pandas_plus` | Exports styled and Excel formats | |
| `simpleitk` | Custom SimpleITK with Elastix | |

To add an install group:

```shell
pip install "magellanmapper[3d]" # add "3d" group
pip install "magellanmapper[3d,gui]" # add two groups
pip install -e ".[3d]" # same but for editable install from clone
```

The same commands can be run to add groups after initial installation.

### Optional Dependency Build and Runtime Requirements

#### Custom packages

In most cases MagellanMapper can be installed without a compiler by using custom dependency packages that we have pre-built and hosted.

| Dependency | Custom Package | Precompiled Run Req | Build Req | Purpose |
|-----|-----|-----|-----|-----|
| Python-Javabridge | Precompiled with later updates than current release | Python 3.6-3.11, Java 8+ | JDK, C compiler| For Python-Bioformats |
| Python-Bioformats | Extended support for older NumPy releases | Python 3.6+ | JDK, C compiler | Import proprietary image formats |
| SimpleITK with Elastix | Precompiled with Elastix support. **Deprecated**: Defaults to ITK-Elastix in MM v1.6b2. | Python 3.6-3.11 | C, C++ compilers | Load medical 3D formats, image regsitration |

C compilers by platform:

- Mac and Linux: `gcc`/`clang`
- Windows: Microsoft Visual Studio (tested on 2017, 2019, Community edition) along with these additional components
  - MSVC C++ x64/x86 build tools
  - Windows 10 SDK
  - See [below](#simpleitk-with-elastix-dependency) for additional requirements when building SimpleElastix

Java versions:

- The Conda setup pathway installs JDK 8
- Python-Javabridge uses JDK v8+ (v12+ in Javabridge 1.0.19; see [below](#image-loading) for image loading times and setup troubleshooting with various Java versions)
- ImageJ/Fiji currently supports Java 8 best in our experience

#### Additional optional packages

- R for additional stats
- Zstd (fallback to Zip) for compression on servers
- MeshLab for 3D surface clean-up

### SimpleITK with Elastix dependency

**Deprecated**: As of MM v1.6b2, ITK-Elastix will be installed instead. SimpleITK is still supported but no longer the default, and custom binaries are no longer generated.

SimpleITK with Elastix is used for loading many 3D image formats (eg `.mhd/.raw` and `.nii`) and registration tasks in MagellanMapper. The library in the standard [PyPi](https://pypi.org/) is not currently built with Elastix support. As the buid process is not trivial, we have uploaded binaries to a [third-party PyPi server](https://pypi.fury.io/dd8/). On Windows, the [Microsoft Visual C++ Redistributable for Visual Studio 2022](https://visualstudio.microsoft.com/downloads/#microsoft-visual-c-redistributable-for-visual-studio-2022) is required to run this package.

If you would prefer to build SimpleITK with Elastix yourself, we have provided a couple build scripts to ease the build process for the SimpleElastix Python wrapper:

- Mac or Linux: Run `bin/build_se.sh` within a Python environment. Building SimpleElastix requires `cmake`, `gcc`, `g++`, and related compiler packages.
- Windows: Run `bin\build_se.bat` within your environment. See the [build_se.bat](https://github.com/sanderslab/magellanmapper/blob/master/bin/build_se.bat) script for required build components. 

As an alternative, the SimpleITK package can provide much of the same functionality except for our image registration pipeline.


## Uninstallation

### Uninstall MagellanMapper

To uninstall MagallanMapper, simply remove the MagellanMapper folder. This folder may be named `magellanmapper` if you downloaded it from Git, `magellanmapper-x.y.z` if you downloaded a specific release, or `magellanmapper-master` if you downloaded the latest ZIP file.

### Conda uninstalls

#### Option 1: Only uninstall Conda environment

This command will uninstall only the Conda environment created by the `setup_conda[.bat]` scripts while keeping the Conda installation:

```
conda remove -n mag --all
```

If you created an environment with a custom name, replace `mag` with this name.

#### Option 2: Fully uninstall Conda

1. Remove the Conda initialization from shell profiles:

    ```
    conda init --reverse
    ```

1. Uninstall Conda, which will also remove the environment created by the setup scripts.
    - Mac and Linux: Remove the Conda folder, typically `<home-path>/miniconda3`.
    - Windows: Run the Anaconda or Miniconda uninstaller from Windows Settings > Apps.

For more details on uninstalling Conda, see the [Conda uninstallation directions](https://docs.anaconda.com/anaconda/install/uninstall/).


## Build Documentation

API documentation can be built using Sphinx. To install docs dependencies, run:

```
pip install -e .[docs]
```

We have provided a convenience Bash script to generate HTML files with Sphinx. To build all required files, run:

```
bin/build_docs.sh -a
```

Output API docs can be accessed from `docs/_build/html/index.html`.

Note that there may be many warnings but otherwise correct output. On subsequent runs, the `-a` flag can be omitted if no new modules are added. The `-c` flag will clean the documentation before reubuilding it.


## Tested Platforms

MagellanMapper has been built and tested to build on:

- MacOS, tested on 10.11-10.15
- Linux, tested on RHEL 7.4-7.5, Ubuntu 18.04-20.04
- Windows, tested on Windows 10 (see below for details) in various environments:
  - Native command-prompt and PowerShell
  - Via built-in Windows Subsystem for Linux (WSL), tested on Ubuntu 18.04 and an X Server
  - Bash scripts in Cygwin (tested on Cygwin 2.10+) and MSYS2

## Troubleshooting

### Installation on Windows

Currently, MagellanMapper uses many Bash scripts, which require Cygwin or more recently Windows Subsystem for Linux (WSL) to run. Theoretically MagellanMapper most likely could run without them, which we will need to test.

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


### Qt errors

```
QXcbConnection: Could not connect to display
qt.qpa.xcb: could not connect to display
```

- As of at least 2018-01-05, Mayavi installation requires a GUI so will not work directly in headless cloud instances
- For servers, use RDP or an X11 forwarding instead
- For non-graphical setups such as WSL, start an X11 server (eg in Windows)

```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
```

- A number of missing libraries may prevent Qt from loading
- Run `export QT_DEBUG_PLUGINS=1` to check error messages during startup, which may display the following missing packages:
  - `libxkbcommon-x11.so.0`: `xkbcommon` was removed from Qt [starting in 5.12.1](https://code.qt.io/cgit/qt/qtbase.git/tree/dist/changes-5.12.1?h=5.12.1); install `libxkbcommon-x11-0`
  - `libfontconfig.so.1`: Install `libfontconfig1`
  - `libXrender.so.1`: Install `libxrender1`
  - `libxcb-xinerama.so.0`: Install `libxcb-xinerama0`

```
Could not load the Qt platform plugin "xcb" in "" even though it was found.
WARNING: This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: eglfs, linuxfb, minimal, minimalegl, offscreen, vnc, wayland-egl, wayland, wayland-xcomposite-egl, wayland-xcomposite-glx, webgl, xcb.
```
- For non-graphical environments with this additional error info, may need to set an offscreen plugin: `export QT_QPA_PLATFORM=offscreen`


Additional errors:

- PyQt5 5.12 may give an `FT_Get_Font_Format` error, requiring manual downgrade to 5.11.3, though 5.12 works on Ubuntu 18.04

### Mayavi/VTK errors

```
root - ERROR - The traitsui.qt4.* modules have moved to traitsui.qt.*.

Applications which require backwards compatibility can either:

- set the ETS_QT4_IMPORTS environment variable
- set the ETS_TOOLKIT environment variable to "qt4",
- the ETSConfig.toolkit to "qt4"
- install a ShadowedModuleFinder into sys.meta_path::

    import sys
    from pyface.ui import ShadowedModuleFinder

    sys.meta_path.append(ShadowedModuleFinder(
        package="traitsui.qt4.",
        true_package="traitsui.qt.",
    ))
```

**UPDATE 2024-01-30**: A workaround in #618 fixes this error. VTK < 9.3 is also required at this time.

*Previously*:
At least as of Mayavi 4.8.1, Mayavi will not load TraitsUI 8. Workaround is to run in the shell before launching MM: `export ETS_TOOLKIT="qt4"`

```
Numpy is required to build Mayavi correctly, please install it first
```

- During installation via `pip install -r envs/requirements.txt`, the Mayavi package [may fail to install](https://github.com/enthought/mayavi/issues/782)
- Rerunning this command appears to allow Mayavi to find Numpy now that it has been installed
- A similar error may occur in other install routes but corrects itself (ie the error can be ignored)
- On Windows, Mayavi appears to require the [Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019](https://visualstudio.microsoft.com/downloads/#microsoft-visual-c-redistributable-for-visual-studio-2019) to install

```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

- Leads to `ImportError: No module named 'vtkRenderingOpenGL2Python'`
- Install the `libgl1-mesa-glx` package in Ubuntu (or similar in other distros)

```
ERROR: In ..\Rendering\OpenGL2\vtkWin32OpenGLRenderWindow.cxx, line 686
vtkWin32OpenGLRenderWindow (00000252735E3450): failed to get wglChoosePixelFormatARB

ERROR: In ..\Rendering\OpenGL2\vtkWin32OpenGLRenderWindow.cxx, line 765
vtkWin32OpenGLRenderWindow (00000252735E3450): failed to get valid pixel format.

ERROR: In ..\Rendering\OpenGL2\vtkOpenGLRenderWindow.cxx, line 741
vtkWin32OpenGLRenderWindow (00000252735E3450): GLEW could not be initialized: Missing GL version
```

Windows running as a virtual machine (eg in VirtualBox) may require installation of Mesa for OpenGL support. Pre-built Mesa software is available from [Mesa-Dist-Win](https://github.com/pal1000/mesa-dist-win).

1. Download and extract Mesa-Dist-Win [20.3.4](https://github.com/pal1000/mesa-dist-win/releases/tag/20.3.4) (later versions apparently require an additional Vulkan library download)
1. Run `systemwidedeploy.cmd`
1. Install the "desktop OpenGL" drivers

```
ERROR: In ../Rendering/OpenGL2/vtkOpenGLRenderWindow.cxx, line 754
vtkXOpenGLRenderWindow (0x7fffc7718590): Unable to find a valid OpenGL 3.2 or later implementation.
```

This error may occur in WSL even with an X11 server open. Workarounds include:

- Run `export LIBGL_ALWAYS_INDIRECT=0` and/or `export QT_X11_NO_MITSHM=1`
- Or simply run the software from within MobaXTerm

Additional errors:

- An error with VTK has prevented display of 3D images at least as of VTK 8.1.2 on RHEL 7.5, though the same VTK version works on Ubuntu 18.04

```Building TVTK classes... Windows fatal exception: code 0xc0000374``` or ```Building TVTK classes... Fatal Python error: Segmentation fault```

* Appears to be a sporadic installation issue (see [issue](https://github.com/sanderslab/magellanmapper/issues/401))
* Workaround: reinstall Mayavi: `pip install mayavi`
* Starting with MM v1.6a4, Mayavi will no longer be installed by defaul

### Display issues

#### Window is too large for screen

- On high-definition (HiDPI) desktops where display scaling is set to a non-integer factor (eg 150%), the window may expand beyond the size of small screens
- This problem is fixed with Qt >= 5.14
- Workaround 1: Use the Venv instead of the Conda install script. As of 2020-10-08, PyQt5/Qt 5.12 is the latest verion available on Conda, but >= 5.14 is available on Pip, including installation using `bin/setup_venv.sh`.
- Workaround 2: Replace PyQt5/Qt from Conda with Pip versions.
    1. Activate the Conda environment: `conda activate mag`
    1. Uninstall Conda packages: `conda uninstall --force pyqt qt`
    1. Install Pip packages: `pip install PyQt5`
- Note that these workarounds are unlikely to work for small, non-HiDPI screens

### Long load times

#### Initial software launch on macOS

- In our experience, initial load times can be very slow on MacOS 10.15 (Catalina) but improve on subsequent loads

#### Bioformats/Java initialization

- Image imports that require Bioformats/Java are slower to initialize in the Conda pathway because it uses an older Java version (Java 8) for backward compatibility
- After initialization, import speed is generally similar as with newer Java versions
- Workaround: Replace the Conda environment with the latest Java version
    - Open the `environment.yml` file and change `openjdk=8` to simply `openjdk`
    - [Uninstall](#option-1-only-uninstall-conda-environment) the Conda environment
    - [Set up](#recommended-install-in-a-conda-environment) a new Conda environment

#### Image loading

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
