# Installation of MagellanMapper

MagellanMapper can be installed many different ways dependening on one's Python preferences.

## Recommended: Install in a Conda environment

Conda simplifies installation by managing all supporting packages such as Java and others that would otherwise need to be compiled. Conda's virtual environment also keeps these packages separate from other Python package installations that may be on your system.

After downloading MagellanMapper, create a new Conda environment with all dependent packages using a provided setup script below for your platform.

**Mac, Linux**:

```
bin/setup_conda [-n name] [-s spec]
```

**Windows**:

```
bin/setup_conda.bat
```

These scripts perform a full installation of Conda if not already present. Specifically, it will install:

- [Miniconda](https://conda.io/miniconda.html), a free minimal installer for the Anaconda Python Distribution, if not already present based on the `conda` command
- A Conda environment with Python 3, named according to the `-n` option, or `mag` by default
- Full dependencies based on `environment.yml`, or an alternative specification if the `-s` option is given, such as `-s environment_light.yml` for headless systems that do not require a GUI

Alternatively, those who already have Conda installed can set up an envirionment directly based on the environment spec:

```
conda env create -n mag -f environment.yml
```

See the [Readme](../README.md#run-magellanmapper) for instructions on running MagellanMapper.

## Option 2: Install through Venv+Pip

Venv is a virtual environment manager included with Python 3.3+. We have provided a convenient script to set up a new environment and install all dependencies using Pip:

```
bin/setup_venv.sh [-n name]
```

This option assumes that you have already installed Python 3.6 and a Java Development Kit (JDK) 8. Other versions of Python 3 and the JDK may work but with varying other requirements, such as a C compiler to build dependencies (see [below](#custom-precompiled-packages)).

This setup script will check and install the following dependencies:

- Checks for an existing Python 3.6+, which already includes Venv
- Performs a Pip install of MagellanMapper and all dependencies

## Option 3: Install in another virtual environment or system-wide

Whether in a virtual environment of your choice or none at all, MagellanMapper can be installed through Pip:

```
pip install -e . --extra-index-url https://pypi.fury.io/dd8/
```

The extra URL provides pre-built custom (with [certain requirements](#custom-precompiled-packages)) dependency packages. To include all dependencies, run this command instead:

```
pip install -e .[all] --extra-index-url https://pypi.fury.io/dd8/
```

## Option 4: Even more installation methods

You can also install MagellanMapper these ways in the shell and Python environment of your choice:

- In a Python environment of your choice or none at all, run `pip install -r requirements.txt` to match dependencies in a pinned, current test setup (cross-platform)
- To create a similar environment in Conda, run `conda env create -n [name] -f environment_[os].yml`, where `name` is your desired environment name, and `os` is `win|mac|lin` for your OS (assumes 64-bit)
- To install without Pip, run `python setup.py install` to install the package and only required dependencies

## Dependencies

The main required and optional dependencies in MagellanMapper are:

- Scipy, Numpy, Matplotlib stack
- Mayavi/TraitsUI/Qt stack for GUI and 3D visualization
- Scikit-image for image processing
- Scikit-learn for machine learning based stats
- Pandas for stats
- [SimpleElastix](https://github.com/SuperElastix/SimpleElastix), a fork of SimpleITK with Elastix integrated (see below)
- Python-Bioformats/Javabridge for importing images from propriety formast such as `.czi` (optional, requires Java SDK and C compiler)

### Optional Dependency Build and Runtime Requirements

In most cases MagellanMapper can be installed without a compiler by using custom dependency packages we have provided (see Conda pathway above). Where possible, we have made these dependencies optional for those who would prefer not to use the custom packages. They may also be compiled directly as described here.

### Custom precompiled packages

| Dependency | Precompiled Available? | Precompiled Run Req | Build Req | Purpose | 
| --- | --- | --- | --- | --- |
| Python-Javabridge | Yes, via custom package | Python 3.6, Java 8+ | JDK, C compiler| Import proprietary image formats |
| Traits, Pyface, Traitsui | Yes, via Conda (not PyPI) | Python 3.6+ | C compiler, Python dev | GUI |
| SimpleElastix | Yes, via custom package | Python 3.6 | C, C++ compilers | Load medical 3D formats, image regsitration |
| ImageJ/FIJI | Yes, via direct download | Java 8 | n/a | Image stitching |

C compilers by platform:

- Mac and Linux: `gcc`/`clang`
- Windows: Microsoft Visual Studio Build Tools (tested on 2017, 2019) along with these additional components
  - MSVC C++ x64/x86 build tools
  - Windows 10 SDK

Java versions:

- The Conda setup pathway installs JDK 8
- Python-Javabridge uses JDK v8+ (v12+ in Javabridge 1.0.19; see [below](#image-loading) for image loading times and setup troubleshooting with various Java versions)
- ImageJ/Fiji currently supports Java 8 best in our experience

Our custom packages assume an environment with Python 3.6 and Java 8+.

### Additional optional packages

- R for additional stats
- Zstd (fallback to Zip) for compression on servers
- MeshLab for 3D surface clean-up

### SimpleElastix dependency

SimpleElastix is used for loading many 3D image formats (eg `.mhd/.raw` and `.nii`) and registration tasks in MagellanMapper. The library is not currently available in the standard [PyPi](https://pypi.org/). As the buid process is not trivial, we have uploaded binaries to a [third-party PyPi server](https://pypi.fury.io/dd8/).

If you would prefer to build SimpleElastix yourself, we have provided a couple build scripts to ease the build process for the SimpleElastix Python wrapper:

- Mac or Linux: Run the environment setup with `bin/setup_conda.sh -s` to build and install SimpleElastix during setup using the `bin/build_se.sh` script. SimpleElastix can also be built after envrionment setup by running this script within the environment. Building SimpleElastix requires `cmake`, `gcc`, `g++`, and related compiler packages.
- Windows: Run `bin\build_se.bat` within your environment. See above for required Windows compiler components. Note that CMake 3.14 in the MSVC 2019 build tools package has not worked for us, but CMake 3.15 from the official download site has worked.

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

API documentation can be built using Sphinx. To install Sphinx, run:

```
pip install sphinx
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

```
ERROR: In ..\Rendering\OpenGL2\vtkWin32OpenGLRenderWindow.cxx, line 686
vtkWin32OpenGLRenderWindow (00000252735E3450): failed to get wglChoosePixelFormatARB

ERROR: In ..\Rendering\OpenGL2\vtkWin32OpenGLRenderWindow.cxx, line 765
vtkWin32OpenGLRenderWindow (00000252735E3450): failed to get valid pixel format.

ERROR: In ..\Rendering\OpenGL2\vtkOpenGLRenderWindow.cxx, line 741
vtkWin32OpenGLRenderWindow (00000252735E3450): GLEW could not be initialized: Missing GL version
```

Windows Virtual Machines may require installation of Mesa for OpenGL support. Pre-built Mesa software is available from [mesa-dist-win](https://github.com/pal1000/mesa-dist-win). The Desktop OpenGL drivers can be installed system-wide.

```
ERROR: In ../Rendering/OpenGL2/vtkOpenGLRenderWindow.cxx, line 754
vtkXOpenGLRenderWindow (0x7fffc7718590): Unable to find a valid OpenGL 3.2 or later implementation.
```

This error may occur in WSL even with an X11 server open. Workarounds include:

- Run `export LIBGL_ALWAYS_INDIRECT=0` and/or `export QT_X11_NO_MITSHM=1`
- Or simply run the software from within MobaXTerm

Additional errors:

- An error with VTK has prevented display of 3D images at least as of VTK 8.1.2 on RHEL 7.5, though the same VTK version works on Ubuntu 18.04
- PyQt5 5.12 may give an `FT_Get_Font_Format` error, requiring manual downgrade to 5.11.3, though 5.12 works on Ubuntu 18.04

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
