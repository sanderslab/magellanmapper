# MagellanMapper

MagellanMapper is a graphical imaging informatics suite for 3D reconstruction and automated analysis of whole specimens and atlases. Its design philosophy is to make the raw 3D images as accessible as possible, simplify annotation from nuclei to atlases, and scale from the laptop or desktop to the cloud in cross-platform environments.

![ROI Editor and Atlas Editor screenshots](https://user-images.githubusercontent.com/1258953/83934132-f699aa00-a7e0-11ea-932c-0e58366d5061.png)

<img align="right" width="200" src="https://currentprotocols.onlinelibrary.wiley.com/cms/asset/14209f75-4368-40c9-b83b-1bb866991a8b/cpns76-gra-0001-m.jpg" title="Current Protocols cover image">

Related publications and datasets:
- For more information on the methods used for 3D atlas construction, please see: https://elifesciences.org/articles/61408
- For step-by-step instructions on using the software, please see: https://currentprotocols.onlinelibrary.wiley.com/doi/abs/10.1002/cpns.104 (now [open access on PubMed](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7781073/)!)
- The 3D reconstructed versions of the Allen Developing Mouse Brain Atlas: https://search.kg.ebrains.eu/instances/Project/b8a8e2d3-4787-45f2-b010-589948c33f20
- Sample wild-type whole mouse brains at age P0: https://search.kg.ebrains.eu/instances/Dataset/2423e103-35e9-40cf-ab0c-0e3d08d24d5a


## Quick Reference

- **NEW**: [Full docs are now on ReadTheDocs!](https://magellanmapper.readthedocs.io/en/latest/)
- [Installation](#installation) (more [details](docs/install.md))
- [Intro to running MagellanMapper](#run-magellanmapper)
- [Using the viewers](docs/viewers.md)
- [Command-line interface](docs/cli.md)
- [Configuration and settings](docs/settings.md)


## Installation

### Install a standalone package

Starting with v1.4, standlone [installers](https://github.com/sanderslab/magellanmapper/releases) are now available for Windows, macOS, and Linux!

These packages do not require use of the terminal, and you can skip the installation steps below. As the packages are rough around the edges, please file any [issues](https://github.com/sanderslab/magellanmapper/issues) you may enounter.

Windows users: The installer is not yet signed, meaning that Windows will still give some security warnings. If the Edge browser blocks the download, click the Downloads button -> the `...` button on the right of the file entry -> "Keep" -> "Show more" -> "Keep anyway". In the blue window after opening the file, click "More info" -> "Run anyway" to start the installer.

### Install from source

1. Download MagellanMapper by cloning the git repo (or download the [latest release](https://github.com/sanderslab/magellanmapper/releases/latest)):
```
git clone https://github.com/sanderslab/magellanmapper.git
```
1. Install MagellanMapper using the following script in the `magellanmapper` folder
    - On Mac or Linux: `bin/setup_conda`
    - On Windows: `bin\setup_conda.bat`

- Installation may take up to 5 minutes, depending on internet connection speed.
- The script will also install the Conda package manager if not already installed.
- To update the environment, rerun the appropriate `setup_conda` script above.
- On Mac, it may be necessary to right-click and "Open with" the Terminal app.
- On Linux, it may be necessary to go to "Preferences" in the file browser (eg the Files app), select the "Behavior" tab, and choose to "Run" or "Ask" when executing text files.
- [v1.3.x releases](https://github.com/sanderslab/magellanmapper/releases/tag/v1.3.8) match the [Current Protocols guide](https://currentprotocols.onlinelibrary.wiley.com/doi/abs/10.1002/cpns.104), while v1.4.x introduces small changes to the interface and workflows (see [v1.4 release notes](https://github.com/sanderslab/magellanmapper/blob/master/docs/release/release_v1.4.md) for more details).
- See [Installation](docs/install.md) for more details and install options, including installation Venv+Pip instead of Conda.
  
## Run MagellanMapper

MagellanMapper consists of a graphical user interface (GUI), command-line interface (CLI), and application programming interface (API) for Python programmatic access. Here we introduce using the GUI, and the [CLI docs](docs/cli.md) can be referenced for scripting.

### Using the standalone package

- **Mac**: launch MagellanMapper from LaunchPad, or double-click on the MagellanMapper app
- **Windows**: in the Start Menu, go to "MagallanMapper v.x.y.z" and run "MagellanMapper"
- **Linux**: in a file browser, double-click on `MagellanMapper/MagellanMapper`

On Windows and Mac, you can also use "Open with" on supported file types (eg `.npy`, `.mhd`, `.nii.gz`) to open them in MagellanMapper.

### After installing from source

#### From a file browser

**On Mac or Linux**: Double-click the MagellanMapper icon created during Conda setup. This Unix executable should open with Terminal by default on Mac and after the file browser preference change described above on Linux.

**On Windows**: Run `run.py` through Python.
- It may be necessary to right-click, choose "Open with", and browse to the Conda `pythonw.exe` file to open `run.py`
- If a security warning displays, click on "More info" and "Run anyway" to launch the file

Note that during the first run, there may be a delay of up to several minutes from antivirus scanning for the new Python interpreter location in the new environment. Subsequent launches are typically much faster.

#### From a terminal

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

Licensed under the open-source [BSD-3 license](LICENSE.txt)

Author: David Young, 2017, 2022, Stephan Sanders Lab
