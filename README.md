# MagellanMapper

MagellanMapper is a graphical imaging informatics suite for 3D reconstruction and automated analysis of whole specimens and atlases. Its design philosophy is to make the raw 3D images as accessible as possible, simplify annotation from nuclei to atlases, and scale from the laptop or desktop to the cloud in cross-platform environments.

![ROI Editor and Atlas Editor screenshots](https://user-images.githubusercontent.com/1258953/83934132-f699aa00-a7e0-11ea-932c-0e58366d5061.png)

<a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7781073/"><img align="right" width="200" src="https://user-images.githubusercontent.com/1258953/179440433-0326c4d5-9a9b-4bae-92c7-d09416375bc5.png" title="Current Protocols cover image"></a>


## Quick Reference

- **NEW**: [Full docs are now on ReadTheDocs!](https://magellanmapper.readthedocs.io/en/latest/)
- [Installation](#installation) (more [details](docs/install.md))
- [Intro to running MagellanMapper](#run-magellanmapper)
- [Using the viewers](docs/viewers.md)
- [Command-line interface](docs/cli.md)
- [Configuration and settings](docs/settings.md)
- [Publications](#related-publications-and-datasets)


## Installation

If you use Conda (available [here](https://docs.conda.io/en/latest/miniconda.html)), you can install MagellanMapper into a new environment:

```shell
conda env create -n mag -f https://raw.githubusercontent.com/sanderslab/magellanmapper/master/envs/environment_rel.yml
```

To run in this new environment named `mag`:

```shell
conda activate mag
mm
```

Or install using Pip (virtual environment [recommended](https://realpython.com/python-virtual-environments-a-primer/)):

```shell
pip install magellanmapper[most] --extra-index-url https://pypi.fury.io/dd8/
```

The extra index accesses a few [customized dependencies](docs/install.md#custom-packages) for MagellanMapper.

Conda installs Java to import proprietary image file formats, which can also be installed separately when using Pip (eg from [here](https://www.azul.com/downloads/?package=jdk)).

### Using our installer scripts

We have provided scripts to take care of installing Miniconda (if necessary), creating an environment, and installing MagellanMapper, without requiring command-line/terminal experience. Alternatively, see [here](docs/install.md#option-2-install-through-venvpip) for our Pip+Venv installer scripts.

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

### Using the installer

**Note**: We're in the process of determining how useful these are for the community. If you've liked them, please let us know! (And feedback welcome if you've run into any issues with them.)

The easiest way to install MagellanMapper is using one of the [installers](https://github.com/sanderslab/magellanmapper/releases) now available for Windows, macOS, and Linux.

To run:
- **Mac**: run the MagellanMapper app
- **Windows**: in the Start Menu, go to "MagallanMapper v.x.y.z" and run "MagellanMapper"
- **Linux**: in a file browser, double-click on `MagellanMapper/MagellanMapper`

On Windows and Mac, you can also use "Open with" on supported file types (eg `.npy`, `.mhd`, `.nii.gz`) to open them in MagellanMapper.

## Using MagellanMapper

MagellanMapper consists of a graphical user interface (GUI), command-line interface (CLI), and application programming interface (API) for Python programmatic access. See the [GUI docs](docs/viewers.md) for graphical usage and the [CLI docs](docs/cli.md) for scripting.

For automated tasks, [`sample_cmds.sh`](bin/sample_cmds.sh) is a script that shows examples of common commands. You can also use [`pipelines.sh`](bin/pipelines.sh), a script to run many automated pipelines within MagellanMapper, such as whole volume nuclei detection and image transposition. See [Settings](docs/settings.md) for how to customize parameters for your image analysis.

### Image file import

In the "Import" tab, you can select files, view and update metadata, and import the files.

Medical imaging formats such as `.mha` (or `.mhd/.raw`) and `.nii` (or `.nii.gz`) can be opened with the SimpleITK/SimpleElastix Library and do not require separate import. Standard image formats such as TIFF or proprietary microscopy formats such as CZI can be imported by MagellanMapper into an industry standard Numpy format, which allows on-the-fly loading to reduce memory requirements and initial loading time.

### Sample 3D data

To try out functions with sample images, download any of these practice files:

- [Sample region of nuclei at 4x (`sample_region.zip`)](https://github.com/sanderslab/magellanmapper/releases/download/v1.1.3/sample_region.zip)
- [Sample downsampled tissue cleared whole brain (`sample_brain.zip`)](https://github.com/sanderslab/magellanmapper/releases/download/v1.1.3/sample_brain.zip)
- [Allen Developing Mouse Brain Atlas E18.5 (`ADMBA-E18pt5.zip`)](https://github.com/sanderslab/magellanmapper/releases/download/v1.1.3/ADMBA-E18pt5.zip)

### Related publications and datasets

- For more information on the methods used for 3D atlas construction, please see: https://elifesciences.org/articles/61408
- For step-by-step instructions on using v1.3.x of the software, please see: https://currentprotocols.onlinelibrary.wiley.com/doi/abs/10.1002/cpns.104 (now [open access on PubMed](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7781073/)!); see [ReadTheDocs](https://magellanmapper.readthedocs.io/en/latest/) for ongoing updates
- The 3D reconstructed versions of the Allen Developing Mouse Brain Atlas: https://search.kg.ebrains.eu/instances/Project/b8a8e2d3-4787-45f2-b010-589948c33f20
- Sample wild-type whole mouse brains at age P0: https://search.kg.ebrains.eu/instances/Dataset/2423e103-35e9-40cf-ab0c-0e3d08d24d5a

Licensed under the open-source [BSD-3 license](LICENSE.txt)

Author: David Young, 2017, 2022, Stephan Sanders Lab
