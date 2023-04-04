# MagellanMapper

MagellanMapper is a graphical imaging informatics suite for 3D reconstruction and automated analysis of whole specimens and atlases. Its design philosophy is to make the raw 3D images as accessible as possible, simplify annotation from nuclei to atlases, and scale from the laptop or desktop to the cloud in cross-platform environments.

![ROI Editor and Atlas Editor screenshots](https://user-images.githubusercontent.com/1258953/195321971-955fc46a-f44d-4282-8c78-21708ebaeef1.png)

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

To run in this new environment named `mag`<sup>*</sup>:

```shell
conda activate mag
mm
```

<sup>*</sup> `mm` was added in v1.6.0 to launch from installed packages. When installed [from source](docs/install.md#developer-installs), `python run.py` still works!

Or install using Pip (virtual environment [recommended](https://realpython.com/python-virtual-environments-a-primer/)):

```shell
pip install "magellanmapper[most]" --extra-index-url https://pypi.fury.io/dd8/
```

The extra index accesses a few [customized dependencies](docs/install.md#custom-packages) for MagellanMapper.

Conda installs Java to import proprietary image file formats, which can also be installed separately when using Pip (eg from [here](https://www.azul.com/downloads/?package=jdk)).

### More ways to install and run

- [Developer installs](docs/install.md#developer-installs) from source code
- [Installer scripts](docs/install.md#installer-scripts), which install Miniconda and MagellanMapper without requiring command-line usage
- [Installer packages](docs/install.md#installer-packages), for point-and-click installation

## Using MagellanMapper

MagellanMapper consists of a graphical user interface (GUI), command-line interface (CLI), and application programming interface (API) for Python programmatic access. See the [GUI docs](docs/viewers.md) for graphical usage and the [CLI docs](docs/cli.md) for scripting.

For automated tasks, see the [`sample_cmds_bash.ipynb`](bin/sample_cmds_bash.ipynb) Jupyter Notebook (or the older [`sample_cmds.sh`](bin/sample_cmds.sh) script) that shows examples of running the CLI and exploring images in the GUI. See ReadTheDocs for more details, including [viewer shortcuts](docs/viewers.md) and customizing [settings](docs/settings.md) for your image analysis.

Have a question? Found a bug? Want a feature? Please [ask](https://github.com/sanderslab/magellanmapper/issues)!

### Image file import

Large images or proprietary microscopy formats such as CZI can be imported by MagellanMapper into NumPy format, which allows on-the-fly loading to reduce memory requirements and initial loading time. In the "Import" tab, you can select files, view and update metadata, and import the files.

Medical imaging formats such as `.mha` (or `.mhd/.raw`) and `.nii` (or `.nii.gz`) can be opened with the SimpleITK/SimpleElastix library and do not require separate import. 

### Sample 3D data

To try out functions with sample images, download any of these files:

- [Sample region of nuclei at 4x (`sample_region.zip`)](https://github.com/sanderslab/magellanmapper/releases/download/v1.1.3/sample_region.zip)
- [Sample downsampled tissue cleared whole brain (`sample_brain.zip`)](https://github.com/sanderslab/magellanmapper/releases/download/v1.1.3/sample_brain.zip)
- [Allen Developing Mouse Brain Atlas E18.5 (`ADMBA-E18pt5.zip`)](https://github.com/sanderslab/magellanmapper/releases/download/v1.1.3/ADMBA-E18pt5.zip)

## Related publications and datasets

- For more information on the methods used for 3D atlas construction, please see: https://elifesciences.org/articles/61408
- For step-by-step instructions on using v1.3.x of the software, please see: https://currentprotocols.onlinelibrary.wiley.com/doi/abs/10.1002/cpns.104 (now [open access on PubMed](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7781073/)!); see [ReadTheDocs](https://magellanmapper.readthedocs.io/en/latest/) for ongoing updates
- The 3D reconstructed versions of the Allen Developing Mouse Brain Atlas: https://search.kg.ebrains.eu/instances/Project/b8a8e2d3-4787-45f2-b010-589948c33f20
- Sample wild-type whole mouse brains at age P0: https://search.kg.ebrains.eu/instances/Dataset/2423e103-35e9-40cf-ab0c-0e3d08d24d5a

Licensed under the open-source [BSD-3 license](LICENSE.txt)

Author: David Young, 2017, 2023, Stephan Sanders Lab
