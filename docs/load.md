# Load MagellanMapper and Image Files

## Start MagellanMapper

Once you've [installed](install.md) MagellanMapper, activate its environment and launch the app:

```shell
conda activate mag
mm
```

Replace the `conda` command if you are using another type of virtual environment.

The graphical user interface (GUI) should open, allowing you to load images using the browsing controls.

## Import images

Importing image files allows them to be read on-the-fly for lower memory requirements. The "Import" panel in the GUI allows you to select files to import, specify metadata, and load them into a NumPy (NPY) format.

TODO: add details

## Load TIF files directly

v1.6 added *EXPERIMENTAL* support for loading TIF images similarly to NPY images but without requiring prior import. Loading a `.tif` file in the "ROI > Image path" controls will attempt to load the image directly.

## Using the CLI

See our [Jupyter Notebook](https://github.com/sanderslab/magellanmapper/blob/master/bin/sample_cmds_bash.ipynb) for examples of loading and viewing various types of images in MagellanMapper using the command-line interface (CLI).
