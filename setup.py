# MagellanMapper setup script
# Author: David Young, 2017, 2020

import setuptools

# optional dependencies to import files via BioFormats, which req Java 8+
_EXTRAS_IMPORT = [
    # Javabridge pre-built on Java 8
    "javabridge==1.0.19.post4+gbebed64",
    # Python-Bioformats built to depend on the vanilla (non-forked) Javabridge
    "python-bioformats==4.0.5.post2+g51eb88a",
]

# optional dependencies for Pandas
_EXTRAS_PANDAS = [
    "openpyxl",  # export to Excel files
    "jinja2",  # style output
]

# optional dependencies for AWS interaction
_EXTRAS_AWS = ["boto3", "awscli"]

# optional dependencies to build API documentation
_EXTRAS_DOCS = ["sphinx", "sphinx-autodoc-typehints", "myst-parser"]

# optional dependencies for Jupyter notebooks
_EXTRAS_JUPYTER = ["jupyterlab", "bash_kernel"]

# installation configuration
config = {
    "name": "magellanmapper",
    "description": "3D atlas analysis and annotation",
    "long_description": open("README.md").read(),
    "long_description_content_type": "text/markdown",
    "author": "David Young",
    "url": "https://github.com/sanderslab/magellanmapper",
    "author_email": "david@textflex.com",
    "license": "BSD-3",
    "version": "1.6a1",
    "packages": setuptools.find_packages(),
    "scripts": [],
    "python_requires": ">=3.6",
    "entry_points": {
        # gui_scripts doesn't load because of TraitsUI issue #1032
        "console_scripts": ["mm = magmap.io.load_env:launch_magmap"],
    },
    "install_requires": [
        "scikit-image",
        # PlotEditor performance regression with 3.3.0-3.3.1
        "matplotlib != 3.3.0, != 3.3.1",
        "vtk",
        "mayavi",
        "pandas",
        "PyQt5",
        "pyface",
        "traitsui",
        "simpleitk==2.0.2rc2.dev785+g8ac4f",  # pre-built SimpleElastix
        "PyYAML",
        "appdirs",
        # part of stdlib in Python >= 3.8
        "importlib-metadata >= 1.0 ; python_version < '3.8'",
        "tifffile",
        # required with tifffile >= 2022.7.28
        "imagecodecs",
        # part of stdlib in Python >= 3.7
        "dataclasses ; python_version < '3.7'",
        # BrainGlobe dependencies for access to cloud-hosted atlases
        "bg-atlasapi @ https://github.com/brainglobe/bg-atlasapi/archive/refs/heads/master.zip",
    ],
    "extras_require": {
        "import": _EXTRAS_IMPORT,
        "aws": _EXTRAS_AWS,
        "pandas_plus": _EXTRAS_PANDAS,
        "docs": _EXTRAS_DOCS,
        "jupyter": _EXTRAS_JUPYTER,
        
        # dependencies for most common tasks
        "most": [
            "matplotlib_scalebar",
            "pyamg",  # for Random-Walker segmentation "cg_mg" mode
            *_EXTRAS_IMPORT,
        ],
        
        # (almost) all optional dependencies
        "all": [
            "matplotlib_scalebar",
            "pyamg",  # for Random-Walker segmentation "cg_mg" mode
            "seaborn",  # for Seaborn-based plots
            "scikit-learn",
            *_EXTRAS_PANDAS,
            *_EXTRAS_IMPORT,
            *_EXTRAS_AWS,
            *_EXTRAS_JUPYTER,
        ]
    },
}


if __name__ == "__main__":
    # perform setup
    setuptools.setup(**config)
