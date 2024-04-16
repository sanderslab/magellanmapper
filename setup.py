# MagellanMapper setup script
# Author: David Young, 2017, 2023

import setuptools

# optional dependencies to import files via BioFormats, which req Java 8+
_EXTRAS_IMPORT = [
    # Javabridge pre-built on Java 8
    "javabridge==1.0.19.post4+gbebed64 ; python_version < '3.8'",
    "javabridge==1.0.19.post9+gc8c12b4 ; python_version >= '3.8'",
    
    # Python-Bioformats built to depend on the vanilla (non-forked) Javabridge
    "python-bioformats==4.0.5.post2+g51eb88a ; python_version < '3.8'",
    "python-bioformats==4.0.7.post5+g52309d1 ; python_version >= '3.8'",
]

# optional dependencies for Pandas
_EXTRAS_PANDAS = [
    "openpyxl",  # export to Excel files
    "jinja2",  # style output
]

# optional dependencies for AWS interaction
_EXTRAS_AWS = ["boto3", "awscli"]

# optional dependencies to build API documentation
_EXTRAS_DOCS = [
    "sphinx",
    "sphinx-autodoc-typehints",
    "myst-parser",
    "furo",  # theme
]

# optional dependencies for Jupyter notebooks
_EXTRAS_JUPYTER = ["jupyterlab", "bash_kernel"]

# optional dependencies for classification
_EXTRAS_CLASSIFER = ["tensorflow"]

# optional dependencies for main GUI; note that this group is not necessary
# for the Matplotlib-based viewers (eg ROI Editor, Atlas Editor)
_EXTRAS_GUI = [
    # backend error with 5.15.8
    "PyQt5 != 5.15.8",
    "pyface",
    "traitsui",
]

#: Optional dependencies for the 3D viewer.
_EXTRAS_3D = [
    "mayavi",
    # WORKAROUND: error in VTK 9.3.0 with Mayavi 4.8.1
    "vtk < 9.3.0",
]

#: Optional pre-built SimpleITK with Elastix for image I/O and registration.
_EXTRAS_SIMPLEITK = [
    "simpleitk==2.0.2rc2.dev785+g8ac4f ; python_version < '3.8'",
    "simpleitk==2.3.0.dev117+g0640d ; python_version >= '3.8'",
]

#: Optional ITK and Elastix for image I/O and registration.
#: `itk` not included since it does not load properly when installed with rest
#: of dependencies. `itk-elastix` also installs a later `itk` version.
_EXTRAS_ITK = [
    "itk-elastix",
]


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
    "version": "1.6b5",
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
        "pandas",
        "PyYAML",
        "appdirs",
        # part of stdlib in Python >= 3.8
        "importlib-metadata >= 1.0 ; python_version < '3.8'",
        "tifffile",
        # required with tifffile >= 2022.7.28
        "imagecodecs",
        # part of stdlib in Python >= 3.7
        "dataclasses ; python_version < '3.7'",
        "bg-atlasapi",
        "typing_extensions",
    ],
    "extras_require": {
        "import": _EXTRAS_IMPORT,
        "aws": _EXTRAS_AWS,
        "pandas_plus": _EXTRAS_PANDAS,
        "docs": _EXTRAS_DOCS,
        "jupyter": _EXTRAS_JUPYTER,
        "classifier": _EXTRAS_CLASSIFER,
        "gui": _EXTRAS_GUI,
        "3d": _EXTRAS_3D,
        "itk": _EXTRAS_ITK,
        "simplitk": _EXTRAS_SIMPLEITK,
        
        # dependencies for most common tasks
        "most": [
            "matplotlib_scalebar",
            "pyamg",  # for Random-Walker segmentation "cg_mg" mode
            *_EXTRAS_GUI,
            *_EXTRAS_ITK,
            *_EXTRAS_IMPORT,
        ],
        
        # (almost) all optional dependencies
        "all": [
            "matplotlib_scalebar",
            "pyamg",  # for Random-Walker segmentation "cg_mg" mode
            "seaborn",  # for Seaborn-based plots
            "scikit-learn",
            *_EXTRAS_GUI,
            *_EXTRAS_PANDAS,
            *_EXTRAS_ITK,
            *_EXTRAS_IMPORT,
            *_EXTRAS_AWS,
            *_EXTRAS_JUPYTER,
            *_EXTRAS_3D,
        ]
    },
}


if __name__ == "__main__":
    # perform setup
    setuptools.setup(**config)
