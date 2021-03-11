# MagellanMapper setup script
# Author: David Young, 2017, 2020

import os
import pkg_resources
import setuptools

# optional dependencies to import files via BioFormats, which req Java 8+
_EXTRAS_IMPORT = [
    # pre-built Javabridge with Python 3.6, Java 8
    "javabridge==1.0.18.post21+g4526f53",
    "python-bioformats==1.1.0",
]

# optional dependencies for Pandas
_EXTRAS_PANDAS = [
    "openpyxl",  # export to Excel files
    "jinja2",  # style output
]

# optional dependencies for AWS interaction
_EXTRAS_AWS = ["boto3", "awscli"]

# installation configuration
config = {
    "name": "magellanmapper", 
    "description": "3D atlas analysis and annotation",
    "author": "David Young",
    "url": "https://github.com/sanderslab/magellanmapper",
    "author_email": "david@textflex.com",
    "version": "1.4.0",
    "packages": setuptools.find_packages(),
    "scripts": [], 
    "python_requires": ">=3.6",  # may work on earlier versions
    "install_requires": [
        "scikit-image",
        # PlotEditor performance regression with 3.3.0-3.3.1
        "matplotlib != 3.3.0, != 3.3.1",
        "vtk<9.0.0",  # Mayavi 4.7.1 is not compatible with VTK 9
        "mayavi", 
        "pandas", 
        "PyQt5",
        "pyface",
        "traitsui",
        "scikit-learn",
        "simpleitk==1.2.0rc2.dev1162+g2a79d",  # pre-built SimpleElastix
        "PyYAML",
        "appdirs",
    ], 
    "extras_require": {
        "import": _EXTRAS_IMPORT, 
        "aws": _EXTRAS_AWS,
        "pandas_plus": _EXTRAS_PANDAS,
        "all": [
            "matplotlib_scalebar", 
            "pyamg",  # for Random-Walker segmentation "cg_mg" mode
            *_EXTRAS_PANDAS,
            *_EXTRAS_IMPORT,  
            *_EXTRAS_AWS, 
        ]
    }, 
}


def get_pkg_dirs(name, prefix=None):
    """Get package directory names

    Args:
        name (str): Package name.
        prefix (str): Start of output path; defaults to None.

    Returns:
        tuple(str, str): Tuples in the form,
        ``(egg-info-path, output-path)`, for the given package.

    """
    distrib = pkg_resources.get_distribution(name)
    egg_name = os.path.basename(distrib.egg_info)
    if prefix:
        egg_name = os.path.join(prefix, egg_name)
    paths = (distrib.egg_info, egg_name)
    print("Adding package path:", paths)
    return paths


if __name__ == "__main__":
    # perform setup
    setuptools.setup(**config)
