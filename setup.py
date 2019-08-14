# Clrbrain setup script
# Author: David Young, 2017, 2019

import setuptools

# optional dependencies to import files via BioFormats, which requires 
# a Java SDK installation
_EXTRAS_IMPORT = ["python-bioformats==1.1.0"]

# installation configuration
config = {
    "name": "clrbrain", 
    "description": "3D atlas analysis and annotation",
    "author": "David Young",
    "url": "URL",
    "author_email": "david@textflex.com",
    "version": "0.9.4",
    "packages": setuptools.find_packages(),
    "scripts": [], 
    "python_requires": ">=3",  # TODO: may need to increase; tested on >=3.6
    "install_requires": [
        "scikit-image", 
        "matplotlib", 
        "mayavi", 
        "pandas", 
        "PyQt5", 
        "simpleitk", 
    ], 
    "extras_require": {
        "import": _EXTRAS_IMPORT, 
        "all": [
            "matplotlib_scalebar", 
            "pyamg", 
            *_EXTRAS_IMPORT,  
        ]
    }
}

# perform setup
setuptools.setup(**config)
