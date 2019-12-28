# MagellanMapper setup script
# Author: David Young, 2017, 2019

import setuptools

# optional dependencies to import files via BioFormats, which requires 
# a Java SDK installation
_EXTRAS_IMPORT = ["python-bioformats==1.1.0"]
_EXTRAS_AWS = ["boto3", "awscli"]

# installation configuration
config = {
    "name": "magellanmapper", 
    "description": "3D atlas analysis and annotation",
    "author": "David Young",
    "url": "URL",
    "author_email": "david@textflex.com",
    "version": "1.1.0",
    "packages": setuptools.find_packages(),
    "scripts": [], 
    "python_requires": ">=3.6",  # TODO: consider testing on earlier versions
    "install_requires": [
        "scikit-image", 
        "matplotlib", 
        "mayavi", 
        "pandas", 
        "PyQt5", 
        "scikit-learn",
        "simpleitk==1.2.0rc2.dev1162+g2a79d",  # pre-built SimpleElastix
    ], 
    "extras_require": {
        "import": _EXTRAS_IMPORT, 
        "aws": _EXTRAS_AWS, 
        "all": [
            "matplotlib_scalebar", 
            "pyamg", 
            *_EXTRAS_IMPORT,  
            *_EXTRAS_AWS, 
        ]
    }, 
}

# perform setup
setuptools.setup(**config)
