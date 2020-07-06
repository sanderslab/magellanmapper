# MagellanMapper setup script
# Author: David Young, 2017, 2020

import setuptools

# optional dependencies to import files via BioFormats, which req Java 8+
_EXTRAS_IMPORT = [
    # pre-built Javabridge with Python 3.6, Java 8
    "javabridge==1.0.18.post21+g4526f53",
    "python-bioformats==1.1.0",
]
_EXTRAS_AWS = ["boto3", "awscli"]

# installation configuration
config = {
    "name": "magellanmapper", 
    "description": "3D atlas analysis and annotation",
    "author": "David Young",
    "url": "https://github.com/sanderslab/magellanmapper",
    "author_email": "david@textflex.com",
    "version": "1.3.3",
    "packages": setuptools.find_packages(),
    "scripts": [], 
    "python_requires": ">=3.6",  # TODO: consider testing on earlier versions
    "install_requires": [
        "scikit-image", 
        "matplotlib",
        "vtk<9.0.0",  # Mayavi 4.7.1 is not compatible with VTK 9
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
            "openpyxl",  # for Pandas export to Excel files
            *_EXTRAS_IMPORT,  
            *_EXTRAS_AWS, 
        ]
    }, 
}

# perform setup
setuptools.setup(**config)
