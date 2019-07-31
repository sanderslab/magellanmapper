# Clrbrain setup script
# Author: David Young, 2017, 2019

try:
    import setuptools
except ImportError:
    import distutils.core

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
    # TODO: consider adding pythong-bioformats, simpleitk (SimpleElastix fork)
    "install_requires": [
        "scikit-image", 
        "matplotlib", 
        "mayavi", 
        "pandas", 
    ]
}

setuptools.setup(**config)
