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
}

setuptools.setup(**config)
