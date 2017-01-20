try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    "name" = "clrbrain"
    "description": "3D brain imaging analysis",
    "author": "David Young",
    "url": "URL",
    "author_email": "david@textflex.com",
    "version": "0.1",
    "packages": ["clrbrain"],
    "scripts": []
}

setup(**config)