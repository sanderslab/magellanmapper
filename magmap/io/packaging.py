# Utilities for packaging
"""Provide utilities for packaging the application."""

import os
import pkg_resources
import importlib_resources


def get_pkg_egg(name, prefix=None):
    """Get package egg-info path.

    Args:
        name (str): Package name.
        prefix (str): Start of output path; defaults to None.

    Returns:
        str, str: The egg-info path and output path for the given package.

    """
    distrib = pkg_resources.get_distribution(name)
    egg_name = os.path.basename(distrib.egg_info)
    if prefix:
        egg_name = os.path.join(prefix, egg_name)
    paths = (distrib.egg_info, egg_name)
    print("Adding package egg-info path:", paths)
    return paths


def get_pkg_path(name, prefix=None):
    """Get path to the installed package.

    Args:
        name (str): Package name.
        prefix (str): Start of output path; defaults to None.

    Returns:
        str, str: The package path and output path for the given package.

    """
    pkg_dir = name
    for entry in importlib_resources.files(name).iterdir():
        pkg_dir = os.path.dirname(entry)
        break
    out_path = os.path.basename(pkg_dir)
    if prefix:
        out_path = os.path.join(prefix, out_path)
    paths = (pkg_dir, out_path)
    print("Adding package path:", paths)
    return paths
