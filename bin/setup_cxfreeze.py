# Setup for freezing an environment for distribution via cx_Freeze

import os
import pkg_resources

from cx_Freeze import setup, Executable

import setup as mag_setup


def get_pkg_dirs(name):
    """Get package directory names
    
    Args:
        name (name): Package name . 

    Returns:
        tuple(str, str): Tuples in the form,
        ``(egg-info-path, output-path)`, for the given package.

    """
    distrib = pkg_resources.get_distribution(name)
    paths = (distrib.egg_info,
             os.path.join("lib", os.path.basename(distrib.egg_info)))
    print("Adding package path:", paths)
    return paths


build_exe_options = {
    "packages": [
        "pyface.ui.qt4",
        "tvtk.pyface.ui.qt4",
        "skimage.feature._orb_descriptor_positions",
    ],
    "include_files": [
        get_pkg_dirs(p) for p in ("mayavi", "pyface", "traitsui")],
    "excludes": "Tkinter",
}

executables = [
    Executable('run.py')
]

setup(
    name=mag_setup.config["name"],
    version=mag_setup.config["version"],
    description=mag_setup.config["description"],
    options={"build_exe": build_exe_options},
    executables=executables,
)
