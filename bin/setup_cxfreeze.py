# Setup for freezing an environment for distribution via cx_Freeze

from cx_Freeze import setup, Executable

from magmap.io import packaging
import setup as mag_setup


build_exe_options = {
    "packages": [
        "pyface.ui.qt4",
        "tvtk.pyface.ui.qt4",
        "skimage.feature._orb_descriptor_positions",
    ],
    "include_files": [
        packaging.get_pkg_egg(p, "lib") for p
        in ("mayavi", "pyface", "traitsui")],
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
