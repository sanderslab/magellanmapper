# -*- mode: python ; coding: utf-8 -*-
# Setup specification file for Pyinstaller
"""Pyinstaller specification for freezing MagellanMapper and its environment.

Originally auto-generated by Pyinstaller, modified here to include
modules and packages not detected but required for MagellanMapper.

"""

import os
import pathlib
import platform

from magmap.io import packaging
from magmap.settings import config

block_cipher = None

# WORKAROUND: PyQt5 as of v5.15.4 gives a segmentation fault when the "Qt5"
# folder is not present; even an empty folder bypasses this error, but a stub
# must be added here for Pyinstaller to include the file
path_qt5 = pathlib.Path("build") / "Qt5"
path_qt5.mkdir(parents=True, exist_ok=True)
(path_qt5 / "stub").touch(exist_ok=True)

# get path to the Java Runtime Environment extracted by jlink from environment
# variable, defaulting to a relative path designated by platform to accommodate
# JREs across platforms
path_jre = os.getenv("JRE_PATH")
if not path_jre:
    path_jre = str(pathlib.Path(
        "..") / ".." / "JREs" / f"jre_{platform.system().lower()}")
print("Using JRE from:", path_jre)

a = Analysis(
    ["../run.py"],
    pathex=[],
    binaries=[],
    datas=[
        # app resources
        ("../images", "images"),  # images folder with icons
        ("../LICENSE.txt", "."),
        
        # add full package folders since they contain many modules that
        # are dynamically discovered or otherwise not found by Pyinstaller
        *[packaging.get_pkg_path(p) for p in (
            "mayavi",
            "pyface",
            "traitsui",
            "tvtk",
            "bioformats",
            "javabridge",
        )],
        
        # add egg-info folders required for these packages' entry points
        *[packaging.get_pkg_egg(p) for p in (
            "mayavi",
            "pyface",
            "traitsui",
        )],
        
        # workaround for error when folder is missing
        (path_qt5.resolve(), pathlib.Path("Pyqt5") / "Qt5"),
        
        # JRE distributable
        (path_jre, "jre"),
    ],
    hiddenimports=[
        "sklearn.utils._weight_vector",
        "traits.util.clean_strings",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=config.APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon="../images/magmap.ico")

a.binaries = a.binaries - [
    # remove Java virtual machine library that takes precedence over bundled
    # JRE or any Java home setting on Windows, preventing JVM initialization
    ("jvm.dll", None, None),
    
    # may conflict with corresponding library on newer platforms for VTK
    ("libstdc++.so.6", None, None),
]

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=config.APP_NAME)
app = BUNDLE(
    coll,
    name="{}.app".format(config.APP_NAME),
    icon="../images/magmap.icns",
    bundle_identifier=None,
    info_plist={
        "NSRequiresAquaSystemAppearance": False,
        "LSEnvironment": {
            # "./" at the start will be translated to the .app root directory
            # since the working directory is "/" rather than the app root
            "JAVA_HOME": "../Resources/jre"
        }
    })