# -*- mode: python ; coding: utf-8 -*-
# Setup specification file for Pyinstaller
"""Pyinstaller specification for freezing MagellanMapper and its environment.

Originally auto-generated by Pyinstaller, modified here to include
modules and packages not detected but required for MagellanMapper.

"""

from magmap.io import packaging
from magmap.settings import config
import setup as mag_setup

block_cipher = None

a = Analysis(
    ["../run.py"],
    pathex=[],
    binaries=[],
    datas=[
        *[packaging.get_pkg_path(p)
          for p in ("mayavi", "pyface", "traitsui", "tvtk")],
        *[packaging.get_pkg_egg(p)
          for p in ("mayavi", "pyface", "traitsui")],
    ],
    hiddenimports=[
        "sklearn.utils._weight_vector",
        "pyface.ui.qt4.clipboard",
        "pyface.ui.qt4.image_resource",
        "pyface.ui.qt4.resource_manager",
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
    name=mag_setup.config["name"],
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="run")
app = BUNDLE(
    coll,
    name="{}.app".format(config.APP_NAME),
    icon=None,
    bundle_identifier=None,
    info_plist={
        "NSRequiresAquaSystemAppearance": False,
    })
