#!/usr/bin/env python
"""Pipeline to import a series of RAW files from mesoSPIM and export to TIFs.

Designed to import RAW files output by the mesoSPIM microscope and export
to a TIF format that the BigStitcher ImageJ/FIJI plugin can stitch together.
Assumes that files are in the format, `<chl>_<tile-coords>.raw` in a single
folder.

"""

import glob
import os
import pathlib
import re

from magmap.io import cli, np_io
from magmap.settings import config


def main():
    # get RAW file paths
    paths = sorted(glob.glob(f"{config.filename}*.raw"))
    print("paths", paths)
    paths = paths
    tiles = []
    chls = []
    
    # set up output paths
    prefix_orig = config.prefix
    prefix_npy = prefix_orig
    if not os.path.basename(prefix_npy):
        # ensure a basename is present to allow NPY file to be opened
        prefix_npy += "export"
    
    for path in paths:
        # import RAW file to NPY
        
        # parse mesoSPIM metadata file
        meta = {}
        with open(f"{path}_meta.txt") as meta_file:
            for line in meta_file:
                # metadata assumed to be in the format: `[key] = val`
                m = re.match(r"^(?P<key>\[.*\]) (?P<val>.*)$", line)
                if m:
                    meta[m.group("key").strip("[]")] = m.group("val")
        config.meta_dict[config.MetaKeys.SHAPE] = [
            1, int(meta["z_planes"]), int(meta["y_pixels"]),
            int(meta["x_pixels"]), 1]
        config.meta_dict[config.MetaKeys.DTYPE] = "uint16"
        config.meta_dict[config.MetaKeys.RESOLUTIONS] = [
            float(meta["z_stepsize"]), float(meta["Pixelsize in um"]),
            float(meta["Pixelsize in um"])]
        # config.meta_dict[config.MetaKeys.MAGNIFICATION] = 1
        config.meta_dict[config.MetaKeys.ZOOM] = float(meta["Zoom"].strip("x"))
        
        # import RAW file, overwriting the same NPY file
        config.filename = path
        config.prefix = prefix_npy
        # config.filename = str(f"{pathlib.Path(path).parent}_out/export")
        cli.process_proc_tasks()

        # construct output filename, assuming input filename is in the format,
        # `<chl>_<tile-coords>.raw` and output is to tile_<t>_ch_<c>.tif
        path_split = os.path.basename(path).split("_", 1)
        metas = []
        for i, (meta_str, meta_list) in enumerate(zip(
                path_split, (chls, tiles))):
            if meta_str not in meta_list:
                meta_list.append(meta_str)
            metas.append(meta_list.index(meta_str))
        filename_out = f"tile_{metas[1]}_ch_{metas[0]}"
        
        # export imported file to TIF file
        print(f"Exporting file from '{path}' to '{filename_out}'")
        config.prefix = prefix_orig
        np_io.write_tif(
            config.image5d,
            pathlib.Path(path).parent / filename_out, imagej=True)


if __name__ == "__main__":
    cli.main(process_args_only=True)
    main()
