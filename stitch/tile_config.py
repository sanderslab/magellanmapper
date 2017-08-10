# Build tile configuration for ImageJ/Fiji Stitching pluing
# Author: David Young, 2017
"""Builds the tile grid layout for the ImageJ/Fiji Stitching plugin.

Attributes:
    OPTIONS_DIRECTIONALITY: direction of travel from one row to the next,
        where "uni" = unidirectional, "bi" = bidirectional
    OPTIONS_START_DIRECTION: starting direction of travel along the 1st
        row, where "right" = moving toward the right, "left" = moving
        toward the left
    TILE_CONFIG_FILE = default tile configuration name
"""

import os
import argparse

OPTIONS_DIRECTIONALITY = ["uni", "bi"]
OPTIONS_START_DIRECTION = ["right", "left"]
TILE_CONFIG_FILE = "TileConfiguration.txt"

def main():
    parser = argparse.ArgumentParser(
        description="Configure tiling for image stitching")
    parser.add_argument("--img")
    parser.add_argument("--target_dir")
    parser.add_argument("--rows", type=int)
    parser.add_argument("--cols", type=int)
    parser.add_argument("--directionality")
    parser.add_argument("--start_direction")
    parser.add_argument("--size")
    parser.add_argument("--overlap", type=float)
    args = parser.parse_args()
    
    img = args.img
    grid_x_size = args.cols
    grid_y_size = args.rows
    overlap = args.overlap
    size = args.size.split(",")
    size = [float(i) for i in size]
    directionality = args.directionality
    start_direction = args.start_direction
    target_dir = args.target_dir
    
    with open(os.path.join(target_dir, TILE_CONFIG_FILE), "w") as f:
        f.write("dim = {}\n".format(len(size)))
        tiles = grid_x_size * grid_y_size
        for i in range(tiles):
            grid_x = i % grid_x_size
            grid_y = i // grid_x_size
            row_alt = grid_y
            if start_direction == OPTIONS_START_DIRECTION[0]:
                row_alt += 1
            if directionality == OPTIONS_DIRECTIONALITY[1] and row_alt % 2 == 0:
                grid_x = grid_x_size - grid_x - 1
            frac = abs(1 - overlap)
            offset_x = size[0] * grid_x * frac
            offset_y = size[1] * grid_y * frac
            print("offset_x: {}, offset_y: {}".format(offset_x, offset_y))
            f.write("{}; ; ({}, {}, {})\n".format(img, offset_x, offset_y, 0.0))

if __name__ == "__main__":
    print("Starting tile configurator...")
    main()
