# Stitching script via ImageJ/Fiji
# Author: David Young, 2017
"""Stitches multi-series imaginge files (eg CZI format) through the Fiji
Stitching script.

Attributes:
    in_file: Path to the image input file.
"""

from ij import IJ
import os

#@String in_file
out_dir = os.path.dirname(in_file)
print("in_file: {}".format(in_file))
print("out_dir: {}".format(out_dir))
options = ("type=[Positions from file] "
           "order=[Defined by TileConfiguration] "
           "multi_series_file=" + in_file + " "
           "output_directory=" + out_dir + " "
           "fusion_method=[Linear Blending] "
           "regression_threshold=0.30 "
           "max/avg_displacement_threshold=2.50 "
           "absolute_displacement_threshold=3.50 "
           "compute_overlap "
           "use_virtual_input_images "
           "computation_parameters=[Save memory (but be slower)] "
           "image_output=[Write to disk]")

print(options)
IJ.run("Grid/Collection stitching", options);