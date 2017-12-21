# BigStitcher script via ImageJ/Fiji
# Author: David Young, 2017
"""Stitches multi-series imaginge files (eg CZI format) through the Fiji
BigStitcher plugin.

Attributes:
    in_file: Path to the image input file.
"""

from ij import IJ
import os

#@String in_file
#@int compute_overlap
#@int write_fused

def perform_task(task, options):
    print("Running {}".format(task))
    print(options)
    IJ.run(task, options)

out_dir = os.path.dirname(in_file)
print("in_file: {}".format(in_file))
print("out_dir: {}".format(out_dir))
print("compute_overlap: {}".format(type(compute_overlap)))
if write_fused == 1:
    fusion_method = "Linear Blending"
else:
    fusion_method = "Do not fuse images (only write TileConfiguration)"
print("fusion method: {}".format(fusion_method))

options = (
    "select=" + in_file + " "
    "selection=[Pick brightest]"
)
#perform_task("Select Illuminations", options);

options = (
    "select=" + in_file + " "
    "process_angle=[All angles] "
    "process_channel=[All channels] "
    "process_illumination=[All illuminations] "
    "process_tile=[Multiple tiles (Select from List)] "
    "process_timepoint=[All Timepoints] "
    "tile_0 tile_1 method=[Phase Correlation] "
    "downsample_in_x=4 "
    "downsample_in_y=4 "
    "downsample_in_z=1"
)
#perform_task("Calculate pairwise shifts ...", options)

options = (
    "select=" + in_file + " "
    "min_r=0 "
    "max_r=1 "
    "max_shift_in_x=0 "
    "max_shift_in_y=0 "
    "max_shift_in_z=0 "
    "max_displacement=0"
)
#perform_task("Filter pairwise shifts ...", options);

options = (
    "select=" + in_file + " "
    "process_angle=[All angles] "
    "process_channel=[All channels] "
    "process_illumination=[All illuminations] "
    "process_tile=[Multiple tiles (Select from List)] "
    #"process_tile=[All tiles] "
    "process_timepoint=[All Timepoints] "
    "tile_0 tile_1 "
    "how_to_treat_timepoints=[treat individually] "
    "how_to_treat_channels=group "
    "how_to_treat_illuminations=group "
    "how_to_treat_angles=[treat individually] "
    "how_to_treat_tiles=compare "
    "relative=2.500 "
    "absolute=3.500 "
    "global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles] "
    "fix_group_0-0 fix_group_0-1"
)
perform_task("Optimize globally and apply shifts ...", options);

options = (
    "select=" + in_file + " "
    "process_angle=[All angles] "
    "process_channel=[All channels] "
    "process_illumination=[All illuminations] "
    "process_tile=[Multiple tiles (Select from List)] "
    "process_timepoint=[All Timepoints] "
    "tile_0 tile_1 "
    "bounding_box=[Currently Selected Views (3648x1921x4385px)] "
    "downsampling=5 pixel_type=[16-bit unsigned integer] "
    "interpolation=[Linear Interpolation] "
    "image=Virtual blend preserve_original "
    "produce=[All views together] "
    "fused_image=[Save as (compressed) TIFF stacks] "
    "output_file_directory=" + out_dir
)
#perform_task("Fuse dataset ...", options);