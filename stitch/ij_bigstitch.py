# BigStitcher script via ImageJ/Fiji
# Author: David Young, 2017, 2018
"""Stitches multi-series imaginge files (eg CZI format) through the Fiji
BigStitcher plugin.

Attributes:
    in_file: Path to the image input file. The dataset filename will be 
        based on this input name, using the part of the name before the 
        first dash (eg "M05" from "M05-WT-P0...").
"""

from ij import IJ
import os
from time import time

#@String in_file
#@int write_fused

def perform_task(task, options):
    print("Running {}".format(task))
    print(options)
    time_start = time()
    IJ.run(task, options)
    print("\n{} elapsed time: {}".format(task, time() - time_start))

out_dir = os.path.dirname(in_file)
print("in_file: {}".format(in_file))
print("out_dir: {}".format(out_dir))

# dataset name based on experiment code, assumed to be the 
# section before the first dash in the input filename
basename = os.path.basename(in_file)
exp_code = basename.split("-")[0]
dataset_name_base = "dataset_{}".format(exp_code)
dataset_name_xml = "{}.xml".format(dataset_name_base)
dataset_path = os.path.join(out_dir, dataset_name_base)
dataset_path_xml = os.path.join(out_dir, dataset_name_xml)
print("dataset path: {}".format(dataset_path_xml))

time_start = time()
# import into HDF5 format (.h5 file)
'''
# these settings don't appear to change the output but appear
# during macro recording
    "subsampling_factors=[{ {1,1,1} }] "
    "hdf5_chunk_sizes=[{ {32,32,4} }] "
    "timepoints_per_partition=1 "
    "setups_per_partition=0 "
'''
options_define = (
    "define_dataset=[Automatic Loader (Bioformats based)] "
    "project_filename=" + dataset_name_xml + " "
    "path=" + in_file + " "
    "exclude=10 "
    "bioformats_series_are?=Tiles "
    "move_tiles_to_grid_(per_angle)?=[Do not move Tiles to Grid (use Metadata if available)] "
    "use_virtual_images_(cached) "
    "dataset_save_path=" + out_dir + " "
    "check_stack_sizes "
    "resave_as_hdf5 "
    "use_deflate_compression "
    "export_path=" + dataset_path
)

# choose the illumination for each tile
options_illuminators = (
    "select=" + dataset_path_xml + " "
    "selection=[Pick brightest]"
)

# calculate tile shifts
options_shifts = (
    "select=" + dataset_path_xml + " "
    "process_angle=[All angles] "
    "process_channel=[All channels] "
    "process_illumination=[All illuminations] "
    #"process_tile=[Multiple tiles (Select from List)] "
    "process_tile=[All tiles] "
    "process_timepoint=[All Timepoints] "
    #"tile_0 tile_1 "
    "method=[Phase Correlation] "
    "downsample_in_x=4 "
    "downsample_in_y=4 "
    "downsample_in_z=1"
)

# filter out poor alignments (currently set to filter none)
options_filter = (
    "select=" + dataset_path_xml + " "
    "filter_by_link_quality "
    "min_r=0.80 "
    "max_r=1 "
    "max_shift_in_x=0 "
    "max_shift_in_y=0 "
    "max_shift_in_z=0 "
    "max_displacement=0"
)

# apply the shifts in an iterative manner
options_apply = (
    "select=" + dataset_path_xml + " "
    "process_angle=[All angles] "
    "process_channel=[All channels] "
    "process_illumination=[All illuminations] "
    #"process_tile=[Multiple tiles (Select from List)] "
    "process_tile=[All tiles] "
    "process_timepoint=[All Timepoints] "
    #"tile_0 tile_1 "
    "how_to_treat_timepoints=[treat individually] "
    "how_to_treat_channels=group "
    "how_to_treat_illuminations=group "
    "how_to_treat_angles=[treat individually] "
    "how_to_treat_tiles=compare "
    "relative=2.500 "
    "absolute=3.500 "
    "global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles] "
    "fix_group_0-0"
)

# fuse the image and export to .tiff files; 
# for single-channel: swap "process_channel" and uncomment "processing_channel"
options_fuse = (
    "select=" + dataset_path_xml + " "
    "process_angle=[All angles] "
    "process_channel=[All channels] "
    #"process_channel=[Single channel (Select from List)]"
    "process_illumination=[All illuminations] "
    #"process_tile=[Multiple tiles (Select from List)] "
    "process_tile=[All tiles] "
    "process_timepoint=[All Timepoints] "
    #"processing_channel=[channel Cam2] "
    #"tile_0 tile_1 "
    #"bounding_box=[All Views] "
    "downsampling=1 "
    "pixel_type=[16-bit unsigned integer] "
    "interpolation=[Linear Interpolation] "
    "image=Virtual blend preserve_original "
    #"produce=[All views together] "
    "produce=[Each timepoint & channel] "
    "fused_image=[Save as (compressed) TIFF stacks] "
    "output_file_directory=" + out_dir
)


# perform stitching tasks

if write_fused in (0, 2):
    # 0 for import/align only; 2 for all
    perform_task("Define dataset ...", options_define)
    perform_task("Select Illuminations", options_illuminators)
    perform_task("Calculate pairwise shifts ...", options_shifts)
    perform_task("Filter pairwise shifts ...", options_filter)
    perform_task("Optimize globally and apply shifts ...", options_apply)

if write_fused in (1, 2):
    # 1 and 2 both including writing fused file(s)
    perform_task("Fuse dataset ...", options_fuse)

print("\nTotal elapsed time: {}".format(time() - time_start))

IJ.run("Quit")
