#!/bin/bash
# Template for running Clrbrain
# Author: David Young 2017

################################################
# Sample scenarios and workflows for Clrbrain
#
# Use this file as a template for your own scenarioes. Edit path 
# variables with your own file paths. Uncomment (remove the 
# "#" symbol) to apply the given scenario for your files.
################################################

# run from script directory
BASE_DIR="`dirname $0`"
cd "$BASE_DIR"
echo $PWD

####################################
# Basic usage

IMG="/path/to/your/image"
S3_DIR="path/to/your/bucket/artifact"

# Import raw image stack into Numpy array
#python -m clrbrain.cli --img "$IMG" --channel 0 --proc importonly

# Load ROI, starting at the given offset and ROI size
#./run --img "$IMG" --channel 0 --offset 30,30,8 --size 50,50,10 --savefig pdf

# Process an entire stack locally
#python -m clrbrain.cli --img "$IMG" --proc processing_mp --channel 0 -v

# Process an entire stack on AWS
#./process_aws.sh -f "$IMG" -s $S3_DIR --  --microscope "2p_20x"

# Extract a single z-plane
#python -m clrbrain.cli --img "$IMG" --proc extract --channel 0 --offset 0,0,0 -v --savefig jpeg

# Process a sub-region of the stack and load it
SIZE=700,90,50
OFFSET=50,580,230
#python -m clrbrain.cli --img "$IMG" --proc processing_mp --channel 0 -v --offset $OFFSET --size $SIZE --microscope "2p_20x"
#./run --img "$IMG($OFFSET)x($SIZE)" -v --channel 0 -v --proc load --offset 50,20,20 --size 50,50,10 --savefig pdf --microscope "2p_20x"


####################################
# Stitching Workflow

EXP="experiment_folder_name"
NAME="image_filename"
OUT_DIR="/path/to/output/folder"
IMG="${OUT_DIR}/${NAME}"
NAME_BASE="output_image_name_without_extension"
TIFF_DIR="${OUT_DIR}/${NAME_BASE}"
TIFF_IMG="${OUT_DIR}/${NAME_BASE}.tiff"

# Get large, unstitched file from cloud
#mkdir $OUT_DIR
#aws s3 cp "s3:${S3_DIR}/${EXP}/${NAME}" $OUT_DIR


# ALTERNATIVE 1: Stitching plugin (old)

# Replace the tile parameters with your image's setup
#python -m stitch.tile_config --img "$NAME" --target_dir "$OUT_DIR" --cols 6 --rows 7 --size 1920,1920,1000 --overlap 0.1 --directionality bi --start_direction right
#./stitch.sh -f "$IMG" -o "$TIFF_DIR" -c

# Before the next steps, please manually check alignments to ensure that they 
# fit properly, especially since unregistered tiles may be shifted to (0, 0, 0)
#./stitch.sh -f "$IMG" -o "$TIFF_DIR" -w
#python -u -m clrbrain.cli --img "$TIFF_DIR" --res 0.913,0.913,4.935 --mag 5.0 --zoom 1.0 -v --channel 0 --proc importonly


# ALTERNATIVE 2: BigStitcher plugin

#./stitch.sh -f "$IMG" -b
#python -u -m clrbrain.cli --img "$TIFF_IMG" --res 0.913,0.913,4.935 --mag 5.0 --zoom 1.0 -v --channel 0 --proc importonly


# Upload stitched image to cloud
#aws s3 cp $OUT_DIR "s3:${S3_DIR}/${EXP}" --recursive --exclude "*" --include *.npz
