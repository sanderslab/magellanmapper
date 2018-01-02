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

# Replace with your paths
IMG="/path/to/your/image"
S3_DIR="path/to/your/bucket/artifact"

# Replace microscope type with available profiles, such as "lightsheet_5x", 
# "2p_20x", or "lightsheet_5x_02"
MICROSCOPE="lightsheet_5x_02"

# Replace region of interest (ROI) size and offset
SIZE=700,90,50
OFFSET=50,580,230

# Import raw image stack into Numpy array
#python -m clrbrain.cli --img "$IMG" --channel 0 --proc importonly

# Load ROI, starting at the given offset and ROI size
#./run --img "$IMG" --channel 0 --offset $OFFSET --size $SIZE --savefig pdf

# Process an entire stack locally
#python -m clrbrain.cli --img "$IMG" --proc processing_mp --channel 0 -v

# Process an entire stack on AWS
#./process_aws.sh -f "$IMG" -s $S3_DIR --  --microscope "$MICROSCOPE"

# Extract a single z-plane
#python -m clrbrain.cli --img "$IMG" --proc extract --channel 0 --offset 0,0,0 -v --savefig jpeg

# Process a sub-region of the stack and load it
#python -m clrbrain.cli --img "$IMG" --proc processing_mp --channel 0 -v --offset $OFFSET --size $SIZE --microscope "$MICROSCOPE"
#./run --img "$IMG($OFFSET)x($SIZE)" -v --channel 0 -v --proc load --offset $OFFSET --size $SIZE --savefig pdf --microscope "$MICROSCOPE"


####################################
# Stitching Workflow

# Replace with your paths
EXP="experiment_folder_name"
OUT_DIR="/path/to/output/parent/$EXP"
NAME_BASE="image_name_without_extension"
NAME="${NAME_BASE}.czi"
IMG="${OUT_DIR}/${NAME}"

# Replace with your lens objective settings
RESOLUTIONS="0.913,0.913,4.935"
MAGNIFICATION="5.0"
ZOOM="1.0"


# Get large, unstitched image file from cloud; image should be fused for 
# Alternative 1 and unfused for 2
#echo "Stitching $IMG to $OUT_DIR with resolutions $RESOLUTIONS, magnification $MAGNIFICATION, and zoom $ZOOM"
#mkdir $OUT_DIR
#aws s3 cp s3://"${S3_DIR}/${EXP}/${NAME}" $OUT_DIR


# ALTERNATIVE 1: Stitching plugin (old)

OUT_NAME_BASE="${NAME_BASE}_stitched"
TIFF_DIR="${OUT_DIR}/${NAME_BASE}"
# Replace the tile parameters with your image's setup
#python -m stitch.tile_config --img "$NAME" --target_dir "$OUT_DIR" --cols 6 --rows 7 --size 1920,1920,1000 --overlap 0.1 --directionality bi --start_direction right
#./stitch.sh -f "$IMG" -o "$TIFF_DIR" -c

# Before the next steps, please manually check alignments to ensure that they 
# fit properly, especially since unregistered tiles may be shifted to (0, 0, 0)
#./stitch.sh -f "$IMG" -o "$TIFF_DIR" -w
#python -u -m clrbrain.cli --img "$TIFF_DIR" --res "$RESOLUTIONS" --mag "$MAGNIFICATION" --zoom "$ZOOM" -v --channel 0 --proc importonly


# ALTERNATIVE 2: BigStitcher plugin

OUT_NAME_BASE="${NAME_BASE}_bigstitched"
TIFF_IMG="${OUT_DIR}/${NAME_BASE}.tiff"
#./stitch.sh -f "$IMG" -b
#python -u -m clrbrain.cli --img "$TIFF_IMG" --res "$RESOLUTIONS" --mag "$MAGNIFICATION" --zoom "$ZOOM" -v --channel 0 --proc importonly


# Upload stitched image to cloud
#aws s3 cp $OUT_DIR s3://"${S3_DIR}/${EXP}" --recursive --exclude "*" --include *.npz
