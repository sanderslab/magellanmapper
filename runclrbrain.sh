#!/bin/bash
# Template for running Clrbrain
# Author: David Young 2017, 2018

################################################
# Sample scenarios and workflows for Clrbrain
#
# Use this file as a template for your own scenarioes. Edit path 
# variables with your own file paths. Uncomment (ie remove the 
# "#" symbol) to apply the given scenario for your files.
################################################


####################################
# REPLACE WITH YOUR PATHS

# full path to original image, assumed to be within a directory 
# that names the experiment and also used for S3; eg 
# "/data/exp_yyyy-mm-dd/WT-cortex.czi"
IMG="/path/to/your/image"

# parent path of image file in cloud such as AWS S3; eg
# "MyName/ClearingExps", where the image would be found in 
# $S3_DIR/exp_yyyy-mm-dd
S3_DIR="path/to/your/bucket/artifact"


####################################
# Basic Usage

# Parsing names from your image path
OUT_DIR="`dirname $IMG`"
EXP="`basename $OUT_DIR`"
NAME="`basename $IMG`"
NAME_BASE="${NAME%.*}"
IMG_PATH_BASE="${OUT_DIR}/${NAME_BASE}"
EXT="${IMG##*.}"

# run from script directory
BASE_DIR="`dirname $0`"
cd "$BASE_DIR"
echo $PWD

# Replace microscope type with available profiles, such as "lightsheet", 
# "2p_20x", or "lightsheet_v02", or with modifiers, such as 
# "lightsheet_contrast" or "lightsheet_contrast_cytoplasm". Multiple 
# profiles can also be given for multiple channels, such as 
# "lightsheet lightsheet_cytoplasm" for a nuclear marker in channel 0 
# and cytoplasmic marker in channel 1
MICROSCOPE="lightsheet"

# Replace region of interest (ROI) size and offset
SIZE=700,90,50
OFFSET=50,580,230

# Import raw image stack into Numpy array
#python -u -m clrbrain.cli --img "$IMG" --channel 0 --proc importonly

# Load ROI, starting at the given offset and ROI size
#./run --img "$IMG" --channel 0 --offset $OFFSET --size $SIZE --savefig pdf --microscope "$MICROSCOPE"

# Extract a single z-plane
#python -u -m clrbrain.cli --img "$IMG" --proc extract --channel 0 --offset 0,0,0 -v --savefig jpeg --microscope "$MICROSCOPE"

# Process a sub-region of the stack and load it
#python -m clrbrain.cli --img "$IMG" --proc processing_mp --channel 0 -v --offset $OFFSET --size $SIZE --microscope "$MICROSCOPE"
IMG_ROI="${IMG_PATH_BASE}_(${OFFSET})x(${SIZE}).${EXT}"
#./run --img "$IMG_ROI" -v --channel 0 -v --proc load --offset $OFFSET --size $SIZE --savefig pdf --microscope "$MICROSCOPE"

# Transpose an image and generate an animated GIF
#python -u -m clrbrain.cli --img "$IMG" --proc transpose --rescale 0.05 --plane yz
IMG_TRANSPOSED="${IMG_PATH_BASE}_transposed.${EXT}"
#python -u -m clrbrain.cli --img "$IMG_TRANSPOSED" --proc animated --interval 5 --rescale 1.0


####################################
# Whole Image Processing Workflow

# Process an entire image locally on 1st channel, chunked into multiple 
# smaller stacks to minimize RAM usage and multiprocessed for efficiency
#python -u -m clrbrain.cli --img "$IMG" --proc processing_mp --channel 0 --microscope "$MICROSCOPE"

# Similar processing but integrated with S3 access from AWS (run from 
# within EC2 instance)
#./process_aws.sh -f "$IMG" -s $S3_DIR --  --microscope "$MICROSCOPE" --channel 0


####################################
# Stitching Workflow

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
TIFF_DIR="${OUT_DIR}/${OUT_NAME_BASE}"

# Replace the tile parameters with your image's setup; set up tile 
# configuration manually and compute alignment refinement
#python -m stitch.tile_config --img "$NAME" --target_dir "$OUT_DIR" --cols 6 --rows 7 --size 1920,1920,1000 --overlap 0.1 --directionality bi --start_direction right
#./stitch.sh -f "$IMG" -o "$TIFF_DIR" -w 0

# Before the next steps, please manually check alignments to ensure that they 
# fit properly, especially since unregistered tiles may be shifted to (0, 0, 0)
#./stitch.sh -f "$IMG" -o "$TIFF_DIR" -w 1
#python -u -m clrbrain.cli --img "$TIFF_DIR" --res "$RESOLUTIONS" --mag "$MAGNIFICATION" --zoom "$ZOOM" -v --channel 0 --proc importonly


# ALTERNATIVE 2: BigStitcher plugin

OUT_NAME_BASE="${NAME_BASE}_bigstitched"
TIFF_IMG="${OUT_DIR}/${OUT_NAME_BASE}.tiff"

# Import file into BigStitcher HDF5 format (warning: large file, just 
# under size of original file) and find alignments
#./stitch.sh -f "$IMG" -b -w 0

# Before writing stitched file, advise checking alignments; when 
# satisfied, then run this fusion step
#./stitch.sh -f "$IMG" -b -w 1

# Rename output file(s):
FUSED="fused_tp_0"
#for f in ${OUT_DIR}/${FUSED}*.tif; do mv $f ${f/$FUSED/$OUT_NAME_BASE}; done

# Import stacked TIFF file(s) into Numpy arrays for Clrbrain
#python -u -m clrbrain.cli --img "$TIFF_IMG" --res "$RESOLUTIONS" --mag "$MAGNIFICATION" --zoom "$ZOOM" -v --proc importonly


# Upload stitched image to cloud
#aws s3 cp $OUT_DIR s3://"${S3_DIR}/${EXP}" --recursive --exclude "*" --include *.npz

exit 0
