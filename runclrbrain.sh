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
# REPLACE WITH SETTINGS FOR YOUR IMAGE

# full path to original image, assumed to be within a directory 
# that names the experiment and also used for S3; eg 
# "/data/exp_yyyy-mm-dd/WT-cortex.czi"
IMG="/path/to/your/image"

# parent path of image file in cloud such as AWS S3; eg
# "MyName/ClearingExps", where the image would be found in 
# $S3_DIR/exp_yyyy-mm-dd
S3_DIR="path/to/your/bucket/artifact"

# Replace microscope type with available profiles, such as "lightsheet", 
# "2p_20x", or "lightsheet_v02", or with modifiers, such as 
# "lightsheet_contrast" or "lightsheet_contrast_cytoplasm". Multiple 
# profiles can also be given for multiple channels, such as 
# "lightsheet lightsheet_cytoplasm" for a nuclear marker in channel 0 
# and cytoplasmic marker in channel 1
MICROSCOPE="lightsheet"



####################################
# Script setup

# Parsing names from your image path
OUT_DIR="`dirname $IMG`"
EXP="`basename $OUT_DIR`"
NAME="`basename $IMG`"
IMG_PATH_BASE="${OUT_DIR}/${NAME%.*}"
EXT="${IMG##*.}"

# run from script's directory
BASE_DIR="`dirname $0`"
cd "$BASE_DIR"
echo $PWD



####################################
# Basic Usage

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



####################################
# Stitching Workflow

# Replace with your lens objective settings
RESOLUTIONS="0.913,0.913,4.935"
MAGNIFICATION="5.0"
ZOOM="1.0"
# Choose "stitching" or "bigstitcher" for alternative stitching plugin pathways
stitch_pathway="bigstitcher"

# Get large, unstitched image file from cloud, where the fused (all 
# illuminators merged) image is used for the Stitching pathway, and 
# the unfused, original image is used for the BigStitcher pathway
#mkdir $OUT_DIR
#aws s3 cp s3://"${S3_DIR}/${EXP}/${NAME}" $OUT_DIR

out_name_base=""
clr_img=""
if [[ "$stitch_pathway$" == "stitching" ]]; then
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
    clr_img="${OUT_DIR}/${OUT_NAME_BASE}.${EXT}"
    
elif [[ "$stitch_pathway$" == "bigstitcher" ]]; then
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
    clr_img="${OUT_DIR}/${OUT_NAME_BASE}.${EXT}"
fi

# At this point, you can delete the TIFF image since it has been exported into a Numpy-based 
# format for loading into Clrbrain


####################################
# Transpose/Resize Image Workflow

# Replace with your rescaling factor here
scale=0.05
plane="yx"

clr_img_base="${clr_img%.*}"

# Rescale an image to downsample by the scale factor
#python -u -m clrbrain.cli --img "$clr_img" --proc transpose --rescale ${scale}
img_transposed="${clr_img_base}_scale${scale}.${EXT}"

# Both rescale and transpose an image from z-axis (xy plane) to x-axis (yz plane) orientation
#python -u -m clrbrain.cli --img "$clr_img" --proc transpose --rescale ${scale} --plane "$plane"
#img_transposed="${clr_img_base}_plane${plane}_scale${scale}.${EXT}"

scale=1.0

# Export transposed image to an animated GIF (requires ImageMagick)
#python -u -m clrbrain.cli --img "$img_transposed" --proc animated --interval 5 --rescale ${scale}

# Alternatively, export to an MP4 video
#python -u -m clrbrain.cli --img "$img_transposed" --proc animated --interval 5 --rescale ${scale} --savefig "mp4"



####################################
# Whole Image Processing Workflow

# Process an entire image locally on 1st channel, chunked into multiple 
# smaller stacks to minimize RAM usage and multiprocessed for efficiency
#python -u -m clrbrain.cli --img "$clr_img" --proc processing_mp --channel 0 --microscope "$MICROSCOPE"

# Similar processing but integrated with S3 access from AWS (run from 
# within EC2 instance)
#./process_aws.sh -f "$clr_img" -s $S3_DIR --  --microscope "$MICROSCOPE" --channel 0


# Upload stitched image to cloud
#aws s3 cp $OUT_DIR s3://"${S3_DIR}/${EXP}" --recursive --exclude "*" --include *.npz

exit 0
