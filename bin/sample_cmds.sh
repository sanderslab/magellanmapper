#!/usr/bin/env bash
# Sample commands for MagellanMapper tasks
# Author: David Young, 2020

# This script is designed to provide examples of commands for performing
# various tasks in MagellanMapper. Commands can be run directly from this
# script in a Bash shell after changing variables below and uncommenting
# (ie remove the "#" at the start of the line) the desired command.
# Since most of these commands are run directly in Python, the commands can
# also be copied for use in other shells such as Windows command prompt or
# Batch scripts. For complete pipelines, see pipelines.sh.

# PATH AND SETTINGS SETUP
# update paths to your image files and settings

# choose file paths (relative to magellanmapper directory), channels, etc
PREFIXES=(. ../data) # add additional data folders
BASE=sample # replace with your sample file (without extension)
CHL=1
SERIES=0
ABA_DIR=ABA-CCFv3 # replace with atlas of choice
ABA_SPEC=ontology1.json # replace with atlas label map file

# profiles and theme
MIC=lightsheet # add/replace additional microscope profiles, separated by "_"
REG=finer_abaccfv3 # add/replace register/atlas profiles
THEME=(--theme dark) # GUI theme

# Annotation building
SIZE=1000,100,50 # z: 50-6*2 for ROI, -3*2 for border = 32; x/y: 42-5*2 for border
ROI_OFFSET=50,25,13 # get z from [50 (tot size) - 18 (ROI size)] / 2 - 3 (border)
ROI_SIZE=50,50,18

# offsets for ground truth ROIs within a thin, long sub-image
# - increment each ROI in x by 70 (small gap between each ROI)
# - view in "wide region" layout
OFFSETS=(
  "800,1150,250"
)

# subsets of OFFSETS that have been completed for testing
OFFSETS_DONE=("${OFFSETS[@]:0:20}")
offsets_test=($OFFSET)

# current offset
OFFSET="${OFFSETS[0]}"


# APPLY USER SETTINGS
# run from script's parent directory and set up paths
cd "$(dirname "$0")/.." || exit 1
. bin/labmag.sh
setup_image_paths PREFIXES "$BASE"
setup_atlas_paths PREFIXES "$ABA_DIR"

# IMPORT, REGISTRATION, CELL DETECTION, AND STATS

# initial import from TIF files
# - filenames should have the format: name_ch_0.tif, name_ch_1.tif, etc
# - change `res` to resolutions in x,y,z
# - change `mag` to objective magnification
#python -u -m magmap.io.cli --img "${IMG%.*}.tif" --proc import_only --res 10.52,10.52,10 --mag 0.63 -v

# downsample to size of CCFv3 and view
#python -u -m magmap.io.cli --img "$IMG" --proc transform --reg_profile finer_abaccfv3 #--plane xz
#./run.py --img "$IMG_RESIZED" --offset 225,300,150

# register imported CCFv3 to downsampled image and view
# - defaults to using channel 0; add `--channel x` to use channel x instead
#python -u -m magmap.atlas.register --img "$IMG_RESIZED" "$ABA_IMPORT_DIR" --prefix "$IMG" --flip 1 --register single --reg_profile "${REG}_raw" --no_show -v
#./run.py --img "$IMG_MHD" --microscope lightsheet_atlas --labels "$ABA_LABELS" --reg_suffixes exp.mhd annotation.mhd --offset 70,350,150

# full image detection
# - detects cells in channel set in variable `CHL`
#python -m magmap.io.cli --img "$IMG" --proc detect --channel "$CHL" --microscope "$MIC"

# make and view density image (heat map)
#python -u -m magmap.atlas.register -v --img "$IMG" --register make_density_images --no_show
#./run.py --img "$IMG" --microscope lightsheet_contrast --offset 125,250,175 --vmin 0 --vmax 2 --labels "$ABA_LABELS" --reg_suffixes heat.mhd annotation.mhd

# volume metrics
#python -u -m magmap.atlas.register --img "$IMG" --register vol_stats --reg_profile lightsheet_finer --labels "$ABA_LABELS"
#python -u -m magmap.atlas.register --img "$IMG" --register vol_stats --reg_profile lightsheet_finer --labels "$ABA_LABELS" 13

# turn on WT and basic.stats profiles (requires R 3.5+)
#Rscript --verbose clrstats/run.R


# VIEW ORIGINAL IMAGE, BUILD TRUTH SETS

# view original image
#./run.py --img "$IMG" --microscope "$MIC" --offset 800,1150,250 --size 70,70,30 -v #--proc load #--savefig pdf

# generate, view, and build truth set for single offset
#python -m magmap.io.cli --img "$IMG" --proc detect --channel "$CHL" --subimg_offset "$OFFSET" --subimg_size "$SIZE" --microscope "$MIC" --saveroi
#./run.py --img "$IMG" -v --channel "$CHL" --subimg_offset "$ROI_OFFSET" --subimg_size "$ROI_SIZE" --microscope lightsheet_contrast --proc load #--savefig png

# test single offset
#python -m magmap.io.cli --img "$IMG" --proc detect --channel "$CHL" --subimg_offset "$OFFSET" --subimg_size "$SIZE" --microscope "$MIC" --truth_db verify magmap.db --roc --no_show

# view verifications for single offset
#./run.py --img "$IMG" -v --channel "$CHL" --subimg_offset "$OFFSET" --subimg_size "$SIZE" --offset "$ROI_OFFSET" --size "$ROI_SIZE" --microscope lightsheet_contrast --proc load --truth_db verified "${THEME[@]}"

# test all OFFSETS with ROC curve
#python -m magmap.io.cli --img "$IMG" --proc detect --channel "$CHL" --offset ${OFFSETS_DONE[@]} --size $SIZE --microscope "$MIC" --truth_db "verify" --roc
#detect_roc "$IMG" offsets_test $SIZE
#detect_roc "$IMG" OFFSETS_DONE $SIZE

# view annotation (ie segmentation) truth set
#./run.py --img "$IMG_ROI" -v --channel "$CHL" --proc load --offset $ROI_OFFSET --size $ROI_SIZE --microscope "lightsheet_contrast" --db "$(name=$(basename $IMG); echo "${name%.*}_($OFFSET)x($SIZE)_00000_annot.db")"

# edit detections (ie blobs) truth set
#./run.py --img "$IMG_ROI" -v --channel "$CHL" --offset $ROI_OFFSET --size $ROI_SIZE --microscope "lightsheet_contrast" --proc load --truth_db edit magmap.db

# export single ROI after detections
#./run.py --img "$IMG_ROI" -v --channel "$CHL" --proc export_rois --savefig pdf --truth_db view
#export_roi "$IMG" $OFFSET $SIZE verified

# export all completed ROIs
#export_rois "$IMG" offsets_test "$SIZE" verified vols_stats_intensVnuc_rois.csv


# PIPELINES SCRIPT

# image downsampling
#bin/pipelines.sh -p transformation -i "$IMG" -z "$shape_resized" -- --reg_profile "$REG"
