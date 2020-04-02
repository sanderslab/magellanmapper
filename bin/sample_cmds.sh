#!/usr/bin/env bash
# Sample commands for MagellanMapper tasks
# Author: David Young, 2020

# This script is designed to provide examples of commands for performing
# various tasks in MagellanMapper. These commands can be copied into
# Bash scripts or modified slightly for use in other scripts such as Windows
# Batch scripts. For complete pipelines, see pipelines.sh.

# Alternatively, this script can be run directly after a few modifications.
# How to use:
# 1) Copy "sample_settings.sh" to your own file and update it with your settings
# 2) Uncomment (ie remove the "#" at line start) a desired command below
# 3) Run this script (eg from the magellanmapper folder), replacing the
#    settings file with your own: "bin/sample_cmds.sh bin/my_settings.sh"

# APPLY USER SETTINGS

# run from script's parent directory and set up paths
cd "$(dirname "$0")/.." || exit 1
. bin/libmag.sh

# load settings from external file
if [[ -z "$1" ]]; then
  msg="Please provide a settings file, such as "
  msg+="\"bin/sample_cmds.sh bin/my_settings.sh\""
  echo -e "$msg"
  exit 1
fi
. "$1"
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
