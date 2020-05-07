#!/usr/bin/env bash
# Sample commands for MagellanMapper tasks
# Author: David Young, 2020

# This script provides examples of commands for performing various tasks in
# MagellanMapper. You can use the script in various ways:
#
# 1) Use these commands as templates. These commands can be copied into
#    Bash scripts or Windows Batch scripts and customized for your file paths.
#
# 2) Run this script directly after a few modifications:
#    a) Copy `sample_settings.sh` to your own file and customize its settings
#    b) Copy desired commands below into `custom_tasks` in your settings file
#    c) Run this script, pointing it to your custom settings file:
#       `bin/sample_cmds.sh bin/my_settings.sh`
#
# 3) You can also uncomment (ie remove the "#" at line start) a desired command
#    below to run in directly, but this file will need to be reset before
#    any code repository updates.
#
# 4) For complete pipelines, see `pipelines.sh` instead.

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

# initial import from original microscopy file
# - replace `.czi` with the extension of your file
# - use `pipelines.sh` instead for stitching multi-tile images
#python -u -m magmap.io.cli --img "${IMG%.*}.czi" --proc import_only -v

# initial import from TIF files
# - filenames should have the format: name_ch_0.tif, name_ch_1.tif, etc
# - change `res` to resolutions in x,y,z
# - change `mag` to objective magnification
#python -u -m magmap.io.cli --img "${IMG%.*}.tif" --proc import_only --res 10.52,10.52,10 --mag 0.63 -v

# view imported image
#./run.py --img "$IMG"

# downsample to size of atlas; the first option uses the target size in the
# atlas profile, while the next options use a rescaling factor or specific
# output size; use the plane option to transpose the output image
#python -u -m magmap.io.cli --img "$IMG" --proc transform --reg_profile "$REG" #--plane xz
#python -u -m magmap.io.cli --img "$IMG" --proc transform --transform rescale=0.25
#python -u -m magmap.io.cli --img "$IMG" --proc transform --size "$SHAPE_RESIZED"

# view downsampled image (assumes filename output using the first option)
#./run.py --img "$IMG_RESIZED"

# register imported atlas to downsampled image and view
# - defaults to using channel 0; add `--channel x` to use channel x instead
#./run_cli.py --img "$IMG_RESIZED" "$ABA_IMPORT_DIR" --prefix "$IMG" --flip 1 --register single --reg_profile "${REG}_raw" --no_show -v
#./run.py --img "$IMG_MHD" --roi_profile lightsheet_atlas --labels "$ABA_LABELS" --reg_suffixes exp.mhd annotation.mhd --offset 70,350,150

# similar view of registered labels but overlaid on downsampled image
# including all of its channels
#./run.py --img "$IMG_RESIZED" --prefix "$IMG_MHD" --roi_profile lightsheet_atlas --labels "$ABA_LABELS" --reg_suffixes annotation=annotation.mhd --offset 70,350,150

# full image detection
# - detects cells in channel set in variable `CHL`
#./run_cli.py --img "$IMG" --proc detect --channel "$CHL" --roi_profile "$MIC"

# make and view density image (heat map)
#./run_cli.py -v --img "$IMG" --register make_density_images --no_show
#./run.py --img "$IMG" --roi_profile lightsheet_contrast --offset 125,250,175 --vmin 0 --vmax 2 --labels "$ABA_LABELS" --reg_suffixes heat.mhd annotation.mhd

# volume metrics (level 13 includes hierarchical regions through this level)
#./run_cli.py --img "$IMG" --register vol_stats --reg_profile lightsheet_finer --labels "$ABA_LABELS"
#./run_cli.py --img "$IMG" --register vol_stats --reg_profile lightsheet_finer --labels "$ABA_LABELS" 13

# generate CSV of all atlas IDs with names; merge with hierarchical volumes CSV
#./run_cli.py --register export_regions --labels "$ABA_LABELS" 1 --img "$ABA_DIR"
#./run_cli.py --df merge_csvs_cols --img "region_ids_$ABA_DIR.csv" "${IMG%.*}_volumes_level13.csv" --plot_labels id_col=Region --prefix "${IMG%.*}_volumes_level13_named.csv"

# turn on WT and basic.stats profiles (requires R 3.5+)
#Rscript --verbose clrstats/run.R


# VIEW ORIGINAL IMAGE, BUILD TRUTH SETS

# view original image
# - use offset and size to start at a particular ROI position
# - use the load flag to load previously saved whole-image blobs
# - use savefig to automatically save ROI Editor figures
#./run.py --img "$IMG" --roi_profile "$MIC" #--offset 800,1150,250 --size 70,70,30 -v #--proc load #--savefig pdf

# view with overlaid registered labels
# - reg suffixes are given as `atlas annotation borders` to load, where the
#   atlas defaults to the main image (IMG)
# - alphas are the opacities for the main image, labels, and region highlighter
#./run.py --img "$IMG" --roi_profile lightsheet_atlas --labels "$ABA_LABELS" --reg_suffixes annotation=annotation.mhd --alphas 1,0.5,0.4

# detect blobs within a sub-image and export the sub-image for portability
#./run_cli.py --img "$IMG" --proc detect --channel "$CHL" --subimg_offset "$OFFSET" --subimg_size "$SIZE" --roi_profile "$MIC" --saveroi

# view and build truth set for a sub-image; after pressing "Save blobs," the truth set will be in magmap.db
#./run.py --img "$IMG" -v --channel "$CHL" --subimg_offset "$OFFSET" --subimg_size "$SIZE" --offset "$ROI_OFFSET" --size "$ROI_SIZE" --roi_profile lightsheet_contrast --proc load #--savefig png

# edit saved truth set; load saved ROIs from the "ROIs" dropdown box, then press "ROI Editor"
#./run.py --img "$IMG" -v --channel "$CHL" --subimg_offset "$OFFSET" --subimg_size "$SIZE" --roi_profile "lightsheet_contrast" --proc load --truth_db edit magmap.db

# grid-search on single sub-image using the "test" ROC profile
#./run_cli.py --img "$IMG" --proc detect --channel "$CHL" --subimg_offset "$OFFSET" --subimg_size "$SIZE" --roi_profile "$MIC" --truth_db verify magmap.db --grid_search test --no_show

# view verifications for single offset
#./run.py --img "$IMG" -v --channel "$CHL" --subimg_offset "$OFFSET" --subimg_size "$SIZE" --offset "$ROI_OFFSET" --size "$ROI_SIZE" --roi_profile lightsheet_contrast --proc load --truth_db verified "${THEME[@]}"

# test all OFFSETS with ROC curve
#./run_cli.py --img "$IMG" --proc detect --channel "$CHL" --offset ${OFFSETS_DONE[@]} --size $SIZE --roi_profile "$MIC" --truth_db "verify" --grid_search

# view annotation (ie segmentation) truth set
#./run.py --img "$IMG" -v --channel "$CHL" --proc load --subimg_offset "$OFFSET" --subimg_size "$SIZE" --roi_profile "lightsheet_contrast" --db "$(name=$(basename $IMG); echo "${name%.*}_($OFFSET)x($SIZE)_00000_annot.db")"


# EXPORT IMAGES

# export animation of image and registered atlas labels through all z-planes
# - slice is given as `start,stop,step`, where none = all
# - prefix is for images registered to a different image path
#./run_cli.py --img "$IMG_RESIZED" --proc animated --slice none,none,1 --roi_profile atlas --savefig mp4 --prefix "$IMG" --labels "$ABA_LABELS" --reg_suffixes exp.mhd annotation.mhd

# export ROI after detections
#./run.py --img "$IMG" -v --channel "$CHL" --proc export_rois --savefig pdf --truth_db view


# PIPELINES SCRIPT

# image downsampling
#bin/pipelines.sh -p transformation -i "$IMG" -z "$shape_resized" -- --reg_profile "$REG"


# OTHER IMAGE TRANSFORMATIONS

# rotate an image along multiple axes as specified in custom profiles;
# the first profile needs a "preprocess" key with a list of tasks, and
# the second profile specifies the rotation (see `atlas_prof.py`)
# ./run_cli.py --img "$IMG" --proc preprocess --roi_profile profiles/preproc.yaml --reg_profile profiles/rotate.yaml


# CUSTOM TASKS

# add your own commands here, or add them to this function in the your
# script based on `sample_settings.sh` to minimize edits here
custom_tasks
