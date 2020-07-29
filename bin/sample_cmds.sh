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
#    a) Copy `sample_settings.sh` to your own file and customize its settings.
#       Copying the file rather than editing it directly allows you
#       to update the software without overwriting your changes.
#    b) Copy desired commands below into `custom_tasks` in your settings file
#    c) Run this script, pointing it to your custom settings file:
#       `bin/sample_cmds.sh <path-to-your-settings-file>`
#
# 3) For complete pipelines, see `pipelines.sh` instead.

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

# Example commands to learn ways to use MagellanMapper. Copy these commands
# into your own scripts, such as your custom settings script described above.
sample_tasks() {
  # IMPORT, REGISTRATION, CELL DETECTION, AND STATS

  # initial import from original microscopy file
  # - replace `.czi` with the extension of your file
  # - use `pipelines.sh` instead for stitching multi-tile images
  ./run.py --img "${IMG%.*}.czi" --proc import_only -v

  # initial import from TIF files
  # - filenames should have the format: name_ch_0.tif, name_ch_1.tif, etc
  # - change `resolutions` to image resolutions in x,y,z
  # - change `magnification` and `zoom` to microscope objective values
  ./run.py --img "${IMG%.*}.tif" --proc import_only \
    --set_meta resolutions=10.52,10.52,10 magnification=0.63 zoom=1.0 -v

  # view imported image
  ./run.py --img "$IMG"

  # downsample to size of atlas; the first option uses the target size in the
  # atlas profile, while the next options use a rescaling factor or specific
  # output size; use the plane option to transpose the output image
  ./run.py --img "$IMG" --proc transform --atlas_profile "$REG" #--plane xz
  ./run.py --img "$IMG" --proc transform --transform rescale=0.25
  ./run.py --img "$IMG" --proc transform --size "$SHAPE_RESIZED"

  # view downsampled image (assumes filename output using the first option)
  ./run.py --img "$IMG_RESIZED"

  # register imported atlas to downsampled image and view
  # - defaults to using channel 0; add `--channel x` to use channel x instead
  # - use the `transform` parameter for a 180 degree rotation (2 x 90 deg)
  ./run.py --img "$IMG_RESIZED" "$ABA_IMPORT_DIR" --prefix "$IMG" \
    --register single --atlas_profile "${REG},raw" -v #--transform rotate=2
  ./run.py --img "$IMG_MHD" --roi_profile lightsheet,atlas \
    --labels "$ABA_LABELS" --reg_suffixes exp.mhd annotation.mhd

  # similar view of registered labels but overlaid on downsampled image
  # including all of its channels
  ./run.py --img "$IMG_RESIZED" --prefix "$IMG_MHD" \
    --roi_profile lightsheet,atlas --labels "$ABA_LABELS" \
    --reg_suffixes annotation=annotation.mhd --offset 70,350,150

  # full image detection
  # - detects cells in channel set in variable `CHL`
  ./run.py --img "$IMG" --proc detect --channel "$CHL" --roi_profile "$MIC"

  # make and view density image (heat map)
  # TODO: not yet working on Windows
  ./run.py -v --img "$IMG" --register make_density_images
  ./run.py --img "$IMG" --roi_profile lightsheet,contrast \
    --offset 125,250,175 --vmin 0 --vmax 2 --labels "$ABA_LABELS" \
    --reg_suffixes heat.mhd annotation.mhd

  # volume metrics (level 13 includes hierarchical regions through this level)
  # TODO: not yet working on Windows
  ./run.py --img "$IMG" --register vol_stats \
    --atlas_profile lightsheet,finer --labels "$ABA_LABELS"
  ./run.py --img "$IMG" --register vol_stats \
    --atlas_profile lightsheet,finer --labels "$ABA_LABELS" 13

  # generate CSV of all atlas IDs with names
  ./run.py --register export_regions --labels "$ABA_LABELS" 1 --img "$ABA_DIR"

  # merge volume metrics CSV into this atlas names CSV to map IDs to names
  ./run.py --df merge_csvs_cols \
    --img "region_ids_$ABA_DIR.csv" "${IMG%.*}_volumes_level13.csv" \
    --plot_labels id_col=Region --prefix "${IMG%.*}_volumes_level13_named.csv"

  # turn on WT and basic.stats profiles (requires R 3.5+)
  Rscript --verbose clrstats/run.R

  # VIEW ORIGINAL IMAGE, BUILD TRUTH SETS

  # view original image
  # - use offset and size to start at a particular ROI position
  # - use the load flag to load previously saved whole-image blobs
  # - use savefig to automatically save ROI Editor figures
  ./run.py --img "$IMG" --roi_profile "$MIC" #--offset 800,1150,250 --size 70,70,30 -v #--proc load #--savefig pdf

  # view with overlaid registered labels
  # - reg suffixes are given as `atlas annotation borders` to load, where the
  #   atlas defaults to the main image (IMG)
  # - alphas are the opacities for the main image, labels, and region highlighter
  ./run.py --img "$IMG" --roi_profile lightsheet,atlas --labels "$ABA_LABELS" \
    --reg_suffixes annotation=annotation.mhd --alphas 1,0.5,0.4

  # detect blobs within a sub-image and export the sub-image for portability
  ./run.py --img "$IMG" --proc detect --channel "$CHL" \
    --subimg_offset "$OFFSET" --subimg_size "$SIZE" --roi_profile "$MIC" \
    --save_subimg

  # view and build truth set for a sub-image; after pressing "Save blobs,"
  # the truth set will be in magmap.db
  ./run.py --img "$IMG" -v --channel "$CHL" --subimg_offset "$OFFSET" \
    --subimg_size "$SIZE" --offset "$ROI_OFFSET" --size "$ROI_SIZE" \
    --roi_profile lightsheet,contrast --proc load #--savefig png

  # edit saved truth set; load saved ROIs from the "ROIs" dropdown box,
  # then press "ROI Editor"
  ./run.py --img "$IMG" -v --channel "$CHL" --subimg_offset "$OFFSET" \
    --subimg_size "$SIZE" --roi_profile lightsheet,contrast --proc load \
    --truth_db edit magmap.db

  # grid-search on single sub-image using the "test" ROC profile
  ./run.py --img "$IMG" --proc detect --channel "$CHL" \
    --subimg_offset "$OFFSET" --subimg_size "$SIZE" --roi_profile "$MIC" \
    --truth_db verify magmap.db --grid_search gridtest

  # view verifications for single offset
  ./run.py --img "$IMG" -v --channel "$CHL" --subimg_offset "$OFFSET" \
    --subimg_size "$SIZE" --offset "$ROI_OFFSET" --size "$ROI_SIZE" \
    --roi_profile lightsheet,contrast --proc load --truth_db verified "${THEME[@]}"

  # test all OFFSETS with ROC curve
  ./run.py --img "$IMG" --proc detect --channel "$CHL" \
    --subimg_offset ${OFFSETS_DONE[@]} --subimg_size $SIZE --roi_profile "$MIC" \
    --truth_db "verify" --grid_search gridtest

  # view annotation (ie segmentation) truth set
  ./run.py --img "$IMG" -v --channel "$CHL" --proc load \
    --subimg_offset "$OFFSET" --subimg_size "$SIZE" \
    --roi_profile lightsheet,contrast \
    --db "$(name=$(basename $IMG); echo "${name%.*}_($OFFSET)x($SIZE)_00000_annot.db")"

  # EXPORT IMAGES

  # export animation of image and registered atlas labels through all z-planes
  # - slice is given as `start,stop,step`, where none = all
  # - prefix is for images registered to a different image path
  ./run.py --img "$IMG_RESIZED" --proc animated --slice none,none,1 \
    --roi_profile atlas --savefig mp4 --prefix "$IMG" --labels "$ABA_LABELS" \
    --reg_suffixes exp.mhd annotation.mhd

  # export ROI after detections
  ./run.py --img "$IMG" -v --channel "$CHL" --proc export_rois --savefig pdf \
    --truth_db view

  # PIPELINES SCRIPT

  # image downsampling
  bin/pipelines.sh -p transformation -i "$IMG" -z "$shape_resized" \
    -- --atlas_profile "$REG"

  # OTHER IMAGE TRANSFORMATIONS

  # rotate an image along multiple axes as specified in custom profiles;
  # 1) create an ROI profile file with a "preprocess" key giving a list of tasks
  # 2) create an atlas profile file specifying the rotation parameters
  #    (see `atlas_prof.py`)
  ./run.py --img "$IMG" --proc preprocess \
    --roi_profile preproc.yaml --atlas_profile rotate.yaml
}

# This call will run the `custom_tasks` function in your settings script
custom_tasks
