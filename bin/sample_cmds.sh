#!/usr/bin/env bash
# Sample commands for MagellanMapper tasks
# Author: David Young, 2020

# WARNING (2020-09-18): We are brainstorming ways to reorganize the sample
# scripts with the goal of exposing MagellanMapper functionality, keeping
# commands up to date, and minimize setup required by the user. Any and all
# feedback is welcome!

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
  # IMAGE IMPORT

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


  # ATLAS REGISTRATION TO IMPORTED IMAGE
  
  # downsample to size of atlas; the first option uses the target size in the
  # atlas profile, while the next options use a rescaling factor or specific
  # output size; use the plane option to transpose the output image
  ./run.py --img "$IMG" --proc transform --atlas_profile "$REG" #--plane xz
  ./run.py --img "$IMG" --proc transform --transform rescale=0.25
  ./run.py --img "$IMG" --proc transform --size "$SHAPE_RESIZED"

  # view downsampled image (assumes filename output using the first option)
  ./run.py --img "$IMG_RESIZED"

  # register imported atlas to downsampled image
  # - defaults to using channel 0; add `--channel x` to use channel x instead
  # - to rotate: `--transform rotate=<n>`, where n = number of 90 deg rotations
  # - to transpose: `--plane <xy|xz|yz>`, where xz transposes y-plane to z-plane
  ./run.py --img "$IMG_RESIZED" "$ABA_IMPORT_DIR" --prefix "$IMG" \
    --register single --atlas_profile "$REG" -v #--transform rotate=2
  
  # view registered atlas on exported, single-channel downsampled image
  ./run.py --img "$IMG_MHD" --roi_profile lightsheet,atlas \
    --labels "$ABA_LABELS" --reg_suffixes exp.mhd annotation.mhd

  # similar view of registered labels but overlaid on original downsampled image
  # including all of its channels; prefix is used to find registered images
  ./run.py --img "$IMG_RESIZED" --prefix "$IMG_MHD" \
    --roi_profile lightsheet,atlas --labels "$ABA_LABELS" \
    --reg_suffixes annotation=annotation.mhd --offset 70,350,150


  # CELL DETECTION AND STATS BY ATLAS REGION
  
  # full image detection
  # - detects cells in channel set in variable `CHL`
  ./run.py --img "$IMG" --proc detect --channel "$CHL" --roi_profile "$MIC"
  
  # load blobs to view rather than redetecting blobs in each ROI
  ./run.py --img "$IMG" --roi_profile "$MIC" --load blobs

  # make and view density image (heat map)
  ./run.py -v --img "$IMG" --register make_density_images
  ./run.py --img "$IMG" --roi_profile lightsheet,contrast \
    --offset 125,250,175 --vmin 0 --vmax 2 --labels "$ABA_LABELS" \
    --reg_suffixes heat.mhd annotation.mhd

  # volume metrics for each atlas label in the image
  ./run.py --img "$IMG" --register vol_stats --labels "$ABA_LABELS"
  
  # combine metrics for hierarchical levels
  # - eg, level 13 includes hierarchical regions through all levels in
  #   the Allen Developing Mouse Brain Atlas; use 11 for Allen CCFv3
  # - run after generating labels for each image label (the above command)
  # - add `--atlas_profile combinesides` to the command to combine
  #   corresponding regions from opposite hemispheres
  ./run.py --img "$IMG" --register vol_stats --labels "$ABA_LABELS" 13

  # generate CSV of all atlas IDs with names
  ./run.py --register export_regions --labels "$ABA_LABELS" 1 --img "$ABA_DIR"

  # merge volume metrics CSV into this atlas names CSV to map IDs to names
  ./run.py --df merge_csvs_cols \
    --img "region_ids_$ABA_DIR.csv" "${IMG%.*}_volumes_level13.csv" \
    --plot_labels id_col=Region --prefix "${IMG%.*}_volumes_level13_named.csv"

  # turn on WT and basic.stats profiles (requires R 3.5+)
  Rscript --verbose clrstats/run.R
  
  
  # BLOB COLOCALIZATION
  
  # intensity-based colocalization:
  # - perform together with whole image detection
  # - replace channels with desired channels to colocalize
  # - replace "profile0" and "profile1" with desired profiles for these chls
  # - output is in the blobs NPZ file, automatically loaded with blobs
  ./run.py --img "$IMG" --proc detect_coloc --channel 0 1 \
    --roi_profile profile0 profile1

  # match-based colocalization:
  # - perform after whole image detection to find optimal matches between
  #   nearby blobs
  # - replace with desired channels and profiles
  # - output is in the magmap.db database
  ./run.py --img "$IMG" --proc coloc_match --channel 0 1 \
    --roi_profile profile0 profile1
  
  # load blobs and match-based colocalizations
  # - replace with desired profiles
  ./run.py --img "$IMG" --load blobs blob_matches \
    --roi_profile profile0 profile1
  
  # make density image for match-based colocalizations
  # - replace with desired channels
  # - heat.mhd has densities of combined blobs across the given channels
  # - heatColoc.mhd has densities of blobs colocalized for these channels,
  #   preferentially using blob matches if available and falling back
  #   to intensity-based colocalizations
  ./run.py --img "$IMG" --register make_density_images --channel 0 1
  
  # view match-based colocalization density image
  ./run.py --img "$IMG" --roi_profile lightsheet,contrast --vmin 0 --vmax 2 \
    --labels "$ABA_LABELS" --reg_suffixes heatColoc.mhd annotation.mhd \
    --alphas 1,0.1


  # VIEW ORIGINAL IMAGE, BUILD TRUTH SETS

  # view original image
  # - use offset and size to start at a particular ROI position
  # - use the load flag to load previously saved whole-image blobs
  # - use savefig to automatically save ROI Editor figures
  ./run.py --img "$IMG" --roi_profile "$MIC" #--offset 800,1150,250 --size 70,70,30 -v #--load blobs #--savefig pdf

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
    --roi_profile lightsheet,contrast --load blobs #--savefig png

  # edit saved truth set; load saved ROIs from the "ROIs" dropdown box,
  # then press "ROI Editor"
  ./run.py --img "$IMG" -v --channel "$CHL" --subimg_offset "$OFFSET" \
    --subimg_size "$SIZE" --roi_profile lightsheet,contrast --load blobs \
    --truth_db edit magmap.db

  # grid-search on single sub-image using the "test" ROC profile
  ./run.py --img "$IMG" --proc detect --channel "$CHL" \
    --subimg_offset "$OFFSET" --subimg_size "$SIZE" --roi_profile "$MIC" \
    --truth_db verify magmap.db --grid_search gridtest

  # view verifications for single offset
  ./run.py --img "$IMG" -v --channel "$CHL" --subimg_offset "$OFFSET" \
    --subimg_size "$SIZE" --offset "$ROI_OFFSET" --size "$ROI_SIZE" \
    --roi_profile lightsheet,contrast --load blobs --truth_db verified "${THEME[@]}"

  # test all OFFSETS with ROC curve
  ./run.py --img "$IMG" --proc detect --channel "$CHL" \
    --subimg_offset ${OFFSETS_DONE[@]} --subimg_size $SIZE --roi_profile "$MIC" \
    --truth_db "verify" --grid_search gridtest

  # view annotation (ie segmentation) truth set
  ./run.py --img "$IMG" -v --channel "$CHL" --load blobs \
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

  # rotate an image along multiple axes:
  # 1) create an atlas profile file specifying the rotation parameters
  #    (see `profiles/atlas_rotate.yml` for an example)
  # 2) run this command
  ./run.py --img "$img" --proc preprocess=rotate --atlas_profile rotate.yaml
}

# This call will run the `custom_tasks` function in your settings script
custom_tasks
