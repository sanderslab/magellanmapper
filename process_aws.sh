#!/bin/bash
# Process files on Amazon Web Services EC2 server
# Author: David Young 2017

################################################
# Imports files from S3 for processing on EC2 and upload
# back to S3.
#
# To run:
# -Activate conda environment: "source activate clr"
# -Update clrbrain: "cd src/clrbrain; git fetch; git pull"
# -Ensure that swap space and data drive are mounted
# -Update paths to files for processing, ensuring that the
#  *_image5d.npz and *_info.npz files are on S3
# -Sample run command: 
#  nohup ./process_aws.sh -f "/path/to/img.czi" -s "root/path" \
#    -- --microscope "type" > /path/to/output 2>&1 &
# -Track results: "tail -f /path/to/output"
# -If all goes well, pick up processed files from S3
################################################

DEST=/data
IMG=""
S3_DIR=""
SERIES=0 # TODO: make settable
EXTRA_ARGS=""

# workaround for https://github.com/numpy/numpy/issues/5336,
# fixed in https://github.com/numpy/numpy/pull/7133, 
# released in Numpy 1.12.0
TMPDIR="$DEST"/tmp
if [ ! -e "$TMPDIR" ]
then
    mkdir "$TMPDIR"
fi
export TMPDIR="$TMPDIR"

# run from parent directory
BASE_DIR="`dirname $0`"
cd "$BASE_DIR"
echo $PWD

OPTIND=1
while getopts hf:s: opt; do
    case $opt in
        h)  echo $HELP
            exit 0
            ;;
        f)  IMG="$OPTARG"
            echo "Set image file to $IMG"
            ;;
        s)  S3_DIR="$OPTARG"
            echo "Set AWS S3 base path to $S3_DIR"
            ;;
        :)  echo "Option -$OPTARG requires an argument"
            exit 1
            ;;
        --) ;;
    esac
done

# pass arguments after "--" to clrbrain
shift "$((OPTIND-1))"
EXTRA_ARGS="$@"

FOUND_NPZ=0
IMG_BASE=${IMG/.czi/_}$(printf %05d $SERIES)
NPZ_IMG=$IMG_BASE"_image5d.npz"
NPZ_INFO=$IMG_BASE"_info.npz"
for NPZ in $NPZ_IMG $NPZ_INFO
do
    if [ -e "$DEST"/"$NPZ" ]
    then
        echo "Found $DEST/$NPZ"
        FOUND_NPZ=1
    else
        echo "Could not find $DEST/$NPZ, checking on s3..."
        NPZ_LS=`aws s3 ls s3://"$S3_DIR"/"$NPZ"`
        if [ "$NPZ_LS" != "" ]
        then
            aws s3 cp s3://"$S3_DIR"/"$NPZ" "$DEST"/"$NPZ"
            ls -lh "$DEST"/"$NPZ"
            FOUND_NPZ=1
        else
            echo "Could not find $DEST/$NPZ on s3, checking original image..."
            if [ -e "$DEST"/"$IMG" ]
            then
                echo "Found $DEST/$IMG"
            else
                aws s3 cp s3://"$S3_DIR"/"$IMG" "$DEST"/"$IMG"
                ls -lh "$DEST"/"$IMG"
            fi
        fi
    fi
done

# import raw image into Numpy array if not available
if (( $FOUND_NPZ == 0)); then
    echo "Importing $DEST/$IMG..."
    python -m clrbrain.cli --img "$DEST"/"$IMG" --proc importonly $EXTRA_ARGS
fi

# process image and segments
python -u -m clrbrain.cli --img "$DEST"/"$IMG" --proc processing_mp $EXTRA_ARGS

# upload to S3
NPZ_IMG_PROC=$IMG_BASE"_image5d_proc.npz"
NPZ_INFO_PROC=$IMG_BASE"_info_proc.npz"
for NPZ in $NPZ_INFO_PROC #$NPZ_IMG_PROC 
do
    aws s3 cp "$DEST"/"$NPZ" s3://"$S3_DIR"/"$NPZ"
done

