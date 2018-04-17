#!/bin/bash
# Process files on Amazon Web Services EC2 server
# Author: David Young 2017

HELP="
Imports files from S3 for processing on EC2 and upload
back to S3.

Arguments:
    -h: Show help and exit.
    -f: Full path to image file on local drive. The parent  
        directory is assumed to be the experiment directory 
        containing the image on S3.
    -s: S3 parent path, where the associated image files will be 
        assumed to be located in 
        \"s3://[your/s3/path]/[exp]/[name]\"
    -d: Path to destination directory; if not set, the 
        parent path to the image will be used instead.

Usage:
    - Ensure that Anaconda environment is activated: 
        \"source activate clr\"
    - Ensure that swap space and data drive are mounted
    - If running via an SSH session, consider running in nohup 
        or similar environment where the process will not 
        terminate if the session breaks, eg:
        \"nohup ./process_aws.sh -f /path/to/img.czi \\
        -s parent/S3/path -- --microscope type \\
        > /path/to/output 2>&1 &
    - Then track results with \"tail -f /path/to/output\"
    - If all goes well, pick up processed files from S3
"

DEST=""
IMG=""
S3_DIR=""
SERIES=0 # TODO: make settable
EXTRA_ARGS=""

# run from parent directory
BASE_DIR="`dirname $0`"
cd "$BASE_DIR"
echo $PWD

OPTIND=1
while getopts hf:s:d: opt; do
    case $opt in
        h)  echo "$HELP"
            exit 0
            ;;
        f)  IMG="$OPTARG"
            echo "Set image file to $IMG"
            ;;
        s)  S3_DIR="$OPTARG"
            echo "Set AWS S3 base path to $S3_DIR"
            ;;
        d)  DEST="$OPTARG"
            echo "Set destination path to $DEST"
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

# set destination to image directory unless DEST already set
if [ "$DEST" == "" ]
then
    DEST="`dirname $IMG`"
fi

# workaround for https://github.com/numpy/numpy/issues/5336,
# fixed in https://github.com/numpy/numpy/pull/7133, 
# released in Numpy 1.12.0
TMPDIR="$DEST"/tmp
if [ ! -e "$TMPDIR" ]
then
    mkdir -p "$TMPDIR"
fi
export TMPDIR="$TMPDIR"

FOUND_NPZ=0
# image base and NPZ filenames don't include parents in case 
# destination points to a different parent path
IMG_BASE=${IMG/.czi/_}$(printf %05d $SERIES)
IMG_BASE="`basename $IMG_BASE`"
EXP_PATH="`dirname $IMG`"
EXP="`basename $EXP_PATH`"
NPZ_IMG=$IMG_BASE"_image5d.npz"
NPZ_INFO=$IMG_BASE"_info.npz"
for NPZ in $NPZ_IMG $NPZ_INFO
do
    NPZ_PATH="${DEST}/${NPZ}"
    if [ -e "$NPZ_PATH" ]
    then
        echo "Found $NPZ_PATH"
        FOUND_NPZ=1
    else
        echo "Could not find ${NPZ_PATH}, checking on s3..."
        IMG_S3="${S3_DIR}/${EXP}/${NPZ}"
        NPZ_LS=`aws s3 ls s3://"$IMG_S3"`
        if [ "$NPZ_LS" != "" ]
        then
            aws s3 cp s3://"$IMG_S3" "$NPZ_PATH"
            ls -lh "$NPZ_PATH"
            FOUND_NPZ=1
        else
            echo "Could not find $IMG_S3 on s3, checking original image..."
            if [ -e "$IMG" ]
            then
                echo "Found $IMG"
            else
                aws s3 cp s3://"${S3_DIR}/${EXP}/`basename $IMG`" "$IMG"
                ls -lh "$IMG"
            fi
        fi
    fi
done

# import raw image into Numpy array if not available
if (( $FOUND_NPZ == 0)); then
    echo "Importing $IMG..."
    python -u -m clrbrain.cli --img "$IMG" --proc importonly $EXTRA_ARGS
fi

# process image and segments
python -u -m clrbrain.cli --img "$IMG" --proc processing_mp $EXTRA_ARGS

# upload to S3
NPZ_IMG_PROC=$IMG_BASE"_image5d_proc.npz"
NPZ_INFO_PROC=$IMG_BASE"_info_proc.npz"
for NPZ in $NPZ_INFO_PROC #$NPZ_IMG_PROC 
do
    aws s3 cp "${DEST}/${NPZ}" s3://"${S3_DIR}/${EXP}/${NPZ}"
done

