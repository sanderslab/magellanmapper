#!/bin/bash
# Process files on Amazon Web Services EC2 server
# Author: David Young 2017

DEST=/data
IMG=""
S3_DIR=""
SERIES=0
CHANNEL=1
EXTRA_ARGS=""

PAR_IMGNAME="imgname"
PAR_S3="s3"

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

if [ $# -gt 0 ]
then
	echo "Parsing user arguments..."
	for arg in "$@"
	do
		# reads arguments
		if [ "x$arg" = "x--help" -o "x$arg" = "x-h" ] # help docs
		then
			if [ "`command -v more`" != '' ]
			then
				echo "$HELP" | more
			elif [ "`command -v less`" != "" ]
			then
				echo "$HELP" | less
			else
				echo "$HELP"
			fi
			exit 0
			
		# image filename
		elif [ ${arg:0:${#PAR_IMGNAME}} = "$PAR_IMGNAME" ]
		then
			IMG="${arg#${PAR_IMGNAME}=}"
			echo "...set to use \"$IMG\" as the image filename"
		# S3 path
		elif [ ${arg:0:${#PAR_S3}} = "$PAR_S3" ]
		then
			S3_DIR="${arg#${PAR_S3}=}"
			echo "...set to use \"$S3_DIR\" as the S3 directory path"
		# extra arguments
		else
			EXTRA_ARGS+=" $arg"
			echo "...adding \"$arg\" to clrbrain arguments"
		fi
	done
fi

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
python -u -m clrbrain.cli --img "$DEST"/"$IMG" --proc processing_mp -v $EXTRA_ARGS

# upload to S3
NPZ_IMG_PROC=$IMG_BASE"_image5d_proc.npz"
NPZ_INFO_PROC=$IMG_BASE"_info_proc.npz"
for NPZ in $NPZ_IMG_PROC $NPZ_INFO_PROC
do
	aws s3 cp "$DEST"/"$NPZ" s3://"$S3_DIR"/"$NPZ"
done

