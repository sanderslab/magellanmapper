#!/bin/bash

DEST=/data
IMG=""
S3_DIR=""
SERIES=0
CHANNEL=1
if [ ! -e "$DEST"/"$IMG" ]
then
	echo "Could not find $DEST/$IMG, checking for .npz file..."
	NPZ="$IMG"$(printf %05d $SERIES).npz
	if [ ! -e "$DEST"/"$NPZ" ]
	then
		echo "Could not find $DEST/$NPZ, checking on s3..."
		NPZ_LS=`aws s3 ls s3://"$S3_DIR"/"$NPZ"`
		if [ "$NPZ_LS" != "" ]
		then
			aws s3 cp s3://"$S3_DIR"/"$NPZ" "$DEST"/"$NPZ"
			ls -lh "$DEST"/"$NPZ"
		else
			aws s3 cp s3://"$S3_DIR"/"$IMG" "$DEST"/"$IMG"
			ls -lh "$DEST"/"$IMG"
		fi
	fi
fi
./run img="$DEST"/"$IMG" 3d=importonly channel=$CHANNEL series=$SERIES

