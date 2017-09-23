#!/bin/bash
# Stitch files using ImageJ/Fiji plugin
# Author: David Young 2017

################################################
# Stitch files using ImageJ/Fiji plugin.
#
# To run:
# -Run the stitch/tile_config.py utility to build a positions
#  configuration file since the .czi file may not contain
#  position information, such as for Lightsheet files
# -Sample run command, using nohup in case of long server
#  operation: 
#  nohup ./stitch.sh -f "/path/to/img.czi" > /path/to/output 2>&1 &
# -Track results: "tail -f /path/to/output"
# -Note: The resulting TileConfiguration.registered.txt file 
#  places unregistered tiles at (0, 0, 0), which will reduce the
#  intensity of the 1st tile. Please check this file to manually
#  reposition tiles if necessary:
#   -Edit TileConfiguration.registered.txt with positions from
#    surrounding tiles
#   -Copy TileConfiguration.registered.txt to TileConfiguration.txt
#    (TODO: check if necessary)
#   -Kill the current ImageJ process
#   -Edit stitch/ij_stitch.py to remove "compute_overlap" option
#   -Re-run this script
################################################

# run from parent directory
base_dir="`dirname $0`"
cd "$base_dir"
echo $PWD

OPTIND=1
while getopts hf:o: opt; do
    case $opt in
        h)  echo $HELP
            exit 0
            ;;
        f)  IMG="$OPTARG"
            echo "Set image file to $IMG"
            ;;
        o)  OUT_DIR="$OPTARG"
            echo "Set output directory to $OUT_DIR"
            echo "NOTE: This path will be ignored for now"
            echo "and based on image directory instead"
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

# find ImageJ binary name; assumes binary is in the PATH
echo -n "Detecting environment..."
SYSTEM=`uname -a`
bit="32"
if [[ "$SYSTEM" =~ "x86_64" ]]
then
    bit="64"
fi
platform=""
if [[ "$SYSTEM" =~ "CYGWIN" ]] || [[ "$SYSTEM" =~ "WINDOWS" ]]
then
    platform="Windows"
    IJ=Fiji.app/ImageJ-win$bit
elif [[ "$SYSTEM" =~ "Darwin" ]]
then
    platform="MacOSX"
    IJ=/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx
elif [[ "$SYSTEM" =~ "Linux" ]]
then
    platform="Linux"
    IJ=Fiji.app/ImageJ-linux$bit
fi
echo "will use $platform platform with $bit bit for ImageJ"
echo "Assumes Fiji executable is located at $IJ"

# calculates memory to reserve based on image file size with generous
# extra padding room (TODO: check if too much for large files)
mem=`du "$IMG" | awk '{print $1}'`
mem=$((mem/100))
echo "Reserving $mem MB of memory"

# evaluates the options directly from command-line;
# does not appear to work when fed a separate script in "-macro" mode
$IJ --mem "$mem"m --headless --run stitch/ij_stitch.py 'in_file="'"$IMG"'"'
