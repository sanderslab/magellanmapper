#!/bin/bash
# Stitch files using ImageJ/Fiji plugin
# Author: David Young 2017

################################################
# Stitch files using ImageJ/Fiji plugin.
#
# To run:
# -Sample run command, using nohup in case of long server
#  operation: 
#  nohup ./stitch.sh -f "/path/to/img.czi" -s "root/path" \
#    -o "/output/dir" &
# -Track results: "tail -f nohup.out"
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

# evaluates the options directly from command-line;
# does not appear to work when fed a separate script in "-macro" mode
$IJ --mem 100000m --headless -eval 'run("Grid/Collection stitching", "type=[Positions from file] order=[Defined by image metadata] multi_series_file='"$IMG"' output_directory='"$OUT_DIR"' fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 increase_overlap=0 compute_overlap use_virtual_input_images computation_parameters=[Save memory (but be slower)] image_output=[Write to disk]"); -batch'