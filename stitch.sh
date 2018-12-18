#!/bin/bash
# Stitch files using ImageJ/Fiji plugin
# Author: David Young 2017, 2018

HELP="
Stitch files using ImageJ/Fiji plugin.

Arguments:
  -h: Show help and exit.
  -f: Path to image file.
  -i [path]: Path to ImageJ/Fiji executable file. If not given, the 
      the path will be auto-detected, assuming that the app folder is in 
      /Applications on Mac or the parent folder of Clrbrain on Windows/Linux.
  -w [0|1|2]: 0 = Stitch but do not write fused file. Stitching 
      plugin will compute overlap and write registered tile 
      configuration. BigStitcher will import and calculate 
      alignments.
      1 = Write fused file only. Stitching plugin will use 
      coordinates from TileConfiguration.txt.
      2 = Stitch and write fused file(s).
  -s [none|stitching|bigstitcher]: Type of stitcher to use; 
      defaults to BigStitcher. Alternatives are Stitching (the 
      original ImageJ/Fiji plugin), or none. If none, ImageJ/Fiji 
      will simply be opened and left open for stitching review.

To run in normal (not BigStitcher) mode:
-Run the stitch/tile_config.py utility to build a positions
 configuration file since the .czi file may not contain
 position information, such as for Lightsheet files
-Sample run command, using nohup in case of long server
 operation: 
 nohup ./stitch.sh -f \"/path/to/img.czi\" > /path/to/output 2>&1 &
-Track results: \"tail -f /path/to/output\"
-Note: The resulting TileConfiguration.registered.txt file 
 places unregistered tiles at (0, 0, 0), which will reduce the
 intensity of the 1st tile. Please check this file to manually
 reposition tiles if necessary:
  -Edit TileConfiguration.registered.txt with positions from
   surrounding tiles
  -Move TileConfiguration.registered.txt to TileConfiguration.txt
  -Kill the current ImageJ process
  -Edit stitch/ij_stitch.py to remove "compute_overlap" option
  -Re-run this script

To run in BigStitcher mode:
-Run ./stitch.sh -f \"/path/to/img.czi\" -b
"

write_fused=2
out_dir=""
STITCH_TYPES=("none" "stitching" "bigstitcher")
stitch=STITCH_TYPES[2]
ij=""

OPTIND=1
while getopts hf:w:s:i: opt; do
    case $opt in
        h)  echo $HELP
            exit 0
            ;;
        f)  IMG="$OPTARG"
            echo "Set image file to $IMG"
            ;;
        w)  write_fused="$OPTARG"
            echo "Set fuse and write to $write_fused"
            ;;
        s)  stitch="$OPTARG"
            echo "Set stitch type to $stitch"
            ;;
        i)  ij="$OPTARG"
            echo "Set ImageJ/Fiji path to to $ij"
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

# run from script directory
BASE_DIR="`dirname $0`"
cd "$BASE_DIR"
BASE_DIR="$PWD"

# load dependencies
source libclr.sh

detect_platform
if [[ -z "$ij" ]]; then
    # auto-detect ImageJ binary path if not already set; assume that 
    # Fiji.app folder is in the standard Mac Applications directory (Mac)
    # or in the Clrbrain parent directory (Windows/Linux)
    if [[ "$os" = "Windows" ]]; then
        ij=../Fiji.app/ImageJ-win$bit
    elif [[ "$os" = "MacOSX" ]]; then
        ij=/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx
    elif [[ "$os" = "Linux" ]]; then
        ij=../Fiji.app/ImageJ-linux$bit
    fi
fi
echo "Assumes Fiji executable is located at $ij"

# calculates memory to reserve based on image file size with generous
# extra padding room (TODO: check if too much for large files)
if [[ "$stitch" == "${STITCH_TYPES[2]}" ]]; then
    mem=`free|awk '/Mem\:/ { print $2 }'`
    mem=$((mem/1000-2000))
else
    mem=`du "$IMG" | awk '{print $1}'`
    mem=$((mem/100))
fi
MIN_MEM=1000
if ((mem < MIN_MEM))
then
    # ensure a minimum amount of RAM
    mem=$MIN_MEM
fi
echo "Reserving $mem MB of memory"

if [[ "$stitch" == "${STITCH_TYPES[1]}" ]]; then
    # Fiji Stitching plugin; does not appear to work when fed a separate script in "-macro" mode
    "$ij" --mem "$mem"m --headless --run stitch/ij_stitch.py 'in_file="'"$IMG"'",write_fused="'"$write_fused"'"'
    
    # manually move files to output directory since specifying this directory
    # within the Stitching plugin requires the tile configuration file to be
    # there as well
    if [ $write_fused -eq 0 ] || [ "$out_dir" == "" ]
    then
        # exit if not writing fused files or no dir to move into
        exit 0
    fi
    # move into out_dir, assuming output files are in format "img_t..."
    in_dir="`dirname $IMG`"
    echo "Moving files from $in_dir to $out_dir..."
    if [ ! -e "$out_dir" ]
    then
        mkdir "$out_dir"
    fi
    mv "$in_dir"/img_t* "$out_dir"
elif [[ "$stitch" == "${STITCH_TYPES[2]}" ]]; then
    # BigStitcher; not working in headless mode so will require GUI availability
    "$ij" --ij2 --mem "$mem"m --run stitch/ij_bigstitch.py 'in_file="'"$IMG"'",write_fused="'"$write_fused"'"'
else
    # no stitching, just open ImageJ/Fiji
    "$ij" --ij2
fi

