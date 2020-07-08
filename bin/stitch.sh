#!/bin/bash
# Stitch files using ImageJ/Fiji plugin
# Author: David Young 2017, 2020

HELP="
Stitch files using ImageJ/Fiji plugin.

Arguments:
  -h: Show help and exit.
  -f [path]: Path to image file.
  -j [path]: Path to custom JAVA_HOME for ImageJ/Fiji. If not given 
    empty, the JAVA_HOME environment variable will be used instead.
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

To run in \"BigStitcher\" mode (default):
-Run ./stitch.sh -f /path/to/img.czi

To run in \"stitching\" mode:
-Run the \"stitch/tile_config.py\" utility to build a positions
 configuration file since the .czi file may not contain
 position information, such as for Lightsheet files
-Sample run command, using nohup in case of long server
 operation: 
 nohup ./stitch.sh -f /path/to/img.czi > /path/to/output 2>&1 &
-Track results: \"tail -f /path/to/output\"
-Note: The resulting TileConfiguration.registered.txt file 
 places unregistered tiles at (0, 0, 0), which will reduce the
 intensity of the 1st tile. Please check this file to manually
 reposition tiles if necessary:
  -Edit TileConfiguration.registered.txt with positions from
   surrounding tiles
  -Move TileConfiguration.registered.txt to TileConfiguration.txt
  -Kill the current ImageJ process
  -Edit stitch/ij_stitch.py to remove \"compute_overlap\" option
  -Re-run this script
"

write_fused=2
out_dir=""
STITCH_TYPES=("none" "stitching" "bigstitcher")
stitch="${STITCH_TYPES[2]}"
java_home="$JAVA_HOME"

OPTIND=1
while getopts hf:w:s:j: opt; do
  case $opt in
    h)
      echo "$HELP"
      exit 0
      ;;
    f)
      IMG="$OPTARG"
      echo "Set image file to $IMG"
      ;;
    w)
      write_fused="$OPTARG"
      echo "Set fuse and write to $write_fused"
      ;;
    s)
      stitch="$OPTARG"
      echo "Set stitch type to $stitch"
      ;;
    j)
      java_home="$OPTARG"
      echo "Set JAVA_HOME for ImageJ/Fiji to to $java_home"
      ;;
    :)
      echo "Option -$OPTARG requires an argument"
      exit 1
      ;;
    *)
      echo "$HELP" >&2
      exit 1
      ;;
  esac
done

# run from script's parent directory
BASE_DIR="$(dirname "$0")/.."
cd "$BASE_DIR" || { echo "Unable to find folder $BASE_DIR, exiting"; exit 1; }
BASE_DIR="$PWD"

# load dependencies
source bin/libmag.sh

detect_platform
# auto-detect ImageJ binary path if not already set; assume that 
# Fiji.app folder is in the standard Mac Applications directory (Mac)
# or in the MagellanMapper parent directory (Windows/Linux)
bit_short="64"
if [[ "$BIT" =~ "32" ]]; then
  bit_short="32"
fi
if [[ "$OS_NAME" = "Windows" ]]; then
  ij="../Fiji.app/ImageJ-win$bit_short"
elif [[ "$OS_NAME" = "MacOSX" ]]; then
  ij="/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx"
elif [[ "$OS_NAME" = "Linux" ]]; then
  ij="../Fiji.app/ImageJ-linux$bit_short"
fi

# calculates memory to reserve based on image file size with generous
# extra padding room (TODO: check if too much for large files)
if [[ "$stitch" == "${STITCH_TYPES[1]}" ]]; then
  # old stitching plugin, which requires a ton of memory
  mem=$(du "$IMG" | awk '{print $1}')
  mem=$((mem/100))
elif [[ "$stitch" == "${STITCH_TYPES[2]}" ]]; then
  # BigStitcher plugin, which is more memory efficient
  if [[ "$OS_NAME" = "MacOSX" ]]; then
    mem=$(sysctl -a | awk '/hw.memsize\:/ {print $2}')
    mem=$((mem/1024))
  else
    mem=$(free|awk '/Mem\:/ { print $2 }')
  fi
  mem=$((mem/1024*8/10))
fi
MIN_MEM=1000
if ((mem < MIN_MEM))
then
  # ensure a minimum amount of RAM
  mem=$MIN_MEM
fi
echo "Reserving $mem MB of memory"

# setup ImageJ execution with args for JVM
ij=(
  "$ij" 
  "-XX:+HeapDumpOnOutOfMemoryError" 
  "-XX:HeapDumpPath=$(dirname "$IMG")/heapdump.hprof"
  "-Xmx${mem}m"
  "-Xms${mem}m"
  "--"
)
if [[ -n "$java_home" ]]; then
  # specify Java home directly in ImageJ rather than as env var
  ij+=("--java-home" "$java_home")
fi
# add another memory flag since unclear which ones are actually used
ij+=("--mem" "${mem}m")
echo "Will run ImageJ/Fiji executable as:"
echo "${ij[@]}"

if [[ "$stitch" == "${STITCH_TYPES[1]}" ]]; then
  # Fiji Stitching plugin; does not appear to work when fed a separate script 
  # in "-macro" mode
  "${ij[@]}" --headless --run stitch/ij_stitch.py \
    'in_file="'"$IMG"'",write_fused="'"$write_fused"'"'
  
  # manually move files to output directory since specifying this directory
  # within the Stitching plugin requires the tile configuration file to be
  # there as well
  if [ "$write_fused" -eq 0 ] || [ "$out_dir" == "" ]
  then
    # exit if not writing fused files or no dir to move into
    exit 0
  fi
  # move into out_dir, assuming output files are in format "img_t..."
  in_dir="$(dirname "$IMG")"
  echo "Moving files from $in_dir to $out_dir..."
  if [ ! -e "$out_dir" ]
  then
    mkdir "$out_dir"
  fi
  mv "$in_dir"/img_t* "$out_dir"
elif [[ "$stitch" == "${STITCH_TYPES[2]}" ]]; then
  # BigStitcher; not working in headless mode so will require GUI; 
  # --ij2 flag must follow --mem flag or else --mem ignored; 
  # TODO: check if need --ij2 flag
  "${ij[@]}" --ij2 --run stitch/ij_bigstitch.py \
    'in_file="'"$IMG"'",write_fused="'"$write_fused"'"'
else
  # no stitching, just open ImageJ/Fiji
  "${ij[@]}" --ij2
fi
