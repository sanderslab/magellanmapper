#!/bin/bash
# Process via nohup while tracking output
# Author: David Young 2018

HELP="
Run command in nohup, storing output in unique file and 
automatically printing output to screen.

Arguments:
    -h: Show help documentation.
    -d: Set the destination directory. Output will be in a 
        unique file of the format, out[n].txt.
"

DEST="."

# run from parent directory
BASE_DIR="`dirname $0`"
cd "$BASE_DIR"
echo $PWD

OPTIND=1
while getopts hd: opt; do
    case $opt in
        h)  echo "$HELP"
            exit 0
            ;;
        d)  DEST="$OPTARG"
            echo "Set destination output directory to $DEST"
            ;;
        :)  echo "Option -$OPTARG requires an argument"
            exit 1
            ;;
        --) ;;
    esac
done
readonly DEST

# pass arguments after "--" to clrbrain
shift "$((OPTIND-1))"
EXTRA_ARGS="$@"

# get output filename, avoiding overwriting existing file
OUT_BASE="${DEST}/out"
out_path="${OUT_BASE}.txt"
if [[ -e "$out_path" ]]; then
    out_base_last="${OUT_BASE}1"
    i=2
    while [ -e "${out_base_last}.txt" ]; do
        out_base_last="${OUT_BASE}"$i
        let i++
    done
    out_path="${out_base_last}.txt"
fi
echo "Output file: $out_path"

# run rest of args in nohup and display output
nohup $EXTRA_ARGS > "$out_path" 2>&1 &
HISPID=$!
echo "Started process $HISPID in nohup"
tail -f "$out_path"
