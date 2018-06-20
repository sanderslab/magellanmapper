#!/bin/bash
# Process via nohup while tracking output
# Author: David Young 2018

HELP="
Run command in nohup, storing output in unique file and 
automatically printing output to screen.

Arguments given after \"--\" (or not recognized by this script) 
will be passed to nohup to run without hanging up if the 
current session is closed.

Arguments:
    -h: Show help documentation.
    -d [dir]: Set the destination directory. Output will be in a 
        unique file of the format, dir/out[n].txt.
    -o: Pass the output file as an argument (eg \"-o file.txt\"), 
        appended to the extra arguments.
"

DEST="."
pass_output=0

# run from parent directory
BASE_DIR="`dirname $0`"
cd "$BASE_DIR"
echo $PWD

OPTIND=1
while getopts hd:o opt; do
    case $opt in
        h)  echo "$HELP"
            exit 0
            ;;
        d)  DEST="$OPTARG"
            echo "Set destination output directory to $DEST"
            ;;
        o)  pass_output=1
            echo "Set to pass output file to command nohup will run"
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

# create output filename based on first arg in EXTRA_ARGS
out_name="${EXTRA_ARGS%% *}"
out_name="`basename ${EXTRA_ARGS%.*}`"
OUT_BASE="${DEST}/out_${out_name}"
out_path="${OUT_BASE}.txt"
echo "Output file: $out_path"
if [[ -e "$out_path" ]]; then
    # avoid overwriting existing file by appending next 
    # available integer
    out_base_last="${OUT_BASE}(1)"
    i=2
    while [[ -e "${out_base_last}.txt" ]]; do
        out_base_last="${OUT_BASE}(${i})"
        let i++
    done
    out_path_last="${out_base_last}.txt"
    mv "$out_path" "$out_path_last"
    echo "Backed up original $out_path to $out_path_last"
fi
# file may not have been created by time trying to show with tail
touch "$out_path"

# run rest of args in nohup and display output
if [[ $pass_output -eq 1 ]]; then
    EXTRA_ARGS+=" -o $out_path"
fi
nohup $EXTRA_ARGS > "$out_path" 2>&1 &
PID_NOHUP=$!
echo "Started \"$EXTRA_ARGS\" in nohup (PID $PID_NOHUP)"
tail -f "$out_path" &
PID_TAIL=$!

# in case process does in fact complete during this session, 
# notify the user of completion
while ps -p $PID_NOHUP > /dev/null; do
    sleep 1
done
kill $PID_TAIL
echo "$PID_NOHUP completed, exiting."
