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
  -d path/to/file: Set the destination path. Any existing file 
    of the same name will be moved backed up first. Defaults 
    to \"out.txt\".
  -o: Pass the output file as an argument (eg \"-o file.txt\"), 
    prepended to the extra arguments after its first arg.
  -q: Quiet mode to not show nohup log output.
"

dest="out.txt"
pass_output=0
quiet=0

# run from parent directory
BASE_DIR="$(dirname "$0")/.."
cd "$BASE_DIR" || { echo "Unable to find folder $BASE_DIR, exiting"; exit 1; }
echo "$PWD"

OPTIND=1
while getopts hd:oq opt; do
  case $opt in
    h)  echo "$HELP"
      exit 0
      ;;
    d)  dest="$OPTARG"
      echo "Set destination output path to $dest"
      ;;
    o)  pass_output=1
      echo "Set to pass output file to command that nohup will run"
      ;;
    q)  quiet=1
      echo "Set to be quiet"
      ;;
    :)  echo "Option -$OPTARG requires an argument"
      exit 1
      ;;
    --) ;;
    *)
      echo "$HELP" >&2
      exit 1
      ;;
  esac
done

# pass arguments after "--" to magmap
shift "$((OPTIND-1))"
EXTRA_ARGS="$@"

# create output filename based on dest
dest_base="${dest%.*}"
out_path="$dest"
echo "Output file: $out_path"
if [[ -e "$out_path" ]]; then
  # avoid overwriting existing file by appending next 
  # available integer
  out_base_last="${dest_base}(1)"
  i=2
  while [[ -e "${out_base_last}.txt" ]]; do
    out_base_last="${dest_base}(${i})"
    ((i++))
  done
  out_path_last="${out_base_last}.txt"
  mv "$out_path" "$out_path_last"
  echo "Backed up original $out_path to $out_path_last"
fi
# file may not have been created by time trying to show with tail
touch "$out_path"

# run rest of args in nohup and display output
if [[ $pass_output -eq 1 ]]; then
  # insert output arg after first extra arg in case EXTRA_ARGS 
  # contains additional extra args
  args=($EXTRA_ARGS)
  EXTRA_ARGS="${args[0]} -o ${out_path} ${args[@]:1}"
fi
nohup $EXTRA_ARGS > "$out_path" 2>&1 &
PID_NOHUP=$!
echo "Started \"$EXTRA_ARGS\" in nohup (PID $PID_NOHUP)"

if [[ "$quiet" -eq 0 ]]; then
  # show continuous updates from nohup log file
  tail -f "$out_path" &
  PID_TAIL=$!
  
  # in case process does in fact complete during this session, 
  # notify the user of completion
  while ps -p $PID_NOHUP > /dev/null; do
    sleep 1
  done
  kill $PID_TAIL
  echo "$PID_NOHUP completed, exiting."
fi
