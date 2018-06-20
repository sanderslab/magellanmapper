#!/bin/bash
# Prepare run script based on sample script
# Author: David Young 2018

HELP="
Replace defaults in sample script with given arguments.

Arguments:
    -h: Show help and exit.
    -s: Path with which to replace S3_DIR.
    -i: Path with which to replace IMG.
    -a: Abbreviation of image, used to name the output 
        script.
"

RUN_SCRIPT="runclrbrain.sh"
s3_dir=""
img=""
abbr=""
url_notify=""

OPTIND=1
while getopts hs:i:a:n: opt; do
    case $opt in
        h)  echo $HELP
            exit 0
            ;;
        s)  s3_dir="$OPTARG"
            echo "Set S3 directory to $s3_dir"
            ;;
        i)  img="$OPTARG"
            echo "Set image path to $img"
            ;;
        a)  abbr="$OPTARG"
            echo "Set image abbreviation to $abbr"
            ;;
        n)  url_notify="$OPTARG"
            echo "Set notification URL to $url_notify"
            ;;
        --) ;;
    esac
done

# pass arguments after "--" to another script if necessary
shift "$((OPTIND-1))"
EXTRA_ARGS="$@"

# appendn abbreviation to run script filename before ext
output="${RUN_SCRIPT%.*}_${abbr}.sh"

# replace S3_DIR and IMG variable assignments
sed -e "s:IMG=\"/path.*:IMG=\"${img}\":" \
    -e "s:S3_DIR=.*:S3_DIR=\"${s3_dir}\":" \
    -e "s,url_notify=.*,url_notify=\"${url_notify}\"," \
    "$RUN_SCRIPT" > "$output"
chmod 755 "$output"

echo "Updated run script, saved at $output"

exit 0
