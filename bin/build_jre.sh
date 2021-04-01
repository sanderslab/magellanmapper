#!/usr/bin/env bash
# Extract a distributable Java Runtime Environment for MagellanMapper
# dependencies

HELP="
Build a distributable Java Runtime Environment using jlink for MagellanMapper
dependencies.

Arguments:
  -h: Show help and exit.
  -o [path]: Output directory; defaults to \"jre_<platform>\".
"

output_dir=""

OPTIND=1
while getopts ho: opt; do
  case $opt in
    h)
      echo "$HELP"
      exit 0
      ;;
    o)
      output_dir="$OPTARG"
      echo "Set output directory to $output_dir"
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

# load libmag
BASE_DIR="$(dirname "$0")"
. "${BASE_DIR}/libmag.sh"

# get platform name
detect_platform
platform="$(echo "$OS_NAME" | tr '[:upper:]' '[:lower:]')"

if [[ -z "$output_dir" ]]; then
  # set default output directory
  output_dir="jre_${platform}"
fi
backup_file "$output_dir"

# build JRE with modules required for Bioformats package
build_jre \
  "${VIRTUAL_ENV}/lib/python3.6/site-packages/bioformats/jars/loci_tools.jar" \
  "$output_dir"
