#!/bin/bash
# Build MagellanMapper documentation
# Author: David Young 2018, 2019

HELP="
Build documentation files for MagellanMapper through the Sphinx package.

Arguments:
  -a: Rebuild API .rst files.
  -c: Clean docs before rebuilding.
  -h: Show help and exit.

Usage:
- Clean and rebuild all doc files, including auto-generation of .rst files:
  $0 -c -a
- Rebuild docs:
  $0
"

DOCS_DIR="docs"
build_api=0
clean=0

# run from project root directory
BASE_DIR="$(dirname "$0")/.."
cd "$BASE_DIR" || { echo "Unable to find folder $BASE_DIR, exiting"; exit 1; }
echo "$PWD"

OPTIND=1
while getopts hac opt; do
  case $opt in
    h)  
      echo "$HELP"
      exit 0
      ;;
    a)  
      build_api=1
      echo "Set to rebuild API .rst files"
      ;;
    c)  
      clean=1
      echo "Set to clean docs before building"
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

# setup Sphinx configuration
if [[ ! -e "$DOCS_DIR" ]]; then
  mkdir "$DOCS_DIR"
  cd "$DOCS_DIR" || { echo "unable to make/enter $DOCS_DIR"; exit 1; }
  sphinx-quickstart
else
  cd "$DOCS_DIR" || { echo "unable to enter $DOCS_DIR"; exit 1; }
fi

# (re)build .rst files
if [[ $build_api -eq 1 ]]; then
  sphinx-apidoc -f -o . ..
fi

# clean docs
if [[ $clean -eq 1 ]]; then
  make clean
fi

# build docs
make html
