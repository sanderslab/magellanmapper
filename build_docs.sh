#!/bin/bash
# Build Clrbrain documentation

DOCS_DIR="docs"
build_api=0
clean=0

# run from project root directory
BASE_DIR="`dirname $0`"
cd "$BASE_DIR"
echo $PWD

OPTIND=1
while getopts hac opt; do
    case $opt in
        h)  echo $HELP
            exit 0
            ;;
        a)  build_api=1
            echo "Set to rebuild API .rst files"
            ;;
        c)  clean=1
            echo "Set to clean docs before building"
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

# setup Sphinx configuration
if [ ! -e "$DOCS_DIR" ]
then
    mkdir "$DOCS_DIR"
    cd "$DOCS_DIR"
    sphinx-quickstart
else
    cd "$DOCS_DIR"
fi

# (re)build .rst files
if [ $build_api -eq 1 ]
then
    sphinx-apidoc -f -o . ..
fi

# clean docs
if [ $clean -eq 1 ]
then
    make clean
fi

# build docs
make html