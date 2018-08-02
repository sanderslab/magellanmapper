#!/bin/bash
# Build script to compile SimpleElastix
# Author: David Young, 2017, 2018

################################################
# Compiles SimpleElastix in a clean build directory on Mac
# in an Anaconda environment. Includes compiler flags to 
# configure CMake to use clang on later Mac versions. Turns off
# virtual environment creation to avoid conflicts with Anaconda, 
# so assumes the configuration/build process takes place within
# an Anaconda environment. Also includes flags to turn off all 
# wrappers except Python.
#
# Assumptions:
# -If SimpleElastix git repository already exists, it might be
#  usable, but you might want to move it aside to download
#  a fresh clone or update them manually
################################################

BUILD_DIR_BASE="build_se"
PKG="SimpleITK-build/Wrapping/Python/Packaging"
install_wrapper=0

OPTIND=1
while getopts hi opt; do
    case $opt in
        h)  echo $HELP
            exit 0
            ;;
        i)  install_wrapper=1
            echo "Set to install Python wrapper"
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

echo "Building SimpleElastix..."

# base directory is parent of script's directory
base_dir="`dirname $0`"
cd $base_dir/..

# find platform for Anaconda
echo -n "Detecting environment..."
SYSTEM=`uname -a`
compiler_c="gcc"
compiler_cpp="g++"
platform=""
if [[ "$SYSTEM" =~ "CYGWIN" ]] || [[ "$SYSTEM" =~ "WINDOWS" ]]
then
    platform="Windows"
elif [[ "$SYSTEM" =~ "Darwin" ]]
then
    platform="MacOSX"
    compiler_c=clang
    compiler_cpp=clang++
elif [[ "$SYSTEM" =~ "Linux" ]]
then
    platform="Linux"
fi
bit="x86"
if [[ "$SYSTEM" =~ "x86_64" ]]
then
    bit="x86_64"
fi
echo "will use $compiler_c C compiler and $compiler_cpp C++ compiler"
echo "for $platform platform"

# get SimpleElastix git repo if not already present
if [[ ! -e SimpleElastix ]]
then
    echo "Cloning SimpleElastix git repo..."
    #git clone https://github.com/SuperElastix/SimpleElastix.git
    git clone https://github.com/the4thchild/SimpleElastix.git
fi

# build SimpleElastix if install flag is false, in which case 
# the package will only be installed if possible, or if the 
# package folder doesn't exist
build_dir="$BUILD_DIR_BASE"
if [[ $install_wrapper -ne 1 ]] || [[ ! -e "${build_dir}/${PKG}" ]]; then
    # backup old build directories if necessary and create a new one
    if [[ -e "$build_dir" ]]; then
        build_dir_last="${BUILD_DIR_BASE}1"
        i=2
        while [ -e "$build_dir_last" ]; do
            build_dir_last="$BUILD_DIR_BASE"$i
            let i++
        done
        mv "$build_dir" "$build_dir_last"
    fi
    mkdir "$build_dir"
    echo "Created build directory: $build_dir"
    
    cd "$build_dir"
    pwd
    
    echo "Building SimpleElastix"
    # run SuperBuild with flags to find clang on later Mac versions, 
    # turn off all wrappers except the default Python wrapper, and 
    # turn off virtual environment creation
    cmake -DCMAKE_CXX_COMPILER:STRING=/usr/bin/$compiler_cpp \
        -DCMAKE_C_COMPILER:STRING=/usr/bin/$compiler_c \
        -DWRAP_JAVA:BOOL=OFF -DWRAP_LUA:BOOL=OFF \
        -DWRAP_R:BOOL=OFF -DWRAP_RUBY:BOOL=OFF \
        -DWRAP_TCL:BOOL=OFF \
        -DSimpleITK_PYTHON_USE_VIRTUALENV:BOOL=OFF \
        ../SimpleElastix/SuperBuild
    
    # change to -j1 for debugging to avoid multiple processes
    make -j4
    
    cd ..
fi

# install the Python wrapper if flagged
if [ $install_wrapper -eq 1 ]
then
    echo "Installing Python wrapper..."
    cd "${build_dir}/${PKG}"
    python setup.py install
fi

echo "Done building SimpleElastix"
