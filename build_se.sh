#!/bin/bash
# Build script to compile SimpleElastix
# Author: David Young, 2017, 2019

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

build_dir_base="build_se"
PKG="SimpleITK-build/Wrapping/Python"
install_wrapper=0

OPTIND=1
while getopts hid: opt; do
    case $opt in
        h)  echo $HELP
            exit 0
            ;;
        d)  build_dir_base="$OPTARG"
            echo "Changed build directory to $build_dir_base"
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
cd "$base_dir"
base_dir="$PWD"
cd "${base_dir}/.."

# load dependencies
source "${base_dir}/libclr.sh"

# find platform for Anaconda
detect_platform
compiler_c="gcc"
compiler_cpp="g++"
if [[ "$os" = "MacOSX" ]]; then
    compiler_c=clang
    compiler_cpp=clang++
fi
echo "will use $compiler_c C compiler and $compiler_cpp C++ compiler"
echo "for $os platform"

build_dir_parent="$(dirname "$build_dir_base")"
cd "$build_dir_parent"

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
build_dir="$(basename "$build_dir_base")"
if [[ $install_wrapper -ne 1 ]] || [[ ! -e "${build_dir}/${PKG}" ]]; then
    # backup old build directory if necessary and create a new one
    backup_file "$build_dir"
    mkdir "$build_dir"
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
    python Packaging/setup.py install
fi

echo "Done building SimpleElastix"
