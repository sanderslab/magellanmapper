#!/bin/bash
# Build script to compile SimpleElastix
# David Young, 2017

# Compiles SimpleElastix in a clean build directory on Mac
# in an Anaconda environment. Includes compiler flags to 
# configure CMake to use clang on later Mac versions. Turns off
# virtual environment creation to avoid conflicts with Anaconda, 
# so assumes the configuration/build process takes place within
# an Anaconda environment. Also includes flags to turn off all 
# wrappers except Python.

base_dir="`dirname $0`"
cd $base_dir/..

BUILD_DIR_BASE="build_se"

# create a new build directory with unique name
build_dir="$BUILD_DIR_BASE"
i=1
while [ -e "$build_dir" ]; do
    build_dir="$BUILD_DIR_BASE"$i
    let i++
done
mkdir "$build_dir"
echo "Created build directory: $build_dir"
cd "$build_dir"
pwd

# run SuperBuild with flags to find clang on later Mac versions, 
# turn off all wrappers except the default Python wrapper, and 
# turn off virtual environment creation
cmake -DCMAKE_CXX_COMPILER:STRING=/usr/bin/clang++ -DCMAKE_C_COMPILER:STRING=/usr/bin/clang -DWRAP_JAVA:BOOL=OFF -DWRAP_LUA:BOOL=OFF -DWRAP_R:BOOL=OFF -DWRAP_RUBY:BOOL=OFF -DWRAP_TCL:BOOL=OFF -DSimpleITK_PYTHON_USE_VIRTUALENV:BOOL=OFF ../SimpleElastix/SuperBuild
# change to -j1 for debugging to avoid multiple processes
make -j4