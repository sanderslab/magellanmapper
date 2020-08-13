#!/bin/bash
# Build script to compile SimpleElastix
# Author: David Young, 2017, 2019

HELP="
Compiles SimpleElastix in a clean build directory with support for Mac and 
Linux with Python virtual environments.

Includes compiler flags to configure CMake to use clang on later Mac versions. 
Turns off virtual environment creation to avoid conflicts with Anaconda. 
Also includes flags to turn off all wrappers except Python and all 
example and test configurations.

See here for more build details:
https://github.com/sanderslab/magellanmapper/blob/master/docs/install.md#simpleelastix-dependency

Arguments:
  -h: Show help and exit.
  -d [path]: Set a build directory path. Relative paths are relative to 
    this script's directory. Default to \"../build_se\".
  -i: Install the Python wrapper from the build directory.
  -s [path]: Set a SimpleElastix repository path. Relative paths are relative 
    to this script's directory. Default to \"../SimpleElastix\".
"

build_dir=""
se_dir=""
PKG="SimpleITK-build/Wrapping/Python"
install_wrapper=0

OPTIND=1
while getopts hid:s: opt; do
  case $opt in
    h)
      echo "$HELP"
      exit 0
      ;;
    d)
      build_dir="$OPTARG"
      echo "Changed build directory to $build_dir"
      ;;
    s)
      se_dir="$OPTARG"
      echo "Changed SimpleElastix directory to $se_dir"
      ;;
    i)
      install_wrapper=1
      echo "Set to install Python wrapper"
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

echo "Initating build of SimpleElastix..."

# base directory is script's parent directory, and default repo and build dirs 
# are in parent directory of base dir
BASE_DIR="$(dirname "$0")/.."
cd "$BASE_DIR" || { echo "Unable to find folder $BASE_DIR, exiting"; exit 1; }
base_dir="$PWD"
if [[ -z "$se_dir" ]]; then
  se_dir="${base_dir}/../SimpleElastix"
fi
if [[ -z "$build_dir" ]]; then
  build_dir="${base_dir}/../build_se"
fi

# load dependencies
source "${base_dir}/bin/libmag.sh"

# find platform to determine compilers
detect_platform
compiler_c="gcc"
compiler_cpp="g++"
if [[ "$OS_NAME" = "MacOSX" ]]; then
  compiler_c=clang
  compiler_cpp=clang++
fi
msg="will use $compiler_c C compiler and $compiler_cpp C++ compiler "
msg+="for $OS_NAME platform"
echo -e "$msg"

if [[ ! -d "$se_dir" ]]; then
  # get SimpleElastix git repo
  echo "Cloning SimpleElastix git repo to ${se_dir}..."
  git clone https://github.com/SuperElastix/SimpleElastix.git "$se_dir"
fi
# get absolute path of SimpleElastix directory
cd "$se_dir" || { echo "Unable to find folder $se_dir, exiting"; exit 1; }
se_dir="$(pwd)"
cd - || exit 1

if [[ $install_wrapper -ne 1 ]] || [[ ! -d "${build_dir}/${PKG}" ]]; then
  # build SimpleElastix if not set to install or if the package doesn't exist
  backup_file "$build_dir"
  mkdir "$build_dir"
  cd "$build_dir" || { echo "Unable to enter $build_dir, exiting"; exit 1; }

  echo "Building SimpleElastix"
  # run SuperBuild with flags to find clang on later Mac versions,
  # turn off all wrappers except the default Python wrapper, and
  # turn off virtual environment creation
  cmake -DCMAKE_CXX_COMPILER:STRING=/usr/bin/$compiler_cpp \
    -DCMAKE_C_COMPILER:STRING=/usr/bin/$compiler_c \
    -DWRAP_JAVA:BOOL=OFF -DWRAP_LUA:BOOL=OFF \
    -DWRAP_R:BOOL=OFF -DWRAP_RUBY:BOOL=OFF \
    -DWRAP_TCL:BOOL=OFF -DWRAP_CSHARP:BOOL=OFF -DBUILD_EXAMPLES:BOOL=OFF \
    -DBUILD_TESTING:BOOL=OFF -DSimpleITK_PYTHON_USE_VIRTUALENV:BOOL=OFF \
    "${se_dir}/SuperBuild"

  # can change to -j1 for debugging to avoid multiple processes
  make -j4

  # build distributions
  echo "Generating source and platform wheel distributions..."
  cd "$PKG" || { echo "$PKG does not exist, exiting"; exit 1; }
  python Packaging/setup.py sdist
  python Packaging/setup.py bdist_wheel
  cd - || exit 1
fi


if [ $install_wrapper -eq 1 ]
then
  # install the newly or previously built Python wrapper
  echo "Installing Python wrapper..."
  pkg_dir="${build_dir}/${PKG}"
  cd "$pkg_dir" || { echo "$pkg_dir does not exist, exiting"; exit 1; }
  python Packaging/setup.py install
fi

echo "Done building SimpleElastix"
