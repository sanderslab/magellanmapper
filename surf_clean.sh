#!/bin/bash
# 3D object clean-up with MeshLab
# Author: David Young 2018

HELP="
Clean up 3D object surfaces using MeshLab scripting.

Will attempt to identify the MeshLab path based on the current 
platform to run the meshlabserver command-line executable.

Arguments:
  -h: Show help and exit.
  -m [path[: Path to image MeshLab root directory.
  -i [path]: Path to input file for MeshLab script.
  -o [path]: Path to output file for MeshLab script.
  -s [path]: Path to MeshLab script.
"

# run from parent directory
base_dir="`dirname $0`"
cd "$base_dir"
echo $PWD

path_meshlab=""
path_meshlab_server=""
path_input=""
path_output=""
path_script=""

OPTIND=1
while getopts hm:i:o:s: opt; do
  case $opt in
    h) 
      echo $HELP
      exit 0
      ;;
    m)  
      path_meshlab="$OPTARG"
      echo "Set MeshLab binary path to $path_meshlab"
      ;;
    i)  
      path_input="$OPTARG"
      echo "Set input path to $path_input"
      ;;
    o)  
      path_output="$OPTARG"
      echo "Set output path to $path_output"
      ;;
    s)  
      path_script="$OPTARG"
      echo "Set script path to $path_script"
      ;;
    :)  
      echo "Option -$OPTARG requires an argument"
      exit 1
      ;;
    --) ;;
  esac
done

# pass arguments after "--" to magmap
shift "$((OPTIND-1))"
EXTRA_ARGS="$@"

echo -n "Detecting environment..."
SYSTEM=`uname -a`
bit="32"
if [[ "$SYSTEM" =~ "x86_64" ]]
then
  bit="64"
fi
platform=""
if [[ "$SYSTEM" =~ "CYGWIN" ]] || [[ "$SYSTEM" =~ "WINDOWS" ]]
then
  platform="Windows"
  # TODO: find path on Windows
  path_meshlab_server="${path_meshlab}/meshlabserver"
elif [[ "$SYSTEM" =~ "Darwin" ]]
then
  platform="MacOSX"
  if [[ "$path_meshlab" = "" ]]; then
    path_meshlab="/Applications/meshlab.app"
  fi
  export DYLD_FRAMEWORK_PATH="${path_meshlab}/Contents/Frameworks"
  path_meshlab_server="${path_meshlab}/Contents/MacOS/meshlabserver"
elif [[ "$SYSTEM" =~ "Linux" ]]
then
  platform="Linux"
  # TODO: find path on Linux
  path_meshlab_server="${path_meshlab}/meshlabserver"
fi
echo "MeshLab set up for $platform platform"
echo "Assumes MeshLab server is located at $path_meshlab_server"

echo "Apply filter from $path_script to $path_input..."
"$path_meshlab_server" -i "$path_input" -o "$path_output" -m vc vn fc -s "$path_script"
