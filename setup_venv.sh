#!/bin/bash
# Sets up the Clrbrain environment using venv
# Author: David Young 2019

HELP="
Sets up the initial Clrbrain environment using the built-in
venv environment manager.

Arguments:
  -h: Show help and exit.
  -d: Skip dependencies check before environment setup.
  -e [path]: Path to folder where the new venv directory will be placed.
  -n [name]: Set the Conda environment name; defaults to CONDA_ENV.
  -s: Build and install SimpleElastix.
"

CLR_ENV="clr"
env_name="$CLR_ENV"
venv_dir="../venvs"
python="python"

build_simple_elastix=0
deps_check=1

OPTIND=1
while getopts hn:sde: opt; do
  case $opt in
    h)
      echo $HELP
      exit 0
      ;;
    n)
      env_name="$OPTARG"
      echo "Set to create the venv environment $env_name"
      ;;
    s)
      build_simple_elastix=1
      echo "Set to build and install SimpleElastix"
      ;;
    e)
      venv_dir="$OPTARG"
      echo "Set the venv directory to $venv_dir"
      ;;
    d)
      deps_check=0
      echo "Set to skip dependencies check"
      ;;
    :)
      echo "Option -$OPTARG requires an argument"
      exit 1
      ;;
    --) ;;
  esac
done

# pass arguments after "--" to clrbrain
shift "$((OPTIND-1))"
EXTRA_ARGS="$@"

# run from script directory
BASE_DIR="`dirname $0`"
cd "$BASE_DIR"
BASE_DIR="$PWD"

# load dependencies
source libclr.sh

# find platform for Anaconda
detect_platform
ext="sh"
if [[ "$os" = "Windows" ]]; then
  ext="ext"
fi

# Dependencies check
if [[ $deps_check -eq 1 ]]; then
  
  # check for Java jar availability
  if ! command -v "javac" &> /dev/null
  then
  	echo "Please install a JDK or add JAVA_HOME to your path environment "
  	echo "variables. Exiting."
  	exit 1
  fi
  
  # Mac-specific check for command-line tools (CLT) package since the commands 
  # that are not activated will still return
  if [[ "$os" == "MacOSX" ]]; then
    if [[ ! -e "/Library/Developer/CommandLineTools/usr/bin/git" ]]; then
      if [[ "$os_ver" < "10.14" && -e "/usr/include/iconv.h" ]]; then
        # ver <= 10.13 apparently also requires CLT headers here
        :
      else
        echo "Mac command-line tools not present/activated."
        echo "Please run \"xcode-select --install\""
        exit 1
      fi
    fi
  fi
fi

# check for Python availability
if command -v "python" &> /dev/null; then
  echo "Python found..."
elif command -v "python3" &> /dev/null; then
  echo "Python found at python3..."
  python="python3"
else
  echo "Please install Python (version 3.6+)"
  exit 1
fi

py_ver="$("$python" -V 2>&1)"
py_ver="${py_ver#* }"
py_ver_maj="${py_ver%%.*}"
py_ver_rest="${py_ver#*.}"
py_ver_min="${py_ver_rest%%.*}"

if [[ $py_ver_maj -lt 3 || $py_ver_min -lt 6 ]]; then
  echo "Please install Python >= 3.6"
  exit 1
fi

if [[ ! -d "$venv_dir" ]]; then
  mkdir "$venv_dir"
fi
env_path="${venv_dir}/${env_name}"
if [[ -e "$env_path" ]]; then
  echo "$env_path already exists. Please give a different venv directory name."
  exit 1
fi
"$python" -m venv "$env_path"
env_act="${env_path}/bin/activate"
if [[ ! -e "$env_act" ]]; then
  echo "Could not create new venv environment at $env_path"
  exit 1
fi
source "$env_act"

# update pip and install all dependencies for Clrbrain
pip install -U pip
pip install -r requirements.txt


if [[ $build_simple_elastix -eq 1 ]]; then
  # build and install SimpleElastix
  ./build_se.sh -i
else
  # install SimpleITK if not installing SimpleElastix and not a 
  # lightweight install to allow opening files through SimpleITK
  pip install simpleitk
fi

echo "Clrbrain environment setup complete!"
echo "** Please run \"$env_act\" to enter your new environment **"

