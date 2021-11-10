#!/bin/bash
# Sets up the MagellanMapper environment using venv
# Author: David Young 2019

HELP="
Sets up the MagellanMapper environment using the Python built-in
venv environment manager.

Also handles package updates for existing environments.

Arguments:
  -h: Show help and exit.
  -e [path]: Path to folder where the new venv directory will be placed. 
    Defaults to \"../venvs\".
  -n [name]: Set the virtual environment name; defaults to CLR_ENV.
"

CLR_ENV="mag"
env_name="$CLR_ENV"
venv_dir="../venvs"

OPTIND=1
while getopts hn:e: opt; do
  case $opt in
    h)
      echo "$HELP"
      exit 0
      ;;
    n)
      env_name="$OPTARG"
      echo "Set to create the venv environment $env_name"
      ;;
    e)
      venv_dir="$OPTARG"
      echo "Set the venv directory to $venv_dir"
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

# run from script's parent directory
BASE_DIR="$(dirname "$0")/.."
cd "$BASE_DIR" || { echo "Unable to find folder $BASE_DIR, exiting"; exit 1; }
BASE_DIR="$PWD"

# load dependencies
source bin/libmag.sh

# find platform for Anaconda
detect_platform
ext="sh"
if [[ "$OS_NAME" = "Windows" ]]; then
  ext="ext"
fi

# check for Python availability and version requirement
py_ver_majmin="" # found Python version in x.y format
py_ver_min=(3 6) # minimum supported Python version
py_vers=(3.6 3.7 3.8) # range of versions currently supported
py_vers_prebuilt_deps=(3.6 3.7 3.8 3.9) # vers with custom prebuilt deps
for ver in "${py_vers[@]}"; do
  # prioritize specific versions in case "python" points to lower version,
  # calling Python directly since `command` will show Pyenv Python binaries
  # exist even if they will not work with a global/local version conflict
  if "python$ver" -V >/dev/null 2>&1; then
    python=python$ver
    py_ver_majmin="$ver"
    break
  fi
done
if [[ -z "$python" ]]; then
  # fallback to checking version number output by "python"
  if command -v python &> /dev/null; then
    if check_python python "${py_ver_min[@]}"; then
      python=python
      py_ver_majmin="$PY_VER"
    fi
  fi
  if [[ -z "$python" ]]; then
    echo "Sorry, could not detect a compatible Python version."
    echo "Please install one of the following Python versions: ${py_vers[*]}"
    exit 1
  fi
fi
echo "Found $python"

# Compiler and Java dependency checks
check_java # assume if java is not available, javac is not either
warn_prebuilt=true
for ver in "${py_vers_prebuilt_deps[@]}"; do
  if [[ "$ver" = "$py_ver_majmin" ]]; then
    warn_prebuilt=false
    break
  fi
done
if $warn_prebuilt; then
  # custom precompiled dependencies may not be available, in which case
  # compilers are required
  check_clt
  check_gcc
  #check_git # only necessary if deps req git
fi

if [[ ! -d "$venv_dir" ]]; then
  # create directory structure to hold new environment folder
  mkdir -p "$venv_dir"
fi

env_path="${venv_dir}/${env_name}"
env_act="${env_path}/bin/activate"
update=""
if [[ -e "$env_path" ]]; then
  # target venv dir exists
  if [[ ! -e "$env_act" ]]; then
    # assume that dir is not a venv dir if activation file is absent
    echo "$env_path exists but does not appear to be a venv."
    echo "Please choose another venv path. Exiting."
    exit 1
  fi
  
  # check whether user wishes to update existing directory
  read -r -p "$env_path already exists. Update all packages (y/n)? " update
  case "${update:0:1}" in
    y|Y )
      echo "Will update all packages..."
    ;;
    * )
      echo "Will only add new dependencies..."
    ;;
  esac
else
  # create new virtual environment
  "$python" -m venv "$env_path"
fi

# activate the environment
if [[ ! -e "$env_act" ]]; then
  # env generated in Windows does not contain bin folder; will need to 
  # change line endings if running in Cygwin (works as-is in MSYS2)
  env_act="${env_path}/Scripts/activate"
  if [[ ! -e "$env_act" ]]; then
    # check to ensure that the environment was actually created
    echo "Could not create new venv environment at $env_path"
    exit 1
  fi
fi
source "$env_act"
if [[ -z "$VIRTUAL_ENV" ]]; then
  echo "Could not activate new venv environment at $env_path, exiting"
  exit 1
fi

# update pip, using python since python3.y may not be in env, and wheel
# package to speed up installs  
python -m pip install -U pip
pip install -U wheel

# Install MagellanMapper including required dependencies

# Mayavi as of 4.7.3 does not supply wheels, and a wheel is built on the
# current VTK during installation; force rebuild rather than using any cached
# wheel since old builds may be incompatible with updated VTK versions
args_update=(--no-binary=mayavi)
if [[ -n "$update" ]]; then
  # update all dependencies based on setup.py
  args_update+=(--upgrade --upgrade-strategy eager)
fi
pip install "${args_update[@]}" -e .[all] --extra-index-url \
  https://pypi.fury.io/dd8/

echo "MagellanMapper environment setup complete!"
echo "** Please run \"source $env_act\" to enter your new environment **"
