#!/usr/bin/env bash
# Script to set up Venv environments for multiple Python versions

HELP="
Create Venv environments for multiple Python versions.

Arguments:
  -h: Show help and exit.
  -d [path]: Path to folder where the new venv directory will be placed. 
    Defaults to \"../venvs\".
"

venv_dir="../venvs"

OPTIND=1
while getopts hd: opt; do
  case $opt in
    h)
      echo "$HELP"
      exit 0
      ;;
    d)
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

echo "Initating build of Venvs for multiple Python versions..."

# base directory is script's parent directory
BASE_DIR="$(dirname "$0")/.."
cd "$BASE_DIR" || { echo "Unable to find folder $BASE_DIR, exiting"; exit 1; }

# check if Pyenv is installed and if so, store local Python ver setting
pyenv_local=""
if command -v pyenv &> /dev/null; then
  pyenv_local="$(pyenv local)"
fi

# Build Venv environment for the given Python version.
# Args:
#   1: Python version; if Pyenv is available, will get the latest matching ver.
build_venv() {
  local ver="$1"
  # default to Python as pythonx.y, eg python3.6
  local py="python${ver}"
  if [[ -n "$pyenv_local" ]]; then
    # find latest matching Python installed by Pyenv; eg if 3.6.3 and 3.6.11
    # are installed, 3.6 will give 3.6.11
    pyenv_ver="$(pyenv whence python | grep "$ver" | sort -V | tail -1)"
    if [[ -n "$pyenv_ver" ]]; then
      # set found version as the local Pyenv version
      pyenv local "$pyenv_ver"
      py=python
    else
      echo "Could not find Pyenv Python version for ${ver}, skipping"
      return
    fi
  elif ! command -v "$py" &> /dev/null; then
    echo "Could not find ${py}, skipping"
    return
  fi
  
  # generate new env if folder doesn't exist
  echo "Building Venv for Python $ver"
  py_venv="${venv_dir}/py${ver}"
  if [[ -d "$py_venv" ]]; then
    echo "$py_venv exists, skipping"
  else
    "$py" -m venv "$py_venv"
  fi
}

# set up a Venv for each given Python version
py_vers=(3.6 3.7 3.8 3.9)
for py_ver in "${py_vers[@]}"; do
  build_venv "$py_ver"
done

if [[ -n "$pyenv_local" ]]; then
  # reset Pyenv local setting
  pyenv local "$pyenv_local"
fi
