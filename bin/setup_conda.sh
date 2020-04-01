#!/bin/bash
# Sets up the MagellanMapper environment
# Author: David Young 2017, 2020

HELP="
Sets up the initial MagellanMapper environment with Anaconda and all
packages including git repositories.

Downloads and installs Miniconda3 if it is not already present. 
Installs or updates a Conda environment named \"clr\" by 
default in a standard graphical setup or \"clrclu\" for a 
lightweight setup.

Although this installation generally makes use of Conda 
packages, Pip packages are occasionally used instead if the 
package or necessary version is unavailable in Conda. In some 
cases, dependencies that have required updates that are not yet 
fully released but available on Git, in which case shallow Git 
clones will be downloaded and installed through Pip.

Arguments:
  -h: Show help and exit.
  -n [name]: Set the Conda environment name; defaults to CONDA_ENV.
  -s [spec]: Specify the environment specification file; defaults to 
    ENV_CONFIG.
"

# default Conda environment names as found in .yml configs
CONDA_ENV="mag"
env_name="$CONDA_ENV"

# default .yml files
ENV_CONFIG="environment.yml"
config="$ENV_CONFIG"

OPTIND=1
while getopts hn:s: opt; do
  case $opt in
    h)
      echo "$HELP"
      exit 0
      ;;
    n)
      env_name="$OPTARG"
      echo "Set to create the Conda environment $env_name"
      ;;
    s)
      config="$OPTARG"
      echo "Set the environment spec file to $config"
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

# run from script directory
BASE_DIR="$(dirname "$0")/.."
cd "$BASE_DIR" || { echo "Unable to find folder $BASE_DIR, exiting"; exit 1; }
BASE_DIR="$PWD"

# load dependencies
source bin/libmag.sh

# find platform for Anaconda
detect_platform
ext="sh"
if [[ "$os" = "Windows" ]]; then
  ext="ext"
fi

# check for Anaconda installation and download/install if not found
if ! command -v "conda" &> /dev/null; then
  echo "Downloading and installing Miniconda..."
  PLATFORM=$os-$bit
  MINICONDA="Miniconda3-latest-$PLATFORM.${ext}"
  CONDA_URL=https://repo.continuum.io/miniconda/$MINICONDA
  if [[ "$os" == "MacOSX" ]]; then
    curl -O "$CONDA_URL"
  else
    wget "$CONDA_URL"
  fi
  chmod 755 "$MINICONDA"
  ./"$MINICONDA"
  # reload the bash environment, or exit if unable
  bash_profile=~/.bash_profile
  if [[ ! -f $bash_profile ]]; then
    bash_profile=~/.bashrc
  fi
  if [[ -f $bash_profile && "$os" != "Linux" ]]; then
    # Ubuntu and likely other Linux platforms short-circuit sourcing 
    # .bashrc non-interactively so unable to load without hacks
    # TODO: check if base environment gets activated as not yet 
    # by default as of Conda 4.4.10
    source $bash_profile
  else
    echo "Please close and reopen your terminal, then rerun this script"
    exit 0
  fi
fi

# create or update Conda environment; warn of apparent hang since no 
# progress monitor displays during installs by environment spec
check_env="$(conda env list | grep -w "$env_name")"
msg="Installing dependencies (may take awhile and appear to hang after the"
msg+="\n  \"Executing transaction\" step because of additional "
msg+="downloads/installs)..."
eval "$(conda shell.bash hook)"
if [[ "$check_env" == "" ]]; then
  # create an empty environment before setting channel priority to 
  # generate an env-specific .condarc file; Python version duplicated in 
  # .yml for those who want to create env directly from .yml
  echo "Creating new Conda environment from $config..."
  conda create -y -n "$env_name" python=3.6
  conda activate "$env_name"
  conda config --env --set channel_priority strict # for mixed channels
  echo -e "$msg"
  conda env update -f "$config"
else
  echo "$env_name already exists, will update"
  conda activate "$env_name"
  echo -e "$msg"
  conda env update -f "$config"
fi

# check that the environment was created and activate it
echo "Checking and activating conda environment..."
check_env="$(conda env list | grep -w "$env_name")"
if [[ "$check_env" == "" ]]; then
  echo "$env_name could not be found, exiting."
  exit 1
fi

msg="MagellanMapper environment setup complete!"
msg+="\n** Please run \"conda activate $env_name\" or "
msg+="\"source activate $env_name\""
msg+="\n   depending on your Conda setup to enter the environment "
msg+="for MagellanMapper **"
echo -e "$msg"
