#!/bin/bash
# Sets up the Clrbrain environment
# Author: David Young 2017, 2019

HELP="
Sets up the initial Clrbrain environment with Anaconda and all
packages including git repositories.

Downloads and installs Miniconda3 if it is not already present. 
Installs or updates a Conda environment named \"clr3\" by 
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
  -a: Install AWS components.
  -n: Set the Conda environment name; defaults to CONDA_ENV.
  -s: Build and install SimpleElastix.
  -l: Lightweight environment setup, which does not include 
    GUI components such as Matplotlib or Mayavi.
"

# default Conda environment names as found in .yml configs
CONDA_ENV="clr3"
CONDA_ENV_LIGHT="clrclu"
env_name="$CONDA_ENV"

# default .yml files
ENV_CONFIG="environment.yml"
ENV_CONFIG_LIGHT="environment_light.yml"
config="$ENV_CONFIG"

build_simple_elastix=0
lightweight=0
aws=0

OPTIND=1
while getopts hn:sla opt; do
  case $opt in
    h)
      echo $HELP
      exit 0
      ;;
    n)
      env_name="$OPTARG"
      echo "Set to create the Conda environment $env_name"
      ;;
    s)
      build_simple_elastix=1
      echo "Set to build and install SimpleElastix"
      ;;
    l)
      lightweight=1
      env_name="$CONDA_ENV_LIGHT"
      config="$ENV_CONFIG_LIGHT"
      echo "Set to create lightweight (no GUI) environment"
      ;;
    a)
      aws=1
      echo "Set to install AWS components"
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


# Dependencies checks
  
if ! command -v "javac" &> /dev/null
then
    # check for Java compiler availability for Javabridge
	echo "Warning: \"javac\" not found; Python-Bioformats and Python-Javabridge"
	echo "will ot install correctly. Please install a JDK or add JDK_HOME or"
	echo "add JAVA_HOME to your path environment variables"
fi

if [[ "$os" == "MacOSX" ]]; then
  if [[ ! -e "/Library/Developer/CommandLineTools/usr/bin/git" ]]; then
    # Mac-specific check for command-line tools (CLT) package since the 
    # commands that are not activated will still return
    if [[ "$os_ver" < "10.14" && -e "/usr/include/iconv.h" ]]; then
      # ver <= 10.13 apparently also requires CLT headers here
      :
    else
      echo "Warning: Mac command-line tools not present/activated;"
      echo "installations that require compilation may not work properly."
      echo "If you encounter problems related to compilation, please run "
      echo "\"xcode-select --install\""
    fi
  fi
fi

if ! command -v "gcc" &> /dev/null
then
    # check for gcc availability for compiling Scikit-image (if directly from 
    # repo), Traits (if not from Conda), and Javabridge
	echo "Warning: \"gcc\" not found; installations that require compilation"
	echo "may not work properly. If you encounter problems related to"
	echo "compilation, please install \"gcc\"."
fi

if ! command -v "git" &> /dev/null
then
    # check for git availability for downloading repos for any installs 
    # from Git repos
	echo "Warning: \"git\" not found; installations that require repository"
	echo "access may not work properly. If you encounter problems related to"
	echo "repository downloads, please install \"git\"."
fi


# check for Anaconda installation and download/install if not found
if ! command -v "conda" &> /dev/null
then
	echo "Downloading and installing Miniconda..."
	PLATFORM=$os-$bit
	MINICONDA="Miniconda3-latest-$PLATFORM.${ext}"
	CONDA_URL=https://repo.continuum.io/miniconda/$MINICONDA
	if [[ "$os" == "MacOSX" ]]
	then
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

# create or update Conda environment
check_env="`conda env list | grep -w $env_name`"
if [[ "$check_env" == "" ]]
then
	# create an empty environment before setting channel priority to 
	# generate an env-specific .condarc file; Python version duplicated in 
	# .yml for those who want to create env directly from .yml
	echo "Creating new Conda environment from $config..."
	conda create -n "$env_name" python=3.6
	source activate "$env_name"
	conda config --env --set channel_priority strict # for mixed channels
	conda env update -f "$config"
else
	echo "$env_name already exists, will update"
	source activate "$env_name"
	conda env update -f "$config"
fi

# check that the environment was created and activate it
echo "Checking and activating conda environment..."
check_env="`conda env list | grep -w $env_name`"
if [[ "$check_env" == "" ]]
then
	echo "$env_name could not be found, exiting."
	exit 1
fi


############################################
# Download a shallow Git clone and pip install its Python package.
# Globals:
#   None
# Arguments:
#   Git repository URL
# Returns:
#   None
############################################
install_shallow_clone() {
  local folder="`basename $1`"
  local folder="${folder%.*}"
  if [ ! -e "$folder" ]; then
    # download and install fresh repo with shallow clone
    # and editable installation
    # TODO: check whether shallow clone will yield the 
    # correct fetch/merge steps later
    echo "Cloning into $folder"
    target=$1
    if [[ "$#" -gt 1 ]]; then
      target="${target} -b $2"
    fi
    git clone --depth 1 $target
    cd "$folder"
  else
    # update repo if changes found upstream on given branch
    echo "Updating $folder"
    cd "$folder"
    git fetch
    branch="master" # default branch
    if [[ "$#" -gt 1 ]]; then
      # use given branch, if any
      branch="$2"
      echo "Checking for differences with $branch"
    fi
    if [[ `git rev-parse --abbrev-ref HEAD` != "$branch" ]]; then
      echo "Not on $branch branch so will ignore updates"
    elif [[ `git diff-index HEAD --` ]]; then
      echo "Uncommitted file changes exist so will not update"
    elif [[ `git log HEAD..origin/"$branch" --oneline` ]]; then
      # merge in updates only if on same branch as given one, 
      # differences exist between current status and upstream 
      # branch, and no tracked files have uncommitted changes
      git merge origin/$branch
      echo "You may need to run post-update step such as "
      echo "\"python setup.py build_ext -i\""
    else
      echo "No changes found upstream on $branch branch"
    fi
  fi
  if [[ ! `pip list --format=columns | grep $folder` ]]; then
    echo "Installing $folder"
    pip install -e .
  fi
  cd ..
}

: '
# pip dependencies that are not available in Conda, some of which are 
# git-pip installed from Clrbrain parent directory
cd ..
if [[ $lightweight -eq 0 ]]; then
  # install dependencies for GUI requirement
  
  pip install -U matplotlib-scalebar
  
  # if Mayavi install says that vtk not directly required and not 
  # installed, install directly here
  #pip install -U vtk
  
  # pyqt 5.9.2 available in Conda gives a blank screen so need to use nwer 
  # pip-based version until Conda version is updated; 
  # Matplotlib in Conda on Linux64 and Mayavi on Mac at least not still 
  # require Conda pyqt, however, and will install the Conda package as well
  pip install -U PyQt5
  
  # use Conda now that TraitsUI and Pyface 6 available there
  #install_shallow_clone https://github.com/enthought/traits.git
  #install_shallow_clone https://github.com/enthought/pyface.git
  #install_shallow_clone https://github.com/enthought/traitsui.git
  
  # use Mayavi 4.6 release with Python 3 support
  #install_shallow_clone https://github.com/enthought/mayavi.git
  #pip install -U mayavi
fi

# if newer Scikit-image release is on PyPI
#install_shallow_clone https://github.com/the4thchild/scikit-image.git develop
#pip install -U scikit-image

# may need to install Python-Javabridge from Git for fixes for newer JDKs; 
# shallow clone does not work for some reason
#pip install -U javabridge
pip install git+https://github.com/LeeKamentsky/python-javabridge.git

# need older version since ver > 1.1 give heap space error
pip install -U python-bioformats==1.1.0
'

cd "$BASE_DIR"


if [ $build_simple_elastix -eq 1 ]
then
  # build and install SimpleElastix, replacing SimpleITK
  pip uninstall simpleitk
  ./build_se.sh -i
fi

if [[ $aws -eq 1 ]]; then
  # install AWS components
  pip install boto3
  pip install awscli
fi

echo "Clrbrain environment setup complete!"
echo "** Please run \"conda activate $env_name\" or \"source activate $env_name\""
echo "   depending on your Conda setup to enter the environment for Clrbrain **"

