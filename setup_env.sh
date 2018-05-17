#!/bin/bash
# Sets up the Clrbrain environment
# Author: David Young 2017, 2018

################################################
# Sets up the initial Clrbrain environment with Anaconda and all
# packages including git repositories.
# 
# Arguments:
#   -h: Show help and exit.
#   -n: Set the Conda environment name; defaults to CONDA_ENV.
#   -s: Build and install SimpleElastix.
#   -l: Lightweight environment setup, which does not include 
#       GUI components such as Matplotlib or Mayavi.
#
# Assumptions:
# -Assumes that the Clrbrain source package
# -Creates the Anaconda environment, which should be first
#  removed if you want to start with a clean environment
# -Git dependencies will be cloned into the parent folder of 
#  Clrbrain
################################################

# default Conda environment names as found in .yml configs
CONDA_ENV="clr3"
CONDA_ENV_LIGHT="clrclu"
env_default="$CONDA_ENV"
env_name=""

# default .yml files
ENV_CONFIG="environment.yml"
ENV_CONFIG_LIGHT="environment_light.yml"
config_default="$ENV_CONFIG"

build_simple_elastix=0
lightweight=0

OPTIND=1
while getopts hn:sl opt; do
    case $opt in
        h)  echo $HELP
            exit 0
            ;;
        n)  env_name="$OPTARG"
            echo "Set to create the Conda environment $env_name"
            ;;
        s)  build_simple_elastix=1
            echo "Set to build and install SimpleElastix"
            ;;
        l)  lightweight=1
            env_default="$CONDA_ENV_LIGHT"
            config_default="$ENV_CONFIG_LIGHT"
            echo "Set to create lightweight (no GUI) environment"
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

# run from script directory
BASE_DIR="`dirname $0`"
cd "$BASE_DIR"
BASE_DIR="$PWD"

# check for Java jar availability
if ! command -v "javac" &> /dev/null
then
	echo "Please install JDK or add JAVA_HOME to your path environment variables. Exiting."
	exit 1
fi

# check for gcc availability for compiling Scikit-image;
# TODO: does not handle case where xcode tools needs to be installed
if ! command -v "gcc" &> /dev/null
then
	echo "Please install gcc. Exiting."
	exit 1
fi

# check for git availability for downloading repos for some pip installs
if ! command -v "git" &> /dev/null
then
	echo "Please install git. Exiting."
	exit 1
fi

# find platform for Anaconda
echo -n "Detecting environment..."
SYSTEM=`uname -a`
ANACONDA_DOWNLOAD_PLATFORM=""
if [[ "$SYSTEM" =~ "CYGWIN" ]] || [[ "$SYSTEM" =~ "WINDOWS" ]]
then
    ANACONDA_DOWNLOAD_PLATFORM="Windows"
elif [[ "$SYSTEM" =~ "Darwin" ]]
then
    ANACONDA_DOWNLOAD_PLATFORM="MacOSX"
elif [[ "$SYSTEM" =~ "Linux" ]]
then
    ANACONDA_DOWNLOAD_PLATFORM="Linux"
fi
BIT="x86"
if [[ "$SYSTEM" =~ "x86_64" ]]
then
    BIT="x86_64"
fi
echo "will use $ANACONDA_DOWNLOAD_PLATFORM platform with $BIT bit for Anaconda"

# check for Anaconda availability
if ! command -v "conda" &> /dev/null
then
	echo "Downloading and installing Miniconda..."
	PLATFORM=$ANACONDA_DOWNLOAD_PLATFORM-$BIT
	MINICONDA=Miniconda3-latest-$PLATFORM.sh
	CONDA_URL=https://repo.continuum.io/miniconda/$MINICONDA
	if [[ "$ANACONDA_DOWNLOAD_PLATFORM" == "MacOSX" ]]
	then
		curl -O "$CONDA_URL"
	else
		wget "$CONDA_URL"
	fi
	sh $MINICONDA
	# reload the bash environment, or exit if unable
	bash_profile=~/.bash_profile
	if [ ! -f $bash_profile ]
	then
		bash_profile=~/.bashrc
	fi
	if [ -f $bash_profile ]
	then
		# TODO: check if base environment gets activated as not yet 
		# by default as of Conda 4.4.10
		source $bash_profile
	else
		echo "Please close and reopen your terminal, then rerun this script"
		exit 0
	fi
fi

# create or update Conda environment
config="$config_default"
if [[ "$env_name" == "" ]]; then
    echo "Setting up default Conda environment, $env_default"
    env_name="$env_default"
else
    # change name in environment file with user-defined name
    echo "Updating environment configuration for $env_name"
    config="env_${env_name}.yml"
    sed -e "s/$env_default/$env_name/g" "$config_default" > "$config"
fi
check_env="`conda env list | grep -w $env_name`"
if [[ "$check_env" == "" ]]
then
	echo "Creating new Conda environment from $config..."
	conda env create -f "$config"
else
	echo "$env_name already exists, will update"
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
# need to reload conda script for unclear reasons; assume that 
# CONDA_PREFIX has already been set by base environment
#. "$CONDA_PREFIX/etc/profile.d/conda.sh"
source activate $env_name

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

# pip dependencies that are not available in Conda, some of which are 
# git-pip installed from Clrbrain parent directory
cd ..
if [[ $lightweight -eq 0 ]]; then
    # install dependencies for GUI requirement
    pip install -U matplotlib-scalebar
    #pip install -U vtk==8.1.0
    # pyqt 5.9.2 available in Conda gives a blank screen so need to use pip-based 
    # version, currently 5.10.1 as of 2018-05-10, until Conda version updated; 
    # Matplotlib in Conda on Linux64 is not compatible with this release, 
    # however, and will install the Conda package alongside the pip one
    pip install -U PyQt5
    #install_shallow_clone https://github.com/enthought/traits.git
    #install_shallow_clone https://github.com/enthought/pyface.git
    #install_shallow_clone https://github.com/enthought/traitsui.git
    install_shallow_clone https://github.com/enthought/mayavi.git
fi

# install remaining dependencies
install_shallow_clone https://github.com/the4thchild/scikit-image.git develop
# also cannot be installed in Conda environment configuration script 
# for some reason
install_shallow_clone https://github.com/LeeKamentsky/python-javabridge.git
pip install -U python-bioformats==1.1.0
cd "$BASE_DIR"

if [ $build_simple_elastix -eq 1 ]
then
    # build and install SimpleElastix
    ./build_se.sh -i
fi

echo "clrbrain environment setup complete!"
echo "** Please run \"conda activate $env_name\" to enter the environment for Clrbrain **"

