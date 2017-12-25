#!/bin/bash
# Sets up the Clrbrain environment
# Author: David Young 2017

################################################
# Sets up the initial Clrbrain environment with Anaconda and all
# packages including git repositories.
# 
# Arguments:
#   -h: Show help and exit.
#   -n: Set the Conda environment name; defaults to CONDA_ENV.
#   -s: Build and install SimpleElastix.
#
# Assumptions:
# -Assumes that the Clrbrain git repo has already been cloned
# -Creates the Anaconda environment, which should be first
#  removed if you want to start with a clean environment
################################################

CONDA_ENV="clr2"
env_name="$CONDA_ENV"
build_simple_elastix=0

ENV_CONFIG="environment.yml"

OPTIND=1
while getopts hn:s opt; do
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
echo $PWD

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
		source $bash_profile
	else
		echo "Please close and reopen your terminal, then rerun this script"
		exit 0
	fi
fi

# creates "clr" conda environment
echo "Checking for $env_name Anaconda environment..."
check_env="`conda env list | grep -w $env_name`"
if [[ "$check_env" == "" ]]
then
	echo "Creating new conda environment..."
	config="$ENV_CONFIG"
	if [[ "$env_name" != "$CONDA_ENV" ]]
	then
	    # change name in environment file with user-defined name
	    config="env_${env_name}.yml"
	    sed -e "s/$CONDA_ENV/$env_name/g" "$ENV_CONFIG" > "$config"
	fi
	conda env create -f "$config"
else
	echo "$env_name already exists. Exiting."
	exit 1
fi
echo "Activating conda environment..."
source activate $env_name

if [ $build_simple_elastix -eq 1 ]
then
    # build and install SimpleElastix
    ./build_se.sh -i
fi

echo "clrbrain environment setup complete!"
echo "** Please restart your terminal and run \"source activate $env_name\" **"

