#!/bin/bash
# Sets up the Clrbrain environment
# Author: David Young 2017

################################################
# Sets up the initial Clrbrain environment with Anaconda and all
# git repositories.
#
# Assumptions:
# -Assumes that the Clrbrain git repo has already been cloned
# -If any other git repositories already exist, they might be
#  usable, but you might want to move them aside to download
#  fresh clones
# -Creates the Anaconda environment, "clr", which should be first
#  removed if you want to start with a clean environment
################################################

CONDA_ENV=clr

# run from parent directory
BASE_DIR="`dirname $0`"
cd "$BASE_DIR/.."
echo $PWD

# check for Java jar availability
if [ "`command -v javac`" == '' ]
then
	echo "Please install JDK or add JAVA_HOME to your path environment variables. Exiting..."
	exit 1
fi

# check for gcc availability for compiling Scikit-image
if [ "`command -v gcc`" == '' ]
then
	echo "Please install gcc. Exiting..."
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
if [ "`command -v conda`" == '' ]
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
		exit 1
	fi
fi

# creates "clr" conda environment
echo "Activating Anaconda environment..."
check_env="`conda env list | grep $CONDA_ENV`"
if [[ "$check_env" == "" ]]
then
	echo "Creating new conda environment..."
	conda create --name $CONDA_ENV python=3 pyqt=4
fi
source activate $CONDA_ENV

# install the Menpo version of Mayavi to work with Python 3:
# https://github.com/enthought/mayavi/issues/84#issuecomment-266197984
echo "Conda installing Mayavi and Scikit-image..."
conda install -c menpo mayavi
# although Scikit-image will later be uninstalled, installing now will bring
# in all the dependencies through conda
conda install scikit-image

# install the libraries to read CZI through Bioformats
echo "Pip installing Python-bioformats and exchanging Javabridge for GitHub version..."
pip install python-bioformats
pip uninstall javabridge
conda install cython
# need git version for some MacOS-specific bug fixes
git clone https://github.com/the4thchild/python-javabridge.git
cd python-javabridge
pip install -e .
cd ..

# replace the current Scikit-image release with a customized version
echo "Exchanging Scikit-image for GitHub version with 3D blob pull request..."
conda remove scikit-image
# need git version with 3D blob PR pulled into it
git clone https://github.com/the4thchild/scikit-image
cd scikit-image
git checkout blob3d
pip install -e .
python setup.py build_ext -i
cd ..

# install a simple scalebar for Matplotlib
echo "Pip installing additional packages..."
pip install matplotlib-scalebar

echo "clrbrain environment setup complete!"

