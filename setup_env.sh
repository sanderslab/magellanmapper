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

# run from parent directory
BASE_DIR="`dirname $0`"
cd "$BASE_DIR/.."
echo $PWD

# check for Java jar availability
if [ "`command -v jar`" == '' ]
then
	echo "Please add JAVA_HOME and update path environment variables. Exiting..."
	exit 0
fi

# check for gcc availability for compiling Scikit-image
if [ "`command -v gcc`" == '' ]
then
	echo "Please install gcc. Exiting..."
	exit 0
fi

# check for Anaconda availability
if [ "`command -v conda`" == '' ]
then
	echo "Downloading and installing Miniconda..."
	wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh 
	sh Miniconda3-latest-Linux-x86_64.sh
	source ~/.bashrc
fi

# creates "clr" conda environment
echo "Creating new conda environment..."
conda create --name clr python=3 pyqt=4
source activate clr

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

