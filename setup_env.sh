#!/bin/bash

# run from parent directory
BASE_DIR="`dirname $0`"
cd "$BASE_DIR/.."
echo $PWD

if [ "`command -v jar`" == '' ]
then
	echo "Please add Java SDK directory to path. Exiting..."
	exit 0
fi

if [ "`command -v conda`" == '' ]
then
	wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh 
	sh Miniconda3-latest-Linux-x86_64.sh
	source ~/.bashrc
fi
conda create --name clr01 python=3 pyqt=4
source activate clr01
conda install -c menpo mayavi
conda install scikit-image
pip install python-bioformats
pip uninstall javabridge
conda install cython
git clone https://github.com/LeeKamentsky/python-javabridge.git
cd python-javabridge
pip install -e .
cd ..

conda remove scikit-image
git clone https://github.com/scikit-image/scikit-image.git
cd scikit-image
git branch blob3d
git checkout blob3d
git pull origin pull/2114/head
pip install -e .
cd ..

