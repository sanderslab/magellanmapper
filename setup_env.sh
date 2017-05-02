#!/bin/bash

# run from parent directory
BASE_DIR="`dirname $0`"
cd "$BASE_DIR/.."
echo $PWD

if [ "`command -v jar`" == '' ]
then
	echo "Please add JAVA_HOME and update path environment variables. Exiting..."
	exit 0
fi

if [ "`command -v gcc`" == '' ]
then
	echo "Please install gcc. Exiting..."
	exit 0
fi

if [ "`command -v conda`" == '' ]
then
	echo "Downloading and installing Miniconda..."
	wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh 
	sh Miniconda3-latest-Linux-x86_64.sh
	source ~/.bashrc
fi

echo "Creating new conda environment..."
conda create --name clr python=3 pyqt=4
source activate clr

echo "Conda installing Mayavi and Scikit-image..."
conda install -c menpo mayavi
conda install scikit-image

echo "Pip installing Python-bioformats and exchanged Javabridge for GitHub version..."
pip install python-bioformats
pip uninstall javabridge
conda install cython
git clone https://github.com/LeeKamentsky/python-javabridge.git
cd python-javabridge
pip install -e .
cd ..

echo "Exchanging Scikit-image for GitHub version with 3D blob pull request..."
conda remove scikit-image
git clone https://github.com/the4thchild/scikit-image
cd scikit-image
git checkout blob3d
pip install -e .
python setup.py build_ext -i
cd ..

echo "Pip installing additional packages..."
pip install matplotlib-scalebar

echo "clrbrain environment setup complete!"

