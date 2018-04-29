# Clrbrain
Clrbrain is an imaging informatics GUI and pipeline for high-throughput, automated analysis of whole organs. Its design philosophy is to make the raw, original images as accessible as possible; simplify annotation with a focus on nuclei; and scale from the laptop to the cloud in cross-platform environments.

Author: David Young, 2017-2018, Stephan Sanders Lab

## Installation
```
./setup_env.sh
```
The setup script will install the following:

- [Miniconda](https://conda.io/miniconda.html), a light version of the Anaconda package and environment manager for Python, will be installed if an existing installation isn't found
- A `clr3` environment with Python 3
- The Scipy, Numpy, Matplotlib stack
- Mayavi and related Git repositories, downloaded into the parent folder of Clrbrain (ie alongside rather than inside Clrbrain) for the GUI and 3D visualization; note that Mayavi currently requires a graphical environment to install
- Scikit-Image for image processing

## Run Clrbrain
Opening an image file typically involves importing it into a Numpy array format before loading it in the GUI and processing it headlessly.

- Open `runclrbrain.sh` and edit it with the path to your image file and a sample command, such as file import
- Open a new terminal if you just installed Miniconda
- Run the script:

```
source activate clr3
./runclrbrain.sh
```

## Access from Git

- Contact the project managers aboud loading your public key into this project
- Download the repo: `git clone git@bitbucket.org:psychcore/clrbrain.git`

## Start a processing pipeline

Automated processing will attempt to scale based on your system resources but may require some manual intervention. This pipeline has been tested on a Macbook Pro laptop and AWS EC2 Linux (RHEL and Amazon Linux based) instances.

### Local
Use the sample stack processing command in `runclrbrain.sh`.

### Server

- Launch an instance; graphical support and login (eg `vncserver`) currently required during installation because of Mayavi (see above)
- Attach a swap drive (eg 100GB) and storage drive (4-5x your image size)
- Deploy the Clrbrain Git as an archive: `./deploy.sh -p [your_aws_pem] -i [server_ip]`, which will:
 - Archive the Clrbrain Git directory and `scp` it to the server
 - Download and install ImageJ/Fiji onto the server
 - Update Fiji and install BigStitcher for image stitching
 - To only update an existing Clrbrain directory on the server, add `-u`
- Log into your instance
- Setup drives: `./setup_server.sh -s`, where the `-s` flag can be removed on subsequent launches if the drives are already initialized
- Install and run Clrbrain as above


## Troubleshooting

### Java installation
- Tested on Java 8 SE
- Double-check that the Java SDK has truly been installed since the Clrbrain setup script may not catch all missing installations
- You may need to set up the JAVA_HOME environment variable: `JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk1.8.0_111.jdk/Contents/Home`, and add this variable to your PATH in `~/.bash_profile`

### Xcode setup (Mac)
- `xcrun: error: invalid active developer path (/Library/Developer/CommandLineTools), missing xcrun at: /Library/Developer/CommandLineTools/usr/bin/xcrun` error: The Command Line Tools package on Mac may need to be installed or updated. Try `xcode-select --install` to install Xcode. If you get an error (eg "Can't install the software because it is not currently available from the Software Update server"), try downloading Xcode directly from https://developer.apple.com/download/, then run `sudo xcodebuild -license` to accept the license agreement.


### International setup
- If you get a Python locale error, add these lines to your `~/.bash_profile` file:
```
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
```

### Sciki-image installation
- If you continue getting errors during Scikit-image compilation, delete the folder and restart its compilation from scratch
- After updating any Scikit-image Cython files, run `python setup.py build_ext -i` as per https://github.com/scikit-image/scikit-image. If older version of extensions remain, run `git clean -dxf` to completely clear the working directory (check for any working files you need!) before rerunning the extension builder.

### Mayavi installation
- As of at least 2018-01-05, Mayavi installation requires a GUI so will not work on headless cloud instances, giving a `QXcbConnection: Could not connect to display` error. Will need to work on making Mayavi optional for purely headless systems for analysis.

### Image Stitching
- The original stitcher, `Stitching`, requires a large amount of RAM/swap space and runs single-threaded, taking days to stitch a multi-tile image
- The new, recommended stitcher, `BigStitcher`, uses RAM much more efficiently through an HDF5 format and utilizes multiprocessing
- Clrbrain runs these stitchers as ImageJ scripts in an attempt to require minimal intervention
- BigStitcher throws an exception without a graphical environment, however
- The threshold for links between tiles is set high to minimize false links, falling back on metadata, but still may give false alignments, so manual inspection of stitched images is recommended
- To fix alignments in BigStitcher:
 - Copy the `.xml~2` file to `fix.xml` to obtain the state just before the optimization step
 - Use its Link Explorer to remove inappropriate links
 - Run the global optimizer again with two round and metadata fallback
 - If necessary, use the Manually Align tool to move specific tiles

### Additional tips
- If you get an `syntax error near unexpected token (` error, the run script may have been formatted incorrectly, eg through the Mac Text Editor program. Try `dos2unix [runclrbrain.sh]` (replace with your run script filename) or re-copying from `runclrbrain.sh`.