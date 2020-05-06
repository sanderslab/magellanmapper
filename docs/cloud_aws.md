# Tools for AWS Cloud Management in MagellanMapper

MagellanMapper provides a few tools for basic management of an Amazon Web Servicies (AWS) platform for running pipelines with cloud computing. These tools serve as wrappers to the AWS Boto3 Python-based client management to simplify common tasks related to MagellanMapper.

## Dependencies

- `awscli`: AWS Command Line Interface for basic up/downloading of images and processed files S3. Install via Pip.
- `boto3`: AWS Python client to manage EC2 instances.

## Launch EC2 Instances

You can launch a new EC2 instance of a custom AMI via MagellanMapper, such as an AMI with MagellanMapper pre-installed as described below.

To start an instance, use this command:

```
python -u -m magmap.io.aws --ec2_start "Name" "ami-xxxxxxxx" "m5.4xlarge" \
  "subnet-xxxxxxxx" "sg-xxxxxxxx" "UserName" 50,2000 [2]
```

- `Name` is your name of choice
- `ami` is your previously saved AMI with MagellanMapper
- `m5.4xlarge` is the instance type, which can be changed depending on your performance requirements
- `subnet` is your subnet group
- `sg` is your security group
- `UserName` is the user name whose security key will be uploaded for SSH access
- `50,2000` creates a 50GB swap and 2000GB data drive, which can be changed depending on your needs
- `2` starts two instances (optional, defaults to 1)

To log into the server with graphical support, SSH into your server instance with port forwarding to allow VNC access:

```
ssh -L 5900:localhost:5900 -i [your_aws_pem] ec2-user@[your_server_ip]
```

Start a graphical server (eg `vncserver`) to run ImageJ/Fiji for stitching or for Mayavi dependency setup. Now you can access the server graphically using a VNC cient pointing to `localhost` with port `5900`.

## Set Up Volumes

The `setup_server.sh` script sets up volumes on a new server instance:

```
bin/setup_server.sh -d [path_to_data_device] -w [path_to_swap_device] \
    -f [size_of_swap_file] -u [username]
```

- `-d` is the main data drive, typically a drive large enough for your full image file
- `-w` is a swap drive or file path
- `-f` is a swap file size in GB, used if swap is set to a path rather than a device
- `-n` will map device names to NVMe names, which allows drive names such as `sdf` to be mapped to the corresponding NVMe-style (eg `/dev/nvme0n1`) names
- `-u` is the username, used to change ownership of the mounted drives; defaults to `ec2-user` and should be changed to `ubuntu` for Ubunut-based AMIs
- `-s` to set up fresh drives including formatting; exclude when re-mounting drives that have already been formatted

Set up drives on a new server instance to format and mount data and swap drives or create swap files:

```
bin/setup_server.sh -d [path_to_data_device] -w [path_to_swap_device] \
    -f [size_of_swap_file] -u [username]
```

## Set Up a Server with MagellanMapper

After launching a server, you can set up MagellanMapper by downloading it within the server or by deploying custom files such as a local branch using the `depoy.sh` script.

### Install/Update From Main Respository

Typically graphical support (eg via `vncserver`) is required during installation for Mayavi and stitching in the standard setup, but you can alternatively run a lightweight install without GUI (see [Readme](../README.md)).

See [Installation](install.md) for downloading and install MagellanMapper within the server.

### Install/Update From Local Branch 

The deployment script allows deploying MagellanMapper using local files including custom modifications. It also downloads Fiji for the image stitching pipeline. Use this command to deploy local files to a server:

```
bin/deploy.sh -p [path_to_your_aws_pem] -i [server_ip] \
    -d [optional_file0] -d [optional_file1]
```

- This script by default will:
  - Archive the MagellanMapper Git directory and `scp` it to the server, using your `.pem` file to access it
  - Download and install ImageJ/Fiji onto the server
  - Update Fiji and install BigStitcher for image stitching
- To only update an existing MagellanMapper directory on the server, add `-u`
- To add multiple files or folders such as `.aws` credentials, use the `-d` option as many times as you'd like
- After running this script, log in and install MagellanMapper if it has not been previously set up

#### Run MagellanMapper on Server

When returning to a server with MagellanMapper already set up, you'll need to perform the following tasks:

- Re-mount drives using `setup_server.sh`
- Activate the Conda environment set up during installation

Now you can use the `pipelines.sh` script to perform tasks as whole images, such as this command to fully process a multi-tile image with tile stitching, import to Numpy array, and cell detection, with AWS S3 import/export and Slack notifications along the way, followed by server clean-up/shutdown:

```
bin/process_nohup.sh -d "out_experiment.txt" -o -- bin/pipelines.sh \
  -i "/data/HugeImage.czi" -a "my/s3/bucket" -n \
  "https://hooks.slack.com/services/my/incoming/webhook" -p full -c
```
