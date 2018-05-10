#!/bin/bash
# Prepare a server the Clrbrain pipeline
# Author: David Young 2018

HELP="
Sets up a server for processing files in the Clrbrain 
pipeline. Both initial setup and preparing existing servers 
is supported.

Arguments:
   -h: Show help and exit.
   -s: Set up a fresh server, including drive initiation.
   -l: Use legacy drive specifications.

Assumptions:
- Two additional drives are attached:
  1) /dev/nvme1n1: for swap
  2) /dev/nvmme2n1: ext4 format, for data
- If \"-l\" flag is given, legacy devices are assumed: 
  1) /dev/xvdf for swap
  2) /dev/xvdg for data
- Username: "ec2-user", a standard username on AWS
"

DIR_DATA="/data"

setup=0
swap="/dev/nvme1n1"
data="/dev/nvme2n1"

OPTIND=1
while getopts hsl opt; do
    case $opt in
        h)  echo $HELP
            exit 0
            ;;
        s)  setup=1
            echo "Set to prepare a new server instance"
            ;;
        l)  swap="/dev/xvdf"
            data="/dev/xvdg"
            echo "Set to use legacy device specifiers:"
            echo "swap set to $swap"
            echo "data set to $data"
            ;;
        :)  echo "Option -$OPTARG requires an argument"
            exit 1
            ;;
        --) ;;
    esac
done

# pass arguments after "--" to another script if necessary
shift "$((OPTIND-1))"
EXTRA_ARGS="$@"

# run from script directory
BASE_DIR="`dirname $0`"
cd "$BASE_DIR"
BASE_DIR="$PWD"

# show current drive arrangement
lsblk -p

if [[ $setup -eq 1 ]]; then
    # initialize swap and storage drives if setting up 
    # a new server instance
    sudo mkswap "$swap"
    sudo mkfs -t ext4 "$data"
fi

# turn on swap and mount storage drive; these commands 
# should fail if these drives were not initialized or 
# attached
sudo swapon "$swap"
swapon -s
if [[ ! -d "$DIR_DATA" ]]; then
    sudo mkdir "$DIR_DATA"
fi
sudo mount "$data" "$DIR_DATA"
lsblk -p

if [[ $setup -eq 1 ]]; then
    # change ownership if new drive attached
    sudo chown -R ec2-user.ec2-user "$DIR_DATA"
fi

exit 0
