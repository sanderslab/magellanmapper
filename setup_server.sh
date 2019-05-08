#!/bin/bash
# Prepare a server the Clrbrain pipeline
# Author: David Young 2018, 2019

HELP="
Sets up a server for processing files in the Clrbrain 
pipeline. Both initial setup and preparing existing servers 
is supported.

Arguments:
   -h: Show help and exit.
   -d [/dev/path]: Set data device path to mount \"/data\". If an empty 
       string, data mount will not be set up.
   -f [GB]: Size of swap file in GB.
   -s: Set up a fresh server, including drive initiation.
   -w [/dev/path]: Set swap device path. If an empty 
       string, data drive will not be set up.
   -l: Use legacy drive specifications.
   -u [username]: Username on server. Defaults to ec2-user.

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
username="ec2-user" # default on many EC2 distros
swapfile_size=""

OPTIND=1
while getopts hslw:d:u:f: opt; do
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
        w)  swap="$OPTARG"
            echo "Set swap device/file path to $swap"
            ;;
        d)  data="$OPTARG"
            echo "Set data device path to $data"
            ;;
        f)  swapfile_size="$OPTARG"
            echo "Generate a swapfile with size of ${swapfile_size}GB"
            ;;
        u)  username="$OPTARG"
            echo "Changing username to $username"
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

is_formatted() {
    local format="$(lsblk -o FSTYPE -n $1)"
    if [[ -z "${format// }" ]]; then
        return 0
    else
        echo "$1 is already formatted"
        return 1
    fi
}

mount_dev() {
    if ! mountpoint -q "$DIR_DATA"; then
        if [[ ! -d "$2" ]]; then
            sudo mkdir "$2"
        fi
        sudo mount "$1" "$2"
    fi
}

if [[ $setup -eq 1 ]]; then
    # initialize swap and storage drives if setting up a new server instance
    if [[ "$data" != "" ]]; then
        is_formatted "$data"
        newly_formatted="$?"
        if [[ $newly_formatted -eq 0 ]]; then
            sudo mkfs -t ext4 "$data"
        fi
        mount_dev "$data" "$DIR_DATA"
        if [[ $newly_formatted -eq 0 ]]; then
            # need to change ownership if new drive attached
            sudo chown -R "${username}.${username}" "$DIR_DATA"
        fi
    fi
    if [[ "$swap" != "" ]]; then
        if [[ "$swapfile_size" != "" ]]; then
            if [[ -e "$swap" ]]; then
                echo "$swap already exists, will attempt to load as swap file"
            else
                # generate swapfile with given size in GB based on 
                # block size * MB in GB * num GB
                size=$((1024*1024*swapfile_size))
                echo "Making ${swapfile_size}GB swap file at $swap"
                dd if=/dev/zero of="$swap" bs=1024 count=$size
                sudo chown root:root "$swap"
                sudo chmod 0600 "$swap"
                sudo mkswap "$swap"
            fi
        else
            is_formatted "$swap"
            if [[ "$?" -eq 0 ]]; then
                sudo mkswap "$swap"
            fi
        fi
    fi
fi


# turn on swap and mount storage drive, which should fail if these 
# drives were not initialized or attached
if [[ "$swap" != "" ]]; then
    sudo swapon "$swap"
    swapon -s
fi
if [[ "$data" != "" ]]; then
    mount_dev "$data" "$DIR_DATA"
fi
lsblk -p

exit 0
