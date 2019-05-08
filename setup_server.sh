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

OPTIND=1
while getopts hslw:d:u: opt; do
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
    format="$(lsblk -o FSTYPE -n $1)"
    if [[ -z "${format// }" ]]; then
        return 0
    else
        echo "$1 is already formatted"
        return 1
    fi
}

if [[ $setup -eq 1 ]]; then
    # initialize swap and storage drives if setting up 
    # a new server instance
    if [[ "$swap" != "" ]]; then
        is_formatted "$swap"
        if [[ "$?" -eq 0 ]]; then
            sudo mkswap "$swap"
        fi
    fi
    if [[ "$data" != "" ]]; then
        is_formatted "$data"
        if [[ "$?" -eq 0 ]]; then
            sudo mkfs -t ext4 "$data"
        fi
    fi
fi


# turn on swap and mount storage drive; these commands 
# should fail if these drives were not initialized or 
# attached
if [[ "$swap" != "" ]]; then
    sudo swapon "$swap"
    swapon -s
fi

if [[ "$data" != "" ]]; then
    if [[ ! -d "$DIR_DATA" ]]; then
        sudo mkdir "$DIR_DATA"
    fi
    sudo mount "$data" "$DIR_DATA"
fi
lsblk -p

if [[ $setup -eq 1 && -e "$DIR_DATA" ]]; then
    # change ownership if new drive attached
    sudo chown -R "${username}.${username}" "$DIR_DATA"
fi

exit 0
