#!/bin/bash
# Prepare a server the Clrbrain pipeline
# Author: David Young 2018, 2019

HELP="
Sets up a server for processing files in the Clrbrain pipeline. 
Both initial setup and re-setup of existing servers is supported.

Arguments:
   -d [/dev/name]: Set data device path to mount \"/data\". If an empty 
     string, data mount will not be set up. Can be a name (eg \"sdf\") to 
     map to an NVMe name if the NVMe flag is given.
   -f [GB]: Size of swap file in GB.
   -h: Show help and exit.
   -n: Map device names to NVMe names. Assumes that NVMe device names 
     are in the format \"/dev/nvme[n]n1\", where n is from 0-5.
   -s: Set up a fresh server, including drive initiation.
   -u [username]: Username on server. Defaults to ec2-user, a standard 
     username on AWS
   -w [/dev/name]: Set swap device path. If an empty 
     string, data drive will not be set up.

"

DIR_DATA="/data"

setup=0
swap=""
data=""
username="ec2-user" # default on many EC2 distros but not Ubuntu
swapfile_size=""
nvme=0

OPTIND=1
while getopts hslw:d:u:f:n opt; do
  case $opt in
    h)
      echo "$HELP"
      exit 0
      ;;
    s)
      setup=1
      echo "Set to prepare a new server instance"
      ;;
    n)
      nvme=1
      echo "Set to convert device names to NVMe assignments"
      ;;
    w)
      swap="$OPTARG"
      echo "Set swap device/file path to $swap"
      ;;
    d)
      data="$OPTARG"
      echo "Set data device path to $data"
      ;;
    f)
      swapfile_size="$OPTARG"
      echo "Generate a swapfile with size of ${swapfile_size}GB"
      ;;
    u)
      username="$OPTARG"
      echo "Changing username to $username"
      ;;
    :)
      echo "Option -$OPTARG requires an argument"
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

############################################
# Check if a device is formatted.
# Globals:
#   NONE
# Arguments:
#   1: Device path.
# Returns:
#   0 if device is formatted, otherwise 1.
############################################
is_formatted() {
  local format="$(lsblk -o FSTYPE -n $1)"
  if [[ -n "${format// }" ]]; then
    echo "$1 is already formatted"
    return
  fi
  false
}

############################################
# Mount device.
# Globals:
#   DIR_DATA: Data mount point.
# Arguments:
#   1: Device path.
#   2: Mount path.
# Returns:
#   NONE
############################################
mount_dev() {
  if ! mountpoint -q "$DIR_DATA"; then
    if [[ ! -d "$2" ]]; then
      sudo mkdir "$2"
    fi
    sudo mount "$1" "$2"
  fi
}

############################################
# Check whether the given NVMe device and name match.
# Globals:
#   NONE
# Arguments:
#   1: Device path.
#   2: Device name.
# Returns:
#   NONE
############################################
is_nvme_name() {
  if [[ -e "$1" ]]; then
    sudo nvme id-ctrl -v "$1" | grep "$2"
  fi
}

############################################
# Convert a given name to an NVMe device path. Assumes that NVMe paths 
# are in the format /dev/nvme[n]n1, where n is checked from 0-5.
# Globals:
#   dev: Last checked device path.
# Arguments:
#   1: Name of device, eg /dev/sdf.
# Returns:
#   0 if match is found, otherwise 1.
############################################
map_nvme_name() {
  for i in {0..5}; do
    dev="/dev/nvme${i}n1"
    if [[ "$(is_nvme_name ${dev} ${1})" != "" ]]; then
      echo "found match between $1 and $dev"
      return
    fi
  done
  false
}


if [[ "$nvme" -eq 1 ]]; then
  # convert device names to NVMe device paths since these devices are assigned 
  # different names and in inconsistent order
  if [[ -n "$data" ]]; then
    if map_nvme_name "$data"; then
      data="$dev"
    else
      echo "Could not find mapping from $data to its NVMe name, exiting"
      exit 1
    fi
  fi
  if [[ -n "$swap" && -z "$swapfile_size" ]]; then
    if map_nvme_name "$swap"; then
      swap="$dev"
    else
      echo "Could not find mapping from $swap to its NVMe name, exiting"
      exit 1
    fi
  fi
fi

if [[ $setup -eq 1 ]]; then
  # initialize swap and storage drives if setting up a new server instance
  if [[ -n "$data" ]]; then
    # format data device if not already formatted and mount
    is_formatted "$data"
    already_formatted="$?"
    if [[ $already_formatted -ne 0 ]]; then
      sudo mkfs -t ext4 "$data"
    fi
    mount_dev "$data" "$DIR_DATA"
    if [[ $already_formatted -ne 0 ]]; then
      # need to change ownership if new drive attached
      sudo chown -R "${username}.${username}" "$DIR_DATA"
    fi
  fi
  if [[ -n "$swap" ]]; then
    if [[ "$swapfile_size" != "" ]]; then
      # generate swap file
      if [[ -e "$swap" ]]; then
        echo "$swap already exists, will attempt to load as swap file"
      else
        # generate swapfile with given size in GB; 
        # TODO: consider fallback to dd with formats that don't support 
        # fallocate for swap files
        echo "Making ${swapfile_size}GB swap file at $swap"
        sudo fallocate -l "${swapfile_size}G" "$swap"
        sudo chmod 0600 "$swap"
        sudo mkswap "$swap"
      fi
    else
      # generate swap partition if device isn't already formatted
      if ! is_formatted "$swap"; then
        sudo mkswap "$swap"
      fi
    fi
  fi
fi


# mount storage drive and turn on swap, which should fail if these 
# drives were not initialized or attached
if [[ -n "$data" ]]; then
  # mount if not previously mounted
  mount_dev "$data" "$DIR_DATA"
fi
if [[ -n "$swap" ]]; then
  sudo swapon "$swap"
  swapon -s
fi
lsblk -p

exit 0
