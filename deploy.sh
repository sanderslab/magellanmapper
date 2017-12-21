#!/bin/bash
# Deploy Clrbrain to AWS
# Author: David Young 2017

################################################
# Deploy Clrbrain and related files to AWS.
#
# Arguments:
#   -h: Show help and exit.
#   -i: IP address of the AWS EC2 instance.
#   -p: Path to the .pem file for accessing EC2.
#
################################################

FIJI="http://downloads.imagej.net/fiji/latest/fiji-nojre.zip"

# run from parent directory
base_dir="`dirname $0`"
cd "$base_dir"
echo $PWD

OPTIND=1
while getopts hi:p: opt; do
    case $opt in
        h)  echo $HELP
            exit 0
            ;;
        i)  ip="$OPTARG"
            echo "Set IP to $ip"
            ;;
        p)  pem="$OPTARG"
            echo "Set pem key file to $pem"
            ;;
        :)  echo "Option -$OPTARG requires an argument"
            exit 1
            ;;
        --) ;;
    esac
done

# pass arguments after "--" to clrbrain
shift "$((OPTIND-1))"
EXTRA_ARGS="$@"

#today=`date +'%Y-%m-%d-%Hh%M'`
git_hash=`git rev-parse --short HEAD`
archive="clrbrain_${git_hash}.zip"
git archive -o "$archive" HEAD

scp -i "$pem" "$archive" ec2-user@"$ip":~
ssh -t -i "$pem" ec2-user@"$ip" "unzip $archive -d clrbrain && wget $FIJI && unzip fiji-nojre.zip -d clrbrain"
