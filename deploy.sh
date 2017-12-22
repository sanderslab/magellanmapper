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
update=0

# run from parent directory
base_dir="`dirname $0`"
cd "$base_dir"
echo $PWD

OPTIND=1
while getopts hi:p:u opt; do
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
        u)  update=1
            echo "Set to update Clrbrain only"
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
server_cmd="unzip -o $archive -d clrbrain"
if [ $update -eq 0 ]
then
    server_cmd="${server_cmd} "
    server_cmd+="&& wget $FIJI "
    server_cmd+="&& unzip fiji-nojre.zip "
    server_cmd+="&& Fiji.app/ImageJ-linux64 --update add-update-site BigStitcher http://sites.imagej.net/BigStitcher/ "
    server_cmd+="&& Fiji.app/ImageJ-linux64 --update update"
fi
ssh -t -i "$pem" ec2-user@"$ip" "$server_cmd"
