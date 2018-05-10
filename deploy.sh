#!/bin/bash
# Deploy Clrbrain to AWS
# Author: David Young 2017, 2018

################################################
# Deploy Clrbrain and related files to AWS.
#
# Clrbrain will be updated by default. Unless the "-u" flag is given, 
# the rest of the deployment including Fiji/ImageJ installation will 
# be completed.
#
# Arguments:
#   -h: Show help and exit.
#   -i: IP address of the AWS EC2 instance.
#   -p: Path to the .pem file for accessing EC2.
#   -u: Upload and update Clrbrain files only, skipping the rest of 
#       deployment.
#   -f: Update Fiji/ImageJ. Assume that it has already been deployed.
#
################################################

FIJI="http://downloads.imagej.net/fiji/latest/fiji-nojre.zip"
update=0 # update Clrbrain
update_fiji=0 # update Fiji/ImageJ
run_script=""

# run from parent directory
base_dir="`dirname $0`"
cd "$base_dir"
echo $PWD

OPTIND=1
while getopts hi:p:ufr: opt; do
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
        f)  update_fiji=1
            echo "Set to update Fiji only"
            ;;
        r)  run_script="$OPTARG"
            echo "Set run script to $run_script"
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

git_hash=`git rev-parse --short HEAD`
archive="clrbrain_${git_hash}.zip"
git archive -o "$archive" HEAD

# basic update
deploy_files="$archive"
server_cmd="unzip -o $archive -d clrbrain"

# append for initial deployment
if [[ $update -eq 0 ]]; then
    #mv_recon="multiview-reconstruction-*-SNAPSHOT.jar"
    #deploy_files+=" ../multiview-reconstruction/target/$mv_recon"
    server_cmd+=" && wget $FIJI"
    server_cmd+=" && unzip fiji-nojre.zip"
    server_cmd+=" && Fiji.app/ImageJ-linux64 --update add-update-site BigStitcher http://sites.imagej.net/BigStitcher/"
    server_cmd+=" && $cmd_fiji_update"
    #server_cmd+=" ; rm Fiji.app/plugins/multiview?reconstruction-*.jar ; mv $mv_recon Fiji.app/plugins"
fi

# append customized run script, such as that prepared by 
# setup_runclrbrain.sh
if [[ "$run_script" != "" ]]; then
    deploy_files+=" $run_script"
    server_cmd+=" && mv $run_script clrbrain"
fi

# add on Fiji update
if [[ $update_fiji -eq 1 ]]; then
    cmd_fiji_update="Fiji.app/ImageJ-linux64 --update update"
    server_cmd+=" && $cmd_fiji_update"
fi

# summarize files and commands
echo -e "\nFiles to deploy:"
echo "$deploy_files"
echo -e "\nCommand to execute on server:"
echo "$server_cmd"
echo ""

# execute upload and server commands
scp -i "$pem" $deploy_files ec2-user@"$ip":~
ssh -t -i "$pem" ec2-user@"$ip" "$server_cmd"
