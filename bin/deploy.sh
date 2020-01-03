#!/bin/bash
# Deploy MagellanMapper to AWS
# Author: David Young 2017, 2020

HELP="
Deploy MagellanMapper and related files to AWS.

Arguments:
  -h: Show help and exit.
  -d [file]: Deploy file or folder recursively. Can be used 
      multiple times to add additional files for upload.
  -i [IP]: IP address of the AWS EC2 instance.
  -p [path]: Path to the .pem file for accessing EC2.
  -u: Upload and update MagellanMapper files only, skipping the rest of 
      deployment.
  -f: Update Fiji/ImageJ. Assume that it has already been deployed.
  -g [git_hash]: Archive and upload the given specific Git commit; 
      otherwise, defaults to HEAD.
  -r [path]: Run script path to pipe and execute via SSH after all 
      uploads and other commands.
  -n [username]: Username on server. Defaults to ec2-user.
"

FIJI="http://downloads.imagej.net/fiji/latest/fiji-nojre.zip"
FIJI_SITES=("BigStitcher http://sites.imagej.net/BigStitcher/")
update=0 # update MagellanMapper
update_fiji=0 # update Fiji/ImageJ
run_script="" # run script, which will be executed as last cmd
deploy_files=() # files to deploy
git_hash="" # git commit, including short hashes
username="ec2-user" # default on many EC2 distros

# run from parent directory
BASE_DIR="$(dirname "$0")/.."
cd "$BASE_DIR" || { echo "Unable to find folder $BASE_DIR, exiting"; exit 1; }

OPTIND=1
while getopts hi:p:ufr:d:g:n: opt; do
  case $opt in
    h)
      echo "$HELP"
      exit 0
      ;;
    i)
      ip="$OPTARG"
      echo "Set IP to $ip"
      ;;
    p)
      pem="$OPTARG"
      echo "Set pem key file to $pem"
      ;;
    u)
      update=1
      echo "Set to update MagellanMapper only"
      ;;
    f)
      update_fiji=1
      echo "Set to update Fiji only"
      ;;
    r)
      run_script="$OPTARG"
      echo "Set run script to $run_script"
      ;;
    d)
      deploy_files+=("$OPTARG")
      echo "Adding $OPTARG to file deployment list"
      ;;
    g)
      git_hash="$OPTARG"
      echo "Using $git_hash as git hash"
      ;;
    n)
      username="$OPTARG"
      echo "Changing username to $username"
      ;;
    :)
      echo "Option -$OPTARG requires an argument"
      exit 1
      ;;
    *)
      echo "$HELP" >&2
      exit 1
      ;;
  esac
done

if [[ "$git_hash" == "" ]]; then
  git_hash=$(git rev-parse --short HEAD)
fi
archive="magellanmapper_${git_hash}.zip"
git archive -o "$archive" "$git_hash"

# basic deployment
deploy_files+=("$archive")
server_cmd="unzip -o $archive -d magellanmapper"
cmd_fiji_update="Fiji.app/ImageJ-linux64 --update update"

# append for initial deployment
if [[ $update -eq 0 ]]; then
  #mv_recon="multiview-reconstruction-*-SNAPSHOT.jar"
  #deploy_files+=" ../multiview-reconstruction/target/$mv_recon"
  server_cmd+=" && wget $FIJI"
  server_cmd+=" && unzip fiji-nojre.zip"
  server_cmd+=" && Fiji.app/ImageJ-linux64"
  for site in "${FIJI_SITES[@]}"; do
    server_cmd+=" --update add-update-site $site"
  done
  server_cmd+=" && $cmd_fiji_update"
  #server_cmd+=" ; rm Fiji.app/plugins/multiview?reconstruction-*.jar ;"
  #server_cmd+=" mv $mv_recon Fiji.app/plugins"
fi

# add on Fiji update
if [[ $update_fiji -eq 1 ]]; then
  server_cmd+=" && $cmd_fiji_update"
fi

# summarize files and commands
echo -e "\nFiles to deploy:"
echo "${deploy_files[@]}"
echo -e "\nCommand to execute on server:"
echo "$server_cmd"
echo ""

# execute upload and server commands
scp -i "$pem" -r "${deploy_files[@]}" "${username}@${ip}":~
ssh -i "$pem" "${username}@${ip}" "$server_cmd"

if [[ "$run_script" != "" ]]; then
  # pipe and execute custom script via SSH
  ssh -i "$pem" "${username}@${ip}" "bash -s" < "$run_script"
fi
