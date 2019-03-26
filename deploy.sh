#!/bin/bash
# Deploy Clrbrain to AWS
# Author: David Young 2017, 2019

HELP="
Deploy Clrbrain and related files to AWS.

Arguments:
  -h: Show help and exit.
  -d [file]: Deploy file or folder recursively. Can be used 
      multiple times to add additional files for upload.
  -i [IP]: IP address of the AWS EC2 instance.
  -p [path]: Path to the .pem file for accessing EC2.
  -u: Upload and update Clrbrain files only, skipping the rest of 
      deployment.
  -f: Update Fiji/ImageJ. Assume that it has already been deployed.
  -g [git_hash]: Archive and upload the given specific Git commit; 
      otherwise, defaults to HEAD.
  -q: Quiet mode, which will suppress ssh output and cause the 
      script to return immediately.
  -r [path]: Path to the run script that will be uploaded and 
      excecuted as the last server command.
  -n [username]: Username on server. Defaults to ec2-user.
"

FIJI="http://downloads.imagej.net/fiji/latest/fiji-nojre.zip"
update=0 # update Clrbrain
update_fiji=0 # update Fiji/ImageJ
run_script="" # run script, which will be executed as last cmd
deploy_files=() # files to deploy
git_hash="" # git commit, including short hashes
quiet=0
username="ec2-user" # default on many EC2 distros

# run from parent directory
base_dir="`dirname $0`"
cd "$base_dir"
echo $PWD

OPTIND=1
while getopts hi:p:ufr:d:g:qn: opt; do
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
        d)  deploy_files+=("$OPTARG")
            echo "Adding $OPTARG to file deployment list"
            ;;
        g)  git_hash="$OPTARG"
            echo "Using $git_hash as git hash"
            ;;
        q)  quiet=1
            echo "Set to be quiet"
            ;;
        n)  username="$OPTARG"
            echo "Changing username to $username"
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

if [[ "$git_hash" == "" ]]; then
    git_hash=`git rev-parse --short HEAD`
fi
archive="clrbrain_${git_hash}.zip"
git archive -o "$archive" "$git_hash"

# basic deployment
deploy_files+=("$archive")
server_cmd="unzip -o $archive -d clrbrain"
cmd_fiji_update="Fiji.app/ImageJ-linux64 --update update"

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

# add on Fiji update
if [[ $update_fiji -eq 1 ]]; then
    server_cmd+=" && $cmd_fiji_update"
fi

# append customized run script and execute it
if [[ "$run_script" != "" ]]; then
    deploy_files+=("$run_script")
    server_cmd+=" && ./$run_script"
fi

# summarize files and commands
echo -e "\nFiles to deploy:"
echo "${deploy_files[@]}"
echo -e "\nCommand to execute on server:"
echo "$server_cmd"
echo ""

# run remote command on server
run_remote() {
    ssh -i "$pem" "${username}@${ip}" "$server_cmd"
}

# execute upload and server commands
scp -i "$pem" -r ${deploy_files[@]} "${username}@${ip}":~
if [[ $quiet -eq 0 ]]; then
    run_remote
else
    # suppress output and return immediately
    run_remote >&- 2>&- <&- &
fi
