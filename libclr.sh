#!/bin/bash
# Bash library functions for Clrbrain
# Author: David Young 2018


############################################
# Backup old directory if necessary and create a new one.
# Globals:
#   NONE
# Arguments:
#   1: Path of directory to back up. The backed up directory will have the 
#      same name with the next available integer in parentheses appended. 
#      After backup, an emptry directory at the original location will be made.
# Returns:
#   NONE
############################################
backup_dir() {
  curr_dir="$1"
  if [[ -e "$curr_dir" ]]; then
    curr_dir_last="${1}(1)"
    i=2
    while [ -e "$curr_dir_last" ]; do
      # increment 
      curr_dir_last="$1("$i")"
      let i++
    done
    mv "$curr_dir" "$curr_dir_last"
    echo "Backed up directory to $curr_dir_last"
  fi
  mkdir "$curr_dir"
  echo "Created directory: $curr_dir"
}

############################################
# Suppress all output.
# Globals:
#   Redirects all streams to suppress output
# Arguments:
#   NONE
# Returns:
#   NONE
############################################
# TODO: not currently used
suppress_output() {
  exec 3>&1
  exec 4>&2
  if [[ $1 -eq 1 ]]; then
    exec >&-
    exec 2>&-
  else
    exec >&3
    exec 2>&4
  fi
}
