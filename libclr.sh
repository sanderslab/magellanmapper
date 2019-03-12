#!/bin/bash
# Bash library functions for Clrbrain
# Author: David Young 2018, 2019


############################################
# Backup a file or directory if necessary.
# Globals:
#   NONE
# Arguments:
#   1: Path of file/directory to back up. The backed up file will have the 
#      same name with the next available integer in parentheses appended, 
#      either at the end of the filename for directories or before the 
#      extension otherwise.
# Returns:
#   NONE
############################################
backup_file() {
  local curr_path="$1"
  local base="$curr_path"
  local ext=""
  if [[ ! -d "$curr_path" ]]; then
    # split into before and after extension
    base="${curr_path%.*}"
    if [[ "$base" != "$curr_path" ]]; then
      ext=".${curr_path##*.}"
    fi
  fi
  echo "$base $ext"
  if [[ -e "$curr_path" ]]; then
    local curr_path_last="${base}(1)${ext}"
    local i=2
    while [ -e "$curr_path_last" ]; do
      # increment 
      curr_path_last="${base}("$i")${ext}"
      let i++
    done
    mv "$curr_path" "$curr_path_last"
    echo "Backed up directory to $curr_path_last"
  fi
}

############################################
# Detect computer platform including OS and bit.
# Globals:
#   os: Operating system, which is one of Windows, MacOSX, or Linux.
#   os_ver: OS version, identified for Mac.
#   bit: Architecture with bit, such as x86 or x86_64
# Arguments:
#   NONE
# Returns:
#   NONE
############################################
detect_platform() {
  echo -n "Detecting platform..."
  local system=`uname -a`
  
  # detect operating system
  os=""
  os_ver=""
  if [[ "$system" =~ "CYGWIN" ]] || [[ "$system" =~ "WINDOWS" ]]; then
    os="Windows"
  elif [[ "$system" =~ "Darwin" ]]; then
    os="MacOSX"
    os_ver="$(/usr/bin/sw_vers -productVersion)"
  elif [[ "$system" =~ "Linux" ]]; then
    os="Linux"
  fi
  if [[ -z "$os" ]]; then
    echo "Could not detect OS"
  else
    readonly os
    readonly os_ver
  fi
  
  # detect bit
  bit="x86"
  if [[ "$system" =~ "x86_64" ]]; then
    bit="x86_64"
  fi
  readonly bit
  echo "Found $os platform with $bit bit"
}

############################################
# Check Python version.
# Globals:
#   NONE
# Arguments:
#   1: Python command to get version.
#   2: Python minimum major version number.
#   3: Python minimum minor version number.
# Returns:
#   True if the Python version meets or excceds the version requirement.
############################################
check_python() {
  local py_ver="$("$1" -V 2>&1)"
  local py_ver="${py_ver#* }"
  local py_ver_maj="${py_ver%%.*}"
  local py_ver_rest="${py_ver#*.}"
  local py_ver_min="${py_ver_rest%%.*}"
  [[ $py_ver_maj -ge "$2" && $py_ver_min -ge "$3" ]]
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
