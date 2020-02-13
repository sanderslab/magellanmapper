#!/bin/bash
# Bash library functions for MagellanMapper
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
      curr_path_last="${base}($i)${ext}"
      ((i++))
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
  local system
  system="$(uname -a)"
  
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
  echo "found $os platform with $bit bit"
}

############################################
# Check Python version.
# Globals:
#   PY_VER: The found Python version in maj.min format, or an empty string
#     if Python not found.
# Arguments:
#   1: Python command to get version.
#   2: Python minimum major version number.
#   3: Python minimum minor version number.
# Returns:
#   True if the Python version meets or exceeds the version requirement.
############################################
check_python() {
  local py_ver
  PY_VER=""
  if ! command -v "1" &> /dev/null; then
    return 1
  fi
  py_ver="$("$1" -V 2>&1)"
  py_ver="${py_ver#* }"
  local py_ver_maj="${py_ver%%.*}"
  local py_ver_rest="${py_ver#*.}"
  local py_ver_min="${py_ver_rest%%.*}"
  PY_VER="${py_ver_maj}.${py_ver_min}"
  [[ $py_ver_maj -ge "$2" && $py_ver_min -ge "$3" ]]
}

############################################
# Check for existing Java compiler.
# Globals:
#   NONE
# Arguments:
#   NONE
# Returns:
#   0 if javac is found; 1 if otherwise.
############################################
check_javac() {
  if ! command -v "javac" &> /dev/null; then
    # check for Java compiler availability for Javabridge
    echo "Warning: \"javac\" not found; Python-Bioformats and Python-Javabridge"
    echo "will ot install correctly. Please install a JDK or add JDK_HOME or"
    echo "add JAVA_HOME to your path environment variables"
    return 1
  fi
  return 0
}

############################################
# Check for existing, installed Mac Command-Line Tools
# Globals:
#   os: Operating System string to check for MacOS.
# Arguments:
#   NONE
# Returns:
#   0 if an activated CLT installation is found; 1 if otherwise.
############################################
check_clt() {
  if [[ "$os" == "MacOSX" ]]; then
    if [[ ! -e "/Library/Developer/CommandLineTools/usr/bin/git" ]]; then
      # Mac-specific check for command-line tools (CLT) package since the 
      # commands that are not activated will still return
      if [[ "$os_ver" < "10.14" && -e "/usr/include/iconv.h" ]]; then
        # vers < 10.14 require both git and CLT headers
        :
      else
        echo "Warning: Mac command-line tools not present/activated;"
        echo "installations that require compilation may not work properly."
        echo "If you encounter problems related to compilation, please run "
        echo "\"xcode-select --install\""
        return 1
      fi
    fi
  fi
  return 0
}

############################################
# Check for existing gcc compiler.
# Globals:
#   NONE
# Arguments:
#   NONE
# Returns:
#   0 if gcc is found; 1 if otherwise.
############################################
check_gcc() {
  if ! command -v "gcc" &> /dev/null; then
    # check for gcc availability for compiling Scikit-image (if directly from 
    # repo), Traits (if not from Conda), and Javabridge
    echo "Warning: \"gcc\" not found; installations that require compilation"
    echo "may not work properly. If you encounter problems related to"
    echo "compilation, please install \"gcc\"."
    return 1
  fi
  return 0
}

############################################
# Check for existing Git executable.
# Globals:
#   NONE
# Arguments:
#   NONE
# Returns:
#   0 if git is found; 1 if otherwise.
############################################
check_git() {
  if ! command -v "git" &> /dev/null; then
    # check for git availability for downloading repos for any installs 
    # from Git repos
    echo "Warning: \"git\" not found; installations that require repository"
    echo "access may not work properly. If you encounter problems related to"
    echo "repository downloads, please install \"git\"."
    return 1
  fi
  return 0
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

############################################
# Join array elements into a delimited string
# Globals:
#   NONE
# Arguments:
#   1: Name of array to join.
#   2: Separator.
# Returns:
#   NONE
############################################
join_array() {
  local -n arr="$1"
  local sep="$2"
  local ticks
  ticks=$(printf "${sep}%s" "${arr[@]}")
  ticks="${ticks:${#sep}}"
  echo "$ticks"
}
