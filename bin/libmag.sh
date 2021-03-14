#!/usr/bin/env bash
# Bash library functions for MagellanMapper
# Author: David Young 2018, 2020


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
#   OS_NAME: Operating system, which is one of Windows, MacOSX, or Linux.
#   OS_VER: OS version, identified for Mac.
#   BIT: Architecture with bit, such as x86 or x86_64
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
  OS_NAME=""
  OS_VER=""
  if [[ "$system" =~ "CYGWIN" ]] || [[ "$system" =~ "WINDOWS" ]]; then
    OS_NAME="Windows"
  elif [[ "$system" =~ "Darwin" ]]; then
    OS_NAME="MacOSX"
    OS_VER="$(/usr/bin/sw_vers -productVersion)"
  elif [[ "$system" =~ "Linux" ]]; then
    OS_NAME="Linux"
  fi
  if [[ -z "$OS_NAME" ]]; then
    echo "Could not detect OS"
  else
    readonly OS_NAME
    export OS_NAME
    readonly OS_VER
    export OS_VER
  fi
  
  # detect bit
  BIT="x86"
  if [[ "$system" =~ "x86_64" ]]; then
    BIT="x86_64"
  fi
  readonly BIT
  export BIT
  echo "found $OS_NAME platform with $BIT bit"
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
# Check for existing Java runtime.
# Globals:
#   NONE
# Arguments:
#   NONE
# Returns:
#   0 if java is found; 1 if otherwise.
############################################
check_java() {
  if ! command -v "java" &> /dev/null; then
    # check for Java availability for Javabridge
    echo "Warning: \"java\" not found; Python-Bioformats and Python-Javabridge"
    echo "may not install or run correctly. Please install a JDK or add"
    echo "JDK_HOME or add JAVA_HOME to your path environment variables and"
    echo "reinstall if you need these dependencies."
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
# Build a Java Runtime Environment image.
# Globals:
#   JAVA_HOME: Path to JRE home directory.
# Arguments:
#   1: Path for jdeps to find dependencies.
#   2: Output path.
# Returns:
#   NONE
############################################
build_jre() {
  echo "Building custom JRE in $2"
  # find dependencies for the given CLASS or JAR file
  deps="$(jdeps --print-module-deps --recursive --ignore-missing-deps -q "$1")"
  echo "Includes dependencies: $deps"
  
  # build a lightweight runtime image
  "$JAVA_HOME/bin/jlink" --no-header-files --no-man-pages \
    --compress=2 --strip-java-debug-attributes --add-modules "$deps" \
    --output "$2"
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
# Download a shallow Git clone and pip install its Python package.
# Globals:
#   None
# Arguments:
#   Git repository URL
# Returns:
#   None
############################################
install_shallow_clone() {
  local folder
  folder="$(basename "$1")"
  folder="${folder%.*}"
  if [ ! -e "$folder" ]; then
    # download and install fresh repo with shallow clone
    # and editable installation
    # TODO: check whether shallow clone will yield the
    # correct fetch/merge steps later
    echo "Cloning into $folder"
    target=$1
    if [[ "$#" -gt 1 ]]; then
      target="${target} -b $2"
    fi
    git clone --depth 1 $target
    cd "$folder"
  else
    # update repo if changes found upstream on given branch
    echo "Updating $folder"
    cd "$folder"
    git fetch
    branch="master" # default branch
    if [[ "$#" -gt 1 ]]; then
      # use given branch, if any
      branch="$2"
      echo "Checking for differences with $branch"
    fi
    if [[ $(git rev-parse --abbrev-ref HEAD) != "$branch" ]]; then
      echo "Not on $branch branch so will ignore updates"
    elif [[ $(git diff-index HEAD --) ]]; then
      echo "Uncommitted file changes exist so will not update"
    elif [[ $(git log HEAD..origin/"$branch" --oneline) ]]; then
      # merge in updates only if on same branch as given one,
      # differences exist between current status and upstream
      # branch, and no tracked files have uncommitted changes
      git merge "origin/$branch"
      echo "You may need to run post-update step such as "
      echo "\"python setup.py build_ext -i\""
    else
      echo "No changes found upstream on $branch branch"
    fi
  fi
  if [[ ! "$(pip list --format=columns | grep $folder)" ]]; then
    echo "Installing $folder"
    pip install -e .
  fi
  cd ..
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

############################################
# Find directory that contains a matching file from a list of directories.
# Prioritizes files ending with "npy" (eg "image5d.npy").
# Globals:
#   PWD: Default echo value.
# Arguments:
#   1: Name of array with directory names.
#   2: Start of file name to match.
# Returns:
#   0 if any matching path is found, 1 if not. Echoes directory from $1
#   where a match was found.
############################################
find_prefix() {
  if ! local -n dirs="$1"; then
    # fallback to indirect expansion if nameref is not available (Bash < 4.3)
    local dirs="${1}[@]"
    dirs=("${!dirs}")
  fi
  local name="$2"
  prefix=""
  for p in "${dirs[@]}"; do
    for f in "$p/$name"*npy; do
      # prioritize checking for NPY files (image5d)
      if [[ -f "$f" ]]; then
        echo "$p"
        return 0
      fi
    done
  done
  for p in "${dirs[@]}"; do
    for f in "$p/$name"*; do
      # fall back to any file extension
      if [[ -f "$f" ]]; then
        echo "$p"
        return 0
      fi
    done
  done
  echo "$PWD"
  return 1
}

############################################
# Set up MagellanMapper image file paths.
# Globals:
#   IMG: Main image path.
#   IMG_MHD: Path to main image in MHD format.
#   SHAPE_RESIZED: Shape in x,y,z for resized image name.
#   IMG_RESIZED: Resized main image.
# Arguments:
#   1: Name of array with directory names.
#   2: Start of file name to match.
############################################
setup_image_paths() {
  # set image prefix based on identified location of files matching BASE
  local name="$2"
  local prefix
  if prefix="$(find_prefix "$1" "$name")"; then
    echo "Found image files starting with $name in $prefix"
  else
    msg="WARNING: could not find image file associated with \"$name.\" "
    msg+="\nWill assume image files are located in \"$prefix\"."
    echo -e "$msg"
  fi
  IMG="$prefix/${name}."

  # set paths from identified prefix
  IMG_MHD="$prefix/${name}.mhd"
  IMG_RESIZED="$prefix/${name}_resized($SHAPE_RESIZED)."
}

############################################
# Set up atlas image file paths.
# Globals:
#   ABA_SPEC: Atlas mapping/specification filename in atlas directory.
#   ABA_PATH: Main atlas directory path.
#   ABA_LABELS: Path to atlas labels map file.
#   ABA_IMPORT_DIR: Path to atlas imported from ABA_PATH, assumed to have
#     the same name as ABA_PATH but "_import" appended.
# Arguments:
#   1: Name of array with names of directories that may hold the atlas dir.
#   2: Atlas directory name.
############################################
setup_atlas_paths() {
  # set label directory paths
  local aba_dir="$2"
  if ABA_PATH="$(find_prefix "$1" "$aba_dir/$ABA_SPEC")/$aba_dir"; then
    echo "Found atlas spec file $aba_dir/$ABA_SPEC in $(dirname "$ABA_PATH")"
  else
    msg="WARNING: could not find atlas directory associated with \"$aba_dir.\" "
    msg+="\nWill use the atlas path: \"$ABA_PATH\"."
    echo -e "$msg"
  fi
  ABA_LABELS="$ABA_PATH/$ABA_SPEC"
  ABA_IMPORT_DIR="${ABA_PATH}_import"
}
