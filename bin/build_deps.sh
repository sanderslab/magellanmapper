#!/usr/bin/env bash
# Build wheels for MagellanMapper dependencies

venv_dir="../venvs"
output_dir="../build_deps"
HELP="
Build dependencies in Venv environments for multiple Python versions.

Assumes that environments for each Python version have been set up using
\"setup_multi_venvs.sh\".

Arguments:
  -h: Show help and exit.
  -d [path]: Path to build directory; defaults to \"$output_dir\".
  -e [path]: Path to folder where the new venv directory will be placed. 
    Defaults to \"$venv_dir\".
  -j [path]: Path to Javabridge directory, passed to \"build_jb.sh\".
  -s [path]: Path to SimpleElastix directory, passed to \"build_se.sh\".
"

se_dir=()
jb_dir=()

OPTIND=1
while getopts hd:e:j:s: opt; do
  case $opt in
    h)
      echo "$HELP"
      exit 0
      ;;
    d)
      output_dir="$OPTARG"
      echo "Changed build directory to $output_dir"
      ;;
    e)
      venv_dir="$OPTARG"
      echo "Set the venv directory to $venv_dir"
      ;;
    j)
      jb_dir+=(-d "$OPTARG")
      echo "Changed Python-Javabridge directory to $OPTARG"
      ;;
    s)
      se_dir+=(-s "$OPTARG")
      echo "Changed SimpleElastix directory to $OPTARG"
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

echo "Initating build of SimpleElastix..."

# base directory is script's parent directory
BASE_DIR="$(dirname "$0")/.."
cd "$BASE_DIR" || { echo "Unable to find folder $BASE_DIR, exiting"; exit 1; }

# Activate Venv environment for the given Python version and install wheels
# package.
# Args:
#   1: Python version given as x.y, eg "3.6".
# Returns:
#   1 if the environment could not be activated, 0 otherwise.
activate_venv() {
  # activate the environment
  local ver="$1"
  local vdir
  vdir="$(cd "${venv_dir}/py${ver}" && pwd)"
  echo "Activating $vdir"
  . "${vdir}/bin/activate"
  if [[ "$VIRTUAL_ENV" != "$vdir" ]]; then
    echo "Could not activate $vdir Venv environment, skipping"
    return 1
  fi
  
  # install package to build wheels
  pip install wheel
  return 0
}

# Build SimpleElastix in the given Python version environment.
# Args:
#   1: Python version, used to activate the corresponding Venv environment.
build_se_ver() {
  # build SimpleElastix
  local ver="$1"
  local output="${output_dir}/build_se_py${ver}"
  bin/build_se.sh -d "$output" "${se_dir[@]}"
  
  # copy wheel to output dir
  local dist="${output}/SimpleITK-build/Wrapping/Python/dist"
  cp -v "${dist}/"*.whl "${dist}/"*.tar.gz "$output_dir"
}

# Build Javabridge in the given Python version environment.
# Also installs Cython and Numpy.
build_jb_ver() {
  pip install cython
  pip install 'numpy~=1.19' # v1.19 is last ver supporting Python 3.6
  local java_args=()
  if command -v "/usr/libexec/java_home" &> /dev/null; then
    java_args+=(-j "$(/usr/libexec/java_home -v 1.8)")
  fi
  bin/build_jb.sh "${jb_dir[@]}" -o "$output_dir" "${java_args[@]}"
}

if [[ ! -d "$output_dir" ]]; then
  # make output directory and its parents
  mkdir -p "$output_dir"
fi

# build dependencies for each of all supported Python versions
py_vers=(3.6 3.7 3.8 3.9)
for py_ver in "${py_vers[@]}"; do
  if activate_venv "$py_ver"; then
    build_se_ver "$py_ver"
    build_jb_ver
    deactivate
  fi
done
