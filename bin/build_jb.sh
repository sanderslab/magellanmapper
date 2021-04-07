#!/usr/bin/env bash
# Build script to compile Javabridge

HELP="
Compile Python-Javabridge and build wheels.

The repository will be cleaned to its original state, removing all untracked
files, to ensure that the build does not have stale artifacts.

Assumes that the Venv environments have been set up by
\"bin/setup_multi_venvs.sh\".

Arguments:
  -h: Show help and exit.
  -d [path]: Javabridge directory.
  -j [path]: Java home path.
  -o [path]: Output directory to copy wheel and source distribution.
"

jb_dir=""
out_dir=""
java_home=""

OPTIND=1
while getopts hd:j:o: opt; do
  case $opt in
    h)
      echo "$HELP"
      exit 0
      ;;
    d)
      jb_dir="$OPTARG"
      echo "Changed Python-Javabridge directory to $jb_dir"
      ;;
    j)
      java_home="$OPTARG"
      echo "Changed Java home to $java_home"
      ;;
    o)
      out_dir="$OPTARG"
      echo "Changed output directory to $out_dir"
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

# attempt compatibility with lowest Mac target; Python must have been
# compiled with at least this target, or setting will be ignored
export MACOSX_DEPLOYMENT_TARGET=10.9

if [[ -n "$java_home" ]]; then
  # use the given Java
  export JAVA_HOME="$java_home"
  export PATH="${JAVA_HOME}/bin:$PATH"
  java -version
fi

echo "Initating build of Javabridge..."

# base directory is script's parent directory, and default repo and build dirs 
# are in parent directory of base dir
BASE_DIR="$(dirname "$0")/.."
cd "$BASE_DIR" || { echo "Unable to find folder $BASE_DIR, exiting"; exit 1; }
base_dir="$PWD"
if [[ -z "$jb_dir" ]]; then
  jb_dir="${base_dir}/../python-javabridge"
fi

# get and/or enter git repo
if [[ ! -d "$jb_dir" ]]; then
  echo "Cloning Python-Javabridge git repo to ${jb_dir}..."
  git clone https://github.com/LeeKamentsky/python-javabridge.git "$jb_dir"
fi
cd "$jb_dir" || { echo "Unable to find folder $jb_dir, exiting"; exit 1; }

# restore repo dir to pristine state
git clean -dfx

# build binaries, wheel, and source distribution
echo "Building Python-Javabridge"
python setup.py build
python setup.py bdist_wheel
python setup.py sdist

if [[ -n "$out_dir" ]]; then
  # copy wheel and source distribution to output dir
  if [[ ! -d "$out_dir" ]]; then
    mkdir "$out_dir"
  fi
  cp -v "dist/"*.whl "dist/"*.tar.gz "$out_dir"
fi

echo "Done building Python-Javabridge"
