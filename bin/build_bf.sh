#!/usr/bin/env bash
# Build script to compile a wheel for Python-Bioformats

HELP="
Compile a build a wheel for Python-Bioformats.

The repository will be cleaned to its original state, removing all untracked
files, to ensure that the build does not have stale artifacts.

By default, a forked Python-Bioformats repository will be downloaded.

Arguments:
  -h: Show help and exit.
  -d [path]: Python-Bioformats directory.
  -o [path]: Output directory to copy wheel and source distribution.
"

bf_dir=""
out_dir=""

OPTIND=1
while getopts hd:j:o: opt; do
  case $opt in
    h)
      echo "$HELP"
      exit 0
      ;;
    d)
      bf_dir="$OPTARG"
      echo "Changed Python-Python-Bioformats directory to $bf_dir"
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

echo "Initating build of Python-Bioformats..."

# base directory is script's parent directory
BASE_DIR="$(dirname "$0")/.."
cd "$BASE_DIR" || { echo "Unable to find folder $BASE_DIR, exiting"; exit 1; }
base_dir="$PWD"
if [[ -z "$bf_dir" ]]; then
  # default to look for repo in parent of base dir
  bf_dir="${base_dir}/../python-Python-Bioformats"
elif [[ "$bf_dir" != /* ]]; then
  # convert relative Python-Bioformats to abs dir path
  bf_dir="${base_dir}/${bf_dir}"
fi
if [[ -n "$out_dir" && "$out_dir" != /* ]]; then
  # convert relative output to abs dir path
  out_dir="${base_dir}/${out_dir}"
fi

# get and/or enter git repo
if [[ ! -d "$bf_dir" ]]; then
  # get forked repo that depends on the original Javabridge package rather
  # than the forked Python-Javabridge version
  echo "Cloning Python-Python-Bioformats git repo to ${bf_dir}..."
  git clone https://github.com/yoda-vid/python-bioformats.git "$bf_dir"
fi
cd "$bf_dir" || { echo "Unable to find folder $bf_dir, exiting"; exit 1; }

# restore repo dir to pristine state
git clean -dfx

# build binaries, wheel, and source distribution
echo "Building Python-Python-Bioformats"
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

echo "Done building Python-Python-Bioformats"
