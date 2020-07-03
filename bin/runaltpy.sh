#!/usr/bin/env bash
# Simple Bash wrapper script to launch MagellanMapper without relying on
# the python binary specified in the run script shebang line
# Author: David Young, 2020

# assumes run.py is in current directory
if command -v python &> /dev/null; then
  # launch run script directly from python, allowing it to manage Conda
  python run.py
elif command -v python3 &> /dev/null; then
  # use python3 instead
  python3 run.py
else
  # attempt launch through python installed with Conda, using base env
  # since the run script will handle the appropriate environment
  bash -i -c "conda run python run.py"
fi
