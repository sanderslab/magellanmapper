#!/usr/bin/env bash
# Simple wrapper script to launch MagellanMapper if `python3` is available
# but `python` is not, or `python` is only available through Conda
# Author: David Young, 2020

# assumes run.py is in current directory
if command -v python3 &> /dev/null; then
  # launch run script directly, allowing it to manage Conda
  python3 run.py
else
  # attempt launch through python installed with Conda, using base env
  # since the run script will handle the appropriate environment
  bash -i -c "conda run python run.py"
fi
