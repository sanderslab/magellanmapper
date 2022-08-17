#!/usr/bin/env python
# Simple startup script for MagellanMapper
# Author: David Young, 2017, 2020
"""MagellanMapper launcher script.

See :module:`magmap.io.load_env` for details. Use ``bin/runaltpy.sh`` instead
if executing this script directly (eg `./run.py`) and only ``python3`` is
available (eg on Linux) or ``python`` is only available through Conda.

"""

import multiprocessing

from magmap.io import load_env

if __name__ == "__main__":
    # support multiprocessing in frozen environments, necessary for Windows;
    # no effect on other platforms or non-frozen environments
    multiprocessing.freeze_support()

    # start MagellanMapper
    print("Starting MagellanMapper run script...")
    load_env.main()
