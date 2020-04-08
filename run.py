#! /usr/bin/env python
# Simple startup script for MagellanMapper
# Author: David Young, 2017, 2020

import os
import subprocess
import sys

#: str: Name of Conda or Venv environment
ENV_NAME = "mag"

#: str: Directory with Venvs directories.
VENV_DIR = "../venvs"

#: List[str]: Shell commands to launch :meth:`magmap.gui.visualizer.main`
ARGS_VIS = [
    "python -u -c \"from magmap.gui import visualizer; visualizer.main()\" {}"
    .format(" ".join(sys.argv[1:])),
]

#: List[str]: Shell commands to activate a Conda environment
ARGS_CONDA = [
    "eval \"$(conda shell.bash hook)\"",
    "conda activate {}".format(ENV_NAME),
]


def is_conda_activated():
    """Check whether a Conda environment is active.

    Simply checks whether any environment is active to allow flexibility
    in case the given environment does not have the exact same name as
    :const:`ENV_NAME`.

    Returns:
        True if a Conda environment is currently activated.

    """
    return "CONDA_PREFIX" in os.environ


def is_venv_activated():
    """Check whether a Venv or virtualenv environment is active.

    Returns:
        True if the environment is activated.

    """
    return (hasattr(sys, "real_prefix") or
            (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix))


def launch_subprocess(args, working_dir=None):
    """Launch a subprocess with multiple commands strung together by
    logical ands.

    Args:
        args (List): List of shell commands.
        working_dir (str): Working directory path; defaults to None.

    Returns:
        int: 0 if the return code was 0; otherwise, raises a
        :class:`subprocess.CalledProcessError`.

    """
    return subprocess.check_call("&&".join(args), shell=True, cwd=working_dir)


def launch_vis():
    """Launch :meth:`magmap.gui.visualizer.main`."""
    from magmap.gui import visualizer
    visualizer.main()


def main():
    """Launch the main MagellanMapper GUI.

    If necessary, attempt to activate a virtual environment created
    by MagellanMapper.
    """
    working_dir = os.path.dirname(os.path.abspath(__file__))
    if is_conda_activated() or is_venv_activated():
        # launch GUI if environment is already active
        launch_vis()
        return

    try:
        # activate Conda environment, assuming default name in setup script
        # and need to initialize shell, and launch GUI
        print("Attempting to activate Conda environment")
        launch_subprocess(ARGS_CONDA + ARGS_VIS, working_dir)
    except subprocess.CalledProcessError:
        try:
            # non-POSIX shells do not accept eval but may run without
            # initializing the Conda shell hook
            launch_subprocess(ARGS_CONDA[1:] + ARGS_VIS, working_dir)
        except subprocess.CalledProcessError:
            try:
                # if unable to activate Conda env, try Venv
                print("Conda environment not available, trying Venv")
                launch_subprocess(
                    ["source {}/{}/bin/activate".format(VENV_DIR, ENV_NAME)]
                    + ARGS_VIS, working_dir)
            except subprocess.CalledProcessError:
                # as fallback, attempt to launch without activating
                # an environment
                print("Neither environment is available, attempting to launch "
                      "without environment")
                launch_vis()


if __name__ == "__main__":
    print("Starting MagellanMapper run script...")
    main()
