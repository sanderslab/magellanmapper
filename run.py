#! /usr/bin/env python
# Simple startup script for MagellanMapper
# Author: David Young, 2017, 2020
"""Cross-platform environment activation and MagellanMapper launcher script.

Launches MagellanMapper within a Conda or Venv environment. Environment
activation is in this order:
* If a Conda or Venv environment is already activate, launch in current
  environment
* Attempt to activate a Conda environment
* Attempt to activate a Venv environment

Conda activation assumes that the ``conda`` command is accessible, typically
from a previous initialization through ``conda init`` (preferred) or
by adding the ``conda`` binary directory to the ``PATH``.

"""

import os
import platform
import subprocess
import sys

#: str: Name of Conda or Venv environment
ENV_NAME = "mag"

#: str: Directory with Venvs directories.
VENV_DIR = "../venvs"

#: List[str]: Shell commands to launch CLI.
ARGS_CLI = ["python -u -c \"from magmap.io import cli; cli.main()\" "]

#: List[str]: Shell commands to launch the main GUI.
ARGS_VIS = [
    "python -u -c \"from magmap.gui import visualizer; visualizer.main()\" ",
]

# Conda hook for Windows Command Prompt
_ARG_CONDA_HOOK_WIN = "conda_hook.bat"

#: List[str]: Shell commands to activate a Conda environment.
# add Conda hook for Bash shells to temporarily initialize Conda if
# `conda init` was not run, allowing commands such as `conda activate`
# TODO: add other shells
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


def launch_cli():
    """Launch the command-line interface."""
    from magmap.io import cli
    cli.main()


def launch_vis():
    """Launch the main graphical user interface."""
    from magmap.gui import visualizer
    visualizer.main()


def main(gui=True):
    """Launch the main MagellanMapper GUI.

    If necessary, attempt to activate a virtual environment created
    by MagellanMapper.

    Args:
        gui (bool): True to start the main GUI; defaults to True. If False,
            the CLI will be started instead.

    """
    working_dir = os.path.dirname(os.path.abspath(__file__))
    if is_conda_activated() or is_venv_activated():
        # launch MagellanMapper if environment is already active
        if gui:
            launch_vis()
        else:
            launch_cli()
        return

    args = ARGS_VIS if gui else ARGS_CLI
    args[0] += " ".join(sys.argv[1:])
    if platform.system() == "Windows":
        # replace Conda hook with Command Prompt shell hook
        # TODO: check whether this hook command is necessary in Windows
        ARGS_CONDA[0] = _ARG_CONDA_HOOK_WIN
    try:
        # activate Conda environment, assuming default name in setup script
        # and need to initialize shell, and launch MagellanMapper
        print("Attempting to activate Conda environment")
        launch_subprocess(ARGS_CONDA + args, working_dir)
    except subprocess.CalledProcessError:
        try:
            # non-POSIX shells do not accept eval but may run without
            # initializing the Conda shell hook
            print("Retrying Conda activation without shell hook")
            launch_subprocess(ARGS_CONDA[1:] + args, working_dir)
        except subprocess.CalledProcessError:
            try:
                # if unable to activate Conda env, try Venv
                print("Conda environment not available, trying Venv")
                launch_subprocess(
                    ["source {}/{}/bin/activate".format(VENV_DIR, ENV_NAME)]
                    + args, working_dir)
            except subprocess.CalledProcessError:
                # as fallback, attempt to launch without activating
                # an environment
                print("Neither environment is available, attempting to launch "
                      "without environment")
                launch_vis()


if __name__ == "__main__":
    print("Starting MagellanMapper run script...")
    main()
