#!/usr/bin/env python
# Simple startup script for MagellanMapper
# Author: David Young, 2017, 2020
"""Cross-platform environment activation and MagellanMapper launcher script.

Launches MagellanMapper within a Conda or Venv environment. Environment
activation is in this order:
* If a Conda or Venv environment is already activate, launch in current
  environment
* Attempt to activate a Conda environment
* Attempt to activate a Venv environment

Executing this script as a text file (ie not through Python directly)
assumes that the ``python`` command is accessible without a Python
environment. Use ``bin/runaltpy.sh`` instead if only ``python3`` is
available (eg on Linux) or ``python`` is only available through Conda.

Conda activation assumes that the ``conda`` command is accessible, typically
from a previous initialization through ``conda init`` (preferred) or
by adding the ``conda`` binary directory to the ``PATH``. Note that ``conda``
may not be available in environments that do not load the full shell
configuration such as Python launched via Finder in the MacOS platform.

"""

import os
import platform
import subprocess
import sys

#: str: Name of Conda or Venv environment
ENV_NAME = "mag"

#: str: Directory with Venvs directories.
VENV_DIR = "../venvs"

#: str: Conda environment variable for the currently activated environment name.
_CONDA_ENV_KEY = "CONDA_DEFAULT_ENV"

#: List[str]: Shell commands to launch the MagellanMapper.
ARGS_MAGMAP = [
    "python -u -c \"from magmap.io import cli; cli.main(); "
    "from magmap.gui import visualizer; visualizer.main()\" {}"
    .format(" ".join(sys.argv[1:])),
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
    """Check whether the MagellanMapper Conda environment is active.
    
    The environment name does not have to exactly match :const:`ENV_NAME`
    but at least start with this name to provide some flexibility for
    different versions of this environment, such as ``mag2``.

    Returns:
        bool: True if a Conda environment starting with the name
        :const:`ENV_NAME` is currently activated.

    """
    return (_CONDA_ENV_KEY in os.environ
            and os.environ[_CONDA_ENV_KEY].startswith(ENV_NAME))


def is_venv_activated():
    """Check whether a Venv or virtualenv environment is active.

    Returns:
        True if the environment is activated.

    """
    return (hasattr(sys, "real_prefix") or
            (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix))


def launch_subprocess(args, working_dir=None, sys_shell=False):
    """Launch a subprocess with multiple commands strung together by
    logical ands.

    Args:
        args (List): List of shell commands.
        working_dir (str): Working directory path; defaults to None.
        sys_shell (bool): True to launch the subprocess in the system shell;
            otherwise, ``args`` will be run in an interactive Bash shell.
            Defaults to False.

    Returns:
        int: 0 if the return code was 0; otherwise, raises a
        :class:`subprocess.CalledProcessError`.

    """
    subproc_args = {
        "args": "&&".join(args),
        "cwd": working_dir,
        "shell": sys_shell,
        "stderr": subprocess.STDOUT,
    }
    if not sys_shell:
        subproc_args["args"] = ["bash", "-i", "-c", subproc_args["args"]]
    return subprocess.check_output(**subproc_args)


def launch_magmap():
    """Launch MagellanMapper.
    
    First launch the CLI to process user arguments, which will shut down
    the process afterward if a CLI task has been selected. If the process
    remains alive, the GUI will be launched.
    
    """
    from magmap.io import cli
    cli.main()
    from magmap.gui import visualizer
    visualizer.main()


def main():
    """Launch MagellanMapper with environment activation.

    If necessary, attempt to activate a virtual environment created
    by MagellanMapper.

    """
    working_dir = os.path.dirname(os.path.abspath(__file__))
    if is_conda_activated() or is_venv_activated():
        # launch MagellanMapper if environment is already active
        launch_magmap()
        return
    
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        # bypass environment activation when in PyIntaller frozen env
        print("Launching from from bundled environment")
        launch_magmap()
        return

    use_sys_shell = False
    if platform.system() == "Windows":
        # replace Conda hook with Command Prompt shell hook
        # TODO: check whether this hook command is necessary in Windows
        ARGS_CONDA[0] = _ARG_CONDA_HOOK_WIN
        use_sys_shell = True
    try:
        # activate Conda environment, assuming default name in setup script
        # and need to initialize shell, and launch MagellanMapper
        print("Attempting to activate Conda environment")
        launch_subprocess(ARGS_CONDA + ARGS_MAGMAP, working_dir, use_sys_shell)
    except subprocess.CalledProcessError as e:
        print(e.output)
        try:
            # non-POSIX shells do not accept eval but may run without
            # initializing the Conda shell hook
            print("Retrying Conda activation without shell hook")
            launch_subprocess(
                ARGS_CONDA[1:] + ARGS_MAGMAP, working_dir, use_sys_shell)
        except subprocess.CalledProcessError:
            print(e.output)
            try:
                # if unable to activate Conda env, try Venv
                print("Conda environment not available, trying Venv")
                launch_subprocess(
                    ["source {}/{}/bin/activate".format(VENV_DIR, ENV_NAME)]
                    + ARGS_MAGMAP, working_dir, use_sys_shell)
            except subprocess.CalledProcessError:
                print(e.output)
                # as fallback, attempt to launch without activating
                # an environment
                print("Neither environment is available, attempting to launch "
                      "without environment")
                launch_magmap()


if __name__ == "__main__":
    print("Starting MagellanMapper run script...")
    main()
