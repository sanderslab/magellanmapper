# Input/Output with the Python Subprocess module
# Author: David Young, 2020
"""File management using underlying system commands accessed through the
Python :mod:`subprocess` module.
"""

import os
import subprocess

from magmap.io import libmag


def compress_file(path_in, path_out=None):
    """Compress a file or files by tar archving and compressing with.

    Assumes that ``tar`` and ``zstd`` are available shell commands.

    Args:
        path_in (str, List[str]): Input file path; can be a sequence of paths,
            in which case the first path will determine the working directory
            and the default ``path_out`` if none is given.
        path_out (str): Output file path; defaults to None to use the
            same path as ``path_in`` with ``.tar.zstd`` appended.

    """
    tar_args = ["tar", "cfv", "-"]
    if libmag.is_seq(path_in):
        # set up sequence of paths, using the first path as template
        path_first = path_in[0]
        tar_args.extend([os.path.basename(p) for p in path_in])
    else:
        # set up a single path
        path_first = path_in
        tar_args.append(os.path.basename(path_in))
    if path_out is None:
        # default to using the first path for compressed file name
        path_out = os.path.splitext(path_first)[0] + ".tar.zst"
    # use first path for working directory
    wd = os.path.dirname(path_first)
    if not wd:
        wd = None

    # archive with tar and pipe to zstd for compression
    print("Compressing \"{}\" to \"{}\"".format(path_in, path_out))
    tar = subprocess.Popen(
        tar_args, cwd=wd, stdout=subprocess.PIPE, bufsize=0)
    with open(path_out, "wb") as f:
        zstd = subprocess.Popen(
            ["pzstd", "-v"], stdin=tar.stdout, stdout=f, bufsize=0)
        tar.stdout.close()
        # appears to work for large image with memory issues because data
        # are backed by files
        stderr = zstd.communicate()[1]
        if stderr:
            print(stderr)


def test_compression(path):
    """Test the integrity of a file compressed by ZSTD.

    Args:
        path (str): Path to compressed file.

    Returns:
        bool: True if the integrity check completed without error; False
        if otherwise.

    """
    print("Testing integrity of compressed file:", path)
    try:
        subprocess.check_call(["pzstd", "-t", path])
        print("Integrity test of \"{}\" completed without error".format(path))
        return True
    except subprocess.CalledProcessError as e:
        print(e)
        print("Error during compression integrity testing of", path)
    return False


def decompress_file(path_in, dir_out=None):
    """Decompress and unarchive a file.

    Assumes that the file has been archived by ``tar`` and compressed
    by ``zstd``, both available as shell commands.

    Args:
        path_in (str): Input path.
        dir_out (str): Output directory path; defaults to None to output
            to the current directory.

    """
    tar_args = ["tar", "xvf", "-"]
    if dir_out:
        if not os.path.isdir(dir_out):
            os.makedirs(dir_out)
        tar_args.extend(["-C", dir_out])
    else:
        dir_out = "."
    print("decompressing {} to {}".format(path_in, dir_out))
    zst = subprocess.Popen(
        ["pzstd", "-dc", path_in], stdout=subprocess.PIPE, bufsize=0)
    tar = subprocess.Popen(tar_args, stdin=zst.stdout, bufsize=0)
    zst.stdout.close()
    stderr = tar.communicate()[1]
    if stderr:
        print(stderr)
