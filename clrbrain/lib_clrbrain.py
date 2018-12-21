#!/bin/bash
# Library functions shared within Clrbrain
# Author: David Young, 2017, 2018
"""Shared functions with the Clrbrain package.
"""

import os
import shutil
import numpy as np

from clrbrain import config

# file types that are associated with other types
_FILE_TYPE_GROUPS = {
    "obj": "mtl", 
    "mhd": "raw"
}

# Numpy numerical dtypes with various ranges
_DTYPES = {
    "uint": [np.uint8, np.uint16, np.uint32, np.uint64], 
    "int": [np.int8, np.int16, np.int32, np.int64], 
    "float": [np.float16, np.float32, np.float64]
}

def swap_elements(arr, axis0, axis1, offset=0):
    """Swap elements within an list or tuple.
    
    Args:
        arr: List or tuple in which to swap elements.
        axis0: Index of first element to swap.
        axis1: Index of second element to swap.
        offset: Offsets for indices; defaults to 0.
    
    Returns:
        The array with elements swapped. If the original array is actually
            a tuple, a new tuple with the elements swapped will be returned,
            so the return object may be different from the passed one.
    """
    axis0 += offset
    axis1 += offset
    check_tuple = isinstance(arr, tuple)
    if check_tuple:
        arr = list(arr)
    arr[axis0], arr[axis1] = arr[axis1], arr[axis0]
    if check_tuple:
        arr = tuple(arr)
    return arr

def transpose_1d(arr, plane):
    transposed = arr
    if plane == config.PLANE[1]:
        # make y the "z" axis
        transposed = swap_elements(arr, 0, 1)
    elif plane == config.PLANE[2]:
        # yz plane
        transposed = swap_elements(arr, 0, 2)
        transposed = swap_elements(arr, 1, 2)
    return transposed

def transpose_1d_rev(arr, plane):
    transposed = arr
    if plane == config.PLANE[1]:
        # make y the "z" axis
        transposed = swap_elements(arr, 1, 0)
    elif plane == config.PLANE[2]:
        # yz plane
        transposed = swap_elements(arr, 2, 1)
        transposed = swap_elements(arr, 2, 0)
    return transposed

def roll_elements(arr, shift, axis=None):
    """Roll elements in a tuple safe manner.
    
    Essentially calls Numpy.roll, but checks for tuple beforehand and converts 
    it to a Numpy array beforehand and back to a new tuple afterward.
    
    Args:
        arr: Array, which can be a tuple, list, or Numpy array. 
    
    Returns:
        The array with elements rolled. If arr is a tuple, the returned value 
            will be a new tuple. If arr is a Numpy array, a view of the array 
            will be turned.
    """
    #print("orig: {}".format(arr))
    check_tuple = isinstance(arr, tuple)
    if check_tuple:
        arr = np.array(arr)
    arr = np.roll(arr, shift, axis)
    if check_tuple:
        arr = tuple(arr)
    #print("after moving: {}".format(arr))
    return arr

def insert_before_ext(name, insert):
    """Inserts a string before an extension.
    
    Args:
        name: Path.
        insert: String to insert before the extension in the name.
    
    Returns:
        Modified path.
    """
    return "{0}{2}.{1}".format(*name.rsplit(".", 1) + [insert])

def get_filename_ext(filename):
    ext = ""
    filename_split = filename.rsplit(".", 1)
    if len(filename_split) > 1:
        ext = filename_split[1]
    return ext

def get_filename_without_ext(path):
    """Get filename without extension.
    
    Args:
        path: Full path.
    
    Returns:
        Filename alone without extension; simply returns the filename if 
        no extension exists.
    """
    name = os.path.basename(path)
    name_split = os.path.splitext(name)
    if len(name_split) > 1: return name_split[0]
    return name

def combine_paths(base_path, suffix, sep="_", ext=None):
    """Merge two paths.
    
    Args:
        base_path: Path whose dot-extension will be replaced by ``suffix``.
        suffix: Replacement including new extension.
        sep: Separator between ``base_path`` and ``suffix``.
        ext: Extension to add or substitute; defaults to None to use 
            the current extension.
    
    Returns:
        Merged path.
    """
    if not base_path: return suffix
    path = os.path.splitext(base_path)[0] + sep + suffix
    if ext: path = "{}.{}".format(os.path.splitext(path)[0], ext)
    return path

def normalize(array, minimum, maximum, background=None):
    """Normalizes an array to fall within the given min and max.
    
    Args:
        min: Minimum value for the array.
        max: Maximum value for the array.
    
    Returns:
        The normalized array, operated on in-place.
    """
    #print(array)
    if len(array) <= 0:
        return array
    foreground = array
    if background is not None:
        foreground = foreground[foreground > background]
    array -= np.min(foreground)
    print("min: {}".format(np.min(foreground)))
    array /= np.max(array) / (maximum - minimum)
    array += minimum
    if background is not None:
        array[array < minimum] = minimum
    return array

def printv(s):
    if config.verbose:
        print(s)

def get_int(val):
    """Cast a value as an integer, returning the value if any error.
    
    Args:
        val: Value to cast. If a tuple or list, each entry will be casted 
            recursively.
    
    Returns:
        Value casted to int, or the original value if any error occurs during 
        casting.
    """
    if isinstance(val, (tuple, list)):
        return [get_int(elt) for elt in val]
    try:
        return int(val)
    except:
        return val

def convert_indices_to_int(dict_to_convert):
    """Convert indices of a dictionary to int if possible, including nested 
    indices.
    
    Args:
        dict_to_convert: Dictionary whose indices will be converted.
    
    Returns:
        The converted dictionary.
    """
    dict_converted = {
        get_int(k): [get_int(i) for i in v] if isinstance(v, list) else v 
        for k, v in dict_to_convert.items()
    }
    return dict_converted

def show_full_arrays(on=True):
    """Show full Numpy arrays, except for particularly large arrays.
    
    Args:
        on: True if full printing should be turned on; False to reset
            all settings.
    """
    if on:
        np.set_printoptions(linewidth=500, threshold=10000000)
    else:
        np.set_printoptions()

def print_compact(arr, msg=None, mid=False):
    """Print a Numpy array in a compact form to visual comparison with 
    other arrays.
    
    The array will be rounded, converted to integer, and optionally 
    reduced to a single plane to maximize chance of printing a 
    non-truncated array (or plane) to compare more easily with modified 
    versions of the array or other arrays.
    
    Args:
        arr: Numpy array.
        msg: Message to print on separate line before the array; defaults to 
            None, in which case the message will not be printed.
        mid: True to only print the middle element of the array.
    
    Returns:
        The compact array as a new array.
    """
    compact = np.around(arr).astype(int)
    if msg: print(msg)
    if mid:
        i = len(compact) // 2
        print(compact[i])
    else:
        print(compact)
    return compact

def backup_file(path, modifier="", i=None):
    """Backup a file to the next given available path with an index number 
    before the extension.
    
    The backed up path will be in the format 
    ``path-before-ext[modifier](i).ext``, where ``[modifier]`` is an optional 
    additional string, and ``i`` is the index number, which will be 
    incremented to avoid overwriting an existing file. Will also backup 
    any associated files as given by :const:``_FILE_TYPE_GROUPS``.
    
    Args:
        path: Path of file to backup.
        modifier: Modifier string to place before the index number.
        i: Index to use; typically use default of None to iniate recursivie 
            backup calls.
    """
    if not i:
        if not os.path.exists(path):
            # original path does not exist, so no need to back up
            return
        i = 1
    backup_path = None
    suffix = "{}({})".format(modifier, i)
    backup_path = insert_before_ext(path, suffix)
    if not os.path.exists(backup_path):
        # backup file to currently non-existent path
        shutil.move(path, backup_path)
        print("Backed up {} to {}".format(path, backup_path))
        path_split = os.path.splitext(path)
        if len(path_split) > 1:
            # remove ".", which should exist if path was split, and get 
            # any associated file to backup as well
            ext_associated = _FILE_TYPE_GROUPS.get(path_split[1][1:])
            if ext_associated:
                # back up associated file, using the corresponding i if possible
                backup_file(
                    "{}.{}".format(path_split[0], ext_associated), modifier, i)
    else:
        # recursively try backing up with next index
        backup_file(path, modifier, i + 1)

def is_binary(img):
    """Check if image is binary.
    
    Args:
        img: Image array.
    
    Returns:
        True if the image is composed of only 0 and 1 values.
    """
    return ((img == 0) | (img == 1)).all()

def last_lines(path, n):
    """Get the last lines of a file by simply loading the entire file and 
    returning only the last specified lines, without depending on any 
    underlying system commands.
    
    Args:
        path: Path to file.
        n: Number of lines at the end of the file to extract; if the file is 
            shorter than this number, all lines are returned.
    
    Returns:
        The last ``n`` lines as a list, or all lines if ``n`` is greater than 
        the number of lines in the file.
    """
    lines = None
    with open(path, "r") as f:
        lines = f.read().splitlines()
        num_lines = len(lines)
        if n > num_lines:
            return lines
    return lines[-1*n:]

def coords_for_indexing(coords):
    """Convert array of coordinates to array for fancy indexing in a 
    Numpy array.
    
    Args:
        coords: Array of shape (n, m), where n = number of coordinate sets, 
            and m = number of coordinate dimensions.
    
    Returns:
        Array of coordinates split into axes, such as 
        `nd.array(rows_array, columns_array)`). This array can then be used 
        to index another array through `arr[tuple(indices)]`.
    """
    coordsi = np.transpose(coords)
    coordsi = np.split(coordsi, coordsi.shape[0])
    return coordsi

def dtype_within_range(min_val, max_val, integer, signed=None):
    """Get a dtype that will contain the given range.
    
    :const:``_DTYPES`` will be used to specify the possible dtypes.
    
    Args:
        min_val: Minimum required value, inclusive.
        max_val: Maximim required value, inclusive.
        integer: True to get an int type, False for float.
        signed: True for a signed int, False for unsigned; ignored for float. 
            Defaults to None to determine automatically based on ``min_val``.
    
    Returns:
        The dtype fitting the range specifications.
    
    Raise:
        TypeError if a dtype with the appropriate range cannot be found.
    """
    if signed is None:
        # determine automatically based on whether min val is neg
        signed = min_val < 0
    if integer:
        type_group = "int" if signed else "uint"
        fn_info = np.iinfo
    else:
        type_group = "float"
        fn_info = np.finfo
    types = _DTYPES[type_group]
    for dtype in types:
        if fn_info(dtype).min <= min_val and fn_info(dtype).max >= max_val:
            return dtype
    raise TypeError(
        "unable to find numerical type (integer {}, signed {}) containing "
        "range {} through {}".format(integer, signed, min_val, max_val))

if __name__ == "__main__":
    print(insert_before_ext("test.name01.jpg", "_modifier"))
    a = np.arange(2 * 3).reshape(2, 3).astype(np.float)
    a = np.ones(3 * 4).reshape(3, 4).astype(np.float)
    a *= 100
    a[0, 0] = 0
    a[1, 1] = 50
    print("a:\n{}".format(a))
    a_norm = normalize(np.copy(a), 1, 2)
    print("a_norm without background:\n{}".format(a_norm))
    a_norm = normalize(np.copy(a), 1, 2, background=0)
    print("a_norm with background:\n{}".format(a_norm))
