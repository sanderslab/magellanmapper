#!/bin/bash
# Library functions shared within Clrbrain
# Author: David Young, 2017, 2018
"""Shared functions with the Clrbrain package.
"""

import os
import shutil
import numpy as np

from clrbrain import config

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
    """Cast a value as an integer, returning the value instead if any error.
    
    Args:
        val: Value to cast.
    
    Returns:
        Value casted to int, or the value if any error occurs during casting.
    """
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

def backup_file(path, modifier=""):
    """Backup a file to the next given available path with an index number 
    before the extension.
    
    The backed up path will be in the format 
    ``path-before-ext[modifier](i).ext``, where ``[modifier]`` is an optional 
    additional string, and ``i`` is the index number, which will be 
    incremented to avoid overwriting an existing file.
    
    Args:
        path: Path of file to backup.
        modifier: Modifier string to place before the index number.
    """
    i = 1
    ext = os.path.splitext(path)
    backup_path = None
    while True:
        suffix = "{}({})".format(modifier, i)
        backup_path = insert_before_ext(path, suffix)
        if not os.path.exists(backup_path):
            shutil.move(path, backup_path)
            print("Backed up {} to {}".format(path, backup_path))
            break
        i += 1

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
