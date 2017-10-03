#!/bin/bash
# Library functions shared within Clrbrain
# Author: David Young, 2017
"""Shared functions with the Clrbrain package.
"""

def swap_elements(arr, axis0, axis1, offset=0):
    """Swap elements within an array.
    
    Params:
        arr: Array in which to swap elements.
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
