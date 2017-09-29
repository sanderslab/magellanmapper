#!/bin/bash
# Library functions shared within Clrbrain
# Author: David Young, 2017
"""Shared functions with the Clrbrain package.
"""

def swap_elements(arr, axis0, axis1, offset=0):
    axis0 += offset
    axis1 += offset
    check_tuple = isinstance(arr, tuple)
    if check_tuple:
        arr = list(arr)
    arr[axis0], arr[axis1] = arr[axis1], arr[axis0]
    if check_tuple:
        arr = tuple(arr)
    return arr
