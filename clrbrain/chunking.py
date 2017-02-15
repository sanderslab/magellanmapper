#!/bin/bash
# Chunking image stacks
# Author: David Young, 2017

import numpy as np

from clrbrain import detector

max_pixels = 500
overlap_base = 5

def _num_units(size):
    num = size // max_pixels
    if size % max_pixels > 0:
        num += 1
    return num

def _len_side(size, overlap, i):
    start = i * max_pixels
    end = start + max_pixels + overlap
    if end > size:
        end = size
    return (start, end)

def stack_splitter(roi):
    size = roi.shape
    overlap = overlap_base * detector.scaling_factor
    num_x = _num_units(size[2])
    num_y = _num_units(size[1])
    sub_rois = np.zeros((num_x, num_y), dtype=object)
    for j in range(num_y):
        for i in range(num_x):
            x_bounds = _len_side(size[2], overlap, i)
            y_bounds = _len_side(size[1], overlap, j)
            sub_rois[i, j] = roi[:, slice(*y_bounds), slice(*x_bounds)]
    return sub_rois, overlap

def _get_sub_roi(sub_rois, overlap, i, j):
    sub_roi = sub_rois[i, j]
    size = sub_roi.shape
    edge_x = size[2]
    edge_y = size[1]
    # remove overlap if not at last sub_roi or row or column
    if i != sub_rois.shape[0] - 1:
        edge_x -= overlap
    if j != sub_rois.shape[1] - 1:
        edge_y -= overlap
    return sub_roi[:, :edge_y, :edge_x]

def merge_split_stack(sub_rois, overlap):
    size = sub_rois.shape
    merged = None
    merged_x = None
    for j in range(size[1]):
        merged_x = None
        for i in range(size[0]):
            sub_roi = _get_sub_roi(sub_rois, overlap, i, j)
            if merged_x is None:
                merged_x = sub_roi
            else:
                merged_x = np.concatenate((merged_x, sub_roi), axis=2)
        if merged is None:
            merged = merged_x
        else:
            merged = np.concatenate((merged, merged_x), axis=1)
    return merged

def remove_duplicate_blobs(blobs, region):
    #blobs_tuple = [tuple(blob) for blob in blobs]
    #blobs_unique = np.unique(blobs_tuple)
    # workaround while awaiting https://github.com/numpy/numpy/pull/7742
    # to become a reality, presumably in Numpy 1.13
    blobs_region = blobs[:, region]
    blobs_contig = np.ascontiguousarray(blobs_region)
    blobs_type = np.dtype((np.void, blobs_region.dtype.itemsize * blobs_region.shape[1]))
    blobs_contig = blobs_contig.view(blobs_type)
    _, unique_indices = np.unique(blobs_contig, return_index=True)
    return blobs[unique_indices]

if __name__ == "__main__":
    print("Starting chunking...")
    #roi = np.arange(1920 * 1920 * 500)
    max_pixels = 2
    overlap_base = 1
    roi = np.arange(2 * 4 * 4)
    roi = roi.reshape((2, 4, 4))
    print("roi: {}".format(roi))
    sub_rois, overlap = stack_splitter(roi)
    print("sub_rois shape: {}".format(sub_rois.shape))
    print("sub_rois: {}".format(sub_rois))
    print("overlap: {}".format(overlap))
    merged = merge_split_stack(sub_rois, overlap)
    print("merged: {}".format(merged))
    print("merged shape: {}".format(merged.shape))
    print("test roi == merged: {}".format(np.all(roi == merged)))
    #blobs = np.random.randint(0, high=1920, size=(10, 3))
    blobs = np.array([[1, 3, 4, 2.2342], [1, 8, 5, 3.13452], [1, 3, 4, 5.1234]])
    print("sample blobs: {}".format(blobs))
    end = 3
    blobs_unique = remove_duplicate_blobs(blobs, slice(0, end))
    print("blobs_unique through first {} elements: {}".format(end, blobs_unique))
    