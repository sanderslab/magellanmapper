#!/bin/bash
# Regional volume and density management
# Author: David Young, 2018
"""Measure volumes and densities by regions.
"""

import multiprocessing as mp
from time import time

import numpy as np
from skimage import measure

from clrbrain import plot_3d

class LabelToEdge:
    """Convert a label to an edge with class methods as an encapsulated 
    way to use in multiprocessing without requirement for global variable.
    
    Attributes:
        labels_img_np: Integery labels images as a Numpy array.
    """
    labels_img_np = None
    
    @classmethod
    def set_labels_img_np(cls, val):
        """Set the labels image.
        
        Args:
            val: Labels image to set as class attribute.
        """
        cls.labels_img_np = val
    
    @classmethod
    def find_label_edge(cls, label_id):
        """Convert a label into just its border.
        
        Args:
            label_id: Integer of the label to extract from 
                :attr:``labels_img_np``.
        
        Returns:
            Tuple of the given label ID; list of slices defining the 
            location of the ROI where the edges can be found; and the 
            ROI as a volume mask defining where the edges exist.
        """
        print("getting edge for {}".format(label_id))
        slices = None
        borders = None
        
        # get mask of label to get bounding box
        label_mask = cls.labels_img_np == label_id
        props = measure.regionprops(label_mask.astype(np.int))
        if len(props) > 0 and props[0].bbox is not None:
            _, slices = plot_3d.get_bbox_region(props[0].bbox)
            
            # work on a view of the region for efficiency, obtaining borders 
            # as eroded region and writing into new array
            region = cls.labels_img_np[tuple(slices)]
            label_mask_region = region == label_id
            borders = plot_3d.perimeter_nd(label_mask_region)
        return label_id, slices, borders

def make_labels_edge(labels_img_np):
    """Convert labels image into label borders image.
    
    The atlas is assumed to be a sample (eg microscopy) image on which 
    an edge-detection filter will be applied. 
    
    Args:
        labels_img_np: Image as a Numpy array, assumed to be an 
            annotated image whose edges will be found by obtaining 
            the borders of all annotations.
    
    Returns:
        Binary image array the same shape as ``labels_img_np`` with labels 
        reduced to their corresponding borders.
    """
    start_time = time()
    labels_edge = np.zeros_like(labels_img_np)
    label_ids = np.unique(labels_img_np)
    
    # use a class to set and process the label without having to 
    # reference the labels image as a global variable
    label_to_edge = LabelToEdge()
    label_to_edge.set_labels_img_np(labels_img_np)
    
    pool = mp.Pool()
    pool_results = []
    for label_id in label_ids:
        pool_results.append(pool.apply_async(
            label_to_edge.find_label_edge, 
            args=(label_id, )))
    metrics = []
    for result in pool_results:
        label_id, slices, borders = result.get()
        if slices is not None:
            borders_region = labels_edge[tuple(slices)]
            borders_region[borders] = label_id
    pool.close()
    pool.join()
    
    print("time elapsed to make labels edge:", time() - start_time)
    
    return labels_edge
