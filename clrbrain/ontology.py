#!/bin/bash
# Anatomical ontology management
# Author: David Young, 2019
"""Handle ontology lookup.
"""

from collections import OrderedDict
import json

from clrbrain import config

NODE = "node"
PARENT_IDS = "parent_ids"

def load_labels_ref(path):
    labels_ref = None
    with open(path, "r") as f:
        labels_ref = json.load(f)
        #pprint(labels_ref)
    return labels_ref

def get_node(nested_dict, key, value, key_children):
    """Get a node from a nested dictionary by iterating through all 
    dictionaries until the specified value is found.
    
    Args:
        nested_dict: A dictionary that contains a list of dictionaries in
            the key_children entry.
        key: Key to check for the value.
        value: Value to find, assumed to be unique for the given key.
        key_children: Name of the children key, which contains a list of 
            further dictionaries but can be empty.
    
    Returns:
        The node matching the key-value pair, or None if not found.
    """
    try:
        #print("checking for key {}...".format(key), end="")
        found_val = nested_dict[key]
        #print("found {}".format(found_val))
        if found_val == value:
            return nested_dict
        children = nested_dict[key_children]
        for child in children:
            result = get_node(child, key, value, key_children)
            if result is not None:
                return result
    except KeyError as e:
        print(e)
    return None

def create_aba_reverse_lookup(labels_ref):
    """Create a reverse lookup dictionary for Allen Brain Atlas style
    ontology files.
    
    Args:
        labels_ref: The ontology file as a parsed JSON dictionary.
    
    Returns:
        Reverse lookup dictionary as output by 
        :func:`ontology.create_reverse_lookup`.
    """
    return create_reverse_lookup(
        labels_ref["msg"][0], config.ABAKeys.ABA_ID.value, 
        config.ABAKeys.CHILDREN.value)

def create_reverse_lookup(nested_dict, key, key_children, id_dict=OrderedDict(), 
                          parent_list=None):
    """Create a reveres lookup dictionary with the values of the original 
    dictionary as the keys of the new dictionary.
    
    Each value of the new dictionary is another dictionary that contains 
    "node", the dictionary with the given key-value pair, and "parent_ids", 
    a list of all the parents of the given node. This entry can be used to 
    track all superceding dictionaries, and the node can be used to find 
    all its children.
    
    Args:
        nested_dict: A dictionary that contains a list of dictionaries in
            the key_children entry.
        key: Key that contains the values to use as keys in the new dictionary. 
            The values of this key should be unique throughout the entire 
            nested_dict and thus serve as IDs.
        key_children: Name of the children key, which contains a list of 
            further dictionaries but can be empty.
        id_dict: The output dictionary as an OrderedDict to preserve key 
            order (though not hierarchical structure) so that children 
            will come after their parents; if None is given, an empty 
            dictionary will be created.
        parent_list: List of values for the given key in all parent 
            dictionaries.
    
    Returns:
        A dictionary with the original values as the keys, which each map 
        to another dictionary containing an entry with the dictionary 
        holding the given value and another entry with a list of all parent 
        dictionary values for the given key.
    """
    value = nested_dict[key]
    sub_dict = {NODE: nested_dict}
    if parent_list is not None:
        sub_dict[PARENT_IDS] = parent_list
    id_dict[value] = sub_dict
    try:
        children = nested_dict[key_children]
        parent_list = [] if parent_list is None else list(parent_list)
        parent_list.append(value)
        for child in children:
            #print("parents: {}".format(parent_list))
            create_reverse_lookup(
                child, key, key_children, id_dict, parent_list)
    except KeyError as e:
        print(e)
    return id_dict

def _get_children(labels_ref_lookup, label_id, children_all=[]):
    """Recursively get the children of a given non-negative atlas ID.
    
    Used as a helper function to :func:``get_children_from_id``.
    
    Args:
        labels_ref_lookup: The labels reference lookup, assumed to be 
            generated by :func:`create_reverse_lookup` to look up by ID.
        label_id: ID of the label to find, assumed to be >= 0 since 
            IDs in ``labels_ref_lookup`` are generally non-negative.
        children_all: List of all children of this ID, used recursively; 
            defaults to an empty list. To include the ID itself, pass in a 
            list with this ID alone.
    
    Returns:
        A list of all children of the given ID, in order from highest 
        (numerically lowest) level to lowest.
    """
    label = labels_ref_lookup.get(label_id)
    if label:
        # recursively gather the children of the label
        children = label[NODE][config.ABAKeys.CHILDREN.value]
        for child in children:
            child_id = child[config.ABAKeys.ABA_ID.value]
            #print("child_id: {}".format(child_id))
            children_all.append(child_id)
            _get_children(labels_ref_lookup, child_id, children_all)
    return children_all

def get_children_from_id(labels_ref_lookup, label_id, incl_parent=True, 
                         both_sides=False):
    """Get the children of a given atlas ID.
    
    Args:
        labels_ref_lookup: The labels reference lookup, assumed to be 
            generated by :func:`create_reverse_lookup` to look up by ID.
        label_id: ID of the label to find, which can be negative.
        incl_parent: True to include ``label_id`` itself in the list of 
            children; defaults to True.
        both_sides: True to include both sides, ie positive and negative 
            values of each ID. Defaults to False.
    
    Returns:
        A list of all children of the given ID, in order from highest 
        (numerically lowest) level to lowest.
    """
    id_abs = abs(label_id)
    children_all = [id_abs] if incl_parent else []
    region_ids = _get_children(labels_ref_lookup, id_abs, children_all)
    if both_sides:
        region_ids.extend([-1 * n for n in region_ids])
    elif label_id < 0:
        region_ids = [-1 * n for n in region_ids]
    #print("region IDs: {}".format(region_ids))
    return region_ids
