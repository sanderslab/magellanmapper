# YAML Input/Output
# Author: David Young, 2020
"""YAML file format input/output."""

import numpy as np
import yaml

from magmap.io import libmag


def _filter_dict(d, fn_parse_val):
    """Recursively filter keys and values within nested dictionaries
    
    Args:
        d (dict): Dictionary to filter.
        fn_parse_val (func): Function to apply to each value. Should call
            this parent function if deep recursion is desired.

    Returns:
        dict: Filtered dictionary.

    """
    out = {}
    for key, val in d.items():
        if isinstance(val, dict):
            # recursively filter nested dictionaries
            val = fn_parse_val(val)
        elif libmag.is_seq(val):
            # filter each val within list
            val = [fn_parse_val(v) for v in val]
        else:
            # filter a single val
            val = fn_parse_val(val)
        # filter key
        key = fn_parse_val(key)
        out[key] = val
    return out


def load_yaml(path, enums=None):
    """Load a YAML file with support for multiple documents and Enums.

    Args:
        path (str): Path to YAML file.
        enums (dict): Dictionary mapping Enum names to Enum classes; defaults
            to None. If a key or value in the YAML file matches an Enum name
            followed by a period, the corresponding Enum will be used.

    Returns:
        List[dict]: Sequence of parsed dictionaries for each document within
        a YAML file.

    """
    def parse_enum_val(val):
        # recursively parse Enum values
        if isinstance(val, dict):
            val = _filter_dict(val, parse_enum_val)
        elif libmag.is_seq(val):
            val = [parse_enum_val(v) for v in val]
        elif isinstance(val, str):
            val_split = val.split(".")
            if len(val_split) > 1 and val_split[0] in enums:
                # replace with the corresponding Enum class
                val = enums[val_split[0]][val_split[1]]
        return val

    with open(path) as yaml_file:
        # load all documents into a generator
        docs = yaml.load_all(yaml_file, Loader=yaml.FullLoader)
        data = []
        for doc in docs:
            if enums:
                doc = _filter_dict(doc, parse_enum_val)
            data.append(doc)
    return data


def save_yaml(path, data, use_primitives=False):
    """Save a dictionary to YAML file format.
    
    Args:
        path (str): Output path.
        data (dict): Dictionary to output.
        use_primitives (bool): True to replace Numpy data types to primitives;
            defaults to False.

    Returns:
        dict: ``data`` with Numpy arrays and data types converted to Python
        primitives if ``use_primitives`` is true, otherwise ``data``
        unchanged.

    """
    def convert_numpy_val(val):
        # recursively convert Numpy data types to primitives
        if isinstance(val, dict):
            val = _filter_dict(val, convert_numpy_val)
        elif libmag.is_seq(val):
            # also replaces any tuples with lists, avoiding tuple flags in
            # the output file for simplicity
            val = [convert_numpy_val(v) for v in val]
        else:
            try:
                val = val.item()
            except AttributeError:
                pass
        return val
    
    if use_primitives:
        # replace Numpy arrays and types with Python primitives
        data = _filter_dict(data, convert_numpy_val)
    
    with open(path, "w") as yaml_file:
        # save to YAML format
        yaml.dump(data, yaml_file)
    print("Saved data to:", path)
    return data
