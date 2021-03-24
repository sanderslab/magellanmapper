# YAML Input/Output
# Author: David Young, 2020
"""YAML file format input/output."""

import numpy as np
import yaml

from magmap.io import libmag


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
            val = parse_enum(val)
        elif libmag.is_seq(val):
            val = [parse_enum_val(v) for v in val]
        elif isinstance(val, str):
            val_split = val.split(".")
            if len(val_split) > 1 and val_split[0] in enums:
                # replace with the corresponding Enum class
                val = enums[val_split[0]][val_split[1]]
        return val

    def parse_enum(d):
        # recursively parse Enum keys and values within nested dictionaries
        out = {}
        for key, val in d.items():
            if isinstance(val, dict):
                # recursively parse nested dictionaries
                val = parse_enum(val)
            elif libmag.is_seq(val):
                # parse vals within lists
                val = [parse_enum_val(v) for v in val]
            else:
                # parse a single val
                val = parse_enum_val(val)
            key = parse_enum_val(key)
            out[key] = val
        return out

    with open(path) as yaml_file:
        # load all documents into a generator
        docs = yaml.load_all(yaml_file, Loader=yaml.FullLoader)
        data = []
        for doc in docs:
            if enums:
                doc = parse_enum(doc)
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
    def strip_numpy_val(val):
        # recursively convert Numpy data types to primitives
        if isinstance(val, dict):
            val = strip_numpy(val)
        elif libmag.is_seq(val):
            val = [strip_numpy_val(v) for v in val]
        else:
            try:
                val = val.item()
            except AttributeError:
                pass
        return val
    
    def strip_numpy(d):
        # recursively convert Numpy arrays to lists
        out = {}
        for key, val in d.items():
            if isinstance(val, dict):
                # recursively parse nested dictionaries
                val = strip_numpy(val)
            elif isinstance(val, np.ndarray):
                # parse vals within lists
                val = [strip_numpy_val(v) for v in val]
                print("converted", key, val)
            out[key] = strip_numpy_val(val)
            print("adding", key, val, type(val))
        return out
    
    if use_primitives:
        # replace Numpy arrays and types with Python primitives
        data = strip_numpy(data)
    
    with open(path, "w") as yaml_file:
        # save to YAML format
        yaml.dump(data, yaml_file)
    print("Saved data to:", path)
    return data
