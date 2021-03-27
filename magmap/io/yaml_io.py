# YAML Input/Output
# Author: David Young, 2020
"""YAML file format input/output."""

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
