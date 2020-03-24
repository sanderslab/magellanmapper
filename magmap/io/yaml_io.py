# YAML Input/Output
# Author: David Young, 2020
"""YAML file format input/output."""

import yaml


def load_yaml(path, enums=None):
    """Load a YAML file with support for multiple documents and Enums.

    Args:
        path (str): Path to YAML file.
        enums (dict): Dictionary mapping Enum names to Enum classes; defaults
            to None. If a key in the YAML file matches an Enum name followed
            by a period, the corresponding Enum class will be used.

    Returns:

    """
    def parse_enum(d):
        # recursively parse Enum keys within nested dictionaries
        out = {}
        for key, val in d.items():
            key_split = key.split(".")
            if isinstance(val, dict):
                # parse nested dictionaries
                val = parse_enum(val)
            if len(key_split) > 1 and key_split[0] in enums:
                # replace the entry's key with the corresponding Enum class
                print(enums[key_split[0]])
                out[enums[key_split[0]][key_split[1]]] = val
            else:
                # add the key as-is
                out[key] = val
        return out

    with open(path) as yaml_file:
        # load all documents into a generator
        docs = yaml.load_all(yaml_file, Loader=yaml.FullLoader)
        data = []
        for doc in docs:
            print(doc)
            if enums:
                doc = parse_enum(doc)
            data.append(doc)
    return data
