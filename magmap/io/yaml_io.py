# YAML Input/Output
# Author: David Young, 2020
"""YAML file format input/output."""

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Union

import yaml

from magmap.io import libmag

if TYPE_CHECKING:
    import pathlib


def _filter_dict(d: Dict, fn_parse_val: Callable[[Any], Any]) -> Dict:
    """Recursively filter keys and values within nested dictionaries
    
    Args:
        d: Dictionary to filter.
        fn_parse_val: Function to apply to each value. Should call
            this parent function if deep recursion is desired.

    Returns:
        Filtered dictionary.

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


def load_yaml(
        path: Union[str, "pathlib.Path"],
        enums: Optional[Dict[str, Enum]] = None) -> List[Dict]:
    """Load a YAML file with support for multiple documents and Enums.

    Args:
        path: Path to YAML file.
        enums: Dictionary mapping Enum names to Enum classes; defaults
            to None. If a key or value in the YAML file matches an Enum name
            followed by a period, the corresponding Enum will be used.

    Returns:
        Sequence of parsed dictionaries for each document within
        a YAML file.
    
    Raises:
        FileNotFoundError: if ``path`` could not be found or loaded.

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
    
    try:
        with open(path) as yaml_file:
            # load all documents into a generator
            docs = yaml.load_all(yaml_file, Loader=yaml.FullLoader)
            data = []
            for doc in docs:
                if not doc:
                    # skip empty document
                    continue
                if enums:
                    doc = _filter_dict(doc, parse_enum_val)
                data.append(doc)
    except (FileNotFoundError, UnicodeDecodeError) as e:
        raise FileNotFoundError(e)
    return data


def save_yaml(
        path: Union[str, "pathlib.Path"], data: Dict,
        use_primitives: bool = False, convert_enums: bool = False) -> Dict:
    """Save a dictionary to YAML file format.
    
    Args:
        path: Output path.
        data: Dictionary to output.
        use_primitives: True to replace Numpy data types with Python primitives;
            defaults to False.
        convert_enums: True to convert keys and vals that are Enums to strings;
            defaults to False.

    Returns:
        ``data`` with any conversions.

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
    
    def convert_enum(val):
        # convert Enums to class.name strings
        if isinstance(val, Enum):
            return f"{val.__class__.__name__}.{val.name}"
        return val
    
    if use_primitives:
        # replace Numpy arrays and types with Python primitives
        data = _filter_dict(data, convert_numpy_val)
    
    if convert_enums:
        data = _filter_dict(data, convert_enum)
    
    with open(path, "w") as yaml_file:
        # save to YAML format
        yaml.dump(data, yaml_file)
    print("Saved data to:", path)
    return data
