# Library functions shared within MagellanMapper
# Author: David Young, 2017, 2023
"""Shared functions with the MagellanMapper package.
"""

from functools import lru_cache
import os
import pathlib
import shutil
import sys
from typing import Any, Callable, Generator, List, Optional, Sequence, Tuple, \
    Union
import warnings

if sys.version_info >= (3, 8):
    # included with Python >= 3.8
    from importlib import metadata
else:
    import importlib_metadata as metadata

import numpy as np
from skimage import exposure

from magmap.settings import config

_logger = config.logger.getChild(__name__)


# file types that are associated with other types
_FILE_TYPE_GROUPS = {
    "obj": "mtl",
    "mhd": "raw",
}

# Numpy numerical dtypes with various ranges
_DTYPES = {
    "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
    "int": [np.int8, np.int16, np.int32, np.int64],
    "float": [np.float16, np.float32, np.float64]
}

# the start of extensions that may have multiple periods
_EXTENSIONS_MULTIPLE = (".tar", ".nii")


def swap_elements(arr, axis0, axis1, offset=0):
    """Swap elements within an list or tuple.
    
    Args:
        arr: List or tuple in which to swap elements.
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


def transpose_1d(arr, plane):
    transposed = arr
    if plane == config.PLANE[1]:
        # make y the "z" axis
        transposed = swap_elements(arr, 0, 1)
    elif plane == config.PLANE[2]:
        # yz plane
        transposed = swap_elements(arr, 0, 2)
        transposed = swap_elements(arr, 1, 2)
    return transposed


def transpose_1d_rev(arr, plane):
    transposed = arr
    if plane == config.PLANE[1]:
        # make y the "z" axis
        transposed = swap_elements(arr, 1, 0)
    elif plane == config.PLANE[2]:
        # yz plane
        transposed = swap_elements(arr, 2, 1)
        transposed = swap_elements(arr, 2, 0)
    return transposed


def roll_elements(arr, shift, axis=None):
    """Roll elements in a tuple safe manner.
    
    Essentially calls Numpy.roll, but checks for tuple beforehand and converts 
    it to a Numpy array beforehand and back to a new tuple afterward.
    
    Args:
        arr: Array, which can be a tuple, list, or Numpy array. 
    
    Returns:
        The array with elements rolled. If arr is a tuple, the returned value 
            will be a new tuple. If arr is a Numpy array, a view of the array 
            will be turned.
    """
    #print("orig: {}".format(arr))
    check_tuple = isinstance(arr, tuple)
    if check_tuple:
        arr = np.array(arr)
    arr = np.roll(arr, shift, axis)
    if check_tuple:
        arr = tuple(arr)
    #print("after moving: {}".format(arr))
    return arr


def pad_seq(
        seq: Sequence[Any], n: int, pad: Optional[Any] = None) -> Sequence[Any]:
    """Pad a sequence with a given value or truncate to fit a given length.
    
    Args:
        seq: Sequence to fill in-place.
        n: Target length.
        pad: Value with which to fill; defaults to None.
            If a sequence, missing values in ``seq`` will be filled with
            corresponding values in ``pad``.
    
    Returns:
        A truncated view of ``seq`` if the sequence is longer than ``n``
        or ``seq`` with ``pad`` appended to reach a length of ``n``. ``seq``
        is modified in-place if it is extended and not a NumPy array.
    
    """
    len_seq = len(seq)
    if len_seq >= n:
        # truncate if seq is longer than n is
        seq = seq[:n]
    
    else:
        if isinstance(seq, np.ndarray):
            # convert to list if ndarray to allow mixing with None values
            seq = seq.tolist()
        elif isinstance(seq, tuple):
            # convert to list to allow adding values
            seq = list(seq)

        # pad with the given value
        if is_seq(pad):
            if len(pad) > len_seq:
                # fill starting with corresponding first missing value
                seq += pad[len_seq:]
            # recursively call to truncate if seq now exceeds n or to further
            # fill with last pad value
            seq = pad_seq(seq, n, pad[-1])
        else:
            # fill with pad value repeats
            seq += [pad] * (n - len_seq)
    
    return seq


def replace_seq(
        seq: Sequence[Any], replacement: Sequence[Any]) -> np.ndarray:
    """Replace elements in one sequence with those from another.
    
    Sequences will be converted to NumPy arrays to support multiple dimensions.
    
    Args:
        seq: Sequence to be replaced.
        replacement: Sequence to replace in ``seq``. Must be of the same
            shape and type as those of ``seq``.

    Returns:
        ``seq`` as a new NumPy array with as many values ass possible replaced
        by ``replacement``.

    """
    # convert to np to support multi-dimensional arrays
    seq_np = np.array(seq)
    tgt_np = np.array(replacement)
    
    # truncate to smallest shape in all dimension
    shape = np.minimum(seq_np.shape, tgt_np.shape)
    slices = []
    for s in shape:
        slices.append(slice(s))
    slices = tuple(slices)
    seq_np[slices] = tgt_np[slices]
    
    return seq_np


def combine_arrs(arrs, filter_none=True, fn=None, **kwargs):
    """Combine arrays with array filtering.
    
    Args:
        arrs (Sequence): Sequence of arrays.
        filter_none (bool): True to filter out ``None``s in ``arrs``; defaults
            to True.
        fn (func): Function to combine filtered arrays; defaults to None to
            use :meth:`numpy.concatenate`.
        **kwargs: Additional arguments to ``fn``.

    Returns:
        Any: Output of ``fn`` applied to the filtered ``arrs``, or None if
        the filtered ``arrs`` is None or empty. If the filtered ``arrs`` has
        only one element, return this element as-is, without combining.

    """
    if arrs is None: return None
    if fn is None:
        fn = np.concatenate
    if filter_none:
        # filter out Nones
        arrs = [a for a in arrs if a is not None]
    len_arrs = len(arrs)
    if len_arrs < 1:
        return None
    elif len_arrs == 1:
        # return as-is, without combining
        return arrs[0]
    else:
        # combine with given function
        return fn(arrs, **kwargs)


def flatten(vals: Sequence[Any]) -> Generator[Any, None, None]:
    """Flatten an arbitrarily nested sequence.
    
    Args:
        vals: Sequence of values with arbitrary levels of nesting.

    Yields:
        ``vals`` flattened to a single sequence.

    """
    for val in vals:
        if is_seq(val):
            yield from flatten(val)
        else:
            yield val


def insert_before_ext(
        name: Union[str, pathlib.Path], insert: str, sep: str = "") -> str:
    """Merge two paths by splicing in ``insert`` just before the extention 
    in ``name``.
    
    Args:
        name: Path; if no dot is present in the basename, simply
            merge the string components.
        insert: String to insert before the extension in ``name``.
        sep: Separator between ``name`` and ``insert``; defaults to an
           empty string.
    
    Returns:
        str: ``name`` with ``insert`` inserted just before the extension.
    
    See Also:
        :func:``combine_paths`` to use the extension from ``insert``.
    """
    name = str(name)
    if os.path.basename(name).find(".") == -1:
        # no extension in basename, so simply combine
        return name + sep + insert
    return "{0}{2}{3}.{1}".format(*name.rsplit(".", 1), sep, insert)


def splitext(path):
    """Split a path at its extension in a way that supports extensions 
    with multiple periods as identified in :const:``_EXTENSIONS_MULTIPLE``.
    
    Args:
        path: Path to split.
    
    Returns:
        Tuple of path prior to extension and the extension, including 
        leading period. If an extension start is not found in 
        :const:``_EXTENSIONS_MULTIPLE``, the path will simply be split 
        by :meth:``os.path.splitext``.
    """
    i = -1
    for ext in _EXTENSIONS_MULTIPLE:
        i = path.rfind(ext)
        if i != -1: break
    if i == -1:
        path_split = os.path.splitext(path)
    else:
        path_split = (path[:i], path[i:])
    return path_split


def match_ext(path, path_to_match):
    """Match extensions for two paths.
    
    Args:
        path: Path with extension that will be kept; will be ignored if only 
            an extension with dot.
        path_to_match: Path whose extension will be replaced with that of 
            ``path``.
    
    Returns:
        ``path_to_match`` with extension replaced by that of ``path`` if 
        it has an extension; otherwise, ``path_to_match`` is simply returned.
    """
    path_split = splitext(path)
    if path_split[1] and not path_to_match.endswith(path_split[1]):
        path_to_match = os.path.splitext(path_to_match)[0] + path_split[1]
    return path_to_match


def get_filename_without_ext(path: str) -> str:
    """Wrapper to :meth:`splitext` for getting only the filename.
    
    Args:
        path: Full path.
    
    Returns:
        Filename alone without extension; simply returns the filename if 
        no extension exists.
    """
    name = os.path.basename(path)
    name_split = splitext(name)
    if len(name_split) > 1: return name_split[0]
    return name


def combine_paths(
        base_path: str, suffix: str, sep: str = "_", ext: str = None,
        check_dir: bool = False, keep_ext: bool = False):
    """Merge two paths by appending ``suffix``, replacing the extension
    in ``base_path``.
    
    Args:
        base_path: Path whose dot-extension will be replaced by
            ``suffix``. If None, ``suffix`` will be returned. If a directory
            as indicated by a trailing file separator, will simply be joined
            to ``suffix``.
        suffix: Replacement including new extension.
        sep: Separator between ``base_path`` and ``suffix``.
        ext: Extension to add or substitute; defaults to None to use
            the extension in ``suffix``.
        check_dir: True to check if ``base_path`` is an existing directory,
            in which case it is simply joined to ``suffix``; defaults to False.
        keep_ext: True to keep the `base_path` extension; defaults to False.
    
    Returns:
        Merged path.
    
    See Also:
        :func:`insert_before_ext` to splice in ``suffix`` instead.
    """
    if not base_path: return suffix
    if not os.path.basename(base_path) or check_dir and os.path.isdir(
            base_path):
        # dir if trailing file separator or flagged to check for existing dir;
        # unnecessary to split out ext and adding sep
        path = os.path.join(base_path, suffix)
    else:
        # remove the extension if flagged and combine with separator and suffix
        path = base_path if keep_ext else splitext(base_path)[0]
        path = path + sep + suffix
    if ext:
        # replace extension from suffix with given ext
        path = f"{splitext(path)[0]}.{ext}"
    return path


def make_out_path(
        base_path: Optional[str] = None, prefix: Optional[str] = None,
        suffix: Optional[str] = None, combine_prefix: bool = False) -> str:
    """Make output path based on prefix and suffix settings.
    
    The base path is typically the default path, while the prefix and suffix
    are given by the user. The prefix will override all other settings unless
    flagged to combine it with them.
    
    Args:
        base_path: Base path from which to construct the output path.
            Defaults to None to use :attr:`magmap.settings.config.filename`
            if no prefix is given.
        prefix: Path that normally overrides ``base_path`` and ``suffix``.
            Defaults to None, which causes
            :attr:`magmap.settings.config.prefix_out` to be used if available,
            falling back to :attr:`magmap.settings.config.prefix`.
            Set to "" to ignore. If given with a trailing file separator,
            the prefix will be combined with ``base_path`` and ``suffix``
            even if ``combine_prefix`` is False.
        suffix: String to append to end of path just before the
            extension; defaults to None to use
            :attr:`magmap.settings.config.suffix`.
        combine_prefix: True to combine the prefix with the basename of the
            base path and with the suffix; defaults to False. Often used when
            multiple files will be output.

    Returns:
        Output path.

    """
    # prioritize any prefix available in the order: 1) prefix parameter,
    # 2) prefix-out from config, 3) prefix from config
    out_path = config.prefix_out if prefix is None else prefix
    if out_path is None:
        out_path = config.prefix
    
    suffix = config.suffix if suffix is None else suffix
    if out_path:
        # prefix used as-is unless set to combine or explicitly given as a dir
        if combine_prefix or not os.path.basename(out_path):
            # combine prefix to base name of base path and suffix
            if base_path:
                out_path += os.path.basename(base_path)
            if suffix:
                out_path = insert_before_ext(out_path, suffix)
    else:
        # construct from base path (or filename) and suffix if no prefix
        out_path = base_path if base_path else config.filename
        if suffix:
            out_path = insert_before_ext(out_path, suffix)
    return out_path


def remove_file(path):
    """Remove a file with error catching.

    Exceptions from files that are not found or are directories will be
    caught and displayed.

    Args:
        path (str): Path of file to remove.

    Returns:
        bool: True if the file was successfully deleted.

    """
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        print("File not found to remove:", path)
    except IsADirectoryError:
        print("Path is a directoy, will not be removed:", path)
    return False


def normalize(
        array: Sequence[Union[int, float]], minimum: Union[int, float],
        maximum: Union[int, float],
        in_range: Union[str, Tuple[
            Union[int, float], Union[int, float]]] = "image"
) -> Union[Sequence[Union[int, float]], np.ndarray]:
    """Normalizes an array to fall within the given min and max.
    
    Args:
        array: Array to normalize.
        minimum: Minimum value for the array.
        maximum: Maximum value for the array. Assumed to be greater than
            ``min``.
        in_range: Range within ``array`` to rescale. Defaults to "image",
            which uses the range from ``array`` itself.
    
    Returns:
        The normalized array.
    
    """
    if len(array) <= 0:
        return array
    
    if not isinstance(array, np.ndarray):
        # rescale_intensity requires Numpy arrays
        array = np.array(array)
    
    if isinstance(array.flat[0], (int, np.integer)) and (
            isinstance(minimum, float) or isinstance(maximum, float)):
        # convert to float if min/max are float but array is not
        array = 1.0 * array
    
    # rescale the array intensity range
    array = exposure.rescale_intensity(
        array, out_range=(minimum, maximum), in_range=in_range)
    
    return array


def printv(*s):
    """Print to console only if verbose.
    
    Args:
        s: Variable number of strings to be printed 
            if :attr:``config.verbose`` is true.
    """
    if config.verbose:
        print(*s)


def printcb(s, fn_callback):
    """Print a string to the terminal and call a callback function.
    
    Args:
        s (str): String to print and call in ``fn_callback``.
        fn_callback (func): Callback function taking a string argument.

    """
    print(s)
    if fn_callback:
        fn_callback(s)


def warn(msg, category=UserWarning, stacklevel=2):
    """Print a warning message.
    
    Args:
        msg (str): Message to print.
        category (Exception): Warning category class.
        stacklevel: Warning message level.

    """
    warnings.warn(msg, category, stacklevel=stacklevel)


@lru_cache(10)
def log_once(fn_log: Callable[[str], None], msg: str):
    """Log a message only once.
    
    Args:
        fn_log: Log function.
        msg: Message

    """
    # use cache to ignore repeated calls
    # TODO: check appropriate maxsize
    fn_log(msg)


def series_as_str(series):
    """Get the series as a string for MagellanMapper filenames, ensuring 5 
    characters to allow for a large number of series.
    
    Args:
        series: Series number, to be padded to 5 characters.
    
    Returns:
        Padded series.
    """
    return str(series).zfill(5)


def splice_before(
        base: str, search: str, splice: str, post_splice: str = "") -> str:
    """Splice in a string before a given substring.
    
    Args:
        base: String in which to splice.
        search: Splice before this substring. If not found, ``splice`` is
            simply appended to ``base``.
        splice: Splice in this string.
        post_splice: String to add after the spliced string; defaults to "".
    
    Returns:
        Spliced string.
    
    """
    i = base.rfind(search)
    if i == -1:
        # default to append
        i = len(base)
    return base[:i] + splice + post_splice + base[i:]


def str_to_disp(s):
    """Convert a string to a user-friendly, displayable string by replacing 
    underscores with spaces and trimming outer whitespace.
    
    Args:
        s: String to make displayable.
    
    Returns:
        New, converted string.
    """
    return s.replace("_", " ").strip()


def crop_mid_str(
        vals: Sequence[str], max_chars: int = 10, unique: bool = True
) -> List[str]:
    """Crop out the middle portion of strings while keeping them unique.
    
    The middle section is replaced with "...".
    
    Args:
        vals: Sequence of strings to crop.
        max_chars: Maximum characters to retain. Half the characters will be
            from the start and the remaining characters from the end of the
            string. Defaults to 10. The final output size of each string
            is larger than this size by the number of intervening periods.
        unique: True to ensure that cropped strings are unique. A extra "."
            is added iteratively so the string is unique in the output list.
            Uncropped strings are not checked for uniqueness.

    Returns:
        A list of cropped strings.

    """
    cropped = []
    crop_in = max_chars // 2
    crop_out = max_chars - crop_in
    for val in vals:
        if len(val) > max_chars:
            # crop out middle section
            val_crs = (val[:crop_in], val[-crop_out:])
            sep = "..."
            while True:
                val_cr = sep.join(val_crs)
                if not unique or val_cr not in cropped:
                    break
                # add extra periods until the cropped string is unique
                sep += "."
        else:
            # add as-is if within size, without checking for uniqueness
            val_cr = val
        cropped.append(val_cr)
    return cropped


def make_acronym(
        val: Optional[str], delim: str = " ", ignore: Sequence[str] = None,
        caps: bool = False, num_single: int = 3) -> str:
    """Make an acronymn from a string.
    
    Args:
        val: String to abbreviate.
        delim: Delimiter to split ``val``.
        ignore: Sequence of split strings to ignore. Defaults to None, in
            which case "of" and "the" will be ignored, case-insensitive.
        caps: True to capitalize the abbreviation; defaults to False.
        num_single: Number of characters to keep if ``val`` splits into only
            a single word.

    Returns:
        Abbreviation of ``val``. Return ``val`` unchanged if it is empty.

    """
    if not val:
        # return as-is if empty
        return val
    
    if ignore is None:
        # default splits to ignore
        ignore = ("of", "the")
    
    # split string and remove empty splits or those in ignore, case-insensitive
    split = [s for s in val.split(delim) if s and s.lower() not in ignore]
    
    if len(split) == 1:
        # get custom number of characters if single word
        joined = split[0][:num_single]
    else:
        # get first character of each word
        joined = "".join([s[0] for s in split])
    
    if caps:
        # capitalize whole abbreviation
        joined = joined.upper()
    
    return joined


def get_int(val):
    """Cast a value as an integer or a float if not an integer, if possible.
    
    Args:
        val: Value to cast. If a tuple or list, each entry will be casted 
            recursively.
    
    Returns:
        Value casted to int, falling back to a float, None if ``none``
        (case-insensitive), or the original value if any error occurs during
        casting.
    """
    if isinstance(val, (tuple, list)):
        return [get_int(elt) for elt in val]
    try:
        # prioritize casting to int before float if possible
        return int(val)
    except ValueError:
        try:
            # strings of floating point numbers will give an error when casting 
            # to int, so try casting to float
            return float(val)
        except ValueError:
            if isinstance(val, str) and val.lower() == "none":
                # convert to None if string is "none" (case-insensitive)
                return None
            return val


def is_int(val):
    """Check if a value is an integer, with support for alternate integer
    types such as Numpy integers.
    
    Args:
        val (Any): Value to check.

    Returns:
        bool: True if ``val`` is equal to itself after casting to an int.

    """
    try:
        return val == int(val)
    except ValueError:
        return False


def is_number(val):
    """Check if a value is a number by attempting to cast to ``float``.
    
    Args:
        val: Value to check.
    
    Returns:
        True if the value was successfully cast to ``float``; False otherwise.
    """
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


def is_nan(val):
    """Check if a value can be cast to NaN.
    
    Args:
        val (Any): Value or sequence of values to check.

    Returns:
        Any: True if the value can be cast to ``np.nan``; False if not
            or any error, including strings that do cannot be cast to float;
            or sequence of booleans if ``val`` is a sequence.

    """
    try:
        return np.isnan(np.array(val).astype(float))
    except (ValueError, TypeError):
        return False
        

def format_num(val, dec_digits=1, allow_scinot=True):
    """Format a value to a given number of digits if the value is a number.
    
    Args:
        val (float): The value to format.
        dec_digits (int): Maximum number of decimal place digits to keep;
            defaults to 1.
        allow_scinot (bool): True to allow scientific notation in output;
            defaults to True.
    
    Returns:
        str: A formatted string with the number of decimal place digits
        reduced to ``digits`` if ``val`` is a number, or otherwise simply
        ``val``.
    
    """
    formatted = val
    if is_number(val):
        if isinstance(val, str): val = float(val)
        format_char = "g" if allow_scinot else "f"
        formatted = ("{:." + str(dec_digits) + format_char + "}").format(
            float(val))
    return formatted


def format_bytes(size: Union[int, float, str]) -> str:
    """Format bytes to human-readable unit.
    
    Args:
        size: Size in bytes.

    Returns:
        ``size`` formatted to human-readable unit.

    """
    units = ("KB", "MB", "GB", "TB")
    size_list = [f"{int(size):,} B"]  # initialize with size in bytes
    for i, unit in enumerate(units):
        # get size in unit
        size_unit = f"{int(size) / 1024 ** (i + 1):,.1f} {unit}"
        if size_unit.startswith("0."):
            # stop if absolute val in this unit is < 1
            break
        size_list.append(size_unit)
    
    # get largest unit before size is < 1
    return size_list[-1]


def truncate_decimal_digit(val, repeats=3, trim_near=False):
    """Truncate floats that may have gained unintended decimal point digits
    because of floating point representation.

    Floating point representation may cause a number input as 3.0 to be
    displayed as 3.0000000000000004, for example. To remove the
    unintended numbers, assume that consecutive repeats of a given digit
    indicates the place at which the decimal points can be truncated.

    Repeated 0's immediately after the decimal point in values between 0-1
    (eg. 0.00000000012 will be retained on the assumption that the repeated
    0's are intended for the small value.

    Args:
        val (float): Number to truncate.
        repeats (int): Number of consecutive decimal digits repeats at which
            to truncate `val`; defaults to 3. Only the first digit of the
            repeated value stretch is retained.
        trim_near (bool): True to remove even this first digit if the repeated
            value is "near", ie 0 or 9, which are assumed to be floating
            point errors (eg 3.0000001 or 2.999999998); defaults to False.

    Returns:
        str: String representation of ``val`` with truncated decimals.

    """
    # convert to string, which will remove all but last trailing 0 decimals
    val_str = str(val)
    val_str_split = val_str.split(".")
    if len(val_str_split) > 1:
        last = None
        n = 0
        for i, s in enumerate(val_str_split[1]):
            if s == last:
                # increment consecutively repeated value
                n += 1
                if n >= repeats:
                    # retain the first instance of the value that is repeated
                    # unless trimming "near" (likely floating point error) vals
                    extra = 1 if trim_near and last in "09" else 2
                    return format_num(val, i - n + extra, allow_scinot=False)
            elif val_str_split[0] != "0" or not (last is None and s == "0"):
                # track a new potentially repeated value as long as not 0
                # at start of value between 0-1 since these 0's may
                # intentionally indicate a small value
                last = s
                n = 1
    return val_str


def convert_bin_magnitude(val, orders):
    """Convert a number to another binary order of magnitude.

    Args:
        val (int, float): Value to convert.
        orders (int): Orders of magnitude, such as 3 to convert a value in
            bytes to gibibytes (GiB), or -2 to convert a value in
            tebibytes (TiB) to mebibytes (MiB).

    Returns:

    """
    return val / 1024 ** orders


def convert_indices_to_int(dict_to_convert):
    """Convert indices of a dictionary to int if possible, including nested 
    indices.
    
    Args:
        dict_to_convert: Dictionary whose indices will be converted.
    
    Returns:
        The converted dictionary.
    """
    dict_converted = {
        get_int(k): [get_int(i) for i in v] if isinstance(v, list) else v 
        for k, v in dict_to_convert.items()
    }
    return dict_converted


def npstr_to_array(s, shape=None):
    """Convert a string representation of a Numpy array back to an array.

    Args:
        s: String representation of a Numpy array such as from a ``print``
            command.
        shape: Tuple of ints by which to reshape the Numpy array. Defaults
            to None, in which case the output will be a 1-D array even
            if the input is multi-dimensional.

    Returns:
        A numpy array, or None if the string could not be converted.

    """
    arr = None
    if isinstance(s, str):
        try:
            # outputs 1-D array
            arr = np.fromstring(s.replace("[", "").replace("]", ""), sep=" ")
            if shape is not None:
                arr = arr.reshape(shape)
        except ValueError:
            pass
    return arr


def show_full_arrays(on=True):
    """Show full Numpy arrays, except for particularly large arrays.
    
    Args:
        on: True if full printing should be turned on; False to reset
            all settings.
    """
    if on:
        np.set_printoptions(linewidth=500, threshold=10000000)
    else:
        np.set_printoptions()


def print_compact(arr, msg=None, mid=False):
    """Print a Numpy array in a compact form to visual comparison with 
    other arrays.
    
    The array will be rounded, converted to integer, and optionally 
    reduced to a single plane to maximize chance of printing a 
    non-truncated array (or plane) to compare more easily with modified 
    versions of the array or other arrays.
    
    Args:
        arr: Numpy array.
        msg: Message to print on separate line before the array; defaults to 
            None, in which case the message will not be printed.
        mid: True to only print the middle element of the array.
    
    Returns:
        The compact array as a new array.
    """
    compact = np.around(arr).astype(int)
    if msg: print(msg)
    if mid:
        i = len(compact) // 2
        print(compact[i])
    else:
        print(compact)
    return compact


def compact_float(n, max_decimals=None):
    """Reduce a float to a more compact value.
    
    Args:
        n: Floating point number.
        max_decimals: Maximum decimals to keep; defaults to None.
    
    Returns:
        An integer if `n` is essentially an integer, or a string 
        representation of `n` reduced to `max_decimals` numbers after 
        the decimal point. Otherwise, simply returns `n`.
    """
    compact = n
    if float(n).is_integer():
        compact = int(n)
    elif max_decimals is not None:
        compact = "{0:.{1}f}".format(n, max_decimals)
    return compact


def backup_file(
        path: Union[str, pathlib.Path], modifier: str = "",
        i: Optional[int] = None):
    """Backup a file to the next given available path with an index number 
    before the extension.
    
    The backed up path will be in the format 
    ``path-before-ext[modifier](i).ext``, where ``[modifier]`` is an optional 
    additional string, and ``i`` is the index number, which will be 
    incremented to avoid overwriting an existing file. Will also backup 
    any associated files as given by :const:``_FILE_TYPE_GROUPS``.
    
    Args:
        path: Path of file to backup.
        modifier: Modifier string to place before the index number.
        i: Index to use; typically use default of None to iniate recursivie 
            backup calls.
    """
    if not i:
        if not os.path.exists(path):
            # original path does not exist, so no need to back up
            return
        i = 0
    while True:
        if i == 0 and modifier != "":
            # check modifier directly first
            backup_path = insert_before_ext(path, modifier)
        else:
            # start incrementing from 1
            if i == 0: i = 1
            backup_path = insert_before_ext(path, "{}({})".format(modifier, i))
        if not os.path.exists(backup_path):
            # backup file to currently non-existent path
            shutil.move(path, backup_path)
            print("Backed up {} to {}".format(path, backup_path))
            path_split = os.path.splitext(path)
            if len(path_split) > 1:
                # remove ".", which should exist if path was split, and get 
                # any associated file to backup as well
                ext_associated = _FILE_TYPE_GROUPS.get(path_split[1][1:])
                if ext_associated:
                    # back up associated file with i
                    backup_file(
                        "{}.{}".format(path_split[0], ext_associated), 
                        modifier, i)
            break
        i += 1


def copy_backup(src: str, target: str):
    """Wrapper to copy a file with backup for the target location.
    
    Uses :meth:`shutil.copy2` for the copying. Backs up ``target`` beforehand
    using :meth:`backup_file` unless ``target`` is a directory.
    
    Args:
        src: Source path. If None or non-existant, no copy or backup will occur.
        target: Target path.

    """
    try:
        if not src or not os.path.exists(src):
            # stop if source does not exist
            raise IOError(f"Cannot copy '{src}', does not exist")
        if os.path.isdir(target):
            # get full target path for backup
            target = os.path.join(target, os.path.basename(src))
        backup_file(target)
        shutil.copy2(src, target)
        _logger.debug("Copied '%s' to '%s'", src, target)
    except IOError as e:
        _logger.error(e)


def create_symlink(
        src: Union[str, pathlib.Path], target: Union[str, pathlib.Path],
        **kwargs):
    """Wrapper to create symbolic link with error handling.
    
    Creates a symlink through :meth:`os.symlink`.
    
    Args:
        src: Source path.
        target: Target path.
        **kwargs: Extra arguments to :meth:`os.symlink`.

    """
    try:
        os.symlink(src, target, **kwargs)
    except (FileExistsError, OSError) as e:
        _logger.debug(e)
        _logger.debug("Skipping link from '%s' to '%s'", src, target)


def is_binary(img):
    """Check if image is binary.
    
    Args:
        img: Image array.
    
    Returns:
        True if the image is composed of only 0 and 1 values.
    """
    return ((img == 0) | (img == 1)).all()


def last_lines(path, n):
    """Get the last lines of a file by simply loading the entire file and 
    returning only the last specified lines, without depending on any 
    underlying system commands.
    
    Args:
        path: Path to file.
        n: Number of lines at the end of the file to extract; if the file is 
            shorter than this number, all lines are returned.
    
    Returns:
        The last ``n`` lines as a list, or all lines if ``n`` is greater than 
        the number of lines in the file.
    """
    lines = None
    with open(path, "r") as f:
        lines = f.read().splitlines()
        num_lines = len(lines)
        if n > num_lines:
            return lines
    return lines[-1*n:]


def coords_for_indexing(coords):
    """Convert array of coordinates to array for fancy indexing in a 
    Numpy array.
    
    Args:
        coords: Array of shape (n, m), where n = number of coordinate sets, 
            and m = number of coordinate dimensions.
    
    Returns:
        Array of coordinates split into axes, such as 
        `nd.array(rows_array, columns_array)`). This array can then be used 
        to index another array through `arr[tuple(indices)]`.
    """
    coordsi = np.transpose(coords)
    coordsi = np.split(coordsi, coordsi.shape[0])
    return coordsi


def dtype_within_range(min_val, max_val, integer=None, signed=None):
    """Get a dtype that will contain the given range.
    
    :const:``_DTYPES`` will be used to specify the possible dtypes.
    
    Args:
        min_val: Minimum required value, inclusive.
        max_val: Maximim required value, inclusive.
        integer: True to get an int type, False for float. Defaults to None
            to determine automatically based on ``max_val``.
        signed: True for a signed int, False for unsigned; ignored for float. 
            Defaults to None to determine automatically based on ``min_val``.
    
    Returns:
        The dtype fitting the range specifications.
    
    Raise:
        TypeError if a dtype with the appropriate range cannot be found.
    """
    if signed is None:
        # determine automatically based on whether min val is neg
        signed = min_val < 0
    if integer is None:
        integer = is_int(max_val)
    if integer:
        type_group = "int" if signed else "uint"
        fn_info = np.iinfo
    else:
        type_group = "float"
        fn_info = np.finfo
    types = _DTYPES[type_group]
    for dtype in types:
        if fn_info(dtype).min <= min_val and fn_info(dtype).max >= max_val:
            return dtype
    raise TypeError(
        "unable to find numerical type (integer {}, signed {}) containing "
        "range {} through {}".format(integer, signed, min_val, max_val))


def get_dtype_info(arr):
    """Get the type information for the given array's data type.

    Args:
        arr (:obj:`np.ndarray`): Numpy array, assumed to be either an integer
            or floating point array.

    Returns:
        :obj:`np.iinfo`, :obj:`np.finfo`: Numpy integer or floating point
        information object.

    """
    try:
        # assume integer
        return np.iinfo(arr.dtype)
    except ValueError:
        pass
    # get floating point info
    return np.finfo(arr.dtype)


def is_seq(val):
    """Check if a value is a non-string sequence.
    
    Arg:
        val (Any): Value to check.
    
    Returns:
        bool: True if the value is a list, tuple, or Numpy array.
    """
    # Numpy rec instead of isscalar to handle more cases such as 0d Numpy
    # arrays and third-party objects, first checking if list/tuple to
    # avoid converting them to an ndarray
    return isinstance(val, (list, tuple)) or np.ndim(val) != 0


def to_seq(val, non_none=True):
    """Wrap a value in a sequence if not already a sequence.
    
    Args:
        val (Any): Value to wrap in a sequence.
        non_none (bool): True to only wrap in a sequence if the value is
            not None; False to allow creating ``[None]``.

    Returns:
        list: A sequence of the value if it is not already a sequence.

    """
    if not is_seq(val):
        if not non_none or val is not None:
            # avoid wrapping None in a sequence unless flagged to
            val = [val]
    return val


def get_if_within(val, i, default=None):
    """Get a value from a sequence if available, otherwise returning the
    value if it is a scalar or a default value if the value at the index
    is not available.

    Args:
        val (Any): Scalar or sequence.
        i (int): Index to extract an element from ``val`` if the index
            is available.
        default (Any): Default value to return if ``val`` is a sequence
            but shorter than or equal in length to ``i``.

    Returns:
        Any: Element ``i`` from ``val`` if present, or ``default`` unless
        ``val`` is not a sequence, in which case ``val`` is simply returned.

    """
    if not is_seq(val):
        return val
    elif len(val) > i:
        return val[i]
    return default


def enum_names_aslist(c, lower=True):
    """Get an Enum class as a list of enum names.

    Args:
        c (:class:`Enum`): Enum class.
        lower (bool): True to get names as lower case; defaults to True for
            easier comparison with other strings.

    Returns:
        List: List of enum names.

    """
    return [e.name.lower() if lower else e.name for e in c]


def enum_dict_aslist(d):
    """Summarize a dictionary with enums as keys as a shortened 
    list with only the names of each enum member.
    
    Args:
        d: Dictionary with enums as keys.
    
    Returns:
        List of tuples for all dictionary key-value pairs, with each 
        tuple containing the enum member name in place of the full 
        enum as the key.
    """
    return [(key.name, val) for key, val in d.items()]


def get_enum(s, enum_class):
    """Get an enum from a string where the enum class is assumed to have 
    all upper-case keys, returning None if the key is not found.
    
    Args:
        s (str): Key of enum to find, case-insensitive.
        enum_class (:class:`Enum`): Enum class to search.

    Returns:
        The enum if found, otherwise None.

    """
    enum = None
    if s:
        try:
            s_upper = s.upper()
            enum = enum_class[s_upper]
        except (AttributeError, KeyError):
            # AttributeError if s is not a str; KeyError if not in enum
            pass
    return enum


def get_dict_keys_from_val(d, val):
    """Get keys whose value in a dictionary match a given value.
    
    Args:
        d (dict): Dictionary from which to get keys.
        val (Any): Value whose matching keys will be extracted.

    Returns:
        List[Any]: List of keys whose values match ``val`` in ``d``.

    """
    return [k for k, v in d.items() if v == val]


def add_missing_keys(d_src, d_target, override=None):
    """Add dictionary items without overriding existing items.
    
    Add key-value pairs from one dictionary to another if the key does
    not exist in the target dictionary or the corresponding value is an
    overridable value. Thisupdating allows these values to be overridden
    while protecting values that are explicitly set.
    
    Args:
        d_src (dict): Source dictionary, from which key-value pairs will be
            added to ``d_target``.
        d_target (dict): Target dictionary, to which which key-value pairs
            from ``d_src`` if each key is not in ``d_target`` or if
            ``d_target[key]`` is a value in ``override``.
        override (Sequence[Any]): Sequence of values to override even
            if the key is present; defaults to None to use ``(None,)``,
            where any existing value that is None will be overridden.

    Returns:
        dict: ``d_target``, modified in-place.

    """
    if override is None:
        override = (None,)
    for k, v in d_src.items():
        if k not in d_target or d_target[k] not in override:
            d_target[k] = v
    return d_target


def scale_slice(sl, scale, size):
    """Scale slice values by a given factor.
    
    Args:
        sl (slice): Slice object to scale.
        scale (int, float): Scaling factor.
        size (int): Size of the full range, used if ``sl.stop`` is None 
            and generating a sequence of indices.

    Returns:
        Either a new slice object after scaling if ``scale`` is >= 1, or
        a :obj:`np.ndarray` of scaled indices with the same number of elements
        as would be in the original `sl` range.

    """
    scaled = [sl.start, sl.stop, sl.step]
    scaled = [s if s is None else int(s * scale) for s in scaled]
    if scale >= 1:
        # should produce the same number of elements
        return slice(*scaled)
    # interval would be < 1 if scaling down, so need to construct a sequence
    # of indices including repeated indices to get the same total number of
    # elements
    start = 0 if scaled[0] is None else scaled[0]
    end = size if scaled[1] is None else scaled[1]
    return np.linspace(start, end, sl.stop - sl.start, dtype=int)


def get_git_commit(repo_dir: str) -> Optional[str]:
    """Get git commit hash.
    
    Args:
        repo_dir: Path to repository root directory.

    Returns:
        Commit hash, or None if not found.

    """
    # get HEAD file from .git folder
    git_dir = pathlib.Path(repo_dir) / ".git"
    head_path = git_dir / "HEAD"
    if not head_path.is_file(): return None
    
    # get ref path from HEAD file
    with head_path.open("r") as head_file:
        ref = head_file.readline().split(" ")[-1].strip()
    ref_path = git_dir / ref
    if not ref_path.is_file(): return None
    
    # get hash from ref file
    with ref_path.open("r") as ref_file:
        return ref_file.readline().strip()


def get_version(git: bool = False) -> str:
    """Get package version from installed metadata.
    
    The version string is based on the version at time of installation,
    which is only updated when reinstalling the package.
    
    Args:
        git: True to add a short hash of the current Git commit.

    Returns:
        The version string. If ``git`` is True, the current Git commit
        is appended as ``<ver>-<short-hash>``. If package metadata is not
        available, the version is given as "n/a".

    """
    try:
        # get version from installed metadata; note that this version may differ
        # from the imported version:
        # https://packaging.python.org/guides/single-sourcing-package-version/
        ver = metadata.version(config.APP_NAME.lower())
    except metadata.PackageNotFoundError as e:
        # fall back to N/A version
        _logger.exception(e)
        ver = "n/a"
    
    if git:
        git_commit = get_git_commit(
            str(pathlib.Path(__file__).parent.parent.parent))
        if git_commit:
            # add git short hash if available
            ver = f"{ver}-{git_commit[:8]}"
    return ver


if __name__ == "__main__":
    print("Initializing MagellanMapper general library module")
