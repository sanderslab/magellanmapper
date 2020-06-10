# Profile settings
# Author: David Young, 2019, 2020
"""Profile settings for grid search hyperparameter tuning."""

from collections import OrderedDict

import numpy as np

from magmap.settings import profiles


def make_hyperparm_arr(start, stop, num_steps, num_col, coli, base=1):
    """Make a hyperparameter 2D array that varies across the first axis
    for the given index.

    The 2D array is used for grid searches, where each row is given as a
    parameter. Each parameter is a 1D array with the same values except
    at a given index, which varies across these 1D arrays. The varying
    values are constructed by :meth:`np.linspace`.

    Args:
        start (int, float): Starting value for varying parameter.
        stop (int, float): Ending value for varying parameter, inclusive.
        num_steps (int): Number of steps from ``start`` to ``stop``, which
            determines the number of rows in the output array.
        num_col (int): Number of columns in the output array.
        coli (int): Index of column to vary.
        base (int, float): All values are set to this number except for the
            varying values.

    Returns:
        :obj:`np.ndarray`: 2D array in the format ``[[start, base, base, ...],
        [start0, base, base, ...], ...]``.

    """
    steps = np.linspace(start, stop, num_steps)
    arr = np.ones((len(steps), num_col)) * base
    arr[:, coli] = steps
    return arr


class GridSearchProfile(profiles.SettingsDict):
    """Grid search profile dictionary.

    This dictionary is used to set up combinations of parameter values for
    hyperparameter tuning. The dictionary is in the format:
    ``self[profile_name]: {
        hyperparam1: [param1_val1, param1_val2, ...],
        hyperparm2: [param2_val1, param2_val2, ...], ...
    }, ...``.

    Hyperparameter combinations can thus be accessed via profile names.
    Unlike other :class:`profiles.SettingsDict` sub-classes, adding new
    profile dicts does not update corresponding keys, but rather adds
    the entire dict as the item value.

    Attributes:
        PATH_PREFIX (str): Prefix for grid search profile files.

    """

    PATH_PREFIX = "grid"

    def __init__(self, *args, **kwargs):
        """Initialize a grid search dict of hyperparameter ranges.

        Args:
            *args:
            **kwargs:
        """
        super().__init__(self)
        self._add_mod_directly = True
        self[self.NAME_KEY] = ""

        #: OrderedDict[List[int]]: Nested dictionary where each sub-dictionary
        # contains a sequence of values over which to perform a grid search to
        # generate a receiver operating characteristic curve
        self.profiles = OrderedDict([
            ("gridtest", OrderedDict([
                # test single value by iterating on value that should not affect
                # detection ability
                ("points_3d_thresh", [0.7]),

                # unfused baseline
                #("clip_vmax", 98.5),
                #("clip_max", 0.5),
                #("clip_vmax", np.arange(98.5, 99, 0.5)),
                #("clip_max", np.arange(0.5, 0.6, 0.1)),

                # test parameters
                #("isotropic", make_hyperparm_arr(0.2, 1, 9, 3, 0),
                #("isotropic", np.array([(0.96, 1, 1)])),
                #("overlap", np.arange(0.1, 1.0, 0.1)),
                #("prune_tol_factor", np.array([(4, 1.3, 1.3)])),
                #("prune_tol_factor", make_hyperparm_arr(0.5, 1, 2, 0.9, 0)),
                #("clip_min", np.arange(0.0, 0.2, 0.1)),
                #("clip_vmax", np.arange(97, 100.5, 0.5)),
                #("clip_max", np.arange(0.3, 0.7, 0.1)),
                #("erosion_threshold", np.arange(0.16, 0.35, 0.02)),
                #"denoise_size", np.arange(5, 25, 2)
                #("unsharp_strength", np.arange(0.0, 1.1, 0.1)),
                #("tot_var_denoise", (False, True)),
                #("num_sigma", np.arange(5, 16, 1)),
                #("detection_threshold", np.arange(0.001, 0.01, 0.001)),
                #("segment_size", np.arange(130, 160, 20)),
            ])),
            ("size5x", OrderedDict([
                ("min_sigma_factor", np.arange(2, 2.71, 0.1)),
                ("max_sigma_factor", np.arange(2.7, 3.21, 0.1)),
            ])),
            ("size4x", OrderedDict([
                ("min_sigma_factor", np.arange(2.5, 3.51, 0.3)),
                ("max_sigma_factor", np.arange(3.5, 4.51, 0.3)),
            ])),
            ("sizeiso", OrderedDict([
                ("min_sigma_factor", np.arange(2, 3.1, 1)),
                ("max_sigma_factor", np.arange(3, 4.1, 1)),
                ("isotropic", make_hyperparm_arr(0.2, 1, 9, 3, 0)),
            ])),
        ])

    @staticmethod
    def get_files(profiles_dir=None, filename_prefix=None):
        """Get grid search profile files.

        Args:
            profiles_dir (str): Directory from which to get files; defaults
                to None.
            filename_prefix (str): Only get files starting with this string;
                defaults to None to use :const:`PATH_PREFIX`.

        Returns:
            List[str]: List of files in ``profiles_dir`` matching the given
            ``filename_prefix``.

        """
        if not filename_prefix:
            filename_prefix = GridSearchProfile.PATH_PREFIX
        return super(GridSearchProfile, GridSearchProfile).get_files(
            profiles_dir, filename_prefix)

