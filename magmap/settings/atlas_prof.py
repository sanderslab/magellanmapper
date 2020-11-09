# Microscope profile settings
# Author: David Young, 2020
"""Profile settings for processing regions of interests.

These setting typically involve high-resolution such as microscopy images.
"""

from magmap.settings import config
from magmap.settings import profiles
from magmap.settings.profiles import RegKeys


class RegParamMap(dict):
    """Registration parameter map dictionary initialized with required keys."""
    def __init__(self, *args, **kwargs):
        super().__init__(self)
        self["map_name"] = None
        self["metric_similarity"] = "AdvancedMattesMutualInformation"
        # fallback to alternate similarity metric if below DSC threshold as
        # given by (threshold, alternate_metric)
        self["max_iter"] = None
        self["grid_space_voxels"] = None
        self["num_resolutions"] = "4"
        self["grid_spacing_schedule"] = None
        # True to prevent artificial edges from entering ROI during smoothing,
        # but leave False to use area around mask for the registration
        # (see Elastix manual section 5.4)
        self["erode_mask"] = None
        self["point_based"] = False


class AtlasProfile(profiles.SettingsDict):
    """Atlas profile dictionary.

    Attributes:
        PATH_PREFIX (str): Prefix for atlas profile files.

    """

    PATH_PREFIX = "atlas"

    def __init__(self, *args, **kwargs):
        super().__init__(self)
        self[self.NAME_KEY] = self.DEFAULT_NAME

        # REGISTRATION SETTINGS

        # translation transform
        reg_translation = RegParamMap()
        reg_translation["map_name"] = "translation"
        reg_translation["max_iter"] = "2048"
        self["reg_translation"] = reg_translation

        # affine/scaling transform
        reg_affine = RegParamMap()
        reg_affine["map_name"] = "affine"
        reg_affine["max_iter"] = "1024"
        self["reg_affine"] = reg_affine

        # b-spline transform
        reg_bspline = RegParamMap()
        reg_bspline["map_name"] = "bspline"
        reg_bspline["max_iter"] = "512"
        reg_bspline["grid_space_voxels"] = "50"
        self["reg_bspline"] = reg_bspline

        # similarity metric fallback given as ``(threshold_dsc, metric)``,
        # where metric will be used if the DSC falls below the threshold
        self["metric_sim_fallback"] = None
        self["groupwise_iter_max"] = "1024"
        self["resize_factor"] = 0.7
        self["preprocess"] = False
        # True to use point-based registration during b-spline reg
        self["point_based"] = False
        self["curate"] = True  # carve image; in-paint if generating atlas

        # erase labels outside of ((x_start, x_end), (y_start, ...) ...)
        # (applied after transposition), where each val is given as fractions
        # of the full range or None to not truncate that at that position;
        # None for the entire setting turns off truncation
        self["truncate_labels"] = None

        # LABELS CURATION

        # ACTIVE (bool): True to apply the setting to the final image and
        # metrics; False to use only for metrics and cropping, etc
        # start (float): fractions of the total planes (0-1); use -1 to
        # set automatically, None to turn off the entire setting group

        self["smooth"] = None  # smooth labels
        self["crop_to_labels"] = False  # crop labels and atlas to non-0 labels

        # mirror labels onto the unlabeled hemisphere
        self["labels_mirror"] = {
            RegKeys.ACTIVE: False,
            "start": None,  # reflect planes starting here
            "neg_labels": True,  # invert values of mirrored labels
            "atlas_mirror": True,  # also mirror intensity image
        }
        # extend edge labels
        self["labels_edge"] = {
            RegKeys.ACTIVE: False,
            RegKeys.SAVE_STEPS: False,
            "start": None,  # start plane index
            "surr_size": 5,  # dilation filter size for finding histology region
            # smoothing filter size to remove artifacts (None or 0 to ignore)
            "smoothing_size": 3,
            "in_paint": True,  # True to fill pxs missing labels
            # erosion filter size for watershed markers (0 to ignore), which
            # are weighted by distance to border
            RegKeys.MARKER_EROSION: 10,
            # no erosion if weighted filter size is below min; None to use
            # default size (half of erosion filter size); 0 for no min
            RegKeys.MARKER_EROSION_MIN: None,
            # if filter size is below min, still erode at this size if True
            RegKeys.MARKER_EROSION_USE_MIN: False,  # don't erode if reach min
            # preferentially weight erosion filter sizes in lateral planes
            # so that sizes are reduced linearly to this fraction in the most
            # medial plane; 0 = no weighting, 1 = full weighting
            "wt_lat": 0
        }
        self["labels_dup"] = None  # start duplicating planes til last labels

        # expand labels within bounds given by
        # (((x_pixels_start, x_pixels_end), ...), (next_region...)), or None
        # to avoid expansion
        self["expand_labels"] = None

        # crop atlas and intensity to fit a mask that excludes this sequence
        # of labels; note that labels within this cropped region will remain
        self["crop_out_labels"] = None

        # atlas and labels rotation by ((angle0, axis0), ...), or None to
        # avoid rotation, where an axis value of 0 is the z-axis, 1 is y, etc
        self["rotate"] = {
            "rotation": None,
            "resize": False,  # True to keep full image rather than clipping
            # spline interpolation; 0 for labels (ignored for known labels imgs)
            "order": 1,
        }

        # atlas thresholds for microscopy images
        self["atlas_threshold"] = 10.0  # raise for finer segmentation
        self["atlas_threshold_all"] = 10.0  # keep low to include all signal

        self["target_size"] = None  # x,y,z in exp orientation

        self["rescale"] = None  # rescaling factor

        # carving and max size of small holes for removal, respectively
        self["carve_threshold"] = None
        self["holes_area"] = None

        # paste in region from first image during groupwise reg;
        # x,y,z, same format as truncate_labels except in pixels
        self["extend_borders"] = None

        # affine transformation as a dict of ``axis_along`` for the axis along
        # which to perform transformation (ie the planes that will be
        # affine transformed); ``axis_shift`` for the axis or
        # direction in which to shear; ``shift`` for a tuple of indices
        # of starting to ending shift while traveling from low to high
        # indices along ``axis_along``; ``bounds`` for a tuple of
        # ``((z_start z_end), (y_start, ...) ...)`` indices (note the
        # z,y,x ordering to use directly); and an optional ``axis_attach``
        # for the axis along which to perform another affine to attach the
        # main affine back to the rest of the image
        self["affine"] = None

        # Laplacian of Gaussian
        self["log_sigma"] = 5  # Gaussian sigma; use None to skip
        # use atlas_threshold on atlas image to generate mask for finding
        # background rather than using labels and thresholded LoG image,
        # useful when ventricular spaces are labeled
        self["log_atlas_thresh"] = False

        # edge-aware reannotation: label erosion to generate watershed
        # seeds/markers for resegmentation; also used to demarcate the interior
        # of regions for metrics; can turn on/off with erode_labels
        self[RegKeys.EDGE_AWARE_REANNOTATION] = {
            RegKeys.MARKER_EROSION: 8,  # filter size for labels to markers
            RegKeys.MARKER_EROSION_MIN: 1,  # None for default, 0 for no min
            RegKeys.WATERSHED_MASK_FILTER: (config.SmoothingModes.opening, 2),
        }
        # target eroded size as frac of orig, used when generating interiors
        # of regions but not for watershed seeds; can be None
        self["erosion_frac"] = 0.5
        self["erode_labels"] = {"markers": True, "interior": False}

        # crop labels back to their original background after smoothing
        # (ignored during atlas import if no smoothing), given as the filter
        # size used to open up the background before cropping, 0 to use
        # the original background as-is, or False not to crop
        self["crop_to_orig"] = 1
        
        # crop labels images to foreground of first labels image
        self["crop_to_first_image"] = False

        # type of label smoothing
        self["smoothing_mode"] = config.SmoothingModes.opening

        # combine values from opposite sides when measuring volume stats;
        # default to use raw values for each label and side to generate
        # a data frame that can be used for fast aggregation when
        # grouping into levels
        self["combine_sides"] = False

        # make the far hemisphere neg if it is not, for atlases (eg P56) with
        # bilateral pos labels where one half should be made neg for stats
        self["make_far_hem_neg"] = False

        # the atlas refiner assumes that the z-axis is sagittal planes; use
        # this setting to transpose an image to this orientation for
        # refinement, after which the image will be transposed back
        self["pre_plane"] = None

        # labels range given as ``((start0, end0), (start1, end1), ...)``,
        # where labels >= start and < end will be treated as foreground
        # when measuring overlap, eg labeled ventricles that would be
        # background in histology image
        self["overlap_meas_add_lbls"] = None

        # METRICS

        # sequence of :class:`config.MetricGroups` enums to measure in
        # addition to basic metrics
        self["extra_metric_groups"] = None

        # cluster metrics
        self[RegKeys.METRICS_CLUSTER] = {
            RegKeys.KNN_N: 5,  # num of neighbors for k-nearest-neighbors
            RegKeys.DBSCAN_EPS: 20,  # epsilon for max dist in cluster
            RegKeys.DBSCAN_MINPTS: 6,  # min points/samples per cluster
        }

        # the default unit is microns (um); use this factor to convert
        # atlases in other units to um (eg 1000 for atlases in mm)
        self["unit_factor"] = None

        # ATLAS EDITOR

        # downsample images shown in the Atlas Editor to improve performance
        # when loaded through these I/O packages; default to Numpy because
        # its memory mapping reduces memory but is slower to load x-planes
        self["editor_downsample_io"] = [config.LoadIO.NP]
        # downsample image planes with an edge size exceeding these values,
        # given as edge sizes of x,y,z-planes; decrease sizes to improve
        # performance, especially in x
        self["editor_max_sizes"] = (500, 1000, 2000)

        self.profiles = {

            # turn off bspline registrations
            "nobspline": {
                "reg_bspline": None,
            },

            # turn off affine and bspline registrations
            "noaffinebspline": {
                "reg_affine": None,
                "reg_bspline": None,
            },

            # Normalized Correlation Coefficient similarity metric for
            # registration
            "ncc": {
                "reg_translation": {
                    "metric_similarity": "AdvancedNormalizedCorrelation"},
                "reg_affine": {
                    "metric_similarity": "AdvancedNormalizedCorrelation"},
                "reg_bspline": {
                    "metric_similarity": "AdvancedNormalizedCorrelation",
                    "grid_space_voxels": "60",
                },
                # fallback to MMI since it has been rather reliable
                "metric_sim_fallback":
                    (0.85, self.DEFAULT_NAME),
            },

            # groupwise registration
            "groupwise": {
                # larger bspline voxels to avoid over deformation of internal
                # structures
                "reg_bspline": {
                    "grid_space_voxels": "130",
                    # increased num of resolutions with anisotropic size
                    # (x0, y0, z0, x1, y1, z1, x2, ...) and overall increased
                    # spacing since it appears to improve internal alignment
                    "grid_spacing_schedule": [
                        "8", "8", "4", "4", "4", "2", "2", "2", "1",
                        "1", "1", "1"],
                },
                "bspline_grid_space_voxels": "130",

                # need to empirically determine
                "carve_threshold": 0.01,
                "holes_area": 10000,

                # empirically determined to add variable tissue area from
                # first image since this tissue may be necessary to register
                # to other images that contain this variable region
                "extend_borders": ((60, 180), (0, 200), (20, 110)),
            },

            # test registration function with all registrations turned off
            "testreg": {
                "reg_translation": {"max_iter": "0"},  # need at least one reg
                "reg_affine": None,
                "reg_bspline": None,
                "curate": False,
            },

            # test adding parameter maps for each type of registration
            # without actually performing any iterations
            "testnoiter": {
                "reg_translation": {"max_iter": "0"},
                "reg_affine": {"max_iter": "0"},
                "reg_bspline": {"max_iter": "0"},
                "curate": False,
            },

            # test a target size
            "testsize": {
                "target_size": (50, 50, 50),
            },

            # atlas is big relative to the experimental image, so need to
            # more aggressively downsize the atlas
            "big": {
                "resize_factor": 0.625,
            },

            # new atlas generation: turn on preprocessing
            # TODO: likely remove since not using preprocessing currently
            "new": {
                "preprocess": True,
            },

            # registration to new atlas assumes images are roughly same size and
            # orientation (ie transposed) and already have mirrored labels aligned
            # with the fixed image toward the bottom of the z-dimension
            "generated": {
                "resize_factor": 1.0,
                "truncate_labels": (None, (0.18, 1.0), (0.2, 1.0)),
                "labels_mirror": {RegKeys.ACTIVE: False},
                "labels_edge": None,
            },

            # atlas that uses groupwise image as the atlas itself should
            # determine atlas threshold dynamically
            "grouped": {
                "atlas_threshold": None,
            },

            # ABA E11pt5 specific settings
            "abae11pt5": {
                "target_size": (345, 371, 158),
                "resize_factor": None,  # turn off resizing
                "labels_mirror": {RegKeys.ACTIVE: True, "start": 0.52},
                "labels_edge": {RegKeys.ACTIVE: False, "start": None},
                "log_atlas_thresh": True,
                "atlas_threshold": 75,  # avoid over-extension into ventricles
                "atlas_threshold_all": 5,  # include ventricles since labeled
                # rotate axis 0 to open vertical gap for affines (esp 2nd)
                "rotate": {
                    "rotation": ((-5, 1), (-1, 2), (-30, 0)),
                    "resize": False,
                },
                "affine": ({
                   # shear cord opposite the brain back toward midline
                   "axis_along": 1, "axis_shift": 0, "shift": (25, 0),
                   "bounds": ((None, None), (70, 250), (0, 150))
                }, {
                   # shear distal cord where the tail wraps back on itself
                   "axis_along": 2, "axis_shift": 0, "shift": (0, 50),
                   "bounds": ((None, None), (0, 200), (50, 150))
                }, {
                   # counter shearing at far distal end, using attachment for
                   # a more gradual shearing along the y-axis to preserve the
                   # cord along that axis
                   "axis_along": 2, "axis_shift": 0, "shift": (45, 0),
                   "bounds": ((None, None), (160, 200), (90, 150)),
                   "axis_attach": 1
                }),
                "crop_to_labels": True,  # req because of 2nd affine
                "smooth": 2,
                "overlap_meas_add_lbls": ((126651558, 126652059),),
            },

            # ABA E13pt5 specific settings
            "abae13pt5": {
                "target_size": (552, 673, 340),
                "resize_factor": None,  # turn off resizing
                "labels_mirror": {RegKeys.ACTIVE: True, "start": 0.48},
                # small, default surr size to avoid capturing 3rd labeled area
                # that becomes an artifact
                "labels_edge": {
                    RegKeys.ACTIVE: True,
                    "start": -1,
                },
                "atlas_threshold": 55,  # avoid edge over-extension into skull
                "rotate": {
                    "rotation": ((-4, 1), (-2, 2)),
                    "resize": False,
                },
                RegKeys.EDGE_AWARE_REANNOTATION: {
                    # use a small closing filter avoid label loss
                    RegKeys.WATERSHED_MASK_FILTER: (
                        config.SmoothingModes.closing, 1),
                },
                "crop_to_labels": True,
                "smooth": 2,
            },

            # ABA E15pt5 specific settings
            "abae15pt5": {
                "target_size": (704, 982, 386),
                "resize_factor": None,  # turn off resizing
                "labels_mirror": {RegKeys.ACTIVE: True, "start": 0.49},
                "labels_edge": {
                    RegKeys.ACTIVE: True,
                    "start": -1,
                    "surr_size": 12,
                    # increase template smoothing to prevent over-extension of
                    # intermediate stratum of Str
                    "smoothing_size": 5,
                    # larger to allow superficial stratum of DPall to take over
                    RegKeys.MARKER_EROSION: 19,
                },
                "atlas_threshold": 45,  # avoid edge over-extension into skull
                "rotate": {
                    "rotation": ((-4, 1),),
                    "resize": False,
                },
                RegKeys.EDGE_AWARE_REANNOTATION: {
                    # turn off filtering to avoid label loss
                    RegKeys.WATERSHED_MASK_FILTER: (None, 0),
                },
                "crop_to_labels": True,
                "smooth": 2,
            },

            # ABA E18pt5 specific settings
            "abae18pt5": {
                "target_size": (278, 581, 370),
                "resize_factor": None,  # turn off resizing
                "labels_mirror": {RegKeys.ACTIVE: True, "start": 0.525},
                # start from smallest BG; remove spurious label pxs around
                # medial pallium by smoothing
                "labels_edge": {
                    RegKeys.ACTIVE: True, "start": 0.137, "surr_size": 12,
                    RegKeys.MARKER_EROSION: 12,
                    RegKeys.MARKER_EROSION_USE_MIN: True,
                },
                "expand_labels": (((None,), (0, 279), (103, 108)),),
                "rotate": {
                    "rotation": ((1.5, 1), (2, 2)),
                    "resize": False,
                },
                "smooth": 3,
                RegKeys.EDGE_AWARE_REANNOTATION: {
                    RegKeys.MARKER_EROSION_MIN: 4,
                }
            },

            # ABA P4 specific settings
            "abap4": {
                "target_size": (724, 403, 398),
                "resize_factor": None,  # turn off resizing
                "labels_mirror": {RegKeys.ACTIVE: True, "start": 0.487},
                "labels_edge": {
                    RegKeys.ACTIVE: True,
                    "start": -1,
                    "surr_size": 12,
                    # balance eroding medial pallium and allowing dorsal
                    # pallium to take over
                    RegKeys.MARKER_EROSION: 8,
                },
                # open caudal labels to allow smallest mirror plane index,
                # though still cross midline as some regions only have
                # labels past midline
                "rotate": {
                    "rotation": ((0.22, 1),),
                    "resize": False,
                },
                "smooth": 4,
            },

            # ABA P14 specific settings
            "abap14": {
                "target_size": (390, 794, 469),
                "resize_factor": None,  # turn off resizing
                # will still cross midline since some regions only have labels
                # past midline
                "labels_mirror": {RegKeys.ACTIVE: True, "start": 0.5},
                "labels_edge": {
                    RegKeys.ACTIVE: True,
                    "start": 0.078,  # avoid alar part size jitter
                    "surr_size": 12,
                    RegKeys.MARKER_EROSION: 40,
                    RegKeys.MARKER_EROSION_MIN: 10,
                },
                # rotate conservatively for symmetry without losing labels
                "rotate": {
                    "rotation": ((-0.4, 1),),
                    "resize": False,
                },
                "smooth": 5,
            },

            # ABA P28 specific settings
            "abap28": {
                "target_size": (863, 480, 418),
                "resize_factor": None,  # turn off resizing
                # will still cross midline since some regions only have labels
                # past midline
                "labels_mirror": {RegKeys.ACTIVE: True, "start": 0.48},
                "labels_edge": {
                    RegKeys.ACTIVE: True,
                    "start": 0.11,  # some lat labels only partially complete
                    "surr_size": 12,
                    "smoothing_size": 0,  # no smoothing to avoid loss of detail
                    # large erosion in outer planes (weighted ~off medially)
                    RegKeys.MARKER_EROSION: 50,
                    RegKeys.MARKER_EROSION_MIN: 20,
                    # most erosion in lateral planes; minimal in medial planes
                    "wt_lat": 0.9,
                },
                # "labels_dup": 0.48,
                # rotate for symmetry, which also reduces label loss
                "rotate": {
                    "rotation": ((1, 2),),
                    "resize": False,
                },
                "smooth": 2,
            },

            # ABA P56 (developing mouse) specific settings
            "abap56": {
                "target_size": (528, 320, 456),
                "resize_factor": None,  # turn off resizing
                # stained sections and labels almost but not symmetric
                "labels_mirror": {RegKeys.ACTIVE: True, "start": 0.5},
                "labels_edge": {
                    RegKeys.ACTIVE: True,
                    "start": 0.138,  # some lat labels only partially complete
                    "surr_size": 12,
                    "smoothing_size": 0,  # no smoothing to avoid loss of detail
                    # only mild erosion to minimize layer loss since histology
                    # contrast is low
                    RegKeys.MARKER_EROSION: 5,
                },
                "smooth": 2,
                "make_far_hem_neg": True,
            },

            # ABA P56 (adult) specific settings
            "abap56adult": {
                # same atlas image as ABA P56dev
                "target_size": (528, 320, 456),
                "resize_factor": None,  # turn off resizing
                # same stained sections as for P56dev;
                # labels are already mirrored starting at z=228, but atlas is
                # not here, so mirror starting at the same z-plane to make both
                # sections and labels symmetric and aligned with one another;
                # no need to extend lateral edges
                "labels_mirror": {RegKeys.ACTIVE: True, "start": 0.5},
                "smooth": 2,
                "make_far_hem_neg": True,
            },

            # ABA CCFv3 specific settings
            "abaccfv3": {
                # for "25" image, which has same shape as ABA P56dev, P56adult
                "target_size": (456, 528, 320),
                "resize_factor": None,  # turn off resizing
                # atlas is almost (though not perfectly) symmetric, so turn
                # off mirroring but specify midline (z=228) to make those
                # labels negative; no need to extend lateral edges
                "labels_mirror": {RegKeys.ACTIVE: False, "start": 0.5},
                "make_far_hem_neg": True,
                "smooth": 0,
            },

            # Waxholm rat atlas specific settings
            "whsrat": {
                "target_size": (441, 1017, 383),
                "pre_plane": config.PLANE[2],
                "resize_factor": None,  # turn off resizing
                # mirror, but no need to extend lateral edges
                "labels_mirror": {RegKeys.ACTIVE: True, "start": 0.48},
                "crop_to_labels": True,  # much extraneous, unlabeled tissue
                "smooth": 4,
                "unit_factor": 1000,
            },

            # Profile modifiers to turn off settings. These "no..." profiles
            # can be applied on top of atlas-specific profiles to turn off
            # specific settings. Where possible, the ACTIVE flags will be turned
            # off to retain the rest of the settings within the given group
            # so that they can be used for metrics, cropping, etc.

            # turn off most image manipulations to show original atlas and labels
            # while allowing transformations set as command-line arguments
            "raw": {
                "labels_edge": {RegKeys.ACTIVE: False},
                "labels_mirror": {RegKeys.ACTIVE: False},
                "expand_labels": None,
                "rotate": None,
                "affine": None,
                "smooth": None,
                "crop_to_labels": False,
            },

            # turn off atlas rotation
            "norotate": {
                "rotate": None,
            },

            # turn off edge extension along with smoothing
            "noedge": {
                "labels_edge": {RegKeys.ACTIVE: False},
                "labels_mirror": {RegKeys.ACTIVE: True},
                "smooth": None,
            },

            # turn off mirroring along with smoothing
            "nomirror": {
                "labels_edge": {RegKeys.ACTIVE: True},
                "labels_mirror": {RegKeys.ACTIVE: False},
                "smooth": None,
            },

            # turn off both mirroring and edge extension along with smoothing
            # while preserving their settings for measurements and cropping
            "noext": {
                "labels_edge": {RegKeys.ACTIVE: False},
                "labels_mirror": {RegKeys.ACTIVE: False},
                "smooth": None,
            },

            # turn off label smoothing
            "nosmooth": {
                "smooth": None,
            },

            # turn off negative labels
            "noneg": {
                # if mirroring, do not invert mirrored labels
                "labels_mirror": {"neg_labels": False},
                # do not invert far hemisphere labels
                "make_far_hem_neg": False,
            },

            # set smoothing to 4
            "smooth4": {
                "smooth": 4,
            },

            # turn off labels markers generation
            "nomarkers": {
                RegKeys.EDGE_AWARE_REANNOTATION: None,
            },

            # turn off cropping atlas to extent of labels
            "nocropatlas": {
                "crop_to_labels": False,
            },

            # turn off cropping labels to original size
            "nocroplabels": {
                "crop_to_orig": False,
            },

            # test label smoothing over range
            "smoothtest": {
                "smooth": (0, 1, 2, 3, 4, 5, 6, 7, 8),
                # "smooth": (0, ),
            },

            # test label smoothing over longer range
            "smoothtestlong": {
                "smooth": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            },

            # save intermediate steps where supported
            "savesteps": {
                "labels_edge": {RegKeys.SAVE_STEPS: True}
            },

            # crop ventral and posterior regions
            "cropventropost": {
                "truncate_labels": (None, (0.2, 1.0), (0.45, 1.0)),
            },

            # crop anterior region of labels during single registration
            "cropanterior": {
                "truncate_labels": (None, (0.2, 0.8), (0.45, 1.0)),
            },

            # turn off image curation to avoid post-processing with carving
            # and in-painting
            "nopostproc": {
                "curate": False,
                "truncate_labels": None
            },

            # smoothing by Gaussian blur
            "smoothgaus": {
                "smoothing_mode": config.SmoothingModes.gaussian,
                "smooth": 0.25
            },

            # smoothing by Gaussian blur
            "smoothgaustest": {
                "smoothing_mode": config.SmoothingModes.gaussian,
                "smooth": (0, 0.25, 0.5, 0.75, 1, 1.25)
            },

            # combine sides for volume stats
            "combinesides": {
                "combine_sides": True,
            },

            # more volume stats
            "morestats": {
                # "extra_metric_groups": (config.MetricGroups.SHAPES,),
                "extra_metric_groups": (config.MetricGroups.POINT_CLOUD,),
            },

            # measure interior-border stats
            "interiorlabels": {
                "erode_labels": {"markers": True, "interior": True},
            },

        }

    @staticmethod
    def get_files(profiles_dir=None, filename_prefix=None):
        """Get atlas profile files.

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
            filename_prefix = AtlasProfile.PATH_PREFIX
        return super(AtlasProfile, AtlasProfile).get_files(
            profiles_dir, filename_prefix)

