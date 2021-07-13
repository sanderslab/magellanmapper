"""Labels image metadata."""

import pathlib
from typing import Optional, Sequence

from magmap.io import libmag, yaml_io
from magmap.settings import config

_logger = config.logger.getChild(__name__)


class LabelsMeta:
    """Labels image metadata container and I/O.
    
    Attributes:
        prefix: Path prefix for saving metadata and locating the reference
            file path.
        save_path: Path to save this metadata.
        path_ref: Path to labels reference file.
        region_ids_orig: Sequence of original label IDs.
    
    """
    PATH_LABELS_META: str = "meta_labels.yml"
    
    def __init__(self, prefix: Optional[str] = None):
        """Initialize metadata."""
        super().__init__()
        # output paths
        self.prefix: Optional[str] = prefix
        self.save_path: Optional[str] = None
        
        # metadata
        self.path_ref: Optional[str] = None
        self.region_ids_orig: Optional[Sequence[int]] = None
    
    @property
    def save_path(self):
        """Get the save path.
        
        Returns:
            The save path if set, otherwise a path constructed from
            :attr:`prefix` if set and :const:`PATH_LABELS_META`, or the
            constant alone.

        """
        if not self._save_path:
            if self.prefix:
                return libmag.combine_paths(
                    self.prefix, self.PATH_LABELS_META, check_dir=True)
            return self.PATH_LABELS_META
        return self._save_path
    
    @save_path.setter
    def save_path(self, val):
        """Set the save path."""
        self._save_path = val
    
    def save(self):
        """Save the metadata.
        
        Also copies the reference file to the metadata's directory.
        
        """
        labels_ref_name = None
        if self.path_ref:
            # if provided, copy labels reference file to output dir
            labels_ref_name = pathlib.Path(self.path_ref).name
            labels_ref_out = labels_ref_name
            if self.prefix:
                labels_ref_out = libmag.combine_paths(
                    self.prefix, labels_ref_name, check_dir=True)
                labels_ref_name = pathlib.Path(labels_ref_out).name
            libmag.copy_backup(self.path_ref, labels_ref_out)
        
        # save metadata as YAML file
        meta = {
            # reference filename is relative to output directory
            "path_ref": labels_ref_name,
            "region_ids_orig": self.region_ids_orig,
        }
        yaml_io.save_yaml(self.save_path, meta, True, True)

    def load(self):
        """Load metadata from a YAML file."""
        # load from save location
        meta_path = pathlib.Path(self.save_path)
        meta = None
        if meta_path.is_file():
            # load YAML file from save location
            loaded = yaml_io.load_yaml(str(meta_path))
            if loaded:
                # get first YAML document
                meta = loaded[0]
                _logger.debug("Loaded labels metadata from: %s", meta_path)
                _logger.debug(meta)
                
                path_ref = meta["path_ref"]
                if path_ref:
                    # path ref is relative to metadata file; make absolute
                    path_ref = str(
                        pathlib.Path(self.save_path).parent / path_ref)
                
                # set attributes to loaded metadata
                self.path_ref = path_ref
                self.region_ids_orig = meta["region_ids_orig"]
        if not meta:
            _logger.debug("Unable to load labels metadata from '%s'", meta_path)
        return self

