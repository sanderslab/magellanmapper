import pathlib

from magmap.io import libmag, yaml_io
from magmap.settings import config


_logger = config.logger.getChild(__name__)


class LabelsMeta:
    PATH_LABELS_META: str = "meta_labels.yml"
    
    def __init__(self, prefix=None):
        super().__init__()
        self.prefix = prefix
        self.save_path = None
        
        self.path_ref = None
        self.region_ids_orig = None
    
    @property
    def save_path(self):
        if not self._save_path:
            if self.prefix:
                return libmag.combine_paths(
                    self.prefix, self.PATH_LABELS_META, check_dir=True)
            return self.PATH_LABELS_META
        return self._save_path
    
    @save_path.setter
    def save_path(self, val):
        self._save_path = val
    
    def save(self):
        labels_ref_name = None
        if self.path_ref:
            # if provided, copy labels file to import directory
            labels_ref_name = pathlib.Path(self.path_ref).name
            labels_ref_out = labels_ref_name
            if self.prefix:
                labels_ref_out = libmag.combine_paths(
                    self.prefix, labels_ref_name, check_dir=True)
                labels_ref_name = pathlib.Path(labels_ref_out).name
            libmag.copy_backup(self.path_ref, labels_ref_out)
        meta = {
            "path_ref": labels_ref_name,
            "region_ids_orig": self.region_ids_orig,
        }
        yaml_io.save_yaml(self.save_path, meta, True, True)

    def load(self):
        meta_path = pathlib.Path(self.save_path)
        meta = None
        if meta_path.exists():
            loaded = yaml_io.load_yaml(str(meta_path))
            _logger.debug("Loaded labels metadata from: %s", meta_path)
            if loaded:
                meta = loaded[0]
                print(meta)
                path_ref = meta["path_ref"]
                if path_ref and self.prefix:
                    prefix = pathlib.Path(self.prefix)
                    if not prefix.is_dir():
                        prefix = prefix.parent
                    path_ref = str(prefix / path_ref)
                self.path_ref = path_ref
                self.region_ids_orig = meta["region_ids_orig"]
        if not meta:
            _logger.debug("Unable to load labels metadata from '%s'", meta_path)
        return self

