import pathlib

from magmap.io import libmag, yaml_io
from magmap.settings import config

PATH_LABELS_META: str = "meta_labels.yml"

_logger = config.logger.getChild(__name__)


class LabelsMeta:
    def __init__(self, save_dir=None):
        super().__init__()
        if save_dir is None:
            save_dir = "."
        self.save_dir = save_dir
        self.save_path = None
        
        self.path_ref = None
        self.region_ids_orig = None
    
    @property
    def save_path(self):
        if not self._save_path:
            return str(pathlib.Path(self.save_dir) / PATH_LABELS_META)
        return self._save_path
    
    @save_path.setter
    def save_path(self, val):
        self._save_path = val
    
    def save(self):
        labels_ref_out = None
        if self.path_ref:
            # if provided, copy labels file to import directory
            labels_ref_out = pathlib.Path(self.path_ref).name
            libmag.copy_backup(
                self.path_ref, str(pathlib.Path(self.save_dir, labels_ref_out)))
        meta = {
            "path_ref": labels_ref_out,
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
                if path_ref:
                    path_ref = pathlib.Path(path_ref)
                    if not path_ref.is_absolute():
                        path_ref = self.save_dir / path_ref
                    path_ref = str(path_ref)
                self.path_ref = path_ref
                self.region_ids_orig = meta["region_ids_orig"]
        if not meta:
            _logger.debug("Unable to load labels metadata from '%s'", meta_path)
        return self

