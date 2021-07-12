from enum import Enum, auto
import pathlib

from magmap.io import libmag, yaml_io
from magmap.settings import config

PATH_LABELS_META: str = "meta_labels.yml"

_logger = config.logger.getChild(__name__)


class LabelsMetaNames(Enum):
    """Labels image metadata enumerations."""
    PATH_REF = auto()
    REGION_IDS_ORIG = auto()


class LabelsMeta(dict):
    def __init__(self, save_dir=None):
        super().__init__()
        self.update({n: None for n in LabelsMetaNames})
        if save_dir is None:
            save_dir = "."
        self.save_dir = save_dir
        
        self.path_ref = None
        self.save_path = None
    
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
        self.path_ref = self[LabelsMetaNames.PATH_REF]
        if self.path_ref:
            # if provided, copy labels file to import directory
            labels_ref_out = pathlib.Path(self.path_ref).name
            libmag.copy_backup(
                self.path_ref, str(pathlib.Path(self.save_dir, labels_ref_out)))
        self[LabelsMetaNames.PATH_REF] = labels_ref_out
        yaml_io.save_yaml(self.save_path, self, True, True)

    def load(self):
        meta_path = pathlib.Path(self.save_path)
        meta = None
        if meta_path.exists():
            loaded = yaml_io.load_yaml(str(meta_path), {
                LabelsMetaNames.__name__: LabelsMetaNames})
            _logger.debug("Loaded labels metadata from: %s", meta_path)
            if loaded:
                meta = loaded[0]
                self.update(meta)
                print(meta)
                self.path_ref = self[LabelsMetaNames.PATH_REF]
        if not meta:
            _logger.debug("Unable to load labels metadata from '%s'", meta_path)
        return self

