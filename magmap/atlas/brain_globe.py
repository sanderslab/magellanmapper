"""BrainGlobe integration in MageallanMapper"""

from requests.exceptions import ConnectionError
import shutil
from typing import Dict

from bg_atlasapi import list_atlases, bg_atlas

from magmap.settings import config

_logger = config.logger.getChild(__name__)


class BrainGlobeMM:
    def __init__(self):
        self.atlases_avail: Dict[str, str] = {}
        self.atlases_local: Dict[str, str] = {}
    
    def get_avail_atlases(self):
        try:
            self.atlases_avail = list_atlases.get_all_atlases_lastversions()
        except ConnectionError:
            _logger.warn("Unable to get BrainGlobe available atlases")
        return self.atlases_avail
    
    def get_local_atlases(self):
        self.atlases_local = {
            a: list_atlases.get_local_atlas_version(a)
            for a in list_atlases.get_downloaded_atlases()
        }
        return self.atlases_local
    
    def get_atlas(self, name, download=True):
        if not download and name not in self.atlases_local:
            return None
        atlas = bg_atlas.BrainGlobeAtlas(name)
        return atlas
    
    def remove_local_atlas(self, name):
        atlas = self.get_atlas(name, False)
        print("atlas to remove", name, atlas)
        if not atlas:
            _logger.warn("'%s' atlas not found", name)
            return
        try:
            print("path", atlas.root_dir)
            if atlas.root_dir.is_dir():
                print("removing")
                shutil.rmtree(atlas.root_dir)
        except FileExistsError as e:
            _logger.warn(e)
