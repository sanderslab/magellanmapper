"""BrainGlobe model for integration in MagellanMapper"""

from requests.exceptions import ChunkedEncodingError, ConnectionError
import shutil
from typing import Callable, Dict, Optional

from bg_atlasapi import bg_atlas, descriptors, list_atlases

from magmap.settings import config

_logger = config.logger.getChild(__name__)


class BrainGlobeMM:
    """Model for BrainGlobe-MagellanMapper interactions.
    
    Attributes:
        atlases_avail: Dictionary of the names of available atlases from
            BrainGlobe to their latest version string.
        atlases_local: Dictionary of the names of locally downloaded BrainGlobe
            atlases to their version string.
    
    """
    def __init__(self):
        self.atlases_avail: Dict[str, str] = {}
        self.atlases_local: Dict[str, str] = {}
    
    def get_avail_atlases(self) -> Dict[str, str]:
        """Fetch the available atlases from BrainGlobe.
        
        Returns:
            Dictionary of the names of available atlases from BrainGlobe to
            their latest version string.

        """
        try:
            self.atlases_avail = list_atlases.get_all_atlases_lastversions()
        except ConnectionError:
            _logger.warn("Unable to get BrainGlobe available atlases")
        return self.atlases_avail
    
    def get_local_atlases(self) -> Dict[str, str]:
        """Get local, downloaded BrainGlobe atlases.
        
        Returns:
            Dictionary of the names of locally downloaded BrainGlobe atlases
            to their version string.

        """
        self.atlases_local = {
            a: list_atlases.get_local_atlas_version(a)
            for a in list_atlases.get_downloaded_atlases()
        }
        return self.atlases_local
    
    def get_atlas(
            self, name: str, download: bool = True,
            fn_update: Callable[[int, int], None] = None
    ) -> Optional[bg_atlas.BrainGlobeAtlas]:
        """Get a BrainGlobe atlas.
        
        Args:
            name: Name of atlas to retrieve.
            download: True to download the atlas if not available locally;
                False to return None if the atlas is not present.
            fn_update: Handler for progress updates during atlas download.

        Returns:
            The BrainGlobe atlas instance. None if the atlas could not be
            found or downloaded.

        """
        if not download and name not in self.atlases_local:
            return None
        
        try:
            try:
                # get atlas with progress handler using bg-atlasapi > v1.0.2
                atlas = bg_atlas.BrainGlobeAtlas(name, fn_update=fn_update)
            except TypeError:
                # fall back to showing a busy progress indicator
                fn_update(0, 0)
                atlas = bg_atlas.BrainGlobeAtlas(name)
        except ChunkedEncodingError as e:
            # download error
            _logger.error("Error downloading atlas: %s", e)
            return None
        
        # add an attribute to store the path to the structures file
        atlas.structures_path = atlas.root_dir / descriptors.STRUCTURES_FILENAME
        
        return atlas
    
    def remove_local_atlas(self, name: str):
        """Remove local copy of downloaded BrainGlobe atlas.
        
        Args:
            name: Name of atlas to remove

        """
        atlas = self.get_atlas(name, False)
        if not atlas:
            _logger.warn("'%s' atlas not found", name)
            return
        try:
            if atlas.root_dir.is_dir():
                shutil.rmtree(atlas.root_dir)
                _logger.debug(
                    "Removed '%s' atlas from '%s'", name, atlas.root_dir)
        except FileExistsError as e:
            _logger.warn(e)
