"""Preferences profile"""

import dataclasses

from magmap.settings import profiles


@dataclasses.dataclass
class PrefsProfile(profiles.SettingsDict):
    """Application preferences profile."""
    
    #: ROI Editor circles style.
    roi_circles: str = ""
    
    #: ROI Editor planes selection.
    roi_plane: str = ""
    
    #: ROI Editor styles selection.
    roi_styles: str = ""
    
    #: Figure save directory path.
    fig_save_dir: str = ""
    
    def __init__(self, *args, **kwargs):
        """Initialize a preferences profile dictionary.
        
        Args:
            *args: 
            **kwargs: 
        """
        super().__init__(self, *args, **kwargs)
