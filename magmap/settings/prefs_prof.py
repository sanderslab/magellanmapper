"""Preferences profile"""

from magmap.settings import profiles


class PrefsProfile(profiles.SettingsDict):
    def __init__(self, *args, **kwargs):
        """Initialize a preferences profile dictionary.
        
        Args:
            *args: 
            **kwargs: 
        """
        super().__init__(self)
        self[self.NAME_KEY] = self.DEFAULT_NAME
        
        self["fig_save_dir"] = ""
        
        # update with args
        self.update(*args, **kwargs)
