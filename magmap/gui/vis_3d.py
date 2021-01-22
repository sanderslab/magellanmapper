# 3D visualization in MagellanMapper

class Vis3D:
    """3D visualization object for handling Mayavi/VTK tasks.
    
    Attributes:
        surfaces (list): List of Mayavi surfaces for each displayed channel;
            defaults to None.
    
    """
    def __init__(self):
        """Initialize a 3D visualization object."""
        self.surfaces = None

