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

    def update_img_display(self, minimum=None, maximum=None, brightness=None,
                           contrast=None, alpha=None):
        """Update the displayed image settings.
        
        Args:
            minimum (float): Minimum intensity.
            maximum (float): Maximum intensity.
            brightness (float): Brightness gamma.
            contrast (float): Contrast factor.
            alpha (float): Opacity, from 0-1, where 1 is fully opaque.

        Returns:

        """
        if self.surfaces:
            for surface in self.surfaces:
                if alpha is not None:
                    surface.actor.property.opacity = alpha
