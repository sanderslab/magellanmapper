"""Image viewer support"""
from typing import List, Optional, Sequence, TYPE_CHECKING

from magmap.settings import config

if TYPE_CHECKING:
    from matplotlib import artist, backend_bases, figure

_logger = config.logger.getChild(__name__)


# inspired by Matplotlib example:
# https://matplotlib.org/stable/tutorials/advanced/blitting.html#class-based-example
class Blitter:
    """Controller for blitting in Matplotlib graphics.
    
    Improves interactive graphics performance by reducing repetitive drawing.
    
    """
    def __init__(self, fig, artists=None):
        """Initialized the blit controller."""
        #: Matplotlib figure.
        self.fig: "figure.Figure" = fig
        #: Internal representation of tracked artists.
        self._artists = []
        self.artists = artists
        #: Canvas background.
        self._bkgd = None
        
        #: Event listener IDs.
        self._listeners: List[int] = [
            fig.canvas.mpl_connect("draw_event", self.on_draw)
        ]
    
    @property
    def artists(self) -> List["artist.Artist"]:
        """Tracked artists for blitting."""
        return self._artists
    
    @artists.setter
    def artists(self, vals: Optional[Sequence["artist.Artist"]]):
        """Set tracked artists.
        
        Args:
            vals: Artists to add. Can be None, which will reset artists to an
                empty list.

        """
        if vals is None:
            # reset artists to empty list
            self._artists = []
            return
        
        for val in vals:
            # add artist
            self.add_artist(val)
    
    def add_artist(self, arist: "artist.Artist"):
        """Add tracked artist.
        
        Args:
            arist: Artist to track. Only artists in :attr:`fig` will be added.

        Returns:

        """
        if arist.figure != self.fig:
            # skip artists in other figs
            _logger.warn(
                f"Artist from different figure added for blitting: {arist}")
            return
        
        # flag as animated for update, which appears to prevent updating
        # non-animated artists as well
        arist.set_animated(True)
        self._artists.append(arist)
    
    def on_draw(self, evt: Optional["backend_bases.DrawEvent"]):
        """Recapture backgrouna and draw the figure canvas."""
        canvas = self.fig.canvas
        if evt and evt.canvas != canvas:
            # skip drawing if event is from another canvas
            return
        
        # recapture the canvas background and draw artists
        self._bkgd = canvas.copy_from_bbox(self.fig.bbox)
        self._draw_artists()
    
    def _draw_artists(self):
        """Draw arists."""
        for art in self.artists:
            self.fig.draw_artist(art)
    
    def update(self):
        """Update the canvas."""
        canvas = self.fig.canvas
        if self._bkgd is None:
            # get background if not yet set
            self.on_draw(None)
        else:
            # restore saved background, draw artists, and copy to GUI state
            canvas.restore_region(self._bkgd)
            self._draw_artists()
            canvas.blit(self.fig.bbox)
        
        # flush GUI events and repaint if necessary
        canvas.flush_events()
