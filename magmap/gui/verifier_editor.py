# Viewer for verifying blobs
"""Blob verifier viewer GUI."""

from typing import Optional

from matplotlib import pyplot as plt
from matplotlib import figure
from matplotlib import gridspec

from magmap.cv import detector
from magmap.gui import plot_editor
from magmap.io import libmag
from magmap.plot import plot_3d, plot_support
from magmap.settings import config


class VerifierEditor(plot_support.ImageSyncMixin):
    """Editor to verify blobs."""

    def __init__(self, img5d, blobs, title=None, fig=None):
        """Initialize the viewer."""
        super().__init__(img5d)
        self.blobs: "detector.Blobs" = blobs
        self.title: Optional[str] = title
        self.fig: Optional[figure.Figure] = fig
        
    def show_fig(self):
        """Set up the figure."""
        # set up the figure
        if self.fig is None:
            fig = figure.Figure(self.title)
            self.fig = fig
        else:
            fig = self.fig
        fig.clear()
        nrows = 3
        ncols = 3
        gs = gridspec.GridSpec(
            nrows, ncols, wspace=0.1, hspace=0.1, figure=fig,
            left=0.06, right=0.94, bottom=0.02, top=0.98)
        
        # get blobs with confirmation flags
        blobs = self.blobs.blobs
        blobs = blobs[self.blobs.get_blob_confirmed(blobs) >= 0]
        nblobs = len(blobs)
        offsets = self.blobs.get_blob_abs_coords(blobs).astype(int)
        subimg_shape = (50, 50)
        for row in range(nrows):
            for col in range(ncols):
                # add axes
                ax = fig.add_subplot(gs[row, col])
                plot_support.hide_axes(ax)
                aspect, origin = plot_support.get_aspect_ratio(config.PLANE[0])
                
                # get offset from blob's absolute coordinates
                n = row * ncols + col
                if n >= nblobs:
                    break
                offset = offsets[n]
        
                # display plot editor centered on blob
                overlayer = plot_support.ImageOverlayer(
                    ax, aspect, origin, rgb=config.rgb)
                plot_ed = plot_editor.PlotEditor(
                    overlayer, self.img5d.img[0], None, None)
                plot_ed.coord = offset
                plot_ed.show_overview()
                offset_ctr = plot_3d.roi_center_to_offset(
                    offset[1:], subimg_shape)
                plot_ed.view_subimg(offset_ctr, subimg_shape)
                self.plot_eds[n] = plot_ed
        
        # attach listeners
        fig.canvas.mpl_connect("close_event", self.on_close)
        
        plt.ion()  # avoid the need for draw calls
        self.fig.canvas.draw_idle()
