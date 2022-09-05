# Viewer for verifying blobs
"""Blob verifier viewer GUI."""

import dataclasses
from typing import Optional, Dict, Sequence, Any, Callable

import numpy as np

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
    
    @dataclasses.dataclass
    class BlobView:
        """Storage class for each view."""
        #: Plot Editor.
        plot_ed: "plot_editor.PlotEditor"
        #: Displayed blob.
        blob: np.ndarray
    
    def __init__(self, img5d, blobs, title=None, fig=None, fn_update_blob=None):
        """Initialize the viewer."""
        super().__init__(img5d)
        self.blobs: "detector.Blobs" = blobs
        self.title: Optional[str] = title
        self.fig: Optional[figure.Figure] = fig
        #: Handler for updating a blob; defaults to None.
        self.fn_update_blob: Optional[Callable[
            [np.ndarray, Optional[np.ndarray]], np.ndarray]] = fn_update_blob
        
        #: Available blob flags, sorted in ascending order.
        self._blob_flags: Sequence[Any] = []
        #: Dictionary of sub-plot index to data object for the plot.
        self._blob_views: Dict[int, "VerifierEditor.BlobView"] = {}
        
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
        
        # get blobs with confirmation flags set by user (ie non-neg)
        blobs = self.blobs.blobs
        blobs_mask = self.blobs.get_blob_confirmed(blobs) >= 0
        self._blob_flags = sorted(np.unique(
            self.blobs.get_blob_confirmed(blobs[blobs_mask]).astype(int)))
        
        # get indices of these blobs to access by view rather than copy
        blobs_inds = np.argwhere(blobs_mask)
        nblobs = len(blobs_inds)
        subimg_shape = (50, 50)
        for row in range(nrows):
            for col in range(ncols):
                n = row * ncols + col
                if n >= nblobs:
                    break
                
                # add axes
                ax = fig.add_subplot(gs[row, col])
                plot_support.hide_axes(ax)
                aspect, origin = plot_support.get_aspect_ratio(config.PLANE[0])

                # display plot editor centered on blob
                overlayer = plot_support.ImageOverlayer(
                    ax, aspect, origin, rgb=config.rgb)
                plot_ed = plot_editor.PlotEditor(
                    overlayer, self.img5d.img[0], None, None)
                
                # get blob as view and use absolute coordinates as ROI offset
                blob = blobs[blobs_inds[n][0]]
                offset = self.blobs.get_blob_abs_coords(blob).astype(int)
                plot_ed.coord = offset
                plot_ed.show_overview()
                
                # center plot on offset
                offset_ctr = plot_3d.roi_center_to_offset(
                    offset[1:], subimg_shape)
                plot_ed.view_subimg(offset_ctr, subimg_shape)
                self.plot_eds[n] = plot_ed
                
                # store plot and blob
                blob_view = self.BlobView(plot_ed, blob)
                self._set_ax_title(blob_view)
                self._blob_views[n] = blob_view
        
        # attach listeners
        fig.canvas.mpl_connect("button_press_event", self.on_btn_press)
        fig.canvas.mpl_connect("close_event", self.on_close)
        
        plt.ion()  # avoid the need for draw calls
        self.fig.canvas.draw_idle()

    def on_btn_press(self, evt):
        """Respond to mouse button press events."""
        for key, view in self._blob_views.items():
            # ignore presses outside the given plot
            if evt.inaxes != view.plot_ed.axes: continue
            
            # get the index of the current blob confirmed flag and increment
            i = np.argwhere(self._blob_flags == self.blobs.get_blob_confirmed(
                view.blob))[0][0] + 1
            if i >= len(self._blob_flags):
                # reset if the index exceeds the flag list
                i = 0
            
            # update the blob's flag and show in axes title
            blob_orig = np.copy(view.blob)
            self.blobs.set_blob_confirmed(view.blob, self._blob_flags[i])
            self._set_ax_title(view)
            
            if self.fn_update_blob:
                # handle any blob changes
                self.fn_update_blob(view.blob, blob_orig)
            
    def _set_ax_title(self, view: "BlobView"):
        """Set the axes title for a blob view.
        
        Args:
            view: Blob view.

        """
        # show the blob's confirmed flag in the title
        view.plot_ed.axes.set_title(
            f"Class: {self.blobs.get_blob_confirmed(view.blob).astype(int)}")
    
