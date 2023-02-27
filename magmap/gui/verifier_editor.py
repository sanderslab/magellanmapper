# Viewer for verifying blobs
"""Blob verifier viewer GUI."""

import dataclasses
from typing import Optional, Dict, Sequence, Any, Callable

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import figure
from matplotlib import gridspec
from matplotlib.widgets import Button, Slider, TextBox

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
        
        # GUI elements
        self._grid_spec = None
        self._row_slider = None
        self._col_slider = None
        self._back_btn = None
        self._next_btn = None
        self._page_txt = None
        
        # blob views grid layout
        self._nrows: int = 3
        self._ncols: int = 3
        
        #: Available blob flags, sorted in ascending order.
        self._blob_flags: Sequence[Any] = []
        #: Dictionary of sub-plot index to data object for the plot.
        self._blob_views: Dict[int, "VerifierEditor.BlobView"] = {}
        #: Mask of blobs to show.
        self._blobs_show: Sequence = []
        #: Starting index of blob to show in current page.
        self._blob_offset: int = 0
        
    def show_fig(self):
        """Set up the figure."""
        # set up the figure and main layout
        if self.fig is None:
            self.fig = figure.Figure(self.title)
        self.fig.clear()
        self._grid_spec = gridspec.GridSpec(
            2, 1, wspace=0.1, hspace=0.1, height_ratios=(20, 1),
            figure=self.fig, left=0.06, right=0.94, bottom=0.02, top=0.96)
        
        # add controls with sub-grid-spec that includes gaps to accommodate
        # labels, which otherwise overlap other controls
        gs_controls = gridspec.GridSpecFromSubplotSpec(
            1, 8, subplot_spec=self._grid_spec[1, 0], wspace=0.1,
            width_ratios=(30, 5, 30, 3, 11, 11, 3, 7))
        self._row_slider = Slider(
            self.fig.add_subplot(gs_controls[0, 0]), "Rows", 0, 10,
            valinit=self._nrows, valstep=1, valfmt="%d")
        self._col_slider = Slider(
            self.fig.add_subplot(gs_controls[0, 2]), "Cols", 0, 10,
            valinit=self._ncols, valstep=1, valfmt="%d")
        self._back_btn = Button(
            self.fig.add_subplot(gs_controls[0, 4]), "Back")
        self._next_btn = Button(
            self.fig.add_subplot(gs_controls[0, 5]), "Next")
        self._page_txt = TextBox(
            self.fig.add_subplot(gs_controls[0, 7]), "Page", "1",
            label_pad=0.05)
        
        for btn in (self._back_btn, self._next_btn, self._page_txt):
            # enable button and color theme
            self.enable_btn(btn)
        
        # set up and display blobs
        self._setup_blobs()
        self.show_views()
        
        # attach listeners
        self._listeners.append(self.fig.canvas.mpl_connect(
            "button_press_event", self._on_mouse_press))
        self._listeners.append(self.fig.canvas.mpl_connect(
            "close_event", self.on_close))
        
        # attach handlers
        self._row_slider.on_changed(self._change_row)
        self._col_slider.on_changed(self._change_col)
        self._back_btn.on_clicked(self._back_page)
        self._next_btn.on_clicked(self._next_page)
        self._page_txt.on_submit(self._select_page)
        
        plt.ion()  # avoid the need for draw calls
        self.fig.canvas.draw_idle()
    
    def _setup_blobs(self):
        """Set up blobs."""
        blobs = self.blobs.blobs
        if blobs is None:
            # reset blobs setup
            self._blobs_show = []
            self._blob_flags = []
        else:
            # get blobs with confirmation flags set by user (ie non-neg)
            self._blobs_show = self.blobs.get_blob_confirmed(blobs) >= 0
            self._blob_flags = sorted(np.unique(
                self.blobs.get_blob_confirmed(
                    blobs[self._blobs_show]).astype(int)))
    
    def show_views(self):
        """Show blob views."""
        
        # clear all prior axes
        for view in self._blob_views.values():
            view.plot_ed.axes.clear()
        
        # set up grid spect in main view area
        gs_viewers = gridspec.GridSpecFromSubplotSpec(
            self._nrows, self._ncols, subplot_spec=self._grid_spec[0, 0])

        # get indices of these blobs to access by view rather than copy
        blobs_inds = np.argwhere(self._blobs_show)
        nblobs = len(blobs_inds)
        subimg_shape = (50, 50)
        for row in range(self._nrows):
            for col in range(self._ncols):
                n = row * self._ncols + col + self._blob_offset
                if n >= nblobs:
                    break
                
                # add axes
                ax = self.fig.add_subplot(gs_viewers[row, col])
                plot_support.hide_axes(ax)
                aspect, origin = plot_support.get_aspect_ratio(config.PLANE[0])

                # display plot editor centered on blob
                overlayer = plot_support.ImageOverlayer(
                    ax, aspect, origin, rgb=config.rgb,
                    additive_blend=self.additive_blend)
                plot_ed = plot_editor.PlotEditor(
                    overlayer, self.img5d.img[0], None, None)
                
                # get blob as view and use absolute coordinates as ROI offset
                blob = self.blobs.blobs[blobs_inds[n][0]]
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
    
    def _on_mouse_press(self, evt):
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
            
            # update the blob's flag and show in axes title, which will update
            # the underlying blobs array since this blob is a view, not a copy
            self.blobs.set_blob_confirmed(view.blob, self._blob_flags[i])
            self._set_ax_title(view)
            
            if self.fn_update_blob:
                # handle any blob changes with the same updated blob as both
                # new and old blobs since blobs array has already been updated
                self.fn_update_blob(view.blob, view.blob)
            
    def _set_ax_title(self, view: "BlobView"):
        """Set the axes title for a blob view.
        
        Args:
            view: Blob view.

        """
        # show the blob's confirmed flag in the title
        view.plot_ed.axes.set_title(
            f"Class: {self.blobs.get_blob_confirmed(view.blob).astype(int)}")

    def _change_row(self, evt):
        """Handle change to the number of rows."""
        self._nrows = int(evt)
        self.show_views()
    
    def _change_col(self, evt):
        """Handle change to the number of cols."""
        self._ncols = int(evt)
        self.show_views()
    
    def _back_page(self, evt):
        """Scroll back one page of views."""
        if self._blob_offset == 0: return
        self._blob_offset -= self._nrows * self._ncols
        if self._blob_offset < 0:
            self._blob_offset = 0
        self.show_views()

    def _next_page(self, evt):
        """Scroll forward one page of views."""
        nblobs = np.sum(self._blobs_show)
        offset = self._blob_offset + self._nrows * self._ncols
        if offset >= nblobs: return
        self._blob_offset = offset
        print("offset:", self._blob_offset)
        self.show_views()
    
    def _select_page(self, text):
        """Handle selecting a page of blob views."""
        # only accept numbers and convert floats to ints
        if not libmag.is_number(text): return
        page = int(float(text))
        
        # limit to max page
        nblobs = np.sum(self._blobs_show)
        nblobs_per_page = self._nrows * self._ncols
        npages = np.ceil(nblobs / nblobs_per_page).astype(int)
        if page > npages:
            page = npages
        
        # convert page to offset
        offset = (page - 1) * nblobs_per_page
        if offset < 0:
            offset = 0
            page = 0
        
        # update displayed page
        self._page_txt.set_val(page)
        self._blob_offset = offset
        self.show_views()
    
