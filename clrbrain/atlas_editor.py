#!/bin/bash
# Atlas Editor with orthogonal viewing
# Author: David Young, 2018
"""Atlas editing GUI in the Clrbrain package.
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button

from clrbrain import config
from clrbrain import lib_clrbrain
from clrbrain import plot_editor
from clrbrain import plot_support
from clrbrain import plot_3d

class AtlasEditor:
    def __init__(self, image5d, labels_img, channel, offset, fn_close_listener, 
                 borders_img=None, fn_show_label_3d=None):
        """Plot ROI as sequence of z-planes containing only the ROI itself.
        
        Args:
            image5d: Numpy image array in t,z,y,x,[c] format.
            channel: Channel of the image to display.
            offset: Index of plane at which to start viewing in x,y,z (user) 
                order.
            fn_close_listener: Handle figure close events.
            borders_img: Numpy image array in z,y,x,c format to show label 
                borders, such as that generated during label smoothing. 
                Defaults to None.
            fn_show_label_3d: Function to call to show a label in a 
                3D viewer. Defaults to None.
        """
        self.image5d = image5d
        self.labels_img = labels_img
        self.channel = channel
        self.offset = offset
        self.fn_close_listener = fn_close_listener
        self.borders_img = borders_img
        self.fn_show_label_3d = fn_show_label_3d
        
        self.plot_eds = {}
        self.alpha_slider = None
        self.alpha_reset_btn = None
        self.interp_planes = InterpolatePlanes()
        self.interp_btn = None
        
    def show_atlas(self):
        # set up the figure
        fig = plt.figure()
        gs = gridspec.GridSpec(
            2, 1, wspace=0.1, hspace=0.1, height_ratios=(20, 1))
        gs_viewers = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=gs[0, 0])
        
        # set up the image to display
        colormaps = config.process_settings["channel_colors"]
        cmap_labels = plot_support.DiscreteColormap(
            self.labels_img, 0, 255, False, 150, 50, (0, (0, 0, 0, 0)))
        cmap_borders = None
        if self.borders_img is not None:
            # brightest colormap for original (channel 0) borders
            cmap_borders = [
                cmap_labels.modified_cmap(int(40 / (channel + 1)))
                for channel in range(self.borders_img.shape[-1])]
        coord = list(self.offset[::-1])
        
        # transparency controls
        gs_controls = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=gs[1, 0], width_ratios=(4, 1, 1))
        ax_alpha = plt.subplot(gs_controls[0, 0])
        self.alpha_slider = Slider(
            ax_alpha, "Opacity", 0.0, 1.0, 
            valinit=plot_editor.PlotEditor.ALPHA_DEFAULT)
        ax_alpha_reset = plt.subplot(gs_controls[0, 1])
        self.alpha_reset_btn = Button(ax_alpha_reset, "Reset", hovercolor="0.5")
        ax_interp = plt.subplot(gs_controls[0, 2])
        self.interp_btn = Button(ax_interp, "Interpolate", hovercolor="0.5")
    
        def setup_plot_ed(plane, gs_spec):
            # subplot grid, with larger height preference for plot for 
            # each increased row to make sliders of approx equal size and  
            # align top borders of top images
            rows_cols = gs_spec.get_rows_columns()
            extra_rows = rows_cols[3] - rows_cols[2]
            gs_plot = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=gs_spec, 
                height_ratios=(1, 10 + 14 * extra_rows), 
                hspace=0.1/(extra_rows*1.4+1))
            
            # image plot with arrays transformed to this editor's 
            # orthogonal direction
            ax = plt.subplot(gs_plot[1, 0])
            plot_support.hide_axes(ax)
            max_size = plot_support.max_plane(self.image5d[0], plane)
            arrs_3d = [self.image5d[0], self.labels_img]
            if self.borders_img is not None:
                # overlay borders if available
                arrs_3d.append(self.borders_img)
            arrs_3d, arrs_1d, aspect, origin = plot_support.transpose_images(
                plane, arrs_3d, [config.labels_scaling])
            img3d_transposed = arrs_3d[0]
            labels_img_transposed = arrs_3d[1]
            borders_img_transposed = None
            if len(arrs_3d) >= 3:
                borders_img_transposed = arrs_3d[2]
            scaling = arrs_1d[0]
            
            # slider through image planes
            ax_scroll = plt.subplot(gs_plot[0, 0])
            plane_slider = Slider(
                ax_scroll, plot_support.get_plane_axis(plane), 0, 
                len(img3d_transposed) - 1, valfmt="%d", valinit=0, valstep=1)
            
            # plot editor
            plot_ed = plot_editor.PlotEditor(
                ax, img3d_transposed, labels_img_transposed, cmap_labels, 
                plane, aspect, origin, self.update_coords, self.refresh_images, 
                scaling, plane_slider, img3d_borders=borders_img_transposed, 
                cmap_borders=cmap_borders, 
                fn_show_label_3d=self.fn_show_label_3d, 
                interp_planes=self.interp_planes)
            return plot_ed
        
        # setup plot editor for all 3 orthogonal directions
        self.plot_eds[config.PLANE[0]] = setup_plot_ed(
            config.PLANE[0], gs_viewers[:2, 0])
        self.plot_eds[config.PLANE[1]] = setup_plot_ed(
            config.PLANE[1], gs_viewers[0, 1])
        self.plot_eds[config.PLANE[2]] = setup_plot_ed(
            config.PLANE[2], gs_viewers[1, 1])
        
        # attach listeners
        fig.canvas.mpl_connect("scroll_event", self.scroll_overview)
        fig.canvas.mpl_connect("key_press_event", self.scroll_overview)
        fig.canvas.mpl_connect("close_event", self.fn_close_listener)
        fig.canvas.mpl_connect("axes_leave_event", self.axes_exit)
        
        self.alpha_slider.on_changed(self.alpha_update)
        self.alpha_reset_btn.on_clicked(self.alpha_reset)
        self.interp_btn.on_clicked(self.interpolate)
        
        # initialize planes in all plot editors
        self.update_coords(coord, config.PLANE[0])
        
        # extra padding for slider labels
        gs.tight_layout(fig)
        plt.ion()
        plt.show()
        
    def update_coords(self, coord, plane_src=config.PLANE[0]):
        """Update all plot editors with given coordinates.
        
        Args:
            coord: Coordinate at which to center images, in z,y,x order.
            plane_src: One of :const:``config.PLANE`` to specify the 
                orientation from which the coordinates were given; defaults 
                to :const:``config.PLANE[0]``.
        """
        coord_rev = lib_clrbrain.transpose_1d_rev(list(coord), plane_src)
        for plane in config.PLANE:
            coord_transposed = lib_clrbrain.transpose_1d(list(coord_rev), plane)
            self.plot_eds[plane].update_coord(coord_transposed)
    
    def refresh_images(self, plot_ed):
        for key in self.plot_eds.keys():
            ed = self.plot_eds[key]
            if ed != plot_ed: ed.update_image()
    
    def scroll_overview(self, event):
        for key in self.plot_eds.keys():
            self.plot_eds[key].scroll_overview(event)
    
    def alpha_update(self, event):
        for key in self.plot_eds.keys():
            self.plot_eds[key].alpha_updater(event)
    
    def alpha_reset(self, event):
        self.alpha_slider.reset()
    
    def axes_exit(self, event):
        for key in self.plot_eds.keys():
            self.plot_eds[key].on_axes_exit(event)
    
    def interpolate(self, event):
        try:
            self.interp_planes.interpolate(self.labels_img)
            self.refresh_images(None)
        except ValueError as e:
            print(e)

class InterpolatePlanes:
    """Track manually edited planes between which to interpolate changes 
    for a given label.
    
    This interpolation replaces unedited planes based on the trends of 
    the edited ones to avoid the need to manually edit every single plane.
    
    Attribtes:
        plane: Plane in which editing has occurred.
        bounds: Unsorted start and end planes.
        label_id: Label ID of the edited region.
    """
    def __init__(self):
        """Initialize ``InterpolatePlanes`` with empty attibutes.
        """
        self.plane = None
        self.bounds = None
        self.label_id = None
    
    def update_plane(self, plane, i, label_id):
        """Update the current plane.
        
        Args:
            plane: Plane direction, which will overwrite any current direction.
            i: Index of the plane to add, which will overwrite the oldest 
                bounds element.
            label_id: ID of label, which will overwrite any current ID.
        """
        if self.plane is not None and (
            plane != self.plane or label_id != self.label_id):
            # reset bounds if new plane or label ID don't match prior settings 
            # and previously set (plane and label_id should have been set 
            # together)
            self.bounds = None
        self.plane = plane
        self.label_id = label_id
        self.bounds = i
        print(self)
        
    def interpolate(self, labels_img):
        """Interpolate between :attr:``bounds`` in the given :attr:``plane`` 
        direction in the bounding box surrounding :attr:``label_id``.
        
        Args:
            labels_img: Labels image as a Numpy array of x,y,z dimensions.
        """
        if not any(self.bounds):
            raise ValueError("boundaries not fully set: {}".format(self.bounds))
        plot_3d.interpolate_between_planes(
            labels_img, self.label_id, config.PLANE.index(self.plane), 
            self.bounds)
    
    def __str__(self):
        return "{}: {} (ID: {})".format(
            plot_support.get_plane_axis(self.plane), self.bounds, self.label_id)
    
    @property
    def bounds(self):
        return self._bounds
    
    @bounds.setter
    def bounds(self, val):
        if val is None:
            self._bounds = [None, None]
        elif isinstance(val, int):
            self._bounds.append(val)
            del self._bounds[0]
        else:
            self._bounds = val

if __name__ == "__main__":
    print("Starting atlas editor")
