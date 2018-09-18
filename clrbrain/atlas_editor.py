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

class AtlasEditor:
    def __init__(self, image5d, labels_img, channel, offset, fn_close_listener):
        """Plot ROI as sequence of z-planes containing only the ROI itself.
        
        Args:
            image5d: Numpy image array in t,z,y,x,c format.
            channel: Channel of the image to display.
            offset: Index of plane at which to start viewing.
            fn_close_listener: Handle figure close events.
        """
        self.image5d = image5d
        self.labels_img = labels_img
        self.channel = channel
        self.offset = offset
        self.fn_close_listener = fn_close_listener
        
        self.plot_eds = {}
        self.alpha_slider = None
        self.alpha_reset_btn = None
        
    def show_atlas(self):
        # set up the figure
        fig = plt.figure()
        gs = gridspec.GridSpec(
            2, 1, wspace=0.1, hspace=0.1, height_ratios=(20, 1))
        gs_viewers = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=gs[0, 0])
        
        # set up the image to display
        colormaps = config.process_settings["channel_colors"]
        cmap_labels, norm = plot_support.get_labels_colormap(
            self.labels_img, 0, 255, False, 150, 50, (0, (0, 0, 0, 255)))
        coord = list(self.offset[::-1])
        
        # transparency controls
        gs_controls = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs[1, 0], width_ratios=(5, 1))
        ax_alpha = plt.subplot(gs_controls[0, 0])
        self.alpha_slider = Slider(
            ax_alpha, "Opacity", 0.0, 1.0, 
            valinit=plot_editor.PlotEditor.ALPHA_DEFAULT)
        ax_alpha_reset = plt.subplot(gs_controls[0, 1])
        self.alpha_reset_btn = Button(ax_alpha_reset, "Reset", hovercolor="0.5")
    
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
            arrs_3d, arrs_1d, aspect, origin = plot_support.transpose_images(
                plane, [self.image5d[0], self.labels_img], 
                [config.labels_scaling])
            img3d_transposed = arrs_3d[0]
            labels_img_transposed = arrs_3d[1]
            scaling = arrs_1d[0]
            
            # slider through image planes
            ax_scroll = plt.subplot(gs_plot[0, 0])
            plane_slider = Slider(
                ax_scroll, plot_support.get_plane_axis(plane), 0, 
                len(img3d_transposed) - 1, valfmt="%d", valinit=0, valstep=1)
            
            # plot editor
            plot_ed = plot_editor.PlotEditor(
                ax, img3d_transposed, labels_img_transposed, cmap_labels, norm, 
                plane, aspect, origin, self.update_coords, self.refresh_images, 
                scaling, plane_slider)
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
        
        # initialize planes in all plot editors
        self.update_coords(coord, config.PLANE[0])
        
        # extra padding for slider labels
        gs.tight_layout(fig, rect=[0.1, 0, 0.9, 1])
        plt.ion()
        plt.show()
        
    def update_coords(self, coord, plane_src):
        coord_rev = lib_clrbrain.transpose_1d_rev(list(coord), plane_src)
        for plane in config.PLANE:
            #if plane != plane_src:
            coord_transposed = lib_clrbrain.transpose_1d(list(coord_rev), plane)
            #print("xy coord: {}, {} coord: {}".format(coord_rev, plane, coord_transposed))
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
    

if __name__ == "__main__":
    print("Starting atlas editor")
