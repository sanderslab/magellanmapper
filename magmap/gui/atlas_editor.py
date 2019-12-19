#!/bin/bash
# Atlas Editor with orthogonal viewing
# Author: David Young, 2018, 2019
"""Atlas editing GUI in the MagellanMapper package.
"""

import datetime

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, TextBox

from magmap.plot import colormaps
from magmap.settings import config
from magmap.io import libmag
from magmap.gui import plot_editor
from magmap.plot import plot_support
from magmap.cv import cv_nd
from magmap.io import sitk_io


class AtlasEditor:
    """Graphical interface to view an atlas in multiple orthogonal 
    dimensions and edit atlas labels.
    
    Attributes:
        image5d: Numpy image array in t,z,y,x,[c] format.
        labels_img: Numpy image array in z,y,x format.
        channel: Channel of the image to display.
        offset: Index of plane at which to start viewing in x,y,z (user) 
            order.
        fn_close_listener: Handle figure close events.
        borders_img: Numpy image array in z,y,x,[c] format to show label 
            borders, such as that generated during label smoothing. 
            Defaults to None. If this image has a different number of 
            labels than that of ``labels_img``, a new colormap will 
            be generated.
        fn_show_label_3d: Function to call to show a label in a 
            3D viewer. Defaults to None.
        plot_eds: Dictionary of :class:`magmap.plot_editor.PlotEditor`, with 
            key specified by one of :const:`magmap.config.PLANE` 
            plane orientations.
        alpha_slider: Matplotlib alpha slider control.
        alpha_reset_btn: Maplotlib button for resetting alpha transparency.
        alpha_last: Float specifying the previous alpha value.
        interp_planes: Current :class:`InterpolatePlanes` object.
        interp_btn: Matplotlib button to initiate plane interpolation.
        save_btn: Matplotlib button to save the atlas.
    """

    _EDIT_BTN_LBLS = ("Edit", "Editing")
    
    def __init__(self, image5d, labels_img, channel, offset, fn_close_listener, 
                 borders_img=None, fn_show_label_3d=None):
        """Plot ROI as sequence of z-planes containing only the ROI itself."""
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
        self.alpha_last = None
        self.interp_planes = None
        self.interp_btn = None
        self.save_btn = None
        self.edit_btn = None
        self.color_picker_box = None
        
    def show_atlas(self):
        """Set up the atlas display with multiple orthogonal views."""
        # set up the figure
        fig = plt.figure()
        gs = gridspec.GridSpec(
            2, 1, wspace=0.1, hspace=0.1, height_ratios=(20, 1))
        gs_viewers = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=gs[0, 0])
        
        # set up colormaps, using a labels image to generate a template 
        # for the borders image if it has the same number of colors; ideally 
        # use the original labels for consistent ID-color mapping
        cmap_labels = colormaps.get_labels_discrete_colormap(
            self.labels_img, 0, dup_for_neg=True, use_orig_labels=True)
        cmap_borders = colormaps.get_borders_colormap(
            self.borders_img, self.labels_img, cmap_labels)
        coord = list(self.offset[::-1])
        
        # editor controls, split into a slider sub-spec to allow greater
        # spacing for labels on either side and a separate sub-spec for
        # buttons and other fields
        gs_controls = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs[1, 0], width_ratios=(1, 1),
            wspace=0.15)
        self.alpha_slider = Slider(
            plt.subplot(gs_controls[0, 0]), "Opacity", 0.0, 1.0,
            valinit=plot_editor.PlotEditor.ALPHA_DEFAULT)
        gs_controls_btns = gridspec.GridSpecFromSubplotSpec(
            1, 5, subplot_spec=gs_controls[0, 1], wspace=0.1)
        self.alpha_reset_btn = Button(
            plt.subplot(gs_controls_btns[0, 0]), "Reset")
        self.interp_btn = Button(
            plt.subplot(gs_controls_btns[0, 1]), "Fill Label")
        self.interp_planes = InterpolatePlanes(self.interp_btn)
        self.interp_planes.update_btn()
        self.save_btn = Button(plt.subplot(gs_controls_btns[0, 2]), "Save")
        enable_btn(self.save_btn, False)
        self.edit_btn = Button(plt.subplot(gs_controls_btns[0, 3]), "Edit")
        self.color_picker_box = TextBox(
            plt.subplot(gs_controls_btns[0, 4]), None)
    
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
            plot_support.max_plane(self.image5d[0], plane)
            arrs_3d = [self.image5d[0]]
            if self.labels_img is not None:
                # overlay borders if available
                arrs_3d.append(self.labels_img)
            if self.borders_img is not None:
                # overlay borders if available
                arrs_3d.append(self.borders_img)
            scaling = config.labels_scaling
            if scaling is not None: scaling = [scaling]
            arrs_3d, arrs_1d = plot_support.transpose_images(
                plane, arrs_3d, scaling)
            aspect, origin = plot_support.get_aspect_ratio(plane)
            img3d_transposed = arrs_3d[0]
            labels_img_transposed = None
            if len(arrs_3d) >= 2:
                labels_img_transposed = arrs_3d[1]
            borders_img_transposed = None
            if len(arrs_3d) >= 3:
                borders_img_transposed = arrs_3d[2]
            if arrs_1d is not None and len(arrs_1d) > 0: scaling = arrs_1d[0]
            
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
                interp_planes=self.interp_planes,
                fn_update_intensity=self.update_color_picker)
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
        fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        fig.canvas.mpl_connect("close_event", self.fn_close_listener)
        fig.canvas.mpl_connect("axes_leave_event", self.axes_exit)
        
        self.alpha_slider.on_changed(self.alpha_update)
        self.alpha_reset_btn.on_clicked(self.alpha_reset)
        self.interp_btn.on_clicked(self.interpolate)
        self.save_btn.on_clicked(self.save_atlas)
        self.edit_btn.on_clicked(self.toggle_edit_mode)
        self.color_picker_box.on_text_change(self.color_picker_changed)
        
        # initialize planes in all plot editors
        self.update_coords(coord, config.PLANE[0])
        
        # extra padding for slider labels
        gs.tight_layout(fig)
        plt.ion()
        plt.show()
    
    def on_key_press(self, event):
        """Respond to key press events.
        """
        if event.key == "a":
            # toggle between current and 0 opacity
            if self.alpha_slider.val == 0:
                # return to saved alpha if available and reset
                if self.alpha_last is not None:
                    self.alpha_slider.set_val(self.alpha_last)
                self.alpha_last = None
            else:
                # make transluscenct, saving alpha if not already saved 
                # during a halve-opacity event
                if self.alpha_last is None:
                    self.alpha_last = self.alpha_slider.val
                self.alpha_slider.set_val(0)
        elif event.key == "A":
            # halve opacity, only saving alpha on first halving to allow 
            # further halving or manual movements while still returning to 
            # originally saved alpha
            if self.alpha_last is None:
                self.alpha_last = self.alpha_slider.val
            self.alpha_slider.set_val(self.alpha_slider.val / 2)
        elif event.key == "up" or event.key == "down":
            # up/down arrow for scrolling planes
            self.scroll_overview(event)
        elif event.key == "w":
            # shortcut to toggle editing mode
            self.toggle_edit_mode(event)
    
    def update_coords(self, coord, plane_src=config.PLANE[0]):
        """Update all plot editors with given coordinates.
        
        Args:
            coord: Coordinate at which to center images, in z,y,x order.
            plane_src: One of :const:`magmap.config.PLANE` to specify the 
                orientation from which the coordinates were given; defaults 
                to the first element of :const:`magmap.config.PLANE`.
        """
        coord_rev = libmag.transpose_1d_rev(list(coord), plane_src)
        for plane in config.PLANE:
            coord_transposed = libmag.transpose_1d(list(coord_rev), plane)
            self.plot_eds[plane].update_coord(coord_transposed)
    
    def refresh_images(self, plot_ed):
        """Refresh images in a plot editor.
        
        Args:
            plot_ed: :class:`magmap.plot_editor.PlotEditor` whose images 
                will be refreshed.
        """
        for key in self.plot_eds:
            ed = self.plot_eds[key]
            if ed != plot_ed: ed.update_image()
            if ed.edited:
                # display save button as enabled if any editor has been edited
                enable_btn(self.save_btn)
    
    def scroll_overview(self, event):
        """Scroll images and crosshairs in all plot editors
        
        Args:
            event: Scroll event.
        """
        for key in self.plot_eds:
            self.plot_eds[key].scroll_overview(event)
    
    def alpha_update(self, event):
        """Update the alpha transparency in all plot editors.
        
        Args:
            event: Slider event.
        """
        for key in self.plot_eds:
            self.plot_eds[key].alpha_updater(event)
    
    def alpha_reset(self, event):
        """Reset the alpha transparency in all plot editors.
        
        Args:
            event: Button event, currently ignored.
        """
        self.alpha_slider.reset()
    
    def axes_exit(self, event):
        """Trigger axes exit for all plot editors.
        
        Args:
            event: Axes exit event.
        """
        for key in self.plot_eds:
            self.plot_eds[key].on_axes_exit(event)
    
    def interpolate(self, event):
        """Interpolate planes using :attr:`interp_planes`.
        
        Args:
            event: Button event, currently ignored.
        """
        try:
            self.interp_planes.interpolate(self.labels_img)
            self.refresh_images(None)
        except ValueError as e:
            print(e)
    
    def save_atlas(self, event):
        """Save atlas labels.
        
        Args:
            event: Button event, currently not used.
        """
        # only save if at least one editor has been edited
        if not any([ed.edited for ed in self.plot_eds.values()]): return
        sitk_io.load_registered_img(
            config.filename, config.RegNames.IMG_LABELS.value, 
            replace=config.labels_img)
        # reset edited flag in all editors and show save button as disabled
        for ed in self.plot_eds.values(): ed.edited = False
        enable_btn(self.save_btn, False)
        print("Saved labels image at {}".format(datetime.datetime.now()))

    def toggle_edit_mode(self, event):
        """Toggle editing mode, determining the current state from the
        first :class:`magmap.plot_editor.PlotEditor` and switching to the 
        opposite value for all plot editors.

        Args:
            event: Button event, currently not used.
        """
        edit_mode = False
        for i, ed in enumerate(self.plot_eds.values()):
            if i == 0:
                # change edit mode based on current mode in first plot editor
                edit_mode = not ed.edit_mode
                toggle_btn(self.edit_btn, edit_mode, text=self._EDIT_BTN_LBLS)
            ed.edit_mode = edit_mode
        if not edit_mode:
            # reset the color picker text box when turning off editing
            self.color_picker_box.set_val("")

    def update_color_picker(self, val):
        """Update the color picker :class:`TextBox` with the given value.

        Args:
            val: Color value.
        """
        self.color_picker_box.set_val(val)

    def color_picker_changed(self, text):
        """Respond to color picker :class:`TextBox` changes by updating
        the specified intensity value in all plot editors to an integer
        of the given value if it is a number and the first editor does
        not already have an intensity set to this value.

        Args:
            text: String of text box value.
        """
        if not libmag.is_number(text): return
        intensity = int(text)
        for i, ed in enumerate(self.plot_eds.values()):
            if i == 0:
                if intensity == ed.intensity:
                    # ignore if intensity already set, eg if
                    # triggered by update_color_picker
                    return
                else:
                    print("updating specified color to", text)
            ed.intensity_spec = intensity


def enable_btn(btn, enable=True):
    """Display a button as enabled or disabled.
    
    Note that the button's active state will not change since doing so 
    prevents the coloration from changing.
    
    Args:
        btn: Button widget to change.
        enable: True to enable (default), False to disable.
    """
    if enable:
        # "enable" button by changing to default grayscale color intensities
        btn.color = "0.85"
        btn.hovercolor = "0.95"
    else:
        # "disable" button by making darker and no hover response
        btn.color = "0.5"
        btn.hovercolor = "0.5"


def toggle_btn(btn, on=True, shift=0.2, text=None):
    """Toggle a button between on/off modes.

    Args:
        btn: Button widget to change.
        on: True to display the button as on, False as off.
        shift: Float of amount to shift the button color intensity;
            defaults to 0.2.
        text: Tuple of ``(on_text, off_text)`` for the button label;
            defaults to None to keep the original text.
    """
    if on:
        # turn button "on" by darkening intensities and updating label
        btn.color = str(float(btn.color) - shift)
        btn.hovercolor = str(float(btn.hovercolor) - shift)
        if text: btn.label.set_text(text[1])
    else:
        # turn button "off" by lightening intensities and updating label
        btn.color = str(float(btn.color) + shift)
        btn.hovercolor = str(float(btn.hovercolor) + shift)
        if text: btn.label.set_text(text[0])


class InterpolatePlanes:
    """Track manually edited planes between which to interpolate changes 
    for a given label.
    
    This interpolation replaces unedited planes based on the trends of 
    the edited ones to avoid the need to manually edit every single plane.
    
    Attributes:
        btn (:obj:`matplotlib.widgets.Button`): Button to initiate plane 
            interpolation.
        plane (str): Plane in which editing has occurred.
        bounds (List[int]): Unsorted start and end planes.
        label_id (int): Label ID of the edited region.
    """
    def __init__(self, btn):
        """Initialize plane interpolation object."""
        self.btn = btn
        self.plane = None
        self.bounds = None
        self.label_id = None
    
    def update_btn(self):
        """Update text and color of button to interpolate planes.
        """
        if any(self.bounds):
            # show current values if any exist
            self.btn.label.set_text(
                "Fill {} {}\nID {}"
                .format(plot_support.get_plane_axis(self.plane), self.bounds,
                        self.label_id))
            self.btn.label.set_fontsize("xx-small")
        enable_btn(self.btn, all(self.bounds))
        
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
            # reset bounds if new plane or label ID (plane and label_id 
            # should have been set together)
            self.bounds = None
        self.plane = plane
        self.label_id = label_id
        self.bounds = i
        self.update_btn()
    
    def interpolate(self, labels_img):
        """Interpolate between :attr:`bounds` in the given :attr:`plane` 
        direction in the bounding box surrounding :attr:`label_id`.
        
        Args:
            labels_img: Labels image as a Numpy array of x,y,z dimensions.
        """
        if not any(self.bounds):
            raise ValueError("boundaries not fully set: {}".format(self.bounds))
        cv_nd.interpolate_label_between_planes(
            labels_img, self.label_id, config.PLANE.index(self.plane), 
            self.bounds)
    
    def __str__(self):
        return "{}: {} (ID: {})".format(
            plot_support.get_plane_axis(self.plane), self.bounds, self.label_id)
    
    @property
    def bounds(self):
        """Get the bounds property."""
        return self._bounds
    
    @bounds.setter
    def bounds(self, val):
        """Set the bounds property.
        
        Args:
            val: Integer to append to the bounds list if not already
                present, removing the first value in the list, or a value to
                replace the current bounds value. None to reset.
        """
        if val is None:
            self._bounds = [None, None]
        elif isinstance(val, int):
            if val not in self._bounds:
                self._bounds.append(val)
                del self._bounds[0]
        else:
            self._bounds = val


if __name__ == "__main__":
    print("Starting atlas editor")
