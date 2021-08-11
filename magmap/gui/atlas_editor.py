# Atlas Editor with orthogonal viewing
# Author: David Young, 2018, 2020
"""Atlas editing GUI in the MagellanMapper package.
"""

import datetime
import os

from matplotlib import pyplot as plt
from matplotlib import figure
from matplotlib import gridspec
from matplotlib.widgets import Slider, Button, TextBox

from magmap.cv import cv_nd
from magmap.gui import plot_editor
from magmap.io import libmag, naming, sitk_io
from magmap.plot import colormaps, plot_support
from magmap.settings import config


class AtlasEditor(plot_support.ImageSyncMixin):
    """Graphical interface to view an atlas in multiple orthogonal 
    dimensions and edit atlas labels.
    
    :attr:`plot_eds` are dictionaries of keys specified by one of
    :const:`magmap.config.PLANE` plane orientations to Plot Editors.
    
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
        title (str): Window title; defaults to None.
        fn_refresh_atlas_eds (func): Callback for refreshing other
            Atlas Editors to synchronize them; defaults to None.
            Typically takes one argument, this ``AtlasEditor`` object
            to refreshing it. Defaults to None.
        alpha_slider: Matplotlib alpha slider control.
        alpha_reset_btn: Maplotlib button for resetting alpha transparency.
        alpha_last: Float specifying the previous alpha value.
        interp_planes: Current :class:`InterpolatePlanes` object.
        interp_btn: Matplotlib button to initiate plane interpolation.
        save_btn: Matplotlib button to save the atlas.
        fn_status_bar (func): Function to call during status bar updates
            in :class:`pixel_display.PixelDisplay`; defaults to None.
        fn_update_coords (func): Handler for coordinate updates, which
            takes coordinates in z-plane orientation; defaults to None.
    """

    _EDIT_BTN_LBLS = ("Edit", "Editing")

    def __init__(self, image5d, labels_img, channel, offset, fn_close_listener, 
                 borders_img=None, fn_show_label_3d=None, title=None,
                 fn_refresh_atlas_eds=None, fig=None, fn_status_bar=None):
        """Plot ROI as sequence of z-planes containing only the ROI itself."""
        super().__init__()
        self.image5d = image5d
        self.labels_img = labels_img
        self.channel = channel
        self.offset = offset
        self.fn_close_listener = fn_close_listener
        self.borders_img = borders_img
        self.fn_show_label_3d = fn_show_label_3d
        self.title = title
        self.fn_refresh_atlas_eds = fn_refresh_atlas_eds
        self.fig = fig
        self.fn_status_bar = fn_status_bar
        
        self.alpha_slider = None
        self.alpha_reset_btn = None
        self.alpha_last = None
        self.interp_planes = None
        self.interp_btn = None
        self.save_btn = None
        self.edit_btn = None
        self.color_picker_box = None
        self.fn_update_coords = None
        
        self._labels_img_sitk = None  # for saving labels image
        
    def show_atlas(self):
        """Set up the atlas display with multiple orthogonal views."""
        # set up the figure
        if self.fig is None:
            fig = figure.Figure(self.title)
            self.fig = fig
        else:
            fig = self.fig
        fig.clear()
        gs = gridspec.GridSpec(
            2, 1, wspace=0.1, hspace=0.1, height_ratios=(20, 1), figure=fig,
            left=0.06, right=0.94, bottom=0.02, top=0.98)
        gs_viewers = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=gs[0, 0])
        
        # set up a colormap for the borders image if present
        cmap_borders = colormaps.get_borders_colormap(
            self.borders_img, self.labels_img, config.cmap_labels)
        coord = list(self.offset[::-1])
        
        # editor controls, split into a slider sub-spec to allow greater
        # spacing for labels on either side and a separate sub-spec for
        # buttons and other fields
        gs_controls = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs[1, 0], width_ratios=(1, 1),
            wspace=0.15)
        self.alpha_slider = Slider(
            fig.add_subplot(gs_controls[0, 0]), "Opacity", 0.0, 1.0,
            valinit=plot_editor.PlotEditor.ALPHA_DEFAULT)
        gs_controls_btns = gridspec.GridSpecFromSubplotSpec(
            1, 5, subplot_spec=gs_controls[0, 1], wspace=0.1)
        self.alpha_reset_btn = Button(
            fig.add_subplot(gs_controls_btns[0, 0]), "Reset")
        self.interp_btn = Button(
            fig.add_subplot(gs_controls_btns[0, 1]), "Fill Label")
        self.interp_planes = InterpolatePlanes(self.interp_btn)
        self.interp_planes.update_btn()
        self.save_btn = Button(
            fig.add_subplot(gs_controls_btns[0, 2]), "Save")
        self.edit_btn = Button(
            fig.add_subplot(gs_controls_btns[0, 3]), "Edit")
        self.color_picker_box = TextBox(
            fig.add_subplot(gs_controls_btns[0, 4]), None)

        # adjust button colors based on theme and enabled status; note
        # that colors do not appear to refresh until fig mouseover
        for btn in (self.alpha_reset_btn, self.edit_btn):
            enable_btn(btn)
        enable_btn(self.save_btn, False)
        enable_btn(self.color_picker_box, color=config.widget_color+0.1)
    
        def setup_plot_ed(axis, gs_spec):
            # set up a PlotEditor for the given axis

            # get subplot grid with extra height ratio weighting for
            # each increased row to make sliders of approx equal size and  
            # align top borders of top images
            extra_rows = gs_spec.rowspan.stop - gs_spec.rowspan.start - 1
            gs_plot = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=gs_spec, 
                height_ratios=(1, 10 + 14 * extra_rows), 
                hspace=0.1/(extra_rows*1.4+1))
            
            # transform arrays to the given orthogonal direction
            ax = fig.add_subplot(gs_plot[1, 0])
            plot_support.hide_axes(ax)
            plane = config.PLANE[axis]
            arrs_3d, aspect, origin, scaling = \
                plot_support.setup_images_for_plane(
                    plane,
                    (self.image5d[0], self.labels_img, self.borders_img))
            img3d_tr, labels_img_tr, borders_img_tr = arrs_3d
            
            # slider through image planes
            ax_scroll = fig.add_subplot(gs_plot[0, 0])
            plane_slider = Slider(
                ax_scroll, plot_support.get_plane_axis(plane), 0, 
                len(img3d_tr) - 1, valfmt="%d", valinit=0, valstep=1)
            
            # plot editor
            max_size = max_sizes[axis] if max_sizes else None
            plot_ed = plot_editor.PlotEditor(
                ax, img3d_tr, labels_img_tr, config.cmap_labels,
                plane, aspect, origin, self.update_coords, self.refresh_images, 
                scaling, plane_slider, img3d_borders=borders_img_tr,
                cmap_borders=cmap_borders, 
                fn_show_label_3d=self.fn_show_label_3d, 
                interp_planes=self.interp_planes,
                fn_update_intensity=self.update_color_picker,
                max_size=max_size, fn_status_bar=self.fn_status_bar)
            return plot_ed
        
        # setup plot editors for all 3 orthogonal directions
        max_sizes = plot_support.get_downsample_max_sizes()
        for i, gs_viewer in enumerate(
                (gs_viewers[:2, 0], gs_viewers[0, 1], gs_viewers[1, 1])):
            self.plot_eds[config.PLANE[i]] = setup_plot_ed(i, gs_viewer)
        self.set_show_crosslines(True)
        
        # attach listeners
        fig.canvas.mpl_connect("scroll_event", self.scroll_overview)
        fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        fig.canvas.mpl_connect("close_event", self._close)
        fig.canvas.mpl_connect("axes_leave_event", self.axes_exit)
        
        self.alpha_slider.on_changed(self.alpha_update)
        self.alpha_reset_btn.on_clicked(self.alpha_reset)
        self.interp_btn.on_clicked(self.interpolate)
        self.save_btn.on_clicked(self.save_atlas)
        self.edit_btn.on_clicked(self.toggle_edit_mode)
        self.color_picker_box.on_text_change(self.color_picker_changed)
        
        # initialize and show planes in all plot editors
        if self._max_intens_proj is not None:
            self.update_max_intens_proj(self._max_intens_proj)
        self.update_coords(coord, config.PLANE[0])

        plt.ion()  # avoid the need for draw calls

    def _close(self, evt):
        """Handle figure close events by calling :attr:`fn_close_listener`
        with this object.

        Args:
            evt (:obj:`matplotlib.backend_bases.CloseEvent`): Close event.

        """
        self.fn_close_listener(evt, self)

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
                # make translucent, saving alpha if not already saved
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
        elif event.key == "ctrl+s" or event.key == "cmd+s":
            # support default save shortcuts on multiple platforms;
            # ctrl-s will bring up save dialog from fig, but cmd/win-S
            # will bypass
            self.save_fig(self.get_save_path())
    
    def update_coords(self, coord, plane_src=config.PLANE[0]):
        """Update all plot editors with given coordinates.
        
        Args:
            coord: Coordinate at which to center images, in z,y,x order.
            plane_src: One of :const:`magmap.config.PLANE` to specify the 
                orientation from which the coordinates were given; defaults 
                to the first element of :const:`magmap.config.PLANE`.
        """
        coord_rev = libmag.transpose_1d_rev(list(coord), plane_src)
        for i, plane in enumerate(config.PLANE):
            coord_transposed = libmag.transpose_1d(list(coord_rev), plane)
            if i == 0:
                self.offset = coord_transposed[::-1]
                if self.fn_update_coords:
                    # update offset based on xy plane, without centering
                    # planes are centered on the offset as-is
                    self.fn_update_coords(coord_transposed, False)
            self.plot_eds[plane].update_coord(coord_transposed)

    def view_subimg(self, offset, shape):
        """Zoom all Plot Editors to the given sub-image.

        Args:
            offset: Sub-image coordinates in ``z,y,x`` order.
            shape: Sub-image shape in ``z,y,x`` order.
        
        """
        for i, plane in enumerate(config.PLANE):
            offset_tr = libmag.transpose_1d(list(offset), plane)
            shape_tr = libmag.transpose_1d(list(shape), plane)
            self.plot_eds[plane].view_subimg(offset_tr[1:], shape_tr[1:])
        self.fig.canvas.draw_idle()

    def refresh_images(self, plot_ed=None, update_atlas_eds=False):
        """Refresh images in a plot editor, such as after editing one
        editor and updating the displayed image in the other editors.
        
        Args:
            plot_ed (:obj:`magmap.plot_editor.PlotEditor`): Editor that
                does not need updating, typically the editor that originally
                changed. Defaults to None.
            update_atlas_eds (bool): True to update other ``AtlasEditor``s;
                defaults to False.
        """
        for key in self.plot_eds:
            ed = self.plot_eds[key]
            if ed != plot_ed: ed.refresh_img3d_labels()
            if ed.edited:
                # display save button as enabled if any editor has been edited
                enable_btn(self.save_btn)
        if update_atlas_eds and self.fn_refresh_atlas_eds is not None:
            # callback to synchronize other Atlas Editors
            self.fn_refresh_atlas_eds(self)
    
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
            # flag Plot Editors as edited so labels can be saved
            for ed in self.plot_eds.values(): ed.edited = True
            self.refresh_images(None, True)
        except ValueError as e:
            print(e)
    
    def save_atlas(self, event):
        """Save atlas labels using the registered image suffix given by
        :attr:`config.reg_suffixes[config.RegSuffixes.ANNOTATION]`.
        
        Args:
            event: Button event, currently not used.
        
        """
        # only save if at least one editor has been edited
        if not any([ed.edited for ed in self.plot_eds.values()]): return
        
        # save to the labels reg suffix; use sitk Image if loaded and store
        # any Image loaded during saving
        reg_name = config.reg_suffixes[config.RegSuffixes.ANNOTATION]
        if self._labels_img_sitk is None:
            self._labels_img_sitk = config.labels_img_sitk
        self._labels_img_sitk = sitk_io.write_registered_image(
            self.labels_img, config.filename, reg_name, self._labels_img_sitk,
            overwrite=True)
        
        # reset edited flag in all editors and show save button as disabled
        for ed in self.plot_eds.values(): ed.edited = False
        enable_btn(self.save_btn, False)
        print("Saved labels image at {}".format(datetime.datetime.now()))
    
    def get_save_path(self):
        """Get figure save path based on filename, ROI, and overview plane
         shown.
        
        Returns:
            str: Figure save path.

        """
        ext = config.savefig if config.savefig else config.DEFAULT_SAVEFIG
        return "{}.{}".format(naming.get_roi_path(
            os.path.basename(self.title), self.offset), ext)
    
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
            val (str): Color value. If None, only :meth:`color_picker_changed`
                will be triggered.
        """
        if val is None:
            # updated picked color directly
            self.color_picker_changed(val)
        else:
            # update text box, which triggers color_picker_changed
            self.color_picker_box.set_val(val)

    def color_picker_changed(self, text):
        """Respond to color picker :class:`TextBox` changes by updating
        the specified intensity value in all plot editors.

        Args:
            text (str): String of text box value. Converted to an int if
                non-empty.
        """
        intensity = text
        if text:
            if not libmag.is_number(intensity): return
            intensity = int(intensity)
        print("updating specified color to", intensity)
        for i, ed in enumerate(self.plot_eds.values()):
            ed.intensity_spec = intensity


def enable_btn(btn, enable=True, color=None, max_color=0.99):
    """Display a button or other widget as enabled or disabled.
    
    Note that the button's active state will not change since doing so 
    prevents the coloration from changing.
    
    Args:
        btn (:class:`matplotlib.widgets.AxesWidget`): Widget to change,
            which must have ``color`` and ``hovercolor`` attributes.
        enable (bool): True to enable (default), False to disable.
        color (float): Intensity value from 0-1 for the main color. The
            hovercolor will be just above this value, while the disabled
            main and hovercolors will be just below this value. Defaults
            to None, which will use :attr:`config.widget_color`.
        max_color (float): Max intensity value for hover color; defaults
            to 0.99 to provide at least some contrast with white backgrounds.
    """
    if color is None:
        color = config.widget_color
    if enable:
        # "enable" button by changing to default grayscale color intensities
        btn.color = str(color)
        hover = color + 0.1
        if hover > max_color:
            # intensities > 1 appear to recycle, so clip to max allowable val
            hover = max_color
        btn.hovercolor = str(hover)
    else:
        # "disable" button by making darker and no hover response
        color_disabled = color - 0.2
        if color_disabled < 0: color_disabled = 0
        color_disabled = str(color_disabled)
        btn.color = color_disabled
        btn.hovercolor = color_disabled


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
        if not all(self.bounds):
            raise ValueError("boundaries not fully set: {}".format(self.bounds))
        print("interpolating edits between planes", self.bounds)
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
        elif libmag.is_int(val):
            if val not in self._bounds:
                self._bounds.append(val)
                del self._bounds[0]
        else:
            self._bounds = val


if __name__ == "__main__":
    print("Starting atlas editor")
