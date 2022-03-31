# 3D visualization in MagellanMapper

import math
from time import time
from typing import Any, Callable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
from skimage import filters, restoration, transform

from magmap.cv import colocalizer, segmenter
from magmap.io import libmag
from magmap.plot import colormaps, plot_3d
from magmap.settings import config

if TYPE_CHECKING:
    from magmap.cv import detector
    from mayavi.modules.glyph import Glyph
    from mayavi.tools.mlab_scene_model import MlabSceneModel


class Vis3D:
    """3D visualization object for handling Mayavi/VTK tasks.
    
    Attributes:
        scene: Mayavi scene.
        fn_update_coords: Callback to update coordinates; defaults to
            None.
        surfaces: List of Mayavi surfaces for each displayed channel;
            defaults to None.
        blobs3d: List of Mayavi glyphs, where each glyph typically contains
            many 3D points representing blob positions; defaults to None.
        blobs3d_in: Mayavi glyphs of blobs inside the ROI; defaults to None.
        matches3d: Mayavi glyphs of blob matches; defaults to None.
    
    """
    #: float: Maximum number of points to show.
    _MASK_DIVIDEND = 10000.0  # 3D max points
    
    def __init__(self, scene):
        """Initialize a 3D visualization object.
        
        Args:
            scene (:class:`mayavi.tools.mlab_scene_model.MlabSceneModel`):
                Mayavi scene.
        
        """
        self.scene: "MlabSceneModel" = scene
        
        # callbacks
        self.fn_update_coords: Optional[Callable[[np.ndarray], None]] = None
        
        # generated Mayavi objects
        self.surfaces: Optional[List] = None
        self.blobs3d_in: Optional["Glyph"] = None
        self.matches3d: Optional["Glyph"] = None
        self.blobs3d: Optional[List["Glyph"]] = None

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
    
    def plot_3d_points(self, roi, channel, flipz=False, offset=None):
        """Plots all pixels as points in 3D space.

        Points falling below a given threshold will be removed, allowing
        the viewer to see through the presumed background to masses within
        the region of interest.

        Args:
            roi (:class:`numpy.ndarray`): Region of interest either as a 3D
                ``z,y,x`` or 4D ``z,y,x,c`` array.
            channel (int): Channel to select, which can be None to indicate all
                channels.
            flipz (bool): True to invert the ROI along the z-axis to match
                the handedness of Matplotlib with z progressing upward;
                defaults to False.
            offset (Sequence[int]): Origin coordinates in ``z,y,x``; defaults
                to None.

        Returns:
            bool: True if points were rendered, False if no points to render.
        
        """
        print("Plotting ROI as 3D points")
    
        # streamline the image
        if roi is None or roi.size < 1: return False
        roi = plot_3d.saturate_roi(roi, clip_vmax=98.5, channel=channel)
        roi = np.clip(roi, 0.2, 0.8)
        roi = restoration.denoise_tv_chambolle(roi, weight=0.1)
    
        # separate parallel arrays for each dimension of all coordinates for
        # Mayavi input format, with the ROI itself given as a 1D scalar array ;
        # TODO: consider using np.mgrid to construct the x,y,z arrays
        time_start = time()
        shape = roi.shape
        isotropic = plot_3d.get_isotropic_vis(config.roi_profile)
        z = np.ones((shape[0], shape[1] * shape[2]))
        for i in range(shape[0]):
            z[i] = z[i] * i
        if flipz:
            # invert along z-axis to match handedness of Matplotlib with z up
            z *= -1
            if offset is not None:
                offset = np.copy(offset)
                offset[0] *= -1
        y = np.ones((shape[0] * shape[1], shape[2]))
        for i in range(shape[0]):
            for j in range(shape[1]):
                y[i * shape[1] + j] = y[i * shape[1] + j] * j
        x = np.ones((shape[0] * shape[1], shape[2]))
        for i in range(shape[0] * shape[1]):
            x[i] = np.arange(shape[2])
        
        if offset is not None:
            offset = np.multiply(offset, isotropic)
        coords = [z, y, x]
        for i, _ in enumerate(coords):
            # scale coordinates for isotropy
            coords[i] *= isotropic[i]
            if offset is not None:
                # translate by offset
                coords[i] += offset[i]
        
        multichannel, channels = plot_3d.setup_channels(roi, channel, 3)
        for chl in channels:
            roi_show = roi[..., chl] if multichannel else roi
            roi_show_1d = roi_show.reshape(roi_show.size)
            if chl == 0:
                x = np.reshape(x, roi_show.size)
                y = np.reshape(y, roi_show.size)
                z = np.reshape(z, roi_show.size)
            settings = config.get_roi_profile(chl)
        
            # clear background points to see remaining structures
            thresh = 0
            if len(np.unique(roi_show)) > 1:
                # need > 1 val to threshold
                try:
                    thresh = filters.threshold_otsu(roi_show, 64)
                except ValueError as e:
                    thresh = np.median(roi_show)
                    print("could not determine Otsu threshold, taking median "
                          "({}) instead".format(thresh))
                thresh *= settings["points_3d_thresh"]
            print("removing 3D points below threshold of {}".format(thresh))
            remove = np.where(roi_show_1d < thresh)
            roi_show_1d = np.delete(roi_show_1d, remove)
        
            # adjust range from 0-1 to region of colormap to use
            roi_show_1d = libmag.normalize(roi_show_1d, 0.6, 1.0)
            points_len = roi_show_1d.size
            if points_len == 0:
                print("no 3D points to display")
                return False
            mask = math.ceil(points_len / self._MASK_DIVIDEND)
            print("points: {}, mask: {}".format(points_len, mask))
            if any(np.isnan(roi_show_1d)):
                # TODO: see if some NaNs are permissible
                print("NaN values for 3D points, will not show 3D visualization")
                return False
            pts = self.scene.mlab.points3d(
                np.delete(x, remove), np.delete(y, remove), np.delete(z, remove),
                roi_show_1d, mode="sphere",
                scale_mode="scalar", mask_points=mask, line_width=1.0, vmax=1.0,
                vmin=0.0, transparent=True)
            cmap = colormaps.get_cmap(config.cmaps, chl)
            if cmap is not None:
                pts.module_manager.scalar_lut_manager.lut.table = cmap(
                    range(0, 256)) * 255
            
            # scale glyphs to partially fill in gaps from isotropic scaling;
            # do not use actor scaling as it also translates the points when
            # not positioned at the origin
            pts.glyph.glyph.scale_factor = 2 * max(isotropic)

        # keep visual ordering of surfaces when opacity is reduced
        self.scene.renderer.use_depth_peeling = True
        print("time for 3D points display: {}".format(time() - time_start))
        return True

    def plot_3d_surface(self, roi, channel, segment=False, flipz=False,
                        offset=None):
        """Plots areas with greater intensity as 3D surfaces.

        The scene will be cleared before display.
        
        Args:
            roi (:class:`numpy.ndarray`): Region of interest either as a 3D
                ``z,y,x`` or 4D ``z,y,x,c`` array.
            channel (int): Channel to select, which can be None to indicate all
                channels.
            segment (bool): True to denoise and segment ``roi`` before
                displaying, which may remove artifacts that might otherwise
                lead to spurious surfaces. Defaults to False.
            flipz: True to invert ``roi`` along z-axis to match handedness
                of Matplotlib with z progressing upward; defaults to False.
            offset (Sequence[int]): Origin coordinates in ``z,y,x``; defaults
                to None.

        Returns:
            list: List of Mayavi surfaces for each displayed channel, which
            are also stored in :attr:`surfaces`.

        """
        # Plot in Mayavi
        print("viewing 3D surface")
        pipeline = self.scene.mlab.pipeline
        settings = config.roi_profile
        if flipz:
            # invert along z-axis to match handedness of Matplotlib with z up
            roi = roi[::-1]
            if offset is not None:
                # invert z-offset and translate by ROI z-size so ROI is
                # mirrored across the xy-plane
                offset = np.copy(offset)
                offset[0] = -offset[0] - roi.shape[0]
        isotropic = plot_3d.get_isotropic_vis(settings)
    
        # saturate to remove noise and normalize values
        roi = plot_3d.saturate_roi(roi, channel=channel)
    
        # turn off segmentation if ROI too big (arbitrarily set here as 
        # > 10 million pixels) to avoid performance hit and since likely showing 
        # large region of downsampled image anyway, where don't need hi res
        num_pixels = np.prod(roi.shape)
        to_segment = num_pixels < 10000000
    
        time_start = time()
        multichannel, channels = plot_3d.setup_channels(roi, channel, 3)
        surfaces = []
        for chl in channels:
            roi_show = roi[..., chl] if multichannel else roi
        
            # clip to minimize sub-nuclear variation
            roi_show = np.clip(roi_show, 0.2, 0.8)
        
            if segment:
                # denoising makes for much cleaner images but also seems to
                # allow structures to blend together
                # TODO: consider segmenting individual structures and rendering
                # as separate surfaces to avoid blending
                roi_show = restoration.denoise_tv_chambolle(
                    roi_show, weight=0.1)
            
                # build surface from segmented ROI
                if to_segment:
                    vmin, vmax = np.percentile(roi_show, (40, 70))
                    walker = segmenter.segment_rw(
                        roi_show, chl, vmin=vmin, vmax=vmax)
                    roi_show *= np.subtract(walker[0], 1)
                else:
                    print("deferring segmentation as {} px is above threshold"
                          .format(num_pixels))
        
            # ROI is in (z, y, x) order, so need to transpose or swap x,z axes
            roi_show = np.transpose(roi_show)
            surface = pipeline.scalar_field(roi_show)
        
            # Contour -> Surface pipeline
        
            # create the surface
            surface = pipeline.contour(surface)
            # remove many more extraneous points
            surface = pipeline.user_defined(
                surface, filter="SmoothPolyDataFilter")
            surface.filter.number_of_iterations = 400
            surface.filter.relaxation_factor = 0.015
            # distinguishing pos vs neg curvatures?
            surface = pipeline.user_defined(surface, filter="Curvatures")
            surface = self.scene.mlab.pipeline.surface(surface)
            module_manager = surface.module_manager
            module_manager.scalar_lut_manager.data_range = np.array([-2, 0])
            module_manager.scalar_lut_manager.lut_mode = "gray"
        
            '''
            # Surface pipleline with contours enabled (similar to above?)
            surface = pipeline.contour_surface(
                surface, color=(0.7, 1, 0.7), line_width=6.0)
            surface.actor.property.representation = 'wireframe'
            #surface.actor.property.line_width = 6.0
            surface.actor.mapper.scalar_visibility = False
            '''
        
            '''
            # IsoSurface pipeline

            # uses unique IsoSurface module but appears to have 
            # similar output to contour_surface
            surface = pipeline.iso_surface(surface)

            # limit contours for simpler surfaces including smaller file sizes; 
            # TODO: consider making settable as arg or through profile
            surface.contour.number_of_contours = 1
            try:
                # increase min to further reduce complexity
                surface.contour.minimum_contour = 0.5
                surface.contour.maximum_contour = 0.8
            except Exception as e:
                print(e)
                print("ignoring min/max contour for now")
            '''
        
            if offset is not None:
                # translate to offset scaled by isotropic factor
                surface.actor.actor.position = np.multiply(
                    offset, isotropic)[::-1]
            # scale surfaces, which expands/contracts but does not appear
            # to translate the surface position
            surface.actor.actor.scale = isotropic[::-1]
            surfaces.append(surface)
        
        # keep visual ordering of surfaces when opacity is reduced
        self.scene.renderer.use_depth_peeling = True
        print("time to render 3D surface: {}".format(time() - time_start))
        self.surfaces = surfaces
        return surfaces

    def _shadow_blob(self, x, y, z, cmap_indices, cmap, scale):
        """Shows blobs as shadows projected parallel to the 3D visualization.

        Parmas:
            x: Array of x-coordinates of blobs.
            y: Array of y-coordinates of blobs.
            z: Array of z-coordinates of blobs.
            cmap_indices: Indices of blobs for the colormap, usually given as a
                simple ascending sequence the same size as the number of blobs.
            cmap: The colormap, usually the same as for the segments.
            scale: Array of scaled size of each blob.
        
        """
        pts_shadows = self.scene.mlab.points3d(
            x, y, z, cmap_indices, mode="2dcircle", scale_mode="none",
            scale_factor=scale * 0.8, resolution=20)
        pts_shadows.module_manager.scalar_lut_manager.lut.table = cmap
        return pts_shadows

    def show_blobs(
            self,
            blobs: "detector.Blobs",
            segs_in_mask: np.ndarray,
            cmap: np.ndarray,
            roi_offset: Sequence[int], roi_size: Sequence[int],
            show_shadows: bool = False, flipz: bool = None
    ) -> float:
        """Show 3D blobs as points.

        Args:
            blobs: Detected blobs. Blob matches will also be displayed.
            segs_in_mask: Boolean mask for segments within the ROI; all other 
                segments are assumed to be from padding and border regions 
                surrounding the ROI.
            cmap: Colormap as a 2D Numpy array in the
                format  ``[[R, G, B, alpha], ...]``.
            roi_offset: Region of interest offset in ``z,y,x``.
            roi_size: Region of interest size in ``z,y,x``.
                Used to show the ROI outline.
            show_shadows: True if shadows of blobs should be depicted on planes 
                behind the blobs; defaults to False.
            flipz: True to invert blobs along the z-axis to match
                the handedness of Matplotlib with z progressing upward;
                defaults to False.

        Returns:
            The current size of the points.
        
        """
        segments = blobs.blobs
        if segments.shape[0] <= 0:
            return 0
        if roi_offset is None:
            roi_offset = np.zeros(3, dtype=np.int)
        if self.blobs3d:
            for blob in self.blobs3d:
                # remove existing blob glyphs from the pipeline
                blob.remove()
        self.blobs3d_in = None
        self.matches3d = None
        self.blobs3d = []
        
        settings = config.roi_profile
        # copy blobs with duplicate columns to access original values for
        # the coordinates callback when a blob is selected
        segs = np.concatenate((segments[:, :4], segments[:, :4]), axis=1)
        
        matches = None
        matches_cmap = None
        if blobs.blob_matches is not None:
            # set up match-based colocalizations
            matches = blobs.blob_matches.coords
            matches_cmap = blobs.blob_matches.cmap
        
        isotropic = plot_3d.get_isotropic_vis(settings)
        if flipz:
            # invert along z-axis within the same original space, eg to match
            # handedness of Matplotlib with z up
            segs[:, 0] *= -1
            roi_offset = np.copy(roi_offset)
            roi_offset[0] *= -1
            roi_size = np.copy(roi_size)
            roi_size[0] *= -1
            segs[:, :3] = np.add(segs[:, :3], roi_offset)
            if matches is not None:
                matches[:, 0] *= -1
                matches[:, :3] = np.add(matches[:, :3], roi_offset)
        if isotropic is not None:
            # adjust position based on isotropic factor
            roi_offset = np.multiply(roi_offset, isotropic)
            roi_size = np.multiply(roi_size[:3], isotropic)
            segs[:, :3] = np.multiply(segs[:, :3], isotropic)
            if matches is not None:
                matches[:, :3] = np.multiply(matches[:, :3], isotropic)
    
        radii = segs[:, 3]
        scale = 5 if radii is None else np.mean(np.mean(radii) + np.amax(radii))
        print("blob point scaling: {}".format(scale))
        # colormap has to be at least 2 colors
        segs_in = segs[segs_in_mask]
        cmap_indices = np.arange(segs_in.shape[0])
    
        if show_shadows:
            # show projections onto side planes, assumed to be at -10 units
            # along the given axis
            segs_ones = np.ones(segs.shape[0])
            # xy
            self._shadow_blob(
                segs_in[:, 2], segs_in[:, 1], segs_ones * -10, cmap_indices,
                cmap, scale)
            # xz
            shadows = self._shadow_blob(
                segs_in[:, 2], segs_in[:, 0], segs_ones * -10, cmap_indices,
                cmap, scale)
            shadows.actor.actor.orientation = [90, 0, 0]
            shadows.actor.actor.position = [0, -20, 0]
            # yz
            shadows = self._shadow_blob(
                segs_in[:, 1], segs_in[:, 0], segs_ones * -10, cmap_indices,
                cmap, scale)
            shadows.actor.actor.orientation = [90, 90, 0]
            shadows.actor.actor.position = [0, 0, 0]
    
        # show blobs within the ROI
        points_len = len(segs)
        mask = math.ceil(points_len / self._MASK_DIVIDEND)
        print("points: {}, mask: {}".format(points_len, mask))
        if len(segs_in) > 0:
            # each Glyph contains multiple 3D points, one for each blob
            self.blobs3d_in = self.scene.mlab.points3d(
                segs_in[:, 2], segs_in[:, 1],
                segs_in[:, 0], cmap_indices,
                mask_points=mask, scale_mode="none", scale_factor=scale,
                resolution=50)
            self.blobs3d_in.module_manager.scalar_lut_manager.lut.table = cmap
            self.blobs3d.append(self.blobs3d_in)
        
        # show blobs within padding or border region as black and more
        # transparent
        segs_out_mask = np.logical_not(segs_in_mask)
        if np.sum(segs_out_mask) > 0:
            self.blobs3d.append(self.scene.mlab.points3d(
                segs[segs_out_mask, 2], segs[segs_out_mask, 1],
                segs[segs_out_mask, 0], color=(0, 0, 0),
                mask_points=mask, scale_mode="none", scale_factor=scale / 2,
                resolution=50, opacity=0.2))
        
        # blob match display
        if matches is not None:
            # default to yellow
            color = (0.5, 0.5, 0) if matches_cmap is None else None
            self.matches3d = self.scene.mlab.points3d(
                matches[:, 2], matches[:, 1], matches[:, 0],
                np.arange(len(matches)),
                color=color, opacity=0.5, mask_points=mask,
                scale_mode="none", scale_factor=scale,
                resolution=50, mode="cube")
            if matches_cmap is not None:
                self.matches3d.module_manager.scalar_lut_manager.lut.table = \
                    matches_cmap
            self.blobs3d.append(self.matches3d)

        def pick_callback(pick):
            # handle picking blobs/glyphs
            if pick.actor in self.blobs3d_in.actor.actors:
                # get the blob corresponding to the picked glyph actor
                blobi = pick.point_id // glyph_points.shape[0]
            else:
                # find the closest blob to the pick position
                dists = np.linalg.norm(
                    segs_in[:, :3] - pick.pick_position[::-1], axis=1)
                blobi = np.argmin(dists)
                if dists[blobi] > max_dist:
                    # remove blob if not within a tolerated distance
                    blobi = None
            if blobi is None:
                # revert outline to full ROI if no blob is found
                self.show_roi_outline(roi_offset, roi_size)
            else:
                # move outline cube to surround picked blob; each glyph has
                # has many points, and each point ID maps to a data index
                # after floor division by the number of points
                z, y, x, r = segs_in[blobi, :4]
                outline.bounds = (x - r, x + r, y - r, y + r, z - r, z + r)
                if self.fn_update_coords:
                    # callback to update coordinates using blob's orig coords
                    self.fn_update_coords(np.add(
                        segs_in[blobi, 4:7], roi_offset).astype(np.int))
        
        # show ROI outline and make blobs pickable, falling back to closest
        # blobs within 20% of the longest ROI edge to be picked if present
        outline = self.show_roi_outline(roi_offset, roi_size)
        print(outline)
        glyph_points = self.blobs3d_in.glyph.glyph_source.glyph_source.\
            output.points.to_array()
        max_dist = max(roi_size) * 0.2
        self.scene.mlab.gcf().on_mouse_pick(pick_callback)
        
        return scale

    def _shadow_img2d(self, img2d, shape, axis):
        """Shows a plane along the given axis as a shadow parallel to
        the 3D visualization.

        Args:
            img2d: The plane to show.
            shape: Shape of the ROI.
            axis: Axis along which the plane lies.

        Returns:
            The displayed plane.
        """
        img2d = np.swapaxes(img2d, 0, 1)
        img2d[img2d < 1] = 0
        # expands the plane to match the size of the xy plane, with this
        # plane in the middle
        extra_z = (shape[axis] - shape[0]) // 2
        if extra_z > 0:
            img2d_full = np.zeros(shape[1] * shape[2])
            img2d_full = np.reshape(img2d_full, [shape[1], shape[2]])
            img2d_full[:, extra_z:extra_z + img2d.shape[1]] = img2d
            img2d = img2d_full
        return self.scene.mlab.imshow(img2d, opacity=0.5, colormap="gray")

    def plot_2d_shadows(self, roi, flipz=False):
        """Plots 2D shadows in each axis around the 3D visualization.

        Args:
            roi (:class:`numpy.ndarray`): Region of interest.
            flipz (bool): True to invert ``roi`` along z-axis to match
                handedness of Matplotlib with z progressing upward; defaults
                to False.
        
        """
        # set up shapes, accounting for any isotropic resizing
        if flipz:
            # invert along z-axis to match handedness of Matplotlib with z up
            roi = roi[::-1]
        if len(roi.shape) > 2:
            # covert 4D to 3D array, using only the 1st channel
            roi = roi[:, :, :, 0]
        isotropic = plot_3d.get_isotropic_vis(config.roi_profile)
        shape = roi.shape
        shape_iso = np.multiply(roi.shape, isotropic).astype(np.int)
        shape_iso_mid = shape_iso // 2
        
        # TODO: shift z by +10?
    
        # xy-plane, positioned just below the 3D ROI
        img2d = roi[shape[0] // 2, :, :]
        img2d = transform.resize(
            img2d, np.multiply(img2d.shape, isotropic[1:]).astype(np.int),
            preserve_range=True)
        img2d_mlab = self._shadow_img2d(img2d, shape_iso, 0)
        # Mayavi positions are in x,y,z
        img2d_mlab.actor.position = [
            shape_iso_mid[2], shape_iso_mid[1], -10]
    
        # xz-plane
        img2d = roi[:, shape[1] // 2, :]
        img2d = transform.resize(
            img2d, np.multiply(img2d.shape, isotropic[[0, 2]]).astype(np.int),
            preserve_range=True)
        img2d_mlab = self._shadow_img2d(img2d, shape_iso, 2)
        img2d_mlab.actor.position = [
            -10, shape_iso_mid[1], shape_iso_mid[0]]
        img2d_mlab.actor.orientation = [90, 90, 0]
    
        # yz-plane
        img2d = roi[:, :, shape[2] // 2]
        img2d = transform.resize(
            img2d, np.multiply(img2d.shape, isotropic[:2]).astype(np.int),
            preserve_range=True)
        img2d_mlab = self._shadow_img2d(img2d, shape_iso, 1)
        img2d_mlab.actor.position = [
            shape_iso_mid[2], -10, shape_iso_mid[0]]
        img2d_mlab.actor.orientation = [90, 0, 0]

    def show_roi_outline(self, roi_offset, roi_size):
        """Show plot outline to show ROI borders.
        
        Args:
            roi_offset (Sequence[int]): Region of interest offset in ``z,y,x``.
            roi_size (Sequence[int]): Region of interest size in ``z,y,x``.

        Returns:
            :class:`mayavi.modules.outline.Outline`: Outline object.

        """
        # manually calculate extent since the default bounds do not always
        # capture all objects and to include any empty border spaces
        return self.scene.mlab.outline(
            extent=np.array(tuple(zip(roi_offset, np.add(
                roi_offset, roi_size)))).ravel()[::-1])
    
    def clear_scene(self):
        """Clear the scene."""
        print("Clearing 3D scene")
        self.scene.mlab.clf()
