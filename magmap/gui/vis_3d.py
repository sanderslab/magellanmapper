# 3D visualization in MagellanMapper

import math
from time import time

import numpy as np
from skimage import filters, restoration, transform

from magmap.cv import segmenter
from magmap.io import libmag
from magmap.plot import colormaps, plot_3d
from magmap.settings import config


class Vis3D:
    """3D visualization object for handling Mayavi/VTK tasks.
    
    Attributes:
        surfaces (list): List of Mayavi surfaces for each displayed channel;
            defaults to None.
    
    """
    #: float: Maximum number of points to show.
    _MASK_DIVIDEND = 10000.0  # 3D max points
    
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
    
    @staticmethod
    def _resize_glyphs_isotropic(settings, glyphs=None):
        # resize Mayavi glyphs to make them isotropic based on profile settings
        isotropic = plot_3d.get_isotropic_vis(settings)
        if isotropic is not None and glyphs:
            glyphs.actor.actor.scale = isotropic[::-1]
        return isotropic

    def plot_3d_points(self, roi, scene_mlab, channel, flipz=False):
        """Plots all pixels as points in 3D space.

        Points falling below a given threshold will be
        removed, allowing the viewer to see through the presumed
        background to masses within the region of interest.

        Args:
            roi (:obj:`np.ndarray`): Region of interest either as a 3D (z, y, x) or
                4D (z, y, x, channel) ndarray.
            scene_mlab (:mod:``mayavi.mlab``): Mayavi mlab module. Any
                current image will be cleared first.
            channel (int): Channel to select, which can be None to indicate all
                channels.
            flipz (bool): True to invert blobs along z-axis to match handedness
                of Matplotlib with z progressing upward; defaults to False.

        Returns:
            bool: True if points were rendered, False if no points to render.
        """
        print("plotting as 3D points")
        scene_mlab.clf()
    
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
        z = np.ones((shape[0], shape[1] * shape[2]))
        for i in range(shape[0]):
            z[i] = z[i] * i
        if flipz:
            # invert along z-axis to match handedness of Matplotlib with z up
            z *= -1
            z += shape[0]
        y = np.ones((shape[0] * shape[1], shape[2]))
        for i in range(shape[0]):
            for j in range(shape[1]):
                y[i * shape[1] + j] = y[i * shape[1] + j] * j
        x = np.ones((shape[0] * shape[1], shape[2]))
        for i in range(shape[0] * shape[1]):
            x[i] = np.arange(shape[2])
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
            pts = scene_mlab.points3d(
                np.delete(x, remove), np.delete(y, remove), np.delete(z, remove),
                roi_show_1d, mode="sphere",
                scale_mode="scalar", mask_points=mask, line_width=1.0, vmax=1.0,
                vmin=0.0, transparent=True)
            cmap = colormaps.get_cmap(config.cmaps, chl)
            if cmap is not None:
                pts.module_manager.scalar_lut_manager.lut.table = cmap(
                    range(0, 256)) * 255
            self._resize_glyphs_isotropic(settings, pts)
    
        print("time for 3D points display: {}".format(time() - time_start))
        return True

    def plot_3d_surface(self, roi, scene_mlab, channel, segment=False,
                        flipz=False):
        """Plots areas with greater intensity as 3D surfaces.

        Args:
            roi (:obj:`np.ndarray`): Region of interest either as a 3D
                ``z,y,x`` or 4D ``z,y,x,channel`` ndarray.
            scene_mlab (:mod:``mayavi.mlab``): Mayavi mlab module. Any
                current image will be cleared first.
            channel (int): Channel to select, which can be None to indicate all
                channels.
            segment (bool): True to denoise and segment ``roi`` before
                displaying, which may remove artifacts that might otherwise
                lead to spurious surfaces. Defaults to False.
            flipz: True to invert ``roi`` along z-axis to match handedness
                of Matplotlib with z progressing upward; defaults to False.

        Returns:
            list: List of Mayavi surfaces for each displayed channel, which
            are also stored in :attr:`surfaces`.

        """
        # Plot in Mayavi
        print("viewing 3D surface")
        pipeline = scene_mlab.pipeline
        scene_mlab.clf()
        settings = config.roi_profile
        if flipz:
            # invert along z-axis to match handedness of Matplotlib with z up
            roi = roi[::-1]
    
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
            surface = scene_mlab.pipeline.surface(surface)
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
        
            self._resize_glyphs_isotropic(settings, surface)
            surfaces.append(surface)
    
        print("time to render 3D surface: {}".format(time() - time_start))
        self.surfaces = surfaces
        return surfaces

    def _shadow_blob(self, x, y, z, cmap_indices, cmap, scale, mlab):
        """Shows blobs as shadows projected parallel to the 3D visualization.

        Parmas:
            x: Array of x-coordinates of blobs.
            y: Array of y-coordinates of blobs.
            z: Array of z-coordinates of blobs.
            cmap_indices: Indices of blobs for the colormap, usually given as a
                simple ascending sequence the same size as the number of blobs.
            cmap: The colormap, usually the same as for the segments.
            scale: Array of scaled size of each blob.
            mlab: Mayavi object.
        """
        pts_shadows = mlab.points3d(x, y, z, cmap_indices,
                                    mode="2dcircle", scale_mode="none",
                                    scale_factor=scale * 0.8, resolution=20)
        pts_shadows.module_manager.scalar_lut_manager.lut.table = cmap
        return pts_shadows

    def show_blobs(self, segments, mlab, segs_in_mask, cmap, show_shadows=False,
                   flipz=None):
        """Shows 3D blob segments.

        Args:
            segments: Labels from 3D blob detection method.
            mlab: Mayavi object.
            segs_in_mask: Boolean mask for segments within the ROI; all other 
                segments are assumed to be from padding and border regions 
                surrounding the ROI.
            cmap (:class:`numpy.ndaarry`): Colormap as a 2D Numpy array in the
                format  ``[[R, G, B, alpha], ...]``.
            show_shadows: True if shadows of blobs should be depicted on planes 
                behind the blobs; defaults to False.
            flipz (int): Invert blobs and shift them by this amount along the
                z-axis to match handedness of Matplotlib with z progressing
                upward; defaults to False.

        Returns:
            A 3-element tuple containing ``pts_in``, the 3D points within the 
            ROI; ``cmap'', the random colormap generated with a color for each 
            blob, and ``scale``, the current size of the points.
        """
        if segments.shape[0] <= 0:
            return None, None, 0
        settings = config.roi_profile
        segs = np.copy(segments)
        if flipz:
            # invert along z-axis within the same original space, eg to match
            # handedness of Matplotlib with z up
            segs[:, 0] *= -1
            segs[:, 0] += flipz
        isotropic = self._resize_glyphs_isotropic(settings)
        if isotropic is not None:
            # adjust position based on isotropic factor
            segs[:, :3] = np.multiply(segs[:, :3], isotropic)
    
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
                cmap, scale, mlab)
            # xz
            shadows = self._shadow_blob(
                segs_in[:, 2], segs_in[:, 0], segs_ones * -10, cmap_indices,
                cmap, scale, mlab)
            shadows.actor.actor.orientation = [90, 0, 0]
            shadows.actor.actor.position = [0, -20, 0]
            # yz
            shadows = self._shadow_blob(
                segs_in[:, 1], segs_in[:, 0], segs_ones * -10, cmap_indices,
                cmap, scale, mlab)
            shadows.actor.actor.orientation = [90, 90, 0]
            shadows.actor.actor.position = [0, 0, 0]
    
        # show the blobs
        points_len = len(segs)
        mask = math.ceil(points_len / self._MASK_DIVIDEND)
        print("points: {}, mask: {}".format(points_len, mask))
        pts_in = None
        if len(segs_in) > 0:
            # show segs within the ROI
            pts_in = mlab.points3d(
                segs_in[:, 2], segs_in[:, 1],
                segs_in[:, 0], cmap_indices,
                mask_points=mask, scale_mode="none", scale_factor=scale,
                resolution=50)
            pts_in.module_manager.scalar_lut_manager.lut.table = cmap
        # show segments within padding or border region as more transparent
        segs_out_mask = np.logical_not(segs_in_mask)
        if np.sum(segs_out_mask) > 0:
            mlab.points3d(
                segs[segs_out_mask, 2], segs[segs_out_mask, 1],
                segs[segs_out_mask, 0], color=(0, 0, 0),
                mask_points=mask, scale_mode="none", scale_factor=scale / 2,
                resolution=50, opacity=0.2)
    
        return pts_in, scale

    def _shadow_img2d(self, img2d, shape, axis, vis):
        """Shows a plane along the given axis as a shadow parallel to
        the 3D visualization.

        Args:
            img2d: The plane to show.
            shape: Shape of the ROI.
            axis: Axis along which the plane lies.
            vis: Visualization object.

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
        return vis.scene.mlab.imshow(img2d, opacity=0.5, colormap="gray")

    def plot_2d_shadows(self, roi, vis):
        """Plots 2D shadows in each axis around the 3D visualization.

        Args:
            roi: Region of interest.
            vis: Visualization object on which to draw the contour. Any 
                current image will be cleared first.
        """
        # set up shapes, accounting for any isotropic resizing
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
        img2d_mlab = self._shadow_img2d(img2d, shape_iso, 0, vis)
        # Mayavi positions are in x,y,z
        img2d_mlab.actor.position = [
            shape_iso_mid[2], shape_iso_mid[1], -10]
    
        # xz-plane
        img2d = roi[:, shape[1] // 2, :]
        img2d = transform.resize(
            img2d, np.multiply(img2d.shape, isotropic[[0, 2]]).astype(np.int),
            preserve_range=True)
        img2d_mlab = self._shadow_img2d(img2d, shape_iso, 2, vis)
        img2d_mlab.actor.position = [
            -10, shape_iso_mid[1], shape_iso_mid[0]]
        img2d_mlab.actor.orientation = [90, 90, 0]
    
        # yz-plane
        img2d = roi[:, :, shape[2] // 2]
        img2d = transform.resize(
            img2d, np.multiply(img2d.shape, isotropic[:2]).astype(np.int),
            preserve_range=True)
        img2d_mlab = self._shadow_img2d(img2d, shape_iso, 1, vis)
        img2d_mlab.actor.position = [
            shape_iso_mid[2], -10, shape_iso_mid[0]]
        img2d_mlab.actor.orientation = [90, 0, 0]


