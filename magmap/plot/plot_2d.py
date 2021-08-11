#!/usr/bin/env python
# 2D plot image and graph plotter
# Author: David Young, 2017, 2020
"""Plot 2D views of imaging data and graphs."""

import os
import math
from typing import Optional, Sequence

import numpy as np
from matplotlib import colors as mat_colors, gridspec, pylab, pyplot as plt
import matplotlib.transforms as transforms
import pandas as pd
from skimage import exposure

from magmap.plot import colormaps
from magmap.settings import config
from magmap.io import libmag
from magmap.stats import mlearn
from magmap.plot import plot_support
from magmap.stats import vols


def _show_overlay(ax, img, plane_i, cmap, out_plane, aspect=1.0, alpha=1.0,
                  title=None):
    """Shows an image for overlays in the orthogonal plane specified by 
    :attribute:`plane`.
    
    Args:
        ax: Subplot axes.
        img: 3D image.
        plane_i: Plane index of `img` to show.
        cmap: Name of colormap.
        aspect: Aspect ratio; defaults to 1.0.
        alpha: Alpha level; defaults to 1.0.
        title: Subplot title; defaults to None, in which case no title will 
            be shown.
    """
    if out_plane == config.PLANE[1]:
        # xz plane
        img_2d = img[:, plane_i]
        img_2d = np.flipud(img_2d)
    elif out_plane == config.PLANE[2]:
        # yz plane, which requires a flip when original orientation is 
        # horizontal section
        # TODO: generalize to other original orientations
        img_2d = img[:, :, plane_i]
        #img_2d = np.swapaxes(img_2d, 1, 0)
        #aspect = 1 / aspect
        img_2d = np.flipud(img_2d)
    else:
        # xy plane (default)
        img_2d = img[plane_i]
    ax.imshow(img_2d, cmap=cmap, aspect=aspect, alpha=alpha)
    plot_support.hide_axes(ax)
    if title is not None:
        ax.set_title(title)


def plot_overlays(imgs, z, cmaps, title=None, aspect=1.0):
    """Plot images in a single row, with the final subplot showing an 
    overlay of all images.
    
    Args:
        imgs: List of 3D images to show.
        z: Z-plane to view for all images.
        cmaps: List of colormap names, which should be be the same length as 
            `imgs`, with the colormap applied to the corresponding image.
        title: Figure title; if None, will be given default title.
        aspect: Aspect ratio, which will be applied to all images; 
           defaults to 1.0.
    """
    # TODO: deprecated
    fig = plt.figure()
    fig.suptitle(title)
    imgs_len = len(imgs)
    gs = gridspec.GridSpec(1, imgs_len + 1)
    for i in range(imgs_len):
        print("showing img {}".format(i))
        _show_overlay(plt.subplot(gs[0, i]), imgs[i], z, cmaps[i], aspect)
    ax = plt.subplot(gs[0, imgs_len])
    for i in range(imgs_len):
        _show_overlay(ax, imgs[i], z, cmaps[i], aspect, alpha=0.5)
    if title is None:
        title = "Image overlays"
    gs.tight_layout(fig)
    plt.show()


def plot_overlays_reg(exp, atlas, atlas_reg, labels_reg, cmap_exp, 
                      cmap_atlas, cmap_labels, translation=None, title=None, 
                      out_plane=None, show=True):
    """Plot overlays of registered 3D images, showing overlap of atlas and 
    experimental image planes.
    
    Shows the figure on screen. If :attr:``config.savefig`` is set, 
    the figure will be saved to file with the extensive given by savefig.
    
    Args:
        exp: Experimental image.
        atlas: Atlas image, unregistered.
        atlas_reg: Atlas image, after registration.
        labels_reg: Atlas labels image, also registered.
        cmap_exp: Colormap for the experimental image.
        cmap_atlas: Colormap for the atlas.
        cmap_labels: Colormap for the labels.
        translation: Translation in (z, y, x) order for consistency with 
            operations on Numpy rather than SimpleITK images here; defaults 
            to None, in which case the chosen plane index for the 
            unregistered atlast will be the same fraction of its size as for 
            the registered image.
        title: Figure title; if None, will be given default title.
        out_plane: Output planar orientation.
        show: True if the plot should be displayed on screen; defaults to True.
    """
    fig = plt.figure()
    # give extra space to the first row since the atlas is often larger
    gs = gridspec.GridSpec(2, 3, height_ratios=[3, 2])
    resolution = config.resolutions[0]
    #size_ratio = np.divide(atlas_reg.shape, exp.shape)
    aspect = 1.0
    z = 0
    atlas_z = 0
    plane_frac = 2#5 / 2
    if out_plane is None:
        out_plane = config.plane
    if out_plane == config.PLANE[1]:
        # xz plane
        aspect = resolution[0] / resolution[2]
        z = exp.shape[1] // plane_frac
        if translation is None:
            atlas_z = atlas.shape[1] // plane_frac
        else:
            atlas_z = int(z - translation[1])
    elif out_plane == config.PLANE[2]:
        # yz plane
        aspect = resolution[0] / resolution[1]
        z = exp.shape[2] // plane_frac
        if translation is None:
            atlas_z = atlas.shape[2] // plane_frac
        else:
            # TODO: not sure why needs to be addition here
            atlas_z = int(z + translation[2])
    else:
        # xy plane (default)
        aspect = resolution[1] / resolution[2]
        z = exp.shape[0] // plane_frac
        if translation is None:
            atlas_z = atlas.shape[0] // plane_frac
        else:
            atlas_z = int(z - translation[0])
    print("z: {}, atlas_z: {}, aspect: {}".format(z, atlas_z, aspect))
    
    # invert any neg values (one hemisphere) to minimize range and match other
    # hemisphere
    labels_reg[labels_reg < 0] = np.multiply(labels_reg[labels_reg < 0], -1)
    vmin, vmax = np.percentile(labels_reg, (5, 95))
    print("vmin: {}, vmax: {}".format(vmin, vmax))
    labels_reg = exposure.rescale_intensity(labels_reg, in_range=(vmin, vmax))
    '''
    labels_reg = labels_reg.astype(np.float)
    lib_clrbrain.normalize(labels_reg, 1, 100, background=15000)
    labels_reg = labels_reg.astype(np.int)
    print(labels_reg[290:300, 20, 190:200])
    '''
    
    # experimental image and atlas
    _show_overlay(plt.subplot(gs[0, 0]), exp, z, cmap_exp, out_plane, aspect, 
                              title="Experiment")
    _show_overlay(plt.subplot(gs[0, 1]), atlas, atlas_z, cmap_atlas, out_plane, 
                  alpha=0.5, title="Atlas")
    
    # atlas overlaid onto experiment
    ax = plt.subplot(gs[0, 2])
    _show_overlay(ax, exp, z, cmap_exp, out_plane, aspect, title="Registered")
    _show_overlay(ax, atlas_reg, z, cmap_atlas, out_plane, aspect, 0.5)
    
    # labels overlaid onto atlas
    ax = plt.subplot(gs[1, 0])
    _show_overlay(ax, atlas_reg, z, cmap_atlas, out_plane, aspect, title="Labeled atlas")
    _show_overlay(ax, labels_reg, z, cmap_labels, out_plane, aspect, 0.5)
    
    # labels overlaid onto exp
    ax = plt.subplot(gs[1, 1])
    _show_overlay(ax, exp, z, cmap_exp, out_plane, aspect, title="Labeled experiment")
    _show_overlay(ax, labels_reg, z, cmap_labels, out_plane, aspect, 0.5)
    
    # all overlaid
    ax = plt.subplot(gs[1, 2])
    _show_overlay(ax, exp, z, cmap_exp, out_plane, aspect, title="All overlaid")
    _show_overlay(ax, atlas_reg, z, cmap_atlas, out_plane, aspect, 0.5)
    _show_overlay(ax, labels_reg, z, cmap_labels, out_plane, aspect, 0.3)
    
    if title is None:
        title = "Image Overlays"
    fig.suptitle(title)
    gs.tight_layout(fig)
    plot_support.save_fig(title, config.savefig)
    if show:
        plt.show()


def _bar_plots(ax, lists, errs, legend_names, x_labels, colors, y_label,
               padding=0.2, skip_all_zero=False, rotation=None,
               y_unit=None, vspans=None, vspan_lbls=None, vspan_alt_y=False,
               hline=None):
    """Generate grouped bar plots from lists, where corresponding elements 
    in the lists are grouped together.
    
    Data is given as a list of sublists. Each sublist contains a "set" of 
    values, with one value per "group." The number of groups is thus the 
    number of values per sublist, and the number of bars per group is 
    the number of sublists.
    
    Typically each sublist will represents an experimental set, such 
    as WT or het. Corresponding elements in each set are grouped together 
    to compare sets, such as WT vs het at time point 0.
    
    Args:
        ax: Axes.
        lists: Sequence of main value sequences to display, where each 
            main value sequence will be displayed as separate set of 
            bar plots with a legend entry. All main value sequences 
            should be the same size as one another. The number of 
            main value sequences will equal the number of legend groups, 
            and the number of entries in each main value sequence 
            will equal the number of bar groups.
        errs: Sequence of error sequences (eg standard deviation or 
            error), with a error sequence for each separate set of 
            bar plots. All error sequences should be the same size as one 
            another and each main value sequence in ``lists``.
        legend_names: Sequence of names to display in the legend. Length 
            should be the same as that of ``lists``. If None, a legend 
            will not be displayed.
        x_labels: Sequence of labels for each bar group, where the length 
            should be equal to that of each main value sequence in ``lists``.
        y_label: Y-axis label. Falls back to :meth:``plot_support.set_scinot`` 
            defaults.
        padding: Fraction each bar group's width that should be left 
            unoccupied by bars. Defaults to 0.2.
        skip_all_zero: True to skip any data list that contains only 
            values below :attr:``config.POS_THRESH``; defaults to False.
        rotation: Degrees of x-tick label rotation; defaults to None for 
            vertical text (90 degrees, where 0 degrees is horizontal).
        y_unit: Measurement unit for y-axis; defaults to None, falling 
            back to :meth:``plot_support.set_scinot``.
        vspans: Shade with vertical spans with indices of bar groups 
            at which alternating colors; defaults to None.
        vspan_lbls (List[str]): Sequence of labels of vertical spans; 
            defaults to None.
        vspan_alt_y (bool): True to alternate y-axis placement to avoid 
            overlap; defaults to False.
        hline (str): One of :attr:`config.STR_FN` for a function to apply
            to each list in ``lists`` for a horizontal line to be drawn
            at this y-value; defaults to None.
    """
    if len(lists) < 1: return
    if rotation is None:
        # default rotation to 90 degrees for "no" rotation (vertical text)
        rotation = 90
    hline_fn = None
    if hline:
        # retrieve function for horizontal line summary metric
        hline_fn = config.STR_FN.get(hline.lower())
    bars = []
    
    # convert lists to Numpy arrays to allow fancy indexing
    lists = np.array(lists)
    if errs: errs = np.array(errs)
    x_labels = np.array(x_labels)
    #print("lists: {}".format(lists))
    
    mask = []
    if skip_all_zero:
        # skip bar groups where all bars would be ~0
        mask = np.all(lists > config.POS_THRESH, axis=0)
    #print("mask: {}".format(mask))
    if np.all(mask):
        print("skip none")
    else:
        print("skipping {}".format(x_labels[~mask]))
        x_labels = x_labels[mask]
        lists = lists[..., mask]
        # len(errs) may be > 0 when errs.size == 0?
        if errs is not None and errs.size > 0:
            errs = errs[..., mask]
    num_groups = len(lists[0])
    num_sets = len(lists) # num of bars per group
    indices = np.arange(num_groups)
    #print("lists:\n{}".format(lists))
    if lists.size < 1: return
    width = (1.0 - padding) / num_sets # width of each bar
    #print("x_labels: {}".format(x_labels))
    
    if vspans is not None:
        # show vertical spans alternating in white and black; assume 
        # background is already white, so simply skip white shading
        xs = vspans - padding / 2
        num_xs = len(xs)
        for i, x in enumerate(xs):
            if i % 2 == 0: continue
            end = xs[i + 1] if i < num_xs - 1 else num_groups
            ax.axvspan(x, end, facecolor="k", alpha=0.2)
            
    # show each list as a set of bar plots so that corresponding elements in 
    # each list will be grouped together as bar groups
    for i in range(num_sets):
        err = None if errs is None or errs.size < 1 else errs[i]
        #print("lens: {}, {}".format(len(lists[i]), len(x_labels)))
        #print("showing list: {}, err: {}".format(lists[i], err))
        num_bars = len(lists[i])
        err_dict = {"elinewidth": width * 20 / num_bars}
        bars.append(
            ax.bar(indices + width * i, lists[i], width=width, color=colors[i], 
                   linewidth=0, yerr=err, error_kw=err_dict, align="edge"))
        if hline_fn:
            # dashed horizontal line at the given metric output
            ax.axhline(hline_fn(lists[i]), color=colors[i], linestyle="--")
    
    # show y-label with any unit in scientific notation
    plot_support.set_scinot(ax, lbls=(y_label,), units=(y_unit,))
    # draw x-tick labels with smaller font for increasing number of labels
    font_size = plt.rcParams["axes.titlesize"]
    if libmag.is_number(font_size):
        # scale font size of x-axis labels by a sigmoid function to rapidly 
        # decrease size for larger numbers of labels so they don't overlap
        font_size *= (math.atan(len(x_labels) / 10 - 5) * -2 / math.pi + 1) / 2
    font_dict = {"fontsize": font_size}
    # draw x-ticks based on number of bars per group and align to right 
    # since center shifts the horiz middle of the label to the center; 
    # rotation_mode in dict helps but still slightly off
    ax.set_xticks(indices + width * len(lists) / 2)
    ax.set_xticklabels(
        x_labels, rotation=rotation, horizontalalignment="right", 
        fontdict=font_dict)
    # translate to right since "right" alignment shift the right of labels 
    # too far to the left of tick marks; shift less with more groups
    offset = transforms.ScaledTranslation(
        30 / np.cbrt(num_groups) / ax.figure.dpi, 0, ax.figure.dpi_scale_trans)
    for lbl in ax.xaxis.get_majorticklabels():
        lbl.set_transform(lbl.get_transform() + offset)
    
    if vspans is not None and vspan_lbls is not None:
        # show labels for vertical spans
        ylims = ax.get_ylim()
        y_span = abs(ylims[1] - ylims[0])
        y_top = max(ylims)
        for i, x in enumerate(xs):
            end = xs[i + 1] if i < num_xs - 1 else num_groups
            x = (x + end) / 2
            # position 4% down from top in data coordinates
            y_frac = 0.04
            if vspan_alt_y and i % 2 != 0:
                # shift alternating labels further down to avoid overlap
                y_frac += 0.03
            y = y_top - y_span * y_frac
            ax.text(
                x, y, vspan_lbls[i], color="k", horizontalalignment="center")
    
    if legend_names:
        ax.legend(bars, legend_names, loc="best", fancybox=True, framealpha=0.5)


def plot_bars(path_to_df, data_cols=None, err_cols=None, legend_names=None, 
              col_groups=None, groups=None, y_label=None, y_unit=None, 
              size=None, show=True, col_vspan=None, vspan_fmt=None,
              col_wt=None, df=None, x_tick_labels=None, rotation=None,
              save=True, hline=None, ax=None, suffix=None, **kwargs):
    """Plot grouped bars from Pandas data frame.
    
    Each data frame row represents a group, and each chosen data column 
    will be plotted as a separate bar within each group.
    
    Args:
        path_to_df: Path from which to read saved Pandas data frame.
            The figure will be saved to file if :attr:``config.savefig`` is 
            set, using this same path except with the savefig extension.
        data_cols: Sequence of names of columns to plot as separate sets 
            of bars, where each row is part of a separate group. Defaults 
            to None, which will plot all columns except ``col_groups``.
        err_cols: Sequence of column names with relative error values 
            corresponding to ``data_cols``. Defaults to None, in which 
            case matching columns with "_err" as suffix will be used for 
            error bars if present.
        legend_names: Sequence of names for each set of bars. 
            Defaults to None, which will use ``data_cols`` for names. 
            Use "" to not display a legend.
        col_groups: Name of column specifying names of each group. 
            Defaults to None, which will use the first column for names.
        groups: Sequence of groups to include and by which to sort; 
            defaults to None to include all groups found from ``col_groups``.
        y_label: Name of y-axis; defaults to None to use 
            :attr:`config.plot_labels`. ``(None, )`` prevents label display.
        y_unit: Measurement unit for y-axis; defaults to None to use 
            :attr:`config.plot_labels`. ``(None, )`` prevents unit display.
        size: Sequence of ``width, height`` to size the figure; defaults 
            to None.
        show: True to display the image; otherwise, the figure will only 
            be saved to file, if :attr:``config.savefig`` is set.  
            Defaults to True.
        col_vspan: Name of column with values specifying groups demaracted 
            by vertical spans. Each change in value when taken in sequence 
            will specify a new span in alternating background colors. 
            Defaults to None.
        vspan_fmt: String to format with values from ``col_vspan``; defaults 
            to None to simply use the unformatted values.
        col_wt: Name of column to use for weighting, where the size of 
            bars and error bars will be adjusted as fractions of the max 
            value; defaults to None.
        df: Data frame to use; defaults to None. If set, this data frame
            will be used instead of loading from ``path``.
        x_tick_labels (List[str]): Sequence of labels for each bar group 
            along the x-axis; defaults to None to use ``groups`` instead. 
        rotation: Degrees of x-tick label rotation; defaults to None.
        save (bool): True to save the plot; defaults to True.
        hline (str): One of :attr:`config.STR_FN` for a function to apply
            to each list in ``lists`` for a horizontal line to be drawn
            at this y-value; defaults to None.
        ax (:class:`matplotlib.image.Axes`): Matplotlib axes; defaults to None.
        suffix: String to append to output path before extension;
            defaults to None to ignore.
        kwargs (Any): Extra arguments to :meth:`plot_support.decorate_plot`.
    
    Returns:
        :obj:`matplotlib.image.Axes`, str: Plot axes and save path without
        extension.
    
    """
    # load data frame from CSV and setup figure
    if df is None:
        df = pd.read_csv(path_to_df)
    if not y_label:
        y_label = config.plot_labels[config.PlotLabels.Y_LABEL]
    if not y_unit:
        y_unit = config.plot_labels[config.PlotLabels.Y_UNIT]

    if ax is None:
        fig, gs = plot_support.setup_fig(1, 1, size)
        ax = plt.subplot(gs[0, 0])
    
    if col_groups is None:
        # default to using first col as group names
        col_groups = df.columns.values.tolist()[0]
    
    if data_cols is None:
        # default to using all but the group column as data cols
        data_cols = df.columns.values.tolist()
        data_cols.remove(col_groups)
    
    if groups is not None:
        # get rows in groups and sort by groups that exist
        groups = np.array(groups)
        df = df.loc[df[col_groups].isin(groups)]
        groups_found = np.isin(groups, df[col_groups])
        groups_missing = groups[~groups_found]
        if len(groups_missing) > 0:
            print("could not find these groups:", groups_missing)
            groups = groups[groups_found]
        df = df.set_index(col_groups).loc[groups].reset_index()
    
    vspans = None
    vspan_lbls = None
    if col_vspan is not None:
        # further group bar groups by vertical spans with location based 
        # on each change in value in col_vspan
        # TODO: change .values to .to_numpy when Pandas req >= 0.24
        vspan_vals = df[col_vspan].values
        vspans = np.insert(
            np.where(vspan_vals[:-1] != vspan_vals[1:])[0] + 1, 0, 0)
        vspan_lbls = [vspan_fmt.format(val) if vspan_fmt else str(val) 
                      for val in vspan_vals[vspans]]
    
    if err_cols is None:
        # default to columns corresponding to data cols with suffix appended 
        # if those columns exist
        err_cols = []
        for col in data_cols:
            col += "_err"
            err_cols.append(col if col in df else None)
    
    if legend_names is None:
        # default to using data column names for names of each set of bars
        legend_names = [name.replace("_", " ") for name in data_cols]
    
    wts = 1
    if col_wt is not None:
        # weight by fraction of weights with max weight
        wts = df[col_wt]
        wts /= max(wts)
    
    # build lists to plot
    lists = []
    errs = []
    bar_colors = []
    for i, (col, col_err) in enumerate(zip(data_cols, err_cols)):
        # each column gives a set of bars, where each bar will be in a 
        # separate bar group
        lists.append(df[col] * wts)
        errs_dfs = None
        if libmag.is_seq(col_err):
            # asymmetric error bars
            errs_dfs = [df[e] * wts for e in col_err]
        elif col_err is not None:
            errs_dfs = df[col_err] * wts
        errs.append(errs_dfs)
        bar_colors.append("C{}".format(i))

    # set labels, title, etc
    decorate_plot(ax, **kwargs)

    # plot bars
    if len(errs) < 1: errs = None
    x_labels = x_tick_labels if x_tick_labels else df[col_groups]
    # TODO: remove more redundant decorations
    _bar_plots(
        ax, lists, errs, legend_names, x_labels, bar_colors, y_label, 
        y_unit=y_unit, vspans=vspans, vspan_lbls=vspan_lbls,
        rotation=rotation, hline=hline)

    # save and display
    if save:
        out_path = libmag.make_out_path(path_to_df, suffix=suffix)
        plot_support.save_fig(out_path, config.savefig)
    if show: plt.show()
    return ax


def plot_lines(path_to_df, x_col, data_cols, linestyles=None, labels=None, 
               title=None, size=None, show=True, suffix=None, 
               colors=None, df=None, groups=None, ignore_invis=False, 
               units=None, marker=None, err_cols=None, prefix=None, save=True,
               ax=None, col_annot=None, line_label_style=None, linewidth=None):
    """Plot a line graph from a Pandas data frame.
    
    Args:
        path_to_df: Path from which to read saved Pandas data frame.
            The figure will be saved to file if :attr:``config.savefig`` is 
            set, using this same path except with the savefig extension.
        x_col: Name of column to use for x.
        data_cols: Sequence of column names to plot as separate lines.
            Hierarchical columns will be plotted with the same color
            and style unless ``groups`` is specified. Legend names will
            correspond to these colum names.
        linestyles: Sequence of styles to use for each line; defaults to 
            None, in which case "-" will be used for all lines if
            ``groups`` is None, or each group will use a distinct style.
        labels (List[str]): ``(y_label, x_label)`` to display; defaults 
            to None to use :attr:`config.plot_labels`. Can explicitly set a 
            value to None to prevent unit display. 
        title: Title of figure; defaults to None.
        size: Sequence of ``width, height`` to size the figure; defaults 
            to None.
        show: True to display the image; otherwise, the figure will only 
            be saved to file, if :attr:``config.savefig`` is set.  
            Defaults to True.
        suffix: String to append to output path before extension; 
            defaults to None to ignore.
        colors: Sequence of colors for plot lines; defaults to None to use 
            :meth:``colormaps.discrete_colormap`` while prioritizing the
            default ``CN`` color cycler (``C0``, ``C1``, etc).
        df: Data frame to use; defaults to None. If set, this data frame
            will be used instead of loading from ``path``.
        groups (List[str]): Sequence of groups names within each data column
            to plot separately, assuming that each data column has sub-columns
            that include these group names. If given, all lines within a
            group will have the same style, and a separate group legend will
            be displayed with these line styles. To simply plot with
            different colors, use separate data colums in ``data_cols``
            instead. Defaults to None.
        ignore_invis: True to ignore lines that aren't displayed,
            such as those with only a single value; defaults to False.
        units (List[str]): ``(y_unit, x_unit)`` to display; defaults 
            to None to use :attr:`config.plot_labels`. Can explicitly set a 
            value to None to prevent unit display.
        marker (str): Marker style for points; defaults to None.
        err_cols (List[str]): Sequence of column names with relative error 
            values corresponding to ``data_cols``; defaults to None.
        prefix: Base path for figure output if :attr:``config.savefig`` 
            is set; defaults to None to use ``path_to_df``.
        save (bool): True to save the plot; defaults to True.
        ax (:obj:`matplotlib.image.Axes`: Image axes object; defaults to
            None to generate a new figure and subplot.
        col_annot (str): Name of column for point annotations; defaults to
            None. The first line will be annotated based on this column.
        line_label_style (str): "legend" to label lines through a legend,
            "end" to label the last point of each line, and any other string
            to not label lines; defaults to None to use "legend".
        linewidth (float): Line width; defaults to None.
    
    Returns:
        :obj:`matplotlib.Axes`: Axes object.
    
    """

    def to_ignore(arr):
        # True if set to ignore and fewer than 2 points to plot
        return ignore_invis and np.sum(~np.isnan(arr)) < 2
    
    def plot_line():
        # plot line and error bars
        line_lbl = None
        if line_label_style == "end":
            ax.annotate(label, (xs.iloc[-1], ys.iloc[-1]))
        elif line_label_style == "legend":
            line_lbl = label
        line = ax.plot(
            xs, ys, color=colors[i], linestyle=linestyles[i],
            linewidth=linewidth, label=line_lbl, marker=marker)
        if errs is not None:
            ax.errorbar(xs, df_col, errs)
        return line
    
    # load data frame from CSV unless already given and setup figure
    if df is None:
        df = pd.read_csv(path_to_df)
    if ax is None:
        fig, gs = plot_support.setup_fig(1, 1, size)
        ax = plt.subplot(gs[0, 0])

    if colors is None:
        # default to discrete colors starting with CN colors
        colors = colormaps.discrete_colormap(
            len(data_cols), prioritize_default="cn", seed=config.seed) / 255

    if linestyles is None:
        # default to solid line for all lines if no groups or cycling
        # through all main line styles for each group
        if groups is None:
            linestyles = ["-"] * len(data_cols)
        else:
            linestyles = ["-", "--", ":", "-."]
    if groups is not None:
        # simply repeat line style sets if groups exceed existing styles
        linestyles = linestyles * (len(groups) // (len(linestyles) + 1) + 1)
    if line_label_style is None:
        line_label_style = "legend"

    annots = None
    if col_annot:
        # get point annotations
        annots = df[col_annot]

    # plot selected columns with corresponding styles
    xs = df[x_col]
    lines = []
    lines_groups = None if groups is None else []
    for i, col in enumerate(data_cols):
        # plot columns with unique colors
        df_col = df[col]
        label = str(col).replace("_", " ")
        df_err = df[err_cols[i]] if err_cols else None
        if groups is None:
            if to_ignore(df_col): continue
            ys = df_col
            errs = df_err
            lines.extend(plot_line())
        else:
            # prioritize solid line for main legend
            labelj = linestyles.index("-") if "-" in linestyles else 0
            for j, group in enumerate(groups):
                # plot all lines within group with same color but unique styles
                df_group = df_col[group]
                if to_ignore(df_group): continue
                ys = df_group
                errs = None if df_err is None else df_err[group]
                lines_group = plot_line()
                if j == labelj:
                    # add first line to main legend
                    lines.extend(lines_group)
                if i == 0:
                    # for first data col, add dummy lines only for group legend
                    lines_groups.extend(
                        ax.plot(
                            [], [], color="k", linestyle=linestyles[j],
                            label=group))
        if i == 0 and annots is not None:
            # annotate each point of first line based on annotation col
            for x, y, annot in zip(xs, df_col, annots):
                ax.annotate(annot, (x, y))

    if line_label_style == "legend":
        # add legends, using "best" location for main legend unless also
        # showing a group legend, in which case locations are set explicitly
        legend_main_loc = "best"
        legend_group = None
        if lines_groups is not None:
            # group legend from empty lines to show line style
            legend_group = ax.legend(
                lines_groups, [l.get_label() for l in lines_groups],
                loc="lower right", fancybox=True, framealpha=0.5)
            legend_main_loc = "upper left"
        ax.legend(
            lines, [l.get_label() for l in lines], loc=legend_main_loc,
            fancybox=True, framealpha=0.5)
        if legend_group is not None:
            # only last legend appears to be shown so need to add prior legends
            ax.add_artist(legend_group)

    # add supporting plot components
    plot_support.set_scinot(ax, lbls=labels, units=units)
    if title: ax.set_title(title)
    
    # save and display
    out_path = libmag.make_out_path(path_to_df, prefix, suffix)
    if save: plot_support.save_fig(out_path, config.savefig)
    if show: plt.show()
    return ax


def plot_scatter(path, col_x, col_y, col_annot=None, cols_group=None,
                 names_group=None, fig_size=None, show=True, suffix=None,
                 df=None, xy_line=False, col_size=None, size_mult=5,
                 annot_arri=None, alpha=None, legend_loc="best",
                 ax=None, save=True, annot_thresh_fn=None,
                 colors: Optional[Sequence] = None, **kwargs):
    """Generate a scatter plot from a data frame or CSV file.
    
    Args:
        path: Path from which to read a saved Pandas data frame and the 
            path basis to save the figure if :attr:``config.savefig`` is set.
        col_x: Name of column to plot as x-values. Can also be a sequence 
            of names to define groups with corresponding `col_y` values.
        col_y: Name of column to plot as corresponding y-values. Can 
            also be a sequence corresponding to that of `col_x`.
        col_annot: Name of column with annotations for each point; defaults to
            None. Can be the name of the index column.
        cols_group (Sequence[str]): Sequence of column names; defaults to None.
            Each unique combination in these columns specifies a group
            to plot separately.
        names_group (Sequence[str]): Sequence of names to display;
            defaults to None, in which case a name based on ``cols_groups``
            will be used instead. Length should equal that of groups based
            on ``cols_group``.
        fig_size: Sequence of ``width, height`` to size the figure; defaults 
            to None.
        show: True to display the image; otherwise, the figure will only 
            be saved to file, if :attr:``config.savefig`` is set.  
            Defaults to True.
        suffix: String to append to output path before extension; 
            defaults to None to ignore.
        df: Data frame to use; defaults to None. If set, this data frame 
            will be used instead of loading from ``path``.
        xy_line: Show an xy line; defaults to False.
        col_size: Name of column from which to scale point sizes, where 
            the max value in the column is 1; defaults to None.
        size_mult: Point size multiplier; defaults to 5.
        annot_arri: Int as index or slice of indices of annotation value
            if the annotation is a string that can be converted into a
            Numpy array; defaults to None.
        alpha (float): Point transparency value, from 0-1; defaults to None,
            in which case 1.0 will be used.
        legend_loc (str): Legend location, which should be one of
            :attr:``plt.legend.loc`` values; defaults to "best".
        ax (:class:`matplotlib.image.Axes`): Matplotlib axes; defaults to None.
        save (bool): True to save the plot; defaults to True.
        annot_thresh_fn (func): Function accepting ``x, y`` and returning
            a boolean indicated whether to annotate the given point;
            defaults to False.
        colors: Color or sequence of colors for each point; defaults to None.
            If None, distinct colors are auto-generated for each pair of x-y
            column or for each group.
        kwargs (Any): Extra arguments to :meth:`plot_support.decorate_plot`.
    
    Returns:
        :class:`matplotlib.image.Axes`: Matplotlib plot.
    
    """
    def plot():
        # plot a paired sequence of x/y's and annotate
        
        # single RGB(A) colors use the "color" param, while sequences of colors
        # use the "c" param
        color = colors[i]
        scat_args = {"color" if mat_colors.is_color_like(color) else "c": color}
        
        ax.scatter(
            xs, ys, s=sizes_plot, label=label, **scat_args, 
            marker=markers[i])
        if col_annot:
            # annotate each point with val from annotation col, which can be
            # the name of the index
            annots = None
            if col_annot in df_group:
                annots = df_group[col_annot]
            elif col_annot == df_group.index.name:
                annots = df_group.index
            if annots is not None:
                for xan, yan, annot in zip(xs, ys, annots):
                    if annot_arri is not None:
                        # attempt to convert string into array to extract
                        # the given values
                        annot_arr = libmag.npstr_to_array(annot)
                        if annot_arr is not None:
                            annot = annot_arr[annot_arri]
                    if annot_thresh_fn and not annot_thresh_fn(xan, yan):
                        continue
                    dec_digits = 0 if libmag.is_int(annot) else 3
                    ax.annotate(
                        f"{libmag.format_num(annot, dec_digits, False)}",
                        (xan, yan))
    
    # load data frame from CSV and setup figure
    if df is None:
        df = pd.read_csv(path)
    if ax is None:
        fig, gs = plot_support.setup_fig(1, 1, fig_size)
        ax = plt.subplot(gs[0, 0])
    
    sizes = size_mult
    if col_size is not None:
        # scale point sizes based on max val in given col
        sizes = df[col_size]
        sizes *= size_mult / np.amax(sizes)
    
    if alpha is None:
        alpha = 1.0
    # scale to 255 for colormap generation
    alpha = int(alpha * 255)
    
    # point markers
    markers = ["o", "v", "^", "d", "<", ">"]
    
    # plot selected columns
    sizes_plot = sizes
    df_group = df
    if libmag.is_seq(col_x):
        # treat each pair of col_y and col_y values as a group
        num_groups = len(col_x)
        if colors is None:
            colors = colormaps.discrete_colormap(
                num_groups, prioritize_default="cn", seed=config.seed,
                alpha=alpha) / 255
        markers = libmag.pad_seq(markers, num_groups)
        for i, (x, y) in enumerate(zip(col_x, col_y)):
            label = x if names_group is None else names_group[i]
            xs = df[x]
            ys = df[y]
            plot()
    else:
        # set up groups
        df_groups = None
        if not cols_group:
            groups = [""]  # default to single group of empty string
        else:
            # treat each unique combination of cols_group values as 
            # a separate group
            for col in cols_group:
                df_col = df[col].astype(str)
                if df_groups is None:
                    df_groups = df_col
                else:
                    df_groups = df_groups.str.cat(df_col, sep=",")
            groups = df_groups.unique()
        num_groups = len(groups)
        markers = libmag.pad_seq(markers, num_groups)
        if colors is None:
            colors = colormaps.discrete_colormap(
                num_groups, prioritize_default="cn", seed=config.seed,
                alpha=alpha) / 255
        for i, group in enumerate(groups):
            # plot all points in each group with same color
            df_group = df
            sizes_plot = sizes
            label = None
            if group != "":
                mask = df_groups == group
                df_group = df.loc[mask]
                if col_size is not None: sizes_plot = sizes_plot[mask]
                if names_group is None:
                    # make label from group names and values
                    label = ", ".join(
                        ["{} {}".format(name, libmag.format_num(val, 3))
                         for name, val in zip(cols_group, group.split(","))])
                else:
                    # use given group name directly
                    label = names_group[i]
            xs = df_group[col_x]
            ys = df_group[col_y]
            plot()
    
    if xy_line:
        # add xy line
        xy_line = np.linspace(*ax.get_xlim())
        ax.plot(xy_line, xy_line)
    
    # set labels, title, etc
    decorate_plot(ax, **kwargs)
    
    # constrained layout tightens layout to avoid compressing the graph
    # for large legends
    if len(ax.get_legend_handles_labels()[1]) > 0:
        ax.legend(loc=legend_loc, fancybox=True, framealpha=0.5)
    
    # save and display
    if save:
        out_path = libmag.make_out_path(path, suffix=suffix)
        plot_support.save_fig(out_path, config.savefig)
    if show: plt.show()
    return ax


def plot_probability(path, conds, metric_cols, col_size, **kwargs):
    """Generate a probability plot such as that used in Q-Q or P-P plots.
    
    Serves as a wrapper for :meth:`plot_scatter` with the assumption that
    matching columns for each of two conditions describe each point.
    
    Args:
        path: Path from which to read a saved Pandas data frame and the 
            path basis to save the figure if :attr:``config.savefig`` is set.
        conds: Sequence of conditions, the first of which will be used 
            to find the x-values for each metric, and the second for y-values.
        metric_cols: Sequence of column name prefixes for each metric to 
            plot. Metric column names are assumed to have these values 
            combined with a condition, separated by "_".
        col_size: Name of column from which to scale point sizes, where 
            the max value in the column is 1; defaults to None.
        **kwargs: Additional keyword arguments to pass to 
            :meth:``plot_scatter``.
    """
    metric_cond_cols = []
    for cond in conds:
        metric_cond_cols.append(
            ["{}_{}".format(col, cond) for col in metric_cols])
    plot_scatter(
        path, metric_cond_cols[0], metric_cond_cols[1], None, None, 
        names_group=metric_cols, 
        ylabel=conds[1].capitalize(), xlabel=conds[0].capitalize(),
        xy_line=True, col_size=col_size, **kwargs)


def plot_roc(df, show=True, annot_arri=None, **kwargs):
    """Plot ROC curve generated from :meth:``mlearn.grid_search``.
    
    Args:
        df (:class:`pandas.DataFrame`): Data frame generated from
            :meth:``mlearn.parse_grid_stats``.
        show (bool): True to display the plot in :meth:``plot_scatter``;
            defaults to True.
        annot_arri (int): Int as index or slice of indices of annotation value
            if the annotation is a string that can be converted into a
            Numpy array; defaults to None.
        kwargs (Any): Extra arguments to :meth:`plot_support.plot_scatter`.
    
    Returns:
        :class:`matplotlib.image.Axes`: Matplotlib plot.
    
    """
    # names of hyperparameters for each group name, with hyperparameters 
    # identified by param prefix
    cols_group = [col for col in df
                  if col.startswith(mlearn.GridSearchStats.PARAM.value)]
    start = len(mlearn.GridSearchStats.PARAM.value)
    names_group = [col[start+1:] for col in cols_group]
    
    # add extra arguments unless already set in kwargs
    libmag.add_missing_keys({
        "xlabel": "False Discovery Rate",
        "ylabel": "Sensitivity",
        "xlim": (0, 1),
        "ylim": (0, 1),
        "title": "Nuclei Detection ROC Over {}".format(names_group[-1]),
    }, kwargs)
    if "path" in kwargs:
        path = kwargs["path"]
        del kwargs["path"]
    else:
        path = "gridsearch_roc"
    
    # plot sensitivity by FDR, annotating with col of final hyperparameter
    # rather than using this col in the group specification
    return plot_scatter(
        path, mlearn.GridSearchStats.FDR.value,
        mlearn.GridSearchStats.SENS.value, cols_group[-1], cols_group[:-1],
        names_group, df=df, show=show, annot_arri=annot_arri,
        legend_loc="lower right", **kwargs)


def plot_histogram(df, path, col_x, ax=None, size=None, save=True, suffix=None,
                   show=False, **kwargs):
    """Geneate a histogram plot.
    
    Args:
        df (:class:`pandas.DataFrame`): Data frame to plot.
        path (str): Path to data frame to use if ``df`` is None, also used
            as the basis for output path.
        col_x (str): Name of column with values to plot.
        ax (:class:`matplotlib.axes.Axes`): Matplotlib axes; defaults to
            None to generate a new figure with axes.
        size (Sequence[float]): Sequence of ``width, height`` to size the
            figure; defaults to None.
        save (bool): True to save the plot; defaults to True.
        suffix: String to append to output path before extension;
            defaults to None to ignore.
        show: True to display the image; otherwise, the figure will only
            be saved to file, if :attr:``config.savefig`` is set.
            Defaults to True.
        kwargs (Any): Extra arguments to :meth:`decorate_plot`.

    Returns:
        :class:`matplotlib.axes.Axes`: Matplotlib axes.

    """
    # load data frame from CSV unless already given and set up figure
    if df is None:
        df = pd.read_csv(path)
    if ax is None:
        fig, gs = plot_support.setup_fig(1, 1, size)
        ax = plt.subplot(gs[0, 0])
    
    # generate histogram
    n, bins, patches = ax.hist(df[col_x])
    decorate_plot(ax, **kwargs)
    
    # save and display plot if indicated
    if save:
        out_path = libmag.make_out_path(path, suffix=suffix)
        plot_support.save_fig(out_path, config.savefig)
    if show: plt.show()
    return ax


def plot_image(img, path=None, show=False):
    """Plot a single image in a borderless figure, with option to export 
    directly to file.
    
    Args:
        img (:obj:`np.ndarray`): Image as a Numpy array to display.
        path (str): Path to save image. Defaults to None to not save.
        show (bool): True to show the image; defaults to False, which will
            plot the image for saving.
    """
    # plot figure without frame, axes, or border space
    fig, gs = plot_support.setup_fig(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    plot_support.hide_axes(ax)
    ax.imshow(img)
    plot_support.fit_frame_to_image(fig, img.shape, None)
    
    if path:
        # use user-supplied ext if given
        ext = config.savefig
        if not ext:
            # use path extension if available, or default to png
            path_split = os.path.splitext(path)
            ext = path_split[1][1:] if path_split[1] else config.DEFAULT_SAVEFIG
            print(path_split, ext)
        plot_support.save_fig(path, ext)
    if show: plt.show()
    plt.close()  # prevent display during next show call


def decorate_plot(ax, title=None, xlabel=None, ylabel=None, xunit=None,
                  yunit=None, xlim=None, ylim=None, xscale=None, yscale=None,
                  **kwargs):
    """Decorate a plot with text and configure limits and scaling.

    Args:
        ax (:class:`matplotlib.image.Axes`): Matplotlib plot.
        title (str): Title of figure; defaults to None.
        xlabel (str): X-axis label; defaults to None to use
            :attr:`config.plot_labels`. Can explicitly set to None to prevent
            unit display.
        ylabel (str): Y-axis label; defaults to None to use
            :attr:`config.plot_labels`. Can explicitly set to None to prevent
            unit display.
        xunit (str): X-axis label unit; defaults to None.
        yunit (str): Y-axis label unit; defaults to None.:
        xlim (Sequence[float]): Sequence of min and max boundaries for the
            x-axis; defaults to None.
        ylim (Sequence[float]): Sequence of min and max boundaries for the
            y-axis; defaults to None.
        xscale (str): Scale mode for :meth:`plot_support.scale_axes` x-axis;
            defaults to None to ignore.
        yscale (str): Scale mode for :meth:`plot_support.scale_axes` y-axis;
            defaults to None to ignore.
        **kwargs (Any): Additional arguments, which will be ignored.

    Returns:
        :class:`matplotlib.image.Axes`: Matplotlib plot.

    """
    if config.verbose:
        print("Parameters not recognized and ignored for plot decorations:")
        print(kwargs)
    
    # set x/y axis limits if given
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    
    # axes scaling must follow after scientific notation since non-linear
    # formatters are not compatible with scinot
    plot_support.set_scinot(ax, lbls=(ylabel, xlabel), units=(yunit, xunit))
    plot_support.scale_axes(ax, xscale, yscale)
    
    # set title if given
    if title: ax.set_title(title)
    
    return ax


def setup_style(style=None, rc_params=None):
    """Setup Matplotlib styles and RC parameter themes.
    
    Both styles and themes default to those specified in :mod:`config`.
    
    Args:
        style (str): Name of Matplotlib style to apply. Defaults to None to
            use the style specified in :attr:``config.matplotlib_style``.
        rc_params (List[Enum]): Sequence of :class:`config.Themes` enums
            specifying custom themes to apply after setting the style.
            Themes will be applied in the order listed. Defaults to None,
            which will use the :attr:`config.rc_params` value.
    """
    #print(plt.style.available)
    if style is None:
        style = config.matplotlib_style
    if rc_params is None:
        rc_params = config.rc_params
    print("setting up Matplotlib style", style)
    plt.style.use(style)
    for rc in rc_params:
        if rc is config.Themes.DARK:
            # dark theme requires darker widgets for white text
            config.widget_color = 0.6
        print("applying theme", rc.name)
        pylab.rcParams.update(rc.value)


def post_plot(ax, out_path=None, save_ext=None, show=False):
    """Post plot adjustments, followed by saving and display.
    
    Handles additional :attr:`config.plot_labels` values.
    
    Args:
        ax (:obj:`matplotlib.image.Axes`: Image axes object.
        out_path (str): String to save path without extension; defaults
            to None. Both ``out_path`` and ``save_ext`` must be given to save.
        save_ext (str): String to save extension; defaults to None.
        show (bool): True to show the plot.

    """
    x_lim = config.plot_labels[config.PlotLabels.X_LIM]
    y_lim = config.plot_labels[config.PlotLabels.Y_LIM]
    if x_lim is not None:
        ax.set_xlim(*x_lim)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    if out_path and save_ext:
        plot_support.save_fig(out_path, save_ext)
    else:
        print("Figure not saved as both output path ({}) and file "
              "extension ({}) are required".format(out_path, save_ext))
    if show:
        plt.show()


def main(ax=None):
    """Perform 2D plot tasks.
    
    Args:
        ax (:class:`matplotlib.image.Axes`): Matplotlib plot.
    
    """
    # collect config settings
    size = config.plot_labels[config.PlotLabels.SIZE]
    plot_2d_type = libmag.get_enum(
        config.plot_2d_type, config.Plot2DTypes)
    annot_col = config.plot_labels[config.PlotLabels.ANNOT_COL]
    marker = config.plot_labels[config.PlotLabels.MARKER]
    scale_x = config.plot_labels[config.PlotLabels.X_SCALE]
    scale_y = config.plot_labels[config.PlotLabels.Y_SCALE]
    
    # perform 2D plot task, deferring save until the post-processing step
    if plot_2d_type is config.Plot2DTypes.BAR_PLOT:
        # generic barplot
        title = config.plot_labels[config.PlotLabels.TITLE]
        x_tick_lbls = config.plot_labels[config.PlotLabels.X_TICK_LABELS]
        data_cols = config.plot_labels[config.PlotLabels.Y_COL]
        if data_cols is not None and not libmag.is_seq(data_cols):
            data_cols = (data_cols, )
        y_lbl = config.plot_labels[config.PlotLabels.Y_LABEL]
        y_unit = config.plot_labels[config.PlotLabels.Y_UNIT]
        col_wt = config.plot_labels[config.PlotLabels.WT_COL]
        col_groups = config.plot_labels[config.PlotLabels.GROUP_COL]
        legend_names = config.plot_labels[config.PlotLabels.LEGEND_NAMES]
        hline = config.plot_labels[config.PlotLabels.HLINE]
        ax = plot_bars(
            config.filename, data_cols=data_cols, 
            legend_names=legend_names, col_groups=col_groups, title=title,
            y_label=y_lbl, y_unit=y_unit, hline=hline,
            size=size, show=False, groups=config.groups, 
            prefix=config.prefix, save=False,
            col_wt=col_wt, x_tick_labels=x_tick_lbls, rotation=45)
    
    elif plot_2d_type is config.Plot2DTypes.BAR_PLOT_VOLS_STATS:
        # barplot for data frame from R stats from means/CIs
        ax = plot_bars(
            config.filename, data_cols=("original.mean", "smoothed.mean"), 
            err_cols=("original.ci", "smoothed.ci"), 
            legend_names=("Original", "Smoothed"), col_groups="RegionName", 
            size=size, show=False, groups=config.groups, save=False,
            prefix=config.prefix)
    
    elif plot_2d_type is config.Plot2DTypes.BAR_PLOT_VOLS_STATS_EFFECTS:
        # barplot for data frame from R stats test effect sizes and CIs
        
        # setup labels
        title = config.plot_labels[config.PlotLabels.TITLE]
        x_tick_lbls = config.plot_labels[config.PlotLabels.X_TICK_LABELS]
        y_lbl = config.plot_labels[config.PlotLabels.Y_LABEL]
        y_unit = config.plot_labels[config.PlotLabels.Y_UNIT]
        if y_lbl is None: y_lbl = "Effect size"
        
        # assume stat is just before the extension in the filename, and 
        # determine weighting column based on stat
        stat = os.path.splitext(config.filename)[0].split("_")[-1]
        col_wt = vols.get_metric_weight_col(stat)
        if col_wt: print("weighting bars by", col_wt)
        
        # generate bar plot
        ax = plot_bars(
            config.filename, data_cols=("vals.effect",), 
            err_cols=(("vals.ci.low", "vals.ci.hi"), ), 
            legend_names="", col_groups="RegionName", title=title, 
            y_label=y_lbl, y_unit=y_unit, save=False,
            size=size, show=False, groups=config.groups, 
            prefix=config.prefix, col_vspan="Level", vspan_fmt="L{}", 
            col_wt=col_wt, x_tick_labels=x_tick_lbls, rotation=45)

    elif plot_2d_type is config.Plot2DTypes.LINE_PLOT:
        # generic line plot
        
        title = config.plot_labels[config.PlotLabels.TITLE]
        x_cols = config.plot_labels[config.PlotLabels.X_COL]
        data_cols = libmag.to_seq(
            config.plot_labels[config.PlotLabels.Y_COL])
        labels = (config.plot_labels[config.PlotLabels.Y_LABEL],
                  config.plot_labels[config.PlotLabels.X_LABEL])
        err_cols = libmag.to_seq(
            config.plot_labels[config.PlotLabels.ERR_COL])
        ax = plot_lines(
            config.filename, x_col=x_cols, data_cols=data_cols,
            labels=labels, err_cols=err_cols, title=title, size=size,
            show=False, groups=config.groups, prefix=config.prefix,
            suffix=config.suffix, marker=marker, col_annot=annot_col, ax=ax,
            save=False)

    elif plot_2d_type is config.Plot2DTypes.ROC_CURVE:
        # ROC curve

        # set annotation array index as 0 since most often vary only
        # z-val, but switch or remove when varying other axes
        ax = plot_roc(pd.read_csv(config.filename), config.show, 0, save=False)
    
    elif plot_2d_type is config.Plot2DTypes.SCATTER_PLOT:
        # scatter plot
        
        # get data frame columns and corresponding labels
        cols = (config.plot_labels[config.PlotLabels.Y_COL],
                config.plot_labels[config.PlotLabels.X_COL])
        labels = [config.plot_labels[config.PlotLabels.Y_LABEL],
                  config.plot_labels[config.PlotLabels.X_LABEL]]
        for i, (col, label) in enumerate(zip(cols, labels)):
            # default to use data frame columns
            if not label: labels[i] = col
        
        # get group columns and title
        cols_group = config.plot_labels[config.PlotLabels.GROUP_COL]
        if cols_group and not libmag.is_seq(cols_group):
            cols_group = [cols_group]
        title = config.plot_labels[config.PlotLabels.TITLE]
        if not title: title = "{} Vs. {}".format(*labels)
        
        ax = plot_scatter(
            config.filename, cols[1], cols[0], annot_col,
            cols_group=cols_group, labels=labels, title=title,
            fig_size=size, show=config.show, suffix=config.suffix,
            alpha=config.alphas[0], scale_x=scale_x, scale_y=scale_y,
            ax=ax, save=False)
    
    if ax is not None:
        # perform plot post-processing tasks, including file save unless
        # savefig is None
        post_plot(ax, libmag.make_out_path(), config.savefig, config.show)
    
    return ax


if __name__ == "__main__":
    print("Starting MagellanMapper 2D plotter...")
    # set up command-line args and run main tasks
    from magmap.io import cli
    cli.main(True)
    main()
