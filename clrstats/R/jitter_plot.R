# Plotter for paired and jitter group plots
# Author: David Young, 2018, 2019

kSummaryStats <- c("mean.ci", "boxplot")

jitterPlot <- function(df.region, col, title, group.col=NULL, 
                       split.by.subgroup=TRUE, split.col=NULL, 
                       paired=FALSE, show.sample.legend=FALSE, 
                       plot.size=c(5, 7), summary.stats=kSummaryStats[2], 
                       axes.in.range=FALSE, save=TRUE, sort.groups=TRUE,
                       show.labels=FALSE) {
  # Plot jitter/scatter plots of values by groups with summary stats.
  #
  # Groups specified in a main group column will have separate 
  # x-tick labels. "split.col" can be used to divide groups further into 
  # sub-groups with symbols and labels denoted in the main legend. Paired 
  # plots are assumed to have corresponding samples in each pair of 
  # "split.col" sub-groups. For unpaired plots, "split.col" can specify 
  # an arbitrary number of sub-groups for each group.
  # 
  # Also generates mean and 95% CI for each group, which will be plotted 
  # unless boxplot is specified.
  #
  # Args:
  #   df.region: Date frame sliced by region, assumed to be filtered for 
  #     non-zero values.
  #   col: Name of column for values.
  #   title: Plot figure title.
  #   group.col: Name of column specifying main groups.
  #   split.by.subgroup: True to plot separate sub-scatter plots for each 
  #     region by subgroup; defaults to TRUE.
  #   split.col: Column name by which to split; defaults to NULL, in which 
  #     case "Side" will be used as the column name.
  #   paired: True to show pairing between values, which assumes that values 
  #     are in the same order when filtered by split.col. Jitter will be 
  #     turned off to ensure that start and end x-values are the same for 
  #     pairings. Defaults to FALSE.
  #   show.sample.legend: True to show a separate legend of samples. 
  #     Assumes that the number of samples in each split group is the 
  #     same within each main group. Defaults to FALSE.
  #   plot.size: Vector of width, height for exported plot; defaults to 
  #     c(5, 7).
  #   summary.stats: One of kSummaryStats designating the type of stats 
  #     to display, or NULL to show none. Defaults to "boxplot".
  #   axes.in.range: True to require x- and y-ranges to include axes; 
  #     defaults to FALSE.
  #   save: True to save plot to file; defaults to TRUE.
  #   sort.groups: True to sort groups; defaults to TRUE.
  #   show.labels: Annotate points with sample names; defaults to FALSE.
  #
  # Returns:
  #   List of group names, means, and 95% confidence intervals.
  
  # set up grouping, where "group" specifies the main group, and 
  # "split.col" specifies subgroups
  if (is.null(group.col)) {
    # default group name
    group.col <- "Group"
  }
  id.cols <- list() # group ID columns
  if (is.element(group.col, names(df.region))) {
    groups <- df.region[[group.col]]
    id.cols <- append(group.col, id.cols)
  } else {
    groups <- c("")
  }
  groups.unique <- unique(groups)
  if (sort.groups) groups.unique <- sort(groups.unique)
  if (is.null(split.col)) {
    # default column name by which to split
    split.col <- "Side"
  }
  if (is.element(split.col, names(df.region))) {
    subgroups <- df.region[[split.col]]
    id.cols <- append(split.col, id.cols)
  } else {
    subgroups <- c("")
  }
  subgroups.unique <- getUniqueSubgroups(subgroups, split.by.subgroup)
  num.groups <- length(groups.unique) # total groups
  num.subgroups <- length(subgroups.unique) # total unique subgroups
  num.groupcombos <- num.groups # total group-subgroup combos
  id.cols <- unlist(id.cols, use.names=FALSE)
  if (length(id.cols) > 0) {
    combos <- unique(df.region[, id.cols])
    if (length(id.cols) >= 2) {
      num.groupcombos <- nrow(combos)
    } else {
      num.groupcombos <- length(combos)
    }
  }
  subgroups.by.group <- list() # sets of subgroups within each main group
  names.groupcombos <- vector(length=num.groupcombos) # group-subgroup names
  
  # set up summary stats to display
  mean.ci <- FALSE
  boxplot <- FALSE
  if (!is.null(summary.stats)) {
    mean.ci <- summary.stats == kSummaryStats[1]
    boxplot <- summary.stats == kSummaryStats[2]
  }
  
  # set up coordinates to plot and error ranges
  vals <- df.region[[col]]
  samples <- df.region$Sample
  int.digits <- nchar(trunc(max(vals)))
  vals.groups <- list() # list of vals for each group-subgroup group
  vals.means <- vector(length=num.groupcombos)
  vals.medians <- vector(length=num.groupcombos)
  vals.cis <-vector(length=num.groupcombos)
  vals.sds <-vector(length=num.groupcombos)
  errs <- vector(length=num.groupcombos) # based on CI but 0 if CI is NA
  i <- 1
  for (group in groups.unique) {
    subgroups.in.group <- subgroups
    if (group != "") {
      subgroups.in.group <- df.region[
        df.region[[group.col]] == group, split.col]
    }
    subgroups.in.group.unique <- getUniqueSubgroups(
      subgroups.in.group, split.by.subgroup)
    subgroups.by.group <- append(
      subgroups.by.group, list(subgroups.in.group.unique))
    for (subgroup in subgroups.in.group.unique) {
      # vals for group based on whether to include subgroup
      if (subgroup != "") {
        mask <- groups == group & subgroups == subgroup
      } else {
        mask <- groups == group
      }
      vals.group <- vals[mask]
      vals.groups[[i]] <- list(vals=vals.group, samples=samples[mask])
      
      # error bars
      vals.means[i] <- mean(vals.group)
      vals.medians[i] <- median(vals.group)
      num.vals <- length(vals.group)
      vals.sds[i] <- sd(vals.group)
      vals.sem <- vals.sds[i] / sqrt(num.vals)
      # use 97.5th percentile for 2-tailed 95% confidence level
      vals.cis[i] <- qt(0.975, df=num.vals-1) * vals.sem
      # store max height of error bar setting axis limits
      errs[i] <- if (is.na(vals.cis[i])) 0 else vals.cis[i]
      
      # stats label
      name <- group
      if (num.subgroups > 1) {
        name <- subgroup
        if (num.groups > 1) name <- paste(group, subgroup)
      }
      names.groupcombos[i] <- name
      i <- i + 1
    }
  }
  
  # adjust y-axis to use any replacement label and unit, rescaling to avoid 
  # scientific notation in labels
  if (is.element(col, names(kMeasNames))) {
    meas <- kMeasNames[[col]]
    ylab <- meas[[1]]
    unit <- meas[[2]]
  } else {
    ylab <- gsub("_", " ", col)
    unit <- NULL
  }
  if (int.digits >= 5) {
    # TODO: support scientific notation for decimal values
    power <- int.digits - 1
    denom <- 10 ^ power
    # use single-character numeral prefix abbreviations if possible
    if (power >= 15 | power < 6) {
      prefix <- c("10^", power)
    } else if (power >= 6) {
      if (power >= 12) {
        prefix <- c(denom / 10 ^ 12, "T")
      } else if (power >= 9) {
        prefix <- c(denom / 10 ^ 9, "B")
      } else {
        prefix <- c(denom / 10 ^ 6, "M")
      }
    }
    if (is.null(unit)) {
      ylab <- paste0(ylab, " (", paste0(prefix, collapse=""), ")")
    } else {
      ylab <- bquote(list(.(ylab)~(.(paste0(prefix, collapse=""))~.(unit))))
    }
  } else {
    denom <- 1
    if (!is.null(unit)) ylab <- bquote(list(.(ylab)~(.(unit))))
  }
  
  # define graph limits, with x from 0 to number of groups, and y from 
  # 0 to highest y-val, or highest absolute error bar
  mins <- c(-0.5, min(vals) / denom)
  maxes <- c(num.groupcombos, max(vals) / denom)
  if (mean.ci) {
    mins[2] <- min(mins[2], min(vals.means - errs) / denom)
    maxes[2] <- max(maxes[2], max(vals.means + errs) / denom)
  }
  if (axes.in.range) {
    # ensure that y-axis remains within range; x-axis already within range
    if (mins[2] < 0 & maxes[2] < 0) {
      maxes[2] <- 0
    } else if (mins[2] > 0 & maxes[2] > 0) {
      mins[2] <- 0
    }
  }
  # add 20% padding above and below y-range
  # TODO: consider moving to parameter
  y.pad <- 0.2 * (maxes[2] - mins[2])
  mins[2] <- mins[2] - y.pad
  maxes[2] <- maxes[2] + y.pad
  
  # save current graphical parameters to reset at end, avoiding setting 
  # spillover in subsequent plots
  par.old <- par(no.readonly=TRUE)
  if (show.sample.legend) {
    # setup sample legend names and number of columns based on max name length
    samples.unique <- samples
    if (is.factor(samples.unique)) {
      # get levels but retain original order
      samples.unique <- levels(samples.unique)[samples.unique]
    }
    samples.unique <- unique(samples.unique)
    name.max.len <- max(nchar(samples.unique))
    ncol <- 1
    if (name.max.len <= 10) ncol <- 2
    
    # increase bottom margin based on additional rows for sample legend
    margin <- par()$mar
    margin[1] <- margin[1] + length(samples.unique) / (1.3 * ncol)
    par(mar=margin)
    par(xpd=NA) # for custom legend rect outside of plot
  }
  legend.text.width <- 0.7
  if (plot.size[1] < plot.size[2]) {
    # make legend width larger for narrow plots to avoid overlap
    legend.text.width <- 0.6 + 0.1 * plot.size[2] / plot.size[1]
  }
  
  # draw main plot
  plot(NULL, main=title, xlab="", ylab=ylab, xaxt="n", 
       xlim=range(mins[1], maxes[1] - 0.5), ylim=range(mins[2], maxes[2]), 
       bty="n", las=1)
  
  # subgroup legend, moved outside of plot and positioned at top right 
  # before shifting a full plot unit to sit below the plot
  colors <- RColorBrewer::brewer.pal(num.subgroups, "Dark2")
  if (show.sample.legend) {
    color <- 1
    bty <- "n"
    pt.bg <- "gray"
  } else {
    color <- colors
    bty <- "o"
    pt.bg <- NA
  }
  pt.cex <- 1.5
  # use only solid pch symbols, starting with those that have a distinct 
  # border; repeat symbols if run out
  pch.offset <- 6
  pchs <- rep(15:25, length.out=(num.subgroups+pch.offset))
  legend.subgroups <- NULL
  if (num.subgroups > 1) {
    legend.subgroups <- legend(
      "topleft", legend=subgroups.unique, pch=pchs[pch.offset+1:length(pchs)], 
      xpd=TRUE, inset=c(0, 1), bty=bty, col=color, pt.bg=pt.bg, 
      text.width=legend.text.width, pt.cex=pt.cex, ncol=3)
  }
  
  i <- 1
  group.last <- NULL
  x.pos <- 0:(num.groupcombos-1)
  for (j in seq_along(groups.unique)) {
    group <- groups.unique[j]
    subgroups.in.group.unique <- subgroups.by.group[[j]]
    # plot each group of points
    
    if (length(groups.unique) > 1) {
      # place group label at center of group just below x-axis
      text(i + (length(subgroups.in.group.unique) - 1) / 2 - 1, 
           0, labels=group, pos=1)
    }
    vals.group <- list() # vals within main group, for paired points
    if (show.sample.legend) {
      # distinct color for each member in group, using same set of
      # colors for each set of points
      # TODO: fix colors for unpaired samples
      if (num.subgroups > 0) {
        colors <- RColorBrewer::brewer.pal(length(samples.unique), "Paired")
      }
    }
    for (subgroup in subgroups.in.group.unique) {
      # plot points, adding jitter in x-direction unless paired
      vals.group <- vals.groups[[i]]$vals / denom
      x <- x.pos[i]
      if (i %% 2 == 0) {
        # shift even groups left slightly and connect if paired
        x <- x - 0.05
        if (paired) {
          # assume same order within each group
          for (j in seq_along(vals.group)) {
            color <- if(show.sample.legend) colors[j] else 1
            segments(x.pos[i - 1], group.last[j], x, vals.group[j], col=color)
          }
        }
      }
      x.vals <- rep(x, length(vals.group))
      if (!paired) {
        # add jitter to distinguish points
        x.vals <- jitter(x.vals, amount=0.2)
      }
      colors.group <- if (show.sample.legend) colors else colors[i]
      pch <- pchs[i + pch.offset]
      pch <- pchs[match(subgroup, subgroups.unique) + pch.offset]
      points(x.vals, vals.group, pch=pch, col=colors.group, 
             bg=colors.group, cex=pt.cex)
      if (show.labels) {
        addTextLabels::addTextLabels(
          x.vals, vals.group, label=vals.groups[[i]]$samples, cex=0.5, lwd=0.2)
      }
      
      # plot summary stats, eg means/CIs or boxplots
      x.summary <- x # default to means/CIs under jitter points
      if (paired) {
        # shift to outer sides of paired points since the points' vertical 
        # alignment would occlude summary stats
        x.summary <- if (i %% 2 == 0) x + 0.25 else x - 0.25
      } else if (boxplot) {
        # plot boxplot before subgroups
        x.summary <- x - 0.25
      }
      if (boxplot) {
        # overlay boxplot
        boxplot(vals.group, at=x.summary, add=TRUE, boxwex=0.2, yaxt="n", 
                frame.plot=FALSE)
      } else if (mean.ci) {
        # plot error bars unless CI is NA, such as infinitely large CI (n = 1)
        mean <- vals.means[[i]] / denom
        ci <- vals.cis[[i]] / denom
        if (!is.na(ci)) {
          segments(x.summary - 0.25, mean, x.summary + 0.25, mean)
          arrows(x.summary, mean + ci, x.summary, mean - ci, length=0.05, 
                 angle=90, code=3)
        }
      }
      group.last <- vals.group
      i <- i + 1
    }
    if (show.sample.legend) {
      # add sample legend
      if (is.null(legend.subgroups)) {
        legend.sample <- legend(
          "topleft", inset=c(0, 1), xpd=TRUE, legend=samples.unique, lty=1, 
          col=colors, ncol=ncol, text.width=legend.text.width)
      } else {
        # place below group legend to label colors, with manually 
        # drawn rectangle to enclose group legend as well
        group.rect <- legend.subgroups$rect
        legend.sample <- legend(
          x=group.rect$left, y=(group.rect$top-0.7*group.rect$h), 
          legend=samples.unique, lty=1, col=colors, xpd=TRUE, bty="n", 
          ncol=ncol, text.width=legend.text.width)
        sample.rect <- legend.sample$rect
        rect(sample.rect$left, sample.rect$top - sample.rect$h,
             sample.rect$left + sample.rect$w, group.rect$top)
      }
    }
  }
  
  if (save) {
    # save figure to PDF
    dev.print(
      pdf, width=plot.size[1], height=plot.size[2], 
      file=paste0("../plot_jitter_", col, "_", gsub("/| ", "_", title), ".pdf"))
    }
  par(par.old)
  
  return(list(names.groupcombos, vals.means, vals.medians, vals.sds, vals.cis))
}

getUniqueSubgroups <- function(subgroups, split.by.subgroup) {
  # Get a vector of unique subgroups, defaulting to a vector of a single empty 
  # string if the subgroups vectors is empty or subgroup split flag is False.
  #
  # Args:
  #   subgroups: Vector of subgroups.
  #   split.by.subgroup: False if the subgroups should be ignored.
  #
  # Returns:
  #   A vector of subgroups or of only an empty string if no subgroups are found or 
  #   split.by.subgroup is False.
  
  subgroups.unique <- unique(subgroups)
  single.subgroup <- !split.by.subgroup | length(subgroups.unique) == 0
  if (single.subgroup) {
    # use a single subgroup for one for-loop pass
    subgroups.unique = c("")
  }
  return(subgroups.unique)
}

runJitter <- function(path.in) {
  # Create a generic jitter plot.
  #
  # Arguments:
  #   path.in: Input CSV path.
  df <- read.csv(path.in)
  print(df)
  jitterPlot(df, "Response", "Cas response", split.by.subgroup=FALSE, 
             split.col=NULL, paired=FALSE, show.sample.legend=FALSE, 
             plot.size=c(5, 7), summary.stats=kSummaryStats[1], 
             axes.in.range=FALSE)
}
