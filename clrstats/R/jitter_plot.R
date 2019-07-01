# Plotter for paired and jitter group plots
# Author: David Young, 2018, 2019

kSummaryStats <- c("mean.ci", "boxplot")

jitterPlot <- function(df.region, col, title, geno.col=NULL, 
                       split.by.side=TRUE, split.col=NULL, 
                       paired=FALSE, show.sample.legend=FALSE, 
                       plot.size=c(5, 7), summary.stats=kSummaryStats[2], 
                       axes.in.range=FALSE, save=TRUE, sort.groups=TRUE) {
  # Plot jitter/scatter plots of values by genotype with summary stats.
  #
  # Groups specified in a "Group" or "Geno" column will have separate 
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
  #   geno.col: Name of column specifying main groups.
  #   split.by.side: True to plot separate sub-scatter plots for each 
  #     region by side; defaults to TRUE.
  #   split.col: Column name by which to split; defaults to NULL, in which 
  #     case "Side" will be used as the column name.
  #   paired: True to show pairing between values, which assumes that values 
  #     are in the same order when filtered by split.col. Jitter will be 
  #     turned off to ensure that start and end x-values are the same for 
  #     pairings. Defaults to FALSE.
  #   show.sample.legend: True to show a separate legend of samples for 
  #     each genotype group. Assumes that the number of samples in each 
  #     split group is the same within each genotype. Defaults to FALSE.
  #   plot.size: Vector of width, height for exported plot; defaults to 
  #     c(5, 7).
  #   summary.stats: One of kSummaryStats designating the type of stats 
  #     to display, or NULL to show none. Defaults to "boxplot".
  #   axes.in.range: True to require x- and y-ranges to include axes; 
  #     defaults to FALSE.
  #   save: True to save plot to file; defaults to TRUE.
  #   sort.groups: True to sort groups; defaults to TRUE.
  #
  # Returns:
  #   List of group names, means, and 95% confidence intervals.
  
  # set up grouping, where "geno/group" specifies the main group, and 
  # "side/split.col" specifies sub-groups
  if (is.null(geno.col)) {
    # default group name
    geno.col <- "Group"
  }
  if (is.element(geno.col, names(df.region))) {
    genos <- df.region[[geno.col]]
  } else {
    genos <- c("")
  }
  genos.unique <- unique(genos)
  if (sort.groups) genos.unique <- sort(genos.unique)
  if (is.null(split.col)) {
    # default column name by which to split
    split.col <- "Side"
  }
  sides <- df.region[[split.col]]
  sides.unique <- getUniqueSides(sides, split.by.side)
  num.genos <- length(genos.unique) # total main groups
  num.sides <- length(sides.unique) # total unique subgroups
  num.groups <- num.genos # total group-subgroup combos
  if (is.element(split.col, names(df.region))) {
    num.groups <- nrow(unique(df.region[, c(geno.col, split.col)]))
  }
  sides.by.geno <- list() # sets of sides within each main group
  names.groups <- vector(length=num.groups)
  
  # set up summary stats to display
  mean.ci <- FALSE
  boxplot <- FALSE
  if (!is.null(summary.stats)) {
    mean.ci <- summary.stats == kSummaryStats[1]
    boxplot <- summary.stats == kSummaryStats[2]
  }
  
  # set up coordinates to plot and error ranges
  vals <- df.region[[col]]
  int.digits <- nchar(trunc(max(vals)))
  vals.groups <- list() # list of vals for each geno-side group
  vals.means <- vector(length=num.groups)
  vals.cis <-vector(length=num.groups)
  errs <- vector(length=num.groups) # based on CI but 0 if CI is NA
  i <- 1
  for (geno in genos.unique) {
    sides.in.geno <- sides
    if (geno != "") {
      sides.in.geno <- df.region[df.region[[geno.col]] == geno, split.col]
    }
    sides.in.geno.unique <- getUniqueSides(sides.in.geno, split.by.side)
    sides.by.geno <- append(sides.by.geno, list(sides.in.geno.unique))
    for (side in sides.in.geno.unique) {
      # vals for group based on whether to include side
      if (side != "") {
        vals.geno <- vals[genos == geno & sides == side]
      } else {
        vals.geno <- vals[genos == geno]
      }
      vals.groups <- append(vals.groups, list(vals.geno))
      
      # error bars
      vals.means[i] <- mean(vals.geno)
      num.vals <- length(vals.geno)
      vals.sem <- sd(vals.geno) / sqrt(num.vals)
      # use 97.5th percentile for 2-tailed 95% confidence level
      vals.cis[i] <- qt(0.975, df=num.vals-1) * vals.sem
      # store max height of error bar setting axis limits
      errs[i] <- if (is.na(vals.cis[i])) 0 else vals.cis[i]
      
      # main label
      if (num.sides > 1) {
        name <- side
      } else {
        name <- geno
      }
      names.groups[i] <- name
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
  maxes <- c(num.groups, max(vals) / denom)
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
  # add 10% padding above and below y-range
  y.pad <- 0.1 * (maxes[2] - mins[2])
  mins[2] <- mins[2] - y.pad
  maxes[2] <- maxes[2] + y.pad
  
  # save current graphical parameters to reset at end, avoiding setting 
  # spillover in subsequent plots
  par.old <- par(no.readonly=TRUE)
  if (show.sample.legend) {
    # setup sample legend names and number of columns based on max name length
    samples <- df.region$Sample
    if (is.factor(samples)) {
      samples <- levels(samples)
    }
    names.samples <- unique(samples)
    name.max.len <- max(nchar(names.samples))
    ncol <- 1
    if (name.max.len <= 10) ncol <- 2
    
    # increase bottom margin based on additional rows for sample legend
    margin <- par()$mar
    margin[1] <- margin[1] + length(names.samples) / (1.3 * ncol)
    par(mar=margin)
    par(xpd=NA) # for custom legend rect outside of plot
  }
  legend.text.width <- NULL
  if (plot.size[1] < plot.size[2]) {
    # make legend width larger for narrow plots to avoid overlap
    legend.text.width <- 1 - 0.5 * plot.size[1] / plot.size[2]
  }
  
  # draw main plot
  plot(NULL, main=title, xlab="", ylab=ylab, xaxt="n", 
       xlim=range(mins[1], maxes[1] - 0.5), ylim=range(mins[2], maxes[2]), 
       bty="n", las=1)
  
  # subgroup legend, moved outside of plot and positioned at top right 
  # before shifting a full plot unit to sit below the plot
  colors <- RColorBrewer::brewer.pal(num.sides, "Dark2")
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
  pchs <- rep(15:25, length.out=(num.sides+pch.offset))
  legend.sides <- legend(
    "topleft", legend=sides.unique, pch=pchs[pch.offset+1:length(pchs)], 
    xpd=TRUE, inset=c(0, 1), bty=bty, col=color, pt.bg=pt.bg, 
    text.width=legend.text.width, pt.cex=pt.cex, ncol=3)
  
  i <- 1
  group.last <- NULL
  x.pos <- 0:(num.groups-1)
  for (j in seq_along(genos.unique)) {
    geno <- genos.unique[j]
    sides.in.geno.unique <- sides.by.geno[[j]]
    # plot each group of points
    
    # TODO: consider adding x-tick-label if > 1 genos and sides
    if (length(genos.unique) > 1) {
      text(i + (length(sides.in.geno.unique) - 1) / 2 - 1, 
           0, labels=geno, pos=1)
    }
    vals.geno <- list() # vals within genotype, for paired points
    if (show.sample.legend) {
      # distinct color for each member in group, using same set of
      # colors for each set of points
      if (num.sides > 0) {
        colors <- RColorBrewer::brewer.pal(length(vals.groups[[1]]), "Paired")
      }
    }
    for (side in sides.in.geno.unique) {
      # plot points, adding jitter in x-direction unless paired
      vals.group <- vals.groups[[i]] / denom
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
      pch <- pchs[match(side, sides.unique) + pch.offset]
      points(x.vals, vals.group, pch=pch, col=colors.group, 
             bg=colors.group, cex=pt.cex)
      
      # plot summary stats on outer sides of scatter plots
      x.summary <- x
      if (boxplot || paired) {
        x.summary <- if (i %% 2 == 0) x + 0.25 else x - 0.25
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
      # add sample legend below group legend to label colors, with manually 
      # drawn rectangle to enclose group legend as well
      group.rect <- legend.sides$rect
      legend.sample <- legend(
        x=group.rect$left, y=(group.rect$top-0.7*group.rect$h), 
        legend=names.samples, lty=1, col=colors, xpd=TRUE, bty="n", 
        ncol=ncol, text.width=legend.text.width)
      sample.rect <- legend.sample$rect
      rect(sample.rect$left, sample.rect$top - sample.rect$h,
           sample.rect$left + sample.rect$w, group.rect$top)
    }
  }
  
  if (save) {
    # save figure to PDF
    dev.print(
      pdf, width=plot.size[1], height=plot.size[2], 
      file=paste0("../plot_jitter_", col, "_", gsub("/| ", "_", title), ".pdf"))
    }
  par(par.old)
  
  return(list(names.groups, vals.means, vals.cis))
}

getUniqueSides <- function(sides, split.by.side) {
  # Get a vector of unique sides, defaulting to a vector of a single empty 
  # string if the sides vectors is empty or side split flag is False.
  #
  # Args:
  #   sides: Vector of sides.
  #   split.by.side: False if the sides should be ignored.
  #
  # Returns:
  #   A vector of sides or of only an empty string if no sides are found or 
  #   split.by.side is False.
  
  sides.unique <- unique(sides)
  single.side <- !split.by.side | length(sides.unique) == 0
  if (single.side) {
    # use a single side for one for-loop pass
    sides.unique = c("")
  }
  return(sides.unique)
}

runJitter <- function(path.in) {
  # Create a generic jitter plot.
  #
  # Arguments:
  #   path.in: Input CSV path.
  df <- read.csv(path.in)
  print(df)
  jitterPlot(df, "Response", "Cas response", split.by.side=FALSE, 
             split.col=NULL, paired=FALSE, show.sample.legend=FALSE, 
             plot.size=c(5, 7), summary.stats=kSummaryStats[1], 
             axes.in.range=FALSE)
}
