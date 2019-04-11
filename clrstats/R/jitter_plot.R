# Plotter for paired and jitter group plots
# Author: David Young, 2018, 2019

jitterPlot <- function(df.region, col, title, split.by.side=TRUE, 
                       split.col=NULL, paired=FALSE, show.sample.legend=FALSE, 
                       plot.size=c(5, 7), boxplot=TRUE) {
  # Plot jitter/scatter plots of values by genotype with summary stats.
  # 
  # Also generates mean and 95% CI for each group, which will be plotted 
  # unless boxplot is specified.
  #
  # Args:
  #   df.region: Date frame sliced by region, assumed to be filtered for 
  #     non-zero values.
  #   col: Name of column for values.
  #   title: Plot figure title.
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
  #   boxplot: True to show box plot in place of mean and CIs; defaults to 
  #     TRUE.
  #
  # Returns:
  #   List of group names, means, and 95% confidence intervals.
  
  if (is.null(split.col)) {
    # default column name by which to split
    split.col <- "Side"
  }
  if (is.element("Geno", names(df.region))) {
    genos <- df.region$Geno
  } else {
    genos <- c("")
  }
  genos.unique <- sort(unique(genos))
  multiple.geno <- length(genos.unique) > 1
  sides <- df.region[[split.col]]
  sides.unique <- unique(sides)
  if (!split.by.side | length(sides.unique) == 0) {
    # use a single side for one for-loop pass
    sides.unique = c("")
  }
  
  # setup coordinates to plot and error ranges
  num.sides <- length(sides.unique)
  num.groups <- length(genos.unique) * num.sides
  names.groups <- vector(length=num.groups)
  vals <- df.region[[col]]
  int.digits <- nchar(trunc(max(vals)))
  vals.groups <- list() # list of vals for each geno-side group
  vals.means <- vector(length=num.groups)
  vals.cis <-vector(length=num.groups)
  max.errs <- vector(length=num.groups)
  i <- 1
  for (geno in genos.unique) {
    for (side in sides.unique) {
      # vals for group based on whether to include side
      if (split.by.side) {
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
      err <- if (is.na(vals.cis[i])) 0 else vals.cis[i]
      max.errs[i] <- vals.means[i] + err
      
      # main label
      name <- side
      if (multiple.geno) name <- paste(geno, side)
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
      ylab <- paste0(ylab, " (", paste0(prefix, collapse=""))
    } else {
      ylab <- bquote(list(.(ylab)~(.(paste0(prefix, collapse=""))~.(unit))))
    }
  } else {
    denom <- 1
    if (!is.null(unit)) ylab <- bquote(list(.(ylab)~(.(unit))))
  }
  
  # define graph limits, with x from 0 to number of groups, and y from 
  # 0 to highest y-val, or highest absolute error bar if not boxplot
  maxes <- c(num.groups, max(vals) / denom)
  if (boxplot) maxes[2] <- max(maxes[2], max(max.errs) / denom)
  
  # save current graphical parameters to reset at end, avoiding setting 
  # spillover in subsequent plots
  par.old <- par(no.readonly=TRUE)
  if (show.sample.legend & paired) {
    # setup sample legend names and number of columns based on max name length
    names.samples <- unique(levels(df.region$Sample))
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
  
  # draw main plot and group legends
  plot(NULL, main=title, xlab="", ylab=ylab, xaxt="n", 
       xlim=range(-0.5, maxes[1] - 0.5), ylim=range(0, maxes[2]), bty="n", 
       las=1)
  colors <- RColorBrewer::brewer.pal(num.sides, "Dark2")
  # group legend, moved outside of plot and positioned at top right 
  # before shifting a full plot unit to sit below the plot
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
  legend.group <- legend(
    "topleft", legend=names.groups, pch=21:(21+length(names.groups)), 
    xpd=TRUE, inset=c(0, 1), horiz=TRUE, bty=bty, col=color, pt.bg=pt.bg, 
    text.width=legend.text.width, pt.cex=pt.cex)
  
  i <- 1
  group.last <- NULL
  for (geno in genos.unique) {
    # plot each group of points
    
    # add group label for genotypes if more than one total
    if (length(genos.unique) > 1) mtext(geno, side=1, at=i-0.5)
    x.pos <- 0:(num.sides-1) # group starting x-positions
    vals.geno <- list() # vals within genotype, for paired points
    if (show.sample.legend) {
      # distinct color for each member in group, using same set of
      # colors for each set of points
      if (num.sides > 0) {
        colors <- RColorBrewer::brewer.pal(length(vals.groups[[1]]), "Paired")
      }
    }
    for (side in sides.unique) {
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
      points(x.vals, vals.group, pch=i+20, col=colors.group, bg=colors.group, 
             cex=pt.cex)
      
      # plot summary stats on outer sides of scatter plots
      x.summary <- if (i %% 2 == 0) x + 0.25 else x - 0.25
      if (boxplot) {
        # overlay boxplot
        boxplot(vals.group, at=x.summary, add=TRUE, boxwex=0.2, yaxt="n", 
                frame.plot=FALSE)
      } else {
        # plot error bars unless CI is NA, such as infinitely large CI (n = 1)
        mean <- vals.means[[i]] / denom
        ci <- vals.cis[[i]] / denom
        if (!is.na(ci)) {
          points(x.summary, mean, pch=16, cex=2)
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
      group.rect <- legend.group$rect
      legend.sample <- legend(
        x=group.rect$left, y=(0.7*(group.rect$top-group.rect$h)), 
        legend=names.samples, lty=1, col=colors, xpd=TRUE, bty="n", 
        ncol=ncol, text.width=legend.text.width)
      sample.rect <- legend.sample$rect
      rect(sample.rect$left, sample.rect$top - sample.rect$h,
           sample.rect$left + sample.rect$w, group.rect$top)
    }
  }
  
  # save figure to PDF
  dev.print(
    pdf, width=plot.size[1], height=plot.size[2], 
    file=paste0(
      "../plot_jitter_", col, "_", gsub("/| ", "_", title), ".pdf"))
  par(par.old)
  
  return(list(names.groups, vals.means, vals.cis))
}
