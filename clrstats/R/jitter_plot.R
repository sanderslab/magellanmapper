# Plotter for paired and jitter group plots
# Author: David Young, 2018, 2019

jitterPlot <- function(df.region, col, title, split.by.side=TRUE, 
                       split.col=NULL, paired=FALSE, show.sample.legend=FALSE) {
  # Plot jitter/scatter plots of values by genotype with mean and 95% CI.
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
  print("vals")
  print(vals)
  print(max(vals))
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
  
  # adjust y-axis to use any replacement label and rescale to avoid 
  # scientific notation in labels
  if (is.element(col, names(kMeasNames))) {
    ylab <- kMeasNames[[col]]
  } else {
    ylab <- gsub("_", " ", col)
  }
  if (int.digits >= 5) {
    power <- int.digits - 1
    denom <- 10 ^ power
    ylab <- paste0(ylab, " (10^", power, ")")
  } else {
    denom <- 1
  }
  
  # max y-val or error bar, whichever is higher
  maxes <- c(num.groups, max(max(vals) / denom, max(max.errs) / denom))
  
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
  }
  
  # plot values with means and error bars
  
  plot(NULL, frame.plot=TRUE, main=title, xlab="", ylab=ylab, xaxt="n", 
       xlim=range(-0.5, maxes[1] - 0.5), ylim=range(0, maxes[2]), bty="n", 
       las=1)
  colors <- viridis::viridis(num.sides, begin=0.2, end=0.8)
  i <- 1
  for (geno in genos.unique) {
    # plot each group of points
    
    # add group label for genotypes if more than one total
    if (length(genos.unique) > 1) mtext(geno, side=1, at=i-0.5)
    x.adj <- 0
    x.pos <- vector(length=num.sides) # group base x-positions
    vals.geno <- list() # vals within genotype, for paired points
    if (show.sample.legend) {
      # distinct color for each member in group, using same set of
      # colors for each set of points
      if (num.sides > 0) colors <- RColorBrewer::brewer.pal(length(vals.groups[[1]]), "Paired")
    }
    for (side in sides.unique) {
      # plot points, adding jitter in x-direction unless paired
      vals.group <- vals.groups[[i]]
      vals.geno <- append(vals.geno, vals.groups[i])
      x <- i + x.adj - 1
      x.vals <- rep(x, length(vals.group))
      if (!paired) {
        # add jitter to distinguish points
        x.vals <- jitter(x.vals, amount=0.2)
      }
      colors.group <- if (show.sample.legend) colors else colors[i]
      points(x.vals, vals.group / denom, pch=i+14, col=colors.group)
      
      # plot error bars unless CI is NA, such as infinitely large CI when n = 1
      mean <- vals.means[[i]] / denom
      ci <- vals.cis[[i]] / denom
      if (!is.na(ci)) {
        if (i %% 2 == 0) {
          x.mean <- x + 0.25
        } else {
          x.mean <- x - 0.25
        }
        points(x.mean, mean, pch=16, cex=2)
        arrows(x.mean, mean + ci, x.mean, mean - ci, length=0.05, angle=90, 
               code=3)
      }
      x.pos[i] <- x # store x for connecting paired points
      x.adj <- x.adj + 0.05
      i <- i + 1
    }
    if (paired) {
      # connect pairs of points with segments, assuming same order for each 
      # vector of values
      vals.group <- vals.geno[[1]]
      for (j in seq_along(vals.group)) {
        color <- if(show.sample.legend) colors[j] else 1
        segments(x.pos[1], vals.group[j] / denom, x.pos[2], 
                 vals.geno[[2]][j] / denom, col=color)
      }
    }
    if (show.sample.legend) {
      # add sample legend below group legend to label colors
      legend("topright", legend=names.samples, lty=1, 
             col=colors, xpd=TRUE, inset=c(x.pos[1], 1.05), ncol=ncol, bty="n")
    }
  }
  
  # group legend, moved outside of plot and positioned at top right 
  # before shifting a full plot unit to sit below the plot
  color <- if(show.sample.legend) 1 else colors
  legend("topright", legend=names.groups, pch=15:(14+length(names.groups)), 
         xpd=TRUE, inset=c(0, 1), horiz=TRUE, bty="n", col=color)
  
  # save figure to PDF
  dev.print(
    pdf, width=7, height=7, 
    file=paste0(
      "../plot_jitter_", col, "_", gsub("/| ", "_", title), ".pdf"))
  par(par.old)
  
  return(list(names.groups, vals.means, vals.cis))
}
