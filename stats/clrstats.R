# Clrbrain stats in R
# Author: David Young, 2018

library("MASS")
library("gee")
library("viridis")

# library to avoid overlapping text labels
#install.packages("devtools")
#library("devtools")
#install_github("JosephCrispell/addTextLabels")
library("addTextLabels")

# alternatives for labels overlap, but both still leave some overlap
#library("plotrix")
#library("maptools")

# statistical models
kModel <- c("logit", "linregr", "gee", "logit.ord", "ttest", "wilcoxon", 
           "ttest.paired", "wilcoxon.paired", "fligner")

# measurements, which correspond to columns in main data frame
kMeas <- c("Volume", "Density", "Nuclei", "VarNuclei", "VarIntensity", 
          "EdgeDistSum", "EdgeDistMean", "DSC_atlas_labels")

# named vector to convert measurement columns to display names
kMeasNames <- setNames(
  c("Within-Region Nuclei Variation", 
    "Within-Region Intensity Variation", 
    "Edge Distances to Anatomical Boundaries", 
    "Mean Edge Distances to Anatomical Boundaries", 
    "Dice Similarity Coefficient Between Atlas and Labels"), 
  c(kMeas[4:8]))

# ordered genotype levels
kGenoLevels <- c(0, 0.5, 1)

# regions to ignore (eg duplicates)
kRegionsIgnore <- c(15564)


# File Paths

# raw values from Clrbrain
kStatsFilesIn <- c("vols_by_sample.csv", "vols_by_sample_levels.csv", 
                   "vols_by_sample_summary.csv", "dsc_summary.csv")
kStatsPathOut <- "../vols_stats.csv" # output stats

# region-ID map from Clrbrain, which should contain all regions including 
# hierarchical/ontological ones
kRegionIDsPath <- "../region_ids.csv"


fitModel <- function(model, vals, genos, sides, ids=NULL) {
  # Fit data with the given regression model.
  #
  # Args:
  #   model: Model to use, corresponding to one of kModel.
  #   vals: Main independent variable.
  #   genos: Genotypes vector.
  #   sides: Vector indicated which side the given values is on, eg 
  #     corresponding to left or right
  #   ids: Vector of sample IDs; defaults to NULL.
  #
  # Returns:
  #   Coefficients of the summary statistics. The first row of coefficients 
  #   is removed if it is a non-intercept row. The colums are assumed to have 
  #   effect size in the 2nd column and p-value in the 4th.
  
  coef.tab <- NULL
  if (model == kModel[1]) {
    # logistic regression
    fit <- glm(genos ~ vals * sides, family=binomial)
    coef.tab <- summary.glm(fit)$coefficients
    # remove first ("non-intercept") row
    coef.tab <- coef.tab[-(1:1), ]
  } else if (model == kModel[2]) {
    # linear regression
    # TODO: see whether need to factorize genos
    fit <- lm(vals ~ genos * sides)
    coef.tab <- summary.lm(fit)$coefficients
    # remove first ("non-intercept") row
    coef.tab <- coef.tab[-(1:1), ]
  } else if (model == kModel[3]) {
    # generalized estimating equations
    # TODO: fix model prob "fitted value very close to 1" error
    fit <- gee(
      genos ~ vals * sides, ids, corstr="exchangeable", family=binomial())
    coef.tab <- summary(fit)$coefficients
  } else if (model == kModel[4]) {
    # ordered logistic regression
    vals <- scale(vals)
    genos <- factor(genos, levels=kGenoLevels)
    fit <- tryCatch({
      fit <- polr(genos ~ vals * sides, Hess=TRUE)
      coef.tab <- coef(summary(fit))
      # calculate p-vals and incorporate into coefficients
      p.vals <- pnorm(abs(coef.tab[, "t value"]), lower.tail=FALSE) * 2
      coef.tab <- cbind(coef.tab, "p value"=p.vals)
    },
      error=function(e) {
        print(paste("Could not calculate ordered logistic regression", 
              e, "skipping"))
      }
    )
  } else {
    cat("Sorry, model", model, "not found\n")
  }
  print(coef.tab)
  return(coef.tab)
}

meansModel <- function(vals, conditions, model, paired=FALSE) {
  # Test for differences of means.
  #
  # Args:
  #   vals: List of values to compare.
  #   conditions: List assumed to have at least two conditions by which to 
  #     group; only the first two sorted conditions will be used, and all 
  #     other conditions will be ignored. For paired tests, number of 
  #     values per condition is assumed to be the same.
  #   model: One of kModels to apply, which should be a mean test such as 
  #     "ttest" or "wilcoxon".
  #   paired: True for paired test; defaults to FALSE.
  #
  # Returns:
  #   Data frame of with p-value and mean of differences columns.
  
  # require at least 2 condition to compare
  conditions.unique = sort(unique(conditions))
  if (length(conditions.unique) < 2) {
    cat("need at least 2 conditions, cannot compare means\n")
    return(NULL)
  }
  
  # build lists of value vectors for each condition
  val.conds <- list()
  num.per.cond <- NULL
  for (i in seq_along(conditions.unique)) {
    val.conds[[i]] <- vals[conditions == conditions.unique[i]]
    num.per.cond <- length(val.conds[[i]])
    if (num.per.cond <= 1) {
      cat("0-1 values for at least one condition, cannot calculate stats\n")
      return(NULL)
    }
  }
  
  result <- NULL
  col.effect <- "estimate"
  if (model == kModel[5] | model == kModel[7]) {
    # Student's t-test
    result <- t.test(val.conds[[1]], val.conds[[2]], paired=paired)
  } else if (model == kModel[6] | model == kModel[8]) {
    # Wilcoxon test (Mann-Whitney if not paired)
    result <- wilcox.test(
      val.conds[[1]], val.conds[[2]], paired=paired, conf.int=TRUE)
  } else if (model == kModel[9]) {
    # Fligner-Killen test of variance
    result <- fligner.test(vals, conditions)
    col.effect <- "statistic"
  } else {
    cat("Sorry, model", model, "not found\n")
  }
  print(result)
  
  # basic stats data frame in format for filterStats
  coef.tab <- data.frame(matrix(nrow=1, ncol=4))
  names(coef.tab) <- c("Value", "col2", "col3", "P")
  rownames(coef.tab) <- c("vals")
  coef.tab$Value <- c(result[[col.effect]])
  coef.tab$P <- c(result$p.value)
  print(coef.tab)
  return(coef.tab)
}

statsByCols <- function(df, col.start, model) {
  # Calculates statistics for columns starting with the given string using 
  # the selected model.
  #
  # Values of 0 will be ignored. If all values for a given vector are 0, 
  # statistics will not be computed.
  #
  # Args:
  #   df: Data frame with columns for Genos, Sides, and names starting with 
  #     col.start.
  #   col.start: Columns starting with this string will be included.
  #   model: Model to use, corresponding to one of kModel.
  
  .Deprecated("statsByRegion")
  
  # filter cols only starting with search string
  cols <- names(df)[grepl(col.start, names(df))]
  for (name in cols) {
    # filter out values of 0, using as mask for corresponding columns
    nonzero <- df[[name]] > 0
    cat("---------------------------\n")
    if (any(nonzero)) {
      vals <- df[[name]][nonzero]
      genos <- df$Geno[nonzero]
      sides <- df$Side[nonzero]
      cat(name, ": ", vals, "\n")
      fit <- fitModel(model, vals, genos, sides, ids)
      hist(vals)
    } else {
      cat(name, ": no non-zero samples found\n\n")
    }
  }
}

setupPairing <- function(df.region, col, split.col) {
  # Setup data frame for comparing paired groups.
  #
  # Assume that the data frame has been sorted by sample so that samples 
  # will be matched after splitting by split.col.
  #
  # Args:
  #   df.region: Data frame only the samples to compare.
  #   col: Column name of values to compare.
  #   split.col: Column name by which to split samples into groups, assumed 
  #     to have at least two groups within column.
  #
  # Returns:
  #   New data frame filtering out any pair that lacks positive values for 
  #   both members of the pair, or NULL if unable to pair.
  
  # require at least 2 condition to compare
  conditions <- df.region[[split.col]]
  conditions.unique = sort(unique(conditions))
  if (length(conditions.unique) < 2) {
    cat("need at least 2 conditions, cannot set up pairing\n")
    return(NULL)
  }
  vals <- df.region[[col]]
  
  # build up nonzero mask to filter out any pairs with any zero vals
  nonzero <- NULL
  num.per.cond <- NULL
  for (cond in conditions.unique) {
    val.cond <- vals[cond == conditions]
    # TODO: look for all non-zero, not just pos vals
    nonzero.cond <- val.cond > 0 & !is.nan(val.cond)
    if (is.null(nonzero)) {
      nonzero <- nonzero.cond
    } else if (num.per.cond != length(val.cond)) {
      cat("unequal number of values per conditions, cannot match pairs\n")
      return(NULL)
    } else {
      nonzero <- nonzero & nonzero.cond
    }
    num.per.cond <- length(val.cond)
  }
  
  df.filtered <- NULL
  num.nonzero <- sum(nonzero)
  if (num.nonzero == 0) {
    cat("no non-zero values would remain after filtering\n")
    return(NULL)
  } else if (num.nonzero < num.per.cond) {
    # build new data frame with each condition filtered by mask
    for (cond in conditions.unique) {
      df.nonzero <- df.region[cond == conditions, ][nonzero, ]
      if (is.null(df.filtered)) {
        df.filtered <- df.nonzero
      } else {
        df.filtered <- rbind(df.filtered, df.nonzero)
      }
    }
  } else {
    # no filtering required
    df.filtered <- df.region
  }
  return(df.filtered)
}

statsByRegion <- function(df, col, model, split.by.side=TRUE, 
                          regions.ignore=NULL) {
  # Calculate statistics given by region for columns starting with the given 
  # string using the selected model.
  #
  # Values of 0 will be ignored. If all values for a given vector are 0, 
  # statistics will not be computed.
  #
  # Args:
  #   df: Data frame with columns for Genos, Sides, Region, and name given by 
  #     col.
  #   col: Column from which to find main stats.
  #   model: Model to use, corresponding to one of kModel.
  #   split.by.side: True to keep data split by sides, False to combine 
  #     corresponding regions from opposite sides into single regions; 
  #     defaults to True.
  #   regions.ignore: Vector of regions to ignore; default to NULL.
  
  # find all regions
  regions <- unique(df$Region)
  #regions <- c() # TESTING: insert single region
  cols <- c("Region", "Stats", "Volume")
  stats <- data.frame(matrix(nrow=length(regions), ncol=length(cols)))
  names(stats) <- cols
  # use original order of appearance in Condition column to sort each 
  # region since order may change from region to region
  cond.unique <- unique(df$Condition)
  regions.ignored <- vector()
  
  for (i in seq_along(regions)) {
    region <- regions[i]
    if (!is.null(regions.ignore) & is.element(region, kRegionsIgnore)) next
    
    # filter data frame for the given region and get mask to filter out 
    # NaNs and 0's as they indicate that the label for the region was suppressed
    df.region <- df[df$Region == region, ]
    nonzero <- df.region[[col]] > 0 & !is.nan(df.region[[col]])
    stats$Region[i] <- region
    
    if (any(nonzero)) {
      cat("\nRegion", region, "\n")
      df.region.nonzero <- df.region
      split.col <- NULL
      paired <- is.element(model, kModel[7:9])
      
      if (!paired) {
        # filter each column within region for rows with non-zero values
        df.region.nonzero <- df.region.nonzero[nonzero, ]
        if (is.null(df.region.nonzero)) next
      }
      if (is.element(model, kModel[5:9])) {
        # filter for means tests, which split by "Condition" column; 
        # TODO: reconsider aggregating sides but need way to properly 
        # average variations in a weighted manner
        split.col <- "Condition"
        if (paired) {
          # sort by sample and condition, matching saved condition order, 
          # split by condition, and filter out pairs where either sample 
          # has a zero value
          #print(df.region.nonzero)
          df.region.nonzero <- df.region.nonzero[
            order(df.region.nonzero$Sample, 
                  match(df.region.nonzero$Condition, cond.unique)), ]
          df.region.nonzero <- setupPairing(df.region.nonzero, col, split.col)
          if (is.null(df.region.nonzero)) next
        }
      }
      #print(df.region.nonzero)
      vals <- df.region.nonzero[[col]]
      
      # apply stats and store in stats data frame, using list to allow 
      # arbitrary size and storing mean volume as well
      if (is.element(model, kModel[5:9])) {
        # means tests
        coef.tab <- meansModel(
          vals, df.region.nonzero$Condition, model, paired)
      } else {
        # regression tests
        genos <- df.region.nonzero$Geno
        sides <- df.region.nonzero$Side
        ids <- df.region.nonzero$Sample
        coef.tab <- fitModel(model, vals, genos, sides, ids)
      }
      if (!is.null(coef.tab)) {
        stats$Stats[i] <- list(coef.tab)
        stats$Volume[i] <- mean(df.region.nonzero$Volume)
      }
      
      # show histogram to check for parametric distribution
      #histogramPlot(vals, title, meas)
      
      # plot individual values grouped by genotype and selected column
      df.jitter <- df.region.nonzero
      title <- paste0(df.region.nonzero$RegionName[1], " (", region, ")")
      if (!split.by.side) {
        df.jitter <- aggregate(
          cbind(Volume, Nuclei) ~ Sample + Geno, df.jitter, sum)
        df.jitter$Density <- df.jitter$Nuclei / df.jitter$Volume
        print(df.jitter)
      }
      stats.group <- jitterPlot(
        df.jitter, col, title, split.by.side, split.col, paired)
      
      # add mean and CI for each group to stats data frame
      names <- stats.group[[1]]
      for (j in seq_along(names)) {
        stats[i, paste0(names[j], ".mean")] <- stats.group[[2]][j]
        stats[i, paste0(names[j], ".ci")] <- stats.group[[3]][j]
      }
    } else {
      # ignore region if all values 0, leaving entry for region as NA and 
      # grouping output for empty regions to minimize console output; 
      # TDOO: consider grouping into list and displaying only at end
      regions.ignored <- append(regions.ignored, region)
    }
  }
  if (length(regions.ignored) > 0) {
    cat("\nno non-zero samples found for these regions:")
    print(regions.ignored)
  }
  return(stats)
}

histogramPlot <- function(vals, title, meas) {
  # Plot histogram and save to file.
  #
  # Args:
  #   vals: Values to plot.
  #   title: Title for plot.
  #   meas: Measurement to include in filename
  
  # may need to tweak width divisor to fit exactly in fig
  hist(vals, main=strwrap(title, width=dev.size("px")[1]/10))
  dev.print(
    pdf, file=paste0(
      "../plot_histo_", meas, "_", gsub("/| ", "_", title), ".pdf"))
}

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
  
  # max y-val or error bar, whichever is higher
  maxes <- c(num.groups, max(max(vals), max(max.errs)))
  if (is.element(col, names(kMeasNames))) {
    ylab <- kMeasNames[[col]]
  } else {
    ylab <- gsub("_", " ", col)
  }
  
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
  colors <- viridis(num.sides, begin=0.2, end=0.8)
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
      if (num.sides > 0) colors <- rainbow(length(vals.groups[[1]]), end=0.8)
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
      col <- if (show.sample.legend) colors else colors[i]
      points(x.vals, vals.group / denom, pch=i+14, col=col)
      
      # plot error bars unless CI is NA, such as infinitely large CI when n = 1
      mean <- vals.means[[i]]
      ci <- vals.cis[[i]]
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
        col <- if(show.sample.legend) colors[j] else 1
        segments(x.pos[1], vals.group[j], x.pos[2], 
                 vals.geno[[2]][j], col=col)
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
  col <- if(show.sample.legend) 1 else colors
  legend("topright", legend=names.groups, pch=15:(14+length(names.groups)), 
         xpd=TRUE, inset=c(0, 1), horiz=TRUE, bty="n", col=col)
  
  # save figure to PDF
  dev.print(
    pdf, width=7, height=7, 
    file=paste0(
      "../plot_jitter_", meas, "_", gsub("/| ", "_", title), ".pdf"))
  par(par.old)
  
  return(list(names.groups, vals.means, vals.cis))
}

filterStats <- function(stats, corr=NULL) {
  # Filter regional statistics to remove \code{NA}s and gather the most 
  # pertinent statistical values.
  #
  # Args:
  #   stats: Data frame generated by \code{\link{statsByRegion}}.
  #
  # Returns:
  #   Filtered data frame with columns for Region, Effect, and p.
  
  non.na <- !is.na(stats$Stats)
  stats.filt <- stats[non.na, ]
  filtered <- NULL
  interactions <- NULL
  offset <- 0 # number of columns ahead of coefficients
  
  # get names and mean and CI columns
  cols.names <- names(stats.filt)
  cols.means.cis <- c(
    cols.names[grepl(".mean", cols.names)], 
    cols.names[grepl(".ci", cols.names)])
  
  for (i in 1:nrow(stats.filt)) {
    if (is.na(stats.filt$Stats[i])) next
    # get coefficients, stored in one-element list
    stats.coef <- stats.filt$Stats[i][[1]]
    if (is.null(filtered)) {
      # build data frame if not yet generated to store pertinent coefficients 
      # from each type of main effect or interaction
      interactions <- gsub(":", ".", rownames(stats.coef))
      cols <- list("Region", "Volume")
      offset <- length(cols)
      for (interact in interactions) {
        cols <- append(cols, paste0(interact, ".effect"))
        cols <- append(cols, paste0(interact, ".p"))
      }
      filtered <- data.frame(matrix(nrow=nrow(stats.filt), ncol=length(cols)))
      names(filtered) <- cols
      filtered$Region <- stats.filt$Region
      filtered$Volume <- stats.filt$Volume
    }
    for (j in seq_along(interactions)) {
      # insert effect, p-value, and -log(p) after region name for each 
      # main effect/interaction, ignoring missing rows
      if (nrow(stats.coef) >= j) {
        filtered[i, j * 3 - 2 + offset] <- stats.coef[j, 1]
        filtered[i, j * 3 - 1 + offset] <- stats.coef[j, 4]
      }
    }
    for (col in cols.means.cis) {
      # add all mean and CI column values
      filtered[i, col] <- stats.filt[i, col]
    }
  }
  num.regions <- nrow(filtered)
  if (!is.null(corr)) {
    cat("correcting for", num.regions, "regions\n")
  }
  for (interact in interactions) {
    col <- paste0(interact, ".p")
    col.for.log <- col
    if (!is.null(corr)) {
      # apply correction based on number of comparisons
      col.for.log <- paste0(col, "corr")
      filtered[[col.for.log]] <- p.adjust(
        filtered[[col]], method="bonferroni", n=num.regions)
    }
    # calculate -log-p values
    filtered[[paste0(interact, ".logp")]] <- -1 * log10(filtered[[col.for.log]])
  }
  return(filtered)
}

volcanoPlot <- function(stats, meas, interaction, thresh=NULL, 
                        log.scale.x=TRUE) {
  # Generate a volcano plot.
  #
  # Args:
  #   stats: Data frame generated by \code{\link{filterStats}}.
  #   meas: Measurement to display in plot title.
  #   interaction: Interaction column name whose set of stats should be 
  #     displayed.
  #   thresh: Threshold as a 2-element array corresponding to the x and y 
  #     values, respectively, above which labels will be shown if both 
  #     condition is met. NA for either value will cause the threshold to 
  #     be ignored, or NA for the entire argument will ignore thresholding 
  #     all together.
  #   log.scale.x: True to scale x-axis by log10, first normalizing to 
  #     the minimum x-value before taking the log of the absolute value 
  #     to avoid negative log values, then returning to the original sign.
  
  x <- stats[[paste0(interaction, ".effect")]]
  num.x <- length(x)
  if (num.x < 1) {
    cat("no values found to generate volcano plot, skipping\n")
    return()
  }
  y <- stats[[paste0(interaction, ".logp")]]
  # weight size based on mean volume, replacing NaNs with a small num
  vol <- stats$Volume
  if (isTRUE(all.equal(vol, rep(0, num.x)))) {
    size <- rep(1, num.x)
  } else {
    vol[is.nan(vol) | vol == 0] <- 1
    size <- sqrt(vol / max(vol)) * 3
  }
  #print(data.frame(x, size))
  
  # point colors based on IDs of parents at the level generated for region 
  # IDs file, using a palette with color for each unique parent
  parents <- stats$Parent
  parents.unique <- unique(parents)
  parents.indices <- match(parents, parents.unique)
  colors <- viridis(length(parents.unique))
  colors_parents <- colors[parents.indices]
  
  # base plot -log p vs effect size
  if (log.scale.x) {
    # log scale x, rescaling so abs vals of x are >= 1 to get pos log vals 
    # and changing back to original sign
    x.neg <- x < 0
    x.norm <- x / min(abs(x))
    x.log <- log(abs(x.norm))
    x.log[x.neg] <- -1 * x.log[x.neg]
    print(data.frame(x, x.norm, x.neg, x.log, y))
    x <- x.log
  }
  x.max <- max(abs(x))
  if (is.element(meas, names(kMeasNames))) {
    title <- kMeasNames[[meas]]
  } else {
    title <- paste(meas, "Differences for", interaction)
  }
  plot(
    x, y, xlim=c(-1 * x.max, x.max), 
    main=title, xlab="Effects", 
    ylab="-log10(p)", type="p", las=1, pch=16, cex=size, col=colors_parents)
  # vertical line to denote x = 0
  abline(v=0, lty="dashed", col=gray(0.5, 0.5))
  
  # label points
  x.lbl <- x
  y.lbl <- y
  lbls <- stats$RegionAbbr
  if (!is.null(thresh)) {
    # limit labels only to those within all thresholds, ignoring NAs
    show.lbl <- abs(x) > 0
    if (!is.na(thresh[1])) show.lbl <- abs(x) > thresh[1] & show.lbl
    if (!is.na(thresh[2])) show.lbl <- y > thresh[2] & show.lbl
    x.lbl <- x[show.lbl]
    y.lbl <- y[show.lbl]
    lbls <- lbls[show.lbl]
  }
  if (length(lbls) > 0) {
    # place text labels; "text" has full overlap, "thigmophobe.labels" and 
    # "pointLabels" have varying degrees of overlap, and "addTextLabels" has 
    # the least overlap
    #text(x.lbl, y.lbl, label=lbls, cex=0.2)
    #thigmophobe.labels(x.lbl, y.lbl, label=lbls, cex=0.2)
    #pointLabel(x.lbl, y.lbl, label=lbls, cex=0.2)
    addTextLabels(x.lbl, y.lbl, label=lbls, cex=0.5, lwd=0.2)
  }
  
  # write to PDF file
  dev.print(
    pdf, file=paste("../plot_volcano", meas, paste0(interaction, ".pdf"), 
                    sep="_"))
}

calcVolStats <- function(path.in, path.out, meas, model, region.ids, 
                         split.by.side=TRUE, corr=NULL) {
  # Calculate volumetric stats from the given CSV file.
  #
  # Args:
  #   path.in: Path from which to load CSV to calculate stats, assumed to be 
  #     generated by \code{clrbrain.stats.regions_to_pandas} Python function.
  #   path.out: Path to output CSV file.
  #   meas: Column from which to generate stats, which should be one of 
  #     \code{\link{kMeas}}.
  #   model: Model type to use for stats, which should be one of 
  #     \code{\link{kModel}}.
  #   split.by.side: True to plot separate sub-scatter plots for each 
  #     region by side; defaults to True.
  #
  # Returns:
  #   Filtered data frame from \code{\link{filterStats}}.
  
  # load CSV file output by Clrbrain Python stats module
  df <- read.csv(path.in)
  
  # convert summary regions into "Mus Musculus" (ID 15564), the 
  # over-arching parent, which will be skipped if in kRegionsIgnore
  region.all <- df$Region == "all"
  if (any(region.all)) {
    df$Region <- as.integer(df$Region)
    df$Region[region.all] <- 15564
  }
  
  # merge in region names based on matching IDs
  df <- merge(df, region.ids, by="Region")
  print.data.frame(df)
  print(str(df)) # show data frame structure
  cat("\n\n")
  
  # calculate stats, filter out NAs and extract effects and p-values
  regions.ignore <- NULL
  if (basename(path.in) == kStatsFilesIn[2]) {
    # ignore duplicate when including all levels
    regions.ignore <- kRegionsIgnore
  }
  stats <- statsByRegion(
    df, meas, model, split.by.side=split.by.side, regions.ignore=regions.ignore)
  stats.filtered <- filterStats(stats, corr=corr)
  stats.filtered <- merge(stats.filtered, region.ids, by="Region")
  print(stats.filtered)
  write.csv(stats.filtered, path.out)
  return(stats.filtered)
}

#######################################
# choose measurement and model types
kStatsPathIn <- file.path("..", kStatsFilesIn[2])
meas <- kMeas[4]
model <- kModel[8]
split.by.side <- TRUE # false to combine sides
load.stats <- FALSE # true to load saved stats and only regenerate volcano plots

# set up paramters based on chosen model
stat <- "vals"
if (model == kModel[2]) {
  stat <- "genos"
}

# calculate stats or retrieve from file
region.ids <- read.csv(kRegionIDsPath)
if (load.stats && file.exists(kStatsPathOut)) {
  stats <- read.csv(kStatsPathOut)
} else {
  stats <- calcVolStats(
    kStatsPathIn, kStatsPathOut, meas, model, region.ids, 
    split.by.side=split.by.side, corr="bonferroni")
}

# plot effects and p's
volcanoPlot(stats, meas, stat, thresh=c(NA, 1.3))
volcanoPlot(stats, meas, "sidesR", thresh=c(25, 2.5))
# ":" special character automatically changed to "."
volcanoPlot(stats, meas, paste0(stat, ".sidesR"), thresh=c(1e-04, 25))
