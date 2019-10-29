# Clrbrain stats in R
# Author: David Young, 2018, 2019

# library to avoid overlapping text labels
#install.packages("devtools")
#library("devtools")
#install_github("JosephCrispell/addTextLabels")

# To run stats:
# - Select profiles in runStats
# - Start R in clrstats folder
# - Load source: "devtools::load_all(".")"
# - Run stats: "runStats()"

# stat processing types
kStatTypes <- c(
  "default", "corr", "norm"
)

# statistical models
kModel <- c("logit", "linregr", "gee", "logit.ord", "ttest", "wilcoxon", 
           "ttest.paired", "wilcoxon.paired", "fligner", "basic")

# measurements, which correspond to columns in main data frame
kMeas <- c("Volume", "Density", "Nuclei", "VarIntensity", "VarNuclei", 
          "EdgeDistSum", "EdgeDistMean", "DSC_atlas_labels_hemisphere", 
          "Compactness", 
          "VarIntensBorder", "VarIntensMatch", "VarIntensDiff", 
          "CoefVarIntens", "CoefVarNuc", "MeanIntensity", "MeanNuclei", 
          "Intensity", "DSC", "Smoothing_quality",
          "VolDSC", "NucDSC", "VolOut", "NucOut")

# named list to convert measurement columns to display names, consisting 
# of lists of titles/labels and measurement units
kMeasNames <- setNames(
  list(list("Edge Match (Within-Region Intensity Variation)", "SD size"), 
       list("Edge Match (Within-Region Nuclei Variation)", "SD size"),
       list("Label Edge Distances to Anatomical Edges", bquote(list(mu*"m"))), 
       list("Label Edge Distances to Anatomical Edges (Mean)", 
            bquote(list(mu*"m"))), 
       list(paste("Labeled Hemisphere Atlas and Labels Overlay", 
                  "(Dice Similarity Coefficient)"), NULL), 
       list("Region Homogeneity (Core-Periphery Variation Match)", 
            "SD size difference"), 
       list("Edge Noise (Core-Periphery Variation Difference)", 
            "SD size difference"), 
       list("Edge Match (Within-Region Intensity Variation)", 
            "Coefficient of variation"), 
       list("Edge Match (Within-Region Nuclei Variation)", 
            "Coefficient of variation"), 
       list("Atlas and Labels Overlay (Dice Similarity Coefficient)", NULL)), 
  c(kMeas[c(4:8, 11:14, 18)]))

# ordered genotype levels
kGenoLevels <- c(0, 0.5, 1)

# regions to ignore (eg duplicates)
kRegionsIgnore <- c(15564)


# File Paths

# paths to files of raw metric values from Clrbrain
kStatsFilesIn <- c(
  "vols_by_sample.csv", "vols_by_sample_levels.csv", 
  "vols_by_sample_summary.csv", "dsc_summary.csv", 
  "compactness_summary.csv", "compactness_summary_stats.csv", 
  "reg_stats_melted.csv", "smoothing_gausVopen.csv",
  "vols_by_sample_compare.csv", "vols_by_sample_compare_levels.csv"
)
kStatsPathOut <- "../vols_stats" # output stats

# region-ID map from Clrbrain, which should contain all regions including 
# hierarchical/ontological ones
kRegionIDsPath <- "../region_ids.csv"

# configurable environment
config.env <- new.env()


fitModel <- function(model, vals, genos, sides, ids=NULL) {
  # Fit data with the given regression model.
  #
  # Args:
  #   model: Model to use, corresponding to one of kModel.
  #   vals: Main independent variable.
  #   genos: Genotypes vector.
  #   sides: Vector corresponding to the side of vals, eg left or right.
  #   ids: Vector of sample IDs; defaults to NULL.
  #
  # Returns:
  #   Coefficients of the summary statistics. The first row of coefficients 
  #   is removed if it is a non-intercept row. The colums are assumed to have 
  #   effect size in the 2nd column and p-value in the 4th.
  
  result <- NULL
  col.effect <- "Estimate"
  num.sides <- length(unique(sides))
  if (model == kModel[1]) {
    # logistic regression
    if (num.sides > 1) {
      fit <- glm(genos ~ vals * sides, family=binomial)
    } else {
      fit <- glm(genos ~ vals, family=binomial)
    }
    result <- summary.glm(fit)$coefficients
    # remove first ("non-intercept") row
    result <- result[-(1:1), ]
  } else if (model == kModel[2]) {
    # linear regression
    # TODO: see whether need to factorize genos
    fit <- lm(vals ~ genos * sides)
    result <- summary.lm(fit)$coefficients
    # remove first ("non-intercept") row
    result <- result[-(1:1), ]
  } else if (model == kModel[3]) {
    # generalized estimating equations
    # TODO: fix model prob "fitted value very close to 1" error
    fit <- gee::gee(
      genos ~ vals * sides, ids, corstr="exchangeable", family=binomial())
    result <- summary(fit)$coefficients
  } else if (model == kModel[4]) {
    # ordered logistic regression
    vals <- scale(vals)
    genos <- factor(genos, levels=kGenoLevels)
    fit <- tryCatch({
      fit <- MASS::polr(genos ~ vals * sides, Hess=TRUE)
      result <- coef(summary(fit))
      # calculate p-vals and incorporate into coefficients
      p.vals <- pnorm(abs(result[, "t value"]), lower.tail=FALSE) * 2
      result <- cbind(result, "p value"=p.vals)
    },
      error=function(e) {
        print(paste("Could not calculate ordered logistic regression", 
              e, "skipping"))
      }
    )
  } else {
    cat("Sorry, model", model, "not found\n")
  }
  
  # basic stats data frame in format for filterStats
  coef.tab <- setupBasicStats()
  effect <- result[[col.effect]]
  coef.tab$Value <- c(effect)
  stderr <- "Std. Error"
  if (is.element(stderr, names(result))) {
    # use SEM for the "CI" values
    ci <- result[[stderr]]
    coef.tab$CI.low <- c(effect - ci)
    coef.tab$CI.hi <- c(ci - effect)
  }
  coef.tab$P <- c(result[4])
  coef.tab$N <- c(length(vals))
  print(coef.tab)
  return(coef.tab)
}

meansModel <- function(vals, conditions, model, paired=FALSE, reverse=FALSE) {
  # Test for differences of means.
  #
  # Args:
  #   vals: List of values to compare.
  #   conditions: List assumed to have at least two conditions by which to 
  #     group; only the first two sorted conditions will be used, and all 
  #     other conditions will be ignored. For paired tests, number of 
  #     values per condition is assumed to be the same. Conditions are 
  #     sorted alphabetically, and effect sizes are generated by taking 
  #     values from the second minus that of the first condition.
  #   model: One of kModels to apply, which should be a mean test such as 
  #     "ttest" or "wilcoxon".
  #   paired: True for paired test; defaults to FALSE.
  #   reverse: True to reverse the order of sorted conditions; defaults 
  #     to FALSE.
  #
  # Returns:
  #   Data frame of with p-value and mean of differences columns.
  
  # require at least 2 condition to compare, sorting in direction based 
  # on reverse argument
  conditions.unique = sort(unique(conditions), decreasing=reverse)
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
    if (is.element(model, kModel[c(5, 7)]) & num.per.cond <= 1) {
      # T-tests requires >= 2 values
      cat("<2 values for at least one condition, cannot calculate stats\n")
      return(NULL)
    }
  }
  
  result <- NULL
  col.effect <- "estimate"
  if (model == kModel[5] | model == kModel[7]) {
    # Student's t-test
    result <- t.test(val.conds[[2]], val.conds[[1]], paired=paired)
  } else if (model == kModel[6] | model == kModel[8]) {
    # Wilcoxon test (Mann-Whitney if not paired)
    result <- wilcox.test(
      val.conds[[2]], val.conds[[1]], paired=paired, conf.int=TRUE)
  } else if (model == kModel[9]) {
    # Fligner-Killen test of variance
    result <- fligner.test(vals, conditions)
    col.effect <- "statistic"
  } else {
    cat("Sorry, model", model, "not found\n")
  }
  print(result)
  
  # basic stats data frame in format for filterStats
  coef.tab <- setupBasicStats()
  effect <- result[[col.effect]]
  coef.tab$Value <- c(effect)
  # get relative confidence intervals as pos vals
  if (is.element("conf.int", names(result))) {
    ci <- result$conf.int
    coef.tab$CI.low <- c(effect - ci[1])
    coef.tab$CI.hi <- c(ci[2] - effect)
  }
  coef.tab$P <- c(result$p.value)
  coef.tab$N <- c(num.per.cond)
  print(coef.tab)
  return(coef.tab)
}

setupBasicStats <- function() {
  # Setup a data frame for basic stats.
  #
  # Returns:
  #   Data frame with columns for basic statistics such as mean and 
  #   confidence intervals and a single empty row.
  
  cols <- c("N", "Value", "CI.low", "CI.hi", "P")
  coef.tab <- data.frame(matrix(nrow=1, ncol=length(cols)))
  names(coef.tab) <- cols
  rownames(coef.tab) <- c("vals")
  return(coef.tab)
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
  
  # build up nonnan mask to filter out any pairs with any zero vals
  nonnan <- NULL
  num.per.cond <- NULL
  for (cond in conditions.unique) {
    val.cond <- vals[cond == conditions]
    # TODO: look for all non-zero, not just pos vals
    nonnan.cond <- !is.nan(val.cond)
    if (is.null(nonnan)) {
      nonnan <- nonnan.cond
    } else if (num.per.cond != length(val.cond)) {
      cat("unequal number of values per conditions, cannot match pairs\n")
      return(NULL)
    } else {
      nonnan <- nonnan & nonnan.cond
    }
    num.per.cond <- length(val.cond)
  }
  
  df.filtered <- NULL
  num.nonnan <- sum(nonnan)
  if (num.nonnan == 0) {
    cat("no non-zero values would remain after filtering\n")
    return(NULL)
  } else if (num.nonnan < num.per.cond) {
    # build new data frame with each condition filtered by mask
    for (cond in conditions.unique) {
      df.nonnan <- df.region[cond == conditions, ][nonnan, ]
      if (is.null(df.filtered)) {
        df.filtered <- df.nonnan
      } else {
        df.filtered <- rbind(df.filtered, df.nonnan)
      }
    }
  } else {
    # no filtering required
    df.filtered <- df.region
  }
  return(df.filtered)
}

statsByRegion <- function(df, col, model, split.by.side=TRUE, 
                          regions.ignore=NULL, cond=NULL, group.col=NULL) {
  # Calculate statistics given by region for columns starting with the given 
  # string using the selected model.
  #
  # NaN values will be ignored. If all values for a given vector are NaN, 
  # statistics will not be computed.
  #
  # For non-paired stats, the comparison column is set by group.col. Paired
  # stats compare groups specified in the "Condition" column.
  #
  # Args:
  #   df: Data frame with at least columns for "Sample" and "Region".
  #   col: Column from which to find main stats.
  #   model: Model to use, corresponding to one of kModel.
  #   split.by.side: True to keep data split by sides, False to combine 
  #     corresponding regions from opposite sides into single regions; 
  #     defaults to True.
  #   regions.ignore: Vector of regions to ignore; default to NULL.
  #   cond: Filter df to keep only this condition; defaults to NULL.
  #   group.col: Name of group column; defaults to NULL, which uses "Condition"
  #     for means models and "Geno" otherwise.
  
  if (is.null(group.col)) {
    # set up default group column name
    if (is.element(model, kModel[5:9])) {
      # means models split by condition
      group.col <- "Condition"
    } else {
      # split by genotype
      group.col <- "Geno"
    }
  }
  
  # find all regions
  regions <- unique(df$Region)
  #regions <- c(15565) # TESTING: insert single region
  cols <- c("Region", "Stats", "Volume", "Nuclei")
  stats <- data.frame(matrix(nrow=length(regions), ncol=length(cols)))
  names(stats) <- cols
  if (!is.null(cond)) {
    # filter by the given condition
    df <- df[df$Condition == cond, ]
  }
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
    nonnan <- !is.nan(df.region[[col]])
    stats$Region[i] <- as.character(region)
    
    if (any(nonnan)) {
      cat("\nRegion", region, "\n")
      df.region.nonnan <- df.region
      split.col <- NULL
      paired <- is.element(model, kModel[7:9])
      
      if (!paired) {
        # filter each column within region for rows with non-zero values
        df.region.nonnan <- df.region.nonnan[nonnan, ]
        if (is.null(df.region.nonnan)) next
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
          #print(df.region.nonnan)
          df.region.nonnan <- df.region.nonnan[
            order(df.region.nonnan$Sample, 
                  match(df.region.nonnan$Condition, cond.unique)), ]
          df.region.nonnan <- setupPairing(df.region.nonnan, col, split.col)
          if (is.null(df.region.nonnan)) next
        }
      }
      #print(df.region.nonnan)
      vals <- df.region.nonnan[[col]]
      
      # apply stats and store in stats data frame, using list to allow 
      # arbitrary size and storing mean volume as well
      coef.tab <- NULL
      if (is.element(model, kModel[5:9])) {
        # means tests
        coef.tab <- meansModel(
          vals, df.region.nonnan[[group.col]], model, paired, 
          config.env$ReversePairedStats)
        
      } else if (model == kModel[10]) {
        # basic stats
        coef.tab <- setupBasicStats()
        
        # show histogram to check for parametric distribution
        #histogramPlot(vals, title, meas)
        
      } else {
        # regression tests
        genos <- df.region.nonnan[[group.col]]
        sides <- df.region.nonnan$Side
        ids <- df.region.nonnan$Sample
        coef.tab <- fitModel(model, vals, genos, sides, ids)
      }
      
      if (!is.null(coef.tab)) {
        # collect stats, taking means for weightings
        stats$Stats[i] <- list(coef.tab)
        stats$Volume[i] <- mean(df.region.nonnan$Volume)
        stats$Nuclei[i] <- mean(df.region.nonnan$Nuclei)
      }
      
      # construct title from region identifiers and capitalize first letter
      df.jitter <- df.region.nonnan
      region.name <- df.region.nonnan$RegionName[1]
      if (is.na(region.name)) {
        title <- region
        if (is.factor(title)) title <- as.character(title)
      } else {
        title <- paste0(region.name, " (", region, ")")
      }
      substring(title, 1, 1) <- toupper(substring(title, 1, 1))
      
      # plot individual values grouped by genotype and selected column
      if (!split.by.side) {
        df.jitter <- aggregate(
          cbind(Volume, Nuclei) ~ Sample + group.col, df.jitter, sum)
        df.jitter$Density <- df.jitter$Nuclei / df.jitter$Volume
        print(df.jitter)
      }
      # TODO: allow split.col to be ignored
      group.col.jitter <- group.col
      if (group.col.jitter == split.col) group.col.jitter <- "Geno"
      # TODO: set up groups and generate stats outside of jitter plots
      stats.group <- jitterPlot(
        df.jitter, col, title, group.col.jitter, split.by.side, split.col, 
        paired, config.env$SampleLegend, config.env$PlotSize, 
        axes.in.range=config.env$Axes.In.Range, 
        summary.stats=config.env$SummaryStats, 
        save=config.env$JitterPlotSave, sort.groups=config.env$Sort.Groups,
        show.labels=config.env$JitterLabels)
      
      # add mean, median, and CI for each group to stats data frame
      names <- stats.group[[1]]
      for (j in seq_along(names)) {
        stats[i, paste0(names[j], ".mean")] <- stats.group[[2]][j]
        stats[i, paste0(names[j], ".med")] <- stats.group[[3]][j]
        stats[i, paste0(names[j], ".sd")] <- stats.group[[4]][j]
        stats[i, paste0(names[j], ".ci")] <- stats.group[[5]][j]
      }
    } else {
      # ignore region if all values 0, leaving entry for region as NA and 
      # grouping output for empty regions to minimize console output; 
      # TODO: consider grouping into list and displaying only at end
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
  if (length(stats.filt$Stats) < 1) return(NULL)
  
  filtered <- NULL
  interactions <- NULL
  offset <- 0 # number of columns ahead of coefficients
  
  # get names and mean and CI columns
  cols.names <- names(stats.filt)
  cols.means.cis <- c(
    cols.names[grepl(".mean", cols.names)], 
    cols.names[grepl(".med", cols.names)], 
    cols.names[grepl(".sd", cols.names)], 
    cols.names[grepl(".ci", cols.names)])
  
  # build data frame for pertinent coefficients from each type of main 
  # effect or interaction
  stats.coef <- stats.filt$Stats[1][[1]]
  interactions <- gsub(":", ".", rownames(stats.coef))
  cols <- list("Region", "Volume", "Nuclei")
  cols.orig <- cols # points to original vector if it is mutated
  offset <- length(cols)
  for (interact in interactions) {
    cols <- append(cols, paste0(interact, ".n"))
    cols <- append(cols, paste0(interact, ".effect"))
    cols <- append(cols, paste0(interact, ".ci.low"))
    cols <- append(cols, paste0(interact, ".ci.hi"))
    cols <- append(cols, paste0(interact, ".p"))
    cols <- append(cols, paste0(interact, ".pcorr"))
    cols <- append(cols, paste0(interact, ".logp"))
  }
  filtered <- data.frame(matrix(nrow=nrow(stats.filt), ncol=length(cols)))
  names(filtered) <- cols
  for (col in cols.orig) {
    # copy base columns
    filtered[[col]] <- stats.filt[[col]]
  }
  num.stat.cols <- length(names(stats.coef))
  
  for (i in 1:nrow(stats.filt)) {
    if (is.na(stats.filt$Stats[i])) next
    # get coefficients, stored in one-element list
    stats.coef <- stats.filt$Stats[i][[1]]
    for (j in seq_along(interactions)) {
      # insert effect, p-value, and -log(p) after region name for each 
      # main effect/interaction, ignoring missing rows
      if (nrow(stats.coef) >= j) {
        start <- offset + 6 * (j - 1) + 1
        filtered[i, start:(start+num.stat.cols)] <- stats.coef[j, ]
      }
    }
    for (col in cols.means.cis) {
      # add all mean and CI column values
      filtered[i, col] <- stats.filt[i, col]
    }
  }
  
  num.regions <- nrow(filtered)
  for (interact in interactions) {
    col.for.log <- paste0(interact, ".pcorr")
    if (!is.null(corr)) {
      # apply correction based on number of comparisons
      # TODO: consider not correcting for means stats with < 2 vals
      col <- paste0(interact, ".p")
      cat("correcting", col, "by", corr, "for", num.regions, "regions\n")
      filtered[[col.for.log]] <- p.adjust(
        filtered[[col]], method=corr, n=num.regions)
    }
    # calculate -log-p values
    filtered[[paste0(interact, ".logp")]] <- -1 * log10(filtered[[col.for.log]])
  }
  
  return(filtered)
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
  if (!is.element(meas, names(df))) {
    cat(paste(meas, "not found in data frame", "\n"))
    return(NULL)
  }
  
  # convert summary regions into "Mus Musculus" (ID 15564), the 
  # over-arching parent, which will be skipped if in kRegionsIgnore
  region.all <- df$Region == "all"
  if (any(region.all)) {
    df$Region <- as.integer(df$Region)
    df$Region[region.all] <- 15564
  }
  
  # merge in region names based on matching IDs
  df <- merge(df, region.ids, by="Region", all.x=TRUE)
  #print.data.frame(df)
  print(str(df)) # show data frame structure
  cat("\n\n")
  
  # calculate stats, filter out NAs and extract effects and p-values
  regions.ignore <- NULL
  if (basename(path.in) == kStatsFilesIn[2]) {
    # ignore duplicate when including all levels
    regions.ignore <- kRegionsIgnore
  }
  stats <- statsByRegion(
    df, meas, model, split.by.side=split.by.side, regions.ignore=regions.ignore,
    cond=config.env$Condition, group.col=config.env$GroupCol)
  stats.filtered <- filterStats(stats, corr=corr)
  stats.filtered <- merge(region.ids, stats.filtered, by="Region", all.y=TRUE)
  print(stats.filtered)
  write.csv(stats.filtered, path.out)
  return(stats.filtered)
}

calcCorr <- function(path.in, cols, plot.size=c(5, 7), suffix=NULL) {
  # Calculate correlation coefficient matrix for columns in a data frame 
  # and plot with significance.
  #
  # Args:
  #   path.in: CSV input path for data frame.
  #   cols: Vector of columns for which to build correlation matrix.
  #   plot.size: Vector of width, height for exported plot; defaults to 
  #     c(5, 7).
  #   suffix: String of output path suffix inserted before the stat type;
  #     defaults to NULL.
  
  # load CSV to data frame, calculate correlation coefficient matrix, 
  # and save correlations and p-vals to CSV
  cat("\nloading", path.in, "\n")
  df <- read.csv(path.in)
  base.path <- tools::file_path_sans_ext(basename(path.in))
  corr <- Hmisc::rcorr(as.matrix(df[, cols]), type="spearman")
  print(corr)
  if (!is.null(suffix)) {
    base.path <- paste(base.path, suffix, sep="_")
  }
  out.path <- paste0("../", base.path, "_corr")
  write.csv(corr$r, paste0(out.path, "_r.csv"))
  write.csv(corr$P, paste0(out.path, "_p.csv"))
  
  # plot correlation matrix and save to PDF
  corrplot::corrplot(
    corr$r, method="circle", order="hclust", p.mat=corr$p, 
    sig.level=0.05, insig="p-value")
  plot.path <- paste0("../plot_corr_", base.path, ".pdf")
  dev.print(pdf, width=plot.size[1], height=plot.size[2], file=plot.path)
  cat(paste("Output plot to", plot.path, "\n"))
}

calcNormality <- function(path.in, cols) {
  # Calculate normality statistic using the Shapiro test.
  #
  # Args:
  #   path.in: CSV input path for data frame.
  #   cols: Vector of columns to test normality for each column.
  
  cat("\nloading", path.in, "\n")
  df <- read.csv(path.in)
  for (col in cols) {
    cat("\ntesting normality of col", col, "\n")
    vals <- df[[col]]
    vals <- vals[!is.nan(vals)]
    nor <- shapiro.test(vals)
    print(nor)
  }
}

setupConfig <- function(name=NULL) {
  # Setup configuration environment for the given profile.
  #
  # Args:
  #   name: Name of profile to load. Defaults to NULL, which will initialize  
  #     the environment with default settings.
  
  if (is.null(name)) {
    # initialize environment
    config.env$PlotSize <- c(5, 7)
    config.env$SampleLegend <- FALSE
    config.env$StatsPathIn <- file.path("..", kStatsFilesIn[2])
    config.env$Measurements <- kMeas[6]
    config.env$Model <- kModel[8]
    config.env$PlotVolcano <- TRUE
    config.env$VolcanoLabels <- TRUE
    config.env$VolcanoLogX <- TRUE
    config.env$JitterPlotSave <- TRUE
    config.env$JitterLabels <- FALSE
    config.env$Axes.In.Range <- FALSE
    config.env$ReversePairedStats <- FALSE
    config.env$SummaryStats <- kSummaryStats[2]
    config.env$GroupCol <- NULL
    config.env$Sort.Groups <- TRUE
    config.env$Condition <- NULL
    config.env$P.Corr <- "bonferroni"
    
  } else if (name == "aba") {
    # multiple distinct atlases
    config.env$SampleLegend <- TRUE
    config.env$Measurements <- kMeas[c(6, 13)]
    config.env$PlotVolcano <- FALSE
    setupConfig("skinny")
    
  } else if (name == "dsc") {
    # Dice Similarity Coefficient stats for ABA series
    setupConfig("aba")
    setupConfig("square")
    config.env$StatsPathIn <- file.path("..", kStatsFilesIn[4])
    config.env$Measurements <- kMeas[8]
    
  } else if (name == "smoothing") {
    # smoothing quality comparison between Gaussian and opening filter methods
    setupConfig("aba")
    config.env$StatsPathIn <- file.path("..", kStatsFilesIn[8])
    config.env$Measurements <- kMeas[19]
    
  } else if (name == "compactness") {
    # compactness combined jitter plot for ABA series by treating conditions 
    # as separate genotypes of the same "all" region
    setupConfig("aba")
    setupConfig("square")
    config.env$StatsPathIn <- file.path("..", kStatsFilesIn[5])
    config.env$Measurements <- kMeas[9]
    config.env$Sort.Groups <- FALSE
    
  } else if (name == "compactness.stats") {
    # compactness stats for ABA series by treating conditions as different 
    # regions to get corrected p-vals for all "regions"
    setupConfig("compactness")
    config.env$StatsPathIn <- file.path("..", kStatsFilesIn[6])
    config.env$JitterPlotSave <- FALSE
    
  } else if (name == "reg") {
    # WT registrations
    config.env$StatsPathIn <- file.path("..", kStatsFilesIn[7])
    config.env$Measurements <- kMeas[18]
    config.env$Model <- kModel[10]
    config.env$PlotVolcano <- FALSE
    config.env$Axes.In.Range <- TRUE
    config.env$SummaryStats <- kSummaryStats[1]
    config.env$Sort.Groups <- FALSE
    
  } else if (name == "compare.vol") {
    # basic stats from comparison of two atlases
    setupConfig("aba")
    config.env$StatsPathIn <- file.path("..", kStatsFilesIn[10])
    config.env$Model <- kModel[10]
    config.env$PlotVolcano <- FALSE
    config.env$Measurements <- kMeas[20:23]
    
  } else if (name == "nolevels") {
    # input file from drawn labels only, without levels
    config.env$StatsPathIn <- file.path("..", kStatsFilesIn[1])
    
  } else if (name == "nojittersave") {
    # plot but don't save jitter plots
    config.env$JitterPlotSave <- FALSE
    
  } else if (name == "wt") {
    # WT samples
    config.env$Measurements <- kMeas[c(4:7, 13:14)]
    config.env$VolcanoLabels <- FALSE
    config.env$VolcanoLogX <- FALSE
    config.env$JitterPlotSave <- FALSE
    
  } else if (name == "intensnuc") {
    # WT intensity and nuclei counts
    setupConfig("wt")
    config.env$StatsPathIn <- file.path("..", kStatsFilesIn[1])
    config.env$Measurements <- kMeas[c(3, 17)]
    
  } else if (name == "wt.test") {
    # WT test
    setupConfig("wt")
    config.env$Measurements <- kMeas[14]
    config.env$SampleLegend <- TRUE
    
  } else if (name == "geno") {
    # compare across multiple genotypes
    config.env$Measurements <- kMeas[1:3]
    config.env$Model <- kModel[1]
    config.env$VolcanoLogX <- FALSE
    config.env$Condition <- "smoothed"
    config.env$GroupCol <- "Geno"
    config.env$JitterLabels <- TRUE
    
  } else if (name == "compare.sex") {
    # compare sex instead of genotype
    setupConfig("geno")
    config.env$GroupCol <- "Sex"
    
  } else if (name == "compare.laterality") {
    # compare left/right hemispheres instead of genotype
    setupConfig("geno")
    config.env$GroupCol <- "Side"
    
  } else if (name == "lessstringent") {
    # compare 2 genotypes with slightly less stringent tests
    config.env$Model <- kModel[6]
    config.env$P.Corr <- "BH"
    
  } else if (name == "skinny") {
    # very narrow plots
    config.env$PlotSize <- c(3.5, 7)
    
  } else if (name == "skinny.small") {
    # narrow and short plots
    config.env$PlotSize <- c(3.5, 5)
    
  } else if (name == "square") {
    # square plots
    config.env$PlotSize <- c(7, 7)
    
  } else if (name == "revpairedstats") {
    # reverse the order of conditions in paired stats
    config.env$ReversePairedStats <- TRUE
    
  }
}

runStats <- function(stat.type=NULL) {
  # Load data and run full stats.
  #
  # Args:
  #   stat.type: One of kStatTypes specifying stat processing typest. 
  #     Defaults to NULL to use kStatTypes[1].
  
  # setup configuration environment
  setupConfig()
  
  #setupConfig("dsc")
  #setupConfig("aba")
  #setupConfig("smoothing")
  
  #setupConfig("wt")
  #setupConfig("intensnuc")
  #setupConfig("compactness")
  #setupConfig("compactness.stats")
  #setupConfig("reg")
  #setupConfig("wt.test")
  #setupConfig("compare.vol")
  
  #setupConfig("geno")
  #setupConfig("lessstringent")
  #setupConfig("compare.sex") # M vs F unpaired stats
  setupConfig("compare.laterality") # L vs R paired stats
  
  #setupConfig("nolevels")
  #setupConfig("nojittersave")
  #setupConfig("skinny.small")
  #setupConfig("square")
  setupConfig("revpairedstats")
  
  if (is.null(stat.type) || stat.type == kStatTypes[1]) {
    # default, general stats
    
    # setup measurement and model types
    split.by.side <- TRUE # false to combine sides
    load.stats <- FALSE # true to load saved stats, only regenerate volcano plots
    
    # set up parameters based on chosen model
    stat <- "vals"
    if (config.env$Model == kModel[2]) {
      stat <- "genos"
    }
    region.ids <- read.csv(kRegionIDsPath)
    
    # reset graphics to ensure consistent layout
    while (!is.null(dev.list())) dev.off()
    
    for (meas in config.env$Measurements) {
      print(paste("Calculating stats for", meas))
      # calculate stats or retrieve from file
      path.out <- paste0(kStatsPathOut, "_", meas, ".csv")
      if (load.stats && file.exists(path.out)) {
        stats <- read.csv(path.out)
      } else {
        stats <- calcVolStats(
          config.env$StatsPathIn, path.out, meas, config.env$Model, region.ids, 
          split.by.side=split.by.side, corr=config.env$P.Corr)
      }
      
      if (!is.null(stats) & config.env$PlotVolcano) {
        # plot effects and p's
        volcanoPlot(stats, meas, stat, c(NA, 1.3, 0.2), config.env$VolcanoLogX, 
                    config.env$VolcanoLabels, config.env$PlotSize, 
                    meas.names=kMeasNames)
        volcanoPlot(stats, meas, "sidesR", c(25, 2.5, 0.2), 
                    config.env$VolcanoLogX, config.env$VolcanoLabels, 
                    config.env$PlotSize, meas.names=kMeasNames)
        # ":" special character automatically changed to "."
        volcanoPlot(stats, meas, paste0(stat, ".sidesR"), c(1e-04, 25, 0.2), 
                    config.env$VolcanoLogX, config.env$VolcanoLabels, 
                    config.env$PlotSize, meas.names=kMeasNames)
      }
    }
    
  } else if (stat.type == kStatTypes[2]) {
    # correlation coefficient matrix; 
    # TODO: generalize scenarios
    #calcCorr(config.env$StatsPathIn, kMeas[c(4:6, 10)], config.env$PlotSize)
    
    # intensity vs. nuclei from atlas regions
    calcCorr("../vols_stats_intensVnuc.csv", 
             c("Intensity.original", "Intensity.smoothed", 
               "Nuclei.original", "Nuclei.smoothed"), 
             config.env$PlotSize)
    calcCorr("../vols_stats_intensVnuc.csv", 
             c("Intensity_density.original", 
               "Intensity_density.smoothed", 
               "Nuclei_density.original", "Nuclei_density.smoothed"), 
             config.env$PlotSize, "density")
    
    # from ROIs
    calcCorr("../vols_stats_intensVnuc_rois_combined_condtocol.csv", 
             c("Intensity_detected", "Intensity_truth", 
               "Nuclei_detected", "Nuclei_truth"), 
             config.env$PlotSize)
    
  } else if (stat.type == kStatTypes[3]) {
    # normality test
    calcNormality(config.env$StatsPathIn, config.env$Measurements)
  }
}
