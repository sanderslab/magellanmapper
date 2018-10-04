# Clrbrain stats in R
# Author: David Young, 2018

library("MASS")
library("gee")
# library("plotly")
# library("ggplot2")
library("viridis")

# statistical models to use
kModel = c("logit", "linregr", "gee", "logit.ord")

# measurements, which correspond to columns in main data frame
kMeas = c("Vol", "Dens", "Nuclei")

# ordered genotype levels
kGenoLevels <- c(0, 0.5, 1)

# file paths
kStatsPathIn <- "../vols_by_sample.csv" # raw values from Clrbrain
kStatsPathOut <- "../vols_stats.csv" # output stats
kRegionIDsPath <- "../region_ids.csv" # region-ID map from Clrbrain

fitModel <- function(model, vals, genos, sides, ids=NULL) {
	# Fit data with the given model.
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
		print(coef.tab)
	} else if (model == kModel[2]) {
		# linear regression
		# TODO: see whether need to factorize genos
		fit <- lm(vals ~ genos * sides)
		coef.tab <- summary.lm(fit)$coefficients
		# remove first ("non-intercept") row
		coef.tab <- coef.tab[-(1:1), ]
		print(coef.tab)
	} else if (model == kModel[3]) {
		# generalized estimating equations
		# TODO: fix model prob "fitted value very close to 1" error
		fit <- gee(
			genos ~ vals * sides, ids, corstr="exchangeable", family=binomial())
		coef.tab <- summary(fit)$coefficients
		print(coef.tab)
	} else if (model == kModel[4]) {
		# ordered logistic regression
		vals <- scale(vals)
		genos <- factor(genos, levels=kGenoLevels)
		fit <- polr(genos ~ vals * sides, Hess=TRUE)
		coef.tab <- coef(summary(fit))
		# calculate p-vals and incorporate into coefficients
		p.vals <- pnorm(abs(coef.tab[, "t value"]), lower.tail=FALSE) * 2
		coef.tab <- cbind(coef.tab, "p value"=p.vals)
		print(coef.tab)
	} else {
		cat("Sorry, model", model, "not found\n")
	}
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

statsByRegion <- function(df, col, model, split.by.side=TRUE) {
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
	
	# find all regions
	regions <- unique(df$Region)#[2:2]
	cols <- c("Region", "Stats", "MeanNuclei")
	stats <- data.frame(matrix(nrow=length(regions), ncol=length(cols)))
	names(stats) <- cols
	for (i in seq_along(regions)) {
		region <- regions[i]
		# filter data frame for the given region
		df.region <- df[df$Region == region, ]
		# generate mask to filter out values of 0
		nonzero <- df.region[[col]] > 0
		cat("---------------------------\n")
		stats$Region[i] <- region
		if (any(nonzero)) {
			# filter each column within region for rows with non-zero values
			df.region.nonzero <- df.region[nonzero, ]
			vals <- df.region.nonzero[[col]]
			genos <- df.region.nonzero$Geno
			sides <- df.region.nonzero$Side
			ids <- df.region.nonzero$Sample
			cat("Region", region, ": ", vals, "from ", sum(nonzero), 
					"nonzero regions\n")
			
			# apply stats and store in stats data frame, using list to allow 
			# arbitrary size and storing mean nuclei as well
			coef.tab <- fitModel(model, vals, genos, sides, ids)
			stats$Stats[i] <- list(coef.tab)
			stats$MeanNuclei[i] <- mean(df.region$Nuclei)
			
			# show histogram to check for parametric distribution
			#hist(vals)
			
			title <- paste0(df.region.nonzero$RegionName[1], " (", region, ")")
			df.jitter <- df.region.nonzero
			if (!split.by.side) {
				print(df.region.nonzero)
				df.jitter <- aggregate(
					cbind(Vol, Nuclei) ~ Sample + Geno, df.jitter, sum)
				df.jitter$Dens <- df.jitter$Nuclei / df.jitter$Vol
				print(df.jitter)
			}
			jitterPlot(df.jitter, col, title, split.by.side)
		} else {
			# ignore region if all values 0, leaving entry for region as NA
			cat(region, ": no non-zero samples found\n\n")
		}
	}
	return(stats)
}

jitterPlot <- function(df.region, col, title, split.by.side=TRUE) {
	# Plot jitter/scatter plots of values by genotype with mean and 95% CI.
	#
	# Args:
	#   df.region: Date frame sliced by region, assumed to be filtered for 
	#     non-zero values.
	#   col: Name of column for values.
	#   title: Plot figure title.
	#   split.by.side: True to plot separate sub-scatter plots for each 
	#     region by side; defaults to True.
	
	genos <- df.region$Geno
			genos.unique <- sort(unique(genos))
	sides <- df.region$Side
	sides.unique <- sort(unique(sides))
	if (!split.by.side) {
		# use a single side for one for-loop pass
		sides.unique = c("")
	}
	vals <- df.region[[col]]
	maxes <- c(length(genos.unique) * length(sides.unique), max(vals))
	plot(NULL, frame.plot=TRUE, xlab=title, ylab=col, xaxt="n", 
					 xlim=range(-0.5, maxes[1] - 0.5), ylim=range(0, maxes[2]))
	names <- list()
			i <- 0
			for (geno in genos.unique) {
		x.adj <- 0
		mtext(geno, side=1, at=i+0.5)
		for (side in sides.unique) {
			if (split.by.side) {
			vals.geno <- vals[genos == geno & sides == side]
			} else {
				vals.geno <- vals[genos == geno]
			}
				print(vals.geno)
				num.vals <- length(vals.geno)
				# add jitter to distinguish points
			x <- i + x.adj
			points(jitter(rep(x, num.vals), amount=0.2), vals.geno, col=i+1, pch=16)
				vals.mean <- mean(vals.geno)
				vals.sd <- sd(vals.geno)
				vals.sem <- vals.sd / sqrt(num.vals)
			# use 97.5th percentile for 2-tailed 95% confidence level
				vals.ci <- qt(0.975, df=num.vals-1) * vals.sem
			segments(x - 0.25, vals.mean, x + 0.25, vals.mean)
			arrows(x, vals.mean + vals.ci, x, vals.mean - vals.ci, length=0.05, 
							 angle=90, code=3)
			names <- append(names, paste(geno, side))
				i <- i + 1
			x.adj <- x.adj + 0.05
			}
	}
	legend(0, maxes[2] * 0.5, names, col=1:length(names), pch=16)
	dev.print(
		pdf, file=paste0("../plot_jitter_", meas, "_", gsub("/| ", "_", title), ".pdf"))
}

filterStats <- function(stats) {
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
	for (i in 1:nrow(stats.filt)) {
		if (is.na(stats.filt$Stats[i])) next
		# get coefficients, stored in one-element list
		stats.coef <- stats.filt$Stats[i][[1]]
		if (is.null(filtered)) {
			# build data frame if not yet generated to store pertinent coefficients 
			# from each type of main effect or interaction
			interactions <- gsub(":", ".", rownames(stats.coef))
			cols <- list("Region", "MeanNuclei")
			offset <- length(cols)
			for (interact in interactions) {
				cols <- append(cols, paste0(interact, ".effect"))
				cols <- append(cols, paste0(interact, ".p"))
				cols <- append(cols, paste0(interact, ".logp"))
			}
			filtered <- data.frame(matrix(nrow=nrow(stats.filt), ncol=length(cols)))
			names(filtered) <- cols
			filtered$Region <- stats.filt$Region
			filtered$MeanNuclei <- stats.filt$MeanNuclei
		}
		for (j in seq_along(interactions)) {
			# insert effect, p-value, and -log(p) after region name for each 
			# main effect/interaction, ignoring missing rows
			if (nrow(stats.coef) >= j) {
				filtered[i, j * 3 - 2 + offset] <- stats.coef[j, 1]
				filtered[i, j * 3 - 1 + offset] <- stats.coef[j, 4]
				filtered[i, j * 3 + offset] <- -1 * log(stats.coef[j, 4])
			}
		}
	}
	return(filtered)
}

volcanoPlot <- function(stats, meas, interaction, thresh=NULL) {
	# Generate a volcano plot.
	#
	# Args:
	#   stats: Data frame generated by \code{\link{filterStats}}.
	#   meas: Measurement to display in plot title.
	#   interaction: Interaction column name whose set of stats should be 
	#     displayed.
	#   thresh: Threshold as a 2-element array corresponding to the x and y 
	#     values, respectively, above which labels will be shown if either 
	#     condition is met.
	
	x <- stats[[paste0(interaction, ".effect")]]
	y <- stats[[paste0(interaction, ".logp")]]
	# weight size based on relative num of nuclei
	size <- stats$MeanNuclei / max(stats$MeanNuclei) * 3
	# print(data.frame(x, size))
	
	# point colors based on IDs of parents at the level generated for region 
	# IDs file, using a palette with color for each unique parent
	parents <- stats$Parent
	parents.unique <- unique(parents)
	parents.indices <- match(parents, parents.unique)
	colors <- viridis(length(parents.unique))
	colors_parents <- colors[parents.indices]
	
	# base plot -log p vs effect size
	plot(
		x, y, main=paste(meas, "Differences for", interaction), xlab="Effects", 
		ylab="-log(p)", type="p", pch=16, cex=size, col=colors_parents)
	x.lbl <- x
	y.lbl <- y
	lbls <- paste(stats$Region, stats$RegionName, sep="\n")
	if (!is.null(thresh)) {
		y.high <- abs(x) > thresh[1] | y > thresh[2]
		x.lbl <- x[y.high]
		y.lbl <- y[y.high]
		lbls <- lbls[y.high]
	}
	if (length(lbls) > 0) {
		text(x.lbl, y.lbl, label=lbls, cex=0.1)
	}
	# plot_ly(data=stats, x=x, y=y)
	#g <- ggplot(data=stats, aes(x=x, y=y)) + geom_point(size=2)
	# ggplotly(g, tooltip=c("Region"))
	#print(g)
	dev.print(
		pdf, file=paste("../plot_volcano", meas, paste0(interaction, ".pdf"), sep="_"))
}

calcVolStats <- function(path.in, path.out, meas, model, region.ids) {
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
	#
	# Returns:
	#   Filtered data frame from \code{\link{filterStats}}.
	
	# load CSV file output by Clrbrain Python stats module
	df <- read.csv(path.in)
	# merge in region name based on matching IDs
	df <- merge(df, region.ids, by="Region")
	print.data.frame(df)
	cat("\n\n")
	
	# calculate stats, filter out NAs and extract effects and p-values
	stats <- statsByRegion(df, meas, model)#, split.by.side=FALSE)
	stats.filtered <- filterStats(stats)
	stats.filtered <- merge(stats.filtered, region.ids, by="Region")
	print(stats.filtered)
	write.csv(stats.filtered, path.out)
	return(stats.filtered)
}

#######################################
# choose measurement and model types
meas <- kMeas[2]
model <- kModel[4]


# set up paramters based on chosen model
stat <- "vals"
if (model == kModel[2]) {
	stat <- "genos"
}

# calculate stats or retrieve from file
region.ids <- read.csv(kRegionIDsPath)
if (file.exists(kStatsPathOut)) {
	stats <- read.csv(kStatsPathOut)
} else {
	stats <- calcVolStats(
		kStatsPathIn, kStatsPathOut, meas, model, region.ids)
}

# plot effects and p's
volcanoPlot(stats, meas, stat, thresh=c(1e-05, 0.8))
volcanoPlot(stats, meas, "sidesR", thresh=c(25, 2.5))
# ":" special character automatically changed to "."
volcanoPlot(stats, meas, paste0(stat, ".sidesR"), thresh=c(1e-04, 25))
