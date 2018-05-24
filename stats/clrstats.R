# Clrbrain stats in R
# Author: David Young, 2018

library("gee")

kModel = c("logit", "linregr", "gee")

# logistic regression
statsByCols <- function(df, col.start, model) {
	# Calculates statistics for columns starting with the given string using 
	# the selected model
	#
	# Values of 0 will be ignored. If all values for a given vector are 0, 
	# statistics will not be computed.
	#
	# Args:
	#   df: Data frame with columns for Genos, Sides, and names starting with 
	#     col.start.
	#   col.start: Columns starting with this string will be included.
	#   model: Model to use, corresponding to one of kModel.
	
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
			if (model == kModel[1]) {
				# logistic regression
				fit <- glm(genos ~ vals * sides, family=binomial)
				print(summary.glm(fit))
			} else if (model == kModel[2]) {
				# linear regression
				fit <- lm(vals ~ genos * sides)
				print(summary.lm(fit))
			} else if (model == kModel[3]) {
				# generalized estimating equations
				ids <- df$Sample[nonzero]
				fit <- gee(
					genos ~ vals * sides, ids, corstr="exchangeable", family=binomial())
			}
			hist(vals)
		} else {
			cat(name, ": no non-zero samples found\n\n")
		}
	}
}

# load CSV file output by Clrbrain Python stats module
df <- read.csv("../vols_by_sample.csv")
print.data.frame(df)
cat("\n\n")

print("Calculating logistic regressions...")
statsByCols(df, "Dens", kModel[1])
statsByCols(df, "Vol", kModel[1])

print("Calculating linear regressions...")
statsByCols(df, "Dens", kModel[2])
statsByCols(df, "Vol", kModel[2])

print("Calculating GEEs...")
statsByCols(df, "Dens", kModel[3])
statsByCols(df, "Vol", kModel[3])

