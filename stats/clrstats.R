# Clrbrain stats in R
# Author: David Young, 2018

kModel = c("logit", "linregr")

# logistic regression
statsByCols <- function(df, col.start, model) {
	cols <- names(df)[grepl(col.start, names(df))]
	for (name in cols) {
		nonzero <- df[[name]] > 0
		cat("---------------------------\n")
		if (any(nonzero)) {
			vals = df[[name]][nonzero]
			cat(name, ": ", vals, "\n")
			if (model == kModel[1]) {
				fit.logit <- glm(
					df$Geno[nonzero] ~ vals * df$Side[nonzero], 
					family=binomial(link="logit"))
				print(summary.glm(fit.logit))
			} else if (model == kModel[2]) {
				fit.linregr <- lm(vals ~ df$Geno[nonzero] * df$Side[nonzero])
				print(summary.lm(fit.linregr))
			}
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

