#!/usr/bin/env R
# Simple R stats run script
# Author: David Young, 2020

# Usage:
#   Rscript --verbose clrstats/run.R
# Adjust profiles in clrstats.R before running

setwd("clrstats")
devtools::load_all(file.path(getwd(), "R"))
runStats()
