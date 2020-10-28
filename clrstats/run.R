#!/usr/bin/env R
# MagellanMapper R stats Command-Line Interface
# Author: David Young, 2020

# Usage:
#   1) From a shell: Rscript --verbose <path-to-clrstats>/run.R [options]
#   2) From an R session: source("<path-to-clrstats>/run.R")
#
# Run with `-h` flag to see options. To set options when running from an
# R session, create an `args.parsed` list and set these options in the list
# (eg `args.parsed$file <- path/to/my/file.csv`).

kWorkDir <- "clrstats"
dir.start <- getwd()
if (dir.exists(kWorkDir)) {
  # change working directory
  setwd(kWorkDir)
}

# load all source changes and run stats
devtools::load_all(file.path(getwd(), "R"))
tryCatch({
  # set args.parsed directly if sourcing this file since arguments cannot
  # be given to the parser
  if (!exists("args.parsed")) {
    # parse command-line arguments if args have not been set explicitly
    parser <- optparse::OptionParser()
    parser <- optparse::add_option(
      parser, c("-f", "--file"), type="character",
      help="Input file, generally a CSV/TSV file")
    parser <- optparse::add_option(
      parser, c("-p", "--profiles"), type="character",
      help="Profile names delimited by comma")
    args.parsed <- optparse::parse_args(parser)
  }
  cat("Parsed arguments:", paste(args.parsed), "\n")

  # run main statistics
  runStats()
}, error=function(e) {
  message(e)
}, finally={
  # return to original directory
  setwd(dir.start)
})
