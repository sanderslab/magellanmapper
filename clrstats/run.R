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

# use tryCatchLog to include line numbers with errors, which is not
# available with tryCatch
futile.logger::flog.threshold(futile.logger::ERROR)
tryCatchLog::tryCatchLog({
  # set args.parsed directly if sourcing this file since arguments cannot
  # be given to the parser
  if (!exists("args.parsed")) {
    # parse command-line arguments if args have not been set explicitly
    parser <- optparse::OptionParser()
    parser <- optparse::add_option(
      parser, c("-v", "--verbose"), action="store_true",
      help="Show verbose debugging information")
    parser <- optparse::add_option(
      parser, c("-f", "--file"), type="character",
      help="Input file, generally a CSV/TSV file")
    parser <- optparse::add_option(
      parser, c("-p", "--profiles"), type="character",
      help="Profile names delimited by comma")
    parser <- optparse::add_option(
      parser, c("-m", "--meas"), type="character",
      help="Names of measurent columns on which to perform stats")
    parser <- optparse::add_option(
      parser, "--prefix", type="character",
      help="Path prefix")
    args.parsed <- optparse::parse_args(parser)
  }
  cat("Parsed arguments:", paste(args.parsed), "\n")

  # run main statistics
  print(args.parsed$meas)
  runStats(args.parsed$file, args.parsed$profiles, args.parsed$meas,
           args.parsed$prefix, args.parsed$verbose)
}, finally={
  # return to original directory
  setwd(dir.start)
}, include.full.call.stack=FALSE)
