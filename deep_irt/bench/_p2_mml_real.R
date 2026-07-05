#!/usr/bin/env Rscript
# _p2_mml_real.R -- classical MML (mirt) fit for the REAL-data taxonomy check
# ("Not All Parameters Learn Alike").  Called by the sibling _p2_mml_real.py
# driver; not run standalone in the study.
#
# WHAT.  Fits itemtype='2PL' (dichotomous 2-parameter logistic) or itemtype=
# 'gpcm' (Muraki GPCM, K response categories) by marginal maximum likelihood to
# a learner x item response matrix supplied as SPARSE (learner,item,resp)
# triplets (0-based ids), and writes the recovered item parameters on mirt's
# own IRTpars(a, b[, b2, ...]) scale.  Point estimates only (SE=FALSE): this is
# a concordance/reliability control, not an inferential fit.  Generalizes
# _p2_mml_control.R's pivot+fit+coef skeleton to a caller-supplied itemtype
# instead of a hardcoded 'gpcm', and drops the BIG_Q EM/QMCEM/subset ladder
# (real-data cells here are always Q=250, N<=3000 -- plain EM is fast).
#
# Usage:
#   Rscript _p2_mml_real.R <in_triplets_csv> <out_items_csv> <out_meta_txt> \
#       <itemtype> <ncores>
#   in_triplets_csv : columns learner,item,resp (0-based item ids; only
#                     administered (learner,item) pairs -- sparse triplets)
#   out_items_csv   : written -- columns item,a,b[,b2,...] (0-based item ids,
#                     one row per DISTINCT item id present in the input)
#   out_meta_txt    : written -- KEY: value lines: STATUS, CONVERGED, ITER,
#                     WALL_S, ITEMTYPE, N_ITEMS, N_LEARNERS_USED, MESSAGE (on
#                     error)
#   itemtype        : "2PL" | "gpcm"
#   ncores          : mirtCluster() worker count; 0/1 disables the cluster
#
# Scratch file: new, `_`-prefixed; does not touch any existing script.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop("usage: _p2_mml_real.R <in_triplets_csv> <out_items_csv> ",
       "<out_meta_txt> <itemtype> <ncores>")
}
in_csv    <- args[1]
out_csv   <- args[2]
meta_txt  <- args[3]
itemtype  <- args[4]
ncores    <- as.integer(args[5])

suppressMessages(library(mirt))
suppressMessages(library(data.table))

write_meta <- function(...) writeLines(c(...), meta_txt)

t0 <- Sys.time()

tri <- tryCatch(fread(in_csv, colClasses = c("integer", "integer", "integer")),
                error = function(e) e)
if (inherits(tri, "error")) {
  write_meta("STATUS: ERROR", paste("MESSAGE: read failed:", conditionMessage(tri)),
             paste("WALL_S:", as.numeric(difftime(Sys.time(), t0, units = "secs"))),
             paste("ITEMTYPE:", itemtype))
  quit(status = 1, save = "no")
}

# Pivot long (learner,item,resp) -> wide (one row per learner, one column per
# item; NA where not administered).  data.table::dcast sorts the new columns
# by the ascending numeric value of `item`.
wide <- dcast(tri, learner ~ item, value.var = "resp")
wide[, learner := NULL]
item_ids <- as.integer(colnames(wide))
ord <- order(item_ids)
wide <- wide[, ord, with = FALSE]
item_ids <- item_ids[ord]
n_items <- length(item_ids)
n_learners_used <- nrow(wide)

df <- as.data.frame(lapply(wide, as.integer))

# Real (non-spiraled) data can leave an item with a single observed response
# category in a given fit (whole matrix or a learner half) -- e.g. an item
# every remaining learner got right on their first attempt.  mirt cannot
# estimate such an item; drop it and note the ids rather than let the whole
# fit fail.
n_unique <- vapply(df, function(col) length(unique(col[!is.na(col)])), integer(1))
degenerate <- n_unique < 2
dropped_ids <- item_ids[degenerate]
if (any(degenerate)) {
  df <- df[, !degenerate, drop = FALSE]
  item_ids <- item_ids[!degenerate]
  n_items <- length(item_ids)
}
colnames(df) <- paste0("V", seq_len(n_items))   # mirt-safe names; item id kept in item_ids

if (ncores > 1) {
  suppressMessages(tryCatch(mirtCluster(ncores), error = function(e) NULL))
}

fit <- tryCatch(
  mirt(df, model = 1, itemtype = itemtype, SE = FALSE, method = "EM",
       verbose = TRUE, technical = list(NCYCLES = 800)),
  error = function(e) e
)

wall_s <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

if (inherits(fit, "error")) {
  write_meta("STATUS: ERROR", paste("MESSAGE:", conditionMessage(fit)),
             paste("WALL_S:", wall_s), paste("ITEMTYPE:", itemtype),
             paste("N_ITEMS:", n_items), paste("N_LEARNERS_USED:", n_learners_used))
  quit(status = 1, save = "no")
}

conv  <- isTRUE(fit@OptimInfo$converged)
iters <- fit@OptimInfo$iter

co <- tryCatch(coef(fit, IRTpars = TRUE, simplify = TRUE)$items,
               error = function(e) e)
if (inherits(co, "error")) {
  write_meta("STATUS: ERROR", paste("MESSAGE: coef(IRTpars) failed:", conditionMessage(co)),
             paste("WALL_S:", wall_s), paste("ITEMTYPE:", itemtype),
             paste("N_ITEMS:", n_items), paste("N_LEARNERS_USED:", n_learners_used))
  quit(status = 1, save = "no")
}

bcols <- grep("^b", colnames(co), value = TRUE)
# "2PL" -> single "b"; "gpcm" -> "b1".."b_{K-1}", ordered ascending.
if (length(bcols) > 1) {
  bcols <- bcols[order(as.integer(sub("^b", "", bcols)))]
}
out <- data.frame(item = item_ids, a = co[, "a"], co[, bcols, drop = FALSE])
write.csv(out, out_csv, row.names = FALSE)

write_meta("STATUS: OK", paste("CONVERGED:", conv), paste("ITER:", iters),
           paste("WALL_S:", wall_s), paste("ITEMTYPE:", itemtype),
           paste("N_ITEMS:", n_items), paste("N_LEARNERS_USED:", n_learners_used),
           paste("N_DROPPED_DEGENERATE:", length(dropped_ids)),
           paste("DROPPED_DEGENERATE:", paste(dropped_ids, collapse = ";")))
