#!/usr/bin/env Rscript
# _p2_mml_control_2pl.R -- classical MML (mirt itemtype='2PL') fit for the 2PL
# oracle-ladder control ("Not All Parameters Learn Alike").  Sibling of
# _p2_mml_control.R (which fits itemtype='gpcm'); called by the sibling
# _p2_mml_control_2pl.py driver, not run standalone in the study.
#
# WHAT.  Fits itemtype='2PL' (dichotomous 2-parameter logistic) by marginal
# maximum likelihood to a learner x item response matrix supplied as SPARSE
# (learner,item,resp) triplets (0-based ids), and writes the recovered item
# parameters on the SAME (a, b) scale as the generating truth (2PL is the
# datagen GPCM kernel at K=2: psi_1 = a*(theta-b1), the textbook 2PL logistic;
# verified empirically that mirt's itemtype='2PL' + coef(IRTpars=TRUE) returns
# columns a, b, g, u with g=0/u=1 fixed, i.e. a and b directly, no manual
# conversion).  Point estimates only (SE=FALSE): a recovery-correlation
# control, not an inferential fit.
#
# Usage:
#   Rscript _p2_mml_control_2pl.R <in_triplets_csv> <out_items_csv> \
#       <out_meta_txt> <K> <method> <ncores>
#   Same argument contract as _p2_mml_control.R; <K> must be 2 here (itemtype
#   is hardcoded '2PL', not derived from K, since this script is 2PL-only).
#
# Scratch file: new, `_`-prefixed; does not touch any existing script.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop("usage: _p2_mml_control_2pl.R <in_triplets_csv> <out_items_csv> ",
       "<out_meta_txt> <K> <method> <ncores>")
}
in_csv    <- args[1]
out_csv   <- args[2]
meta_txt  <- args[3]
K         <- as.integer(args[4])
method    <- args[5]
ncores    <- as.integer(args[6])

suppressMessages(library(mirt))
suppressMessages(library(data.table))

write_meta <- function(...) writeLines(c(...), meta_txt)

t0 <- Sys.time()

tri <- tryCatch(fread(in_csv, colClasses = c("integer", "integer", "integer")),
                error = function(e) e)
if (inherits(tri, "error")) {
  write_meta("STATUS: ERROR", paste("MESSAGE: read failed:", conditionMessage(tri)),
             paste("WALL_S:", as.numeric(difftime(Sys.time(), t0, units = "secs"))),
             paste("METHOD:", method))
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
colnames(df) <- paste0("V", seq_len(n_items))   # mirt-safe names; item id kept in item_ids

if (ncores > 1) {
  suppressMessages(tryCatch(mirtCluster(ncores), error = function(e) NULL))
}

fit <- tryCatch(
  mirt(df, model = 1, itemtype = "2PL", SE = FALSE, method = method,
       verbose = TRUE, technical = list(NCYCLES = 800)),
  error = function(e) e
)

wall_s <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

if (inherits(fit, "error")) {
  write_meta("STATUS: ERROR", paste("MESSAGE:", conditionMessage(fit)),
             paste("WALL_S:", wall_s), paste("METHOD:", method),
             paste("N_ITEMS:", n_items), paste("N_LEARNERS_USED:", n_learners_used))
  quit(status = 1, save = "no")
}

conv  <- isTRUE(fit@OptimInfo$converged)
iters <- fit@OptimInfo$iter

co <- tryCatch(coef(fit, IRTpars = TRUE, simplify = TRUE)$items,
               error = function(e) e)
if (inherits(co, "error")) {
  write_meta("STATUS: ERROR", paste("MESSAGE: coef(IRTpars) failed:", conditionMessage(co)),
             paste("WALL_S:", wall_s), paste("METHOD:", method),
             paste("N_ITEMS:", n_items), paste("N_LEARNERS_USED:", n_learners_used))
  quit(status = 1, save = "no")
}

# 2PL IRTpars columns are a, b, g, u (g=0, u=1 fixed); keep only the single
# difficulty column so the output schema (item,a,b1..b_{K-1}) matches the gpcm
# sibling's contract with K-1=1 threshold column, named "b1" for the Python
# side's ``bcols`` sort convention.
out <- data.frame(item = item_ids, a = co[, "a"], b1 = co[, "b"])
write.csv(out, out_csv, row.names = FALSE)

write_meta("STATUS: OK", paste("CONVERGED:", conv), paste("ITER:", iters),
           paste("WALL_S:", wall_s), paste("METHOD:", method),
           paste("N_ITEMS:", n_items), paste("N_LEARNERS_USED:", n_learners_used),
           paste("K:", K))
