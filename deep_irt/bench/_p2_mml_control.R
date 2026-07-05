#!/usr/bin/env Rscript
# _p2_mml_control.R -- classical MML (mirt) fit for the amortization-claim
# control ("Not All Parameters Learn Alike").  Called by the sibling
# _p2_mml_control.py driver; not run standalone in the study.
#
# WHAT.  Fits itemtype='gpcm' (Muraki GPCM, K response categories) by marginal
# maximum likelihood to a learner x item response matrix supplied as SPARSE
# (learner,item,resp) triplets (0-based ids), and writes the recovered item
# parameters on the SAME (a, b1..b_{K-1}) scale as the generating truth
# (deep_irt/bench/datagen.py::_gpcm_probs is the textbook Muraki GPCM
# psi_k = sum_{v<=k} a*(theta - b_v), which is exactly mirt's own
# parameterization, so coef(..., IRTpars=TRUE) hands back a/b directly, no
# manual d -> b conversion needed).  Point estimates only (SE=FALSE): this is
# a recovery-correlation control, not an inferential fit.
#
# Usage:
#   Rscript _p2_mml_control.R <in_triplets_csv> <out_items_csv> <out_meta_txt> \
#       <K> <method> <ncores>
#   in_triplets_csv : columns learner,item,resp (0-based item ids; only
#                     administered (learner,item) pairs -- sparse triplets)
#   out_items_csv   : written -- columns item,a,b1..b_{K-1} (0-based item ids,
#                     one row per DISTINCT item id present in the input)
#   out_meta_txt    : written -- KEY: value lines (no JSON dependency):
#                     STATUS, CONVERGED, ITER, WALL_S, METHOD, N_ITEMS,
#                     N_LEARNERS_USED, MESSAGE (on error)
#   K               : number of response categories (integer, e.g. 4)
#   method          : "EM" | "QMCEM"
#   ncores          : mirtCluster() worker count; 0/1 disables the cluster
#
# Scratch file: new, `_`-prefixed; does not touch any existing script.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop("usage: _p2_mml_control.R <in_triplets_csv> <out_items_csv> ",
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
  mirt(df, model = 1, itemtype = "gpcm", SE = FALSE, method = method,
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

bcols <- grep("^b", colnames(co), value = TRUE)
bcols <- bcols[order(as.integer(sub("^b", "", bcols)))]
out <- data.frame(item = item_ids, a = co[, "a"], co[, bcols, drop = FALSE])
write.csv(out, out_csv, row.names = FALSE)

write_meta("STATUS: OK", paste("CONVERGED:", conv), paste("ITER:", iters),
           paste("WALL_S:", wall_s), paste("METHOD:", method),
           paste("N_ITEMS:", n_items), paste("N_LEARNERS_USED:", n_learners_used),
           paste("K:", K))
