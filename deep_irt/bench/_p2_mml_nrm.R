#!/usr/bin/env Rscript
# _p2_mml_nrm.R -- classical MML (mirt itemtype='nominal') fit for the NRM
# oracle ladder's missing mirt rung ("Not All Parameters Learn Alike").
# Sibling of _p2_mml_control.R (gpcm) / _p2_mml_control_2pl.R (2PL); called by
# the sibling _p2_mml_nrm.py driver, not run standalone in the study.
#
# WHAT.  Fits itemtype='nominal' (Bock 1972 / Thissen & Steinberg 1986 nominal
# response model, K unordered categories) by marginal maximum likelihood to a
# learner x item response matrix supplied as SPARSE (learner,item,resp)
# triplets (0-based ids), and writes the recovered per-category slopes/
# intercepts on the SAME sum-to-zero (Bock-centered) scale as the generating
# truth (deep_irt/bench/nrm_datagen.py::_sample_item_params centers both a_k
# and c_k across the K options: a -= rowMeans(a), c -= rowMeans(c)).
#
# GAUGE.  mirt's raw 'nominal' parameterization fixes ak0=0, ak(K-1)=(K-1),
# d0=0 for identification (verified empirically via mod2values() on a Q=30
# synthetic self-test: a1=slope multiplier, ak0..ak(K-1) category "scoring"
# weights, d0..d(K-1) category intercepts -- see scratch probe referenced in
# the calling instructions).  mirt's OWN coef(fit, IRTpars=TRUE) already
# performs the sum-to-zero recentering: per item, its "a1..aK"/"c1..cK" columns
# equal (a1*ak_k - mean_k(a1*ak_k)) and (d_k - mean_k(d_k)) respectively --
# verified to match a hand-reconstruction from the raw parameters to machine
# precision, and a full simulate->fit->recover round trip on that same Q=30
# bank recovered the planted a_k/c_k at Spearman 0.994 / 0.992.  So NO manual
# gauge conversion is coded here: IRTpars=TRUE's own output IS the sum-to-zero
# contrast the generator uses, column-for-column (a1<->option 0, a2<->option 1,
# ..., aK<->option K-1; same for c).  Point estimates only (SE=FALSE): a
# recovery-correlation control, not an inferential fit.
#
# Usage:
#   Rscript _p2_mml_nrm.R <in_triplets_csv> <out_items_csv> <out_meta_txt> \
#       <K> <method> <ncores>
#   in_triplets_csv : columns learner,item,resp (0-based item ids; only
#                     administered (learner,item) pairs -- sparse triplets)
#   out_items_csv   : written -- columns item,a1..aK,c1..cK (0-based item ids,
#                     one row per DISTINCT item id present in the input; column
#                     "a{k+1}"/"c{k+1}" is category k, k=0..K-1, matching the
#                     generator's (Q,K) column order -- NOT re-indexed)
#   out_meta_txt    : written -- KEY: value lines (no JSON dependency):
#                     STATUS, CONVERGED, ITER, NCYCLES_CAP, CAP_HIT, WALL_S,
#                     METHOD, N_ITEMS, N_LEARNERS_USED, MESSAGE (on error)
#   K               : number of response categories (integer, e.g. 4)
#   method          : "EM" | "QMCEM"
#   ncores          : mirtCluster() worker count; 0/1 disables the cluster
#
# Scratch file: new, `_`-prefixed; does not touch any existing script.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop("usage: _p2_mml_nrm.R <in_triplets_csv> <out_items_csv> ",
       "<out_meta_txt> <K> <method> <ncores>")
}
in_csv    <- args[1]
out_csv   <- args[2]
meta_txt  <- args[3]
K         <- as.integer(args[4])
method    <- args[5]
ncores    <- as.integer(args[6])
NCYCLES_CAP <- 800L    # same generosity as the gpcm/2pl siblings

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
  mirt(df, model = 1, itemtype = "nominal", SE = FALSE, method = method,
       verbose = TRUE, technical = list(NCYCLES = NCYCLES_CAP)),
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
cap_hit <- (!is.null(iters)) && (!is.na(iters)) && (iters >= NCYCLES_CAP)

# coef(IRTpars=TRUE): per item, columns a1..aK (category slopes) and c1..cK
# (category intercepts), ALREADY Bock-centered (sum_k a_k = sum_k c_k = 0) --
# see the module docstring's GAUGE section. NOT the raw ak0../d0.. scale.
co <- tryCatch(coef(fit, IRTpars = TRUE, simplify = TRUE)$items,
               error = function(e) e)
if (inherits(co, "error")) {
  write_meta("STATUS: ERROR", paste("MESSAGE: coef(IRTpars=TRUE) failed:", conditionMessage(co)),
             paste("WALL_S:", wall_s), paste("METHOD:", method),
             paste("N_ITEMS:", n_items), paste("N_LEARNERS_USED:", n_learners_used))
  quit(status = 1, save = "no")
}

acols <- grep("^a[0-9]+$", colnames(co), value = TRUE)
acols <- acols[order(as.integer(sub("^a", "", acols)))]
ccols <- grep("^c[0-9]+$", colnames(co), value = TRUE)
ccols <- ccols[order(as.integer(sub("^c", "", ccols)))]
if (length(acols) != K || length(ccols) != K) {
  write_meta("STATUS: ERROR",
             paste("MESSAGE: expected", K, "a-cols/c-cols, got",
                   length(acols), "/", length(ccols), "-- coef() column names:",
                   paste(colnames(co), collapse = ",")),
             paste("WALL_S:", wall_s), paste("METHOD:", method),
             paste("N_ITEMS:", n_items), paste("N_LEARNERS_USED:", n_learners_used))
  quit(status = 1, save = "no")
}

out <- data.frame(item = item_ids, co[, acols, drop = FALSE], co[, ccols, drop = FALSE])
write.csv(out, out_csv, row.names = FALSE)

write_meta("STATUS: OK", paste("CONVERGED:", conv), paste("ITER:", iters),
           paste("NCYCLES_CAP:", NCYCLES_CAP), paste("CAP_HIT:", cap_hit),
           paste("WALL_S:", wall_s), paste("METHOD:", method),
           paste("N_ITEMS:", n_items), paste("N_LEARNERS_USED:", n_learners_used),
           paste("K:", K))
