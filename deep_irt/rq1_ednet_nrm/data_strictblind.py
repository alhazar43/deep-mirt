"""data_strictblind.py -- strict format-blind encoder history for RQ1.

Closes the one honest caveat on the binary-vs-NRM result
(deep_irt/rq1_ednet_nrm): the encoder VALUE stream in the original run fed
the 4-way NOMINAL option (a/b/c/d) at EVERY position, including BINARY-ONLY
items.  So although a binary-only item's shared difficulty d_i is shaped only
by the 2PL head (its q_embed gets only binary-head gradient), the *chosen
option* a binary-only item still entered the encoder's theta/history pathway.
A strict reviewer could say the history wasn't truly format-blind.

This module builds a STRICT-FORMAT-BLIND value stream so the encoder history
respects the same format mask the heads do:

    BINARY-ONLY positions : value token carries ONLY the binary label
        (correct vs incorrect).  All distractors collapse to one "wrong"
        bucket, so which option (b/c/d) was chosen cannot enter theta/history.
    NRM-ONLY positions    : value token = true nominal option (unchanged).
    BOTH / bridge positions : value token = true nominal option (unchanged).

Implementation note on the "wrong" bucket index
------------------------------------------------
The encoder's value embedding (LearnedEmbedding) maps the response index
through the triangular ordinal kernel ``max(0, 1 - |k - r|/(K-1))``.  Options
were remapped so the CORRECT option is index 0.  A binary-only position is
therefore encoded as a 2-LEVEL signal:

    correct  -> index 0      (kernel peak at 0)
    wrong    -> index K-1    (kernel peak at the far end, K-1)

Index K-1 is the canonical single "wrong" bucket: it is distractor-blind
(every wrong distractor maps to the same token) and maximally distinct from
the correct token under the ordinal kernel.  The binary-only position now
carries exactly one bit -- the binary label -- and nothing about which
distractor was chosen.

Only the ENCODER VALUE STREAM changes.  The NRM training TARGET still uses the
true nominal option (so NRM-head learning on NRM-visible items is untouched),
and the binary TARGET still uses the true binary label.  d_i / head structure
is identical to the original.  ma-irt is FROZEN; this is purely additive.
"""

from __future__ import annotations

from typing import List

import numpy as np

from deep_irt.rq1_ednet_nrm.data import (
    EdNetData,
    load_ednet,
    G_BINARY_ONLY,
    G_BOTH,
    G_NRM_ONLY,
)


def load_ednet_strictblind(**kwargs) -> EdNetData:
    """Load EdNet exactly as :func:`load_ednet`, then add a strict-blind value
    stream to each record.

    All loading / filtering / partition / learner-sampling is IDENTICAL to the
    original :func:`load_ednet` (same kwargs, same seed) so the two runs are
    directly comparable.  The only addition is a per-record ``value`` array:

        value[t] = nominal[t]                  if position t is NRM-visible
                                               (NRM-ONLY or BOTH)
        value[t] = 0 if correct else K-1       if position t is BINARY-ONLY

    The encoder is fed ``value`` instead of ``nominal``; everything else
    (binary target, nominal target, masks) is unchanged.
    """
    data = load_ednet(**kwargs)
    K = data.K
    wrong_bucket = K - 1  # canonical single "wrong" token (far end of kernel)

    for r in data.records:
        questions = r["questions"]                 # (T,) 1-based local ids
        g = data.group[questions - 1]              # (T,) per-position group
        is_binary_only = (g == G_BINARY_ONLY)      # (T,) bool
        binary = r["binary"]                       # (T,) 0/1 correctness
        nominal = r["nominal"]                      # (T,) 0..K-1 chosen option

        # Default: true nominal (NRM-only and both positions keep full signal).
        value = nominal.copy().astype(np.int64)
        # Binary-only positions: collapse to a 2-level distractor-blind token.
        bo_wrong = is_binary_only & (binary == 0)
        bo_right = is_binary_only & (binary == 1)
        value[bo_wrong] = wrong_bucket
        value[bo_right] = 0                          # correct option index
        r["value"] = value

    return data
