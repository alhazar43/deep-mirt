"""
deep_irt.nrmfmt -- Binary-vs-Nominal cross-format transfer experiment.

The binary/nominal analogue of deep_irt/jointfmt (which did GPCM-vs-BT).
Tests whether ONE shared item embedding, read by a 2PL (right/wrong) head AND
a Bock NRM (chosen-option) head, places items on ONE latent difficulty scale --
so a binary-only item lands correctly on the NRM-derived scale and vice versa.

This is the data-rich format pair: multiple-choice option choices are abundant
in real data (EdNet, Eedi), and the chosen-distractor pattern carries signal
beyond right/wrong that a binary 2PL head cannot see.
"""
