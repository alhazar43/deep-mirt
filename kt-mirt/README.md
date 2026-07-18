# kt-mirt

Sideline package of the `deep-mirt` PhD thesis, exploring multi-KC
(multiple knowledge component) knowledge tracing with IRT-flavored
readouts. Standalone from the frozen `kt-irt` submodule, which stays
dedicated to its own paper.

The model core (encoders, decoders, prediction losses, anchored item-bank
extension) was vendored 2026-07-17 from `kt-irt` @
`df3aee1fcfaff20f8ba7784c59853ec4ab528696`. `kt_mirt` has no runtime
dependency on the `deep_irt` package; only import paths were rewritten
(`deep_irt.core.*` -> `kt_mirt.core.*`).

Planning documents (design plan, decision ledger, working notes) live in
`_planning/`. See `_planning/vendor_report.md` for the full vendoring
record.
