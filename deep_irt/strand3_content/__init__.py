"""strand3_content -- the content channel and its cold-start payoff test.

The content channel makes an item's representation a function of its TEXT
(a frozen sentence-transformer embedding projected into the item-embedding
space) rather than a free per-item ID row.  Its load-bearing property is
COLD-START: a new item's representation -- and therefore its difficulty -- is
computable from text alone, with no responses and no trained ID row.  This is
the substitute for an ID-only model's task-specific item-bank trap and the path
to a general (foundation-model) ability scale.

This package is additive and self-contained: it composes the frozen deep_irt
core (encoder + GPCM decoder) with a content-channel item representation and
never modifies the core.  The core's default ID-embedding path is untouched and
bit-for-bit identical.
"""
