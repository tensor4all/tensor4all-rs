# Reporting a `max_bond_dim` truncation in adaptive patching (#1246)

## Decisions

- A report, not an opt-in error. An error would break existing callers of
  `add_with_patching` and `contract_adaptive`, and the issue allowed either. The
  two functions keep their signatures and return values; the reporting path is
  the `_with_report` sibling pair, so a caller opts in by naming it.
- One report type for both entry points: `PatchingReport`, one `PatchCapRecord`
  per returned patch with the patch's bond dimensions and a `cap_truncated`
  flag. Both entry points end in the same closing `truncate_adaptive` sweep, so
  one record type covers both.
- The flag comes from the cutoff-only ranks, not from the stored dimensions of
  the patch that entered the sweep. Reading the stored dimensions reports a cap
  truncation for a patch whose stored bond carries a zero singular value (the
  exact rank is already at the cap), and it conflates a bond the cutoff shrank
  with a bond the cap set. The record is therefore built where the closing sweep
  already knows the threshold: a patch that keeps a bond exactly at the cap is
  truncated once more with the same threshold and no cap, and the flag says
  whether that retained a larger bond dimension.
- Cost is confined to the report path and to patches that can be capped: a patch
  below the cap cannot have been capped and is not re-truncated, and
  `truncate_adaptive`, `add_with_patching` and `contract_adaptive` never collect
  records.
- No discarded weight. `TreeTN::truncate` returns no singular values, so the
  weight the cap discarded would need a separate decomposition of the patch. The
  record carries the bond dimensions and the flag instead, and the rustdoc and
  the design doc say why.
- The report is patch-level, from the largest bond dimension on each side, not
  per bond: pairing two truncations edge by edge would need edge identity that
  the truncation does not preserve.

## Verification conclusions and constraints

- The issue's reproducer (four bits over two nodes, `patch_order = [a]`, cap 1,
  `cutoff = 1e-12`, `Norm`) reports both patches as cap-truncated with bond
  dimension 1, while the relative error stays 4.8e-1: the report is the signal,
  and it does not become an error.
- Controls that report nothing, for both entry points: cap 2 with
  `patch_order = [a]` for addition (the exact rank of each patch is 2, error at
  round-off), cap 4 for contraction (the output patches have exact rank 4), and
  cap 1 with every bit in `patch_order` (the split loop keeps splitting, so no
  patch reaches the cap).
- A stored bond dimension above the cap with a zero singular value is not
  reported; that case is pinned by its own test, because the stored dimensions
  are what made an earlier version of the flag wrong.
- The flag cannot distinguish a cap that discarded nonzero weight from a cutoff
  that would have retained exactly the cap at a different bond. Neither can the
  bond dimensions alone; a caller that needs the weight has to decompose the
  patch itself.
- Contraction records describe the contracted patch, which the contraction
  options may already have truncated before the closing sweep.

Design text: `docs/design/partitioned-treetn.md`, "Truncation convention:
cross-patch discarded-weight `cutoff`".
