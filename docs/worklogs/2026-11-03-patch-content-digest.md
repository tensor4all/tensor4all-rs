# Content digest for patch ordering (#1086)

Follow-up to #1087 / #1084. #1087 removed `DynIndex` allocation ids from the
patch and contribution order, leaving three key components: the patch index
signature, the patch's own projector coordinates in logical labels, and (then
missing) a content tiebreak. This record covers the third component.

## Decisions

- `IdxTensor::content_digest() -> Option<u64>` is the non-materializing content
  accessor. `Some` covers a `Materialized` dense/diagonal/structured payload, a
  host-resident `Eager` payload and a host-resident `Compact` payload; it hashes
  the storage kind, dtype, payload dims, axis classes, every payload value's
  native bit pattern, and the accumulated gradient of a tracked payload. `None`
  covers a deferred payload, a backend-resident eager payload, a non-contiguous
  compact payload and a scalar kind with no compact representation.
- The accessor borrows. `to_storage`/`storage` build a compact snapshot
  (`IdxTensorStorage::materialize`), and `with_dense_slice` /
  `with_dense_host_slice` densify a diagonal or structured payload, so neither
  can be used from a comparator. `payload_view` on compact storage and
  `TensorRead::as_slice` on eager storage both answer in place, and the second
  reports `HostAccess` instead of downloading backend storage.
- The word mix adds the word before multiplying and then folds the high bits
  down. A plain FNV-1a step (`state = (state ^ word) * prime`) is linear over
  GF(2), so `[1.0, 2.0]` and `[-1.0, -2.0]` fold to the same state: paired sign
  flips cancel. The regression test pins that pair.
- The digest includes the accumulated gradient of a tracked payload, because a
  primal alone does not show it. It does **not** separate `x` and `2x` at
  `x = 0`: their primals are bitwise equal and nothing readable distinguishes the
  two graphs before a gradient is accumulated, so that pair stays a documented
  tie. The accessible AD state here is a reverse-mode cotangent, not a forward
  tangent, so the digest can only be as strong as that. A gradient accumulated
  through `SubDomainTreeTN::data()` after construction makes the stored key
  stale, which affects ordering reproducibility only.
- The third key component is per patch: each node tensor's digest folded in
  sorted node-name order, preceded by that node's physical leg roles (which axis
  is a site index, which is an internal bond, with the bond's canonical labels
  and dimension, never its id). Without the leg roles, two networks over the same
  site indices whose node tensors carry the same buffer under a different leg
  order would share a key while being different networks. The key is built by one
  helper and refreshed after `SubDomainTreeTN::truncate`, which rewrites payloads
  in place.
- Rejected: computing the digest at sort time (the comparator would be
  `O(payload)` per comparison and cannot cache without interior mutability);
  recording storage kind / dtype / dims as a separate component for the `None`
  case (a second partial order for a smaller class than it separates);
  returning an error instead of ordering an unfingerprintable payload.
- `StructuredStorage`'s contiguous-view accessors built a temporary stride
  vector through `col_major_strides` on every call, so borrowing a payload
  allocated. They now compare against the expected strides directly; this is
  what lets the digest allocate nothing for compact storage.

## Verification conclusions and constraints

- The digest allocates nothing for compact storage: a dense `Materialized`
  payload, and a diagonal payload whose densification would allocate `8 n^2`,
  hash with zero bytes allocated. An eager payload is read in place; ranking
  above 8 copies only the axis layout, measured at 848 bytes for rank 9, 1088 for
  rank 12 and 1984 for rank 20, i.e. `O(rank)` and not `O(payload)`.
- A diagonal payload is never densified, a strided compact payload reports
  `None` without copying, and a deferred payload reports `None` even though its
  pre-failure source would report `Some`.
- Equal primals with different accumulated gradients get different digests;
  equal primals with different graphs tie, as documented.
- Patch keys separate two patches whose index signature and projector
  coordinates are equal, and the order does not depend on construction order. A
  swapped leg order over identical buffers changes the key. Truncation refreshes
  the cached component.
- Cost, measured on one host (Apple M5 Max, 18 CPUs, rustc 1.96.0, release
  profile, `RAYON/OPENBLAS/OMP/MKL/VECLIB` thread variables all 1,
  `--test-threads=1`, one timing loop per shape, not a benchmark): the digest
  runs at about 5.6 GB/s (2 M `f64` values in 2.9 ms). `SubDomainTreeTN::new`
  grows by about 3 percent at 32 payload values, 27 percent at 8 k values and 43
  percent at 2 M values. This is one linear pass over resident bytes, paid once
  per patch construction.
- Unverified: the `None` path for a backend-resident payload has no CPU test (a
  CUDA build is needed); the CPU-reachable `None` paths are covered.
