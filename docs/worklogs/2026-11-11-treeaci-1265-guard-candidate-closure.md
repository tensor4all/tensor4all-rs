# Guard candidate abstraction (#1265)

This issue is tracked under #854 and relates to #794 and #784. The branch is
based on current main `66841f68`, independently of the strong-pivot-hint branch.

## Fixture-first scope

Following shinaoka's [review comment](https://github.com/tensor4all/tensor4all-rs/issues/1265#issuecomment-6792048853),
the first commit contains only small characterization fixtures. It does not
modify candidate activation, padding, numerical selection, or stopping rules.
The tests explicitly demonstrate a current defect; the abstraction fix will
turn the same observed failures into required-closure assertions.

A four-node chain reproduces false saturation: a child has two physical slots
but only one distinct candidate after paired padding. A subsequent parent
candidate references an unavailable child. A five-node chain with a valid
rank-two initial guess and two identity sweeps reproduces true saturation: the required child already
has two distinct selected samples at the configured cap, while its parent has
remaining headroom. Both fixtures cover the reflected cut orientation, preserve
selected pivot pairs and represented output values, and respect configured and
algebraic bond dimensions. No slow R10 workload or numerical tolerance change
is required to reproduce the abstraction defect.

## Abstraction work to follow

Useful candidate membership must be represented separately from output-axis
padding. Distinct required children either receive representable candidate
capacity or remain unresolved Guard evidence; a parent cannot be published as
usable when its required child is missing. Selected prefixes, paired output
dimensions, caps and rollback remain the invariants. Reset's required
center-directed nesting must not be confused with full outward nesting.
#784/#854 rank-cycling policies and stopping changes stay outside this diff.

The earlier padding-slot prototype is excluded from this change and is not
promoted as a fix. Its experimental records are retained separately. This
fixture commit changes no production algorithm.
