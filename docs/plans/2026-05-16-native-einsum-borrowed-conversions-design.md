# Native Einsum Borrowed Conversions Design

## Context

Issue #441 identifies avoidable memory growth in dense native einsum paths. The
public borrowed bridge `einsum_native_tensors` currently clones every native
operand and forwards to the owned bridge. For CPU host buffers, `NativeTensor`
cloning duplicates the underlying dense storage.

The owned bridge already promotes only operands whose dtype differs from the
common target dtype. The borrowed bridge should preserve that behavior without
deep-cloning operands that can be reused by reference.

## Design

`einsum_native_tensors` will compute the common target dtype, validate each
operand rank against its label list, and build einsum subscripts as before. It
will then allocate converted temporary tensors only for operands whose dtype
differs from the target dtype. Operands already at the target dtype will remain
borrowed from the caller.

The bridge will pass a `Vec<&NativeTensor>` to tenferro's borrowed
`eager_einsum`. References will point either to the original caller-owned tensor
or to the local converted temporary. The converted temporaries live until
`eager_einsum` returns, so the borrowed input slice remains valid.

## Alternatives Considered

- Keep using the owned bridge for mixed dtype inputs. This keeps code simple but
  deep-clones target-dtype operands unnecessarily, which is the memory issue.
- Add a public mixed borrowed/owned tenferro API. This is more general but
  expands the dependency surface before the tensor4all bridge needs it.
- Convert only mismatched operands inside the tensor4all bridge. This solves the
  immediate memory issue while keeping the public API unchanged.

The third option is the chosen design.

## Testing

Tests should cover both native einsum execution paths:

- same-dtype borrowed operands use the borrowed path with no conversions;
- mixed-dtype borrowed operands use the borrowed-with-conversions path and still
  promote to the common dtype;
- existing validation and numerical results remain unchanged.

The profile path enum is test-only observable within the module, so tests can
assert which route was used without adding public API.
