# Three-Mode AD Design for Tensor Networks

## Goal

Provide a single conceptual API for tensor-network algorithms that supports:

- `Primal` execution (plain numeric evaluation)
- `Dual` execution (forward mode / JVP)
- `Tracked` execution (reverse mode / VJP, plus HVP support)

The design should preserve Julia-like AD ergonomics while keeping non-differentiable decisions explicit.

## Scope

This document defines:

- Core value and scalar model for three AD modes
- Trait boundaries between primal kernels and AD rules
- Semantics of `detach` (`stop_gradient`)
- HVP composition from forward and reverse information
- Policy for non-smooth operations (pivot selection, rank truncation)

It does not prescribe migration steps.

## Design Principles

1. `AnyScalar` remains, but must preserve AD mode metadata.
2. Primal numeric kernels and AD logic are separated.
3. AD behavior is rule-driven (`rrule`, `frule`, `hvp`) per operation.
4. Tensor and scalar mode promotion is explicit and centralized.
5. Non-smooth operations are not silently differentiated.

## Core Value Model

### Base scalar domain

```rust
pub enum BaseScalar {
    F64(f64),
    C64(num_complex::Complex64),
}
```

### Mode-preserving scalar

```rust
pub enum AnyScalar {
    Primal(BaseScalar),
    Dual {
        primal: BaseScalar,
        tangent: BaseScalar,
    },
    Tracked {
        primal: BaseScalar,
        node: NodeId,
        tangent: Option<BaseScalar>,
    },
}
```

Rationale:

- Tensor-network APIs (`scale`, `axpby`, `inner_product`) exchange scalar values.
- If scalar results are downgraded to primal, AD paths are broken.
- Therefore scalar values must carry the same mode semantics as tensors.

### Tensor wrappers

```rust
pub struct Primal<T> {
    pub value: T,
}

pub struct Dual<T> {
    pub primal: T,
    pub tangent: T,
}

pub struct Tracked<T> {
    pub primal: T,
    pub node: NodeId,
    pub tape: TapeId,
    pub tangent: Option<T>, // optional direction for HVP workflows
}
```

## Trait Boundaries

### Primal kernel trait

Primal kernels are pure numeric operations. No tape logic, no AD branching.

```rust
pub trait TensorKernel: Clone {
    type Index: IndexLike;

    fn contract(tensors: &[&Self], allowed: AllowedPairs<'_>) -> Result<Self>;
    fn factorize(
        &self,
        left_inds: &[Self::Index],
        options: &FactorizeOptions,
    ) -> Result<FactorizeResult<Self>>;
    fn axpby(&self, a: BaseScalar, other: &Self, b: BaseScalar) -> Result<Self>;
    fn scale(&self, a: BaseScalar) -> Result<Self>;
    fn inner_product(&self, other: &Self) -> Result<BaseScalar>;
}
```

### Rule trait for AD

Each operation is defined by evaluation and AD rules.

```rust
pub trait OpRule<V: Differentiable> {
    fn eval(&self, inputs: &[&V]) -> Result<V>;
    fn rrule(
        &self,
        inputs: &[&V],
        out: &V,
        cotangent: &V::Tangent,
    ) -> AdResult<Vec<V::Tangent>>;
    fn frule(
        &self,
        inputs: &[&V],
        tangents: &[Option<&V::Tangent>],
    ) -> AdResult<V::Tangent>;
    fn hvp(
        &self,
        inputs: &[&V],
        cotangent: &V::Tangent,
        cotangent_tangent: Option<&V::Tangent>,
        input_tangents: &[Option<&V::Tangent>],
    ) -> AdResult<Vec<(V::Tangent, V::Tangent)>>;
}
```

This makes AD behavior explicit and testable per operation.

## Execution Semantics by Mode

- `Primal<T>`: `eval` only.
- `Dual<T>`: `eval + frule` (JVP propagation).
- `Tracked<T>`: `eval + rrule` recording on tape (VJP).
- `HVP`: either
  - operation-local `hvp` rule, or
  - dualized pullback composition (`rrule` linearized by forward tangents).

## Mode and DType Promotion

Promotion is centralized, not scattered across operators.

### DType promotion

- `F64 + F64 -> F64`
- any combination including `C64 -> C64`

### AD mode promotion

- `Primal < Dual < Tracked`
- Mixed-mode ops promote to the maximum mode.

Examples:

- `Primal tensor + Dual scalar -> Dual result`
- `Dual tensor + Tracked scalar -> Tracked result`

## Detach / Stop-Gradient

`detach` is defined as a first-class operator: `stop_gradient(x)`.

Mathematical semantics:

- primal value: identity (`y = x`)
- reverse derivative: zero
- forward tangent: zero
- hvp contribution: zero

Mode behavior:

- `Primal`: no-op
- `Dual`: keep primal, set tangent to zero / `None`
- `Tracked`: keep primal, drop tape connection (`requires_grad = false`)

## Non-Smooth Operations Policy

Operations with discrete decisions (pivoting, rank selection, truncation) are not smoothly differentiable in general.

Define explicit policy:

```rust
pub enum DiffPolicy {
    Strict,        // return ModeNotSupported
    StopGradient,  // allow eval, block derivatives
}
```

Recommended default for tensor-network workflows: `StopGradient`.

This matches practical usage where pivot/rank decisions are treated as non-differentiable control flow.

## HVP Composition

Let graph nodes satisfy:

\[
x_k = \phi_k(x_{p_1(k)}, \dots, x_{p_m(k)}), \quad f = x_L \in \mathbb{R}
\]

Forward pass (Dual/JVP):

\[
\dot{x}_k = \sum_j \frac{\partial \phi_k}{\partial x_{p_j}} \dot{x}_{p_j}
\]

Reverse seeds:

\[
\bar{x}_L = 1,\quad \dot{\bar{x}}_L = 0
\]

Reverse recursion:

\[
\bar{x}_{p_j} \mathrel{+}= J_{k,j}^\top \bar{x}_k
\]

HVP recursion (linearized reverse):

\[
\dot{\bar{x}}_{p_j} \mathrel{+}= J_{k,j}^\top \dot{\bar{x}}_k
+ \dot{J}_{k,j}^\top \bar{x}_k
\]

with
\[
\dot{J}_{k,j} = D J_{k,j}[\dot{x}]
\]

At leaves:

- gradient: \(\nabla f = \bar{\theta}\)
- Hessian-vector product: \(H v = \dot{\bar{\theta}}\)

## Complexity

For one direction \(v\), HVP should remain in the same asymptotic class as one model pass:

- gradient: \(O(C)\)
- one HVP: \(O(C)\)
- \(K\) directions: \(O(KC)\)

where \(C\) is the forward computational cost for the network.

## Testing Requirements

1. Mode invariants:
   - AD metadata is preserved through scalar/tensor operations.
2. Rule consistency:
   - `frule` vs finite-difference directional derivative.
   - `rrule` vs finite-difference VJP checks.
3. HVP checks:
   - `hvp` vs finite-difference of gradient.
4. Detach semantics:
   - primal equality holds, derivatives are zero.
5. Policy checks:
   - non-smooth ops respect `DiffPolicy` exactly.

## Summary

The key architectural decision is:

- keep `AnyScalar`, but make it AD-mode preserving;
- keep primal kernels simple;
- express AD via operation rules;
- treat `detach` and non-smooth behavior explicitly.

This yields a consistent three-mode system with Julia-like AD behavior for tensor-network code while remaining honest about non-differentiable algorithmic branches.
