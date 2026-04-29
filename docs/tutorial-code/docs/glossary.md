# Glossary

Short reminders for terms used in the tutorials.

## Tensor Train

A tensor train represents a large array as a chain of smaller tensors called
cores. Instead of storing every array entry directly, it stores compact local
pieces and the links between them.

## QTT

QTT means Quantics Tensor Train. A one-dimensional array with `2^R` points is
reshaped into `R` binary sites, then stored as a tensor train.

Sketch:

```text
grid index i  ->  binary bits b1 b2 ... bR  ->  TT sites
```

## Site

One small dimension in the QTT. For the examples here, each site is usually a
binary site with dimension `2`.

## Bond Dimension

The size of the link between neighboring tensor-train cores. Larger bond
dimensions mean the representation needs more internal room.

## Rank

In these tutorials, rank usually means the largest bond dimension in the QTT.

## Quantics Bit Depth

The number of binary sites, often called `R` or `bits` in the code. A depth of
`R` gives `2^R` grid points.

## Grid Point

One sample location where the target function can be evaluated.

## Interpolation Tolerance

The target accuracy used by the quantics cross interpolation. Smaller tolerance
usually asks for a more accurate QTT and can increase bond dimensions or runtime.

## DiscretizedGrid

The Tensor4all object that maps integer grid indices to physical coordinates.
Use it when the function lives on an interval such as `[-1, 2]` instead of only
on integer indices.

## 1-Based QTT Indexing

Several Tensor4all QTT APIs use 1-based grid indices. The first grid point is
index `1`, not index `0`.

## Fourier Operator / MPO

The quantics Fourier operator is a tensor-network representation of a discrete
Fourier transform. MPO means matrix product operator, the operator analogue of a
tensor train.
