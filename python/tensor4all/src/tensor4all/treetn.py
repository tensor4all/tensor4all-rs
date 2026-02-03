"""TreeTensorNetwork (TTN) - Tree tensor network with MPS/MPO support.

This module provides a Python interface to the tensor4all-treetn
Rust library via C API.

Examples
--------
>>> from tensor4all import Index, Tensor, TreeTensorNetwork
>>> import numpy as np
>>>
>>> # Create indices
>>> s0 = Index(2)
>>> l01 = Index(3)
>>> s1 = Index(2)
>>>
>>> # Create tensors for a 2-site MPS
>>> t0 = Tensor([s0, l01], np.ones((2, 3)))
>>> t1 = Tensor([l01, s1], np.ones((3, 2)))
>>> mps = TreeTensorNetwork([t0, t1])
>>>
>>> print(mps.num_vertices)  # 2
>>> print(mps.bond_dims)     # [3]
>>> print(mps.maxbonddim)    # 3
"""

from __future__ import annotations

from typing import Optional, Sequence

from ._ffi import ffi
from ._capi import get_lib, check_status, T4A_INVALID_ARGUMENT
from .index import Index
from .tensor import Tensor


# Canonical form constants (matching Rust enum)
CANONICAL_UNITARY = 0
CANONICAL_LU = 1
CANONICAL_CI = 2

# Contract method constants
CONTRACT_ZIPUP = 0
CONTRACT_FIT = 1
CONTRACT_NAIVE = 2

_CONTRACT_METHODS = {
    "zipup": CONTRACT_ZIPUP,
    "fit": CONTRACT_FIT,
    "naive": CONTRACT_NAIVE,
}


class TreeTensorNetwork:
    """Tree tensor network (TTN) supporting MPS, MPO, and general tree topologies.

    Wraps the Rust TreeTN type. Node names are 0-indexed integers internally
    (matching the C API). Users pass 0-indexed vertex names.

    Parameters
    ----------
    tensors : list of Tensor
        Tensors forming the tree tensor network. Connectivity is determined
        by shared index IDs (einsum rule).
    node_names : list, optional
        Custom node names. If None, nodes are named 0, 1, ..., n-1.
        Currently only integer node names are supported via the C API.
    """

    __slots__ = ("_handle", "_node_names", "_node_map")

    def __init__(self, tensors: Sequence[Tensor], node_names: Optional[list] = None):
        """Create a TreeTensorNetwork from a list of Tensor objects."""
        lib = get_lib()
        n = len(tensors)
        if n == 0:
            raise ValueError("Cannot create TreeTensorNetwork from empty tensor list")

        if node_names is None:
            self._node_names = list(range(n))
        else:
            if len(node_names) != n:
                raise ValueError(
                    f"node_names length ({len(node_names)}) must match "
                    f"tensors length ({n})"
                )
            self._node_names = list(node_names)

        self._node_map = {name: i for i, name in enumerate(self._node_names)}

        tensor_ptrs = ffi.new("t4a_tensor*[]", [t._ptr for t in tensors])
        out = ffi.new("t4a_treetn**")
        check_status(lib.t4a_treetn_new(tensor_ptrs, n, out))
        self._handle = out[0]

    @classmethod
    def _from_handle(cls, handle, node_names: list) -> TreeTensorNetwork:
        """Create a TreeTensorNetwork from a raw C handle (internal use)."""
        if handle == ffi.NULL:
            raise ValueError("Cannot create TreeTensorNetwork from null handle")
        obj = object.__new__(cls)
        obj._handle = handle
        obj._node_names = list(node_names)
        obj._node_map = {name: i for i, name in enumerate(node_names)}
        return obj

    def __del__(self):
        if hasattr(self, "_handle") and self._handle != ffi.NULL:
            get_lib().t4a_treetn_release(self._handle)
            self._handle = ffi.NULL

    def _to_c_vertex(self, v):
        """Convert a node name to a 0-based C index."""
        if v not in self._node_map:
            raise KeyError(f"Vertex {v} not found in TTN")
        return self._node_map[v]

    def _from_c_vertex(self, idx: int):
        """Convert a 0-based C index to a node name."""
        return self._node_names[idx]

    # ========================================================================
    # Accessors
    # ========================================================================

    @property
    def num_vertices(self) -> int:
        """Number of vertices (nodes) in the TTN."""
        out = ffi.new("size_t*")
        check_status(get_lib().t4a_treetn_num_vertices(self._handle, out))
        return out[0]

    @property
    def num_edges(self) -> int:
        """Number of edges (bonds) in the TTN."""
        out = ffi.new("size_t*")
        check_status(get_lib().t4a_treetn_num_edges(self._handle, out))
        return out[0]

    def __len__(self) -> int:
        """Number of vertices."""
        return self.num_vertices

    @property
    def vertices(self) -> list:
        """List of vertex names."""
        return list(self._node_names)

    def __getitem__(self, v) -> Tensor:
        """Get the tensor at vertex v."""
        c_v = self._to_c_vertex(v)
        out = ffi.new("t4a_tensor**")
        check_status(get_lib().t4a_treetn_tensor(self._handle, c_v, out))
        return Tensor._from_ptr(out[0])

    def __setitem__(self, v, tensor: Tensor):
        """Set the tensor at vertex v."""
        c_v = self._to_c_vertex(v)
        check_status(
            get_lib().t4a_treetn_set_tensor(self._handle, c_v, tensor._ptr)
        )

    def neighbors(self, v) -> list:
        """Get the neighbors of vertex v."""
        c_v = self._to_c_vertex(v)
        n = self.num_vertices
        buf = ffi.new("size_t[]", n)
        n_out = ffi.new("size_t*")
        check_status(
            get_lib().t4a_treetn_neighbors(self._handle, c_v, buf, n, n_out)
        )
        return [self._from_c_vertex(buf[i]) for i in range(n_out[0])]

    def siteinds(self, v) -> list[Index]:
        """Get the site (physical) indices at vertex v."""
        c_v = self._to_c_vertex(v)
        buf_size = 16
        idx_buf = ffi.new("t4a_index*[]", buf_size)
        n_out = ffi.new("size_t*")
        status = get_lib().t4a_treetn_siteinds(
            self._handle, c_v, idx_buf, buf_size, n_out
        )
        if status == -5:  # T4A_BUFFER_TOO_SMALL
            buf_size = n_out[0]
            idx_buf = ffi.new("t4a_index*[]", buf_size)
            status = get_lib().t4a_treetn_siteinds(
                self._handle, c_v, idx_buf, buf_size, n_out
            )
        check_status(status)
        return [Index._from_ptr(idx_buf[i]) for i in range(n_out[0])]

    def linkind(self, *args) -> Index:
        """Get a link (bond) index.

        linkind(v1, v2) - between vertices v1 and v2
        linkind(i) - between MPS sites i and i+1 (0-indexed)
        """
        if len(args) == 2:
            v1, v2 = args
            c_v1 = self._to_c_vertex(v1)
            c_v2 = self._to_c_vertex(v2)
            out = ffi.new("t4a_index**")
            check_status(
                get_lib().t4a_treetn_linkind(self._handle, c_v1, c_v2, out)
            )
            return Index._from_ptr(out[0])
        elif len(args) == 1:
            i = args[0]
            out = ffi.new("t4a_index**")
            check_status(
                get_lib().t4a_treetn_linkind_at(self._handle, i, out)
            )
            return Index._from_ptr(out[0])
        else:
            raise TypeError("linkind() takes 1 or 2 arguments")

    def linkdim(self, *args) -> int:
        """Get a bond dimension.

        linkdim(v1, v2) - between vertices v1 and v2
        linkdim(i) - between MPS sites i and i+1 (0-indexed)
        """
        if len(args) == 2:
            v1, v2 = args
            c_v1 = self._to_c_vertex(v1)
            c_v2 = self._to_c_vertex(v2)
            out = ffi.new("size_t*")
            check_status(
                get_lib().t4a_treetn_bond_dim(self._handle, c_v1, c_v2, out)
            )
            return out[0]
        elif len(args) == 1:
            i = args[0]
            out = ffi.new("size_t*")
            check_status(
                get_lib().t4a_treetn_bond_dim_at(self._handle, i, out)
            )
            return out[0]
        else:
            raise TypeError("linkdim() takes 1 or 2 arguments")

    @property
    def bond_dims(self) -> list[int]:
        """Bond dimensions for MPS-like TTN (length = num_vertices - 1)."""
        n = self.num_vertices
        if n <= 1:
            return []
        out = ffi.new("size_t[]", n - 1)
        check_status(get_lib().t4a_treetn_bond_dims(self._handle, out, n - 1))
        return [out[i] for i in range(n - 1)]

    @property
    def maxbonddim(self) -> int:
        """Maximum bond dimension across all links."""
        out = ffi.new("size_t*")
        check_status(get_lib().t4a_treetn_maxbonddim(self._handle, out))
        return out[0]

    # ========================================================================
    # Orthogonalization
    # ========================================================================

    def orthogonalize(self, v, *, form: str = "unitary") -> TreeTensorNetwork:
        """Orthogonalize the TTN in-place to vertex v.

        Parameters
        ----------
        v : vertex name
            Target vertex for orthogonality center.
        form : str
            Canonical form: "unitary" (default), "lu", or "ci".

        Returns
        -------
        self
        """
        form_map = {
            "unitary": CANONICAL_UNITARY,
            "lu": CANONICAL_LU,
            "ci": CANONICAL_CI,
        }
        form_int = form_map.get(form.lower())
        if form_int is None:
            raise ValueError(
                f"Unknown canonical form: {form}. Use 'unitary', 'lu', or 'ci'"
            )
        c_v = self._to_c_vertex(v)
        check_status(
            get_lib().t4a_treetn_orthogonalize_with(self._handle, c_v, form_int)
        )
        return self

    @property
    def canonical_form(self) -> Optional[int]:
        """Get the canonical form (0=Unitary, 1=LU, 2=CI).

        Returns None if not set.
        """
        out = ffi.new("int*")
        status = get_lib().t4a_treetn_canonical_form(self._handle, out)
        if status == T4A_INVALID_ARGUMENT:
            return None
        check_status(status)
        return out[0]

    # ========================================================================
    # Operations
    # ========================================================================

    def truncate(
        self,
        *,
        rtol: float = 0.0,
        cutoff: float = 0.0,
        maxdim: int = 0,
    ) -> TreeTensorNetwork:
        """Truncate bond dimensions in-place.

        Parameters
        ----------
        rtol : float
            Relative tolerance (0.0 = not set).
        cutoff : float
            ITensorMPS.jl cutoff (0.0 = not set). Converted to
            rtol = sqrt(cutoff).
        maxdim : int
            Maximum bond dimension (0 = no limit).

        Returns
        -------
        self
        """
        check_status(
            get_lib().t4a_treetn_truncate(self._handle, rtol, cutoff, maxdim)
        )
        return self

    def norm(self) -> float:
        """Compute the norm of the TTN."""
        out = ffi.new("double*")
        check_status(get_lib().t4a_treetn_norm(self._handle, out))
        return out[0]

    def to_dense(self) -> Tensor:
        """Convert to a dense tensor by contracting all link indices.

        Returns
        -------
        Tensor
            The dense tensor with only site indices.
        """
        out = ffi.new("t4a_tensor**")
        check_status(get_lib().t4a_treetn_to_dense(self._handle, out))
        return Tensor._from_ptr(out[0])

    def __add__(self, other: TreeTensorNetwork) -> TreeTensorNetwork:
        """Add two TTNs using direct-sum construction."""
        out = ffi.new("t4a_treetn**")
        check_status(
            get_lib().t4a_treetn_add(self._handle, other._handle, out)
        )
        return TreeTensorNetwork._from_handle(out[0], list(self._node_names))

    def copy(self) -> TreeTensorNetwork:
        """Create a deep copy."""
        ptr = get_lib().t4a_treetn_clone(self._handle)
        if ptr == ffi.NULL:
            raise RuntimeError("Failed to clone TreeTensorNetwork")
        return TreeTensorNetwork._from_handle(ptr, list(self._node_names))

    def __repr__(self) -> str:
        n = self.num_vertices
        e = self.num_edges
        return f"TreeTensorNetwork(nv={n}, ne={e})"


# Type aliases
MPS = TreeTensorNetwork
MPO = TreeTensorNetwork


# ============================================================================
# Module-level functions
# ============================================================================


def inner(a: TreeTensorNetwork, b: TreeTensorNetwork) -> complex:
    """Compute the inner product <a|b>.

    Parameters
    ----------
    a : TreeTensorNetwork
        First TTN.
    b : TreeTensorNetwork
        Second TTN.

    Returns
    -------
    complex
        The inner product.
    """
    out_re = ffi.new("double*")
    out_im = ffi.new("double*")
    check_status(
        get_lib().t4a_treetn_inner(a._handle, b._handle, out_re, out_im)
    )
    return complex(out_re[0], out_im[0])


def lognorm(ttn: TreeTensorNetwork) -> float:
    """Compute the log-norm of the TTN.

    Parameters
    ----------
    ttn : TreeTensorNetwork
        The tree tensor network.

    Returns
    -------
    float
        The log-norm.
    """
    out = ffi.new("double*")
    check_status(get_lib().t4a_treetn_lognorm(ttn._handle, out))
    return out[0]


def contract(
    a: TreeTensorNetwork,
    b: TreeTensorNetwork,
    *,
    method: str = "zipup",
    maxdim: int = 0,
    rtol: float = 0.0,
    cutoff: float = 0.0,
) -> TreeTensorNetwork:
    """Contract two tree tensor networks.

    Parameters
    ----------
    a : TreeTensorNetwork
        First TTN.
    b : TreeTensorNetwork
        Second TTN (must share site indices with a).
    method : str
        Contraction method: "zipup" (default), "fit", or "naive".
    maxdim : int
        Maximum bond dimension (0 = no limit).
    rtol : float
        Relative tolerance (0.0 = not set).
    cutoff : float
        ITensorMPS.jl cutoff (0.0 = not set).

    Returns
    -------
    TreeTensorNetwork
        The contraction result.
    """
    method_int = _CONTRACT_METHODS.get(method.lower())
    if method_int is None:
        raise ValueError(
            f"Unknown contract method: {method}. Use 'zipup', 'fit', or 'naive'"
        )
    out = ffi.new("t4a_treetn**")
    check_status(
        get_lib().t4a_treetn_contract(
            a._handle, b._handle, method_int, rtol, cutoff, maxdim, out
        )
    )
    return TreeTensorNetwork._from_handle(out[0], list(a._node_names))


def linsolve(
    operator: TreeTensorNetwork,
    rhs: TreeTensorNetwork,
    init: TreeTensorNetwork,
    *,
    a0: float = 0.0,
    a1: float = 1.0,
    nsweeps: int = 10,
    rtol: float = 0.0,
    cutoff: float = 0.0,
    maxdim: int = 0,
) -> TreeTensorNetwork:
    """Solve (a0 + a1 * A) * x = b for x.

    Uses DMRG-like sweeps.

    Parameters
    ----------
    operator : TreeTensorNetwork
        The operator A (MPO/TTN operator).
    rhs : TreeTensorNetwork
        The right-hand side b (MPS/TTN state).
    init : TreeTensorNetwork
        Initial guess for x (MPS/TTN state). Cloned internally.
    a0 : float
        Coefficient a0 in (a0 + a1*A)*x = b.
    a1 : float
        Coefficient a1 in (a0 + a1*A)*x = b.
    nsweeps : int
        Number of full sweeps (default 10).
    rtol : float
        Relative tolerance (0.0 = not set).
    cutoff : float
        ITensorMPS.jl cutoff (0.0 = not set).
    maxdim : int
        Maximum bond dimension (0 = no limit).

    Returns
    -------
    TreeTensorNetwork
        The solution x.
    """
    out = ffi.new("t4a_treetn**")
    check_status(
        get_lib().t4a_treetn_linsolve(
            operator._handle, rhs._handle, init._handle,
            a0, a1, nsweeps, rtol, cutoff, maxdim, out
        )
    )
    return TreeTensorNetwork._from_handle(out[0], list(init._node_names))
