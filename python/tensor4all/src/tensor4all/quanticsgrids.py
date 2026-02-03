"""QuanticsGrids - Quantics grid types for coordinate conversions.

This module provides two grid types for quantics tensor train representations:
- ``DiscretizedGrid``: continuous domain with floating-point coordinates
- ``InherentDiscreteGrid``: integer coordinate domain

Both support coordinate conversions between original coordinates, grid indices,
and quantics indices.

Examples
--------
>>> from tensor4all.quanticsgrids import DiscretizedGrid
>>>
>>> # Create a 1D grid with 8 bits (256 grid points) over [0, 1)
>>> grid = DiscretizedGrid(1, 8, [0.0], [1.0])
>>>
>>> # Convert coordinates
>>> quantics = grid.origcoord_to_quantics([0.5])
>>> coord = grid.quantics_to_origcoord(quantics)
"""

from __future__ import annotations

from typing import Sequence, Optional, List

from ._ffi import ffi
from ._capi import get_lib, check_status


def _unfolding_to_int(unfolding: str) -> int:
    """Convert unfolding scheme string to C enum value."""
    if unfolding == "fused":
        return 0
    elif unfolding == "interleaved":
        return 1
    else:
        raise ValueError(f"Unknown unfolding scheme: {unfolding!r}. Use 'fused' or 'interleaved'.")


class DiscretizedGrid:
    """A discretized grid with continuous domain and floating-point coordinates.

    Parameters
    ----------
    ndims : int
        Number of dimensions.
    R : int or Sequence[int]
        Resolution (bits) per dimension. If int, same for all dimensions.
    lower : Sequence[float]
        Lower bounds per dimension.
    upper : Sequence[float]
        Upper bounds per dimension.
    unfolding : str, optional
        Unfolding scheme: "fused" (default) or "interleaved".

    Examples
    --------
    >>> grid = DiscretizedGrid(1, 3, [0.0], [1.0])
    >>> grid.ndims
    1
    >>> grid.rs
    [3]
    >>> grid.grid_step
    [0.125]
    """

    def __init__(
        self,
        ndims: int,
        R,
        lower: Sequence[float],
        upper: Sequence[float],
        unfolding: str = "fused",
    ):
        lib = get_lib()

        if isinstance(R, int):
            rs_list = [R] * ndims
        else:
            rs_list = list(R)
            if len(rs_list) != ndims:
                raise ValueError(f"rs must have length {ndims}, got {len(rs_list)}")

        if len(lower) != ndims:
            raise ValueError(f"lower must have length {ndims}, got {len(lower)}")
        if len(upper) != ndims:
            raise ValueError(f"upper must have length {ndims}, got {len(upper)}")

        rs_c = ffi.new("size_t[]", rs_list)
        lower_c = ffi.new("double[]", list(lower))
        upper_c = ffi.new("double[]", list(upper))
        out = ffi.new("t4a_qgrid_disc**")

        status = lib.t4a_qgrid_disc_new(
            ndims, rs_c, lower_c, upper_c, _unfolding_to_int(unfolding), out
        )
        check_status(status, "DiscretizedGrid.__init__")

        self._ptr = out[0]
        self._ndims = ndims
        self._rs = rs_list

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr != ffi.NULL:
            get_lib().t4a_qgrid_disc_release(self._ptr)
            self._ptr = ffi.NULL

    @property
    def ndims(self) -> int:
        """Number of dimensions."""
        return self._ndims

    @property
    def rs(self) -> List[int]:
        """Resolution (bits) per dimension."""
        return list(self._rs)

    @property
    def local_dimensions(self) -> List[int]:
        """Local dimensions of all tensor sites."""
        max_sites = sum(self._rs) + self._ndims
        out = ffi.new("size_t[]", max_sites)
        n_out = ffi.new("size_t*")
        status = get_lib().t4a_qgrid_disc_local_dims(self._ptr, out, max_sites, n_out)
        check_status(status, "local_dimensions")
        return [out[i] for i in range(n_out[0])]

    @property
    def lower_bound(self) -> List[float]:
        """Lower bounds per dimension."""
        out = ffi.new("double[]", self._ndims)
        status = get_lib().t4a_qgrid_disc_lower_bound(self._ptr, out, self._ndims)
        check_status(status, "lower_bound")
        return [out[i] for i in range(self._ndims)]

    @property
    def upper_bound(self) -> List[float]:
        """Upper bounds per dimension."""
        out = ffi.new("double[]", self._ndims)
        status = get_lib().t4a_qgrid_disc_upper_bound(self._ptr, out, self._ndims)
        check_status(status, "upper_bound")
        return [out[i] for i in range(self._ndims)]

    @property
    def grid_step(self) -> List[float]:
        """Grid spacing per dimension."""
        out = ffi.new("double[]", self._ndims)
        status = get_lib().t4a_qgrid_disc_grid_step(self._ptr, out, self._ndims)
        check_status(status, "grid_step")
        return [out[i] for i in range(self._ndims)]

    def origcoord_to_quantics(self, coord: Sequence[float]) -> List[int]:
        """Convert original (continuous) coordinates to quantics indices.

        Parameters
        ----------
        coord : Sequence[float]
            Original coordinates, one per dimension.

        Returns
        -------
        List[int]
            Quantics indices (1-indexed).
        """
        lib = get_lib()
        coord_c = ffi.new("double[]", list(coord))
        max_sites = sum(self._rs) + self._ndims
        out = ffi.new("int64_t[]", max_sites)
        n_out = ffi.new("size_t*")
        status = lib.t4a_qgrid_disc_origcoord_to_quantics(
            self._ptr, coord_c, len(coord), out, max_sites, n_out
        )
        check_status(status, "origcoord_to_quantics")
        return [out[i] for i in range(n_out[0])]

    def quantics_to_origcoord(self, quantics: Sequence[int]) -> List[float]:
        """Convert quantics indices to original (continuous) coordinates.

        Parameters
        ----------
        quantics : Sequence[int]
            Quantics indices (1-indexed).

        Returns
        -------
        List[float]
            Original coordinates.
        """
        lib = get_lib()
        quantics_c = ffi.new("int64_t[]", list(quantics))
        out = ffi.new("double[]", self._ndims)
        status = lib.t4a_qgrid_disc_quantics_to_origcoord(
            self._ptr, quantics_c, len(quantics), out, self._ndims
        )
        check_status(status, "quantics_to_origcoord")
        return [out[i] for i in range(self._ndims)]

    def origcoord_to_grididx(self, coord: Sequence[float]) -> List[int]:
        """Convert original coordinates to grid indices (1-indexed).

        Parameters
        ----------
        coord : Sequence[float]
            Original coordinates.

        Returns
        -------
        List[int]
            Grid indices (1-indexed).
        """
        lib = get_lib()
        coord_c = ffi.new("double[]", list(coord))
        out = ffi.new("int64_t[]", self._ndims)
        status = lib.t4a_qgrid_disc_origcoord_to_grididx(
            self._ptr, coord_c, len(coord), out, self._ndims
        )
        check_status(status, "origcoord_to_grididx")
        return [out[i] for i in range(self._ndims)]

    def grididx_to_origcoord(self, grididx: Sequence[int]) -> List[float]:
        """Convert grid indices (1-indexed) to original coordinates.

        Parameters
        ----------
        grididx : Sequence[int]
            Grid indices (1-indexed).

        Returns
        -------
        List[float]
            Original coordinates.
        """
        lib = get_lib()
        grididx_c = ffi.new("int64_t[]", list(grididx))
        out = ffi.new("double[]", self._ndims)
        status = lib.t4a_qgrid_disc_grididx_to_origcoord(
            self._ptr, grididx_c, len(grididx), out, self._ndims
        )
        check_status(status, "grididx_to_origcoord")
        return [out[i] for i in range(self._ndims)]

    def grididx_to_quantics(self, grididx: Sequence[int]) -> List[int]:
        """Convert grid indices to quantics indices.

        Parameters
        ----------
        grididx : Sequence[int]
            Grid indices (1-indexed).

        Returns
        -------
        List[int]
            Quantics indices (1-indexed).
        """
        lib = get_lib()
        grididx_c = ffi.new("int64_t[]", list(grididx))
        max_sites = sum(self._rs) + self._ndims
        out = ffi.new("int64_t[]", max_sites)
        n_out = ffi.new("size_t*")
        status = lib.t4a_qgrid_disc_grididx_to_quantics(
            self._ptr, grididx_c, len(grididx), out, max_sites, n_out
        )
        check_status(status, "grididx_to_quantics")
        return [out[i] for i in range(n_out[0])]

    def quantics_to_grididx(self, quantics: Sequence[int]) -> List[int]:
        """Convert quantics indices to grid indices.

        Parameters
        ----------
        quantics : Sequence[int]
            Quantics indices (1-indexed).

        Returns
        -------
        List[int]
            Grid indices (1-indexed).
        """
        lib = get_lib()
        quantics_c = ffi.new("int64_t[]", list(quantics))
        out = ffi.new("int64_t[]", self._ndims)
        n_out = ffi.new("size_t*")
        status = lib.t4a_qgrid_disc_quantics_to_grididx(
            self._ptr, quantics_c, len(quantics), out, self._ndims, n_out
        )
        check_status(status, "quantics_to_grididx")
        return [out[i] for i in range(n_out[0])]

    def __repr__(self) -> str:
        return f"DiscretizedGrid(ndims={self._ndims}, rs={self._rs})"


class InherentDiscreteGrid:
    """A discrete grid for quantics tensor train representations with integer coordinates.

    Parameters
    ----------
    ndims : int
        Number of dimensions.
    R : int or Sequence[int]
        Resolution (bits) per dimension. If int, same for all dimensions.
    origin : Sequence[int] or None, optional
        Origin per dimension (default: all 1s).
    unfolding : str, optional
        Unfolding scheme: "fused" (default) or "interleaved".

    Examples
    --------
    >>> grid = InherentDiscreteGrid(1, 3)
    >>> grid.ndims
    1
    >>> grid.rs
    [3]
    """

    def __init__(
        self,
        ndims: int,
        R,
        origin: Optional[Sequence[int]] = None,
        unfolding: str = "fused",
    ):
        lib = get_lib()

        if isinstance(R, int):
            rs_list = [R] * ndims
        else:
            rs_list = list(R)
            if len(rs_list) != ndims:
                raise ValueError(f"rs must have length {ndims}, got {len(rs_list)}")

        rs_c = ffi.new("size_t[]", rs_list)

        if origin is not None:
            if len(origin) != ndims:
                raise ValueError(f"origin must have length {ndims}, got {len(origin)}")
            origin_c = ffi.new("int64_t[]", list(origin))
        else:
            origin_c = ffi.NULL

        out = ffi.new("t4a_qgrid_int**")
        status = lib.t4a_qgrid_int_new(
            ndims, rs_c, origin_c, _unfolding_to_int(unfolding), out
        )
        check_status(status, "InherentDiscreteGrid.__init__")

        self._ptr = out[0]
        self._ndims = ndims
        self._rs = rs_list

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr != ffi.NULL:
            get_lib().t4a_qgrid_int_release(self._ptr)
            self._ptr = ffi.NULL

    @property
    def ndims(self) -> int:
        """Number of dimensions."""
        return self._ndims

    @property
    def rs(self) -> List[int]:
        """Resolution (bits) per dimension."""
        return list(self._rs)

    @property
    def local_dimensions(self) -> List[int]:
        """Local dimensions of all tensor sites."""
        max_sites = sum(self._rs) + self._ndims
        out = ffi.new("size_t[]", max_sites)
        n_out = ffi.new("size_t*")
        status = get_lib().t4a_qgrid_int_local_dims(self._ptr, out, max_sites, n_out)
        check_status(status, "local_dimensions")
        return [out[i] for i in range(n_out[0])]

    @property
    def origin(self) -> List[int]:
        """Origin per dimension."""
        out = ffi.new("int64_t[]", self._ndims)
        status = get_lib().t4a_qgrid_int_origin(self._ptr, out, self._ndims)
        check_status(status, "origin")
        return [out[i] for i in range(self._ndims)]

    def origcoord_to_quantics(self, coord: Sequence[int]) -> List[int]:
        """Convert original (integer) coordinates to quantics indices."""
        lib = get_lib()
        coord_c = ffi.new("int64_t[]", list(coord))
        max_sites = sum(self._rs) + self._ndims
        out = ffi.new("int64_t[]", max_sites)
        n_out = ffi.new("size_t*")
        status = lib.t4a_qgrid_int_origcoord_to_quantics(
            self._ptr, coord_c, len(coord), out, max_sites, n_out
        )
        check_status(status, "origcoord_to_quantics")
        return [out[i] for i in range(n_out[0])]

    def quantics_to_origcoord(self, quantics: Sequence[int]) -> List[int]:
        """Convert quantics indices to original (integer) coordinates."""
        lib = get_lib()
        quantics_c = ffi.new("int64_t[]", list(quantics))
        out = ffi.new("int64_t[]", self._ndims)
        status = lib.t4a_qgrid_int_quantics_to_origcoord(
            self._ptr, quantics_c, len(quantics), out, self._ndims
        )
        check_status(status, "quantics_to_origcoord")
        return [out[i] for i in range(self._ndims)]

    def origcoord_to_grididx(self, coord: Sequence[int]) -> List[int]:
        """Convert original coordinates to grid indices (1-indexed)."""
        lib = get_lib()
        coord_c = ffi.new("int64_t[]", list(coord))
        out = ffi.new("int64_t[]", self._ndims)
        status = lib.t4a_qgrid_int_origcoord_to_grididx(
            self._ptr, coord_c, len(coord), out, self._ndims
        )
        check_status(status, "origcoord_to_grididx")
        return [out[i] for i in range(self._ndims)]

    def grididx_to_origcoord(self, grididx: Sequence[int]) -> List[int]:
        """Convert grid indices (1-indexed) to original coordinates."""
        lib = get_lib()
        grididx_c = ffi.new("int64_t[]", list(grididx))
        out = ffi.new("int64_t[]", self._ndims)
        status = lib.t4a_qgrid_int_grididx_to_origcoord(
            self._ptr, grididx_c, len(grididx), out, self._ndims
        )
        check_status(status, "grididx_to_origcoord")
        return [out[i] for i in range(self._ndims)]

    def grididx_to_quantics(self, grididx: Sequence[int]) -> List[int]:
        """Convert grid indices to quantics indices."""
        lib = get_lib()
        grididx_c = ffi.new("int64_t[]", list(grididx))
        max_sites = sum(self._rs) + self._ndims
        out = ffi.new("int64_t[]", max_sites)
        n_out = ffi.new("size_t*")
        status = lib.t4a_qgrid_int_grididx_to_quantics(
            self._ptr, grididx_c, len(grididx), out, max_sites, n_out
        )
        check_status(status, "grididx_to_quantics")
        return [out[i] for i in range(n_out[0])]

    def quantics_to_grididx(self, quantics: Sequence[int]) -> List[int]:
        """Convert quantics indices to grid indices."""
        lib = get_lib()
        quantics_c = ffi.new("int64_t[]", list(quantics))
        out = ffi.new("int64_t[]", self._ndims)
        n_out = ffi.new("size_t*")
        status = lib.t4a_qgrid_int_quantics_to_grididx(
            self._ptr, quantics_c, len(quantics), out, self._ndims, n_out
        )
        check_status(status, "quantics_to_grididx")
        return [out[i] for i in range(n_out[0])]

    def __repr__(self) -> str:
        return f"InherentDiscreteGrid(ndims={self._ndims}, rs={self._rs})"
