"""Algorithm selection types (corresponds to tensor4all-core-common).

Provides enum-like types for selecting algorithms in various tensor operations.

Examples
--------
>>> from tensor4all.algorithm import FactorizeAlgorithm, ContractionAlgorithm
>>>
>>> # Factorization algorithms
>>> alg = FactorizeAlgorithm.SVD
>>> print(FactorizeAlgorithm.name(alg))  # "svd"
>>>
>>> # Contraction algorithms
>>> alg = ContractionAlgorithm.ZipUp
"""

from enum import IntEnum
from math import sqrt
from typing import Optional

from ._capi import get_lib


class FactorizeAlgorithm(IntEnum):
    """Algorithm for matrix factorization / decomposition.

    Used in compression, truncation, and various tensor operations.

    Attributes
    ----------
    SVD : int
        Singular Value Decomposition (default, optimal truncation)
    LU : int
        LU decomposition with partial pivoting (faster)
    CI : int
        Cross Interpolation / Skeleton decomposition (adaptive)
    """

    SVD = 0
    LU = 1
    CI = 2

    @classmethod
    def default(cls) -> "FactorizeAlgorithm":
        """Return the default algorithm."""
        return cls.SVD

    @classmethod
    def from_name(cls, name: str) -> "FactorizeAlgorithm":
        """Parse algorithm from string name."""
        lower = name.lower()
        if lower == "svd":
            return cls.SVD
        elif lower == "lu":
            return cls.LU
        elif lower in ("ci", "cross", "crossinterpolation"):
            return cls.CI
        else:
            raise ValueError(f"Unknown FactorizeAlgorithm: {name}")

    def name(self) -> str:
        """Get algorithm name as string."""
        return self._name_.lower()


class ContractionAlgorithm(IntEnum):
    """Algorithm for tensor train contraction (TT-TT or MPO-MPO).

    Attributes
    ----------
    Naive : int
        Exact contraction followed by compression (default)
    ZipUp : int
        On-the-fly compression during contraction (memory efficient)
    Fit : int
        Variational fitting (best for low target rank)
    """

    Naive = 0
    ZipUp = 1
    Fit = 2

    @classmethod
    def default(cls) -> "ContractionAlgorithm":
        """Return the default algorithm."""
        return cls.Naive

    @classmethod
    def from_name(cls, name: str) -> "ContractionAlgorithm":
        """Parse algorithm from string name."""
        lower = name.lower()
        if lower == "naive":
            return cls.Naive
        elif lower in ("zipup", "zip_up", "zip-up"):
            return cls.ZipUp
        elif lower in ("fit", "variational"):
            return cls.Fit
        else:
            raise ValueError(f"Unknown ContractionAlgorithm: {name}")

    def name(self) -> str:
        """Get algorithm name as string."""
        return self._name_.lower()


class CompressionAlgorithm(IntEnum):
    """Algorithm for tensor train compression.

    Attributes
    ----------
    SVD : int
        SVD-based compression (default, optimal)
    LU : int
        LU-based compression (faster)
    CI : int
        Cross Interpolation based compression
    Variational : int
        Variational compression (sweeping optimization)
    """

    SVD = 0
    LU = 1
    CI = 2
    Variational = 3

    @classmethod
    def default(cls) -> "CompressionAlgorithm":
        """Return the default algorithm."""
        return cls.SVD

    @classmethod
    def from_name(cls, name: str) -> "CompressionAlgorithm":
        """Parse algorithm from string name."""
        lower = name.lower()
        if lower == "svd":
            return cls.SVD
        elif lower == "lu":
            return cls.LU
        elif lower in ("ci", "cross", "crossinterpolation"):
            return cls.CI
        elif lower in ("variational", "fit"):
            return cls.Variational
        else:
            raise ValueError(f"Unknown CompressionAlgorithm: {name}")

    def name(self) -> str:
        """Get algorithm name as string."""
        return self._name_.lower()


def get_default_svd_rtol() -> float:
    """Get the default SVD relative tolerance from the Rust library.

    This is the default value used for truncation when no tolerance is specified.

    Returns
    -------
    float
        The default relative tolerance.
    """
    lib = get_lib()
    return lib.t4a_get_default_svd_rtol()


def resolve_truncation_tolerance(
    *,
    cutoff: Optional[float] = None,
    rtol: Optional[float] = None,
) -> float:
    """Resolve truncation tolerance from cutoff (ITensor style) or rtol (tensor4all style).

    Parameters
    ----------
    cutoff : float, optional
        ITensor-style squared error tolerance: sum(discarded sigma^2) / sum(sigma^2) <= cutoff
    rtol : float, optional
        tensor4all-style relative Frobenius error: ||A - A_approx||_F / ||A||_F <= rtol

    Returns
    -------
    float
        The effective rtol value.

    Raises
    ------
    ValueError
        If both cutoff and rtol are specified.

    Notes
    -----
    Conversion: cutoff = rtol^2, so rtol = sqrt(cutoff)

    - If neither specified: returns the default rtol from Rust
    - If only cutoff specified: returns sqrt(cutoff)
    - If only rtol specified: returns rtol
    - If both specified: raises ValueError

    Examples
    --------
    >>> # Use default
    >>> rtol = resolve_truncation_tolerance()
    >>>
    >>> # ITensor style
    >>> rtol = resolve_truncation_tolerance(cutoff=1e-24)  # returns 1e-12
    >>>
    >>> # tensor4all style
    >>> rtol = resolve_truncation_tolerance(rtol=1e-12)
    """
    if cutoff is not None and rtol is not None:
        raise ValueError(
            "Cannot specify both `cutoff` and `rtol`. Use one or the other.\n"
            "- cutoff (ITensor style): sum(discarded sigma^2) / sum(sigma^2) <= cutoff\n"
            "- rtol (tensor4all style): ||A - A_approx||_F / ||A||_F <= rtol\n"
            "Conversion: cutoff = rtol^2, so rtol = sqrt(cutoff)"
        )

    if cutoff is not None:
        return sqrt(cutoff)
    elif rtol is not None:
        return rtol
    else:
        return get_default_svd_rtol()
