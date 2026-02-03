"""Index class for tensor4all."""

from __future__ import annotations

from typing import Sequence

from ._capi import check_status, get_lib, T4AError
from ._ffi import ffi


class Index:
    """A tensor index with dimension, unique ID, and tags.

    An Index represents one dimension of a tensor and has:
    - A dimension (size)
    - A unique 64-bit ID (compatible with ITensors.jl)
    - Optional tags (string labels like "Site", "n=1")

    Examples
    --------
    >>> i = Index(5)  # Create index with dimension 5
    >>> i.dim
    5
    >>> j = Index(3, tags="Site,n=1")  # Create with tags
    >>> j.has_tag("Site")
    True
    """

    __slots__ = ("_ptr",)

    def __init__(
        self,
        dim: int,
        *,
        tags: str = "",
        id: int | None = None,
    ):
        """Create a new Index.

        Parameters
        ----------
        dim : int
            The dimension (size) of the index. Must be > 0.
        tags : str, optional
            Comma-separated tags, e.g., "Site,n=1". Default is no tags.
        id : int, optional
            64-bit ID (compatible with ITensors.jl's UInt64). If None, a random ID is generated.

        Raises
        ------
        ValueError
            If dim <= 0
        T4AError
            If creation fails (e.g., too many tags)
        """
        if dim <= 0:
            raise ValueError(f"Index dimension must be positive, got {dim}")

        lib = get_lib()

        if id is not None:
            tags_bytes = tags.encode("utf-8") if tags else ffi.NULL
            ptr = lib.t4a_index_new_with_id(dim, id, tags_bytes)
        elif tags:
            tags_bytes = tags.encode("utf-8")
            ptr = lib.t4a_index_new_with_tags(dim, tags_bytes)
        else:
            ptr = lib.t4a_index_new(dim)

        if ptr == ffi.NULL:
            raise T4AError("Failed to create Index")

        self._ptr = ptr

    @classmethod
    def _from_ptr(cls, ptr) -> Index:
        """Create an Index from an existing C pointer (internal use)."""
        if ptr == ffi.NULL:
            raise T4AError("Cannot create Index from NULL pointer")
        instance = object.__new__(cls)
        instance._ptr = ptr
        return instance

    def __del__(self):
        """Release the underlying C object."""
        if hasattr(self, "_ptr") and self._ptr != ffi.NULL:
            lib = get_lib()
            lib.t4a_index_release(self._ptr)
            self._ptr = ffi.NULL

    def __repr__(self) -> str:
        tags_str = self.tags
        if tags_str:
            return f"Index(dim={self.dim}, tags='{tags_str}')"
        return f"Index(dim={self.dim})"

    def __eq__(self, other: object) -> bool:
        """Two indices are equal if they have the same ID and tags."""
        if not isinstance(other, Index):
            return NotImplemented
        return self.id == other.id and self.tags == other.tags

    def __hash__(self) -> int:
        """Hash based on ID and tags."""
        return hash((self.id, self.tags))

    @property
    def dim(self) -> int:
        """Get the dimension (size) of the index."""
        lib = get_lib()
        out = ffi.new("size_t*")
        status = lib.t4a_index_dim(self._ptr, out)
        check_status(status, "Failed to get index dimension")
        return out[0]

    @property
    def id(self) -> int:
        """Get the 64-bit ID (compatible with ITensors.jl's UInt64)."""
        lib = get_lib()
        out_id = ffi.new("uint64_t*")
        status = lib.t4a_index_id(self._ptr, out_id)
        check_status(status, "Failed to get index ID")
        return out_id[0]

    @property
    def tags(self) -> str:
        """Get the comma-separated tags string."""
        lib = get_lib()

        # First, query required length
        out_len = ffi.new("size_t*")
        status = lib.t4a_index_get_tags(self._ptr, ffi.NULL, 0, out_len)
        check_status(status, "Failed to get tags length")

        if out_len[0] <= 1:
            return ""

        # Allocate buffer and get tags
        buf = ffi.new(f"uint8_t[{out_len[0]}]")
        status = lib.t4a_index_get_tags(self._ptr, buf, out_len[0], out_len)
        check_status(status, "Failed to get tags")

        return ffi.string(buf).decode("utf-8")

    def has_tag(self, tag: str) -> bool:
        """Check if the index has a specific tag.

        Parameters
        ----------
        tag : str
            The tag to check for.

        Returns
        -------
        bool
            True if the tag exists, False otherwise.
        """
        lib = get_lib()
        tag_bytes = tag.encode("utf-8")
        result = lib.t4a_index_has_tag(self._ptr, tag_bytes)
        if result < 0:
            raise T4AError("Failed to check tag")
        return result == 1

    def add_tag(self, tag: str) -> None:
        """Add a tag to the index.

        Parameters
        ----------
        tag : str
            The tag to add.

        Raises
        ------
        TagOverflowError
            If too many tags
        TagTooLongError
            If tag string is too long
        """
        lib = get_lib()
        tag_bytes = tag.encode("utf-8")
        status = lib.t4a_index_add_tag(self._ptr, tag_bytes)
        check_status(status, f"Failed to add tag '{tag}'")

    def set_tags(self, tags: str) -> None:
        """Set all tags from a comma-separated string (replaces existing).

        Parameters
        ----------
        tags : str
            Comma-separated tags, e.g., "Site,n=1"
        """
        lib = get_lib()
        tags_bytes = tags.encode("utf-8")
        status = lib.t4a_index_set_tags_csv(self._ptr, tags_bytes)
        check_status(status, "Failed to set tags")

    def clone(self) -> Index:
        """Create a copy of this index with the same ID and tags."""
        lib = get_lib()
        ptr = lib.t4a_index_clone(self._ptr)
        if ptr == ffi.NULL:
            raise T4AError("Failed to clone Index")
        return Index._from_ptr(ptr)


def sim(index: Index) -> Index:
    """Create a "similar" index with the same dim/tags but a new ID."""
    return Index(index.dim, tags=index.tags)


def hasind(inds: Sequence[Index], ind: Index) -> bool:
    """Return True if `ind` is present in `inds`."""
    return ind in inds


def hasinds(inds: Sequence[Index], query: Sequence[Index]) -> bool:
    """Return True if all indices in `query` are present in `inds`."""
    return all(q in inds for q in query)


def commoninds(inds1: Sequence[Index], inds2: Sequence[Index]) -> list[Index]:
    """Return indices common to both lists, preserving the order of `inds1`."""
    inds2_set = set(inds2)
    return [i for i in inds1 if i in inds2_set]


def hascommoninds(inds1: Sequence[Index], inds2: Sequence[Index]) -> bool:
    """Return True if there is at least one common index."""
    return len(commoninds(inds1, inds2)) > 0


def commonind(inds1: Sequence[Index], inds2: Sequence[Index]) -> Index:
    """Return the unique common index (error if not exactly one)."""
    cs = commoninds(inds1, inds2)
    if len(cs) != 1:
        raise ValueError(f"Expected exactly 1 common index, got {len(cs)}")
    return cs[0]


def uniqueinds(inds1: Sequence[Index], inds2: Sequence[Index]) -> list[Index]:
    """Return indices that are in `inds1` but not in `inds2`, preserving order."""
    inds2_set = set(inds2)
    return [i for i in inds1 if i not in inds2_set]


def uniqueind(inds1: Sequence[Index], inds2: Sequence[Index]) -> Index:
    """Return the unique index in `inds1` not present in `inds2` (error if not exactly one)."""
    us = uniqueinds(inds1, inds2)
    if len(us) != 1:
        raise ValueError(f"Expected exactly 1 unique index, got {len(us)}")
    return us[0]


def noncommoninds(inds1: Sequence[Index], inds2: Sequence[Index]) -> list[Index]:
    """Return indices that are not shared between the two lists."""
    return uniqueinds(inds1, inds2) + uniqueinds(inds2, inds1)


def replaceinds(
    inds: Sequence[Index],
    old_inds: Sequence[Index],
    new_inds: Sequence[Index],
) -> list[Index]:
    """Replace indices in `inds` by mapping `old_inds[i] -> new_inds[i]`."""
    if len(old_inds) != len(new_inds):
        raise ValueError("old_inds and new_inds must have the same length")
    mapping = {old: new for old, new in zip(old_inds, new_inds)}
    return [mapping.get(i, i) for i in inds]


def replaceind(inds: Sequence[Index], old_ind: Index, new_ind: Index) -> list[Index]:
    """Replace a single index in `inds`."""
    return replaceinds(inds, [old_ind], [new_ind])


__all__ = [
    "Index",
    "sim",
    "hasind",
    "hasinds",
    "hascommoninds",
    "commoninds",
    "commonind",
    "uniqueinds",
    "uniqueind",
    "noncommoninds",
    "replaceinds",
    "replaceind",
]
