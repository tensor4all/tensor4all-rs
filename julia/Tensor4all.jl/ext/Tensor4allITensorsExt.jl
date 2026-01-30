"""
    Tensor4allITensorsExt

Extension module providing bidirectional conversion between
Tensor4all types and ITensors types:
- Tensor4all.Index ↔ ITensors.Index
- Tensor4all.Tensor ↔ ITensors.ITensor

## ID Mapping Policy

Both Rust and ITensors use UInt64 IDs natively, so conversion is direct.

## Storage Mapping

- DenseF64 ↔ Dense{Float64}
- DenseC64 ↔ Dense{ComplexF64}
- DiagF64, DiagC64 → Error (not yet supported)

## Memory Order

Rust uses row-major, Julia/ITensors uses column-major.
Conversion is handled automatically.
"""
module Tensor4allITensorsExt

using Tensor4all
using ITensors

# ============================================================================
# Tensor4all.Index → ITensors.Index
# ============================================================================

"""
    ITensors.Index(idx::Tensor4all.Index)

Convert a Tensor4all.Index to an ITensors.Index.

IDs are natively UInt64 in both systems. Tags are preserved.
"""
function ITensors.Index(idx::Tensor4all.Index)
    d = Tensor4all.dim(idx)
    t = Tensor4all.tags(idx)
    id64 = Tensor4all.id(idx)

    # Create ITensors.Index with explicit ID using full constructor
    # Index(id, space, dir, tags, plev)
    tagset = isempty(t) ? ITensors.TagSet("") : ITensors.TagSet(t)
    return ITensors.Index(id64, d, ITensors.Neither, tagset, 0)
end

# ============================================================================
# ITensors.Index → Tensor4all.Index
# ============================================================================

"""
    Tensor4all.Index(idx::ITensors.Index)

Convert an ITensors.Index to a Tensor4all.Index.

IDs are natively UInt64 in both systems. Tags are preserved.

Note: Tags that exceed Rust limits (max 4 tags, max 16 chars each) will
cause an error.
"""
function Tensor4all.Index(idx::ITensors.Index)
    d = ITensors.dim(idx)
    id64 = ITensors.id(idx)

    # Get tags as comma-separated string
    tag_set = ITensors.tags(idx)
    tags_str = _tags_to_string(tag_set)

    return Tensor4all.Index(d, id64; tags=tags_str)
end

"""
Convert ITensors TagSet to comma-separated string.
"""
function _tags_to_string(ts::ITensors.TagSet)
    n = length(ts)
    if n == 0
        return ""
    end

    tag_strings = String[]
    for i in 1:n
        push!(tag_strings, string(ts[i]))
    end
    return join(tag_strings, ",")
end

# ============================================================================
# Conversion functions
# ============================================================================

"""
    Base.convert(::Type{ITensors.Index}, idx::Tensor4all.Index)

Enable `convert(ITensors.Index, t4a_idx)`.
"""
Base.convert(::Type{ITensors.Index}, idx::Tensor4all.Index) = ITensors.Index(idx)

"""
    Base.convert(::Type{Tensor4all.Index}, idx::ITensors.Index)

Enable `convert(Tensor4all.Index, it_idx)`.
"""
Base.convert(::Type{Tensor4all.Index}, idx::ITensors.Index) = Tensor4all.Index(idx)

# ============================================================================
# Tensor4all.Tensor → ITensors.ITensor
# ============================================================================

"""
    ITensors.ITensor(t::Tensor4all.Tensor)

Convert a Tensor4all.Tensor to an ITensors.ITensor.

Currently only supports DenseF64 and DenseC64 storage types.
Diag storage will raise an error.
"""
function ITensors.ITensor(t::Tensor4all.Tensor)
    kind = Tensor4all.storage_kind(t)

    # Check for supported storage types
    if kind == Tensor4all.DiagF64 || kind == Tensor4all.DiagC64
        error("Diag storage not yet supported for ITensor conversion")
    end

    # Get indices and convert
    t4a_inds = Tensor4all.indices(t)
    it_inds = [ITensors.Index(idx) for idx in t4a_inds]

    # Get data (already column-major from Tensor4all.data)
    arr = Tensor4all.data(t)

    # Create ITensor
    return ITensors.ITensor(arr, it_inds...)
end

# ============================================================================
# ITensors.ITensor → Tensor4all.Tensor
# ============================================================================

"""
    Tensor4all.Tensor(it::ITensors.ITensor)

Convert an ITensors.ITensor to a Tensor4all.Tensor.

Requirements:
- ITensor must be dense (non-dense storage will error)
- Index tags must fit Rust limits (max 4 tags, max 16 chars each)

Note: The ITensor is converted to a contiguous array before passing to Rust.
"""
function Tensor4all.Tensor(it::ITensors.ITensor)
    # Check that ITensor is dense
    if !_is_dense(it)
        error("Only dense ITensors are supported. Non-dense ITensors must be explicitly densified first.")
    end

    # Get indices and convert
    it_inds = ITensors.inds(it)
    t4a_inds = [Tensor4all.Index(idx) for idx in it_inds]

    # Get data in canonical index order (column-major Julia array)
    # Use Array() to get dense array
    arr = Array(it, it_inds...)

    # Create Tensor4all.Tensor
    return Tensor4all.Tensor(t4a_inds, arr)
end

"""
Check if an ITensor has dense storage.
"""
function _is_dense(it::ITensors.ITensor)
    # Check if storage is Dense
    storage = ITensors.storage(it)
    return storage isa ITensors.NDTensors.Dense
end

# ============================================================================
# Conversion functions for Tensor
# ============================================================================

"""
    Base.convert(::Type{ITensors.ITensor}, t::Tensor4all.Tensor)

Enable `convert(ITensors.ITensor, t4a_tensor)`.
"""
Base.convert(::Type{ITensors.ITensor}, t::Tensor4all.Tensor) = ITensors.ITensor(t)

"""
    Base.convert(::Type{Tensor4all.Tensor}, it::ITensors.ITensor)

Enable `convert(Tensor4all.Tensor, itensor)`.
"""
Base.convert(::Type{Tensor4all.Tensor}, it::ITensors.ITensor) = Tensor4all.Tensor(it)

end # module
