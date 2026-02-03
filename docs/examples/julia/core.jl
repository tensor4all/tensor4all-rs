# Julia documentation examples: core types (Index, Tensor)
#
# Run with:
#   julia --project=julia/Tensor4all.jl docs/examples/julia/core.jl

using Tensor4all

# ANCHOR: index_basic
i = Index(2; tags="Site,n=1")
@assert dim(i) == 2
@assert hastag(i, "Site")
@assert hastag(i, "n=1")
@assert id(i) isa UInt64
@assert tags(i) == "Site,n=1"
# ANCHOR_END: index_basic

# ANCHOR: index_utils
j = Index(3; tags="Link,l=1")
k = Index(2; tags="Site,n=2")

j2 = copy(j)
@assert id(j2) == id(j)
@assert tags(j2) == tags(j)

j_sim = sim(j)
@assert dim(j_sim) == dim(j)
@assert tags(j_sim) == tags(j)
@assert id(j_sim) != id(j)

@assert hascommoninds([i, j], [j, k])
@assert commoninds([i, j], [j, k]) == [j]
@assert uniqueinds([i, j], [j, k]) == [i]
@assert noncommoninds([i, j], [j, k]) == [i, k]

new_j = sim(j)
inds_replaced = replaceinds([i, j, k], [j], [new_j])
@assert inds_replaced == [i, new_j, k]
# ANCHOR_END: index_utils

# ANCHOR: tensor_basic
a = reshape(collect(1.0:6.0), 2, 3)
t = Tensor([i, j], a)
@assert rank(t) == 2
@assert dims(t) == (2, 3)
@assert indices(t) == [i, j]
@assert storage_kind(t) == DenseF64
# ANCHOR_END: tensor_basic

# ANCHOR: tensor_onehot
oh = onehot(i => 2, j => 3) # 1-based positions (ITensors.jl style)
oh_arr = Array(oh, [i, j])
@assert oh_arr[2, 3] == 1.0
@assert sum(oh_arr) == 1.0
# ANCHOR_END: tensor_onehot

# ANCHOR: tensor_array
tj_i = Array(t, [j, i])
@assert size(tj_i) == (3, 2)
@assert tj_i[2, 1] == a[1, 2]
# ANCHOR_END: tensor_array

# ANCHOR: tensor_complex
zc = reshape(ComplexF64.(1:6, -(1:6)), 2, 3)
tc = Tensor([i, j], zc)
@assert storage_kind(tc) == DenseC64
@assert Array(tc, [i, j])[1, 1] == zc[1, 1]
# ANCHOR_END: tensor_complex

