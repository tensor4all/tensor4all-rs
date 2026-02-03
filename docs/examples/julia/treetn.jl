# Julia documentation examples: TreeTN (MPS / MPO / TreeTensorNetwork)
#
# Run with:
#   julia --project=julia/Tensor4all.jl docs/examples/julia/treetn.jl

using LinearAlgebra
using Random
using Tensor4all
using Tensor4all.TreeTN

# ANCHOR: create
s1 = Index(2; tags="Site,n=1")
s2 = Index(2; tags="Site,n=2")
l12 = Index(3; tags="Link,l=1")

t1 = Tensor([s1, l12], ones(2, 3))
t2 = Tensor([l12, s2], ones(3, 2))
mps = MPS([t1, t2])
@assert length(mps) == 2
# ANCHOR_END: create

# ANCHOR: accessors
@assert collect(mps) isa Vector{Tensor}
@assert mps[1] isa Tensor
@assert linkdims(mps) == [3]
@assert maxbonddim(mps) == 3
@assert linkind(mps, 1) == l12
@assert linkinds(mps) == [l12]
@assert nv(mps) == 2
@assert ne(mps) == 1
# ANCHOR_END: accessors

# ANCHOR: orthogonalize
orthogonalize!(mps, 1)
@assert canonical_form(mps) == Unitary
# ANCHOR_END: orthogonalize

# ANCHOR: truncate
truncate!(mps; maxdim=2, rtol=1e-12)
@assert maxbonddim(mps) <= 2
# ANCHOR_END: truncate

# ANCHOR: operations
mps_a = copy(mps)
mps_b = copy(mps)
mps_sum = mps_a + mps_b
@assert length(mps_sum) == length(mps)

# MPO x MPO contraction: only the "inner" indices (s1m/s2m) are shared
s1m = Index(2; tags="Site,n=1,Mid")
s2m = Index(2; tags="Site,n=2,Mid")
la = Index(2; tags="Link,a")
lb = Index(2; tags="Link,b")

mpo_a = MPS([
    Tensor([s1, s1m, la], ones(2, 2, 2)),
    Tensor([la, s2, s2m], ones(2, 2, 2)),
])
s1out = Index(2; tags="Site,n=1,Out")
s2out = Index(2; tags="Site,n=2,Out")
mpo_b = MPS([
    Tensor([s1m, s1out, lb], ones(2, 2, 2)),
    Tensor([lb, s2m, s2out], ones(2, 2, 2)),
])
mpo_contracted = contract(mpo_a, mpo_b)
@assert length(mpo_contracted) == length(mpo_a)

# Verify via dense arrays
da = to_dense(mpo_a)
db = to_dense(mpo_b)
dc = to_dense(mpo_contracted)
arr_a = Array(da, [s1, s1m, s2, s2m])   # i,j,k,l
arr_b = Array(db, [s1m, s1out, s2m, s2out])  # j,m,l,n
arr_c = Array(dc, [s1, s1out, s2, s2out])    # i,m,k,n
# Contract shared indices (s1m, s2m) manually
expected = zeros(2, 2, 2, 2)
for i in 1:2, j in 1:2, k in 1:2, l in 1:2, m in 1:2, n in 1:2
    expected[i, m, k, n] += arr_a[i, j, k, l] * arr_b[j, m, l, n]
end
@assert isapprox(arr_c, expected; atol=1e-12)

@assert norm(mps_a) > 0
@assert inner(mps_a, mps_a) isa ComplexF64
dense = to_dense(mps_a)
@assert Tensor4all.rank(dense) == 2
# ANCHOR_END: operations

# ANCHOR: siteinds
sites = siteinds(mps_a, 1)
@assert length(sites) >= 1
si = siteind(mps_a, 1)
@assert si !== nothing
@assert findsite(mps_a, si) == 1
# ANCHOR_END: siteinds

# ANCHOR: random
Random.seed!(0)
sites2 = [Index(2; tags="Site,n=$n") for n in 1:4]
rtt = random_mps(sites2; linkdims=2)
@assert length(rtt) == 4
# ANCHOR_END: random

# ANCHOR: linsolve
# 1-site example (matches the C-API test structure):
# operator is a 2-index tensor (matrix), rhs/init are 1-index tensors (vectors).
s = Index(2; tags="s")
sp = Index(2; tags="sP")

op_t = Tensor([s, sp], Matrix{Float64}(I, 2, 2))
op = MPS([op_t])

rhs_t = Tensor([sp], [3.0, 4.0])
rhs = MPS([rhs_t])

init_t = Tensor([s], [1.0, 1.0])
init = MPS([init_t])

x = linsolve(op, rhs, init; nsweeps=4, maxdim=10, rtol=1e-10)
@assert length(x) == 1
# ANCHOR_END: linsolve
