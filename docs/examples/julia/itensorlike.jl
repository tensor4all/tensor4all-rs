# Julia documentation examples: ITensorLike (TensorTrain / MPS / MPO)
#
# Run with:
#   julia --project=julia/Tensor4all.jl docs/examples/julia/itensorlike.jl

using LinearAlgebra
using Random
using Tensor4all
using Tensor4all.ITensorLike

# ANCHOR: create
s1 = Index(2; tags="Site,n=1")
s2 = Index(2; tags="Site,n=2")
l12 = Index(3; tags="Link,l=1")

t1 = Tensor([s1, l12], ones(2, 3))
t2 = Tensor([l12, s2], ones(3, 2))
tt = TensorTrain([t1, t2])
@assert length(tt) == 2
# ANCHOR_END: create

# ANCHOR: accessors
@assert tensors(tt) isa Vector{Tensor}
@assert tt[1] isa Tensor
@assert bond_dims(tt) == [3]
@assert maxbonddim(tt) == 3
@assert linkind(tt, 1) == l12
@assert linkinds(tt) == [l12]
# ANCHOR_END: accessors

# ANCHOR: orthogonalize
orthogonalize!(tt, 1)
@assert isortho(tt)
@assert orthocenter(tt) == 1
@assert llim(tt) <= rlim(tt)
# ANCHOR_END: orthogonalize

# ANCHOR: truncate
truncate!(tt; maxdim=2, rtol=1e-12)
@assert maxbonddim(tt) <= 2
# ANCHOR_END: truncate

# ANCHOR: operations
tt_a = copy(tt)
tt_b = copy(tt)
tt_sum = tt_a + tt_b
@assert length(tt_sum) == length(tt)

tt_contracted = contract(tt_a, tt_b)
@assert length(tt_contracted) > 0

@assert norm(tt_a) > 0
@assert inner(tt_a, tt_a) isa ComplexF64
dense = to_dense(tt_a)
@assert Tensor4all.rank(dense) == 2
# ANCHOR_END: operations

# ANCHOR: siteinds
sites = siteinds(tt_a)
@assert length(sites) == length(tt_a)
@assert tags(siteind(tt_a, 1)) == "Site,n=1"
@assert findsite(tt_a, siteind(tt_a, 1)) == 1
@assert findsites(tt_a, [siteind(tt_a, 1), siteind(tt_a, 2)]) == [1, 2]
# ANCHOR_END: siteinds

# ANCHOR: random
Random.seed!(0)
sites2 = [Index(2; tags="Site,n=$n") for n in 1:4]
rtt = random_tt(sites2; linkdims=2)
@assert length(rtt) == 4
# ANCHOR_END: random

# ANCHOR: linsolve
# 1-site example (matches the C-API test structure):
# operator is a 2-index tensor (matrix), rhs/init are 1-index tensors (vectors).
s = Index(2; tags="s")
sp = Index(2; tags="sP")

op_t = Tensor([s, sp], Matrix{Float64}(I, 2, 2))
op = TensorTrain([op_t])

rhs_t = Tensor([sp], [3.0, 4.0])
rhs = TensorTrain([rhs_t])

init_t = Tensor([s], [1.0, 1.0])
init = TensorTrain([init_t])

x = linsolve(op, rhs, init; nhalfsweeps=4, maxdim=10, rtol=1e-10, krylov_tol=1e-12, convergence_tol=1e-8)
@assert length(x) == 1
# ANCHOR_END: linsolve
