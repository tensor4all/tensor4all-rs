# Julia documentation examples: SimpleTT (SimpleTensorTrain)
#
# Run with:
#   julia --project=julia/Tensor4all.jl docs/examples/julia/simplett.jl

using LinearAlgebra
using Tensor4all
using Tensor4all.SimpleTT: SimpleTensorTrain, site_dims, link_dims, rank, evaluate, site_tensor

# ANCHOR: create
tt = SimpleTensorTrain([2, 3, 4], 1.0)
zz = zeros(SimpleTensorTrain, [2, 3, 4])
@assert length(tt) == 3
@assert sum(zz) == 0.0
# ANCHOR_END: create

# ANCHOR: properties
@assert site_dims(tt) == [2, 3, 4]
@assert length(link_dims(tt)) == length(tt) - 1
@assert rank(tt) >= 1
# ANCHOR_END: properties

# ANCHOR: evaluate
@assert evaluate(tt, [0, 1, 2]) == 1.0
@assert tt(0, 1, 2) == 1.0
# ANCHOR_END: evaluate

# ANCHOR: operations
@assert isapprox(sum(tt), 24.0; rtol=0, atol=0) # 2 * 3 * 4
@assert isapprox(norm(tt), sqrt(24.0); rtol=0, atol=0) # Frobenius norm of all-ones tensor
core0 = site_tensor(tt, 0)
@assert size(core0, 2) == 2
tt2 = copy(tt)
@assert sum(tt2) == sum(tt)
# ANCHOR_END: operations
