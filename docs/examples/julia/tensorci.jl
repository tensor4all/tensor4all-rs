# Julia documentation examples: TensorCI (cross interpolation)
#
# Run with:
#   julia --project=julia/Tensor4all.jl docs/examples/julia/tensorci.jl

using Tensor4all
using Tensor4all.TensorCI: TensorCI2, crossinterpolate2, crossinterpolate2_tci, rank, add_global_pivots!, to_tensor_train
using Tensor4all.SimpleTT: evaluate

# ANCHOR: basic
f(i, j, k) = Float64((1 + i) * (1 + j) * (1 + k)) # 0-based indices
tt, err = crossinterpolate2(f, [2, 2, 2]; tolerance=1e-12)
@assert length(tt) == 3
@assert err >= 0
# ANCHOR_END: basic

# ANCHOR: evaluate
@assert tt(0, 0, 0) == 1.0
@assert tt(1, 1, 1) == 8.0
@assert evaluate(tt, [1, 0, 1]) == f(1, 0, 1)
# ANCHOR_END: evaluate

# ANCHOR: manual
tci2, err2 = crossinterpolate2_tci(f, [2, 2, 2]; tolerance=1e-12)
@assert length(tci2) == 3
@assert err2 >= 0
tt2 = to_tensor_train(tci2)
@assert length(tt2) == 3
# ANCHOR_END: manual
