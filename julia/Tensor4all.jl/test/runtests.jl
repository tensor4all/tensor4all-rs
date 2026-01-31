using Test
using Tensor4all
import ITensors

# Use qualified names to avoid ambiguity
const T4AIndex = Tensor4all.Index
const T4ATensor = Tensor4all.Tensor

@testset "Tensor4all.jl" begin
    include("test_index.jl")
    include("test_tensor.jl")
    include("test_itensorlike_tensortrain.jl")
    include("test_hdf5.jl")
    include("itensors_ext_test.jl")
    include("test_hdf5_itensors_compat.jl")
end
