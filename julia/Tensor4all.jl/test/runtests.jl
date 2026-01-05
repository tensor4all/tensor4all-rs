using Test
using Tensor4all
using ITensors

# Use qualified names to avoid ambiguity
const T4AIndex = Tensor4all.Index
const T4ATensor = Tensor4all.Tensor

@testset "Tensor4all.jl" begin
    include("test_index.jl")
    include("test_tensor.jl")
    # DISABLED - pending simpletensortrain C-API integration
    # include("test_tensortrain.jl")
    include("itensors_ext_test.jl")
end
