using Test
using Tensor4all
import ITensors

# Use qualified names to avoid ambiguity
const T4AIndex = Tensor4all.Index
const T4ATensor = Tensor4all.Tensor

skip_hdf5 = get(ENV, "T4A_SKIP_HDF5_TESTS", "") == "1"

@testset "Tensor4all.jl" begin
    include("test_index.jl")
    include("test_tensor.jl")
    include("test_treetn.jl")
    if !skip_hdf5
        include("test_hdf5.jl")
    end
    include("itensors_ext_test.jl")
    if !skip_hdf5
        include("test_hdf5_itensors_compat.jl")
    end
end
