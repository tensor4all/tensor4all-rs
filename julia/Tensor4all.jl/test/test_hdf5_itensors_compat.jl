using Test
using Tensor4all: Index as T4AIndex, Tensor as T4ATensor
using Tensor4all: dim, id, tags, rank, dims, indices, data
using Tensor4all: save_itensor, load_itensor
using Tensor4all.TreeTN: MPS, linkdims, save_mps, load_mps
using Tensor4all.TreeTN: random_mps
import ITensors
import HDF5

function temp_path(name::AbstractString)
    return joinpath(tempdir(), "tensor4all_jl_hdf5_compat_$(name).h5")
end

@testset "HDF5 ITensors.jl Compatibility" begin
    @testset "Tensor4all save → ITensors.jl load (f64)" begin
        filepath = temp_path("t4a_to_itensors_f64")

        # Create and save with Tensor4all
        i1 = T4AIndex(2; tags="Site,n=1")
        i2 = T4AIndex(3; tags="Link,l=1")
        arr = [1.0 2.0 3.0; 4.0 5.0 6.0]  # 2×3
        t = T4ATensor([i1, i2], arr)
        save_itensor(filepath, "tensor", t)

        # Load with ITensors.jl
        it = HDF5.h5open(filepath, "r") do f
            ITensors.read(f, "tensor", ITensors.ITensor)
        end

        # Check dimensions
        @test ITensors.dim(ITensors.inds(it)[1]) * ITensors.dim(ITensors.inds(it)[2]) == 6

        # Check data matches (convert ITensor to array)
        # Get ITensors indices in same order as original
        it_inds = ITensors.inds(it)
        it_i1 = nothing
        it_i2 = nothing
        for idx in it_inds
            if ITensors.id(idx) == id(i1)
                it_i1 = idx
            elseif ITensors.id(idx) == id(i2)
                it_i2 = idx
            end
        end
        @test it_i1 !== nothing
        @test it_i2 !== nothing

        it_arr = Array(it, it_i1, it_i2)
        @test it_arr ≈ arr atol=1e-14

        rm(filepath; force=true)
    end

    @testset "Tensor4all save → ITensors.jl load (c64)" begin
        filepath = temp_path("t4a_to_itensors_c64")

        i1 = T4AIndex(2; tags="Site,n=1")
        i2 = T4AIndex(3; tags="Link,l=1")
        arr = ComplexF64[1+0.1im 2+0.2im 3+0.3im; 4+0.4im 5+0.5im 6+0.6im]
        t = T4ATensor([i1, i2], arr)
        save_itensor(filepath, "tensor", t)

        it = HDF5.h5open(filepath, "r") do f
            ITensors.read(f, "tensor", ITensors.ITensor)
        end

        it_inds = ITensors.inds(it)
        it_i1 = nothing
        it_i2 = nothing
        for idx in it_inds
            if ITensors.id(idx) == id(i1)
                it_i1 = idx
            elseif ITensors.id(idx) == id(i2)
                it_i2 = idx
            end
        end
        @test it_i1 !== nothing
        @test it_i2 !== nothing

        it_arr = Array(it, it_i1, it_i2)
        @test it_arr ≈ arr atol=1e-14

        rm(filepath; force=true)
    end

    @testset "ITensors.jl save → Tensor4all load (f64)" begin
        filepath = temp_path("itensors_to_t4a_f64")

        # Create and save with ITensors.jl
        it_i1 = ITensors.Index(2, "Site,n=1")
        it_i2 = ITensors.Index(3, "Link,l=1")
        arr = [1.0 2.0 3.0; 4.0 5.0 6.0]
        it = ITensors.ITensor(arr, it_i1, it_i2)

        HDF5.h5open(filepath, "w") do f
            ITensors.write(f, "tensor", it)
        end

        # Load with Tensor4all
        loaded = load_itensor(filepath, "tensor")

        # Check dims
        @test dims(loaded) == (2, 3)

        # Check data matches
        loaded_inds = indices(loaded)
        # Find matching indices by ID
        t4a_i1 = nothing
        t4a_i2 = nothing
        for idx in loaded_inds
            if id(idx) == ITensors.id(it_i1)
                t4a_i1 = idx
            elseif id(idx) == ITensors.id(it_i2)
                t4a_i2 = idx
            end
        end
        @test t4a_i1 !== nothing
        @test t4a_i2 !== nothing

        loaded_arr = Array(loaded, [t4a_i1, t4a_i2])
        @test loaded_arr ≈ arr atol=1e-14

        rm(filepath; force=true)
    end

    @testset "ITensors.jl save → Tensor4all load (c64)" begin
        filepath = temp_path("itensors_to_t4a_c64")

        it_i1 = ITensors.Index(2, "Site,n=1")
        it_i2 = ITensors.Index(3, "Link,l=1")
        arr = ComplexF64[1+0.1im 2+0.2im 3+0.3im; 4+0.4im 5+0.5im 6+0.6im]
        it = ITensors.ITensor(arr, it_i1, it_i2)

        HDF5.h5open(filepath, "w") do f
            ITensors.write(f, "tensor", it)
        end

        loaded = load_itensor(filepath, "tensor")

        loaded_inds = indices(loaded)
        t4a_i1 = nothing
        t4a_i2 = nothing
        for idx in loaded_inds
            if id(idx) == ITensors.id(it_i1)
                t4a_i1 = idx
            elseif id(idx) == ITensors.id(it_i2)
                t4a_i2 = idx
            end
        end
        @test t4a_i1 !== nothing
        @test t4a_i2 !== nothing

        loaded_arr = Array(loaded, [t4a_i1, t4a_i2])
        @test loaded_arr ≈ arr atol=1e-14

        rm(filepath; force=true)
    end
end
