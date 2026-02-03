using Test
using Tensor4all: Index as T4AIndex, Tensor as T4ATensor
using Tensor4all: dim, id, tags, rank, dims, indices, data
using Tensor4all: save_itensor, load_itensor
using Tensor4all.TreeTN: MPS, linkdims, save_mps, load_mps
using Tensor4all.TreeTN: random_mps

function temp_path(name::AbstractString)
    return joinpath(tempdir(), "tensor4all_jl_hdf5_test_$(name).h5")
end

@testset "HDF5 Save/Load" begin
    @testset "ITensor f64 roundtrip" begin
        filepath = temp_path("itensor_f64")
        i1 = T4AIndex(2; tags="Site,n=1")
        i2 = T4AIndex(3; tags="Link,l=1")
        arr = rand(2, 3)
        t = T4ATensor([i1, i2], arr)

        save_itensor(filepath, "tensor", t)
        loaded = load_itensor(filepath, "tensor")

        @test dims(loaded) == dims(t)

        # Check index properties survive roundtrip
        orig_inds = indices(t)
        loaded_inds = indices(loaded)
        @test length(orig_inds) == length(loaded_inds)
        for (oi, li) in zip(orig_inds, loaded_inds)
            @test id(oi) == id(li)
            @test dim(oi) == dim(li)
            @test tags(oi) == tags(li)
        end

        # Check data
        orig_data = data(t)
        loaded_data = data(loaded)
        @test orig_data ≈ loaded_data atol=1e-14

        rm(filepath; force=true)
    end

    @testset "ITensor c64 roundtrip" begin
        filepath = temp_path("itensor_c64")
        i1 = T4AIndex(2; tags="Site,n=1")
        i2 = T4AIndex(3; tags="Link,l=1")
        arr = rand(ComplexF64, 2, 3)
        t = T4ATensor([i1, i2], arr)

        save_itensor(filepath, "tensor", t)
        loaded = load_itensor(filepath, "tensor")

        @test dims(loaded) == dims(t)

        orig_data = data(t)
        loaded_data = data(loaded)
        @test orig_data ≈ loaded_data atol=1e-14

        rm(filepath; force=true)
    end

    @testset "ITensor 3D roundtrip" begin
        filepath = temp_path("itensor_3d")
        i1 = T4AIndex(2; tags="Link,l=0")
        i2 = T4AIndex(3; tags="Site,n=1")
        i3 = T4AIndex(4; tags="Link,l=1")
        arr = rand(2, 3, 4)
        t = T4ATensor([i1, i2, i3], arr)

        save_itensor(filepath, "tensor3d", t)
        loaded = load_itensor(filepath, "tensor3d")

        @test dims(loaded) == dims(t)
        @test data(loaded) ≈ data(t) atol=1e-14

        rm(filepath; force=true)
    end

    @testset "MPS roundtrip" begin
        filepath = temp_path("mps")
        sites = [T4AIndex(2; tags="Site,n=$i") for i in 1:3]
        mps = random_mps(sites; linkdims=4)

        save_mps(filepath, "mps", mps)
        loaded = load_mps(filepath, "mps")

        @test length(loaded) == length(mps)

        orig_tensors = collect(mps)
        loaded_tensors = collect(loaded)
        for (i, (ot, lt)) in enumerate(zip(orig_tensors, loaded_tensors))
            @test dims(ot) == dims(lt)

            # Check index IDs preserved
            oi = indices(ot)
            li = indices(lt)
            for (o, l) in zip(oi, li)
                @test id(o) == id(l)
                @test dim(o) == dim(l)
            end

            # Check data
            @test data(ot) ≈ data(lt) atol=1e-14
        end

        rm(filepath; force=true)
    end
end
