using Test
using Tensor4all
using ITensors

# Use qualified names to avoid ambiguity
const T4AIndex = Tensor4all.Index
const T4ATensor = Tensor4all.Tensor

@testset "Tensor4all.jl" begin
    @testset "Index creation" begin
        # Basic creation
        i = T4AIndex(5)
        @test Tensor4all.dim(i) == 5
        @test Tensor4all.tags(i) == ""
        @test Tensor4all.id(i) != 0

        # With tags
        j = T4AIndex(3; tags="Site,n=1")
        @test Tensor4all.dim(j) == 3
        @test Tensor4all.hastag(j, "Site")
        @test Tensor4all.hastag(j, "n=1")
        @test !Tensor4all.hastag(j, "Missing")

        # Tags string contains both
        t = Tensor4all.tags(j)
        @test occursin("Site", t)
        @test occursin("n=1", t)
    end

    @testset "Index with custom ID" begin
        id_val = UInt128(0x12345678_9ABCDEF0_FEDCBA98_76543210)
        i = T4AIndex(4, id_val; tags="Custom")
        @test Tensor4all.dim(i) == 4
        @test Tensor4all.id(i) == id_val
        @test Tensor4all.hastag(i, "Custom")
    end

    @testset "Index copy" begin
        i = T4AIndex(5; tags="Original")
        j = copy(i)

        @test Tensor4all.dim(i) == Tensor4all.dim(j)
        @test Tensor4all.id(i) == Tensor4all.id(j)
        @test Tensor4all.tags(i) == Tensor4all.tags(j)
        @test i == j  # Equal by ID
    end

    @testset "Index equality and hashing" begin
        i = T4AIndex(5)
        j = copy(i)
        k = T4AIndex(5)  # Different ID

        @test i == j
        @test i != k
        @test hash(i) == hash(j)
        @test hash(i) != hash(k)  # Very likely different
    end

    @testset "Index display" begin
        i = T4AIndex(3; tags="Site")
        s = sprint(show, i)
        @test occursin("dim=3", s)
        @test occursin("Site", s)
    end

    @testset "Error handling" begin
        @test_throws ArgumentError T4AIndex(0)
        @test_throws ArgumentError T4AIndex(-1)
    end

    @testset "ITensors Extension" begin
        @testset "Tensor4all.Index → ITensors.Index" begin
            t4a_idx = Tensor4all.Index(5; tags="Site,n=1")
            it_idx = ITensors.Index(t4a_idx)

            @test ITensors.dim(it_idx) == 5
            @test ITensors.hastags(it_idx, "Site")
            @test ITensors.hastags(it_idx, "n=1")

            # ID should match (lower 64 bits)
            t4a_id = Tensor4all.id(t4a_idx)
            expected_id = UInt64(t4a_id & 0xFFFFFFFFFFFFFFFF)
            @test ITensors.id(it_idx) == expected_id
        end

        @testset "ITensors.Index → Tensor4all.Index" begin
            it_idx = ITensors.Index(3, "Link,l=2")
            t4a_idx = Tensor4all.Index(it_idx)

            @test Tensor4all.dim(t4a_idx) == 3
            @test Tensor4all.hastag(t4a_idx, "Link")
            @test Tensor4all.hastag(t4a_idx, "l=2")

            # ID should match
            it_id = ITensors.id(it_idx)
            t4a_id = Tensor4all.id(t4a_idx)
            @test UInt64(t4a_id & 0xFFFFFFFFFFFFFFFF) == it_id
        end

        @testset "Roundtrip conversion" begin
            # Tensor4all → ITensors → Tensor4all
            orig = Tensor4all.Index(4; tags="Test")
            it_idx = ITensors.Index(orig)
            back = Tensor4all.Index(it_idx)

            @test Tensor4all.dim(orig) == Tensor4all.dim(back)
            @test Tensor4all.tags(orig) == Tensor4all.tags(back)

            # ITensors → Tensor4all → ITensors
            it_orig = ITensors.Index(6, "Bond")
            t4a_idx = Tensor4all.Index(it_orig)
            it_back = ITensors.Index(t4a_idx)

            @test ITensors.dim(it_orig) == ITensors.dim(it_back)
            @test ITensors.id(it_orig) == ITensors.id(it_back)
        end
    end

    # ========================================================================
    # Tensor Tests
    # ========================================================================

    @testset "Tensor creation (f64)" begin
        i = T4AIndex(2)
        j = T4AIndex(3)

        # Create a 2x3 tensor
        data = Float64[1 2 3; 4 5 6]  # 2x3 matrix
        t = T4ATensor([i, j], data)

        @test Tensor4all.rank(t) == 2
        @test Tensor4all.dims(t) == (2, 3)
        @test Tensor4all.storage_kind(t) == Tensor4all.DenseF64

        # Get indices
        inds = Tensor4all.indices(t)
        @test length(inds) == 2
        @test Tensor4all.dim(inds[1]) == 2
        @test Tensor4all.dim(inds[2]) == 3
    end

    @testset "Tensor data roundtrip (f64)" begin
        i = T4AIndex(2)
        j = T4AIndex(3)

        # Create tensor with known data
        original_data = [1.0 2.0 3.0; 4.0 5.0 6.0]  # 2x3
        t = T4ATensor([i, j], original_data)

        # Get data back
        retrieved = Tensor4all.data(t)
        @test retrieved ≈ original_data
        @test size(retrieved) == (2, 3)
    end

    @testset "Tensor creation (complex)" begin
        i = T4AIndex(2)
        j = T4AIndex(2)

        # Create complex tensor
        data = ComplexF64[1+im 2+2im; 3+3im 4+4im]
        t = T4ATensor([i, j], data)

        @test Tensor4all.rank(t) == 2
        @test Tensor4all.dims(t) == (2, 2)
        @test Tensor4all.storage_kind(t) == Tensor4all.DenseC64
    end

    @testset "Tensor data roundtrip (complex)" begin
        i = T4AIndex(2)
        j = T4AIndex(3)

        original_data = ComplexF64[1+1im 2+2im 3+3im; 4+4im 5+5im 6+6im]
        t = T4ATensor([i, j], original_data)

        retrieved = Tensor4all.data(t)
        @test retrieved ≈ original_data
        @test size(retrieved) == (2, 3)
    end

    @testset "Tensor copy" begin
        i = T4AIndex(3)
        j = T4AIndex(4)
        data = rand(3, 4)

        t = T4ATensor([i, j], data)
        t2 = copy(t)

        @test Tensor4all.rank(t2) == 2
        @test Tensor4all.dims(t2) == (3, 4)
        @test Tensor4all.data(t2) ≈ data
    end

    @testset "Tensor display" begin
        i = T4AIndex(2)
        j = T4AIndex(3)
        t = T4ATensor([i, j], rand(2, 3))

        s = sprint(show, t)
        @test occursin("rank=2", s)
        @test occursin("dims=(2, 3)", s)
    end

    @testset "Higher rank tensor" begin
        i = T4AIndex(2)
        j = T4AIndex(3)
        k = T4AIndex(4)

        # 3D tensor
        data = rand(2, 3, 4)
        t = T4ATensor([i, j, k], data)

        @test Tensor4all.rank(t) == 3
        @test Tensor4all.dims(t) == (2, 3, 4)

        # Roundtrip
        retrieved = Tensor4all.data(t)
        @test retrieved ≈ data
        @test size(retrieved) == (2, 3, 4)
    end

    # ========================================================================
    # ITensors Tensor Conversion Tests
    # ========================================================================

    @testset "ITensors Tensor Extension" begin
        @testset "Tensor4all.Tensor → ITensors.ITensor (f64)" begin
            i = T4AIndex(2; tags="Site")
            j = T4AIndex(3; tags="Link")

            data = [1.0 2.0 3.0; 4.0 5.0 6.0]
            t = T4ATensor([i, j], data)

            # Convert to ITensor
            it = ITensors.ITensor(t)

            # Check structure
            @test length(ITensors.inds(it)) == 2

            # Check data (ITensor should have same data)
            it_inds = ITensors.inds(it)
            arr = Array(it, it_inds...)
            @test arr ≈ data
        end

        @testset "ITensors.ITensor → Tensor4all.Tensor (f64)" begin
            it_i = ITensors.Index(2, "Site")
            it_j = ITensors.Index(3, "Link")

            data = [1.0 2.0 3.0; 4.0 5.0 6.0]
            it = ITensors.ITensor(data, it_i, it_j)

            # Convert to Tensor4all.Tensor
            t = T4ATensor(it)

            @test Tensor4all.rank(t) == 2
            @test Tensor4all.dims(t) == (2, 3)
            @test Tensor4all.data(t) ≈ data
        end

        @testset "Tensor roundtrip: T4A → ITensors → T4A" begin
            i = T4AIndex(2; tags="X")
            j = T4AIndex(3; tags="Y")
            original_data = rand(2, 3)

            t4a_orig = T4ATensor([i, j], original_data)
            it = ITensors.ITensor(t4a_orig)
            t4a_back = T4ATensor(it)

            # Data should be preserved
            @test Tensor4all.data(t4a_back) ≈ original_data

            # Dimensions should be preserved
            @test Tensor4all.dims(t4a_back) == (2, 3)
        end

        @testset "Tensor roundtrip: ITensors → T4A → ITensors" begin
            it_i = ITensors.Index(3, "A")
            it_j = ITensors.Index(4, "B")
            original_data = rand(3, 4)

            it_orig = ITensors.ITensor(original_data, it_i, it_j)
            t4a = T4ATensor(it_orig)
            it_back = ITensors.ITensor(t4a)

            # Check data
            it_back_inds = ITensors.inds(it_back)
            arr_back = Array(it_back, it_back_inds...)
            @test arr_back ≈ original_data

            # Check that index IDs are preserved
            orig_ids = Set(ITensors.id.(ITensors.inds(it_orig)))
            back_ids = Set(ITensors.id.(it_back_inds))
            @test orig_ids == back_ids
        end

        @testset "Complex tensor conversion" begin
            i = T4AIndex(2)
            j = T4AIndex(2)
            data = ComplexF64[1+im 2-im; 3+2im 4-2im]

            t = T4ATensor([i, j], data)
            it = ITensors.ITensor(t)
            t_back = T4ATensor(it)

            @test Tensor4all.data(t_back) ≈ data
        end
    end
end
