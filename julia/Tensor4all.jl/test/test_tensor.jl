@testset "Tensor" begin
    @testset "creation (f64)" begin
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

    @testset "data roundtrip (f64)" begin
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

    @testset "creation (complex)" begin
        i = T4AIndex(2)
        j = T4AIndex(2)

        # Create complex tensor
        data = ComplexF64[1+im 2+2im; 3+3im 4+4im]
        t = T4ATensor([i, j], data)

        @test Tensor4all.rank(t) == 2
        @test Tensor4all.dims(t) == (2, 2)
        @test Tensor4all.storage_kind(t) == Tensor4all.DenseC64
    end

    @testset "data roundtrip (complex)" begin
        i = T4AIndex(2)
        j = T4AIndex(3)

        original_data = ComplexF64[1+1im 2+2im 3+3im; 4+4im 5+5im 6+6im]
        t = T4ATensor([i, j], original_data)

        retrieved = Tensor4all.data(t)
        @test retrieved ≈ original_data
        @test size(retrieved) == (2, 3)
    end

    @testset "copy" begin
        i = T4AIndex(3)
        j = T4AIndex(4)
        data = rand(3, 4)

        t = T4ATensor([i, j], data)
        t2 = copy(t)

        @test Tensor4all.rank(t2) == 2
        @test Tensor4all.dims(t2) == (3, 4)
        @test Tensor4all.data(t2) ≈ data
    end

    @testset "display" begin
        i = T4AIndex(2)
        j = T4AIndex(3)
        t = T4ATensor([i, j], rand(2, 3))

        s = sprint(show, t)
        @test occursin("rank=2", s)
        @test occursin("dims=(2, 3)", s)
    end

    @testset "higher rank" begin
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
end
