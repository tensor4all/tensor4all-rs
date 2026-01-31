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

    @testset "Array with index order" begin
        i = T4AIndex(2)
        j = T4AIndex(3)

        # Create 2x3 tensor
        original_data = [1.0 2.0 3.0; 4.0 5.0 6.0]
        t = T4ATensor([i, j], original_data)

        # Get tensor indices
        t_inds = Tensor4all.indices(t)

        # Array with same order as tensor
        arr1 = Array(t, t_inds)
        @test arr1 ≈ original_data
        @test size(arr1) == (2, 3)

        # Array with reversed order (transpose)
        arr2 = Array(t, [t_inds[2], t_inds[1]])
        @test arr2 ≈ transpose(original_data)
        @test size(arr2) == (3, 2)

        # Varargs form
        arr3 = Array(t, t_inds[1], t_inds[2])
        @test arr3 ≈ original_data
    end

    @testset "Array with index order (3D)" begin
        i = T4AIndex(2)
        j = T4AIndex(3)
        k = T4AIndex(4)

        data = rand(2, 3, 4)
        t = T4ATensor([i, j, k], data)
        t_inds = Tensor4all.indices(t)

        # Original order
        arr1 = Array(t, t_inds)
        @test arr1 ≈ data

        # Permuted order: (k, i, j) -> permutedims(data, (3, 1, 2))
        arr2 = Array(t, [t_inds[3], t_inds[1], t_inds[2]])
        @test arr2 ≈ permutedims(data, (3, 1, 2))
        @test size(arr2) == (4, 2, 3)
    end

    @testset "Tensor from array with source indices" begin
        i = T4AIndex(2)
        j = T4AIndex(3)

        # Array with (j, i) order: shape (3, 2)
        arr = rand(3, 2)

        # Create tensor with [i, j] from array with [j, i]
        t_inds = Tensor4all.indices(T4ATensor([i, j], zeros(2, 3)))
        arr_inds = [t_inds[2], t_inds[1]]  # [j, i]

        t = T4ATensor(t_inds, arr, arr_inds)

        # Data should be permuted to (i, j) order
        @test Tensor4all.dims(t) == (2, 3)
        @test Array(t, t_inds) ≈ permutedims(arr, (2, 1))
    end

    @testset "deepcopy" begin
        i = T4AIndex(3)
        j = T4AIndex(4)
        data = rand(3, 4)

        t = T4ATensor([i, j], data)
        t2 = deepcopy(t)

        @test Tensor4all.rank(t2) == 2
        @test Tensor4all.dims(t2) == (3, 4)
        @test Tensor4all.data(t2) ≈ data
    end

    @testset "onehot 1D" begin
        i = T4AIndex(3)
        t = Tensor4all.onehot(i => 1)
        @test Tensor4all.rank(t) == 1
        @test Tensor4all.dims(t) == (3,)
        d = Tensor4all.data(t)
        @test d ≈ [1.0, 0.0, 0.0]
    end

    @testset "onehot 2D" begin
        i = T4AIndex(3)
        j = T4AIndex(4)
        t = Tensor4all.onehot(i => 2, j => 3)
        @test Tensor4all.rank(t) == 2
        @test Tensor4all.dims(t) == (3, 4)
        d = Tensor4all.data(t)
        expected = zeros(3, 4)
        expected[2, 3] = 1.0
        @test d ≈ expected
    end

    @testset "onehot boundary" begin
        i = T4AIndex(3)
        j = T4AIndex(4)
        t = Tensor4all.onehot(i => 3, j => 4)
        d = Tensor4all.data(t)
        expected = zeros(3, 4)
        expected[3, 4] = 1.0
        @test d ≈ expected
    end

    @testset "onehot error" begin
        i = T4AIndex(3)
        @test_throws ArgumentError Tensor4all.onehot(i => 0)  # 0 is out of 1-based range
        @test_throws ArgumentError Tensor4all.onehot(i => 4)  # 4 > dim=3
    end

    @testset "onehot empty" begin
        t = Tensor4all.onehot()
        @test Tensor4all.rank(t) == 0
        @test Tensor4all.dims(t) == ()
        d = Tensor4all.data(t)
        @test d[] ≈ 1.0
    end
end
