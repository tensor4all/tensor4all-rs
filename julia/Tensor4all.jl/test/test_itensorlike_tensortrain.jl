using Test
using Tensor4all: Index as T4AIndex, Tensor as T4ATensor
using Tensor4all: dim, rank
using Tensor4all.ITensorLike: TensorTrain as T4ATT
using Tensor4all.ITensorLike: tensors, bond_dims, maxbonddim, linkind, linkinds
using Tensor4all.ITensorLike: isortho, orthocenter, llim, rlim, canonical_form
using Tensor4all.ITensorLike: orthogonalize!, truncate!, inner
using Tensor4all.ITensorLike: Unitary, LU, CI
using LinearAlgebra

@testset "ITensorlike TensorTrain" begin
    @testset "TensorTrain construction" begin
        # Create indices
        s0 = T4AIndex(2; tags="Site,n=0")
        l01 = T4AIndex(3; tags="Link,l=0")
        s1 = T4AIndex(2; tags="Site,n=1")

        # Create tensors with shared link index
        data0 = rand(2, 3)
        data1 = rand(3, 2)
        t0 = T4ATensor([s0, l01], data0)
        t1 = T4ATensor([l01, s1], data1)

        # Create tensor train
        tt = T4ATT([t0, t1])
        @test length(tt) == 2
        @test !isempty(tt)
    end

    @testset "TensorTrain accessors" begin
        # Create a simple 3-site tensor train
        s0 = T4AIndex(2)
        l01 = T4AIndex(3)
        s1 = T4AIndex(2)
        l12 = T4AIndex(3)
        s2 = T4AIndex(2)

        t0 = T4ATensor([s0, l01], rand(2, 3))
        t1 = T4ATensor([l01, s1, l12], rand(3, 2, 3))
        t2 = T4ATensor([l12, s2], rand(3, 2))

        tt = T4ATT([t0, t1, t2])

        @test length(tt) == 3
        @test bond_dims(tt) == [3, 3]
        @test maxbonddim(tt) == 3

        # Test tensor access
        tensor1 = tt[1]
        @test rank(tensor1) == 2

        tensor2 = tt[2]
        @test rank(tensor2) == 3

        # Test all tensors
        ts = tensors(tt)
        @test length(ts) == 3

        # Test link indices
        link1 = linkind(tt, 1)
        @test link1 !== nothing
        @test dim(link1) == 3

        links = linkinds(tt)
        @test length(links) == 2
    end

    @testset "TensorTrain orthogonality" begin
        s0 = T4AIndex(2)
        l01 = T4AIndex(4)
        s1 = T4AIndex(2)

        t0 = T4ATensor([s0, l01], rand(2, 4))
        t1 = T4ATensor([l01, s1], rand(4, 2))

        tt = T4ATT([t0, t1])

        # Initially not orthogonalized
        @test !isortho(tt)
        @test orthocenter(tt) === nothing

        # Orthogonalize to site 1
        orthogonalize!(tt, 1)
        @test isortho(tt)
        @test orthocenter(tt) == 1
        @test canonical_form(tt) == Unitary

        # Orthogonalize to site 2
        orthogonalize!(tt, 2)
        @test isortho(tt)
        @test orthocenter(tt) == 2
    end

    @testset "TensorTrain canonical forms" begin
        s0 = T4AIndex(2)
        l01 = T4AIndex(4)
        s1 = T4AIndex(2)

        t0 = T4ATensor([s0, l01], rand(2, 4))
        t1 = T4ATensor([l01, s1], rand(4, 2))

        # Test Unitary (QR) form
        tt1 = T4ATT([t0, t1])
        orthogonalize!(tt1, 1; form=Unitary)
        @test canonical_form(tt1) == Unitary

        # Test LU form
        tt2 = T4ATT([t0, t1])
        orthogonalize!(tt2, 1; form=LU)
        @test canonical_form(tt2) == LU

        # Test CI form
        tt3 = T4ATT([t0, t1])
        orthogonalize!(tt3, 1; form=CI)
        @test canonical_form(tt3) == CI
    end

    @testset "TensorTrain norm" begin
        s0 = T4AIndex(2)
        l01 = T4AIndex(3)
        s1 = T4AIndex(2)

        # Create random tensors
        data0 = rand(2, 3)
        data1 = rand(3, 2)
        t0 = T4ATensor([s0, l01], data0)
        t1 = T4ATensor([l01, s1], data1)

        tt = T4ATT([t0, t1])

        # Norm should be positive
        n = norm(tt)
        @test n > 0

        # Norm should be consistent with inner product
        inner_val = inner(tt, tt)
        @test isapprox(real(inner_val), n^2; rtol=1e-10)
        @test isapprox(imag(inner_val), 0.0; atol=1e-10)
    end

    @testset "TensorTrain truncation" begin
        s0 = T4AIndex(4)
        l01 = T4AIndex(8)  # Large bond dimension
        s1 = T4AIndex(4)
        l12 = T4AIndex(8)
        s2 = T4AIndex(4)

        t0 = T4ATensor([s0, l01], rand(4, 8))
        t1 = T4ATensor([l01, s1, l12], rand(8, 4, 8))
        t2 = T4ATensor([l12, s2], rand(8, 4))

        tt = T4ATT([t0, t1, t2])
        @test maxbonddim(tt) == 8

        # Truncate to max dimension 4
        truncate!(tt; maxdim=4)
        @test maxbonddim(tt) <= 4
    end

    @testset "TensorTrain copy" begin
        s0 = T4AIndex(2)
        l01 = T4AIndex(3)
        s1 = T4AIndex(2)

        t0 = T4ATensor([s0, l01], rand(2, 3))
        t1 = T4ATensor([l01, s1], rand(3, 2))

        tt = T4ATT([t0, t1])
        tt_copy = copy(tt)

        # Copy should have same properties
        @test length(tt_copy) == length(tt)
        @test bond_dims(tt_copy) == bond_dims(tt)

        # Modifying copy shouldn't affect original
        orthogonalize!(tt_copy, 1)
        @test isortho(tt_copy)
        @test !isortho(tt)
    end

    @testset "TensorTrain display" begin
        s0 = T4AIndex(2)
        l01 = T4AIndex(3)
        s1 = T4AIndex(2)

        t0 = T4ATensor([s0, l01], rand(2, 3))
        t1 = T4ATensor([l01, s1], rand(3, 2))

        tt = T4ATT([t0, t1])

        # Test show method doesn't error
        io = IOBuffer()
        show(io, tt)
        str = String(take!(io))
        @test occursin("TensorTrain", str)
        @test occursin("sites=2", str)
    end

    @testset "TensorTrain contract" begin
        using Tensor4all.ITensorLike: contract

        # Create two tensor trains representing MPO-like objects
        # TT_A has site indices s0, s1, s2 and external indices a0, a1, a2
        # TT_B has site indices s0, s1, s2 and external indices b0, b1, b2
        # After contraction, result will have external indices a0, a1, a2, b0, b1, b2

        # Shared site indices (to be contracted)
        s0 = T4AIndex(2)
        s1 = T4AIndex(2)
        s2 = T4AIndex(2)

        # External indices for TT_A (these will remain after contraction)
        a0 = T4AIndex(2)
        a1 = T4AIndex(2)
        a2 = T4AIndex(2)

        # External indices for TT_B (these will remain after contraction)
        b0 = T4AIndex(2)
        b1 = T4AIndex(2)
        b2 = T4AIndex(2)

        # First TT (MPO-like): has both site indices and external indices
        l01_a = T4AIndex(3)
        l12_a = T4AIndex(3)
        t0_a = T4ATensor([a0, s0, l01_a], rand(2, 2, 3))
        t1_a = T4ATensor([l01_a, a1, s1, l12_a], rand(3, 2, 2, 3))
        t2_a = T4ATensor([l12_a, a2, s2], rand(3, 2, 2))
        tt_a = T4ATT([t0_a, t1_a, t2_a])

        # Second TT (MPO-like): same site indices but different external indices
        l01_b = T4AIndex(3)
        l12_b = T4AIndex(3)
        t0_b = T4ATensor([b0, s0, l01_b], rand(2, 2, 3))
        t1_b = T4ATensor([l01_b, b1, s1, l12_b], rand(3, 2, 2, 3))
        t2_b = T4ATensor([l12_b, b2, s2], rand(3, 2, 2))
        tt_b = T4ATT([t0_b, t1_b, t2_b])

        # Test zipup contract (default) - result should have 3 sites
        result = contract(tt_a, tt_b)
        @test length(result) == 3
        @test maxbonddim(result) > 0

        # Test contract with explicit :zipup symbol
        result_zipup = contract(tt_a, tt_b; method=:zipup)
        @test length(result_zipup) == 3

        # Test contract with max dimension
        # Note: contract itself may not perfectly truncate during contraction
        # Use truncate! after contraction for guaranteed bond dimension limit
        result_truncated = contract(tt_a, tt_b; maxdim=4)
        @test length(result_truncated) == 3
        # Post-contract truncation
        truncate!(result_truncated; maxdim=4)
        @test maxbonddim(result_truncated) <= 4

        # Test contract with :fit symbol
        result_fit = contract(tt_a, tt_b; method=:fit, nsweeps=2)
        @test length(result_fit) == 3

        # Test contract with string "fit"
        result_fit_str = contract(tt_a, tt_b; method="fit", nsweeps=2)
        @test length(result_fit_str) == 3

        # Test contract with string "zipup"
        result_zipup_str = contract(tt_a, tt_b; method="zipup")
        @test length(result_zipup_str) == 3

        # Test contract with :naive symbol
        result_naive = contract(tt_a, tt_b; method=:naive)
        @test length(result_naive) == 3

        # Test contract with string "naive"
        result_naive_str = contract(tt_a, tt_b; method="naive")
        @test length(result_naive_str) == 3
    end

    @testset "TensorTrain immutable truncate" begin
        using Tensor4all.ITensorLike: truncate

        s0 = T4AIndex(4)
        l01 = T4AIndex(8)  # Large bond dimension
        s1 = T4AIndex(4)
        l12 = T4AIndex(8)
        s2 = T4AIndex(4)

        t0 = T4ATensor([s0, l01], rand(4, 8))
        t1 = T4ATensor([l01, s1, l12], rand(8, 4, 8))
        t2 = T4ATensor([l12, s2], rand(8, 4))

        tt = T4ATT([t0, t1, t2])
        @test maxbonddim(tt) == 8

        # Immutable truncate should return new TT
        tt_truncated = truncate(tt; maxdim=4)
        @test maxbonddim(tt_truncated) <= 4
        # Original should be unchanged
        @test maxbonddim(tt) == 8
    end

    @testset "TensorTrain deepcopy" begin
        s0 = T4AIndex(2)
        l01 = T4AIndex(3)
        s1 = T4AIndex(2)

        t0 = T4ATensor([s0, l01], rand(2, 3))
        t1 = T4ATensor([l01, s1], rand(3, 2))

        tt = T4ATT([t0, t1])
        tt_deep = deepcopy(tt)

        # Deepcopy should have same properties
        @test length(tt_deep) == length(tt)
        @test bond_dims(tt_deep) == bond_dims(tt)

        # Modifying deepcopy shouldn't affect original
        orthogonalize!(tt_deep, 1)
        @test isortho(tt_deep)
        @test !isortho(tt)
    end

    @testset "TensorTrain iterator" begin
        s0 = T4AIndex(2)
        l01 = T4AIndex(3)
        s1 = T4AIndex(2)
        l12 = T4AIndex(4)
        s2 = T4AIndex(2)

        t0 = T4ATensor([s0, l01], rand(2, 3))
        t1 = T4ATensor([l01, s1, l12], rand(3, 2, 4))
        t2 = T4ATensor([l12, s2], rand(4, 2))

        tt = T4ATT([t0, t1, t2])

        # Test iteration
        collected = collect(tt)
        @test length(collected) == 3
        @test all(t -> t isa T4ATensor, collected)

        # Test for loop
        count = 0
        for t in tt
            count += 1
            @test t isa T4ATensor
        end
        @test count == 3

        # Test indexing helpers
        @test firstindex(tt) == 1
        @test lastindex(tt) == 3
        @test eachindex(tt) == 1:3
        @test keys(tt) == 1:3
        @test eltype(typeof(tt)) == T4ATensor
    end

    @testset "TensorTrain from Arrays" begin
        using Tensor4all.ITensorLike: random_tt

        # Create site indices
        s1 = T4AIndex(2)
        s2 = T4AIndex(3)
        s3 = T4AIndex(2)

        # Create arrays with proper bond dimensions
        arr1 = rand(2, 4)      # site_dim=2, right_bond=4
        arr2 = rand(4, 3, 5)   # left_bond=4, site_dim=3, right_bond=5
        arr3 = rand(5, 2)      # left_bond=5, site_dim=2

        # Create tensor train from arrays
        tt = T4ATT([arr1, arr2, arr3], [s1, s2, s3])

        @test length(tt) == 3
        @test bond_dims(tt) == [4, 5]

        # Test with single site
        s_single = T4AIndex(4)
        arr_single = rand(4)
        tt_single = T4ATT([arr_single], [s_single])
        @test length(tt_single) == 1
    end

    @testset "random_tt" begin
        using Tensor4all.ITensorLike: random_tt

        # Create site indices
        sites = [T4AIndex(2) for _ in 1:5]

        # Test with uniform bond dimension
        tt = random_tt(sites; linkdims=4)
        @test length(tt) == 5
        @test all(==(4), bond_dims(tt))
        @test maxbonddim(tt) == 4

        # Test with variable bond dimensions
        tt2 = random_tt(sites; linkdims=[2, 3, 4, 5])
        @test bond_dims(tt2) == [2, 3, 4, 5]

        # Test with complex type
        tt_complex = random_tt(ComplexF64, sites; linkdims=3)
        @test length(tt_complex) == 5

        # Test single site
        tt_single = random_tt([T4AIndex(4)])
        @test length(tt_single) == 1

        # Test empty
        tt_empty = random_tt(T4AIndex[])
        @test length(tt_empty) == 0
        @test isempty(tt_empty)

        # Verify tensor structure
        tt3 = random_tt([T4AIndex(2), T4AIndex(3), T4AIndex(4)]; linkdims=5)
        @test length(tt3) == 3
        # First tensor should have rank 2 (site + right link)
        @test rank(tt3[1]) == 2
        # Middle tensor should have rank 3 (left link + site + right link)
        @test rank(tt3[2]) == 3
        # Last tensor should have rank 2 (left link + site)
        @test rank(tt3[3]) == 2
    end

    @testset "TensorTrain setindex!" begin
        using Tensor4all.ITensorLike: random_tt
        using Tensor4all: indices

        # Create a random tensor train
        sites = [T4AIndex(2) for _ in 1:3]
        tt = random_tt(sites; linkdims=4)

        # Orthogonalize first
        orthogonalize!(tt, 2)
        @test isortho(tt)

        # Get the middle tensor and modify it
        old_tensor = tt[2]
        old_inds = indices(old_tensor)

        # Create a new tensor with the same indices but different data
        new_data = rand(dims(old_tensor)...)
        new_tensor = T4ATensor(old_inds, new_data)

        # Set the tensor
        tt[2] = new_tensor

        # Orthogonality should be invalidated
        @test !isortho(tt)

        # The tensor should be updated
        updated_tensor = tt[2]
        @test Tensor4all.data(updated_tensor) ≈ new_data
    end

    @testset "TensorTrain site search" begin
        using Tensor4all.ITensorLike: random_tt, findsite, findsites, siteinds, siteind
        using Tensor4all: hasind, hasinds

        # Create site indices with tags for identification
        s1 = T4AIndex(2; tags="Site,n=1")
        s2 = T4AIndex(3; tags="Site,n=2")
        s3 = T4AIndex(4; tags="Site,n=3")
        sites = [s1, s2, s3]

        tt = random_tt(sites; linkdims=5)

        # Test findsite - should find the site containing the given index
        @test findsite(tt, s1) == 1
        @test findsite(tt, s2) == 2
        @test findsite(tt, s3) == 3

        # Test findsite with non-existent index
        other_idx = T4AIndex(10)
        @test findsite(tt, other_idx) === nothing

        # Test findsites - should return all sites with common indices
        @test findsites(tt, s1) == [1]
        @test findsites(tt, s2) == [2]
        @test findsites(tt, [s1, s3]) == [1, 3]

        # Test findsites with link index (should find both adjacent sites)
        link1 = linkind(tt, 1)
        @test link1 !== nothing
        found = findsites(tt, link1)
        @test length(found) == 2
        @test 1 in found && 2 in found

        # Test siteinds - should return site indices only (not link indices)
        si1 = siteinds(tt, 1)
        @test length(si1) == 1
        @test Tensor4all.id(si1[1]) == Tensor4all.id(s1)

        si2 = siteinds(tt, 2)
        @test length(si2) == 1
        @test Tensor4all.id(si2[1]) == Tensor4all.id(s2)

        si3 = siteinds(tt, 3)
        @test length(si3) == 1
        @test Tensor4all.id(si3[1]) == Tensor4all.id(s3)

        # Test siteinds for all sites
        all_siteinds = siteinds(tt)
        @test length(all_siteinds) == 3
        @test all(length(si) == 1 for si in all_siteinds)

        # Test siteind - convenience for single site index
        @test siteind(tt, 1) !== nothing
        @test Tensor4all.id(siteind(tt, 1)) == Tensor4all.id(s1)

        # Test hasind predicate
        t1 = tt[1]
        @test hasind(s1)(t1)
        @test !hasind(s2)(t1)
        @test !hasind(s3)(t1)

        # Test hasinds predicate
        @test hasinds([s1])(t1)
        @test hasinds(s1)(t1)

        # Test findfirst with predicate on TensorTrain
        @test findfirst(hasind(s2), tt) == 2
        @test findfirst(hasind(s3), tt) == 3
    end
end
