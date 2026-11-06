# Generates itensormps_ortho_lims.h5: MPS groups written by ITensorMPS.jl with
# different orthogonality limits, read by tests/test_hdf5.rs (tensor4all-rs#1041).
#
#   julia --project=<env with ITensors, ITensorMPS, HDF5> itensormps_ortho_lims.jl
using ITensors, ITensorMPS, HDF5, Random

Random.seed!(1041)
s = siteinds(2, 4)
m = random_mps(s; linkdims=2)
mps = Dict(
    "noortho" => MPS([m[i] for i in 1:4]),   # llim 0, rlim 5: no center
    "center1" => orthogonalize(m, 1),        # llim 0, rlim 2
    "center3" => orthogonalize(m, 3),        # llim 2, rlim 4
    "center4" => orthogonalize(m, 4),        # llim 3, rlim 5: the last site
)
h5open(joinpath(@__DIR__, "itensormps_ortho_lims.h5"), "w") do f
    for (name, psi) in mps
        write(f, name, psi)
    end
end
for name in sort(collect(keys(mps)))
    psi = mps[name]
    println(name, ": leftlim=", ITensorMPS.leftlim(psi), " rightlim=", ITensorMPS.rightlim(psi))
end
