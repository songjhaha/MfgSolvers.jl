module MFGPI
using LinearAlgebra, SparseArrays

include("utils.jl")
include("MFGOneDimSolver.jl")
include("MFGTwoDimSolver.jl")


export MFGOneDim, MFGTwoDim, solve_mfg

end # module
