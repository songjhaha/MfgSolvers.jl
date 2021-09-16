module MfgSolvers
using LinearAlgebra, SparseArrays

include("utils.jl")
include("MFGOneDimSolver.jl")
include("MFGTwoDimSolver.jl")
include("fixpoint.jl")
include("MFGTwoDimNonQuad.jl")


export MFGOneDim, MFGTwoDim, solve_mfg, solve_mfg_fixpoint, solve_mfg_non_quad

end # module
